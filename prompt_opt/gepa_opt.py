"""
Prompt optimization for HotpotQA using GEPA (via DSPy).

This script creates a DSPy Module that wraps the prompt,
and uses GEPA (from `dspy.teleprompt`) to optimize it.
Uses the first 10 tasks as optional validation dataset.
Training uses all 100 tasks. 
"""
import os
import sys
import time
import json
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import dspy
from dspy import Example, Prediction
from dspy.teleprompt import GEPA

from hotpotqa_eval import create_dataset, evaluate_single, Task


# ---------------------------------------------------------------------------
# 0. Reflection Prompt Template (domain context for the optimizer LLM)
# ---------------------------------------------------------------------------
REFLECTION_PROMPT_TEMPLATE = """You are an expert in agent prompt optimization for HotpotQA (multi-hop reasoning). Questions require reasoning over information spread across multiple context paragraphs. 

Current instructions:
```
<curr_instructions>
```

Examples with feedback:
```
<inputs_outputs_feedback>
```

Create instructions that generalize across all tasks based on local feedback from specific question-answer pairs."""


# ---------------------------------------------------------------------------
# 1. DSPy Signature & Module
# ---------------------------------------------------------------------------
class HotpotQAAnswer(dspy.Signature):
    """Answer the question based on the context."""
    context = dspy.InputField(desc="The context paragraphs for the question")
    question = dspy.InputField(desc="The question to answer")
    answer = dspy.OutputField(desc="A concise, factual answer")


class HotpotQAModule(dspy.Module):
    """DSPy Module that uses a Predict to answer HotpotQA questions."""
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(HotpotQAAnswer)

    def forward(self, context, question):
        result = self.predict(context=context, question=question)
        return result


# ---------------------------------------------------------------------------
# 2. Metric with Feedback for GEPA
# ---------------------------------------------------------------------------
def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    GEPA-compatible metric that returns score + feedback.
    
    gold: dspy.Example with context, question, expected_answer
    pred: dspy.Prediction with answer
    """
    expected = gold.expected_answer
    predicted = pred.answer if hasattr(pred, "answer") else ""
    
    # Check answer
    if predicted is None:
        predicted = ""
    
    output_clean = predicted.strip().lower().rstrip(".,!?;:")
    expected_clean = expected.strip().lower().rstrip(".,!?;:")
    
    is_correct = (output_clean == expected_clean) or (expected_clean in output_clean)
    score = 1.0 if is_correct else 0.0
    
    if is_correct:
        feedback = f"Correct! The model answered '{predicted}' which matches the expected answer '{expected}'."
    else:
        feedback = (
            f"Incorrect. Expected: '{expected}', Got: '{predicted}'. "
            f"Question: {gold.question}. "
            f"The answer should be concise and factual, matching the expected answer exactly."
        )
    
    return dspy.Prediction(score=score, feedback=feedback)


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Optimize HotpotQA prompt with GEPA")
    parser.add_argument("--num_tasks", type=int, default=100, help="Number of tasks (training set)")
    parser.add_argument("--num_val_tasks", type=int, default=100, help="Number of validation tasks")
    parser.add_argument("--max_metric_calls", type=int, default=2000, help="Budget for GEPA metric calls")
    parser.add_argument("--reflection_minibatch_size", type=int, default=10, help="Minibatch size for reflection")
    parser.add_argument("--num_threads", type=int, default=10, help="Parallel threads")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash-lite", help="Model for task execution")
    parser.add_argument("--reflection_model", type=str, default="gemini-2.5-flash-lite", help="Model for GEPA reflection")
    parser.add_argument("--api_base", type=str,
                        default="https://generativelanguage.googleapis.com/v1beta/openai/",
                        help="API base URL")
    parser.add_argument("--reflection_api_base", type=str, default=None,
                        help="API base URL for reflection model (defaults to --api_base)")
    parser.add_argument("--reflection_api_key", type=str, default=None,
                        help="API key for reflection model (defaults to GEMINI/OPENAI key)")
    parser.add_argument("--output_dir", type=str, default="prompt_opt/results/gepa", help="Output directory")
    parser.add_argument("--log_dir", type=str, default="prompt_opt/results/gepa/logs", help="GEPA log directory")
    parser.add_argument("--run_num", type=int, default=1, help="Run number for experimental tracking")
    args = parser.parse_args()

    # Only apply default output_dir override if the user did not specify one
    # (i.e., still at the hardcoded default value)
    if args.output_dir == "prompt_opt/results/gepa":
        args.output_dir = f"prompt_opt/results/gepa_{args.run_num}"
        args.log_dir = os.path.join(args.output_dir, "logs")

    # Ensure a fresh start: backup existing output directory if it exists
    if os.path.exists(args.output_dir):
        import shutil
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{args.output_dir}_bak_{timestamp}"
        print(f"Backing up existing output directory to: {backup_dir}")
        shutil.move(args.output_dir, backup_dir)
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # ---- Configure DSPy ----
    print(f"Configuring DSPy with model: {args.model}")
    task_lm = dspy.LM(
        model=f"openai/{args.model}",
        api_base=args.api_base,
        api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        temperature=0.1,
    )
    dspy.configure(lm=task_lm)

    reflection_api_base = args.reflection_api_base or args.api_base
    reflection_api_key = (
        args.reflection_api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    print(f"Configuring reflection LM: {args.reflection_model} @ {reflection_api_base}")
    reflection_lm = dspy.LM(
        model=f"openai/{args.reflection_model}",
        api_base=reflection_api_base,
        api_key=reflection_api_key,
        temperature=1.0,
        max_tokens=16000,
    )

    # ---- Load dataset ----
    print(f"Loading {args.num_tasks} HotpotQA tasks...")
    raw_tasks = create_dataset(n=args.num_tasks)
    print(f"Loaded {len(raw_tasks)} tasks.")

    # Convert to DSPy Examples
    trainset = []
    for task in raw_tasks:
        ex = Example(
            context=task.context,
            question=task.question,
            expected_answer=task.answer,
        ).with_inputs("context", "question")
        trainset.append(ex)

    # Validation set: first num_val_tasks
    valset = trainset[:args.num_val_tasks]

    print(f"Training set: {len(trainset)} examples")
    print(f"Validation set: {len(valset)} examples")

    # ---- Initialize module ----
    student = HotpotQAModule()

    # ---- Original prompt (the signature docstring) ----
    original_prompt = student.predict.signature.instructions
    print(f"\nOriginal instruction:\n{original_prompt}\n")

    # ---- Run GEPA ----
    print("Starting GEPA optimization...")
    gepa = GEPA(
        metric=gepa_metric,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="pareto",
        max_metric_calls=args.max_metric_calls,
        reflection_minibatch_size=args.reflection_minibatch_size,
        track_stats=True,
        num_threads=args.num_threads,
        log_dir=args.log_dir,
        log_frequency=1,
        seed=args.run_num,  # different seed per run for genuine independence
        gepa_kwargs={"reflection_prompt_template": REFLECTION_PROMPT_TEMPLATE},
    )

    start_time = time.time()
    optimized_program = gepa.compile(
        student=student,
        trainset=trainset,
        valset=valset,
    )
    duration = time.time() - start_time

    # ---- Extract and Save History ----
    history = []
    if hasattr(optimized_program, "detailed_results"):
        dr = optimized_program.detailed_results
        best_score_so_far = -1.0
        
        for i in range(len(dr.candidates)):
            score = dr.val_aggregate_scores[i]
            eval_count = dr.discovery_eval_counts[i]
            if score > best_score_so_far:
                best_score_so_far = score
                instr = ""
                for name, pred in dr.candidates[i].named_predictors():
                    instr = pred.signature.instructions
                    break
                
                history.append({
                    "candidate_index": i,
                    "metric_calls": eval_count,
                    "score": score,
                    "instruction": instr
                })
        
        history_path = os.path.join(args.output_dir, "history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        
        # Save full detailed results
        detailed_results_path = os.path.join(args.output_dir, "gepa_detailed_results.json")
        with open(detailed_results_path, "w") as f:
            json.dump(dr.to_dict(), f, indent=2)
        print(f"Detailed results saved to {detailed_results_path}")

        # Pareto Frontier Analysis (Evolution over time)
        print("\n" + "-" * 80)
        print("GEPA OPTIMIZATION FRONTIER EVOLUTION")
        print(f"{'Discovery #':<12} | {'Metric Calls':<15} | {'Agg Score':<10} | {'On Current Frontier?'}")
        print("-" * 80)
        
        # Current frontier candidates are those that are not dominated by any other in the FINAL result
        final_frontier_indices = set()
        for i in range(len(dr.candidates)):
            is_dominated = False
            for j in range(len(dr.candidates)):
                if i == j: continue
                # j dominates i?
                j_better_or_equal = True
                j_strictly_better = False
                for task_idx in range(len(dr.val_subscores[i])):
                    if dr.val_subscores[j][task_idx] < dr.val_subscores[i][task_idx]:
                        j_better_or_equal = False
                        break
                    if dr.val_subscores[j][task_idx] > dr.val_subscores[i][task_idx]:
                        j_strictly_better = True
                if j_better_or_equal and j_strictly_better:
                    is_dominated = True
                    break
            if not is_dominated:
                final_frontier_indices.add(i)

        for i in range(len(dr.candidates)):
            is_best = (dr.val_aggregate_scores[i] == max(dr.val_aggregate_scores[:i+1]))
            on_final_frontier = "Yes" if i in final_frontier_indices else "No"
            print(f"{i:<12} | {dr.discovery_eval_counts[i]:<15} | {dr.val_aggregate_scores[i]:<10.4f} | {on_final_frontier}")
        print("-" * 80)

    # ---- Report ----
    optimized_instruction = optimized_program.predict.signature.instructions
    
    # Reconstruct a prompt template from the optimized instruction
    optimized_prompt_template = (
        f"{optimized_instruction}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )
    
    print("\n" + "=" * 80)
    print("GEPA OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Duration: {duration:.2f}s")
    print(f"Total Metric Calls: {getattr(optimized_program.detailed_results, 'total_metric_calls', 'N/A')}")
    print(f"\nFinal Optimized Instruction:\n{optimized_instruction}")
    print("=" * 80)

    # ---- Final evaluation using our evaluate_single function in parallel ----
    def parallel_evaluate(prompt_template, dataset):
        from concurrent.futures import ThreadPoolExecutor
        num_threads = args.num_threads
        results = []
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(
                evaluate_single,
                prompt_template=prompt_template,
                task=task,
                api_base=args.api_base,
                model=args.model
            ) for task in dataset]
            for future in tqdm(futures, desc=f"Eval (p={num_threads})", leave=False):
                results.append(future.result())
        
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        return {"accuracy": correct / total if total > 0 else 0.0, "correct": correct, "total": total}

    print("\nRunning final evaluation on full dataset to confirm...")
    final_result = parallel_evaluate(optimized_prompt_template, raw_tasks)
    print(f"\nFinal Accuracy: {final_result['accuracy']:.1%} ({final_result['correct']}/{final_result['total']})")

    # ---- Also evaluate original for comparison ----
    original_prompt_template = (
        "Answer the question based on the context. "
        "Context:\n{context}\n"
        "Question: {question}\n"
        "Answer:"
    )

    print("\nEvaluating original prompt for comparison...")
    original_result = parallel_evaluate(original_prompt_template, raw_tasks)
    print(f"Original Accuracy: {original_result['accuracy']:.1%} ({original_result['correct']}/{original_result['total']})")

    # ---- Save Summary ----
    results = {
        "method": "GEPA (DSPy)",
        "original_prompt": original_prompt_template,
        "optimized_instruction": optimized_instruction,
        "optimized_prompt_template": optimized_prompt_template,
        "original_accuracy": original_result["accuracy"],
        "optimized_accuracy": final_result["accuracy"],
        "improvement": final_result["accuracy"] - original_result["accuracy"],
        "num_tasks": args.num_tasks,
        "num_val_tasks": args.num_val_tasks,
        "max_metric_calls": args.max_metric_calls,
        "total_metric_calls": getattr(optimized_program.detailed_results, 'total_metric_calls', 0),
        "duration_seconds": duration,
        "model": args.model,
        "history": history
    }
    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save optimized program
    save_path = os.path.join(args.output_dir, "optimized_program.json")
    optimized_program.save(save_path)
    print(f"Optimized program saved to {save_path}")


if __name__ == "__main__":
    main()
