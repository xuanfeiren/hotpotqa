"""
Prompt optimization for HotpotQA using Trace (PrioritySearch).

This script wraps the HotpotQA prompt as a trainable parameter
and uses Trace's PrioritySearch algorithm to optimize it.
Follows the same parameter structure as the tau-bench PrioritySearch training script.
"""
import os
import time
import json
import argparse

from opto import trace
from opto.trace import node, bundle
from opto.trace.modules import Module
from opto.optimizers import OptoPrimeV2
from opto.features.priority_search.priority_search_ablation import EpsilonNetPS
from opto.trainer.guide import Guide
from opto.trainer.loggers import WandbLogger, DefaultLogger
from opto.utils.llm import LLM

from hotpotqa_eval import create_dataset, _check_answer, Task

import litellm
import hotpotqa_eval

litellm.drop_params = True
litellm.suppress_debug_info = True

import numpy as np
import torch
np.random.seed(10)
torch.manual_seed(10)

API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL_NAME = "gemini-2.5-flash-lite"
# os.environ["TRACE_LITELLM_MODEL"] = "gemini/gemini-2.5-flash-lite"
os.environ["TRACE_CUSTOMLLM_URL"] = "http://127.0.0.1:8000/v1"
os.environ["TRACE_DEFAULT_LLM_BACKEND"] = "CustomLLM"
os.environ["TRACE_CUSTOMLLM_MODEL"] = "openai/gpt-oss-20b"


OBJECTIVE = """You are an expert in agent prompt optimization for HotpotQA (multi-hop reasoning). Questions require reasoning over information spread across multiple context paragraphs. Your goal is to optimize meta_instructions described in #Variables for an agent.

#Variables:
- meta_instructions: General instructions for the agent's reasoning and formatting.

Goal: Create instructions that generalize across all tasks based on local feedback from specific question-answer pairs.
"""


# ---------------------------------------------------------------------------
# 1. Trainable Prompt Module
# ---------------------------------------------------------------------------
class HotpotQAAgent(Module):
    """Agent that holds the meta_instructions as a trainable parameter and calls LLM."""

    def __init__(self, initial_instructions: str):
        super().__init__()
        self.instructions = node(
            initial_instructions,
            trainable=True,
            name="meta_instructions",
            description=(
                "Meta-instructions for answering HotpotQA multi-hop questions. These instructions guide the model on how to reason over context and format the answer."
            ),
        )
        self.llm = LLM(model="gemini/gemini-2.5-flash-lite")

    @bundle(trainable=False)
    def format_and_call(self, instructions, task):
        """Assembles the prompt by combining meta_instructions, question, and context, then calls the LLM to get the answer."""
        # Use a template that will be formatted inside evaluate_single
        full_prompt_template = f"{instructions}\n\n" + "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        
        # Use the atomic evaluate_single to ensure 1 call = 1 task metric
        # hotpotqa_eval.evaluate_single handles the LLM call and correct/incorrect logic
        result = hotpotqa_eval.evaluate_single(
            prompt_template=full_prompt_template,
            task=task,
            api_base=API_BASE,
            model=MODEL_NAME
        )
        return result["output"]

    def forward(self, task):
        """Forward pass: use optimized instructions to answer the task."""
        response = self.format_and_call(self.instructions, task)
        return response


# ---------------------------------------------------------------------------
# 2. Guide for feedback
# ---------------------------------------------------------------------------
class HotpotQAGuide(Guide):
    """Guide that checks the LLM response against the expected answer."""

    def get_feedback(self, task, response, info, **kwargs):
        expected = info.answer
        is_correct = _check_answer(response, expected)

        if is_correct:
            return 1.0, "Correct."
        else:
            feedback = (
                f"Incorrect. Expected: '{expected}', Got: '{response}'.\n"
                f"Context: {info.context}\n"
                f"Question: {info.question}\n"
                f"Identify the failure mode and update meta_instructions with improved reasoning and formatting guidance."
            )
            return 0.0, feedback


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Train HotpotQA prompt using PrioritySearch algorithm')

    # Dataset parameters
    parser.add_argument('--num_train_samples', type=int, default=10,
                        help='Number of training samples')
    parser.add_argument('--num_validate_samples', type=int, default=10,
                        help='Number of validation samples')
    parser.add_argument('--num_test_samples', type=int, default=1,
                        help='Number of test samples')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Training batch size')
    parser.add_argument('--num_batches', type=int, default=1,
                        help='Number of batches to use from the dataset in each iteration')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--num_steps', type=int, default=5,
                        help='Number of training steps')
    parser.add_argument('--num_threads', type=int, default=20,
                        help='Number of threads for parallel processing')
    parser.add_argument('--test_frequency', type=int, default=None,
                        help='How often to run evaluation')
    parser.add_argument('--log_frequency', type=int, default=1,
                        help='How often to log results')
    parser.add_argument('--save_frequency', type=int, default=None,
                        help='How often to save the agent')
    parser.add_argument('--save_path', type=str, default='checkpoints/priority_search_agent.pkl',
                        help='Path to save the agent')
    parser.add_argument('--num_eval_samples', type=int, default=1,
                        help='Number of times to evaluate each input')

    # PrioritySearch-specific parameters
    parser.add_argument('--num_candidates', type=int, default=5,
                        help='Number of candidates to propose for exploration')
    parser.add_argument('--num_proposals', type=int, default=1,
                        help='Number of proposals to generate per optimizer')
    parser.add_argument('--validate_exploration_candidates', action='store_true', default=False,
                        help='Whether to validate the proposed parameters for exploration')
    parser.add_argument('--use_best_candidate_to_explore', action='store_true', default=False,
                        help='Whether to use the best candidate as part of the exploration candidates')
    parser.add_argument('--memory_size', type=int, default=None,
                        help='Size of the heap memory')
    parser.add_argument('--score_function', type=str, default='mean',
                        choices=['mean', 'ucb'],
                        help='Function to compute the score for the candidates')
    parser.add_argument('--long_term_memory_size', type=int, default=None,
                        help='Size of the long-term memory')
    parser.add_argument('--ucb_exploration_constant', type=float, default=1.0,
                        help='Exploration constant for UCB score function')
    parser.add_argument('--score_range_min', type=float, default=0.0,
                        help='Minimum score for score range')
    parser.add_argument('--score_range_max', type=float, default=1.0,
                        help='Maximum score for score range')
    parser.add_argument('--memory_update_frequency', type=int, default=0,
                        help='Duration of the short-term memory')

    # Algorithm parameters
    parser.add_argument('--algorithm', type=str, default='PS',
                        choices=['PS', 'PS_Summarizer', 'PS_epsNet_Summarizer', 'PS_epsNet'],
                        help='Algorithm variant to use')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Epsilon for EpsilonNetPS')
    parser.add_argument('--epsilon_for_summarizer', type=float, default=0.1,
                        help='Epsilon for summarizer in EpsilonNetPS')

    # Logging parameters
    parser.add_argument('--project_name', type=str, default='hotpotqa-priority-search',
                        help='Wandb project name')
    parser.add_argument('--run_name', type=str, default='debug',
                        help='Wandb run name')

    # Other parameters
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Whether to print verbose output')
    parser.add_argument('--output_dir', type=str, default='prompt_opt/results/trace',
                        help='Output directory')
    parser.add_argument('--run_num', type=int, default=1,
                        help='Run number for experimental tracking')

    args = parser.parse_args()
    
    # Update output directory based on run_num
    if args.run_num > 1 or args.output_dir == "prompt_opt/results/trace":
        args.output_dir = f"prompt_opt/results/trace_{args.run_num}"

    # Ensure a fresh start: backup output directory if it exists
    if os.path.exists(args.output_dir):
        import shutil
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"{args.output_dir}_bak_{timestamp}"
        print(f"Backing up existing output directory to: {backup_dir}")
        shutil.move(args.output_dir, backup_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load dataset ----
    max_samples = max(args.num_train_samples, args.num_validate_samples, args.num_test_samples)
    print(f"Loading {max_samples} HotpotQA tasks...")
    all_tasks = create_dataset(n=max_samples)
    print(f"Loaded {len(all_tasks)} tasks.")

    # Create datasets (same structure as tau-bench reference)
    train_dataset = {
        "inputs": all_tasks[:args.num_train_samples],
        "infos": all_tasks[:args.num_train_samples],
    }
    validate_dataset = {
        "inputs": all_tasks[:args.num_validate_samples],
        "infos": all_tasks[:args.num_validate_samples],
    }
    test_dataset = {
        "inputs": all_tasks[:args.num_test_samples],
        "infos": all_tasks[:args.num_test_samples],
    }

    print(f"Training samples: {len(train_dataset['inputs'])}")
    print(f"Validation samples: {len(validate_dataset['inputs'])}")
    print(f"Test samples: {len(test_dataset['inputs'])}")


    # ---- Initialize agent ----
    print("Initializing agent...")
    agent = HotpotQAAgent("Answer the question based on the context.")
    # response = agent(train_dataset['inputs'][0])
    # print(response)


    # ---- Initialize guide, optimizer, and logger ----
    guide = HotpotQAGuide()
    optimizer = OptoPrimeV2(agent.parameters(), max_tokens=25000, initial_var_char_limit=20000)
    optimizer.objective = OBJECTIVE

    # Prepare configuration for logging
    config_dict = {
        'num_train_samples': args.num_train_samples,
        'num_validate_samples': args.num_validate_samples,
        'num_test_samples': args.num_test_samples,
        'batch_size': args.batch_size,
        'num_batches': args.num_batches,
        'num_epochs': args.num_epochs,
        'num_steps': args.num_steps,
        'memory_update_frequency': args.memory_update_frequency,
        'num_threads': args.num_threads,
        'test_frequency': args.test_frequency,
        'log_frequency': args.log_frequency,
        'save_frequency': args.save_frequency,
        'save_path': args.save_path,
        'num_eval_samples': args.num_eval_samples,
        'num_candidates': args.num_candidates,
        'num_proposals': args.num_proposals,
        'validate_exploration_candidates': args.validate_exploration_candidates,
        'use_best_candidate_to_explore': args.use_best_candidate_to_explore,
        'memory_size': args.memory_size,
        'score_function': args.score_function,
        'ucb_exploration_constant': args.ucb_exploration_constant,
        'score_range_min': args.score_range_min,
        'score_range_max': args.score_range_max,
        'algorithm': args.algorithm,
        'epsilon': args.epsilon,
        'epsilon_for_summarizer': args.epsilon_for_summarizer,
        'verbose': args.verbose,
    }

    logger = WandbLogger(project=args.project_name, verbose=True, name=args.run_name, config=config_dict)

    # ---- Create algorithm ----
    print(f"Creating {args.algorithm} algorithm...")
    if args.algorithm == 'PS':
        algorithm = EpsilonNetPS(
            epsilon=0,
            use_summarizer=False,
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            num_threads=args.num_threads,
        )
    elif args.algorithm == 'PS_Summarizer':
        algorithm = EpsilonNetPS(
            epsilon=0,
            use_summarizer=True,
            agent=agent,
            optimizer=optimizer,
            logger=logger,
            num_threads=args.num_threads,
        )
    elif args.algorithm == 'PS_epsNet_Summarizer':
        algorithm = EpsilonNetPS(
            epsilon=args.epsilon,
            epsilon_for_summarizer=args.epsilon_for_summarizer,
            use_summarizer=True,
            agent=agent,
            optimizer=optimizer,
            logger=logger,
        )
    elif args.algorithm == 'PS_epsNet':
        algorithm = EpsilonNetPS(
            agent=agent,
            optimizer=optimizer,
            use_summarizer=False,
            logger=logger,
            epsilon=args.epsilon,
            num_threads=args.num_threads,
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # Set score range
    score_range = (args.score_range_min, args.score_range_max) if args.score_function == 'ucb' else None

    # Training parameters (same structure as tau-bench reference)
    train_params = {
        "guide": guide,
        "train_dataset": train_dataset,
        "validate_dataset": validate_dataset,
        "test_dataset": test_dataset,
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "score_range": score_range,
        "num_epochs": args.num_epochs,
        "num_steps": args.num_steps,
        "long_term_memory_size": args.long_term_memory_size,
        "memory_update_frequency": args.memory_update_frequency,
        "num_threads": args.num_threads,
        "verbose": args.verbose,
        "test_frequency": args.test_frequency,
        "num_test_samples": args.num_eval_samples,
        "log_frequency": args.log_frequency,
        "save_frequency": args.save_frequency,
        "save_path": args.save_path,
        # PrioritySearch specific parameters
        "num_candidates": args.num_candidates,
        "num_proposals": args.num_proposals,
        "validate_exploration_candidates": args.validate_exploration_candidates,
        "use_best_candidate_to_explore": args.use_best_candidate_to_explore,
        "score_function": args.score_function,
        "ucb_exploration_constant": args.ucb_exploration_constant,
    }

    # ---- Start training ----
    print("Starting training with PrioritySearch...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Num batches: {args.num_batches}")
    print(f"  Num epochs: {args.num_epochs}")
    print(f"  Num steps: {args.num_steps}")
    print(f"  Num threads: {args.num_threads}")
    print(f"  Num candidates: {args.num_candidates}")
    print(f"  Num proposals: {args.num_proposals}")
    print(f"  Score function: {args.score_function}")
    print(f"  Memory update frequency: {args.memory_update_frequency}")
    print(f"  Validate exploration candidates: {args.validate_exploration_candidates}")
    print(f"  Use best candidate to explore: {args.use_best_candidate_to_explore}")

    start_time = time.time()

    try:
        algorithm.train(**train_params)
        duration = time.time() - start_time

        # ---- Report results ----
        optimized_instructions = agent.instructions.data
        optimized_prompt_template = (
            f"{optimized_instructions}\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        print(f"\nTraining completed in {duration:.2f} seconds")
        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"\nOptimized Instructions:\n{optimized_instructions}")
        print("-" * 40)
        print(f"Full Prompt Template:\n{optimized_prompt_template}")
        print("=" * 80)

        # ---- Save results ----
        results = {
            "method": "Trace PrioritySearch",
            "optimized_instructions": optimized_instructions,
            "optimized_prompt_template": optimized_prompt_template,
            "num_train_samples": args.num_train_samples,
            "num_validate_samples": args.num_validate_samples,
            "batch_size": args.batch_size,
            "num_steps": args.num_steps,
            "duration_seconds": duration,
        }
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")

    except Exception as e:
        duration = time.time() - start_time
        print(f"\nError during training: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
