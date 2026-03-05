#!/usr/bin/env python3
"""
Independent Evaluator for HotpotQA Prompts.
Purpose: Fair and robust comparison of different prompt optimization algorithms.
Usage: python independent_eval.py "YOUR_PROMPT_STRING" --num_tasks 100 --evals_per_task 5
"""

import os
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Ensure common modules are visible
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from hotpotqa_eval import load_hotpotqa_dataset, evaluate_single

# Global Settings
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "gemini-2.5-flash-lite"

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained prompt independently.")
    parser.add_argument("prompt", type=str, help="The prompt string to evaluate OR a path to a txt file.")
    parser.add_argument("--num_tasks", type=int, default=100, help="Number of unique tasks to eval.")
    parser.add_argument("--evals_per_task", type=int, default=5, help="Number of repetitions per task for robustness.")
    parser.add_argument("--num_threads", type=int, default=10, help="Number of parallel threads for evaluation.")
    args = parser.parse_args()

    # 1. Resolve Prompt (String or File)
    prompt_content = args.prompt
    if os.path.exists(args.prompt):
        with open(args.prompt, "r") as f:
            prompt_content = f.read().strip()
    
    # Standard template formatting
    full_prompt_template = (
        f"{prompt_content}\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    )

    # 2. Load Dataset
    print(f"[*] Loading {args.num_tasks} unique tasks...")
    base_tasks = load_hotpotqa_dataset(args.num_tasks)
    
    # Duplicate tasks for multiple evaluations per task
    eval_list = []
    for _ in range(args.evals_per_task):
        eval_list.extend(base_tasks)
    
    total_calls = len(eval_list)
    print(f"[*] Starting independent evaluation: {args.num_tasks} tasks x {args.evals_per_task} repetitions = {total_calls} calls.")

    # 3. Parallel Execution
    results = []
    correct_count = 0
    
    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = [
            executor.submit(
                evaluate_single,
                prompt_template=full_prompt_template,
                task=task,
                api_base=API_BASE,
                model=MODEL
            ) for task in eval_list
        ]
        
        for future in tqdm(futures, desc="Evaluating", unit="call"):
            res = future.result()
            results.append(res)
            if res["correct"]:
                correct_count += 1

    # 4. Final Metrics
    accuracy = correct_count / total_calls if total_calls > 0 else 0.0
    
    print("\n" + "="*60)
    print("INDEPENDENT EVALUATION RESULTS")
    print("="*60)
    print(f"Prompt Length:     {len(prompt_content)} chars")
    print(f"Total Tasks:       {args.num_tasks}")
    print(f"Evals Per Task:    {args.evals_per_task}")
    print(f"Total Metric Calls: {total_calls}")
    print(f"Final Accuracy:    {accuracy:.2%} ({correct_count}/{total_calls})")
    print("="*60)

if __name__ == "__main__":
    main()
