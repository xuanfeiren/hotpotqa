#!/usr/bin/env python3
"""
OpenEvolve-based prompt optimization for HotpotQA.

This script uses OpenEvolve's evolutionary search to optimize
the HotpotQA prompt. It evolves the prompt text to maximize
accuracy on the first 100 HotpotQA validation tasks.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from openevolve.api import run_evolution, EvolutionResult


def main():
    parser = argparse.ArgumentParser(description="Optimize HotpotQA prompt with OpenEvolve")
    parser.add_argument("--max_iterations", type=int, default=50, help="Max evolution iterations")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of HotpotQA samples for evaluation")
    parser.add_argument("--parallel_evaluations", type=int, default=4, help="Parallel evaluations")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    initial_prompt_path = script_dir / "initial_prompt.txt"
    evaluator_path = script_dir / "evaluator.py"
    config_path = script_dir / "config.yaml"

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("prompt_opt/results/openevolve") / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set environment variables for the evaluator
    os.environ["OPENEVOLVE_NUM_SAMPLES"] = str(args.num_samples)
    os.environ["OPENEVOLVE_PROMPT"] = str(initial_prompt_path)

    print("=" * 80)
    print("OpenEvolve HotpotQA Prompt Optimization")
    print("=" * 80)
    print(f"Initial prompt: {initial_prompt_path}")
    print(f"Evaluator: {evaluator_path}")
    print(f"Config: {config_path}")
    print(f"Output: {output_dir}")
    print(f"Max iterations: {args.max_iterations}")
    print(f"Evaluation samples: {args.num_samples}")
    print("=" * 80)

    # Save run metadata
    metadata = {
        "method": "OpenEvolve",
        "start_time": datetime.now().isoformat(),
        "max_iterations": args.max_iterations,
        "num_samples": args.num_samples,
        "parallel_evaluations": args.parallel_evaluations,
    }
    with open(output_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Read original prompt
    with open(initial_prompt_path, "r") as f:
        original_prompt = f.read().strip()

    # Run evolution
    print("\nStarting OpenEvolve evolution...\n")
    start_time = time.time()

    try:
        result: EvolutionResult = run_evolution(
            initial_program=str(initial_prompt_path),
            evaluator=str(evaluator_path),
            config=str(config_path),
            output_dir=str(output_dir),
            cleanup=False,
        )

        duration = time.time() - start_time

        print("\n" + "=" * 80)
        print("OPENEVOLVE OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"Duration: {duration:.2f}s")
        print(f"Best score: {result.best_score:.4f}")
        print(f"\nBest evolved prompt:")
        print("-" * 40)
        print(result.best_code)
        print("-" * 40)

        # Save results
        final_results = {
            "method": "OpenEvolve",
            "original_prompt": original_prompt,
            "optimized_prompt": result.best_code,
            "best_score": result.best_score,
            "duration_seconds": duration,
            "num_samples": args.num_samples,
            "max_iterations": args.max_iterations,
        }
        with open(output_dir / "final_results.json", "w") as f:
            json.dump(final_results, f, indent=2)

        with open(output_dir / "best_prompt.txt", "w") as f:
            f.write(result.best_code)

        print(f"\nResults saved to: {output_dir}")

    except Exception as e:
        duration = time.time() - start_time
        print(f"\nError during optimization: {e}")
        import traceback
        traceback.print_exc()

        metadata["status"] = "failed"
        metadata["error"] = str(e)
        metadata["duration_seconds"] = duration
        with open(output_dir / "run_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        raise


if __name__ == "__main__":
    main()
