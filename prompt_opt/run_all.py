#!/usr/bin/env python3
"""
Master script to run all three prompt optimization methods on HotpotQA.

Usage:
    # Run all three methods:
    python prompt_opt/run_all.py

    # Run a specific method:
    python prompt_opt/run_all.py --method trace
    python prompt_opt/run_all.py --method gepa
    python prompt_opt/run_all.py --method openevolve
"""

import subprocess
import sys
import os
import argparse
from datetime import datetime


def run_trace():
    """Run Trace PrioritySearch optimization."""
    print("\n" + "=" * 80)
    print("🔧 RUNNING TRACE PRIORITYSEARCH OPTIMIZATION")
    print("=" * 80 + "\n")
    cmd = [
        sys.executable, "prompt_opt/trace_opt.py",
        "--num_tasks", "100",
        "--batch_size", "2",
        "--num_steps", "20",
        "--num_candidates", "2",
        "--num_threads", "5",
    ]
    subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))


def run_gepa():
    """Run GEPA (DSPy) optimization."""
    print("\n" + "=" * 80)
    print("🧬 RUNNING GEPA (DSPy) OPTIMIZATION")
    print("=" * 80 + "\n")
    cmd = [
        sys.executable, "prompt_opt/gepa_opt.py",
        "--num_tasks", "100",
        "--num_val_tasks", "10",
        "--max_metric_calls", "200",
        "--reflection_minibatch_size", "2",
        "--num_threads", "5",
    ]
    subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))


def run_openevolve():
    """Run OpenEvolve optimization."""
    print("\n" + "=" * 80)
    print("🚀 RUNNING OPENEVOLVE OPTIMIZATION")
    print("=" * 80 + "\n")
    cmd = [
        sys.executable, "prompt_opt/openevolve_opt/run_openevolve.py",
        "--max_iterations", "50",
        "--num_samples", "100",
    ]
    subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Run HotpotQA prompt optimization")
    parser.add_argument("--method", type=str, default="all",
                        choices=["all", "trace", "gepa", "openevolve"],
                        help="Which method to run")
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f"HotpotQA Prompt Optimization — {datetime.now().isoformat()}")
    print(f"Method: {args.method}")
    print(f"{'='*80}")

    if args.method in ("all", "trace"):
        run_trace()
    if args.method in ("all", "gepa"):
        run_gepa()
    if args.method in ("all", "openevolve"):
        run_openevolve()

    print(f"\n{'='*80}")
    print("All requested optimizations complete!")
    print(f"Results are in prompt_opt/results/")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
