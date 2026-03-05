"""
OpenEvolve optimized evaluator for HotpotQA.
Parallelism is handled INTERNALLY via evaluate_single to allow accurate metric call counting.
"""

import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging

# Suppress verbose HTTP logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import hotpotqa_eval
from openevolve.evaluation_result import EvaluationResult

# ---- Configuration ----
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "gemini-2.5-flash-lite"
# Parallelism is handled here in the optimizer script
NUM_THREADS = int(os.environ.get("OPENEVOLVE_PARALLEL_EVALS", "10"))

STAGE1_SAMPLES = 20
STAGE2_SAMPLES = 100

DATA_TEMPLATE = (
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

# ---- Shared State Management ----
import fcntl
import json as _json

_DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "prompt_opt", "results", "openevolve")
_OUTPUT_DIR = os.environ.get("OPENEVOLVE_OUTPUT_DIR", _DEFAULT_OUTPUT)
_STATE_FILE = os.path.join(_OUTPUT_DIR, "_eval_state.json")

def _load_state():
    os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
    try:
        with open(_STATE_FILE, "r") as f:
            return _json.load(f)
    except:
        return {"total_calls": 0, "best_score": 0.0, "step": 0}

def _save_state(state):
    os.makedirs(os.path.dirname(_STATE_FILE), exist_ok=True)
    # Save latest state
    with open(_STATE_FILE, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        _json.dump(state, f)
        fcntl.flock(f, fcntl.LOCK_UN)
    
    # Also log to a history file for detailed tracking
    history_file = os.path.join(_OUTPUT_DIR, "eval_history.jsonl")
    with open(history_file, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        # Add timestamp
        state_log = state.copy()
        import time as _time
        state_log["timestamp"] = _time.strftime("%Y-%m-%d %H:%M:%S")
        f.write(_json.dumps(state_log) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)

# ---- Evaluation Logic ----
_dataset_cache = {}

def _get_dataset(n):
    if n not in _dataset_cache:
        _dataset_cache[n] = hotpotqa_eval.load_hotpotqa_dataset(n)
    return _dataset_cache[n]

def _generic_evaluate(instructions_path: str, num_samples: int) -> EvaluationResult:
    """Run parallel evaluation by calling evaluate_single for each task."""
    try:
        with open(instructions_path, "r") as f:
            instructions = f.read().strip()
            
        full_prompt = f"{instructions}\n\n{DATA_TEMPLATE}"
        dataset = _get_dataset(num_samples)
        
        results = []
        correct = 0
        
        # Parallelism handled internally per algorithm's optimizer
        with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
            futures = []
            for task in dataset:
                # Each call to evaluate_single counts as one task evaluation
                futures.append(executor.submit(
                    hotpotqa_eval.evaluate_single,
                    prompt_template=full_prompt,
                    task=task,
                    api_base=API_BASE,
                    model=MODEL
                ))
            
            for future in tqdm(futures, desc=f"OE Eval (p={NUM_THREADS})", leave=False):
                res = future.result()
                results.append(res)
                if res["correct"]:
                    correct += 1
        
        accuracy = correct / len(dataset) if dataset else 0.0
        
        # Tracking true metric calls
        state = _load_state()
        state["step"] += 1
        state["total_calls"] += len(dataset)
        if accuracy > state["best_score"]:
            state["best_score"] = accuracy
            print(f"*** NEW BEST SCORE: {accuracy:.4f} ***")
        
        # Log this specific evaluation to the trace
        state["last_accuracy"] = accuracy
        state["instructions"] = instructions
        _save_state(state)
        
        print(f"[OpenEvolve] Accuracy: {accuracy:.3f} | Best: {state['best_score']:.3f} | Total Calls: {state['total_calls']} (Tasks: {len(dataset)})")
        
        feedback = []
        for i, d in enumerate(results[:5]):
            if not d["correct"]:
                feedback.append(f"Q: {d['question']}\nExpected: {d['expected']}\nGot: {d['output']}\n")
        
        return EvaluationResult(
            metrics={"combined_score": accuracy, "length": len(instructions)},
            artifacts={"error_examples": "\n".join(feedback)}
        )
    except Exception as e:
        print(f"Evaluation error: {e}")
        traceback.print_exc()
        return EvaluationResult(metrics={"combined_score": 0.0}, artifacts={"error": str(e)})

# ---- Cascade Interface ----
def evaluate_stage1(instructions_path: str) -> EvaluationResult:
    return _generic_evaluate(instructions_path, STAGE1_SAMPLES)

def evaluate_stage2(instructions_path: str) -> EvaluationResult:
    return _generic_evaluate(instructions_path, STAGE2_SAMPLES)

def evaluate(instructions_path: str) -> EvaluationResult:
    return evaluate_stage2(instructions_path)
