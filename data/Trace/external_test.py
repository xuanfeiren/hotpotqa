import os
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Add the parent directory to sys.path to import hotpotqa_eval
sys.path.append("/Users/xuanfeiren/Documents/hotpotQA")
from hotpotqa_eval import load_hotpotqa_dataset, evaluate_single

# Global Settings (from independent_eval.py)
API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = "gemini-2.5-flash-lite"
NUM_TASKS = 100
EVALS_PER_TASK = 5
NUM_THREADS = 1  # Sequential processing for rate stability
CACHE_FILE = "/Users/xuanfeiren/Documents/hotpotQA/data/Trace/test_cache.json"

def get_accuracy(prompt_content):
    # Standard template formatting
    full_prompt_template = (
        f"{prompt_content}\n\n"
        "Context:\n{{context}}\n\n"
        "Question: {{question}}\n\n"
        "Answer:"
    ).replace("{{", "{").replace("}}", "}")

    if not hasattr(get_accuracy, '_tasks'):
        get_accuracy._tasks = load_hotpotqa_dataset(NUM_TASKS)
    base_tasks = get_accuracy._tasks
    
    # Duplicate tasks for multiple evaluations per task
    eval_list = []
    for _ in range(EVALS_PER_TASK):
        eval_list.extend(base_tasks)
    
    total_calls = len(eval_list)
    correct_count = 0
    
    for task in tqdm(eval_list, desc="Evaluating", unit="call", leave=False):
        try:
            res = evaluate_single(
                prompt_template=full_prompt_template,
                task=task,
                api_base=API_BASE,
                model=MODEL
            )
            if res["correct"]:
                correct_count += 1
            # Very small sleep to prevent bursty traffic
            time.sleep(0.05)
        except Exception as e:
            print(f"\n[!] Error in evaluation: {e}")
            time.sleep(1)

    accuracy = correct_count / total_calls if total_calls > 0 else 0.0
    return accuracy

def main():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            try:
                cache = json.load(f)
            except:
                cache = {}
    else:
        cache = {}

    target_files = [
        "/Users/xuanfeiren/Documents/hotpotQA/data/Trace/run_1.json",
        "/Users/xuanfeiren/Documents/hotpotQA/data/Trace/run_2.json",
        "/Users/xuanfeiren/Documents/hotpotQA/data/Trace/run_3.json",
        "/Users/xuanfeiren/Documents/hotpotQA/data/Trace/run_4.json",
        "/Users/xuanfeiren/Documents/hotpotQA/data/Trace/run_5.json",   
        "/Users/xuanfeiren/Documents/hotpotQA/data/Trace/run_6.json",   
        "/Users/xuanfeiren/Documents/hotpotQA/data/Trace_eps0/run_1.json",
        "/Users/xuanfeiren/Documents/hotpotQA/data/Trace_eps0/run_2.json",
        "/Users/xuanfeiren/Documents/hotpotQA/data/Trace_eps0/run_3.json",
    ]

    for input_file in target_files:
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found, skipping.")
            continue

        print(f"\n[*] Processing {input_file} (Sequential)...")
        with open(input_file, "r") as f:
            data = json.load(f)

        # First, handle all cached items for this file
        changed_cache = False
        for item in data:
            n_iters = item.get("Update/n_iters")
            if n_iters is not None and n_iters % 10 == 0:
                prompt = item.get("Parameter/meta_instructions:0_text_content")
                if prompt and prompt in cache:
                    score = cache[prompt]
                    if item.get("Test/score") != score:
                        item["Test/score"] = score
                        changed_cache = True

        if changed_cache:
            print(f"[*] Updating {os.path.basename(input_file)} with cached results...")
            with open(input_file, "w") as f:
                json.dump(data, f, indent=2)

        # Now do the actual testing for missing items in this file
        for item in data:
            n_iters = item.get("Update/n_iters")
            if n_iters is not None and n_iters % 10 == 0:
                if "Test/score" in item:
                    continue
                
                prompt = item.get("Parameter/meta_instructions:0_text_content")
                if not prompt: continue
                
                print(f"[*] Iter {n_iters}: Testing new prompt...")
                score = get_accuracy(prompt)
                print(f"[*] Iter {n_iters}: Score = {score:.4f}")
                
                cache[prompt] = score
                item["Test/score"] = score
                
                # Save cache immediately
                with open(CACHE_FILE, "w") as f:
                    json.dump(cache, f, indent=2)
                
                # Save data file immediately to keep progress
                with open(input_file, "w") as f:
                    json.dump(data, f, indent=2)

        print(f"[*] Done processing {input_file}")

    print("\n[*] All runs completed.")

if __name__ == "__main__":
    main()
