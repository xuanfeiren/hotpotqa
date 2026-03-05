import json
import os
import glob

def process_trace(file_path):
    if not os.path.exists(file_path):
        return
    
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding {file_path}")
            return
    
    if not isinstance(data, list):
        print(f"Skipping {file_path}: not a list")
        return

    for item in data:
        iteration = item.get("Update/n_iters", 0)
        item["prop_step"] = iteration
        item["eval_step"] = 2 * iteration
        item["num_proposals"] = 5 * iteration
        
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Processed {file_path}")

def process_openevolve(file_path):
    if not os.path.exists(file_path):
        return
    
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding {file_path}")
            return
            
    if not isinstance(data, list):
        print(f"Skipping {file_path}: not a list")
        return

    for item in data:
        iteration = item.get("iteration", 0)
        total_calls = item.get("total_calls", 0)
        item["prop_step"] = iteration
        item["eval_step"] = total_calls // 10
        item["num_proposals"] = iteration
        
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Processed {file_path}")

def process_gepa(file_path):
    if not os.path.exists(file_path):
        return
    
    with open(file_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding {file_path}")
            return
            
    if not isinstance(data, list):
        print(f"Skipping {file_path}: not a list")
        return

    current_eval_step = 0
    prev_samples = 0
    
    for item in data:
        iteration = item.get("iteration", 0)
        total_samples = item.get("total_samples", 0)
        
        gap = total_samples - prev_samples
        
        # Applying specified rules
        if gap == 20:
            current_eval_step += 2
        elif gap == 2:
            current_eval_step += 1
        elif gap == 4:
            current_eval_step += 2
        elif gap == 24:
            current_eval_step += 3
        elif gap > 0:
            # For other gaps, we'll try to apply chunks if they fit exactly
            # or just log it. For now, let's handle multiples of 20 as 2 each.
            if gap % 20 == 0:
                current_eval_step += (gap // 20) * 2
            elif gap % 4 == 0:
                current_eval_step += (gap // 4) * 2 # Based on rule gap 4 -> +2
            else:
                print(f"Warning: Unexpected gap {gap} in {file_path} iteration {iteration}. eval_step not increased.")
        
        item["prop_step"] = iteration
        item["num_proposals"] = iteration
        item["eval_step"] = current_eval_step
        
        prev_samples = total_samples
        
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Processed {file_path}")

if __name__ == "__main__":
    base_dir = "/Users/xuanfeiren/Documents/hotpotQA"
    
    # Process Trace
    for fpath in glob.glob(os.path.join(base_dir, "data/Trace/run_*.json")):
        process_trace(fpath)
        
    # Process OpenEvolve
    for fpath in glob.glob(os.path.join(base_dir, "data/openevolve/run_*.json")):
        process_openevolve(fpath)
        
    # Process GEPA
    for fpath in glob.glob(os.path.join(base_dir, "data/gepa/run_*.json")):
        process_gepa(fpath)
