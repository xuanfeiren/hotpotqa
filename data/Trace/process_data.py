import json
import os

def process_file(filename, output_filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    
    keys_to_keep = [
        "Update/n_iters",
        "Update/total_samples",
        "Update/long_term_memory_size",
        "Parameter/meta_instructions:0_text_content"
    ]
    
    processed_data = []
    for step in data:
        new_step = {k: step[k] for k in keys_to_keep if k in step}
        processed_data.append(new_step)
    
    with open(output_filename, 'w') as f:
        json.dump(processed_data, f, indent=2)

if __name__ == "__main__":
    base_dir = "/Users/xuanfeiren/Documents/hotpotQA/data/Trace"
    files = [
        ("run_1_0215_1146.json", "run_1.json"),
        ("run_2_0215_1507.json", "run_2.json"),
        ("run_3_0216_1459.json", "run_3.json")
    ]
    
    for input_f, output_f in files:
        input_path = os.path.join(base_dir, input_f)
        output_path = os.path.join(base_dir, output_f)
        if os.path.exists(input_path):
            print(f"Processing {input_f} -> {output_f}")
            process_file(input_path, output_path)
        else:
            print(f"File not found: {input_path}")
