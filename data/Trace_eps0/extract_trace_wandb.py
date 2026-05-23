import wandb
import json
import os
import pandas as pd
import numpy as np
import re

def extract_trace_history(project_name, run_name, output_json):
    print(f"[*] Fetching history for {run_name}...")
    api = wandb.Api()
    
    try:
        runs = api.runs(project_name)
        target_run = next((r for r in runs if r.name == run_name), None)
        
        if not target_run:
            print(f"[!] Run '{run_name}' not found.")
            return False

        # Get full history
        history = target_run.history()
        
        processed_data = []
        # Mapping our internal JSON keys
        # The instruction is an HTML file, we need to download it
        for _, row in history.iterrows():
            if pd.notna(row.get("Update/n_iters")) or pd.notna(row.get("Update/total_samples")):
                step_data = {
                    "Update/n_iters": int(row["Update/n_iters"]) if pd.notna(row.get("Update/n_iters")) else None,
                    "Update/total_samples": int(row["Update/total_samples"]) if pd.notna(row.get("Update/total_samples")) else None,
                    "Update/long_term_memory_size": int(row["Update/long_term_memory_size"]) if pd.notna(row.get("Update/long_term_memory_size")) else None,
                }
                
                # Handle the HTML instruction file
                wandb_inst_val = row.get("Parameter/meta_instructions:0_text")
                instruction_text = ""
                
                if isinstance(wandb_inst_val, dict) and wandb_inst_val.get("_type") == "html-file":
                    html_path = wandb_inst_val.get("path")
                    if html_path:
                        try:
                            # Use temp directory for downloads
                            file_obj = target_run.file(html_path)
                            local_file = file_obj.download(replace=True, root="temp_wandb").name
                            with open(local_file, "r") as f:
                                html_content = f.read()
                                # Simple strip of HTML tags
                                instruction_text = re.sub('<[^<]+?>', '', html_content).strip()
                        except Exception as e:
                            print(f"      [!] Error downloading html: {e}")
                elif isinstance(wandb_inst_val, str):
                    instruction_text = wandb_inst_val
                
                step_data["Parameter/meta_instructions:0_text_content"] = instruction_text
                processed_data.append(step_data)
        
        if not processed_data:
            print(f"[!] No matching data found.")
            return False

        # Save to JSON
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, 'w') as f:
            json.dump(processed_data, f, indent=2)
            
        print(f"[+] Saved {len(processed_data)} steps to {output_json}")
        return True

    except Exception as e:
        print(f"[!] Error: {e}")
        return False

if __name__ == "__main__":
    PROJECT = "hotpotqa_eps_0"
    # Filled in after experiments finish (see companion resolve_run_names.py or manual edit)
    RUNS = [
        ("run_1_0421_2122", "run_1.json"),
        ("run_2_0421_2303", "run_2.json"),
        ("run_3_0421_2304", "run_3.json"),
    ]
    
    BASE_DIR = "/Users/xuanfeiren/Documents/hotpotQA/data/Trace_eps0"
    
    for run_name, output_name in RUNS:
        output_path = os.path.join(BASE_DIR, output_name)
        extract_trace_history(PROJECT, run_name, output_path)
