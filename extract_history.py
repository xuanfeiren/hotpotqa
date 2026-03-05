import wandb
import re
import os
import pandas as pd

def extract_all_steps(project_name, run_name):
    api = wandb.Api()
    
    try:
        # Get all runs in the project
        runs = api.runs(project_name)
        target_run = None
        for run in runs:
            if run.name == run_name:
                target_run = run
                break
        
        if not target_run:
            print(f"Run '{run_name}' not found.")
            return

        print(f"Extracting history for: {target_run.name}\n")
        
        # Get history focusing on meta_instructions key
        # We use .scan() to get ALL entries if history is long
        history_iter = target_run.scan_history()
        
        all_instructions = []
        
        for row in history_iter:
            step = row.get("_step")
            
            # Find any value that looks like our instruction (logged as HTML file dict)
            for key, value in row.items():
                if "_text" in key and isinstance(value, dict) and value.get("_type") == "html-file":
                    html_path = value.get("path")
                    if html_path:
                        try:
                            # Download the file
                            file_obj = target_run.file(html_path)
                            local_path = file_obj.download(replace=True, root="temp_wandb_files").name
                            
                            with open(local_path, "r") as f:
                                content = f.read()
                                # Strip HTML tags
                                raw_text = re.sub('<[^<]+?>', '', content).strip()
                                all_instructions.append({
                                    "step": step,
                                    "key": key,
                                    "instruction": raw_text
                                })
                        except Exception as download_err:
                            print(f"Error downloading step {step}: {download_err}")

        # Display results
        if not all_instructions:
            print("No instruction history found.")
            return

        # Sort and print
        df = pd.DataFrame(all_instructions).sort_values("step")
        for _, entry in df.iterrows():
            print("-" * 30)
            print(f"STEP: {entry['step']} | KEY: {entry['key']}")
            print(f"INSTRUCTION:\n{entry['instruction']}")
        print("-" * 30)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    PROJECT = "hotpotqa-debug"
    RUN_NAME = "run_1_0215_1139"
    extract_all_steps(PROJECT, RUN_NAME)
