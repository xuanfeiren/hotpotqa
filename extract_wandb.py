import wandb
import os

def extract_wandb_data(project_name, run_name):
    api = wandb.Api()
    
    # Try to find the run by its display name
    # We might need to iterate through entities if not specified, 
    # but usually the API key identifies the user.
    try:
        # Get all runs in the project
        runs = api.runs(project_name)
        target_run = None
        for run in runs:
            if run.name == run_name:
                target_run = run
                break
        
        if not target_run:
            print(f"Run '{run_name}' not found in project '{project_name}'.")
            return

        print(f"Found Run: {target_run.id} ({target_run.name})")
        print(f"Status: {target_run.state}")
        
        # 1. Check Summary (for latest values)
        print("\n--- Summary Data ---")
        for key, value in target_run.summary.items():
            if "_text" in key:
                print(f"{key}: {value}")
        
        # 2. Check History (for values over time)
        print("\n--- History Data (last 5 entries) ---")
        # scan() returns an iterator over the history.
        # We can also use target_run.history() but it's limited in columns by default.
        history = target_run.history(keys=[k for k in target_run.summary.keys() if "_text" in k or "Parameter" in k])
        if not history.empty:
            print(history.tail(5).to_string())
            
            # Extract the actual content from the latest HTML file logged
            print("\n--- Content of Latest Meta-Instructions ---")
            
            # Find the path in the last history row
            last_row = history.iloc[-1]
            for col in history.columns:
                if "_text" in col and isinstance(last_row[col], dict):
                    html_path = last_row[col].get('path')
                    if html_path:
                        print(f"Downloading {html_path}...")
                        file_obj = target_run.file(html_path)
                        # download() returns the local file handle/object
                        local_path = file_obj.download(replace=True).name
                        with open(local_path, "r") as f:
                            content = f.read()
                            # Strip HTML tags to see the raw text
                            import re
                            raw_text = re.sub('<[^<]+?>', '', content)
                            print(f"\n[Latest Instruction]:\n{raw_text.strip()}")
        else:
            print("History looks empty.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    PROJECT = "hotpotqa-debug"
    RUN_NAME = "run_1_0215_1139"
    extract_wandb_data(PROJECT, RUN_NAME)
