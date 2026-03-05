import wandb
import json
import os

def debug_extract(project_name, run_name):
    api = wandb.Api()
    runs = api.runs(project_name)
    target_run = next((r for r in runs if r.name == run_name), None)
    
    if not target_run:
        print("Run not found")
        return

    print(f"Inspecting keys for {run_name}:")
    # Get history and look at columns
    history = target_run.history()
    for col in history.columns:
        if "meta_instruction" in col or "instructions" in col or "n_iters" in col:
            print(f"  FOUND KEY: {col}")

if __name__ == "__main__":
    debug_extract("hotpotqa", "run_4_0221_2201")
    debug_extract("hotpotqa", "run_1_0215_1146")
