"""
Ablation plot: Trace (POLCA, epsilon=0.1) vs Trace_eps0 (POLCA, epsilon=0).

Follows the same style as plot_performance.py so the figures slot into the
paper consistently. Produces 4 PDFs under data/, one per x-axis type:
metric_calls, eval_step, prop_step, num_proposals.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def load_data(filepath, x_key):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error decoding {filepath}")
            return None
    
    # Filter for entries with scores and convert to numbers
    points = []
    for item in data:
        x = item.get(x_key)
        score = item.get("Test/score")
        if x is not None and score is not None:
            points.append((float(x), float(score)))
    
    if not points:
        return None
        
    points.sort()  # Sort by X
    
    # Calculate highest_test_score_so_far and ensure unique X
    processed = {}
    current_max = 0
    for x, score in points:
        current_max = max(current_max, score)
        processed[x] = current_max  # Overwrites duplicates with the max so far for that x
        
    df = pd.DataFrame([{'x': x, 'y': y} for x, y in sorted(processed.items())])
    return df


def create_plot(algorithms_data, output_filename, title, xlabel, ylabel, data_dir, legend_loc='lower right'):
    """
    Create a standardized plot following KernelBench style.
    """
    # KernelBench style parameters
    linewidth = 3.5
    markersize = 10
    figsize = (14, 8)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each algorithm
    for x_values, y_values, std_err, display_name, color, linestyle, marker in algorithms_data:
        # We use plt.step for "so far" hotpotqa plots
        ax.step(x_values, y_values,
                where='post',
                marker=marker,
                label=display_name,
                linewidth=linewidth,
                markersize=markersize,
                color=color,
                linestyle=linestyle,
                markevery=max(1, len(x_values)//10))  # Avoid too many markers
        
        # Plot standard error as shadow
        ax.fill_between(x_values, y_values - std_err, y_values + std_err,
                        step='post',
                        color=color,
                        alpha=0.2)
    
    # Styling following KernelBench guidelines
    ax.set_xlabel(xlabel, fontsize=32, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=32, fontweight='bold')
    ax.set_title(title, fontsize=36, fontweight='bold', pad=20)
    
    # Legend with KernelBench styling
    ax.legend(fontsize=26, loc=legend_loc, frameon=True, shadow=False,
              fancybox=True, framealpha=0.5, borderaxespad=0.8)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linewidth=1.5)
    
    # Axis limits
    ax.set_ylim(0.7, 1.0)  # Requested range 0.7 to 1
    ax.tick_params(axis='both', which='major', labelsize=26)
    
    # Keep all spines visible with thicker lines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.0)
    
    # Save
    plt.tight_layout()
    output_path = data_dir / output_filename
    plt.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0.05)
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    # ========================================================================
    # CONFIGURATION SECTION
    # ========================================================================
    
    data_dir = Path("/Users/xuanfeiren/Documents/hotpotQA/data")
    # Algorithms metadata: (folder_name, display_name, x_key_for_metric_calls)
    algorithms_meta = [
        ("Trace_eps0", "POLCA (\u03b5=0)", "Update/total_samples"),
        ("Trace",      "POLCA (\u03b5=0.1)", "Update/total_samples"),
    ]
    
    # Visual styles (must match order of algorithms_meta)
    # POLCA (eps=0): visually distinguishable blue/dashed/triangle (listed first -> top of legend)
    # POLCA (Ours): keep the same red/solid/circle as plot_performance.py (listed second -> bottom of legend)
    colors = ['#1f77b4', '#d62728']
    linestyles = ['--', '-']
    markers = ['^', 'o']
    
    # Plot configurations: {x_axis_type: (title, xlabel, ylabel, filename)}
    plot_configs = {
        'metric_calls': (
            'HotpotQA Test Score',
            'Number of Metric Calls',
            'Test Score',
            'hotpotqa_ablation_eps_calls.pdf'
        ),
        'eval_step': (
            'HotpotQA Test Score',
            'Evaluation Step',
            'Test Score',
            'hotpotqa_ablation_eps_eval_step.pdf'
        ),
        'prop_step': (
            'HotpotQA Test Score',
            'Proposal Step',
            'Test Score',
            'hotpotqa_ablation_eps_prop_step.pdf'
        ),
        'num_proposals': (
            'HotpotQA Test Score',
            'Number of Proposals',
            'Test Score',
            'hotpotqa_ablation_eps_num_proposals.pdf'
        )
    }
    
    # ========================================================================
    # END CONFIGURATION SECTION
    # ========================================================================

    for x_type, config in plot_configs.items():
        title, xlabel, ylabel, filename = config
        algorithms_plot_data = []

        for i, (folder, display_name, metric_x_key) in enumerate(algorithms_meta):
            x_key = metric_x_key if x_type == 'metric_calls' else x_type
            
            runs = []
            for run_idx in range(1, 4):
                file_path = data_dir / folder / f"run_{run_idx}.json"
                df = load_data(file_path, x_key)
                if df is not None:
                    runs.append(df)
            
            if not runs:
                continue

            # Average runs
            all_xs = sorted(list(set().union(*[run['x'] for run in runs])))
            aligned_scores = []
            for df in runs:
                full_run = pd.DataFrame({'x': all_xs})
                merged = pd.merge(full_run, df, on='x', how='left')
                merged['y'] = merged['y'].ffill().fillna(0)
                aligned_scores.append(merged['y'].values)
                
            aligned_scores = np.array(aligned_scores)
            mean_scores = np.mean(aligned_scores, axis=0)
            std_err = np.std(aligned_scores, axis=0) / np.sqrt(len(runs))
            
            algorithms_plot_data.append((
                all_xs, 
                mean_scores, 
                std_err,
                display_name, 
                colors[i], 
                linestyles[i], 
                markers[i]
            ))
        
        if algorithms_plot_data:
            create_plot(algorithms_plot_data, filename, title, xlabel, ylabel, data_dir)


if __name__ == "__main__":
    main()
