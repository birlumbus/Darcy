import json
import numpy as np
import matplotlib.pyplot as plt

# list of metrics to visualize
metrics_to_average = ['perplexity', 'bleu1', 'bleu2', 'bleu4', 'rouge_l', 'meteor']

# data from aggregated_means.json (loaded from file)
with open('./prompt_results/compiled_analysis/aggregated_means.json', 'r') as f:
    data = json.load(f)
data = data["aggregated_means"]  # extract the nested aggregated_means dictionary

# define color families for each model type
colors = {
    "medium": "skyblue",
    "large": "lightgreen",
    "6b": "salmon"
}

model_types = ["medium", "large", "6b"]
bar_width = 0.618

# loop over each metric and build a graph for each one
for metric in metrics_to_average:
    # set up figure with a single subplot (all models in one row)
    fig, ax = plt.subplots(figsize=(14, 6))
    offset = 0  # starting x position for the first group
    tick_positions = []
    tick_labels = []
    
    # variables to track best overall (highest value) among medium and large only
    best_value = -np.inf
    best_bar = None
    
    for ax_i, model in enumerate(model_types):
        baseline = data[model]["0"][metric]
        individual = data[model]["individual_means"]
        
        # extract version names and their corresponding metric values
        versions = list(individual.keys())
        values = [individual[v][metric] for v in versions]
        
        # positions for each bar
        x = offset + np.arange(len(versions))
        
        # plot bars for version averages
        bars = ax.bar(x, values, width=bar_width, color=colors[model], edgecolor='black', label=f"{model}")
        
        # overlay a marker for the base model (version 0) on each bar:
        # marker is placed at the center of each bar with y value equal to baseline
        ax.scatter(x, [baseline]*len(x), color='white', edgecolor='black', zorder=5, s=100, 
                   label="Base model (v0)" if offset == 0 else "")
        
        # annotate baseline value as horizontal dashed line for clarity
        ax.hlines(baseline, x[0]-bar_width/2, x[-1]+bar_width/2, colors='gray', linestyles='--', linewidth=1)
        
        # check each bar for best overall (exclude 6b models)
        if model != "6b":
            for i, val in enumerate(values):
                if val > best_value:
                    best_value = val
                    best_bar = bars[i]
                    
        # for perplexity, overlay a hatched region for the portion above the baseline
        if metric == "perplexity":
            # for each bar, compute extra height above the baseline
            height_diffs = [v - baseline for v in values]
            # overlay a new bar from the baseline to the top of the bar using a hatch pattern
            # using color 'none' helps preserve the underlying bar’s appearance
            ax.bar(x, height_diffs, width=bar_width, bottom=baseline, color='none', 
                   edgecolor='black', hatch='///', zorder=10)
        
        # set x-axis ticks and labels
        tick_positions.extend(x)
        tick_labels.extend(versions)
        
        # update offset for next group (add a gap between groups)
        offset = x[-1] + 1 + bar_width
    
    # highlight best overall only for bleu4
    if best_bar is not None and metric == "bleu4":
        best_bar.set_edgecolor('black')
        best_bar.set_linewidth(3)
        best_x = best_bar.get_x() + best_bar.get_width() / 2
        best_y = best_bar.get_height()
        ax.annotate("Best Overall", 
                    xy=(best_x, best_y), 
                    xytext=(best_x, best_y + 0.005),
                    ha='center', 
                    arrowprops=dict(facecolor='black', shrink=0.05))
    
    # common labels and layout adjustments
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f"Comparison of {metric.capitalize()} Across Versions\n(Base model value overlaid)")
    ax.legend()
    
    plt.tight_layout()
    plt.show()
