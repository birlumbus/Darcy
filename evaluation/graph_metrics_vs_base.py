import json
import matplotlib.pyplot as plt
import numpy as np

# data from aggregated_means.json (loaded from file)
with open('./prompt_results/compiled_analysis/aggregated_means.json', 'r') as f:
    data = json.load(f)

# choose metric to visualize (e.g., "perplexity", "bleu4", etc.)
metric = "bleu4"

# define color families for each model type
colors = {
    "medium": "skyblue",
    "large": "lightgreen",
    "6b": "salmon"
}

# set up figure with a single subplot (all models in one row)
fig, ax = plt.subplots(figsize=(14, 6))
model_types = ["medium", "large", "6b"]
bar_width = 0.618
offset = 0  # starting x position for the first group
tick_positions = []
tick_labels = []

for ax_i, model in enumerate(model_types):
    baseline = data[model]["0"][metric]
    individual = data[model]["individual_means"]
    
    # extract version names and their corresponding metric values
    versions = list(individual.keys())
    values = [individual[v][metric] for v in versions]
    
    # positions for each bar
    x = offset + np.arange(len(versions))
    
    # plot bars for version averages
    bars = ax.bar(x, values, width=bar_width, color=colors[model], edgecolor='black', label=f"{model} version")
    
    # overlay a marker for the base model (version 0) on each bar
    # marker is placed at center of each bar with y value equal to baseline
    ax.scatter(x, [baseline]*len(x), color='white', edgecolor='black', zorder=5, s=100, label="Base model (v0)" if offset == 0 else "")
    
    # annotate baseline value as horizontal dashed line for clarity
    ax.hlines(baseline, x[0]-bar_width/2, x[-1]+bar_width/2, colors='gray', linestyles='--', linewidth=1)
    
    # set x-axis ticks and labels
    tick_positions.extend(x)
    tick_labels.extend([v for v in versions])
    
    # update offset for next group (add a gap between groups)
    offset = x[-1] + 1 + bar_width

# common labels and layout adjustments
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha='right')
ax.set_xlabel("Version")
ax.set_ylabel(metric.capitalize())
ax.set_title(f"Comparison of {metric.capitalize()} Across Versions\n(Base model value overlaid)")
ax.legend()

plt.tight_layout()
plt.show()
