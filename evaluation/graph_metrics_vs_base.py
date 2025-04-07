import matplotlib.pyplot as plt
import numpy as np

# data from aggregated_means.json
data = {
    "medium": {
        "0": {
            "perplexity": 15.89215564886729,
            "bleu1": 0.4856391003085946,
            "bleu2": 0.23066836395477197,
            "bleu4": 0.023454292875137683,
            "rouge_l": 0.06317688819209881,
            "meteor": 0.2255721986714924
        },
        "individual_means": {
            "gpt2-medium_1": {"perplexity": 31.8, "bleu1": 0.6, "bleu2": 0.26, "bleu4": 0.04, "rouge_l": 0.08, "meteor": 0.18},
            "gpt2-medium_1.1": {"perplexity": 29.31, "bleu1": 0.61, "bleu2": 0.27, "bleu4": 0.04, "rouge_l": 0.08, "meteor": 0.2},
            "gpt2-medium_1.2": {"perplexity": 31.78, "bleu1": 0.6, "bleu2": 0.26, "bleu4": 0.04, "rouge_l": 0.08, "meteor": 0.19},
            "gpt2-medium_2": {"perplexity": 36.84, "bleu1": 0.6, "bleu2": 0.25, "bleu4": 0.05, "rouge_l": 0.07, "meteor": 0.18},
            "gpt2-medium_2.1": {"perplexity": 28.61, "bleu1": 0.58, "bleu2": 0.26, "bleu4": 0.04, "rouge_l": 0.08, "meteor": 0.19},
            "gpt2-medium_2.2": {"perplexity": 35.38, "bleu1": 0.61, "bleu2": 0.26, "bleu4": 0.04, "rouge_l": 0.08, "meteor": 0.18}
        }
    },
    "large": {
        "0": {
            "perplexity": 13.283032890944975,
            "bleu1": 0.47233219597517556,
            "bleu2": 0.22929200440922634,
            "bleu4": 0.023814557027297976,
            "rouge_l": 0.06222120308949199,
            "meteor": 0.2188837554574373
        },
        "individual_means": {
            "gpt2-large_1": {"perplexity": 23.72, "bleu1": 0.6, "bleu2": 0.27, "bleu4": 0.04, "rouge_l": 0.08, "meteor": 0.2},
            "gpt2-large_1.1": {"perplexity": 23.92, "bleu1": 0.56, "bleu2": 0.25, "bleu4": 0.04, "rouge_l": 0.08, "meteor": 0.19},
            "gpt2-large_1.2": {"perplexity": 26.16, "bleu1": 0.58, "bleu2": 0.25, "bleu4": 0.04, "rouge_l": 0.08, "meteor": 0.19},
            "gpt2-large_2": {"perplexity": 30.55, "bleu1": 0.57, "bleu2": 0.25, "bleu4": 0.04, "rouge_l": 0.07, "meteor": 0.18},
            "gpt2-large_2.1": {"perplexity": 30.64, "bleu1": 0.54, "bleu2": 0.23, "bleu4": 0.04, "rouge_l": 0.07, "meteor": 0.17},
            "gpt2-large_2.2": {"perplexity": 31.25, "bleu1": 0.48, "bleu2": 0.19, "bleu4": 0.04, "rouge_l": 0.07, "meteor": 0.15}
        }
    },
    "6b": {
        "0": {
            "perplexity": 11.791919994354249,
            "bleu1": 0.4881322968401878,
            "bleu2": 0.25163458617986817,
            "bleu4": 0.02204958557031764,
            "rouge_l": 0.06070610769737513,
            "meteor": 0.2285012830982082
        },
        "individual_means": {
            "gpt-j-6b_1": {"perplexity": 11.81, "bleu1": 0.54, "bleu2": 0.29, "bleu4": 0.03, "rouge_l": 0.07, "meteor": 0.23},
            "gpt-j-6b_2": {"perplexity": 12.09, "bleu1": 0.52, "bleu2": 0.26, "bleu4": 0.02, "rouge_l": 0.06, "meteor": 0.23},
            "gpt-j-6b_2.1": {"perplexity": 12.18, "bleu1": 0.54, "bleu2": 0.26, "bleu4": 0.02, "rouge_l": 0.07, "meteor": 0.23}
        }
    }
}

# choose metric to visualize (e.g., "perplexity", "bleu4", etc.)
metric = "bleu4"

# define color families for each model type
colors = {
    "medium": "skyblue",  # blue
    "large": "lightgreen",   # green
    "6b": "salmon"       # red
}

# set up figure with a single subplot (all models in one row)
fig, ax = plt.subplots(figsize=(14, 6))
model_types = ["medium", "large", "6b"]
bar_width = 0.809
offset = 0  # starting x position for first group
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
