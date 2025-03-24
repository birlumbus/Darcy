import json
import matplotlib.pyplot as plt

# load data
file_path = 'prompt_results/compiled_analysis/aggregated_results.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# define metrics to plot: mapping display name to JSON key
metrics = {
    "perplexity": "perplexity",
    "bleu1": "bleu1",
    "bleu2": "bleu2",
    "bleu4": "bleu4",
    "rouge_l": "rouge_l",
    "meteor": "meteor"
}

# colors
group_colors = {'medium': 'salmon', 'large': 'lightgreen', '6b': 'skyblue'}

# loop over each stat type: baseline and references
for stat_type in ['baseline', 'references']:
    # loop over each metric
    for metric_name, json_key in metrics.items():
        # for baseline, skip perplexity since it is not available
        if stat_type == 'baseline' and metric_name == 'perplexity':
            continue

        # prep lists for plotting for the current metric and stat type
        x_labels = []
        groups = []
        metric_values = []

        # loop over model groups to extract metric scores, excluding version '0'
        for group in ['medium', 'large', '6b']:
            group_data = data['aggregated_results'][group]
            # sort models by version (skipping version '0')
            for variant in sorted(group_data.keys(), key=lambda v: tuple(map(float, v.split('.')))):
                if variant == "0":
                    continue  # Exclude version '0'
                details = group_data[variant]
                # create a label combining model group and version
                label = f"{group}: {variant}"
                x_labels.append(label)
                groups.append(group)
                value = details['average_metrics'][stat_type].get(json_key, None)
                metric_values.append(value)

        # create bar chart for the current metric and stat type
        plt.figure(figsize=(12, 6))
        bars = plt.bar(x_labels, metric_values, color=[group_colors[g] for g in groups])
        plt.xlabel("Model Variant")
        plt.ylabel(f"{metric_name.upper()} Score ({stat_type.capitalize()})")
        plt.title(f"{metric_name.upper()} Scores Across Model Variants ({stat_type.capitalize()})")

        # highlight best overall only for references bleu4
        if stat_type == "references" and metric_name == "bleu4":
            best_group = data["best_overall"]["model"]
            best_version = data["best_overall"]["version"]
            best_label = f"{best_group}: {best_version}"
            if best_label in x_labels:
                index = x_labels.index(best_label)
                # emphasize best overall variant with a bold black edge
                bars[index].set_edgecolor('black')
                bars[index].set_linewidth(3)
                # add an annotation above the bar
                plt.annotate("Best Overall", 
                             xy=(index, metric_values[index]), 
                             xytext=(index, metric_values[index] + 0.005),
                             ha='center', 
                             arrowprops=dict(facecolor='black', shrink=0.05))

        # improve layout and rotate x-axis labels for readability
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
