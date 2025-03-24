import json
import matplotlib.pyplot as plt


# load data
file_path = 'prompt_results/compiled_analysis/aggregated_results.json'
with open(file_path, 'r') as f:
    data = json.load(f)

# prep lists for plotting BLEU4 (reference) scores
x_labels = []
groups = []
bleu4_values = []

# colors
group_colors = {'medium': 'salmon', 'large': 'lightgreen', '6b': 'skyblue'}

# loop over model groups to extract BLEU4 (reference) scores
for group in ['medium', 'large', '6b']:
    group_data = data['aggregated_results'][group]
    # sort models by version
    for variant in sorted(group_data.keys(), key=lambda v: tuple(map(float, v.split('.')))):
        details = group_data[variant]
        # create a label combining model group and version
        label = f"{group}: {variant}"
        x_labels.append(label)
        groups.append(group)
        bleu4 = details['average_metrics']['references']['bleu4']
        bleu4_values.append(bleu4)

# create bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(x_labels, bleu4_values, color=[group_colors[g] for g in groups])

# title/axes' labels
plt.xlabel("Model Variant")
plt.ylabel("BLEU4 Score (Reference)")
plt.title("BLEU4 Scores Across Model Variants")

# highlight best overall
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
                 xy=(index, bleu4_values[index]), 
                 xytext=(index, bleu4_values[index] + 0.005),
                 ha='center', 
                 arrowprops=dict(facecolor='black', shrink=0.05))

# improve layout and rotate x-axis labels for readability
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
