import json


# list of metric keys to average
metrics_to_average = ['bleu1', 'bleu2', 'bleu4', 'rouge_l', 'meteor']


# load aggregated results from JSON file
in_file = 'prompt_results/compiled_analysis/aggregated_results.json'
with open(in_file, 'r') as f:
    data = json.load(f)

aggregated_results = data['aggregated_results']


# --- set A: all models but ignore versions labeled "0" ---
results_all = {metric: [] for metric in metrics_to_average}


# iterate over each model and version, skipping '0'
for model, versions in aggregated_results.items():
    for version, info in versions.items():
        if version == "0":
            continue
        # extract metrics
        ref_metrics = info['average_metrics']['references']
        for metric in metrics_to_average:
            score = ref_metrics.get(metric)
            if score is not None:
                results_all[metric].append(score)


# calculate averages for each metric in set A
averages_all = {
    metric: (sum(scores) / len(scores)) if scores else None 
    for metric, scores in results_all.items()
}


# --- set B: exclude "6b" models and ignore '0' ---
results_no_6b = {metric: [] for metric in metrics_to_average}


# iterate over all models but '6b'
for model, versions in aggregated_results.items():
    if model == "6b":
        continue
    for version, info in versions.items():
        if version == "0":
            continue  # still skip '0'
        # extract metrics
        ref_metrics = info['average_metrics']['references']
        for metric in metrics_to_average:
            score = ref_metrics.get(metric)
            if score is not None:
                results_no_6b[metric].append(score)


# calculate averages for each metric in set B
averages_no_6b = {
    metric: (sum(scores) / len(scores)) if scores else None 
    for metric, scores in results_no_6b.items()
}


# print results
print("Averages excluding version '0':")
for metric, avg in averages_all.items():
    print(f"{metric}: {avg:.4f}")

print("\nAverages excluding version '0' and 6b models:")
for metric, avg in averages_no_6b.items():
    print(f"{metric}: {avg:.4f}")
