import json


# list of metric keys to average
metrics_to_average = ['bleu1', 'bleu2', 'bleu4', 'rouge_l', 'meteor']


# load aggregated results from JSON file
in_file = 'prompt_results/compiled_analysis/aggregated_results.json'
with open(in_file, 'r') as f:
    data = json.load(f)

aggregated_results = data['aggregated_results']


# --- set a ---
print("\nExtracting means; all models...")
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


# --- set b ---
print("Extracting means; excluding 6b...")
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


# save results to JSON file
out_file = 'aggregated_means.json'
out_path = f'prompt_results/compiled_analysis/{out_file}'
compiled_results = {
    "averages_excluding_version_0": averages_all,
    "averages_excluding_version_0_and_6b_models": averages_no_6b
}


print(f'Saving to {out_file}...')
with open(out_path, 'w') as f:
    json.dump(compiled_results, f, indent=4)
print('Done\n')
