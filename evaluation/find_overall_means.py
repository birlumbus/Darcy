import json
import numpy as np


metrics_to_average = ['perplexity', 'bleu1', 'bleu2', 'bleu4', 'rouge_l', 'meteor']

print("\nLoading in aggregated_results...")
in_file = 'prompt_results/compiled_analysis/aggregated_results.json'
with open(in_file, 'r') as f:
    data = json.load(f)

aggregated_results = data['aggregated_results']


def calculate_mean(metrics_list):
    means = {}
    for metric in metrics_to_average:
        metric_values = [m.get(metric, 0) for m in metrics_list if m.get(metric) is not None]
        means[metric] = np.mean(metric_values) if metric_values else None
    return means


def percentage_change(new_value, old_value):
    if old_value == 0 or old_value is None:
        return None  # Avoid division by zero or None
    return ((new_value - old_value) / old_value) * 100


results = {}

print('Calculating means...')
for model, versions in aggregated_results.items():
    results[model] = {}

    # extract version 0 metrics
    version_0_metrics['perplexity'] = versions['0']['average_metrics']['perplexity']
    version_0_metrics = versions['0']['average_metrics']['references']
    
    results[model]['0'] = version_0_metrics

    # grouping version sets
    version_1_metrics = []
    version_2_metrics = []

    for version, info in versions.items():
        if version in ['1', '1.1', '1.2']:
            metrics = info['average_metrics']['references']
            metrics['perplexity'] = info['average_metrics']['perplexity']
            version_1_metrics.append(metrics)

        elif version in ['2', '2.1', '2.2']:
            metrics = info['average_metrics']['references']
            metrics['perplexity'] = info['average_metrics']['perplexity']
            version_2_metrics.append(metrics)

    # calculate means for each version set
    version_1_mean = calculate_mean(version_1_metrics)
    version_2_mean = calculate_mean(version_2_metrics)
    overall_mean = calculate_mean(version_1_metrics + version_2_metrics)

    results[model]['version_1_mean'] = version_1_mean
    results[model]['version_2_mean'] = version_2_mean
    results[model]['overall_mean'] = overall_mean

    # calculate percentage changes from version 0
    percentage_changes = {}
    for key, mean_metrics in [('version_1_mean', version_1_mean),
                              ('version_2_mean', version_2_mean),
                              ('overall_mean', overall_mean)]:
        percentage_changes[key] = {
            metric: percentage_change(mean_metrics.get(metric), version_0_metrics.get(metric))
            for metric in metrics_to_average
        }

    results[model]['percentage_changes'] = percentage_changes


# save to JSON file
out_file = 'aggregated_means.json'
out_path = f'prompt_results/compiled_analysis/{out_file}'
compiled_results = {
    "aggregated_means": results
}


print(f'Saving to {out_file}...')
with open(out_path, 'w') as f:
    json.dump(compiled_results, f, indent=4)
print('Done\n')



