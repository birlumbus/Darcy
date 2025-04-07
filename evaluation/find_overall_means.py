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
        means[metric] = round(np.mean(metric_values), 2) if metric_values else None
    return means


def percentage_change(new_value, old_value):
    if old_value == 0 or old_value is None:
        return None  # avoid division by zero or None
    return round(((new_value - old_value) / old_value) * 100)


results = {}

print('Calculating means...')
for model, versions in aggregated_results.items():
    results[model] = {}

    # extract version 0 metrics without rounding
    version_0_metrics = {'perplexity': versions['0']['average_metrics']['perplexity']}
    references = versions['0']['average_metrics']['references']
    version_0_metrics.update(references)
    results[model]['0'] = version_0_metrics

    # grouping version sets for group means
    version_1_metrics = []
    version_2_metrics = []

    # compute individual means for each non-zero version
    individual_means = {}

    for version, info in versions.items():
        if version == '0':
            continue

        # create dictionary individual_means of metrics for this version
        ind_metrics = info['average_metrics']['references'].copy()
        # include raw perplexity
        ind_metrics['perplexity'] = info['average_metrics']['perplexity']
        # store in individual_means dict
        individual_means[version] = ind_metrics

        # group into version sets for group means
        if version in ['1', '1.1', '1.2']:
            version_1_metrics.append(ind_metrics)
        elif version in ['2', '2.1', '2.2']:
            version_2_metrics.append(ind_metrics)

    # calculate group means for version 1 and 2 (rounding occurs in calculate_mean)
    group_version_1_mean = calculate_mean(version_1_metrics)
    group_version_2_mean = calculate_mean(version_2_metrics)
    overall_mean = calculate_mean(version_1_metrics + version_2_metrics)

    results[model]['version_1_mean'] = group_version_1_mean
    results[model]['version_2_mean'] = group_version_2_mean
    results[model]['overall_mean'] = overall_mean

    # store individual version means under a separate key, rounding them here after overall means are computed
    results[model]['individual_means'] = {}
    for version, ind_mean in individual_means.items():
        rounded_ind_mean = {metric: round(ind_mean.get(metric), 4) if ind_mean.get(metric) is not None else None for metric in metrics_to_average}  # new code: round each metric
        results[model]['individual_means'][f"version_{version}_mean"] = rounded_ind_mean

    # calculate percentage changes from version 0 for group means
    percentage_changes = {}
    for key, mean_metrics in [('version_1_mean', group_version_1_mean),
                              ('version_2_mean', group_version_2_mean),
                              ('overall_mean', overall_mean)]:
        percentage_changes[key] = {
            metric: percentage_change(mean_metrics.get(metric), version_0_metrics.get(metric))
            for metric in metrics_to_average
        }

    results[model]['percentage_changes'] = percentage_changes

# save to file
out_file = 'aggregated_means.json'
out_path = f'prompt_results/compiled_analysis/{out_file}'
compiled_results = {"aggregated_means": results}

print(f'Saving to {out_file}...')
with open(out_path, 'w') as f:
    json.dump(compiled_results, f, indent=4)
print('Done\n')
