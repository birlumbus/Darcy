import re
from collections import defaultdict
import json


def parse_file(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()

    # blocks correspond to each model output section
    blocks = []
    current_block = None
    current_section = "overall"  # default section if not under a header

    # pattern for a new block: e.g., "From medium-0:"
    model_pattern = re.compile(r'^From\s+([\w\d\.-]+):')
    # pattern for a metric line, e.g., "Perplexity: 19.180458068847656"
    metric_pattern = re.compile(r'^(\w+):\s*([\d.eE+-]+)')
    # section headers for additional score sets
    base_section_pattern = re.compile(r'^Scores against base model output:')
    dialogue_section_pattern = re.compile(r'^Scores against dialogue references:')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # check if the line starts a new model block
        m_model = model_pattern.match(line)
        if m_model:
            # save the previous block (if any)
            if current_block is not None:
                blocks.append(current_block)
            current_block = {"model": m_model.group(1), "sections": {"overall": {}}}
            current_section = "overall"
            continue

        # check for section header changes
        if base_section_pattern.match(line):
            current_section = "base"
            if "base" not in current_block["sections"]:
                current_block["sections"]["base"] = {}
            continue
        if dialogue_section_pattern.match(line):
            current_section = "dialogue"
            if "dialogue" not in current_block["sections"]:
                current_block["sections"]["dialogue"] = {}
            continue

        # check if the line contains a metric
        m_metric = metric_pattern.match(line)
        if m_metric:
            metric = m_metric.group(1)
            try:
                value = float(m_metric.group(2))
            except ValueError:
                continue
            # save the metric under the current section
            current_block["sections"].setdefault(current_section, {})[metric] = value

    # add the last block if it exists
    if current_block is not None:
        blocks.append(current_block)
    return blocks


def group_scores(blocks):
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for block in blocks:
        model_id = block["model"]
        for section, metrics in block["sections"].items():
            for metric, value in metrics.items():
                groups[model_id][section][metric].append(value)
    return groups


def compute_averages(groups):
    # compute the average for each metric and round to five decimal places
    averages = {}
    for group, sections in groups.items():
        # split model identifier into category and version parts
        parts = group.split('-', 1)
        category = parts[0]
        version = parts[1] if len(parts) > 1 else ""
        
        averages[group] = {
            "category": category,
            "version": version
        }
        
        for section, metrics in sections.items():
            averages[group][section] = {}
            for metric, values in metrics.items():
                avg = sum(values) / len(values) if values else 0
                averages[group][section][metric] = round(avg, 5)
    return averages


def aggregate_by_version(results):
    """
    Takes existing results dictionary, adds new objects that are aggregated across models 
    sharing same "version" property. For each unique version, it computes average of metrics 
    across all models with that version and creates a new JSON object with key "version-{version}"
    """
    # group model keys by version
    version_groups = {}
    for model_key, data in results.items():
        version = data.get("version", "")
        if not version:
            continue
        version_groups.setdefault(version, []).append(model_key)

    # for each version, aggregate metrics across models
    aggregated_results = {}
    for version, model_keys in version_groups.items():
        # dictionary holds lists of values for each section and metric
        section_metrics = {}
        for key in model_keys:
            model_data = results[key]
            # process each section (skip "category" and "version" keys)
            for section, metrics in model_data.items():
                if section in ("category", "version"):
                    continue
                section_metrics.setdefault(section, {})
                for metric, value in metrics.items():
                    section_metrics[section].setdefault(metric, []).append(value)
        
        # compute averages per section/metric
        aggregated_sections = {}
        for section, metrics in section_metrics.items():
            aggregated_sections[section] = {}
            for metric, values in metrics.items():
                avg = sum(values) / len(values) if values else 0
                aggregated_sections[section][metric] = round(avg, 5)
        
        # create new aggregated object
        aggregated_results[f"version-{version}"] = {
            "category": "aggregated",
            "version": version,
            "models": model_keys,
            **aggregated_sections
        }
    
    # add aggregated objects to original results
    results.update(aggregated_results)
    return results


def save_results(file_path, results_as_json):
    try:
        with open(file_path, "w") as f:
            f.write(results_as_json)
        print("Done\n")
    except Exception as e:
        print(f"\nError saving results to file: {e}\n")


def main():
    input_file_path = "./prompt_materials/test_results_4.txt"
    output_file_path = "./prompt_materials/compiled_results/analysis_4.json"
    blocks = parse_file(input_file_path)
    groups = group_scores(blocks)
    averages = compute_averages(groups)
    
    # aggregate computed averages
    final_results = aggregate_by_version(averages)
    
    # dump final results as JSON
    results_as_json = json.dumps(final_results, indent=2)
    print(results_as_json)
    print(f"\nSaving results to {output_file_path}...")
    save_results(output_file_path, results_as_json)
    

if __name__ == "__main__":
    main()
