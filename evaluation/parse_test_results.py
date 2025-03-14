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
        averages[group] = {}
        for section, metrics in sections.items():
            averages[group][section] = {}
            for metric, values in metrics.items():
                avg = sum(values) / len(values) if values else 0
                averages[group][section][metric] = round(avg, 5)
    return averages


def save_results(file_path, results_as_json):
	try:
		with open(file_path, "w") as f:
			f.write(results_as_json)
		print("Done\n")
	except Exception as e:
		print(f"\nError saving results to file: {e}\n")


def main():
    input_file_path = "./evaluation/prompt_materials/test_results_4.txt"
    output_file_path = "./evaluation/prompt_materials/compiled_results/analysis_4.json"
    blocks = parse_file(input_file_path)
    groups = group_scores(blocks)
    averages = compute_averages(groups)
    # print the averages as a JSON-formatted string
    results_as_json = json.dumps(averages, indent=2)
    print(results_as_json)
    print(f"\nSaving results to {output_file_path}...")
    save_results(output_file_path, results_as_json)
    

if __name__ == "__main__":
    main()
