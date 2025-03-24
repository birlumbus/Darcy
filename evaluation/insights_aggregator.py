import os
import json
import glob
import statistics


# ---------------------------------
# Helper Functions (reduces nesting)
# ---------------------------------


def is_no_output(output_text):
    """
    Determine if an output object contains no output.
    """
    return output_text.strip() == "[[no output]]"


def init_model_version(results, model, version):
    """
    Initialize a nested dictionary for a given model and version.
    """
    if model not in results:
        results[model] = {}
    if version not in results[model]:
        results[model][version] = {
            "metrics": {
                "perplexity": [],
                "references": {
                    "bleu1": [],
                    "bleu2": [],
                    "bleu4": [],
                    "rouge_l": [],
                    "meteor": []
                },
                "baseline": {  # Only applicable for versions other than "0"
                    "bleu1": [],
                    "bleu2": [],
                    "bleu4": [],
                    "rouge_l": [],
                    "meteor": []
                }
            },
            "no_output_count": 0,
            "total": 0
        }


def process_output(out, model, version, results):
    """
    Process a single output entry:
      - Count total outputs.
      - If marked as no output, increment no_output_count.
      - Otherwise, add perplexity and evaluation metrics separately.
    """
    init_model_version(results, model, version)
    results[model][version]["total"] += 1

    output_text = out.get("output", "").strip()
    if is_no_output(output_text):
        results[model][version]["no_output_count"] += 1
        return  # No evaluation metrics are available for no-output cases.

    # record perplexity (always stored under 'references')
    if "perplexity" in out:
        results[model][version]["metrics"]["references"]["perplexity"].append(out["perplexity"])

    # add evaluation_vs_references metrics
    eval_refs = out.get("evaluation_vs_references", {})
    for metric in ["bleu1", "bleu2", "bleu4", "rouge_l", "meteor"]:
        if metric in eval_refs:
            results[model][version]["metrics"]["references"][metric].append(eval_refs[metric])

    # for non-version "0", add evaluation_vs_baseline metrics separately (if available)
    if str(version) != "0":
        eval_base = out.get("evaluation_vs_baseline", {})
        for metric in ["bleu1", "bleu2", "bleu4", "rouge_l", "meteor"]:
            if metric in eval_base:
                results[model][version]["metrics"]["baseline"][metric].append(eval_base[metric])


def compute_average(values):
    """
    Compute the average of a list of numbers; return None if the list is empty.
    """
    filtered = [v for v in values if v is not None]
    return statistics.mean(filtered) if filtered else None


def compute_aggregated_metrics(results):
    """
    Compute the average metrics for both 'references' and 'baseline'
    for each model and version.
    """
    aggregated = {}
    for model, versions in results.items():
        aggregated[model] = {}
        for version, data in versions.items():
            avg_refs = {metric: compute_average(values)
                        for metric, values in data["metrics"]["references"].items()}
            avg_baseline = {metric: compute_average(values)
                            for metric, values in data["metrics"]["baseline"].items()}
            aggregated[model][version] = {
                "average_metrics": {
                    "references": avg_refs,
                    "baseline": avg_baseline
                },
                "no_output_count": data["no_output_count"],
                "total_samples": data["total"]
            }
    return aggregated


def determine_best_model(aggregated):
    """
    Determine the best overall model/version based on the highest average bleu4 score 
    from evaluation_vs_references.
    """
    best_model = None
    best_version = None
    best_bleu4 = -1
    for model, versions in aggregated.items():
        for version, stats in versions.items():
            bleu4 = stats["average_metrics"]["references"].get("bleu4")
            if bleu4 is not None and bleu4 > best_bleu4:
                best_bleu4 = bleu4
                best_model = model
                best_version = version
    return best_model, best_version, best_bleu4


def read_json_files(input_folder):
    """
    Read all JSON files from the input folder and return a list of data entries.
    Each file is assumed to contain a list of prompt entries.
    """
    all_entries = []
    for filepath in glob.glob(os.path.join(input_folder, "*.json")):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                all_entries.extend(data)
            except json.JSONDecodeError:
                print(f"Skipping file {filepath} due to JSON error.")
    return all_entries


# ------------------------
# Main Aggregation Function
# ------------------------

def aggregate_insights(input_folder, output_file):
    # initialize an empty results dictionary
    results = {}

    # read all prompt entries from JSON files in the folder
    entries = read_json_files(input_folder)

    # process each output in every prompt entry
    for entry in entries:
        outputs = entry.get("outputs", [])
        for out in outputs:
            model = out.get("model")
            version = str(out.get("version"))
            process_output(out, model, version, results)

    # compute averages for all collected metrics
    aggregated = compute_aggregated_metrics(results)

    # determine best overall model/version (based on evaluation_vs_references bleu1 score)
    best_model, best_version, best_bleu4 = determine_best_model(aggregated)

    summary = {
        "aggregated_results": aggregated,
        "best_overall": {
            "model": best_model,
            "version": best_version,
            "average_bleu4": best_bleu4
        }
    }

    # write summary to new JSON file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


def main():
    input_folder = "./prompt_results/json/"
    output_file = "./prompt_results/compiled_analysis/aggregated_results.json"
    print('\nAnalyzing data...')
    aggregate_insights(input_folder, output_file)
    print(f'Saved data at {output_file}\n')


if __name__ == "__main__":
    main()
