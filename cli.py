import sys
import subprocess
import os

# Map short names to script paths relative to project root
COMMANDS = {
    # model training
    "train-gpt2medium-1": "src/scripts/training/model_training/train_gpt2medium_1.py",
    "train-gpt2medium-2": "src/scripts/training/model_training/train_gpt2medium_1.py",
    "train-gpt2large-1": "src/scripts/training/model_training/train_gpt2large_1.py",
    "train-gpt2large-2": "src/scripts/training/model_training/train_gpt2large_2.py",
    "train-gptj-1": "src/scripts/training/model_training/train_gptj6b_1.py",
    "train-gptj-2": "src/scripts/training/model_training/train_gptj6b_1.py",
    # prompting/evaluation
    "prompt": "prompt.py",
    "convert-results-json": "src/scripts/evaluation/results_txt_to_json.py",  # requires an arg
    "evaluate-metrics": "src/scripts/evaluation/bleu_rouge_meteor.py",
    "assemble-results": "src/scripts/evaluation/insights_aggregator.py",
    "prep-for-graphing": "src/scripts/evaluation/find_overall_means.py",
    "collect-best-results": "src/scripts/evaluation/highest_bleu_score.py",
    "graph-vitals": "src/scripts/evaluation/graph_metrics_vs_base.py",
    "graph-all": "src/scripts/evaluation/graph_results.py",
    "test-perplexity-script": "src/scripts/evaluation/calculate_perplexity.py",
    # dataset_builders
    "build-dataset-1": "src/scripts/training/data_prep/build_dataset_1.py",
    "build-dataset-2": "src/scripts/training/data_prep/build_dataset_2.py"
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 cli.py <command> [args]")
        print("\nAvailable commands:")
        for cmd in COMMANDS:
            print(f"  {cmd}")
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command not in COMMANDS:
        print(f"Unknown command: {command}")
        print("Use one of:")
        for cmd in COMMANDS:
            print(f"  {cmd}")
        sys.exit(1)

    script_path = COMMANDS[command]

    # Run the script + optional args
    subprocess.run(["python3", script_path] + args)

if __name__ == "__main__":
    main()
