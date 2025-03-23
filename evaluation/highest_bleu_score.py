import json


def main():
    # open and load JSON file
    file_path = './prompt_results/json/prompt_results_1.json'
    with open(file_path, 'r') as f:
        data = json.load(f)

    max_bleu4 = -1
    max_output = None

    # iterate over each prompt object and its outputs
    for prompt_obj in data:
        for output in prompt_obj.get("outputs", []):
            # get bleu4 score from evaluation_vs_references dict (if available)
            bleu4 = output.get("evaluation_vs_references", {}).get("bleu4")
            if bleu4 is not None and bleu4 > max_bleu4:
                max_bleu4 = bleu4
                max_output = output

    # print results
    print("\nHighest bleu-4 score:", max_bleu4)
    print("Associated JSON object:")
    print(json.dumps(max_output, indent=2))
    print()


if __name__ == "__main__":
    main()
