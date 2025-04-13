import json


'''
DEPRECATED
To be run if unwanted bleu 1,2,4 properties exist
'''


INPUT_FILE = '../../evaluation_data/prompt_results/json/prompt_results_6.json'


def remove_bleu_metrics(data):
    keys_to_remove = ["bleu1", "bleu2", "bleu4"]
    for entry in data:
        for output in entry.get("outputs", []):
            for key in keys_to_remove:
                output.pop(key, None)  # Remove key if exists
    return data

def main():
    # load JSON data
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # remove specified keys from each output object
    cleaned_data = remove_bleu_metrics(data)

    # write cleaned data back to file
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2)

if __name__ == "__main__":
    main()
