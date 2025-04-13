import json


'''
Performance metrics are removed for all outputs where output is '[[no output]]'
This process is no longer expected to be necessary.
'''


def remove_no_output_evaluations(data):
    # iterate over each prompt
    for prompt in data:
        outputs = prompt.get('outputs', [])
        # process each output entry
        for output in outputs:
            if output.get('output') == '[[no output]]':
                # remove eval properties
                output.pop('perplexity', None)
                output.pop('evaluation_vs_references', None)
                output.pop('evaluation_vs_baseline', None)
    return data


def main():
    file_name = 'prompt_results_4.json'
    file_path = f'prompt_results/json/{file_name}'

    # read JSON data from input file
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f'\nCleaning JSON data...')
    cleaned_data = remove_no_output_evaluations(data)

    print(f'Writing new JSON data back to file...')
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2)

    print(f'JSON data written to {file_name}\n')


if __name__ == "__main__":
    main()
