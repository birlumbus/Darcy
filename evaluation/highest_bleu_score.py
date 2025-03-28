import json
import glob
import os

def main():
    threshold = 0.1
    high_score_results = []
    
    print('\nGetting all JSON files...')
    file_list = glob.glob('./prompt_results/json/*.json')
    
    print('Scanning files...')
    for file_path in file_list:
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {file_path}: {e}")
                continue

        # iterate over each prompt object and its outputs in the current file
        for prompt_obj in data:
            for output in prompt_obj.get("outputs", []):
                bleu4 = output.get("evaluation_vs_references", {}).get("bleu4")
                if bleu4 is not None and bleu4 > threshold:
                    # Optionally record the source file for reference
                    output['source_file'] = os.path.basename(file_path)
                    high_score_results.append(output)
    
    # define the output file path for the aggregated results
    
    output_file_name = 'highest_bleu4_results.json'
    output_file_path = f'./prompt_results/json/best_results/{output_file_name}'
    with open(output_file_path, 'w') as out_file:
        json.dump(high_score_results, out_file, indent=2)
    
    print(f"Saved {len(high_score_results)} results with BLEU-4 scores above {threshold} to {output_file_name}\n")

if __name__ == "__main__":
    main()
