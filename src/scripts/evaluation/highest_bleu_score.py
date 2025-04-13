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

        # iterate over prompt objects in current file
        for prompt_obj in data:
            # retrieve prompt text
            prompt_text = prompt_obj.get("prompt", "No prompt provided")
            
            # iterate over outputs in current prompt object
            for output in prompt_obj.get("outputs", []):
                bleu4 = output.get("evaluation_vs_references", {}).get("bleu4")
                if bleu4 is not None and bleu4 > threshold:
                    # record source file for reference
                    output['source_file'] = os.path.basename(file_path)
                    # add prompt text as new property in the output
                    output['prompt'] = prompt_text
                    high_score_results.append(output)
    
    # define output file path for aggregated results
    output_file_name = 'highest_bleu4_results.json'
    output_file_path = f'../../evaluation_data/prompt_results/json/best_results/{output_file_name}'
    with open(output_file_path, 'w') as out_file:
        json.dump(high_score_results, out_file, indent=2)
    
    print(f"Saved {len(high_score_results)} results with BLEU-4 scores above {threshold} to {output_file_name}\n")

if __name__ == "__main__":
    main()
