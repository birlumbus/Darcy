import prompt_scripts.prompt_gpt2medium as medium
import prompt_scripts.prompt_gpt2large as large
import prompt_scripts.prompt_gptj6b as gptj6b
from evaluation.calculate_perplexity import calculate_perplexity
from evaluation.bleu_rouge_meteor import bleu_rouge_meteor


model_categories = {
    "medium": {
        "category": medium,
        "models": {
            "0": "gpt2-medium",
            "1": "./model/darcy-gpt2-medium-1",
            "2": "./model/darcy-gpt2-medium-2"
        }
    },
    "large": {
        "category": large,
        "models": {
            "0": "gpt2-large",
            "1": "./model/darcy-gpt2-large-1",
            "2": "./model/darcy-gpt2-large-2"
        }
    },
    "6b": {
        "category": gptj6b,
        "models": {
            "0": "EleutherAI/gpt-j-6B",
            "1": "./model/darcy-gptj-6b-1",
            "2": "./model/darcy-gptj-6b-2",
            "2.1": "./model/darcy-gptj-6b-2.1"
        }
    }
}


def load_dialogue_references(dialogue_files):
    """
    Load reference dialogue passages from three files into a list.
    
    Each file should contain a single line representing a coherent passage of Darcy's dialogue.
    Ideally, each passage is long enough to capture a complete thought – around 40-60 words is suggested.

    Parameters:
        dialogue_files (list): a list of file paths, each to a to passages of dialogue

    Returns:
        references (list): list containing the text from dialogue_files
    """
    references = []
    for file in dialogue_files:
        try:
            with open(file, "r") as f:
                references.append(f.read().strip())
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return references


def load_models_for_category(category):
    """
    Uses the category associated with a category to load models.
    
    Parameters:
        category (str): The category name (e.g. "medium").
        
    Returns:
        dict: Mapping of model IDs to (model, tokenizer) tuples.
    """
    cat_info = model_categories[category]
    mod = cat_info["category"]
    paths = cat_info["models"]
    return mod.load_models(paths)


def parse_selection(selection):
    """
    Parse user selection and return a dict mapping category to list of model IDs.
    
    Acceptable formats:
      - "all"                -> all models from all categories
      - "medium"             -> all models in medium
      - "large:1"            -> only model "1" from large
      - "6b:all"             -> all models in 6b
      - Comma-separated values, e.g. "medium:1, 6b"
    """
    selection = selection.strip().lower()
    if selection == "all":
        result = {}
        for category, cat_info in model_categories.items():
            result[category] = list(cat_info["models"].keys())
        return result
    
    result = {}
    parts = selection.split(',')
    for part in parts:
        part = part.strip()
        if ':' in part:
            cat, mod = part.split(':', 1)
            cat = cat.strip()
            mod = mod.strip()
            if cat in model_categories:
                if mod == "all":
                    result[cat] = list(model_categories[cat]["models"].keys())
                else:
                    if mod in model_categories[cat]["models"]:
                        result.setdefault(cat, []).append(mod)
                    else:
                        print(f"Warning: Model '{mod}' not found in category '{cat}'.")
            else:
                print(f"Warning: Category '{cat}' not recognized.")
        else:
            cat = part
            if cat in model_categories:
                result[cat] = list(model_categories[cat]["models"].keys())
            else:
                print(f"Warning: Category '{cat}' not recognized.")
    return result


def prompt_single_model(
    category, model_id, prompt_text, loaded_models, 
    base_outputs=None, dialogue_references=None
):
    """
    Prompts a single model and computes its perplexity and BLEU/ROUGE/METEOR scores if applicable.
    
    Parameters:
        category (str): Model category (e.g., "medium").
        model_id (str): Model ID (e.g., "1").
        prompt_text (str): Text to prompt the model with.
        loaded_models (dict): Dictionary mapping model IDs to (model, tokenizer) tuples.
        base_outputs (dict, optional): Stores base model outputs for later BLEU/ROUGE/METEOR comparisons. Defaults to empty dict.
        dialogue_references (list, optional): List of reference dialogue texts for BLEU/ROUGE/METEOR. Defaults to empty list.

    Returns:
        str: Generated model output and relevant metrics (if applicable).
    """
    base_outputs = base_outputs if base_outputs is not None else {}
    dialogue_references = dialogue_references if dialogue_references is not None else []

    if model_id not in loaded_models:
        return f"From {category}-{model_id}:\nModel not found.\n\n"
    
    print(f"\nPROMPTING MODEL: {category}-{model_id}\n")
    model, tokenizer = loaded_models[model_id]

    # generate text
    mod = model_categories[category]["category"]
    output_text = mod.generate_text(prompt_text, model, tokenizer, max_length=256)
    print()  # Line break

    # compute perplexity
    perplexity_val = calculate_perplexity(output_text, model, tokenizer)
    model_output_info = f"From {category}-{model_id}:\n{output_text}\nPerplexity: {perplexity_val}\n"

    # compute BLEU/ROUGE/METEOR scores for non-base models
    if model_id != "0":
        if category in base_outputs:
            reference_texts = [base_outputs[category]] + dialogue_references
            bleu, rouge, meteor = bleu_rouge_meteor(reference_texts, output_text)
            model_output_info += f"BLEU: {bleu}\nROUGE: {rouge}\nMETEOR: {meteor}\n"
        else:
            model_output_info += "Base model output not available for BLEU/ROUGE/METEOR comparison.\n"
    else:
        base_outputs[category] = output_text  # Store base model output

    return model_output_info


def process_category(category, prompt_text, base_outputs, dialogue_references):
    """
    Processes all models in a category and returns their output.
    """
    loaded_models = load_models_for_category(category)
    model_outputs = [
        prompt_single_model(category, model_id, prompt_text, loaded_models, base_outputs, dialogue_references)
        for model_id in model_categories[category]["models"]
    ]
    return model_outputs


def file_mode(file_path, output_file_path, dialogue_files):
    """
    Processes prompts from a file and evaluates model outputs.
    """
    with open(file_path, "r") as f:
        prompt_list = [line.strip() for line in f if line.strip()]

    dialogue_references = load_dialogue_references(dialogue_files)
    base_outputs = {}

    final_outputs = []

    # process each prompt
    for prompt_text in prompt_list:
        print(f"\nNEW PROMPT:\n{prompt_text}\n")
        output_for_prompt = [f"Prompt text:\n{prompt_text}\n"]

        # process each category in parallel
        for category in model_categories:
            output_for_prompt.extend(
                process_category(category, prompt_text, base_outputs, dialogue_references)
            )

        final_outputs.append("\n".join(output_for_prompt))

    # save results to file
    result_text = ("\n" + "-" * 80 + "\n\n").join(final_outputs)
    try:
        with open(output_file_path, "w") as f:
            f.write(result_text)
        print(f"\nResults saved to {output_file_path}\n")
    except Exception as e:
        print(f"\nError writing to file: {e}\n")


def interactive_mode():
    """
    Interactive mode prompts model(s) selected by the user and prints output to the console.
    No testing metrics (perplexity, BLEU/ROUGE/METEOR) are computed in this mode.
    """
    print("Available model categories and models:")
    for category, cat_info in model_categories.items():
        model_ids = ", ".join(cat_info["models"].keys())
        print(f"  {category}: {model_ids}")
    
    print("\nEnter your selection.")
    print("Examples:")
    print("  all")
    print("  medium")
    print("  large:1")
    print("  6b:2.1, large")
    print("  (ctrl-c to exit)")
    
    selected_dict = None
    while not selected_dict:
        selection_input = input("Your selection: ")
        selected_dict = parse_selection(selection_input)
        if not selected_dict:
            print("Invalid input. Try again.")

    prompt_text = input("\nEnter prompt:\n")
    print()
    
    # no file references or base outputs needed for interactive mode
    dialogue_references = []
    base_outputs = {}

    outputs = {}
    
    # iterate over selected category and model
    for category, model_ids in selected_dict.items():
        loaded_models = load_models_for_category(category)
        
        for model_id in model_ids:
            outputs[f"{category}-{model_id}"] = prompt_single_model(
                category, model_id, prompt_text, loaded_models, base_outputs, dialogue_references
            )

    # print results
    print("\nGenerated outputs:")
    for key, text in outputs.items():
        print(f"\n{text}\n")


def main():
    print("Select mode:")
    print("  [s] Interactive mode (single prompt to model(s) of your choice)")
    print(f"  [f] File mode (submit prompts found in {file_path})")
    mode_choice = input("Your selection (default is s): ").strip().lower()
    
    if mode_choice == "f":
        # file init
        file_path = "./test_prompt.txt"
        output_file_path = "./test_result.txt"
        dialogue_files = [
            "./evaluation/reference_dialogue/darcy_dialogue_1.txt",
            "./evaluation/reference_dialogue/darcy_dialogue_2.txt",
            "./evaluation/reference_dialogue/darcy_dialogue_3.txt"
        ]
        file_mode(file_path, output_file_path, dialogue_files)
    else:
        interactive_mode()


if __name__ == '__main__':
    main()
