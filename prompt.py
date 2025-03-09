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


def run_tests_for_model(category, model_id, output_text, base_outputs, dialogue_references, model, tokenizer):
    """
    Computes perplexity along with BLEU, ROUGE, and METEOR scores.
    
    For a base model (model_id "0"), it tests against dialogue_references.
    For non-base models, it uses the base model's output (if available) plus dialogue_references.
    
    Parameters:
        category (str): The model category.
        model_id (str): The model identifier.
        output_text (str): The text generated by the model.
        base_outputs (dict): Dictionary storing outputs from base models.
        dialogue_references (list): List of reference dialogue texts.
        model: The model instance.
        tokenizer: The tokenizer instance.
    
    Returns:
        str: A formatted string containing the perplexity and test metric results.
    """
    if not output_text or not output_text.strip():
        return "[[no output]]"

    perplexity_val = calculate_perplexity(output_text, model, tokenizer)

    if model_id == "0":
        # for base models, test against dialogue references only
        bleu, rouge, meteor = bleu_rouge_meteor(dialogue_references, output_text)
        # also, store the base model's output for future comparisons
        base_outputs[category] = output_text
        return f"\nPerplexity: {perplexity_val}\nBLEU: {bleu}\nROUGE: {rouge}\nMETEOR: {meteor}\n"

    # for non-base models, test against base model output plus dialogue references
    bleu_base, rouge_base, meteor_base = bleu_rouge_meteor([base_outputs[category]], output_text)
    bleu_ref, rouge_ref, meteor_ref = bleu_rouge_meteor(dialogue_references, output_text)
    return f"""
    Perplexity: {perplexity_val}

    Scores against base model output:
    BLEU: {bleu_base}
    ROUGE: {rouge_base}
    METEOR: {meteor_base}

    Scores against dialogue references:
    BLEU: {bleu_ref}
    ROUGE: {rouge_ref}
    METEOR: {meteor_ref}
    """


def prompt_single_model(
    category, model_id, prompt_text, loaded_models, 
    base_outputs=None, dialogue_references=None
):
    """
    Prompts a single model.

    If dialogue_references is provided (as in file_mode), test metrics are computed.
    Otherwise (interactive mode) no tests are run and base model outputs are not preserved.
    """
    base_outputs = base_outputs if base_outputs is not None else {}
    dialogue_references = dialogue_references if dialogue_references is not None else []

    if model_id not in loaded_models:
        return f"From {category}-{model_id}:\nModel not found.\n\n"
    
    print(f"\nPROMPTING MODEL: {category}-{model_id}\n")
    model, tokenizer = loaded_models[model_id]

    mod = model_categories[category]["category"]
    output_text = mod.generate_text(prompt_text, model, tokenizer, max_length=256)
    print() # line break

    # only run tests if dialogue_references are provided (i.e. file_mode)
    if dialogue_references:
        test_metrics = run_tests_for_model(category, model_id, output_text, base_outputs, dialogue_references, model, tokenizer)
        model_output_info = f"From {category}-{model_id}:\n{output_text}\n{test_metrics}"
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


def safe_save(output_file_path, result_text):
    """
    Attempts to save result_text to output_file_path. If an error occurs, prints an error message.
    """
    try:
        with open(output_file_path, "w") as f:
            f.write(result_text)
        print(f"\nResults saved to {output_file_path}\n")
    except Exception as e:
        print(f"\nError saving results to file: {e}\n")


def file_mode(file_path, output_file_path, dialogue_files):
    """
    Processes prompts from a file and evaluates model outputs. In case of an error during processing,
    attempts to save the progress made so far.
    """
    with open(file_path, "r") as f:
        prompt_list = [line.strip() for line in f if line.strip()]

    dialogue_references = load_dialogue_references(dialogue_files)
    base_outputs = {}
    final_outputs = []

    try:
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
    except Exception as e:
        print(f"\nError encountered during processing: {e}\n")
        # attempt to save partial results
        result_text = ("\n" + "-" * 80 + "\n\n").join(final_outputs)
        file_root, file_ext = os.path.splitext(output_file_path)
        incomplete_file_path = file_root + "_INCOMPLETE" + file_ext
        safe_save(incomplete_file_path, result_text)
        raise

    # save results if processing finished without error
    result_text = ("\n" + "-" * 80 + "\n\n").join(final_outputs)
    safe_save(output_file_path, result_text)


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
    print()


def main():
    file_path = "./prompt_materials/test_prompts.txt"
    output_file_path = "./prompt_materials/test_results.txt"
    dialogue_files = [
        "./evaluation/reference_dialogue/darcy_dialogue_1.txt",
        "./evaluation/reference_dialogue/darcy_dialogue_2.txt",
        "./evaluation/reference_dialogue/darcy_dialogue_3.txt"
    ]

    print("Select mode:")
    print("  [s] Interactive mode (single prompt to model(s) of your choice)")
    print(f"  [f] File mode (submit prompts found in {file_path})")
    mode_choice = input("Your selection (default is s): ").strip().lower()
    
    if mode_choice == "f":
        file_mode(file_path, output_file_path, dialogue_files)
    else:
        interactive_mode()


if __name__ == '__main__':
    main()
