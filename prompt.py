import prompt_scripts.prompt_gpt2medium as medium
import prompt_scripts.prompt_gpt2large as large
import prompt_scripts.prompt_gptj6b as gptj6b


model_categories = {
    "medium": {
        "module": medium,
        "models": {
            "0": "gpt2-medium",
            "1": "./model/darcy-gpt2-medium-1",
            "2": "./model/darcy-gpt2-medium-2"
        }
    },
    "large": {
        "module": large,
        "models": {
            "0": "gpt2-large",
            "1": "./model/darcy-gpt2-large-1",
            "2": "./model/darcy-gpt2-large-2"
        }
    },
    "6b": {
        "module": gptj6b,
        "models": {
            "0": "EleutherAI/gpt-j-6B",
            "1": "./model/darcy-gptj-6b-1",
            "2": "./model/darcy-gptj-6b-2",
            "2.1": "./model/darcy-gptj-6b-2.1"
        }
    }
}


def load_models_for_category(category):
    """
    Uses the module associated with a category to load models.
    
    Parameters:
        category (str): The category name (e.g. "gpt2medium").
        
    Returns:
        dict: Mapping of model IDs to (model, tokenizer) tuples.
    """
    cat_info = model_categories[category]
    mod = cat_info["module"]
    paths = cat_info["models"]
    return mod.load_models(paths)


def parse_selection(selection):
    """
    Parse user selection and return a dict mapping category to list of model IDs.
    
    Acceptable formats:
      - "all"                -> all models from all categories
      - "medium"         -> all models in medium
      - "large:1"        -> only model "1" from large
      - "6b:all"     -> all models in 6b
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


def file_mode(file_path, output_file_path):
    """
    In file mode, use all models from every category.
    Input and output file paths are set in main()
    All models are prompted in order, one prompt at a time
    Results are saved in that order.
    """
    with open(file_path, "r") as f:
        prompt_list = [line.strip() for line in f if line.strip()]
    
    # in file mode, use all models from every category.
    selected_dict = {}
    for category, cat_info in model_categories.items():
        selected_dict[category] = list(cat_info["models"].keys())
    
    # query all models in order by category then model ID
    final_outputs = []

    # 1. select question 
    for prompt_text in prompt_list:
        print(f"\nNEW PROMPT:\n{prompt_text}\n")

        output_for_prompt = []
        output_for_prompt.append(f"\n\nPrompt text:\n{prompt_text}\n")

        # 2. select category
        for category in model_categories:
            # load models for current category.
            loaded_models = load_models_for_category(category)

            # 3. iterate over models in category
            for model_id in model_categories[category]["models"]:
                mod = model_categories[category]["module"]
                if model_id in loaded_models:
                    print(f"\nPROMPTING NEW MODEL: {category}-{model_id}\n")

                    model, tokenizer = loaded_models[model_id]
                    output_text = mod.generate_text(prompt_text, model, tokenizer, max_length=150)
                    output_for_prompt.append(f"From {category}-{model_id}:\n{output_text}\n\n")
                else:
                    output_for_prompt.append(f"From {category}-{model_id}:\nModel not found.\n\n")
        final_outputs.append("\n".join(output_for_prompt))
    
    # join results for each prompt with a separator
    result_text = ("\n" + "-"*80 + "\n\n").join(final_outputs)
    try:
        with open(output_file_path, "w") as f:
            f.write(result_text)
        print(f"Results saved to {output_file_path}")
    except Exception as e:
        print(f"Error writing to file: {e}")


def interactive_mode():
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
            print("Invalid input.")
    
    prompt_text = input("\nEnter prompt:\n")
    print()
    
    outputs = {}
    # iterate over each selected category and model.
    for category, model_ids in selected_dict.items():
        # use module’s load_models()
        loaded_models = load_models_for_category(category)
        mod = model_categories[category]["module"]
        for model_id in model_ids:
            if model_id in loaded_models:
                model, tokenizer = loaded_models[model_id]
                output_text = mod.generate_text(prompt_text, model, tokenizer, max_length=150)
                outputs[f"{category}-{model_id}"] = output_text
            else:
                outputs[f"{category}-{model_id}"] = "Model not found."
    
    print("\nGenerated outputs:")
    for key, text in outputs.items():
        print(f"\nFrom {key}:\n{text}\n")


def main():
    file_path = "./test_prompts.txt"
    output_file_path = "./results.txt"

    print("Select mode:")
    print("  [s] Interactive mode (single prompt to model(s) of your choice)")
    print(f"  [f] File mode (submit prompts found in {file_path})")
    mode_choice = input("Your selection (default is s): ").strip().lower()
    
    if mode_choice == "f":
        file_mode(file_path, output_file_path)
    else:
        interactive_mode()


if __name__ == '__main__':
    main()
