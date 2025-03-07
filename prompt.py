import prompt_scripts.prompt_gpt2medium as medium
import prompt_scripts.prompt_gpt2large as large
import prompt_scripts.prompt_gptj6b as gptj6b


# Define available models by category.
# Each entry maps the category name to a dictionary that contains:
#   - The module for that category (which has load_models and generate_text).
#   - A models dict mapping model IDs to their paths.
model_categories = {
    "gpt2medium": {
        "module": medium,
        "models": {
            "0": "gpt2-medium",
            "1": "./model/darcy-gpt2-medium-1",
            "2": "./model/darcy-gpt2-medium-2"
        }
    },
    "gpt2large": {
        "module": large,
        "models": {
            "0": "gpt2-large",
            "1": "./model/darcy-gpt2-large-1",
            "2": "./model/darcy-gpt2-large-2"
        }
    },
    "gptj6b": {
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
      - "gpt2medium"         -> all models in gpt2medium
      - "gpt2large:1"        -> only model "1" from gpt2large
      - "gpt2medium:all"     -> all models in gpt2medium
      - Comma-separated values, e.g. "gpt2medium:1, gptj6b"
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


if __name__ == '__main__':
    # display available categories and model IDs
    print("Available model categories and models:")
    for category, cat_info in model_categories.items():
        model_ids = ", ".join(cat_info["models"].keys())
        print(f"  {category}: {model_ids}")
    
    print("\nEnter your selection.")
    print("Examples:")
    print("  all")
    print("  gpt2medium")
    print("  gpt2large:1")
    print("  gpt2medium:1, gptj6b")
    print("  (ctrl-c to exit)")
    selected_dict = None
    
    while not selected_dict:
        selection_input = input("Your selection: ")
        selected_dict = parse_selection(selection_input)
        
        if not selected_dict:
            print("Invalid input.")
    
    prompt_text = input("\nEnter prompt:\n")
    
    outputs = {}
    # Iterate over each selected category and model.
    for category, model_ids in selected_dict.items():
        # Load the models for this category using the module’s load_models() function.
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
