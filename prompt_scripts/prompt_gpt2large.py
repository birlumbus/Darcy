from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import torch

def load_model(model_path):
    """
    Loads a GPT2 model and its corresponding tokenizer from the given path.
    
    Parameters:
        model_path (str): Path to the model directory.
        
    Returns:
        tuple: (model, tokenizer) with the model set to evaluation mode.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()  # disable dropout layers
    return model, tokenizer

def load_models(models_paths):
    """
    Loads multiple models from a dictionary of paths.
    
    Parameters:
        models_paths (dict): Dictionary with model identifiers as keys and model paths as values.
        
    Returns:
        dict: Mapping of model identifier to a (model, tokenizer) tuple.
    """
    models_dict = {}
    for key, path in models_paths.items():
        models_dict[key] = load_model(path)
    return models_dict

def generate_text(prompt, model, tokenizer, max_length=100):
    """
    Generates text using the provided model and tokenizer based on the given prompt.
    
    Parameters:
        prompt (str): The text prompt to start generation.
        model: The loaded GPT2LMHeadModel.
        tokenizer: The corresponding GPT2TokenizerFast.
        max_length (int): Maximum length of the generated sequence.
        
    Returns:
        str: Generated text.
    """
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # Generate text without gradient tracking
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            do_sample=True,
            top_p=0.95,
            top_k=50
        )
    
    # Decode the generated tokens and return as text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def generate_text_multiple(prompt, models_dict, selected_models, max_length=100):
    """
    Generates text for a given prompt from one or more selected models.
    
    Parameters:
        prompt (str): The text prompt to generate text from.
        models_dict (dict): Dictionary mapping model identifiers to (model, tokenizer) tuples.
        selected_models (list): List of model identifiers to use.
        max_length (int): Maximum length for generated text.
        
    Returns:
        dict: Mapping of model identifier to the generated text.
    """
    results = {}
    for model_key in selected_models:
        if model_key in models_dict:
            model, tokenizer = models_dict[model_key]
            results[model_key] = generate_text(prompt, model, tokenizer, max_length)
        else:
            results[model_key] = "Model not found."
    return results

if __name__ == '__main__':
    # Update the model paths to point to your GPT2-large models
    models_paths = {
        "0": "gpt2-large",
        "1": "../model/darcy-gpt2-large-1",
        "2": "../model/darcy-gpt2-large-2"
    }
    
    models_dict = load_models(models_paths)

    instructions = """Select model to query:
        for base gpt2-large:    '0'   
        for darcy-gpt2-large-1: '1'
        for darcy-gpt2-large-2: '2'
        for all available:      'all'
        (ctrl-c to exit)
    """
    selected_models = None
    
    # User selects which model(s) to use
    while not selected_models:
        selected_input = input(instructions).strip().lower()
        if selected_input == "all":
            selected_models = list(models_dict.keys())
        elif selected_input in models_dict:
            selected_models = [selected_input]
        else:
            print("Invalid input.\n")
    
    # Get the prompt from the user and generate text from the selected model(s)
    prompt = input("Enter prompt:\n")
    outputs = generate_text_multiple(prompt, models_dict, selected_models, max_length=150)
    
    # Display the generated outputs
    for model_key, text in outputs.items():
        print(f"\nGenerated text from {model_key}:\n{text}\n")
