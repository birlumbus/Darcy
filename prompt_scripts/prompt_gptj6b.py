import warnings
from transformers import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)


from transformers import GPTJForCausalLM, AutoTokenizer, AutoConfig, TextIteratorStreamer
import torch
import threading


def load_model(model_path):
    """
    Loads a GPT-J model and its corresponding tokenizer from the given path.
    
    Parameters:
        model_path (str): Path to the model directory.
        
    Returns:
        tuple: (model, tokenizer) with the model set to evaluation mode.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(model_path)
    config.vocab_size = len(tokenizer)
    model = GPTJForCausalLM.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)
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


def generate_text(prompt, model, tokenizer, max_length=256):
    """
    Generates text using the provided GPT-J model and tokenizer based on the given prompt.
    Streams the generated tokens to give real-time progress updates.
    
    Parameters:
        prompt (str): The text prompt to start generation.
        model: The loaded GPTJForCausalLM model.
        tokenizer: The corresponding AutoTokenizer.
        max_length (int): Maximum length of the generated sequence.
        
    Returns:
        str: Generated text.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # create a streamer that will yield tokens as they are generated.
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # prepare generation arguments including the streamer.
    generation_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "max_length": max_length,
        "num_return_sequences": 1,
        "no_repeat_ngram_size": 2,
        "do_sample": True,
        "top_p": 0.95,
        "top_k": 50,
        "streamer": streamer
    }
    
    # launch generation in a separate thread
    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # print tokens as they are generated
    generated_text = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)  # Progress update on console.
        generated_text += new_text
    
    thread.join()
    return generated_text


def generate_text_multiple(prompt, models_dict, selected_models, max_length=256):
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
    models_paths = {
        "0": "EleutherAI/gpt-j-6B",
        "1": "../model/darcy-gptj-6b-1",
        "2": "../model/darcy-gptj-6b-2",
        "2.1": "../model/darcy-gptj-6b-2.1"
    }
    
    models_dict = load_models(models_paths)

    instructions = """Select model to query:
        for base gpt-j-6B:     '0'    
        for darcy-gptj-6b-1:   '1'
        for darcy-gptj-6b-2:   '2'
        for darcy-gptj-6b-2.1: '2.1'
        for all available:     'all'
        (ctrl-c to exit)
    """
    selected_models = None
    
    # user selects model(s) to use
    while not selected_models:
        selected_input = input(instructions).strip().lower()
        if selected_input == "all":
            selected_models = list(models_dict.keys())
        elif selected_input in models_dict:
            selected_models = [selected_input]
        else:
            print("Invalid input.\n")
    
    # get prompt from user and generate text from selected model(s)
    prompt = input("Enter prompt:\n")
    outputs = generate_text_multiple(prompt, models_dict, selected_models, max_length=150)
    
    # display generated outputs
    for model_key, text in outputs.items():
        print(f"\nGenerated text from model {model_key}:\n{text}\n")
