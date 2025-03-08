import warnings
warnings.filterwarnings("ignore", message="loss_type=None was set in the config but it is unrecognised.")


import torch


def calculate_perplexity(generated_text: str, model, tokenizer) -> float:
    """
    Calculates perplexity for generated_text using the provided model and tokenizer.
    
    The function tokenizes generated_text, computes the cross-entropy loss by using the model
    to score generated_text (using the input tokens as labels), and returns the exponentiated loss
    as the perplexity.
    
    Parameters:
        generated_text (str): The text for which to calculate perplexity.
        model: The language model (e.g. a Hugging Face model) that supports a `forward` pass with a 'labels' parameter.
        tokenizer: The tokenizer corresponding to the model.
    
    Returns:
        float: The perplexity score.
    """

    device = next(model.parameters()).device if next(model.parameters(), None) is not None else torch.device("cpu")
    
    # tokenize and encode generated_text
    inputs = tokenizer(generated_text, return_tensors="pt", truncation=True)
    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
    
    with torch.no_grad():
        # compute loss using the model, where input tokens serve as labels
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
    
    # exponentiate loss to get perplexity
    perplexity = torch.exp(loss)
    return perplexity.item()


# if run as a script, allow testing the function with a sample text using GPT-2
if __name__ == '__main__':
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    
    # load a default model and tokenizer (GPT-2 in this case) for demonstration
    model_name = "gpt2"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    
    # ensure model is on the correct device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    sample_text = ("It is a truth universally acknowledged, that a single man in possession "
                   "of a good fortune, must be in want of a wife.")
    ppl = calculate_perplexity(sample_text, model, tokenizer)
    print(f"Perplexity for sample text: {ppl:.2f}")
