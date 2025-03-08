import torch
import math

def calculate_perplexity(model, tokenizer, text, max_length=512):
    """
    Compute the perplexity of a given text using the provided model and tokenizer.
    
    Parameters:
        model: The language model.
        tokenizer: The corresponding tokenizer.
        text (str): Input text to evaluate.
        max_length (int): Maximum sequence length to evaluate (truncation).
    
    Returns:
        float: The perplexity score.
    """

    # tokenize input text
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(model.device)
    
    # compute loss with no gradient; reduces redundant operational overhead
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss # loss per token
    
    # perplexity: the exponential of the loss
    perplexity = math.exp(loss.item())
    return perplexity
