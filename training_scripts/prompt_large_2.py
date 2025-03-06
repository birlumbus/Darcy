import sys
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main(prompt):
    # Use the GPT2Large model
    model_name = "gpt2-large"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Move model to GPU if available for better performance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Encode the prompt and move the tensors to the device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate output text. Adjust generation parameters as needed.
    output_ids = model.generate(
        input_ids,
        max_length=150,             # Adjust maximum length as needed
        num_return_sequences=1,
        no_repeat_ngram_size=2,     # Helps to avoid repetitions
        early_stopping=True
    )
    
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)

if __name__ == "__main__":
    # Use command-line argument as prompt, or default to a simple prompt
    prompt = sys.argv[1] if len(sys.argv) > 1 else "Hello, world!"
    main(prompt)
