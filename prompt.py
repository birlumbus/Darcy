from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the trained model
model_path = "./darcy_gpt2"  # Path to your fine-tuned model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set model to evaluation mode (disables dropout layers)
model.eval()

# Function to generate text from a prompt
def generate_text(prompt, max_length=100):
    # Tokenize input prompt
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    # Ensure model doesn't generate too long sequences
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95, top_k=50)
    
    # Decode and return the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Test prompt
prompt = "My dearest Elizabeth, "
generated = generate_text(prompt, max_length=150)
print(f"Generated text:\n{generated}")
