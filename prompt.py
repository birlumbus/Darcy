from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# load model
model_path = "./model/darcy-gpt"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# disables dropout layers
model.eval()


def generate_text(prompt, max_length=100):
    # tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95, top_k=50)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# test prompt
prompt = input("Enter prompt:\n")
generated = generate_text(prompt, max_length=150)
print(f"Generated text:\n{generated}")
