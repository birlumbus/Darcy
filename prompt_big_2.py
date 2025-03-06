from transformers import GPTJForCausalLM, AutoTokenizer, AutoConfig
import torch


model_path = "./model/darcy-gptj-6b-2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


config = AutoConfig.from_pretrained(model_path)
config.vocab_size = len(tokenizer)
model = GPTJForCausalLM.from_pretrained(model_path, config=config, ignore_mismatched_sizes=True)
model.eval()


def generate_text(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt")
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
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


prompt = input("Enter prompt:\n")
generated = generate_text(prompt, max_length=150)
print(f"Generated text:\n{generated}")
