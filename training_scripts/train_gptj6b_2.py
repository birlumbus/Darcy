import json
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model


gpt_model = "EleutherAI/gpt-j-6B"
darcy_gpt_loc = "../models/darcy-gptj-6b-2.1"
training_data_loc = "../training_data/training_text/final_json/labeled_training_data_2.json"


def load_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_samples = []
    for sample in data:
        formatted_text = f"[CATEGORY: {sample['category']}] [TEXT: {sample['content']}]"
        formatted_samples.append(formatted_text)
    return formatted_samples


texts = load_data(training_data_loc)
data = Dataset.from_dict({"text": texts})


tokenizer = AutoTokenizer.from_pretrained(gpt_model)
tokenizer.pad_token = tokenizer.eos_token
special_tokens_dict = {'additional_special_tokens': ["[CATEGORY:", "[TEXT:"]}
tokenizer.add_special_tokens(special_tokens_dict)


def tokenize(samples):
    tokenized_samples = tokenizer(samples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized_samples["labels"] = tokenized_samples["input_ids"].copy()
    return tokenized_samples

tokenized_data = data.map(tokenize, batched=True)
tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])


# load model without bitsandbytes quantization (LoRA only)
model = AutoModelForCausalLM.from_pretrained(
    gpt_model,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model.resize_token_embeddings(len(tokenizer))


# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)


training_args = TrainingArguments(
    output_dir=darcy_gpt_loc,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    save_steps=200,
    save_total_limit=2,
    fp16=False,
    bf16=True,
    logging_steps=50,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)

# train!
trainer.train()

# save the fine-tuned model and tokenizer
trainer.save_model(darcy_gpt_loc)
tokenizer.save_pretrained(darcy_gpt_loc)