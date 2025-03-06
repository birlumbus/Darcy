import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


gpt_model = "EleutherAI/gpt-j-6B"
darcy_gpt_loc = "../model/darcy-gptj-6b-1"
training_data_loc = "../training_data/training_text/final_json/labeled_training_data_1.json"


# returns list of formatted data; preserves order from json
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
    return tokenizer(samples["text"], padding="max_length", truncation=True, max_length=512)


tokenized_data = data.map(tokenize, batched=True)
tokenized_data.set_format(type="torch", columns=["input_ids", "attention_mask"])

# load model with quantization (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


model = AutoModelForCausalLM.from_pretrained(
    gpt_model,
    quantization_config=bnb_config,
    device_map="auto"
)


model.resize_token_embeddings(len(tokenizer))
model = prepare_model_for_kbit_training(model)


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
    fp16=True,
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

# saving tokenizer because I've made changes to it
trainer.save_model(darcy_gpt_loc)
tokenizer.save_pretrained(darcy_gpt_loc)
