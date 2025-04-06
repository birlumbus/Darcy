import json
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset


gpt_model = "gpt2-large"
darcy_gpt_loc = "../models/darcy-gpt2-large-1"
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


class DarcyDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=512):
        self.examples = []
        for text in texts:
            tokenized = tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=block_size,
                padding="max_length",
            )
            self.examples.append(tokenized["input_ids"])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # in language modeling, input ids and labels are the same
        input_ids = self.examples[idx]
        return {"input_ids": input_ids, "labels": input_ids}


# load tokenizer and model
tokenizer = GPT2TokenizerFast.from_pretrained(gpt_model)
tokenizer.pad_token = tokenizer.eos_token
# special tokens will improve tag recognition during training
special_tokens_dict = {'additional_special_tokens': ["[CATEGORY:", "[TEXT:"]}
tokenizer.add_special_tokens(special_tokens_dict)

# load GPT-2, resize token embeddings
model = GPT2LMHeadModel.from_pretrained(gpt_model)
model.resize_token_embeddings(len(tokenizer))

# instantiate dataset
formatted_texts = load_data(training_data_loc)
dataset = DarcyDataset(formatted_texts, tokenizer)


# adjusted batch size for gpt2-large on Apple M1 Max
training_args = TrainingArguments(
    output_dir=darcy_gpt_loc,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,  # reduced batch size from gpt2medium
    gradient_accumulation_steps=4,
    learning_rate=1e-5,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# train!
trainer.train()

# saving tokenizer because I've made changes to it
trainer.save_model(darcy_gpt_loc)
tokenizer.save_pretrained(darcy_gpt_loc)
