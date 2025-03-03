import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
from torch.utils.data import Dataset


# returns list of formatted data; preserves order from json
def load_data(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_samples = []
    for sample in data:
        formatted_text = f"[CATEGORY: {sample['category']}] [TEXT: {sample['content']}]"
        formatted_samples.append(formatted_text)
    return formatted_samples

# create DarcyDataset class
class DarcyDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=512):
        self.examples = []
        for text in texts:
            tokenized_text = tokenizer.encode(text, add_special_tokens=True)
            if len(tokenized_text) > block_size:
                tokenized_text = tokenized_text[:block_size]
            self.examples.append(tokenized_text)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        # in language modeling, input ids and labels are the same
        input_ids = self.examples[idx]
        return {"input_ids": input_ids, "labels": input_ids}


tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
# hoping special tokens will improve tag recognition during training
special_tokens_dict = {'additional_special_tokens': ["[CATEGORY:", "[TEXT:"]}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# load GPT-2, resize token embeddings
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.resize_token_embeddings(len(tokenizer))

# instantiate dataset
formatted_texts = load_data("labeled_training_data.json")
dataset = DarcyDataset(formatted_texts, tokenizer)

# build trainer
training_args = TrainingArguments(
    output_dir="./model/darcy-gpt",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
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
