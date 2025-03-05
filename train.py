import json
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
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


tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-medium")
tokenizer.pad_token = tokenizer.eos_token
# hoping special tokens will improve tag recognition during training
special_tokens_dict = {'additional_special_tokens': ["[CATEGORY:", "[TEXT:"]}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# load GPT-2, resize token embeddings
model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model.resize_token_embeddings(len(tokenizer))

# instantiate dataset
formatted_texts = load_data("./training_data/training_text/labeled_training_data.json")
dataset = DarcyDataset(formatted_texts, tokenizer)

# # data collator for dynamic input padding
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# build trainer
training_args = TrainingArguments(
    output_dir="./model/darcy-gpt",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    learning_rate=1e-5,
    save_steps=500,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
#     data_collator=data_collator,
)


# train!
trainer.train()

# explicitly save model
trainer.save_model("./model/darcy-gpt")
# also save tokenizer, because I've made changes to it
tokenizer.save_pretrained("./model/darcy-gpt")

