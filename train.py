from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from transformers import EarlyStoppingCallback
import torch

# load tokenizer and model
model_name = "gpt2-medium"  # Can be adjusted if needed
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# bfloat16 for efficient training on Apple M1
if torch.backends.mps.is_available():
    model.to(torch.device("mps"))
    torch.set_default_dtype(torch.bfloat16)

# prep dataset
train_file = "./training_data/labeled_training_data.txt"
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=256
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# optimized training arguments
training_args = TrainingArguments(
    output_dir="./darcy_gpt2",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,  # stabilizes training
    learning_rate=2e-5,
    weight_decay=0.01,  # regularization
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # stop early if no improvement
)

# train!
trainer.train()

trainer.save_model("./darcy_gpt2")
tokenizer.save_pretrained("./darcy_gpt2")
