from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from transformers import EarlyStoppingCallback
import torch

# Load the tokenizer and model
model_name = "gpt2-medium"  # You can choose a different model size here
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Prepare the dataset
train_file = "darcy.txt"  # Your dataset file
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./darcy_gpt2",
    overwrite_output_dir=True,
    num_train_epochs=10,  # Total epochs to train, you can adjust
    per_device_train_batch_size=1,
    save_steps=500,
    save_total_limit=2,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss",
)

# Implement Early Stopping
class EarlyStoppingAtEpochs(EarlyStoppingCallback):
    def __init__(self, patience=1, threshold=0.0):
        super().__init__(patience=patience, threshold=threshold)
        self.best_loss = None

    def on_epoch_end(self, args, state, control, logs=None, **kwargs):
        current_loss = logs.get('eval_loss', None)
        if current_loss is not None:
            if self.best_loss is None or current_loss < self.best_loss - self.threshold:
                self.best_loss = current_loss
            else:
                print(f"Early stopping triggered at epoch {state.epoch}")
                control.should_early_stop = True
        return control

early_stopping = EarlyStoppingAtEpochs(patience=1, threshold=0.0)

# Create Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    callbacks=[early_stopping],
)

# Training loop
trainer.train()

# Save final model and tokenizer
model.save_pretrained("./darcy_gpt2")
tokenizer.save_pretrained("./darcy_gpt2")
