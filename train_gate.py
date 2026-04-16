import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Assuming this script is run in the same directory as reasoningData.py
from reasoningData import RL_DATASET

def prepare_data():
    texts = []
    labels = []
    
    # Flatten the RL dataset into prompt-response pairs with rewards
    for mode, entries in RL_DATASET.items():
        for entry in entries:
            user_prompt = entry["user"]
            for resp in entry["responses"]:
                # Format exactly how the gate will see it at inference time
                text_input = f"Prompt: {user_prompt}\nResponse: {resp['text']}"
                texts.append(text_input)
                labels.append(float(resp["reward"]))
    
    # Convert to HuggingFace Dataset
    return Dataset.from_dict({
        "text": texts,
        "label": labels
    })

def main():
    print("Preparing dataset from reasoningData.py...")
    dataset = prepare_data()
    print(f"Loaded {len(dataset)} examples for training the Gate.")
    
    # Split into train and test sets (using a deterministic seed)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    # Using distilbert as a small, fast local model capable of regression
    model_name = "distilbert-base-uncased"
    print(f"\nLoading tokenizer ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_fn(examples):
        # Truncate and pad to ensure uniform tensor shapes
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(tokenize_fn, batched=True)
    
    print(f"\nLoading model ({model_name}) for Regression (num_labels=1)...")
    # num_labels=1 forces HuggingFace to treat this as an MSE regression task rather than classification
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    output_dir = "./rl_gate_model"
    os.makedirs(output_dir, exist_ok=True)
    
    # Training Arguments designed to be lightweight on a local machine
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        eval_strategy="epoch",      # Evaluate at the end of every epoch
        save_strategy="epoch",      # Save checkpoint at the end of every epoch
        logging_dir="./logs",
        logging_steps=5,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True, # Ensure we keep the weights that scored best on the test set
        remove_unused_columns=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )
    
    print("\nStarting Training... (This may take some time depending on your hardware)")
    trainer.train()
    
    print(f"\nSaving final trained Gate model to {os.path.abspath(output_dir)}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("=========================================================")
    print(" Gate model training complete! Ready for parallel inference.")
    print("=========================================================")

if __name__ == "__main__":
    main()
