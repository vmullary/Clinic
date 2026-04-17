import os
import torch
import argparse
import json
import glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Assuming this script is run in the same directory as reasoningData.py
try:
    from reasoningData import RL_DATASET
except ImportError:
    RL_DATASET = {}

def load_jsonl_data(data_path):
    """
    Finds and loads RL data from .jsonl files in a directory or a specific file.
    Expected format: {"user": "...", "responses": [{"text": "...", "reward": ...}]}
    """
    entries = []
    if os.path.isdir(data_path):
        files = glob.glob(os.path.join(data_path, "**/*.jsonl"), recursive=True)
    else:
        files = [data_path]
    
    print(f"Found {len(files)} .jsonl files.")
    
    for file_path in files:
        print(f"Loading {file_path}...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
    return entries

def prepare_data(external_entries=None):
    texts = []
    labels = []
    
    if external_entries:
        # Use provided external data
        for entry in external_entries:
            user_prompt = entry.get("user", "")
            for resp in entry.get("responses", []):
                text_input = f"Prompt: {user_prompt}\nResponse: {resp.get('text', '')}"
                texts.append(text_input)
                labels.append(float(resp.get("reward", 0.0)))
    else:
        # Fallback to internal RL_DATASET
        if not RL_DATASET:
            print("Warning: RL_DATASET not found in reasoningData.py and no external data provided.")
            return Dataset.from_dict({"text": [], "label": []})
            
        for mode, entries in RL_DATASET.items():
            for entry in entries:
                user_prompt = entry["user"]
                for resp in entry["responses"]:
                    text_input = f"Prompt: {user_prompt}\nResponse: {resp['text']}"
                    texts.append(text_input)
                    labels.append(float(resp["reward"]))
    
    return Dataset.from_dict({
        "text": texts,
        "label": labels
    })

def main():
    parser = argparse.ArgumentParser(description="Train the Gate model for Reasoning mode selection.")
    parser.add_argument("--data_dir", type=str, help="Directory containing .jsonl files or a single .jsonl file.")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased", help="Base model for training.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per device.")
    args = parser.parse_args()

    if args.data_dir:
        print(f"Loading data from {args.data_dir}...")
        external_entries = load_jsonl_data(args.data_dir)
        dataset = prepare_data(external_entries)
    else:
        print("Preparing dataset from reasoningData.py (default)...")
        dataset = prepare_data()

    print(f"Loaded {len(dataset)} examples for training the Gate.")
    
    if len(dataset) == 0:
        print("Error: No data loaded. Training cannot proceed.")
        return

    # Split into train and test sets (using a deterministic seed)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    model_name = args.model_name
    print(f"\nLoading tokenizer ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    
    print("Tokenizing datasets...")
    tokenized_datasets = dataset.map(tokenize_fn, batched=True)
    
    print(f"\nLoading model ({model_name}) for Regression (num_labels=1)...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)
    
    output_dir = "./rl_gate_model"
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        remove_unused_columns=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
    )
    
    print("\nStarting Training...")
    trainer.train()
    
    print(f"\nSaving final trained Gate model to {os.path.abspath(output_dir)}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("=========================================================")
    print(" Gate model training complete!")
    print("=========================================================")

if __name__ == "__main__":
    main()
