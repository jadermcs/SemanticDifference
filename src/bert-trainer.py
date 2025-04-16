#!/usr/bin/env python
# coding: utf-8
import argparse
import torch
import os
import json
import numpy as np
from pathlib import Path
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    set_seed
)
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
NUM_EPOCHS = 10

def load_data(datasets, split="train", mark_target=False):
    """Load and process the WiC dataset."""
    data = []
    for dataset in datasets.split(","):
        with open(f"{dataset}.{split}.json", 'r') as f:
            data.extend(json.load(f))
        
    # Convert to Hugging Face dataset format
    processed_data = []
    for item in data:
        processed_data.append({
            'sentence1': item['USAGE_x'].replace(item['WORD_x'], f"[TGT]{item['WORD_x']}[/TGT]"),
            'sentence2': item['USAGE_y'].replace(item['WORD_y'], f"[TGT]{item['WORD_y']}[/TGT]"),
            'label': 1 if item['LABEL'] == 'identical' else 0
        })
    
    return Dataset.from_list(processed_data)

def preprocess_function(examples, tokenizer):
    """Tokenize the input sentences."""
    return tokenizer(
        examples['sentence1'],
        examples['sentence2'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length'
    )

def compute_metrics(pred):
    """Compute metrics for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    parser = argparse.ArgumentParser(description='Train a BERT model for WiC classification')
    parser.add_argument('--model', type=str, default='FacebookAI/roberta-base',
                        help='Pre-trained model to use')
    parser.add_argument('--dataset', type=str, default='wic',
                        help='Path to the dataset file')
    parser.add_argument('--mark_target', action='store_true', default=False,
                        help='Mark the target word in the sentences')
    parser.add_argument('--output_dir', type=str, default='output/bert-classifier',
                        help='Directory to save the model')
    parser.add_argument('--wandb_project', type=str, default='semantic-difference',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use FP16 precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Initialize wandb
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.model.split('/')[-1]}-{args.dataset}-classifier"
    
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args)
    )
    
    # Load dataset
    train_dataset = load_data(args.dataset, split="train", mark_target=args.mark_target)
    test_dataset = load_data(args.dataset, split="test", mark_target=args.mark_target)

    datasets = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    # Tokenize dataset
    tokenized_dataset = datasets.map(
        lambda x: preprocess_function(x, tokenizer),
        batched=True,
        remove_columns=datasets.column_names
    )
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2
    )
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        warmup_steps=WARMUP_STEPS,
        report_to="wandb",
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate the model
    metrics = trainer.evaluate()
    
    # Log final metrics to wandb
    wandb.log(metrics)
    
    # Save the model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
