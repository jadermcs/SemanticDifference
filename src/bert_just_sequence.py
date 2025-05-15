#!/usr/bin/env python
# coding: utf-8
import argparse

import torch
import json
from tqdm import tqdm
from transformers import (
    ModernBertForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
    set_seed,
)
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
from typing import Optional, Tuple

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
MAX_LENGTH = 128
MLM_PROBABILITY = 0.3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01
IGNORE_ID = -100


def load_data(datasets, split="train"):
    """Load and process the WiC dataset."""
    all_data = []
    for dataset in datasets.split(","):
        with open(f"data/{dataset}.{split}.json", "r") as f:
            all_data.extend(json.load(f))

    processed_data = []
    for item in tqdm(all_data):
        w1, w2 = item["WORD_x"], item["WORD_y"]
        s1 = item["USAGE_x"]
        s2 = item["USAGE_y"]

        processed_entry = {
            "sentence1": s1,
            "sentence2": s2,
            "labels": int(item["LABEL"] == "identical"),
        }

        processed_data.append(processed_entry)

    return Dataset.from_list(processed_data)

def tokenize(sentence, tokenizer):
    return tokenizer(
        sentence,
        truncation=True,
        return_offsets_mapping=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    )

def align(examples, tokenizer):
    inputs = tokenize(examples["sentences"], tokenizer)
    inputs["labels"] = examples["labels"]
    return inputs

def preprocess_function(examples, tokenizer):
    """Tokenize input sentences and optionally process supersense labels."""
    examples["sentences"] = f"{examples['sentence1']} {tokenizer.sep_token} {examples['sentence2']}"
    return examples

def compute_metrics(pred):
    """Compute metrics for both sequence and token classification."""
    # Unpack predictions and labels
    preds, labels = pred

    seq_preds_argmax = preds.argmax(-1)
    seq_precision, seq_recall, seq_f1, _ = precision_recall_fscore_support(
        labels, seq_preds_argmax, average="weighted"
    )
    seq_acc = accuracy_score(labels, seq_preds_argmax)

    metrics = {
        "seq_accuracy": seq_acc,
        "seq_f1": seq_f1,
        "seq_precision": seq_precision,
        "seq_recall": seq_recall,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train a MLM model for difference classification"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="answerdotai/ModernBERT-base",
        help="Pre-trained model to use",
    )
    parser.add_argument(
        "--dataset", type=str, default="wic", help="Path to the dataset file"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument(
        "--supersense",
        action="store_true",
        default=False,
        help="Use supersense classification",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/bert-classifier",
        help="Directory to save the model",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="semantic-difference",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="Weights & Biases run name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True, help="Use FP16 precision"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--steps", type=int, default=500, help="Frequency to run evaluation"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Initialize wandb
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.model.split('/')[-1]}-{args.dataset}-classifier"
        args.wandb_run_name += "-supersense" if args.supersense else ""

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Load dataset
    train_dataset = load_data(args.dataset, split="train")
    test_dataset = load_data("wic", split="test")

    datasets = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    datasets = datasets.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=4,
    )

    datasets["train"] = datasets["train"].map(
        align,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        batched=True,
        remove_columns=datasets["train"].column_names,
        num_proc=4,
    )
    # During test we show all the tokens however we don't give supersense embeddings
    datasets["test"] = datasets["test"].map(
        align,
        fn_kwargs={
            "tokenizer": tokenizer,
        },
        batched=True,
        remove_columns=datasets["test"].column_names,
        num_proc=4,
    )

    # Initialize model
    config = AutoConfig.from_pretrained(args.model, num_labels=2)
    config.embedding_dropout = 0.1
    config.classifier_dropout = 0.1
    model = ModernBertForSequenceClassification(config)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="steps",
        eval_steps=args.steps,
        load_best_model_at_end=True,
        metric_for_best_model="seq_f1",
        greater_is_better=True,
        label_names=["labels"],
        fp16=args.fp16,
        warmup_steps=WARMUP_STEPS,
        gradient_accumulation_steps=32//args.batch_size,
        dataloader_num_workers=4,
        dataloader_pin_memory=True if device.type == "cuda" else False,
        report_to="wandb",
        run_name=args.wandb_run_name,
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
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
