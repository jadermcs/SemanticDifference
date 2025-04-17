#!/usr/bin/env python
# coding: utf-8
import argparse
import torch
import torch.nn as nn
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from functools import lru_cache
from bert_multitask import CustomMultiTaskModel, DataCollatorForJointMLMClassification, MultitaskTrainerJoint
import json
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
    AutoConfig,
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

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Get all supersense classes from WordNet
def get_supersense_classes():
    """Extract all supersense classes from WordNet."""
    supersenses = set()
    for synset in wordnet.all_synsets():
        if hasattr(synset, 'lexname'):
            supersenses.add(synset.lexname())
    return sorted(list(supersenses))


SUPERSENSE_CLASSES = get_supersense_classes()
SUPERSENSE_TO_ID = {supersense: idx for idx, supersense in enumerate(SUPERSENSE_CLASSES)}
NUM_SUPERSENSE_CLASSES = len(SUPERSENSE_CLASSES)


@lru_cache(maxsize=200000)
def get_word_supersenses(word):
    if len(word) < 4:
        return set()
    # Lemmatize the word
    word = lemmatizer.lemmatize(word)
    synsets = wordnet.synsets(word)
    return set(synset.lexname() for synset in synsets)


def load_data(datasets, split="train", mark_target=False, supersense=False):
    """Load and process the WiC dataset."""
    data = []
    for dataset in datasets.split(","):
        with open(f"data/{dataset}.{split}.json", 'r') as f:
            data.extend(json.load(f))
        
    # Convert to Hugging Face dataset format
    processed_data = []
    for item in data:
        w1 = item['WORD_x']
        w2 = item['WORD_y']
        s1 = item["USAGE_x"] if not mark_target else item['USAGE_x'].replace(w1, f"[TGT]{w1}[/TGT]")
        s2 = item["USAGE_y"] if not mark_target else item['USAGE_y'].replace(w2, f"[TGT]{w2}[/TGT]")
        s1 = word_tokenize(s1)
        s2 = word_tokenize(s2)
        data = {
            'sentence1': s1,
            'sentence2': s2,
            'supersenses1': [[SUPERSENSE_TO_ID[supersense] for supersense in get_word_supersenses(word)] for word in s1],
            'supersenses2': [[SUPERSENSE_TO_ID[supersense] for supersense in get_word_supersenses(word)] for word in s2],
            'label_diff': 1 if item['LABEL'] == 'identical' else 0
        }
        processed_data.append(data)
    
    return Dataset.from_list(processed_data)

def preprocess_function(examples, tokenizer):
    """Tokenize the input sentences."""
    tokenized_examples = tokenizer(
        examples['sentence1'],
        examples['sentence2'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        is_split_into_words=True
    )
    new_supersenses = []
    word_ids = tokenized_examples.word_ids()
    supersenses = examples['supersenses1']
    passed = False
    for word_id in word_ids:
        if word_id is None:
            new_supersenses.append([])
        elif not passed and word_id + 1 == len(examples['supersenses1']):
            new_supersenses.append(supersenses[word_id])
            supersenses = examples['supersenses2']
            passed = True
        else:
            new_supersenses.append(supersenses[word_id])
    tokenized_examples['supersenses'] = new_supersenses
    
    tokenized_examples['label_diff'] = examples['label_diff']
    return tokenized_examples

def compute_metrics(pred):
    print("compute_metrics")
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
    parser.add_argument('--supersense', action='store_true', default=False,
                        help='Use supersense classification')
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
    train_dataset = load_data(args.dataset, split="train", mark_target=args.mark_target, supersense=args.supersense)
    test_dataset = load_data("wic", split="test", mark_target=args.mark_target, supersense=args.supersense)

    datasets = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=True)
    datasets = datasets.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer}
    )

    custom_collator = DataCollatorForJointMLMClassification(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    
    model = CustomMultiTaskModel(
        args.model,
        num_sequence_labels=2,
        num_token_labels=NUM_SUPERSENSE_CLASSES if args.supersense else 0,
    )

    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size//2,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        warmup_steps=WARMUP_STEPS,
        remove_unused_columns=False, # CRITICAL
        report_to="wandb",
        run_name=args.wandb_run_name
    )
    
    # Initialize trainer
    trainer = MultitaskTrainerJoint(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        data_collator=custom_collator, # Use the custom multitask collator
        compute_metrics=compute_metrics, # Define function to calc cls + mlm metrics (perplexity)
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
