#!/usr/bin/env python
# coding: utf-8
import argparse
import torch
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from functools import lru_cache
import json
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
START_TARGET_TOKEN = "[TGT]"
END_TARGET_TOKEN = "[/TGT]"

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
        s1 = item["USAGE_x"] if not mark_target else item['USAGE_x'].replace(w1, f"{START_TARGET_TOKEN}{w1}{END_TARGET_TOKEN}")
        s2 = item["USAGE_y"] if not mark_target else item['USAGE_y'].replace(w2, f"{START_TARGET_TOKEN}{w2}{END_TARGET_TOKEN}")
        data = {
            'sentence1': s1,
            'sentence2': s2,
            'labels': 1 if item['LABEL'] == 'identical' else 0
        }
        if supersense:
            s1 = word_tokenize(s1)
            s2 = word_tokenize(s2)
            supersenses1 = [[SUPERSENSE_TO_ID[supersense] for supersense in get_word_supersenses(word)] for word in s1]
            supersenses2 = [[SUPERSENSE_TO_ID[supersense] for supersense in get_word_supersenses(word)] for word in s2]
            supersenses1 = [[1 if i in sp1 else -100 for i in range(NUM_SUPERSENSE_CLASSES)] for sp1 in supersenses1]
            supersenses2 = [[1 if i in sp2 else -100 for i in range(NUM_SUPERSENSE_CLASSES)] for sp2 in supersenses2]
            data['supersenses1'] = supersenses1
            data['supersenses2'] = supersenses2
        processed_data.append(data)

    return Dataset.from_list(processed_data)


def preprocess_function(examples, tokenizer, supersense=False):
    """Tokenize the input sentences."""
    labels = examples['labels']
    if supersense:
        s1 = examples['supersense1']
        s2 = examples['supersense2']
    examples = tokenizer(
        examples['sentence1'],
        examples['sentence2'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        is_split_into_words=supersense
    )
    if supersense:
        new_supersenses = []
        word_ids = examples.word_ids()
        supersenses = s1
        passed = False
        for word_id in word_ids:
            if word_id is None:
                new_supersenses.append([-100 for _ in range(NUM_SUPERSENSE_CLASSES)])
            elif not passed and word_id + 1 == len(s1):
                new_supersenses.append(supersenses[word_id])
                supersenses = s2
                passed = True
            else:
                new_supersenses.append(supersenses[word_id])
        examples['supersenses'] = new_supersenses

    examples['labels'] = labels
    return examples


def compute_metrics(pred):
    """Compute metrics for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    print(labels.shape)
    print(preds.shape)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

import torch.nn as nn
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class MultiTaskModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    sequence_logits: torch.FloatTensor = None
    token_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class CustomMultiTaskModel(nn.Module):
    def __init__(self, model_name_or_path, num_sequence_labels, num_token_labels, loss_weights=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=self.config) # Or your specific base model

        # MLM Head (often part of the base model architecture for BERT-like models, e.g., via BertLMPredictionHead)
        # If using AutoModel, you might need to add the MLM head manually or use BertForMaskedLM as the base and access its components.
        # For simplicity, let's assume we might need to re-implement or fetch it if not using BertForMaskedLM directly.
        # We'll assume self.bert has or we add an MLM prediction capability later if needed.
        # A dedicated MLM head might look like this if needed:

        # Sequence Classification Head
        self.sequence_classifier = nn.Linear(self.config.hidden_size, num_sequence_labels)

        # Token Classification Head
        self.token_classifier = nn.Linear(self.config.hidden_size, num_token_labels)

        self.num_sequence_labels = num_sequence_labels
        self.num_token_labels = num_token_labels

        # Loss weights (optional, for combining losses)
        self.loss_weights = loss_weights if loss_weights else {"mlm": 1.0, "seq": 1.0, "token": 1.0}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,        # Labels for MLM
        sequence_labels=None, # Labels for sequence classification
        token_labels=None,    # Labels for token classification
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        sequence_output = outputs.hidden_states[-1] # Last hidden state (batch_size, sequence_length, hidden_size)
        
        # --- Calculate Logits ---
        # MLM Logits (predicting masked tokens)
        # If using BertForMaskedLM, this would be handled differently.
        # If using AutoModel, apply the head. Note: this is a simplification.
        # A proper MLM head often involves transformations + LayerNorm + bias.
        # mlm_logits = outputs.logits

        # Sequence Classification Logits
        sequence_logits = self.sequence_classifier(sequence_output[:,0,:]) # (batch_size, num_sequence_labels)
        # Token Classification Logits
        # token_logits = self.token_classifier(sequence_output) # (batch_size, sequence_length, num_token_labels)

        # --- Calculate Losses ---
        total_loss = 0.0
        mlm_loss = 0.0
        seq_loss = 0.0
        token_loss = 0.0
        mlm_logits = None
        token_logits = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss() # Common loss function
            seq_loss = loss_fct(sequence_logits.view(-1, self.num_sequence_labels), labels.view(-1))
            total_loss += seq_loss
        if labels is not None and sequence_labels is not None and token_labels is not None:
            print("rodou")
            loss_fct = nn.CrossEntropyLoss() # Common loss function
            bce_loss = nn.BCEWithLogitsLoss()

            # MLM Loss
            mlm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            # Token Classification Loss (ignoring padding, typically -100)
            # Only compute loss for non-special tokens if needed
            active_loss = attention_mask.view(-1) == 1 # Or based on token_labels != -100
            active_logits = token_logits.view(-1, self.num_token_labels)[active_loss]
            active_labels = token_labels.view(-1)[active_loss]
            if active_logits.shape[0] > 0: # Ensure there are valid tokens to compute loss on
                 token_loss = bce_loss(active_logits, active_labels)
            else:
                 # Handle cases where the batch might only contain padding after filtering
                 # Or if no token labels are present for the active parts
                 token_loss = torch.tensor(0.0, device=sequence_logits.device) # Ensure loss is on the correct device


            # Combine losses (e.g., weighted sum)
            total_loss = (self.loss_weights["mlm"] * mlm_loss +
                          self.loss_weights["seq"] * seq_loss +
                          self.loss_weights["token"] * token_loss)

        return MultiTaskModelOutput(
            loss=total_loss,
            mlm_logits=mlm_logits,
            sequence_logits=sequence_logits,
            token_logits=token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



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
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=args.supersense)

    datasets = datasets.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer}
    )

    # Initialize model
    model = CustomMultiTaskModel(
        args.model,
        num_sequence_labels=2,
        num_token_labels=NUM_SUPERSENSE_CLASSES if args.supersense else 0
    )
    if args.mark_target:
        tokenizer.add_tokens([START_TARGET_TOKEN, END_TARGET_TOKEN])
        model.resize_token_embeddings(len(tokenizer))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        warmup_steps=WARMUP_STEPS,
        report_to="wandb",
        run_name=args.wandb_run_name
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
