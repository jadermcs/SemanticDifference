#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForMaskedLM, 
    AutoTokenizer, 
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
from nltk.corpus import wordnet
import nltk
from datasets import load_dataset

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
MLM_PROBABILITY = 0.15

# Get all supersense classes from WordNet
def get_supersense_classes():
    """Extract all supersense classes from WordNet."""
    supersenses = set()
    for synset in wordnet.all_synsets():
        if hasattr(synset, 'lexname'):
            supersenses.add(synset.lexname())
    return sorted(list(supersenses))

# Create a mapping from supersense to index
SUPERSENSE_CLASSES = get_supersense_classes()
SUPERSENSE_TO_ID = {supersense: idx for idx, supersense in enumerate(SUPERSENSE_CLASSES)}
NUM_SUPERSENSE_CLASSES = len(SUPERSENSE_CLASSES)

print(f"Found {NUM_SUPERSENSE_CLASSES} supersense classes: {SUPERSENSE_CLASSES}")

class MultiTaskBertModel(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_supersense_classes=NUM_SUPERSENSE_CLASSES):
        super().__init__()
        # Load pre-trained BERT model
        self.bert = AutoModelForMaskedLM.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Supersense classification head - now applied to each token position
        self.supersense_classifier = nn.Linear(self.bert.config.hidden_size, num_supersense_classes)
        
    def forward(self, input_ids, attention_mask, labels=None, supersense_labels=None):
        # Get BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        # MLM loss
        mlm_loss = outputs.loss
        
        # Get hidden states for supersense classification
        hidden_states = outputs.hidden_states[-1] if hasattr(outputs, 'hidden_states') else outputs.logits
        
        # Apply supersense classifier to all token positions
        # Shape: [batch_size, seq_length, num_supersense_classes]
        supersense_logits = self.supersense_classifier(hidden_states)
        
        # Apply sigmoid to get probabilities for multilabel classification
        supersense_probs = torch.sigmoid(supersense_logits)
        
        # Calculate supersense classification loss if labels are provided
        supersense_loss = None
        if supersense_labels is not None:
            # Apply attention mask to ignore padding tokens
            # attention_mask shape: [batch_size, seq_length]
            # Expand to match supersense_logits shape
            mask = attention_mask.unsqueeze(-1).expand(-1, -1, NUM_SUPERSENSE_CLASSES)
            
            # Apply mask to logits and labels
            masked_logits = supersense_logits * mask
            masked_labels = supersense_labels * mask
            
            # Calculate binary cross entropy loss for multilabel classification
            # We need to handle the special value -100 in the labels
            valid_positions = (masked_labels != -100)
            
            # Only compute loss on valid positions
            if valid_positions.any():
                supersense_loss = F.binary_cross_entropy_with_logits(
                    masked_logits[valid_positions],
                    masked_labels[valid_positions],
                    reduction='mean'
                )
        
        # Total loss is the sum of MLM loss and supersense loss
        total_loss = mlm_loss
        if supersense_loss is not None:
            total_loss = mlm_loss + supersense_loss
        
        return {
            'loss': total_loss,
            'mlm_loss': mlm_loss,
            'supersense_loss': supersense_loss,
            'supersense_logits': supersense_logits,
            'supersense_probs': supersense_probs
        }

class WordNetDataset(Dataset):
    def __init__(self, tokenizer, max_length=MAX_LENGTH, split="train"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load WordNet data
        data_file = f"data/wordnet.{split}.json"
        if os.path.exists(data_file):
            with open(data_file, 'r') as f:
                self.data = json.load(f)
            data = []
            for item in self.data:
                usage_x = item["USAGE_x"].replace(item["WORD_x"], "[TGT]" + item["WORD_x"] + "[/TGT]")
                usage_y = item["USAGE_y"].replace(item["WORD_y"], "[TGT]" + item["WORD_y"] + "[/TGT]")
                item["text"] = usage_x + tokenizer.sep_token + usage_y
                item["masked_text"] = usage_x.replace(item["WORD_x"], self.tokenizer.mask_token) +\
                                    tokenizer.sep_token + usage_y.replace(item["WORD_y"], self.tokenizer.mask_token)
                data.append(item)
            self.data = data
        else:
            raise FileNotFoundError(f"Data file {data_file} not found")
        
        print(f"Loaded {len(self.data)} examples from {split} set")
    
    def _create_masked_example(self, text, target_word):
        """Create a masked version of the text where the target word is masked."""
        # Simple approach: replace the target word with [MASK]
        return text.replace(target_word, self.tokenizer.mask_token)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        synsets = wordnet.synsets(item["LEMMA"])
        item["supersenses"] = set(supersense.lexname() for supersense in synsets)
        item["supersense_ids"] = [SUPERSENSE_TO_ID[supersense] for supersense in item["supersenses"]]
        
        # Tokenize the original text
        original_encoding = self.tokenizer( 
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # Tokenize the masked text
        masked_encoding = self.tokenizer(
            item["masked_text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=True
        )
        
        # Find the position of the [MASK] token in the masked text
        mask_token_id = self.tokenizer.mask_token_id
        mask_positions = (masked_encoding["input_ids"][0] == mask_token_id).nonzero().squeeze()
                
        # Create multilabel supersense labels for each token position
        # Initialize with zeros (no supersense)
        supersense_labels = torch.zeros((self.max_length, NUM_SUPERSENSE_CLASSES), dtype=torch.float)
        
        # Set the supersense labels for the target word position
        for supersense_id in item["supersense_ids"]:
            supersense_labels[mask_positions, supersense_id] = 1.0
        
        # Set special tokens (CLS, SEP, PAD) to a special value (e.g., -100)
        # This will be ignored in the loss calculation
        special_tokens = [self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id]
        for i, token_id in enumerate(masked_encoding["input_ids"][0]):
            if token_id in special_tokens:
                supersense_labels[i] = -100
        
        return {
            "input_ids": masked_encoding["input_ids"][0],
            "attention_mask": masked_encoding["attention_mask"][0],
            "labels": original_encoding["input_ids"][0],
            "supersense_labels": supersense_labels
        }

def train_model(model, train_dataloader, val_dataloader=None):
    """Train the multi-task BERT model."""
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Create learning rate scheduler
    num_training_steps = len(train_dataloader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=num_training_steps
    )
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        total_mlm_loss = 0
        total_supersense_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            supersense_labels = batch["supersense_labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                supersense_labels=supersense_labels
            )
            
            loss = outputs["loss"]
            mlm_loss = outputs["mlm_loss"]
            supersense_loss = outputs["supersense_loss"]
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Update progress bar
            total_loss += loss.item()
            total_mlm_loss += mlm_loss.item()
            total_supersense_loss += supersense_loss.item() if supersense_loss is not None else 0
            
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "mlm_loss": f"{mlm_loss.item():.4f}",
                "supersense_loss": f"{supersense_loss.item():.4f}" if supersense_loss is not None else "N/A"
            })
        
        # Calculate average loss
        avg_loss = total_loss / len(train_dataloader)
        avg_mlm_loss = total_mlm_loss / len(train_dataloader)
        avg_supersense_loss = total_supersense_loss / len(train_dataloader)
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Average MLM Loss: {avg_mlm_loss:.4f}")
        print(f"Average Supersense Loss: {avg_supersense_loss:.4f}")
        
        # Validation
        if val_dataloader is not None:
            model.eval()
            val_loss = 0
            val_mlm_loss = 0
            val_supersense_loss = 0
            correct_supersense = 0
            total_supersense = 0
            partial_correct = 0
            
            with torch.no_grad():
                progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
                for batch in progress_bar:
                    # Move batch to device
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)
                    supersense_labels = batch["supersense_labels"].to(device)
                    
                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        supersense_labels=supersense_labels
                    )
                    
                    loss = outputs["loss"]
                    mlm_loss = outputs["mlm_loss"]
                    supersense_loss = outputs["supersense_loss"]
                    
                    val_loss += loss.item()
                    val_mlm_loss += mlm_loss.item()
                    val_supersense_loss += supersense_loss.item() if supersense_loss is not None else 0
                    
                    # Calculate supersense accuracy
                    if supersense_loss is not None:
                        supersense_probs = outputs["supersense_probs"]
                        # For multilabel, we use a threshold to determine positive classes
                        predicted_supersense = (supersense_probs > 0.5).float()
                        
                        # Only consider non-special tokens
                        valid_positions = (supersense_labels != -100).any(dim=-1)
                        
                        # Calculate accuracy metrics for multilabel classification
                        # Exact match: all labels must match exactly
                        exact_match = ((predicted_supersense == supersense_labels) & (supersense_labels != -100)).all(dim=-1)
                        correct_supersense += exact_match.sum().item()
                        total_supersense += valid_positions.sum().item()
                        
                        # Partial match: at least one label matches
                        partial_match = ((predicted_supersense == supersense_labels) & (supersense_labels != -100)).any(dim=-1)
                        partial_correct += partial_match.sum().item()
            
            # Calculate average validation loss
            avg_val_loss = val_loss / len(val_dataloader)
            avg_val_mlm_loss = val_mlm_loss / len(val_dataloader)
            avg_val_supersense_loss = val_supersense_loss / len(val_dataloader)
            
            print(f"Validation Loss: {avg_val_loss:.4f}")
            print(f"Validation MLM Loss: {avg_val_mlm_loss:.4f}")
            print(f"Validation Supersense Loss: {avg_val_supersense_loss:.4f}")
            
            if total_supersense > 0:
                supersense_accuracy = correct_supersense / total_supersense
                partial_accuracy = partial_correct / total_supersense
                print(f"Supersense Classification Exact Match Accuracy: {supersense_accuracy:.4f}")
                print(f"Supersense Classification Partial Match Accuracy: {partial_accuracy:.4f}")
        
        # Save model checkpoint
        os.makedirs("output/bert", exist_ok=True)
        torch.save(model.state_dict(), f"output/bert/bert_weak_supervision_epoch_{epoch+1}.pt")

def evaluate_mlm(model, dataloader):
    """Evaluate the MLM performance of the model."""
    model.eval()
    total_mlm_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            mask_positions = batch["mask_positions"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            mlm_loss = outputs["mlm_loss"]
            total_mlm_loss += mlm_loss.item()
            
            # Get predictions for masked tokens
            logits = outputs.logits
            for i, pos in enumerate(mask_positions):
                if pos < logits.size(1):  # Ensure position is within bounds
                    predicted_token_id = torch.argmax(logits[i, pos]).item()
                    true_token_id = labels[i, pos].item()
                    
                    if predicted_token_id == true_token_id:
                        correct_predictions += 1
                    total_predictions += 1
    
    avg_mlm_loss = total_mlm_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    print(f"MLM Loss: {avg_mlm_loss:.4f}")
    print(f"MLM Accuracy: {accuracy:.4f}")
    
    return avg_mlm_loss, accuracy

def evaluate_supersense(model, dataloader):
    """Evaluate the supersense classification performance of the model."""
    model.eval()
    total_supersense_loss = 0
    exact_match_correct = 0
    partial_match_correct = 0
    total_predictions = 0
    
    # Metrics for each supersense class
    class_metrics = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(NUM_SUPERSENSE_CLASSES)}
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            supersense_labels = batch["supersense_labels"].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                supersense_labels=supersense_labels
            )
            
            supersense_loss = outputs["supersense_loss"]
            total_supersense_loss += supersense_loss.item() if supersense_loss is not None else 0
            
            # Get predictions
            supersense_probs = outputs["supersense_probs"]
            predicted_supersense = (supersense_probs > 0.5).float()
            
            # Only consider non-special tokens
            valid_positions = (supersense_labels != -100).any(dim=-1)
            
            # Calculate accuracy metrics for multilabel classification
            # Exact match: all labels must match exactly
            exact_match = ((predicted_supersense == supersense_labels) & (supersense_labels != -100)).all(dim=-1)
            exact_match_correct += exact_match.sum().item()
            
            # Partial match: at least one label matches
            partial_match = ((predicted_supersense == supersense_labels) & (supersense_labels != -100)).any(dim=-1)
            partial_match_correct += partial_match.sum().item()
            
            total_predictions += valid_positions.sum().item()
            
            # Calculate per-class metrics
            for i in range(NUM_SUPERSENSE_CLASSES):
                # True positives: predicted 1, actual 1
                tp = ((predicted_supersense[:, :, i] == 1) & (supersense_labels[:, :, i] == 1) & (supersense_labels[:, :, i] != -100)).sum().item()
                # False positives: predicted 1, actual 0
                fp = ((predicted_supersense[:, :, i] == 1) & (supersense_labels[:, :, i] == 0)).sum().item()
                # False negatives: predicted 0, actual 1
                fn = ((predicted_supersense[:, :, i] == 0) & (supersense_labels[:, :, i] == 1)).sum().item()
                
                class_metrics[i]["tp"] += tp
                class_metrics[i]["fp"] += fp
                class_metrics[i]["fn"] += fn
    
    avg_supersense_loss = total_supersense_loss / len(dataloader)
    exact_match_accuracy = exact_match_correct / total_predictions if total_predictions > 0 else 0
    partial_match_accuracy = partial_match_correct / total_predictions if total_predictions > 0 else 0
    
    print(f"Supersense Classification Loss: {avg_supersense_loss:.4f}")
    print(f"Supersense Classification Exact Match Accuracy: {exact_match_accuracy:.4f}")
    print(f"Supersense Classification Partial Match Accuracy: {partial_match_accuracy:.4f}")
    
    # Print per-class metrics
    print("\nPer-Class Metrics:")
    print("Class", "Precision", "Recall", "F1", sep="\t")
    for i in range(NUM_SUPERSENSE_CLASSES):
        metrics = class_metrics[i]
        precision = metrics["tp"] / (metrics["tp"] + metrics["fp"]) if (metrics["tp"] + metrics["fp"]) > 0 else 0
        recall = metrics["tp"] / (metrics["tp"] + metrics["fn"]) if (metrics["tp"] + metrics["fn"]) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{SUPERSENSE_CLASSES[i][:3]}", f"{precision:.4f}", f"{recall:.4f}", f"{f1:.4f}", sep="\t")
    
    return avg_supersense_loss, exact_match_accuracy

def main():
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_tokens(["[TGT]", "[/TGT]"])
    
    # Create datasets
    train_dataset = WordNetDataset(tokenizer, split="train")
    val_dataset = WordNetDataset(tokenizer, split="test")
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = MultiTaskBertModel().to(device)
    model.resize_token_embeddings(len(tokenizer))

    # Train model
    train_model(model, train_dataloader, val_dataloader)
    
    # Evaluate model
    print("\nEvaluating MLM performance...")
    evaluate_mlm(model, val_dataloader)
    
    print("\nEvaluating supersense classification performance...")
    evaluate_supersense(model, val_dataloader)
    
    # Save final model
    os.makedirs("output/bert", exist_ok=True)
    torch.save(model.state_dict(), "output/bert/bert_weak_supervision_final.pt")
    print("\nModel saved to output/bert/bert_weak_supervision_final.pt")

if __name__ == "__main__":
    main()
