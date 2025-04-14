import torch
import torch.nn.functional as F
import os
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

import torch.nn as nn
from torch.utils.data import DataLoader
from torch. optim import AdamW

device = "cpu"

def get_batches(dataset, batch_size):
    """
    Generator function to yield batches from a dataset.

    Args:
        dataset (Dataset): The dataset to batch.
        batch_size (int): The size of each batch.

    Yields:
        dict: A batch of data.
    """
    for i in range(0, len(dataset), batch_size):
        yield dataset[i:i + batch_size]

class BertContrastiveModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(self, input_ids1, attention_mask1, token_pos1,
                input_ids2, attention_mask2, token_pos2):
        # Get BERT outputs
        outputs1 = self.bert(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.bert(input_ids=input_ids2, attention_mask=attention_mask2)

        # Extract hidden states
        hidden_states1 = outputs1.last_hidden_state
        hidden_states2 = outputs2.last_hidden_state

        # Extract specific token representations
        # For each example in the batch, get the token at the specified position
        token_vectors1 = hidden_states1[:, token_pos1, :].mean(dim=1).squeeze(1)
        token_vectors2 = hidden_states2[:, token_pos2, :].mean(dim=1).squeeze(1)

        # Normalize vectors
        token_vectors1 = F.normalize(token_vectors1, p=2, dim=1)
        token_vectors2 = F.normalize(token_vectors2, p=2, dim=1)

        return token_vectors1, token_vectors2


# Contrastive loss function
def contrastive_loss(vec1, vec2, labels, margin=1.0):
    # Cosine similarity
    cosine_sim = torch.sum(vec1 * vec2, dim=1)
    # Loss: minimize distance for similar pairs (label=1),
    # maximize distance for dissimilar pairs (label=0)
    loss = torch.mean((1 - labels) * torch.pow(cosine_sim, 2) +
                      labels * torch.pow(torch.clamp(margin - cosine_sim, min=0.0), 2))
    return loss


def train_bert_contrastive(model, train_dataloader, optimizer, num_epochs=3, val_dataloader=None):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in get_batches(train_dataloader, 32):
            # Get data from batch
            input_ids1 = torch.tensor(batch['input_ids1']).to(device)
            attention_mask1 = torch.tensor(batch['attention_mask1']).to(device)
            token_pos1 = batch['token_pos1']
            input_ids2 = torch.tensor(batch['input_ids2']).to(device)
            attention_mask2 = torch.tensor(batch['attention_mask2']).to(device)
            token_pos2 = batch['token_pos2']
            labels = torch.tensor([int(label == "identical") for label in batch['LABEL']]).to(device)  # 1 for similar, 0 for dissimilar tokens

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            vec1, vec2 = model(input_ids1, attention_mask1, token_pos1,
                               input_ids2, attention_mask2, token_pos2)

            # Calculate loss
            loss = contrastive_loss(vec1, vec2, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            total_loss += loss.item()
            
            # Calculate accuracy
            # Compute cosine similarity between vectors
            cosine_sim = torch.sum(vec1 * vec2, dim=1)
            # Predict similar (1) if cosine similarity > 0.5, dissimilar (0) otherwise
            predictions = (cosine_sim > 0.5).float()
            # Count correct predictions
            correct_predictions += (predictions == labels).sum().item()
            total_predictions += labels.size(0)

        avg_loss = total_loss / len(train_dataloader)
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        # Evaluate on validation set if available
        if val_dataloader is not None:
            model.eval()
            val_correct = 0
            val_total = 0
            val_loss = 0
            
            with torch.no_grad():
                for batch in get_batches(val_dataloader, 32):
                    # Get data from batch
                    input_ids1 = torch.tensor(batch['input_ids1']).to(device)
                    attention_mask1 = torch.tensor(batch['attention_mask1']).to(device)
                    token_pos1 = batch['token_pos1']
                    input_ids2 = torch.tensor(batch['input_ids2']).to(device)
                    attention_mask2 = torch.tensor(batch['attention_mask2']).to(device)
                    token_pos2 = batch['token_pos2']
                    labels = torch.tensor([int(label == "identical") for label in batch['LABEL']]).to(device)
                    
                    # Forward pass
                    vec1, vec2 = model(input_ids1, attention_mask1, token_pos1,
                                       input_ids2, attention_mask2, token_pos2)
                    
                    # Calculate loss
                    loss = contrastive_loss(vec1, vec2, labels)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    cosine_sim = torch.sum(vec1 * vec2, dim=1)
                    predictions = (cosine_sim > 0.5).float().squeeze(-1)
                    val_correct += (predictions == labels).sum().item()
                    val_total += labels.size(0)
            
            avg_val_loss = val_loss / len(val_dataloader) if len(val_dataloader) > 0 else 0
            val_accuracy = val_correct / val_total if val_total > 0 else 0
            print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
            
            # Switch back to training mode
            model.train()

if __name__ == "__main__":
    model_path = "FacebookAI/roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BertContrastiveModel(model_path)

    os.makedirs("output/bert", exist_ok=True)

    for dataset in ["wordnet"]:#, "semcor", "masc", "wordnet", "fews", "wic"]:
        print("running experiments for", dataset)
        train_data = load_dataset("json", data_files=f"data/{dataset}.train.json")
        
        def get_token_embedding(word, sentence, suffix, tokenizer=tokenizer, model=model):
            tokens = tokenizer.tokenize(sentence, add_special_tokens=True)
            tokens_ids = tokenizer(sentence, padding="max_length", truncation=True, max_length=128, add_special_tokens=True)
            word_tokens = [t for t in tokens if word in t or t.lstrip("Ġ") == word]  # Handle Ġ
            word_indices = [i for i, token in enumerate(tokens) if token in word_tokens]
            tokens_ids["input_ids"+suffix] = tokens_ids.pop("input_ids")
            tokens_ids["attention_mask"+suffix] = tokens_ids.pop("attention_mask")
            tokens_ids["token_pos"+suffix] = word_indices
            return tokens_ids

        def preprocess(example):
            tokens1 = get_token_embedding(example["LEMMA"], example["USAGE_x"], "1")
            tokens2 = get_token_embedding(example["LEMMA"], example["USAGE_y"], "2")
            tokens1.update(tokens2)
            return tokens1

        train_data = train_data["train"].map(preprocess)
        
        # Try to load validation data if available
        val_data = None
        try:
            val_data = load_dataset("json", data_files=f"data/{dataset}.test.json")
            val_data = val_data["train"].map(preprocess)
            print(f"Loaded validation data with {len(val_data)} examples")
        except Exception as e:
            print(f"No validation data found: {e}")

        optim = AdamW(model.parameters(), lr=5e-5)
        train_bert_contrastive(model, train_data, optim, val_dataloader=val_data)
