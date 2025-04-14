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
        print(hidden_states1.shape)
        print(token_pos1)
        print(hidden_states2.shape)
        print(token_pos2)
        token_vectors1 = torch.stack([hidden_states1[i, pos, :]
                                     for i, pos in enumerate(token_pos1)])
        token_vectors2 = torch.stack([hidden_states2[i, pos, :]
                                     for i, pos in enumerate(token_pos2)])

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


def train_bert_contrastive(model, train_dataloader, optimizer, num_epochs=3):
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in get_batches(train_dataloader, 32):
            print(batch)
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

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    model_path = "FacebookFacebookAI/roberta-base"
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

        optim = AdamW(model.parameters(), lr=5e-5)
        train_bert_contrastive(model, train_data, optim)
        # for pos in ["verb", "noun", "adverb", "adjective"]:
        #     train_data.filter(lambda x: x["POS"] == pos).to_json(f"output/bert/{dataset}.{pos}.predict.json")
