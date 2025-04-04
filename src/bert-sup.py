import torch
import torch.nn.functional as F
import os
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

import torch.nn as nn
from transformers import BertModel, BertTokenizer, AdamW


class BertContrastiveModel(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
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
        batch_size = input_ids1.size(0)
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
        
        for batch in train_dataloader:
            # Get data from batch
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            token_pos1 = batch['token_pos1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            token_pos2 = batch['token_pos2'].to(device)
            labels = batch['labels'].to(device)  # 1 for similar, 0 for dissimilar tokens
            
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
    optim = AdamW()

    model_path = "Alibaba-NLP/gte-modernbert-base"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    train_bert_contrastive(model, train_data, optim)

os.makedirs("output/bert", exist_ok=True)

for dataset in ["dwug", "semcor", "masc", "wordnet", "fews", "wic"]:
    print("running experiments for", dataset)
    data = load_dataset("json", data_files=f"data/{dataset}.test.json")

    def get_token_embedding(word, sentence, tokenizer=tokenizer, model=model):
        tokens = tokenizer.tokenize(sentence, add_special_tokens=True)
        tokens_ids = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
        word_tokens = [t for t in tokens if word in t or t.lstrip("Ġ") == word]  # Handle Ġ
        word_indices = [i for i, token in enumerate(tokens) if token in word_tokens]
        with torch.no_grad():
            outputs = model(**tokens_ids)
            hidden = outputs.last_hidden_state
        word_embeddings = hidden[0, word_indices, :]
        return word_embeddings.mean(0)

    def preprocess(example):
        emb_x = get_token_embedding(example["WORD_x"], example["USAGE_x"])
        emb_y = get_token_embedding(example["WORD_y"], example["USAGE_y"])
        embs = [emb_x, emb_y]
        embs = torch.stack(embs)
        embeddings = F.normalize(embs, p=2, dim=1)
        score = ((embeddings[0] @ embeddings[1].T) * 100).tolist()
        example["score"] = score
        return example

    data = data["train"].map(preprocess)
    for pos in ["verb", "noun", "adverb", "adjective"]:
        data.filter(lambda x: x["POS"] == pos).to_json(f"output/bert/{dataset}.{pos}.predict.json")
