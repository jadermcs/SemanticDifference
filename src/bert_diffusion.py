#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import trange

MLM_PROBABILITY=0.3
NUM_EPOCHS=3

def mask_tokens(inputs, tokenizer, mlm_probability=MLM_PROBABILITY):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    mask_replace_prob = 0.8
    random_replace_prob = 0.1
    # We sample a few tokens in each sequence for MLM training (with probability self.mlm_probability)
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = IGNORE_ID  # We only compute loss on masked tokens

    # mask_replace_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, mask_replace_prob)).bool()
        & masked_indices
    )
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    if mask_replace_prob == 1 or random_replace_prob == 0:
        return inputs, labels

    remaining_prob = 1 - mask_replace_prob
    # scaling the random_replace_prob to the remaining probability for example if
    # mask_replace_prob = 0.8 and random_replace_prob = 0.1,
    # then random_replace_prob_scaled = 0.1 / 0.2 = 0.5
    random_replace_prob_scaled = random_replace_prob / remaining_prob

    # random_replace_prob% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, random_replace_prob_scaled)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time ((1-random_replace_prob-mask_replace_prob)% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def corrupt_tokens(x, t, mask_token_id):
    B, L = x.shape
    x_corrupt = x.clone()

    for i in range(B):
        p = t[i].item() / 1000
        mask = torch.rand(L) < p
        x_corrupt[i][mask] = mask_token_id

    return x_corrupt


class DiffusionMLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim=768, max_timesteps=1000):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.time_embed = nn.Embedding(max_timesteps, hidden_dim)
        encoder = nn.TransformerEncoderLayer(hidden_dim, 12)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=12)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_t, t):
        B, L = x_t.shape
        token_emb = self.token_embed(x_t)
        time_emb = self.time_embed(t).unsqueeze(1).expand(B, L, -1)
        h = token_emb + time_emb
        h = self.transformer(h.permute(1, 0, 2))  # L, B, D
        return self.output(h.permute(1, 0, 2))    # B, L, V


def train_step(model, optimizer, x_0, vocab_size, mask_token_id, device):
    B, L = x_0.shape
    t = torch.randint(1, 1000, (B,), device=device)
    x_t = corrupt_tokens(x_0, t, mask_token_id).to(device)

    logits = model(x_t, t)
    loss = F.cross_entropy(logits.view(-1, vocab_size), x_0.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

vocab_size = tokenizer.vocab_size
mask_token_id = tokenizer.mask_token_id


def tokenize_function(example):
    tokens = tokenizer(example["text"])
    return tokens

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train").select(range(100))
#dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

tokenized_dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names, batched=True, num_proc=8)

block_size = 128

def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_dataset = dataset.map(group_texts, batched=True, num_proc=8)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DiffusionMLM(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

progress_bar = trange(500, desc="Training", leave=True)

@torch.no_grad()
def sample_sequence(model, seq_len, mask_token_id, num_steps=1000):
    B = 1
    device = next(model.parameters()).device
    start = tokenizer("I like", return_tensors="pt")["input_ids"][0,:-1]
    append = torch.full((seq_len,), mask_token_id, dtype=torch.long)
    x_t = torch.cat([start, append]).unsqueeze(0).to(device)

    for t_val in reversed(range(1, num_steps)):
        t = torch.full((B,), t_val, dtype=torch.long, device=device)
        logits = model(x_t, t)
        x_t = logits.argmax(dim=-1)
        print(tokenizer.batch_decode(x_t, skip_special_tokens=False), end='\r')
    print("")

for i, step in enumerate(progress_bar):
    batch = tokenized_dataset.shuffle(seed=42).select(range(8))
    x_0 = torch.tensor(batch["input_ids"]).to(device)
    t = torch.randint(1, 1000, (x_0.size(0),), device=device)
    x_t = corrupt_tokens(x_0, t, mask_token_id).to(device)

    logits = model(x_t, t)
    loss = F.cross_entropy(logits.view(-1, vocab_size), x_0.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 100 == 0 and i > 0:
        sample_sequence(model, seq_len=20, mask_token_id=mask_token_id)

    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

@torch.no_grad()
def sample_sequence(model, seq_len, mask_token_id, num_steps=1000):
    B = 1
    device = next(model.parameters()).device
    x_t = torch.full((B, seq_len), mask_token_id, dtype=torch.long, device=device)

    for t_val in reversed(range(1, num_steps)):
        t = torch.full((B,), t_val, dtype=torch.long, device=device)
        logits = model(x_t, t)
        x_t = logits.argmax(dim=-1)

    return x_t

sampled_ids = sample_sequence(model, seq_len=32, mask_token_id=mask_token_id)
print(tokenizer.batch_decode(sampled_ids, skip_special_tokens=True))
