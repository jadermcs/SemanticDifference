#!/usr/bin/env python
# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from datasets import load_dataset
from tqdm import trange
import math

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


def get_rotary_embedding(seq_len, dim, device):
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
    pos = torch.arange(seq_len, device=device).type_as(inv_freq)
    sinusoid_inp = torch.einsum("i,j->ij", pos, inv_freq)
    return torch.sin(sinusoid_inp).to(device), torch.cos(sinusoid_inp).to(device)

def apply_rotary_emb(q, k, sin, cos):
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q_rotated = torch.cat([q1 * cos + q2 * sin, q2 * cos - q1 * sin], dim=-1)
    k_rotated = torch.cat([k1 * cos + k2 * sin, k2 * cos - k1 * sin], dim=-1)
    return q_rotated, k_rotated

class RotaryMultiheadAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv_proj = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv_proj(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (B, L, H, D_head)

        sin, cos = get_rotary_embedding(L, self.head_dim, x.device)
        sin, cos = sin[None, :, None, :], cos[None, :, None, :]  # (1, L, 1, D//2)

        q, k = apply_rotary_emb(q, k, sin, cos)

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)

class DiTBlock(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RotaryMultiheadAttention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x

class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, t):
        half_dim = self.linear1.in_features // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        return self.linear2(self.act(self.linear1(emb)))

class DiTForMLM(nn.Module):
    def __init__(self, vocab_size, seq_len, dim, depth, n_heads):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.time_mlp = TimestepEmbedding(dim)
        self.transformer_blocks = nn.ModuleList([DiTBlock(dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)
        self.seq_len = seq_len

    def forward(self, x_t, t):
        B, L = x_t.shape
        x = self.token_emb(x_t)
        t_emb = self.time_mlp(t).unsqueeze(1)  # (B, 1, D)
        x = x + t_emb

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        return self.output(x)  # logits over vocabulary

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
    tokens = tokenizer(example["text"], truncation=True)
    return tokens

dataset = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train").select(range(100_000))
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

tokenized_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=8)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = DiTForMLM(vocab_size, 128, 768, 12, 12).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

progress_bar = trange(5000, desc="Training", leave=True)


@torch.no_grad()
def sample_sequence(model, seq_len, mask_token_id, num_steps=1000, temperature=1.0):
    device = next(model.parameters()).device
    B = 1

    # Starting prompt
    start = tokenizer("I like", return_tensors="pt")["input_ids"][0, :-1]
    append_len = seq_len
    x_t = torch.full((B, append_len), mask_token_id, dtype=torch.long, device=device)

    # Full sequence: start + to-be-generated
    x_t = torch.cat([start.unsqueeze(0).to(device), x_t], dim=1)  # (1, start_len + seq_len)
    total_len = x_t.size(1)

    for t_val in reversed(range(1, num_steps)):
        t = torch.full((B,), t_val, dtype=torch.long, device=device)

        # Model predicts logits
        logits = model(x_t, t)  # (B, L, vocab)
        probs = F.softmax(logits / temperature, dim=-1)

        # Sample new tokens from probabilities
        sampled_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), num_samples=1).view(B, total_len)

        # Gradual denoising:
        # At early steps, keep most tokens noisy; at later steps, keep predictions
        mask_ratio = t_val / num_steps
        mask = torch.bernoulli(torch.full(x_t.shape, mask_ratio, device=device)).bool()

        # Only update tokens where mask == 1
        x_t = torch.where(mask, sampled_tokens, x_t)

        print(tokenizer.batch_decode(x_t, skip_special_tokens=False), end='\r')

    print("\nFinal Output:", tokenizer.decode(x_t[0], skip_special_tokens=True))
    return x_t

for i, step in enumerate(progress_bar):
    batch = tokenized_dataset.shuffle(seed=42).select(range(32))
    x_0 = torch.tensor(batch["input_ids"]).to(device)
    t = torch.randint(1, 1000, (x_0.size(0),), device=device)
    x_t = corrupt_tokens(x_0, t, mask_token_id).to(device)

    logits = model(x_t, t)
    loss = F.cross_entropy(logits.view(-1, vocab_size), x_0.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 500 == 0 and i > 0:
        sample_sequence(model, seq_len=20, mask_token_id=mask_token_id)

    progress_bar.set_postfix(loss=f"{loss.item():.4f}")

sample_sequence(model, seq_len=20, mask_token_id=mask_token_id)
