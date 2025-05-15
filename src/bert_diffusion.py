#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

def corrupt_tokens(x, vocab_size, t, mask_token_id, strategy="mask"):
    """
    Corrupt input tokens with a discrete noising schedule.
    x: [B, L] input token ids
    t: [B] timestep ids
    """
    B, L = x.shape
    x_corrupt = x.clone()

    for i in range(B):
        noise_prob = t[i].item() / 1000  # e.g. 1000 steps total
        mask = torch.rand(L) < noise_prob

        if strategy == "mask":
            x_corrupt[i][mask] = mask_token_id
        elif strategy == "random":
            x_corrupt[i][mask] = torch.randint(0, vocab_size, (mask.sum(),))
        elif strategy == "mixed":
            use_mask = torch.rand(L) < 0.5
            x_corrupt[i][mask & use_mask] = mask_token_id
            x_corrupt[i][mask & ~use_mask] = torch.randint(0, vocab_size, (mask.sum(),))

    return x_corrupt


class DiffusionMLM(nn.Module):
    def __init__(self, vocab_size, hidden_dim=256, num_layers=6, max_timesteps=1000):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, hidden_dim)
        self.time_embed = nn.Embedding(max_timesteps, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x_t, t):
        B, L = x_t.shape
        token_emb = self.token_embed(x_t)
        time_emb = self.time_embed(t).unsqueeze(1).expand(B, L, -1)
        h = token_emb + time_emb
        h = self.transformer(h.permute(1, 0, 2))  # Transformer expects [L, B, D]
        logits = self.output(h.permute(1, 0, 2))
        return logits


def train_step(model, optimizer, x_0, vocab_size, mask_token_id, device):
    B, L = x_0.shape
    t = torch.randint(1, 1000, (B,), device=device)
    x_t = corrupt_tokens(x_0, vocab_size, t, mask_token_id).to(device)

    logits = model(x_t, t)
    loss = F.cross_entropy(logits.view(-1, vocab_size), x_0.view(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


vocab_size = 30522  # e.g., BERT's vocab size
mask_token_id = 103  # [MASK] token
model = DiffusionMLM(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    for batch in dataloader:
        x_0 = batch["input_ids"].to(device)
        loss = train_step(model, optimizer, x_0, vocab_size, mask_token_id, device)
        print(f"Loss: {loss:.4f}")


@torch.no_grad()
def sample_sequence(model, seq_len, vocab_size, mask_token_id, num_steps=1000, strategy="argmax"):
    """
    Generate a sequence by reverse diffusion.
    """
    device = next(model.parameters()).device
    B = 1  # batch size
    x_t = torch.full((B, seq_len), mask_token_id, dtype=torch.long, device=device)

    for t_val in reversed(range(1, num_steps + 1)):
        t = torch.full((B,), t_val, dtype=torch.long, device=device)
        logits = model(x_t, t)  # [B, L, V]

        if strategy == "argmax":
            x_t = logits.argmax(dim=-1)
        elif strategy == "sample":
            probs = torch.softmax(logits, dim=-1)
            x_t = torch.multinomial(probs.view(-1, vocab_size), 1).view(B, seq_len)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return x_t  # final denoised tokens


from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
generated_ids = sample_sequence(model, seq_len=20, vocab_size=tokenizer.vocab_size, mask_token_id=tokenizer.mask_token_id)
print(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
