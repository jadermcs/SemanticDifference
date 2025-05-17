import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from datasets import load_dataset


# Masking schedule: alpha(t) = exp(-int_0^t beta(s) ds)
def alpha_t(t):
    return torch.exp(-20 * t)  # e.g., exponential schedule with β(t) = 20


def d_alpha_dt(t):
    return -20 * alpha_t(t)


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = tokenizer.vocab_size
mask_token_id = tokenizer.mask_token_id


# Dummy Transformer or other neural net to predict token distributions
class MD4Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 256)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=4), num_layers=4
        )
        self.output = nn.Linear(256, vocab_size)

    def forward(self, x, t):
        emb = self.embedding(x)  # [B, L, D]
        t = t.view(-1, 1, 1)
        time_emb = torch.sin(
            t * torch.arange(1, emb.size(-1) + 1, device=t.device) * 3.14
        )
        emb = emb + time_emb
        out = self.transformer(emb.permute(1, 0, 2))  # transformer expects [L, B, D]
        return self.output(out.permute(1, 0, 2))  # [B, L, vocab_size]


# Sample from q(x_t | x0)
def forward_diffusion(x0, t):
    # alpha(t) * x0 + (1 - alpha(t)) * mask
    B, L = x0.shape
    alpha = alpha_t(t).view(B, 1, 1)  # shape: [B, 1, 1]

    x0_onehot = F.one_hot(x0, num_classes=vocab_size).float()  # [B, L, m+1]
    mask_onehot = F.one_hot(
        torch.full_like(x0, mask_token_id), num_classes=vocab_size
    ).float()

    probs = alpha * x0_onehot + (1 - alpha) * mask_onehot
    xt = torch.distributions.Categorical(probs=probs).sample()
    return xt


# Continuous-time ELBO loss
def md4_loss(model, x0):
    B, L = x0.shape
    t = torch.rand(B, device=x0.device)  # Sample random time
    xt = forward_diffusion(x0, t)

    logits = model(xt, t)  # [B, L, vocab_size]
    log_probs = F.log_softmax(logits, dim=-1)

    x0_onehot = F.one_hot(x0, num_classes=vocab_size).float()
    mask = xt == mask_token_id

    # Only calculate cross-entropy on masked tokens
    ce_loss = -(x0_onehot * log_probs).sum(dim=-1)
    masked_loss = ce_loss * mask

    weight = d_alpha_dt(t) / (1 - alpha_t(t))
    return (weight.view(-1, 1) * masked_loss).sum() / mask.sum().clamp_min(1)


model = MD4Model(vocab_size).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


def tokenize_function(example):
    tokens = tokenizer(example["text"], truncation=True)
    return tokens


# dataset = load_dataset(
#     "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train"
# ).select(range(100_000))
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

tokenized_dataset = dataset.map(
    tokenize_function, remove_columns=dataset.column_names, batched=True, num_proc=8
)

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
for step in range(100000):
    batch = tokenized_dataset.shuffle(seed=42).select(range(32))
    x0 = torch.tensor(batch["input_ids"]).to(device)
    loss = md4_loss(model, x0)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}")
