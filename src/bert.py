import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import spacy
import numpy as np # For potential averaging if needed, torch handles it here

# --- Setup ---
# Load spaCy model for lemmatization
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def get_token_embedding(word, sentence, tokenizer):
    doc = nlp(sentence.lower())
    tokens = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
    tokenized_text = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    print(tokenized_text)
    token_indices = [i for i, token in enumerate(tokenized_text) if word in token]
    return token_indices[0], tokens


input_texts = [
    ("plane", "I love plane s and want to be a pilot."),
    ("plane", "There was a plane crash yesterday."),
    ("plane", "Find a plane which is perpendicular to the vector."),
    ("apple", "I love apple and want to be an agricultor."),
    ("aircraft", "The aircraft F-12-X is an excelent war machine.")
]

model_path = "Alibaba-NLP/gte-modernbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# Tokenize the input texts
batch_dict = tokenizer(
    input_texts,
    max_length=1024,
    padding=True,
    truncation=True,
    add_special_tokens=True,
    return_tensors="pt",
)

embs = []
for w, context in input_texts:
    print(context)
    word_idx, tokenized = get_token_embedding(w, context, tokenizer)
    print(tokenized)
    outputs = model(**tokenized)
    embeddings = outputs.last_hidden_state[:, word_idx]
    embs.append(embeddings[0])
    print(embeddings.shape)

embs = torch.stack(embs)
# (Optionally) normalize embeddings
embeddings = F.normalize(embs, p=2, dim=1)
scores = (embeddings[:1] @ embeddings[1:].T) * 100
print(scores.tolist())
# [[42.89073944091797, 71.30911254882812, 33.664554595947266]]
