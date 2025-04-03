import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


model_path = "Alibaba-NLP/gte-modernbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

for dataset in ["dwug", "semcor", "masc", "wordnet", "fews"]:
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
        data.filter(lambda x: x["POS"] == pos).to_json(f"{dataset}.{pos}.predict.json")
