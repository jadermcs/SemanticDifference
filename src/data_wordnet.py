#!/usr/bin/env python
# coding: utf-8
from nltk.corpus import wordnet
import pandas as pd
import spacy
from fuzzywuzzy import process, fuzz
from tqdm import tqdm

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def get_in_context_word(lemma, usage):
    if lemma in usage.split():
        return lemma
    doc = nlp(usage)
    words = {token.lemma_.lower(): token.text for token in doc}
    if lemma in words:
        return words[lemma]
    else:
        # Try fuzzy match
        best_match, _ = process.extractOne(
            lemma, words.values(), scorer=fuzz.partial_ratio)
        return best_match


def convert(pos):
    if pos == "n":
        pos = "noun"
    elif pos == "v":
        pos = "verb"
    elif pos == "a":
        pos = "adjective"
    elif pos == "r":
        pos = "adverb"
    elif pos == "s":
        pos = "adverb"
    else:
        raise "error"
    return pos


def main():
    print("Getting wordnet data.")
    data = []
    for k in tqdm(wordnet.all_eng_synsets()):
        for synset in k.lemmas():
            sense = k.name()
            name = synset.name()
            pos = convert(k.pos())
            usages = [e for e in k.examples() if name in e]
            for usage in usages:
                word = get_in_context_word(name, usage)
                data.append({"LEMMA": name, "SENSE_KEY": sense,
                             "USAGE": usage, "POS": pos, "WORD": word})

    df = pd.DataFrame(data)
    df = df.dropna()

    merged = pd.merge(df, df, on="LEMMA")

    filterm = (merged["POS_x"] == merged["POS_y"]) & (
        merged["SENSE_KEY_x"] <= merged["SENSE_KEY_y"]) & (
        merged["USAGE_x"] != merged["USAGE_y"])

    merged = merged[filterm]

    merged["POS"] = merged["POS_x"]

    merged["LABEL"] = "identical"
    merged.loc[merged["SENSE_KEY_x"] != merged["SENSE_KEY_y"], "LABEL"] = "different"

    df = merged[["LEMMA", "WORD_x", "WORD_y",
                 "USAGE_x", "USAGE_y", "POS", "LABEL"]]

    train_data = df.sample(frac=0.8, random_state=42)
    test_data = df.drop(train_data.index)
    dev_data = test_data.sample(frac=.5, random_state=42)
    test_data = test_data.drop(dev_data.index)
    train_data.to_json("data/wordnet.train.json", orient="records", indent=2)
    dev_data.to_json("data/wordnet.dev.json", orient="records", indent=2)
    test_data.to_json("data/wordnet.test.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
