import json
import gzip
import pandas as pd
from tqdm import tqdm
from nltk.corpus import wordnet as wn


def synset_to_sense_key(synset):
    """
    Convert a WordNet synset to its corresponding sense keys.

    Args:
        synset (wn.Synset): A WordNet synset.

    Returns:
        list: A list of sense keys for the given synset.
    """
    lem_s = synset.split(".")[0]
    synset = wn.synset(synset)
    sense_keys = [
            lemma.key() for lemma in synset.lemmas() if lemma.name() == lem_s]
    return sense_keys


def main():
    entries = []
    with gzip.open("data/semcor_en.jsonl.gz", "r") as fin:
        data = json.load(fin)
        print("Getting semcor data.")
        for item in tqdm(data):
            syn = [x for x in item["synsets"] if x.startswith(item["lemma"])]
            if syn:
                key = synset_to_sense_key(syn[0])
            if syn and key:
                key = key[0]
                if counter[key] > 2:
                    continue
                counter[key] += 1
                instance_data = {
                        "SENSE_KEY": key,
                        "LEMMA": item["lemma"],
                        "USAGE": item["text"].strip(),
                        "POS": 
                }
                entries.append(instance_data)
    df = pd.DataFrame(entries)
    merged = pd.merge(df, df, on="name")
    filterm = (
        (merged["POS_x"] == merged["POS_y"])
        & (merged["SENSE_x"] <= merged["SENSE_y"])
        & (merged["USAGE_x"] != merged["USAGE_y"])
    )

    merged = merged[filterm]

    merged["POS"] = merged["POS_x"]

    merged["LABEL"] = "identical"
    merged.loc[merged["SENSE_x"] != merged["SENSE_y"], "LABEL"] = "different"

    df = merged[["LEMMA", "USAGE_x", "USAGE_y", "POS", "LABEL"]]
    df = df.dropna()

    filtered = df["LEMMA"] <= "j"
    df[filtered].to_json("data/fews.train.json", orient="records", indent=2)
    df[~filtered].to_json("data/fews.test.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
