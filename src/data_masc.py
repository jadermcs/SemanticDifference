import json
import gzip
import pandas as pd
from tqdm import tqdm
from data_wordnet import convert, get_in_context_word
from nltk.corpus import wordnet as wn


def main():
    print("Getting masc data.")
    entries = []
    with gzip.open("data/masc.json.gz", "r") as fin:
        data = json.load(fin)
        for item in tqdm(data):
            key = item["sense_key"].split(";")[0]
            try:
                synset = wn.synset_from_sense_key(key)
                lemma = [lemma.name() for lemma in synset.lemmas() if key.startswith(lemma.name())]
                lemma = lemma[0]
            except:
                continue
            instance = {
                    "SENSE_KEY": key,
                    "LEMMA": lemma,
                    "USAGE": item["usage"],
                    "POS": convert(synset.pos())
            }
            instance["WORD"] = get_in_context_word(
                instance["LEMMA"], instance["USAGE"]
            )
            entries.append(instance)
    df = pd.DataFrame(entries)
    df = df.dropna()
    df = df.groupby("SENSE_KEY").head(3)
    merged = pd.merge(df, df, on="LEMMA")
    filterm = (
        (merged["POS_x"] == merged["POS_y"])
        & (merged["SENSE_KEY_x"] <= merged["SENSE_KEY_y"])
        & (merged["USAGE_x"] != merged["USAGE_y"])
    )

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
    train_data.to_json("data/masc.train.json", orient="records", indent=2)
    dev_data.to_json("data/masc.dev.json", orient="records", indent=2)
    test_data.to_json("data/masc.test.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
