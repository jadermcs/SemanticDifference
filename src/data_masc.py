import json
import gzip
import pandas as pd
from tqdm import tqdm
from data_wordnet import convert
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
            instance_data = {
                    "SENSE_KEY": key,
                    "LEMMA": lemma,
                    "USAGE": item["usage"],
                    "POS": convert(synset.pos())
            }
            entries.append(instance_data)
    df = pd.DataFrame(entries)
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

    df = merged[["LEMMA", "USAGE_x", "USAGE_y", "POS", "LABEL"]]
    df = df.dropna()

    filtered = df["LEMMA"] <= "j"
    df[filtered].to_json("data/masc.train.json", orient="records", indent=2)
    df[~filtered].to_json("data/masc.test.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
