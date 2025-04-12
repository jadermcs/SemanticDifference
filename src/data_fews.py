import json
import re
import gzip
import pandas as pd
from tqdm import tqdm
from data_wordnet import get_in_context_word


def main():
    print("Getting fews data.")
    entries = []
    with gzip.open("data/fews.json.gz", "r") as fin:
        data = json.load(fin)
        for item in tqdm(data):
            key = item["key"]
            instance = {
                    "SENSE_KEY": key,
                    "LEMMA": item["lemma"],
                    "USAGE": re.sub(r"</?WSD>", "", item["usage"]),
                    "POS": key.split(".")[1]
            }
            instance["WORD"] = get_in_context_word(
                instance["LEMMA"], instance["USAGE"]
            )
            entries.append(instance)
    df = pd.DataFrame(entries)
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
    train_data.to_json("data/fews.train.json", orient="records", indent=2)
    dev_data.to_json("data/fews.dev.json", orient="records", indent=2)
    test_data.to_json("data/fews.test.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
