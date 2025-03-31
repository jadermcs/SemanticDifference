import json
import re
import gzip
import pandas as pd
from tqdm import tqdm


def main():
    entries = []
    with gzip.open("data/fews.jsonl.gz", "r") as fin:
        data = json.load(fin)
        print("Getting fews data.")
        for item in tqdm(data):
            key = item["key"]
            instance_data = {
                    "SENSE_KEY": key,
                    "LEMMA": item["lemma"],
                    "USAGE": re.sub(r"</?WSD>", "", item["usage"]),
                    "POS": key.split(".")[1]
            }
            entries.append(instance_data)
    df = pd.DataFrame(entries)
    merged = pd.merge(df, df, on="name")
    filterm = (merged["POS_x"] == merged["POS_y"]) & (
        merged["SENSE_x"] <= merged["SENSE_y"]) & (
        merged["USAGE_x"] != merged["USAGE_y"])

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
