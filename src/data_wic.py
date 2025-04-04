#!/usr/bin/env python
# coding: utf-8
import pandas as pd


def convert(pos):
    if pos == "N":
        pos = "noun"
    elif pos == "V":
        pos = "verb"
    else:
        raise "error"
    return pos


def main():
    print("Getting wic data.")
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(f"data/wic/{split}/{split}.data.txt", sep="\t",
                         names=["LEMMA", "POS", "INDEX", "USAGE_x", "USAGE_y"])
        df2 = pd.read_csv(f"data/wic/{split}/{split}.gold.txt", names=["LABEL"])
        df["LABEL"] = df2["LABEL"].apply(lambda x: "identical" if x == "T" else "different")
        df["POS"] = df["POS"].apply(convert)
        df["WORD_x"] = df.apply(lambda x: x["USAGE_x"].split()[int(x["INDEX"].split("-")[0])], axis=1)
        df["WORD_y"] = df.apply(lambda x: x["USAGE_y"].split()[int(x["INDEX"].split("-")[1])], axis=1)

        df = df[["LEMMA", "WORD_x", "WORD_y",
                 "USAGE_x", "USAGE_y", "POS", "LABEL"]]
        df = df.dropna()

        df.to_json(f"data/wic.{split}.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
