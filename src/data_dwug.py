#!/usr/bin/env python
# coding: utf-8
from glob import glob
import pandas as pd
import numpy


def match(x):
    if isinstance(x, numpy.ndarray):
        x = min(x)
    if x == 4:
        return "identical"
    return "different"


def convert(pos):
    if pos == "nn":
        pos = "noun"
    elif pos == "vb":
        pos = "verb"
    else:
        raise "error"
    return pos


def main():
    print("Getting masc data.")
    uses = []
    for file in glob("dwug_en_resampled/data/*/uses.csv"):
        uses.append(pd.read_csv(file, sep="\t"))
    uses = pd.concat(uses)

    mapper = {k: v for k, v in uses[["identifier", "context"]].values}

    judgments = []
    for file in glob("dwug_en_resampled/data/*/judgments.csv"):
        judgments.append(pd.read_csv(file, sep="\t"))
    judgments = pd.concat(judgments)

    judgments_mode = judgments.groupby(["lemma", "identifier1", "identifier2"])[
        "judgment"].agg(pd.Series.mode).reset_index()
    judgments_mode["judgment"] = judgments_mode["judgment"].apply(match)

    judgments_mode["identifier1"] = judgments_mode["identifier1"].apply(mapper.get)
    judgments_mode["identifier2"] = judgments_mode["identifier2"].apply(mapper.get)

    judgments_mode = judgments_mode.rename(columns={
            "identifier1": "USAGE_x",
            "identifier2": "USAGE_y",
            "judgment": "LABEL"
            })

    judgments_mode[["LEMMA", "POS"]] = judgments_mode["lemma"].str.split("_", n=1, expand=True)
    judgments_mode["POS"] = judgments_mode["POS"].apply(convert)
    df = judgments_mode[["LEMMA", "USAGE_x", "USAGE_y", "POS", "LABEL"]]
    df = df.dropna()

    filtered = df["LEMMA"] <= "j"
    df[filtered].to_json("data/dwug.train.json", orient="records", indent=2)
    df[~filtered].to_json("data/dwug.test.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
