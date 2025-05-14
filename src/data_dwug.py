#!/usr/bin/env python
# coding: utf-8
from glob import glob
from data_wordnet import get_in_context_word
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
    print("Getting dwug data.")
    uses = []
    for file in glob("data/dwug/*/uses.csv"):
        uses.append(pd.read_csv(file, sep="\t"))
    uses = pd.concat(uses)

    mapper = {k: v for k, v in uses[["identifier", "context"]].values}

    judgments = []
    for file in glob("data/dwug/*/judgments.csv"):
        judgments.append(pd.read_csv(file, sep="\t"))
    judgments = pd.concat(judgments)

    judgments_mode = judgments.groupby([
        "lemma", "identifier1", "identifier2"])[
        "judgment"].agg(pd.Series.mode).reset_index()
    judgments_mode["judgment"] = judgments_mode["judgment"].apply(match)

    judgments_mode["identifier1"] = judgments_mode["identifier1"].apply(mapper.get)
    judgments_mode["identifier2"] = judgments_mode["identifier2"].apply(mapper.get)

    df = judgments_mode.rename(columns={
            "identifier1": "USAGE_x",
            "identifier2": "USAGE_y",
            "judgment": "LABEL"
            }).dropna()
    df[["LEMMA", "POS"]] = df["lemma"].str.split(
        "_", n=1, expand=True)
    df["POS"] = df["POS"].apply(convert)
    df["WORD_x"] = df.apply(
        lambda x: get_in_context_word(x["LEMMA"], x["USAGE_x"]), axis=1)
    df["WORD_y"] = df.apply(
        lambda x: get_in_context_word(x["LEMMA"], x["USAGE_y"]), axis=1)

    df = df[["LEMMA", "WORD_x", "WORD_y",
             "USAGE_x", "USAGE_y", "POS", "LABEL"]]
    df = df.dropna()

    train_data = df.sample(frac=0.8, random_state=42)
    test_data = df.drop(train_data.index)
    dev_data = test_data.sample(frac=.5, random_state=42)
    test_data = test_data.drop(dev_data.index)
    train_data.to_json("data/dwug.train.json", orient="records", indent=2)
    dev_data.to_json("data/dwug.dev.json", orient="records", indent=2)
    test_data.to_json("data/dwug.test.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
