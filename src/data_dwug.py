#!/usr/bin/env python
# coding: utf-8
from glob import glob
import pandas as pd
import numpy

uses = []
for file in glob("dwug_en_resampled/data/*/uses.csv"):
    uses.append(pd.read_csv(file, sep="\t"))
uses = pd.concat(uses)

mapper = {k: v for k, v in uses[["identifier", "context"]].values}

judgments = []
for file in glob("dwug_en_resampled/data/*/judgments.csv"):
    judgments.append(pd.read_csv(file, sep="\t"))
judgments = pd.concat(judgments)


def match(x):
    if isinstance(x, numpy.ndarray):
        x = min(x)
    if x == 4:
        return "identical"
    return "different"


judgments_mode = judgments.groupby(["lemma", "identifier1", "identifier2"])[
    "judgment"].agg(pd.Series.mode).reset_index()
judgments_mode["judgment"] = judgments_mode["judgment"].apply(match)


judgments_mode["identifier1"] = judgments_mode["identifier1"].apply(mapper.get)
judgments_mode["identifier2"] = judgments_mode["identifier2"].apply(mapper.get)


judgments_mode = judgments_mode.rename(columns={
        "identifier1": "usage1",
        "identifier2": "usage2",
        "judgment": "label"
        })


def convert(pos):
    if pos == "nn":
        pos = "noun"
    elif pos == "vb":
        pos = "verb"
    else:
        raise "error"
    return pos


judgments_mode[["lemma", "pos"]] = judgments_mode["lemma"].str.split("_", n=1, expand=True)
judgments_mode["pos"] = judgments_mode["pos"].apply(convert)

filtered = judgments_mode["lemma"] <= "j"
judgments_mode[filtered].to_json("data/dwug.train.json", orient="records", indent=2)
judgments_mode[~filtered].to_json("data/dwug.test.json", orient="records", indent=2)
