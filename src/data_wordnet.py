from nltk.corpus import wordnet
import pandas as pd


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
        pos = "adverb (s)"
    else:
        raise "error"
    return pos


data = []
for k in wordnet.all_eng_synsets():
    for synset in k.lemmas():
        sense = k.name()
        name = synset.name()
        pos = convert(k.pos())
        usages = [e for e in k.examples() if name in e]
        for usage in usages:
            data.append({"name": name, "sense": sense,
                        "usage": usage, "pos": pos})

df = pd.DataFrame(data)

merged = pd.merge(df, df, on="name")

filterm = (merged["pos_x"] == merged["pos_y"]) & (
    merged["sense_x"] <= merged["sense_y"]) & (
    merged["usage_x"] != merged["usage_y"])

merged = merged[filterm]

merged["pos"] = merged["pos_x"]

merged["label"] = "identical"
merged.loc[merged["sense_x"] != merged["sense_y"], "label"] = "different"

df = merged[["name", "usage_x", "usage_y", "label", "pos"]]
df = df.dropna()

filtered = df["name"] <= "j"
df[filtered].to_json("data/wordnet.train.json", orient="records", indent=2)
df[~filtered].to_json("data/wordnet.test.json", orient="records", indent=2)
