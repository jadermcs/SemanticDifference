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
        pos = "adverb"
    else:
        raise "error"
    return pos


def main():
    print("Getting wordnet data.")
    data = []
    for k in wordnet.all_eng_synsets():
        for synset in k.lemmas():
            sense = k.name()
            name = synset.name()
            pos = convert(k.pos())
            usages = [e for e in k.examples() if name in e]
            for usage in usages:
                data.append({"LEMMA": name, "SENSE_KEY": sense,
                            "USAGE": usage, "POS": pos})

    df = pd.DataFrame(data)

    merged = pd.merge(df, df, on="LEMMA")

    filterm = (merged["POS_x"] == merged["POS_y"]) & (
        merged["SENSE_KEY_x"] <= merged["SENSE_KEY_y"]) & (
        merged["USAGE_x"] != merged["USAGE_y"])

    merged = merged[filterm]

    merged["POS"] = merged["POS_x"]

    merged["LABEL"] = "identical"
    merged.loc[merged["SENSE_KEY_x"] != merged["SENSE_KEY_y"], "LABEL"] = "different"

    df = merged[["LEMMA", "USAGE_x", "USAGE_y", "POS", "LABEL"]]
    df = df.dropna()

    filtered = df["LEMMA"] <= "j"
    df[filtered].to_json("data/wordnet.train.json", orient="records", indent=2)
    df[~filtered].to_json("data/wordnet.test.json", orient="records", indent=2)


if __name__ == "__main__":
    main()
