import os
import pandas as pd
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

df = pd.read_json("data/dwug_sensediff.json").sample(100, random_state=42)
df[["label"]].to_csv("truth.txt", header=False, index=False)

model_name = "gpt-4o-mini"
pred = []
text = []
correct = 0
count = 0
for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    usage1 = row["usage1"]
    usage2 = row["usage2"]
    word = row["lemma"].split("_")[0]
    prompt = f"""
    Given two different sentences your task is to determine if a word has a identical
    or a different meaning between usages. For this task use zeugma, join both senteces
    by the same word and if the zeugma preserves the same sense (doesn't sound like a bad
    pun), they are identical, otherwise they are differente. After thinking give, the answer in the
    last line of your response.
    Examples:

    Word: plane
    Sentences:
    1) He loves planes and want to become a pilot.
    2) The plane landed just now.
    A: identical
    ---
    Word: cell
    Sentences:
    1) Anyone leaves a cell phone or handheld at home, many of them faculty members from nearby.
    2) I just watch the dirty shadow the window bar makes across the wall of my cell.
    A: different
    ---
    Word: arm
    Sentences:
    1) He has a short arm, a stubby hand, and a left eye that is open twice as wide.
    2) I could feel all his muscles tense up in his arm.
    A: identical

    Task:
    Word: {word}
    Sentences:
    1) {usage1}
    2) {usage2}
    """
    data = [
        {"role": "system", "content": "You are a linguist specialized in word meaning."},
        {"role": "user", "content": prompt},
        ]
    completion = client.chat.completions.create(
        model=model_name,
        messages=data
    )

    full_response = completion.choices[0].message.content
    response = full_response.split("\n")[-1].split(":")[-1].strip()
    pred.append(response)
    if row["label"] in response:
        correct += 1
    count += 1

with open(f"{model_name}.txt", "w") as fout:
    for response in pred:
        fout.write(f"{response}\n")
with open(f"{model_name}-full.txt", "w") as fout:
    for response in text:
        fout.write(f"{response}\n")
