import argparse
import pandas as pd
from transformers import set_seed
from tqdm import tqdm
from guidance import models, gen, select


def main(raw_args=None):
    parser = argparse.ArgumentParser(
                    prog='prompt_generate.py',
                    description='What the program does',
                    epilog="""Generate the prompt for querying an LLM.

    Usage:
        prompt_generate.py [-n] <c1> <c2> <targets>

    Arguments:

        <dataset> = dataset
        <instruction> = give examples on how to do the task
        <task> = task to generate the prompt
        <reasoning> = prompt the model to reason before answering

    """)
    parser.add_argument('dataset')
    parser.add_argument('--rhetorics', action='store_true')
    parser.add_argument('--ctx', type=int, default=2048)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', required=True)
    parser.add_argument('--sample', action='store_true')
    args = parser.parse_args(raw_args)
    set_seed(args.seed)
    model = models.LlamaCpp(
            "models/"+args.model,
            n_gpu_layers=-1,
            n_ctx=args.ctx,
            flash_attn=True,
            echo=False)

    dataset = pd.read_json(args.dataset)

    user = "<｜User｜>"
    assistant = "<｜Assistant｜>"

    pred = []
    for idx, example in tqdm(dataset.iterrows(), total=dataset.shape[0]):
        lm = model
        lm += user
        lm += f"Does the word \'{example['LEMMA']}\' have the same meaning in the following sentences?\n"
        lm += f"1. {example['USAGE_x']}\n"
        lm += f"2. {example['USAGE_y']}\n\n"
        print(lm)
        lm += assistant
        lm += "<think>\n"
        if args.rhetorics:
            lm += "For this task I have to use zeugma for sense differentiation, I have to join both usages and check if it makes a bad pun if it does the senses are different."
        lm += gen(stop="</think>", max_tokens=1024, temperature=0.6)
        lm += "</think>"
        lm += "\nBased on my reasoning, here is my final answer:\n"
        lm += "\nA: " + select(["Yes", "No"], name="label")
        print(lm["label"])
        pred.append("identical" if lm["label"] == "Yes" else "different")
    dataset["pred"] = pred
    print(sum(dataset["LABEL"] == dataset["pred"])/dataset.shape[0])


if __name__ == '__main__':
    main()
