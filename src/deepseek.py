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

        <corpus> = corpus
        <instruction> = give examples on how to do the task
        <task> = task to generate the prompt
        <reasoning> = prompt the model to reason before answering

    """)
    parser.add_argument('dataset')
    parser.add_argument('--rhetorics', action='store_true')
    parser.add_argument('--ctx', type=int, default=4096)
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

    corpus = pd.read_json(args.dataset)

    user = "<｜User｜>"
    assistant = "<｜Assistant｜>"

    for x in tqdm(corpus.iterrows()):
        print(x)
        exit()
        lm = model
        lm += user
        lm += f"Does the word '{x}' have the same meaning in the following sentences?"
        lm += assistant
        lm += "<think>"
        lm += gen(stop="</think>", max_tokens=2048)
        lm += "\nBased on my reasoning, here is my final answer:\n"
        lm += "\nA:" + select(["Yes", "No"])


if __name__ == '__main__':
    main()
