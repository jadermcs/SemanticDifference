# python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic --wandb_run_name "roberta-base-wic"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic --mask --wandb_run_name "roberta-base-wic-mask"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic --target --wandb_run_name "roberta-base-wic-target"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic --supersense --wandb_run_name "roberta-base-wic-supersense"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic --mask --target --wandb_run_name "roberta-base-wic-mask-target"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic --mask --supersense --wandb_run_name "roberta-base-wic-mask-supersense"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic --target --supersense --wandb_run_name "roberta-base-wic-target-supersense"

# python bert-weak-supervision.py --model FacebookAI/roberta-base --dataset dwug
# python bert-weak-supervision.py --model FacebookAI/roberta-base --dataset dwug --mask
# python bert-weak-supervision.py --model FacebookAI/roberta-base --dataset dwug --target
# python bert-weak-supervision.py --model FacebookAI/roberta-base --dataset dwug --supersense
# python bert-weak-supervision.py --model FacebookAI/roberta-base --dataset dwug --mask --target
# python bert-weak-supervision.py --model FacebookAI/roberta-base --dataset dwug --mask --supersense
# python bert-weak-supervision.py --model FacebookAI/roberta-base --dataset dwug --target --supersense


# python bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wordnet
# python bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wordnet --mask
# python bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wordnet --target
# python bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wordnet --supersense
# python bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wordnet --mask --target
# python bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wordnet --mask --supersense
# python bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wordnet --target --supersense


# python bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic
# python bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic --mask
# python bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic --target
# python bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic --supersense
# python bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic --mask --target
# python bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic --mask --supersense
# python bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic --target --supersense

# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --mask
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --target
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --supersense
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --mask --target
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --mask --supersense
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --target --supersense
