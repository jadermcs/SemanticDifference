python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic,semcor --wandb_run_name "roberta-base-fews-masc-semcor"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic,semcor --mask --wandb_run_name "roberta-base-fews-masc-semcor-mask"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic,semcor --target --wandb_run_name "roberta-base-fews-masc-semcor-target"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic,semcor --supersense --wandb_run_name "roberta-base-fews-masc-semcor-supersense"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic,semcor --mask --target --wandb_run_name "roberta-base-fews-masc-semcor-mask-target"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic,semcor --mask --supersense --wandb_run_name "roberta-base-fews-masc-semcor-mask-supersense"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic,semcor --target --supersense --wandb_run_name "roberta-base-fews-masc-semcor-target-supersense"

python src/bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic,semcor --wandb_run_name "ModernBERT-base-fews-masc-semcor"
python src/bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic,semcor --mask --wandb_run_name "ModernBERT-base-fews-masc-semcor-mask"
python src/bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic,semcor --target --wandb_run_name "ModernBERT-base-fews-masc-semcor-target"
python src/bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic,semcor --supersense --wandb_run_name "ModernBERT-base-fews-masc-semcor-supersense"
python src/bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic,semcor --mask --target --wandb_run_name "ModernBERT-base-fews-masc-semcor-mask-target"
python src/bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic,semcor --mask --supersense --wandb_run_name "ModernBERT-base-fews-masc-semcor-mask-supersense"
python src/bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic,semcor --target --supersense --wandb_run_name "ModernBERT-base-fews-masc-semcor-target-supersense"

# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --mask
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --target
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --supersense
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --mask --target
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --mask --supersense
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --target --supersense
