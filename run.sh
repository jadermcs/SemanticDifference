python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic,semcor --wandb_run_name "roberta-base-wic-semcor"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic,semcor --mask --wandb_run_name "roberta-base-wic-semcor-mask"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic,semcor --target --wandb_run_name "roberta-base-semcor-target"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic,semcor --supersense --wandb_run_name "roberta-base-semcor-supersense"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic,semcor --mask --target --wandb_run_name "roberta-base-semcor-mask-target"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic,semcor --mask --supersense --wandb_run_name "roberta-base-wic-semcor-mask-supersense"
python src/bert-weak-supervision.py --model FacebookAI/roberta-base --dataset wic,semcor --target --supersense --wandb_run_name "roberta-base-wic-semcor-target-supersense"

python src/bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic,semcor --wandb_run_name "ModernBERT-base-wic-semcor"
python src/bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic,semcor --mask --wandb_run_name "ModernBERT-base-wic-semcor-mask"
python src/bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic,semcor --target --wandb_run_name "ModernBERT-base-wic-semcor-target"
python src/bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic,semcor --supersense --wandb_run_name "ModernBERT-base-wic-semcor-supersense"
python src/bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic,semcor --mask --target --wandb_run_name "ModernBERT-base-wic-semcor-mask-target"
python src/bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic,semcor --mask --supersense --wandb_run_name "ModernBERT-base-wic-semcor-mask-supersense"
python src/bert-weak-supervision.py --model answerdotai/ModernBERT-base --dataset wic,semcor --target --supersense --wandb_run_name "ModernBERT-base-wic-semcor-target-supersense"

# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --mask
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --target
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --supersense
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --mask --target
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --mask --supersense
# python bert-weak-supervision.py --model microsoft/deberta-v3-base --dataset wic --target --supersense
