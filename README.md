# Multitask BERT Trainer

This project uses Hugging Face's Trainer API to train a BERT model for multiple tasks on the Word-in-Context (WiC) dataset:
1. **Masked Language Modeling (MLM)** - Predicts masked tokens in sentences
2. **Supersense Prediction** - Predicts the supersense of target words
3. **Binary Classification** - Classifies whether two sentences use the same word in the same sense (identical) or different senses (different)

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Make sure you have the WiC dataset in the `data` directory. The script expects the dataset to be in JSON format with the following structure:
   ```json
   [
     {
       "USAGE_x": "First sentence with target word",
       "USAGE_y": "Second sentence with target word",
       "WORD_x": "Target word in first sentence",
       "WORD_y": "Target word in second sentence",
       "LABEL": "identical" or "different",
       "SUPERSENSE_x": "noun.Tops" (optional),
       "SUPERSENSE_y": "noun.Tops" (optional)
     },
     ...
   ]
   ```

## Usage

Run the script with default parameters:
```
python src/bert-trainer.py
```

### Command-line Arguments

- `--model`: Pre-trained model to use (default: "FacebookAI/roberta-base")
- `--dataset`: Dataset name (default: "wic")
- `--mark_target`: Mark target words with special tokens (default: False)
- `--output_dir`: Directory to save the model (default: "output/multitask-bert")
- `--wandb_project`: Weights & Biases project name (default: "semantic-difference")
- `--wandb_run_name`: Weights & Biases run name (default: model name + "-multitask")
- `--batch_size`: Batch size for training (default: 16)
- `--fp16`: Use FP16 precision (default: True)
- `--seed`: Random seed for reproducibility (default: 42)
- `--mlm_weight`: Weight for MLM loss (default: 1.0)
- `--classifier_weight`: Weight for classifier loss (default: 1.0)
- `--supersense_weight`: Weight for supersense loss (default: 1.0)

### Example

```
python src/bert-trainer.py --model "bert-base-uncased" --dataset "wic" --mark_target --output_dir "output/multitask-bert" --wandb_project "semantic-difference" --wandb_run_name "bert-multitask" --batch_size 16 --mlm_weight 0.5 --classifier_weight 1.0 --supersense_weight 0.8
```

## Metrics

The script reports the following metrics for each task:

### Classification Metrics
- Accuracy
- F1 Score
- Precision
- Recall

### Supersense Prediction Metrics
- Supersense Accuracy
- Supersense F1 Score
- Supersense Precision
- Supersense Recall

### MLM Metrics
- MLM Perplexity

All metrics are logged to Weights & Biases for tracking and visualization.

## Model Architecture

The multitask model consists of:
1. A base BERT model for MLM
2. A classification head for binary classification
3. A supersense prediction head that uses the [CLS] token representation

The model combines losses from all three tasks with configurable weights.

## Output

The trained model is saved to the specified output directory. You can load it using the Hugging Face Transformers library:

```python
from transformers import AutoTokenizer
from src.bert-trainer import MultiTaskModel

tokenizer = AutoTokenizer.from_pretrained("output/multitask-bert")
model = MultiTaskModel.from_pretrained("output/multitask-bert")
```

## Supersense Mapping

The model uses a predefined mapping of supersenses to indices. The default mapping includes:
- 26 noun supersenses (noun.Tops, noun.act, noun.animal, etc.)
- 14 verb supersenses (verb.body, verb.change, verb.cognition, etc.)
- 2 adjective supersenses (adj.all, adj.pert)
- 1 adverb supersense (adv.all)

You can modify the `SUPERSENSE_MAPPING` in the script to match your specific supersense taxonomy.

