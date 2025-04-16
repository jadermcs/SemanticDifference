# BERT Trainer for WiC Classification

This project uses Hugging Face's Trainer API to train a BERT model for binary classification on the Word-in-Context (WiC) dataset. The model classifies whether two sentences use the same word in the same sense (identical) or different senses (different).

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
       "LABEL": "identical" or "different"
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
- `--dataset`: Path to the dataset file (default: "data/wic.test.json")
- `--output_dir`: Directory to save the model (default: "output/bert-classifier")
- `--wandb_project`: Weights & Biases project name (default: "semantic-difference")
- `--wandb_run_name`: Weights & Biases run name (default: model name + "-wic-classifier")
- `--batch_size`: Batch size for training (default: 32)
- `--fp16`: Use FP16 precision (default: True)
- `--seed`: Random seed for reproducibility (default: 42)

### Example

```
python src/bert-trainer.py --model "bert-base-uncased" --dataset "data/wic.test.json" --output_dir "output/bert-wic" --wandb_project "semantic-difference" --wandb_run_name "bert-wic-classifier" --batch_size 16
```

## Metrics

The script reports the following metrics:
- Accuracy
- F1 Score
- Precision
- Recall

All metrics are logged to Weights & Biases for tracking and visualization.

## Output

The trained model is saved to the specified output directory. You can load it using the Hugging Face Transformers library:

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("output/bert-classifier")
tokenizer = AutoTokenizer.from_pretrained("output/bert-classifier")
```

