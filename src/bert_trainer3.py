#!/usr/bin/env python
# coding: utf-8
import argparse
import torch
import torch.nn as nn
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from functools import lru_cache
import json
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
    set_seed
)
# import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import Dataset, DatasetDict
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
MAX_LENGTH = 128
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
NUM_EPOCHS = 10
START_TARGET_TOKEN = "[TGT]"
END_TARGET_TOKEN = "[/TGT]"
PAD_SENSE_ID = 0 # Make sure sense ID 0 is reserved for this

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()


# Get all supersense classes from WordNet
def get_supersense_classes():
    """Extract all supersense classes from WordNet."""
    supersenses = set()
    for synset in wordnet.all_synsets():
        if hasattr(synset, 'lexname'):
            supersenses.add(synset.lexname())
    return sorted(list(supersenses))


SUPERSENSE_CLASSES = get_supersense_classes()
SUPERSENSE_TO_ID = {supersense: idx for idx, supersense in enumerate(SUPERSENSE_CLASSES)}
NUM_SUPERSENSE_CLASSES = len(SUPERSENSE_CLASSES)


@lru_cache(maxsize=200000)
def get_word_supersenses(word):
    if len(word) < 4:
        return set()
    # Lemmatize the word
    word = lemmatizer.lemmatize(word)
    synsets = wordnet.synsets(word)
    return set(synset.lexname() for synset in synsets)


def load_data(datasets, split="train", mark_target=False, supersense=False):
    """Load and process the WiC dataset."""
    data = []
    for dataset in datasets.split(","):
        with open(f"data/{dataset}.{split}.json", 'r') as f:
            data.extend(json.load(f))

    # Convert to Hugging Face dataset format
    processed_data = []
    for item in data:
        w1 = item['WORD_x']
        w2 = item['WORD_y']
        s1 = item["USAGE_x"] if not mark_target else item['USAGE_x'].replace(w1, f"{START_TARGET_TOKEN}{w1}{END_TARGET_TOKEN}")
        s2 = item["USAGE_y"] if not mark_target else item['USAGE_y'].replace(w2, f"{START_TARGET_TOKEN}{w2}{END_TARGET_TOKEN}")
        data = {
            'sentence1': s1,
            'sentence2': s2,
            'labels': 1 if item['LABEL'] == 'identical' else 0
        }
        if supersense:
            s1 = word_tokenize(s1)
            s2 = word_tokenize(s2)
            supersenses1 = [[SUPERSENSE_TO_ID[supersense] for supersense in get_word_supersenses(word)] for word in s1]
            supersenses2 = [[SUPERSENSE_TO_ID[supersense] for supersense in get_word_supersenses(word)] for word in s2]
            supersenses1 = [[1 if i in sp1 else -100 for i in range(NUM_SUPERSENSE_CLASSES)] for sp1 in supersenses1]
            supersenses2 = [[1 if i in sp2 else -100 for i in range(NUM_SUPERSENSE_CLASSES)] for sp2 in supersenses2]
            data['supersenses1'] = supersenses1
            data['supersenses2'] = supersenses2
        processed_data.append(data)

    return Dataset.from_list(processed_data)


def preprocess_function(examples, tokenizer, supersense=False):
    """Tokenize the input sentences."""
    tokens = tokenizer(
        examples['sentence1'],
        examples['sentence2'],
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        return_token_type_ids=True,
        is_split_into_words=supersense
    )
    if supersense:
        s1 = examples['supersense1']
        s2 = examples['supersense2']
        new_supersenses = []
        word_ids = examples.word_ids()
        supersenses = s1
        passed = False
        for word_id in word_ids:
            if word_id is None:
                new_supersenses.append([-100 for _ in range(NUM_SUPERSENSE_CLASSES)])
            elif not passed and word_id + 1 == len(s1):
                new_supersenses.append(supersenses[word_id])
                supersenses = s2
                passed = True
            else:
                new_supersenses.append(supersenses[word_id])
        tokens['supersenses'] = new_supersenses

    tokens['labels'] = examples['labels']
    return tokens


def compute_metrics(pred):
    """Compute metrics for evaluation."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


@dataclass
class MultiTaskModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor|float] = None
    mlm_logits: Optional[torch.FloatTensor] = None
    sequence_logits: Optional[torch.FloatTensor] = None
    token_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class CustomMultiTaskModel(nn.Module):
    def __init__(self, model_name_or_path, num_labels, num_token_labels, pad_sense_id, loss_weights=None):
        super().__init__()
        self.num_labels = num_labels
        self.num_token_labels = num_token_labels
        self.pad_sense_id = pad_sense_id
        self.loss_weights = loss_weights
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=self.config) # Or your specific base model
        if num_token_labels > 0:
            self.sense_embeddings = nn.Embedding(num_token_labels, self.config.hidden_size, padding_idx=pad_sense_id)
            # Optional: Initialize sense embeddings (e.g., Xavier initialization)
            nn.init.xavier_uniform_(self.sense_embeddings.weight)

        # We need access to components of the standard embeddings layer
        # self.word_embeddings = self.model.roberta.embeddings.word_embeddings
        # self.position_embeddings = self.model.roberta.embeddings.position_embeddings
        # self.token_type_embeddings = self.model.roberta.embeddings.token_type_embeddings
        # self.LayerNorm = self.model.roberta.embeddings.LayerNorm
        # self.dropout = self.model.roberta.embeddings.dropout

        # Example: Add a classification head
        self.sequence_classifier = nn.Linear(self.config.hidden_size, num_labels) # Binary classification
        # self.token_classifier = nn.Linear(self.config.hidden_size, num_token_labels) # Token classification

    def forward(
        self,
        input_ids,
        sense_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,        # Labels for sequence classification
        mlm_labels=None, # Labels for MLM
        token_labels=None,    # Labels for token classification
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        seq_length = input_ids.size(1)

        # 1. Get standard word embeddings
        # word_embeds = self.word_embeddings(input_ids)

        # 2. Get sense embeddings
        if sense_ids is not None:
            sense_embeds = self.sense_embeddings(sense_ids)
            # 3. Sum word and sense embedings
            # Make sure sense embeddings for PAD tokens are zero (handled by padding_idx)
            # or explicitly zero them out if needed based on attention mask/padding id.
            word_embeds = word_embeds + sense_embeds

        # --- Replicate the rest of BertEmbeddings forward pass ---
        # 4. Add position embeddings
        # position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # position_embeds = self.position_embeddings(position_ids)

        # 5. Add token type embeddings
        # token_type_embeds = self.token_type_embeddings(token_type_ids)

        # 6. Sum all embeddings
        # final_embeddings = word_embeds + position_embeds + token_type_embeds

        # 7. Apply LayerNorm and Dropout
        # final_embeddings = self.LayerNorm(final_embeddings)
        # final_embeddings = self.dropout(final_embeddings)
        # --- End Replication ---

        # 8. Pass final embeddings to BERT encoder
        # We pass `inputs_embeds` instead of `input_ids`
        # We also need to pass the `attention_mask`
        # `token_type_ids` are effectively handled by the embedding addition above
        outputs = self.model(
            # inputs_embeds=final_embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=mlm_labels,
            token_type_ids=None, # Not needed here as types are in final_embeddings
            output_hidden_states=True,
            return_dict=True # Recommended
        )

        sequence_output = outputs.hidden_states[-1] # Hidden states of the last layer
        sequence_logits = self.sequence_classifier(sequence_output[:,0,:]) # (batch_size, num_sequence_labels)

        # --- Calculate Losses ---
        loss = 0.0
        mlm_loss = None
        sequence_loss = None
        token_loss = None

        if mlm_labels is not None:
            mlm_loss = outputs.loss
            loss += mlm_loss
        if token_labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            token_logits = self.token_classifier(outputs.hidden_states[-1]) # (batch_size, sequence_length, num_token_labels)
            token_loss = loss_fct(token_logits.view(-1), token_labels.view(-1))
            loss += token_loss
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            sequence_loss = loss_fct(sequence_logits.view(-1, self.num_labels), labels.view(-1))
            loss += sequence_loss

        if not return_dict:
            output = (sequence_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultiTaskModelOutput(
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



def main():
    parser = argparse.ArgumentParser(description='Train a MLM model for difference classification')
    parser.add_argument('--model', type=str, default='FacebookAI/roberta-base',
                        help='Pre-trained model to use')
    parser.add_argument('--dataset', type=str, default='wic',
                        help='Path to the dataset file')
    parser.add_argument('--mark_target', action='store_true', default=False,
                        help='Mark the target word in the sentences')
    parser.add_argument('--supersense', action='store_true', default=False,
                        help='Use supersense classification')
    parser.add_argument('--output_dir', type=str, default='output/bert-classifier',
                        help='Directory to save the model')
    parser.add_argument('--wandb_project', type=str, default='semantic-difference',
                        help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                        help='Weights & Biases run name')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--fp16', action='store_true', default=True,
                        help='Use FP16 precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Initialize wandb
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.model.split('/')[-1]}-{args.dataset}-classifier"

    # wandb.init(
    #     project=args.wandb_project,
    #     name=args.wandb_run_name,
    #     config=vars(args)
    # )

    # Load dataset
    train_dataset = load_data(args.dataset, split="train", mark_target=args.mark_target, supersense=args.supersense)
    test_dataset = load_data("wic", split="test", mark_target=args.mark_target, supersense=args.supersense)

    datasets = DatasetDict({
        "train": train_dataset,
        "test": test_dataset
    })

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, add_prefix_space=args.supersense)

    datasets = datasets.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer}
    )

    # Initialize model
    model = CustomMultiTaskModel(
        args.model,
        num_labels=2,
        num_token_labels=NUM_SUPERSENSE_CLASSES if args.supersense else 0,
        pad_sense_id=PAD_SENSE_ID
    )
    # model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    if args.mark_target:
        tokenizer.add_tokens([START_TARGET_TOKEN, END_TARGET_TOKEN])
        model.resize_token_embeddings(len(tokenizer))

    # Define training arguments
    training_args = TrainingArguments(
        # output_dir=args.output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=args.fp16,
        warmup_steps=WARMUP_STEPS,
        # report_to="wandb",
        # run_name=args.wandb_run_name
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["test"],
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    metrics = trainer.evaluate()

    # Log final metrics to wandb
    # wandb.log(metrics)

    # Save the model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Finish wandb run
    # wandb.finish()


if __name__ == "__main__":
    main()
