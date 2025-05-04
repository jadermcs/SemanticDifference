#!/usr/bin/env python
# coding: utf-8
import argparse
from nltk.corpus.reader.sinica_treebank import WORD
import torch
import torch.nn as nn
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from functools import lru_cache
import json
from tqdm import tqdm
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
    PreTrainedModel,
    set_seed
)
import wandb
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
MLM_PROBABILITY = .3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
START_TARGET_TOKEN = "[TGT]"
END_TARGET_TOKEN = "[/TGT]"
PAD_SENSE_ID = 0 # Make sure sense ID 0 is reserved for this
WEIGHT_DECAY = 0.01
EVAL_STEPS = 500

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
    all_data = []
    for dataset in datasets.split(","):
        with open(f"data/{dataset}.{split}.json", 'r') as f:
            all_data.extend(json.load(f))

    processed_data = []
    for item in tqdm(all_data):
        w1, w2 = item['WORD_x'], item['WORD_y']
        s1 = item["USAGE_x"]
        s2 = item["USAGE_y"]

        if mark_target:
            s1 = s1.replace(w1, f"{START_TARGET_TOKEN}{w1}{END_TARGET_TOKEN}")
            s2 = s2.replace(w2, f"{START_TARGET_TOKEN}{w2}{END_TARGET_TOKEN}")

        if supersense:
            s1_tokens = word_tokenize(s1)
            s2_tokens = word_tokenize(s2)

            def encode_supersenses(tokens):
                ids = [SUPERSENSE_TO_ID[s] for word in tokens for s in get_word_supersenses(word)]
                return [
                    [1 if i in ids else 0 if ids else -100 for i in range(NUM_SUPERSENSE_CLASSES)]
                    for word in tokens
                ]

            processed_entry = {
                'sentence1': s1_tokens,
                'sentence2': s2_tokens,
                'labels': int(item['LABEL'] == 'identical'),
                'supersenses1': encode_supersenses(s1_tokens),
                'supersenses2': encode_supersenses(s2_tokens),
            }
        else:
            processed_entry = {
                'sentence1': s1,
                'sentence2': s2,
                'labels': int(item['LABEL'] == 'identical')
            }

        processed_data.append(processed_entry)

    return Dataset.from_list(processed_data)


# def mask_tokens(inputs, tokenizer, mlm_probability=.15):
#     """
#     Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
#     """
#     # We sample a few tokens in each sequence for MLM training (with probability self.mlm_probability)
#     labels = inputs.clone()
#     probability_matrix = torch.full(labels.shape, mlm_probability)
#     special_tokens_mask = tokenizer.get_special_tokens_mask(labels, already_has_special_tokens=True)
#     special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

#     probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
#     masked_indices = torch.bernoulli(probability_matrix).bool()
#     labels[~masked_indices] = -100  # We only compute loss on masked tokens

#     # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
#     indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
#     inputs[indices_replaced] = tokenizer.mask_token_id

#     # 10% of the time, we replace masked input tokens with random word
#     indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
#     random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
#     inputs[indices_random] = random_words[indices_random]

#     # The rest of the time (10% of the time) we keep the masked input tokens unchanged
#     return inputs, labels


def mask_tokens(inputs, tokenizer, mlm_probability=MLM_PROBABILITY):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    # We sample a few tokens in each sequence for MLM training (with probability self.mlm_probability)
    probability_matrix = torch.full(inputs.shape, mlm_probability)
    special_tokens_mask = tokenizer.get_special_tokens_mask(inputs, already_has_special_tokens=True)
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    # we replace masked input tokens with tokenizer.mask_token ([MASK])
    inputs[masked_indices] = tokenizer.mask_token_id

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, masked_indices

def preprocess_function(examples, tokenizer, supersense=False):
    """Tokenize input sentences and optionally process supersense labels."""
    if supersense:
        sent = examples['sentence1'] + [tokenizer.sep_token] + examples['sentence2']
    else:
        sent = examples['sentence1'] + tokenizer.sep_token + examples['sentence2']
    tokens = tokenizer(
        sent,
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
        return_token_type_ids=True,
        return_tensors='pt',
        is_split_into_words=supersense
    )
    tokens['input_ids'], mask_array = mask_tokens(tokens['input_ids'][0], tokenizer)

    if supersense:
        senses = examples['supersenses1'] + [[-100] * NUM_SUPERSENSE_CLASSES] + examples['supersenses2']
        word_ids = tokens.word_ids()
        word_ids = torch.tensor([-1 if id is None else id for id in word_ids])
        word_ids[~mask_array] = -1
        all_supersenses = torch.full((len(word_ids), NUM_SUPERSENSE_CLASSES), -100)

        valid_positions = word_ids != -1
        valid_word_ids = word_ids[valid_positions]

        # Convert senses to tensor if it's not already
        senses_tensor = torch.tensor(senses)  # shape: (num_words, NUM_SUPERSENSE_CLASSES)

        # Assign labels using advanced indexing
        all_supersenses[valid_positions] = senses_tensor[valid_word_ids]
        tokens['token_labels'] = all_supersenses

    tokens['labels'] = examples['labels']
    return tokens


@dataclass
class MultiTaskModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor | float] = None
    mlm_logits: Optional[torch.FloatTensor] = None
    sequence_logits: Optional[torch.FloatTensor] = None
    token_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class CustomMultiTaskModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.model = AutoModelForMaskedLM.from_pretrained(config._name_or_path, config=config) # Or your specific base model
        if config.num_token_labels > 0:
            self.sense_embeddings = nn.Embedding(config.num_token_labels, config.hidden_size, padding_idx=config.pad_sense_id)
            # Optional: Initialize sense embeddings (e.g., Xavier initialization)
            nn.init.xavier_uniform_(self.sense_embeddings.weight)
#
        # Example: Add a classification head
        self.sequence_classifier = nn.Linear(config.hidden_size, config.num_labels) # Binary classification
        self.token_classifier = nn.Linear(config.hidden_size, config.num_token_labels) # Token classification
        self.post_init()

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
        # 1. Get standard word embeddings
        embed = self.model.model.embeddings
        word_embeds = embed.tok_embeddings(input_ids)

        # 2. Get sense embeddings
        if sense_ids is not None:
            sense_embeds = self.sense_embeddings(sense_ids)
            # 3. Sum word and sense embedings
            # Make sure sense embeddings for PAD tokens are zero (handled by padding_idx)
            # or explicitly zero them out if needed based on attention mask/padding id.
            word_embeds = word_embeds + sense_embeds

        # --- Replicate the rest of BertEmbeddings forward pass ---
        # 4. Add position embeddings
        # position_ids = torch.arange(
        #     self.config.pad_token_id + 1, input_ids.size(1) + self.config.pad_token_id + 1, dtype=torch.long, device=input_ids.device
        #     )
        # position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        # position_embeds = self.model.roberta.embeddings.position_embeddings(position_ids)

        # 5. Add token type embeddings
        # token_type_embeds = self.model.roberta.embeddings.token_type_embeddings(token_type_ids.squeeze(1))

        # 6. Sum all embeddings
        # final_embeddings = word_embeds + position_embeds + token_type_embeds

        # 7. Apply LayerNorm and Dropout
        final_embeddings = embed.drop(embed.norm(word_embeds))
        # final_embeddings = self.model.roberta.embeddings.LayerNorm(final_embeddings)
        # final_embeddings = self.model.roberta.embeddings.dropout(final_embeddings)
        # --- End Replication ---

        # 8. Pass final embeddings to BERT encoder
        # We pass `inputs_embeds` instead of `input_ids`
        # We also need to pass the `attention_mask`
        # `token_type_ids` are effectively handled by the embedding addition above
        outputs = self.model(
            inputs_embeds=final_embeddings,
            # input_ids=input_ids,
            attention_mask=attention_mask.squeeze(1),
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
        token_logits = None
        mlm_logits = None
        token_loss = None

        if mlm_labels is not None:
            mlm_loss = outputs.loss
            loss += mlm_loss
        if token_labels is not None:
            # Compute token classification loss only on masked and valid positions
            token_logits = self.token_classifier(sequence_output)  # (B, L, C)

            # Create mask for valid positions (masked tokens and non -100 labels)
            valid_mask = (token_labels != -100)

            # Apply the mask
            token_labels = token_labels[valid_mask].float()  # match logits' shape
            token_logits = token_logits[valid_mask]

            # Compute loss
            loss_fct = nn.BCEWithLogitsLoss()
            token_loss = loss_fct(token_logits, token_labels)
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
            token_logits=token_logits,
            mlm_logits=mlm_logits,
            sequence_logits=sequence_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# def compute_metrics(pred):
#     """Compute metrics for evaluation."""
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
#     acc = accuracy_score(labels, preds)

#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }

def compute_metrics(pred):
    """Compute metrics for both sequence and token classification."""
    # Unpack predictions and labels
    seq_preds = pred.predictions['sequence']
    token_preds = pred.predictions['token']
    seq_labels = pred.label_ids['sequence']
    token_labels = pred.label_ids['token']

    # Sequence Classification (e.g., sentiment classification)
    seq_preds_argmax = seq_preds.argmax(-1)
    seq_precision, seq_recall, seq_f1, _ = precision_recall_fscore_support(
        seq_labels, seq_preds_argmax, average='weighted'
    )
    seq_acc = accuracy_score(seq_labels, seq_preds_argmax)

    # Token Classification (e.g., NER)
    token_preds_argmax = token_preds > .5

    # Flatten inputs, ignore special tokens (commonly labeled -100)
    true_token_labels = token_labels.flatten()
    pred_token_labels = token_preds_argmax.flatten()

    mask = true_token_labels != 100
    true_token_labels = true_token_labels[mask]
    pred_token_labels = pred_token_labels[mask]


    token_precision, token_recall, token_f1, _ = precision_recall_fscore_support(
        true_token_labels, pred_token_labels, average='weighted'
    )
    token_acc = accuracy_score(true_token_labels, pred_token_labels)

    return {
        # Sequence classification
        'seq_accuracy': seq_acc,
        'seq_f1': seq_f1,
        'seq_precision': seq_precision,
        'seq_recall': seq_recall,

        # Token classification
        'token_accuracy': token_acc,
        'token_f1': token_f1,
        'token_precision': token_precision,
        'token_recall': token_recall
    }

class MultiTaskTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = inputs.copy()
        labels = {
            "sequence": inputs.get("labels"),
            "token": inputs.get("token_labels")
        }
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        logits = {
            "sequence": outputs.sequence_logits,
            "token": outputs.token_logits
        }
        return None, logits, labels


def main():
    parser = argparse.ArgumentParser(description='Train a MLM model for difference classification')
    parser.add_argument('--model', type=str, default='FacebookAI/roberta-base', help='Pre-trained model to use')
    parser.add_argument('--dataset', type=str, default='wic', help='Path to the dataset file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--mark_target', action='store_true', default=False, help='Mark the target word in the sentences')
    parser.add_argument('--supersense', action='store_true', default=False, help='Use supersense classification')
    parser.add_argument('--output_dir', type=str, default='output/bert-classifier', help='Directory to save the model')
    parser.add_argument('--wandb_project', type=str, default='semantic-difference', help='Weights & Biases project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Weights & Biases run name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use FP16 precision')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Initialize wandb
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.model.split('/')[-1]}-{args.dataset}-classifier"

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args)
    )

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
        fn_kwargs={"tokenizer": tokenizer, "supersense": args.supersense},
        num_proc=4,
    )

    # Initialize model
    config = AutoConfig.from_pretrained(args.model, num_labels=2)
    config.num_token_labels = NUM_SUPERSENSE_CLASSES if args.supersense else 0
    config.pad_sense_id = PAD_SENSE_ID
    config.mask_token_id = tokenizer.mask_token_id
    model = CustomMultiTaskModel(config)
    # model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    if args.mark_target:
        tokenizer.add_tokens([START_TARGET_TOKEN, END_TARGET_TOKEN])
        model.resize_token_embeddings(len(tokenizer))

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="seq_f1",
        greater_is_better=True,
        label_names=["labels"],
        fp16=args.fp16,
        warmup_steps=WARMUP_STEPS,
        report_to="wandb",
        run_name=args.wandb_run_name
    )

    # Initialize trainer
    trainer = MultiTaskTrainer(
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
    wandb.log(metrics)

    # Save the model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
