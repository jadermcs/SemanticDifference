#!/usr/bin/env python
# coding: utf-8
import argparse
import torch
import torch.nn as nn
import json
from wordnet_utils import encode_supersenses, word_tokenize, NUM_SUPERSENSE_CLASSES
from tqdm import tqdm
from transformers import (
    ModernBertModel,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    AutoConfig,
    ModernBertPreTrainedModel,
    set_seed,
)
from transformers.models.modernbert.modeling_modernbert import ModernBertPredictionHead, _unpad_modernbert_input

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
MLM_PROBABILITY = 0.3
LEARNING_RATE = 2e-5
WARMUP_STEPS = 500
START_TARGET_TOKEN = "[TGT]"
END_TARGET_TOKEN = "[/TGT]"
WEIGHT_DECAY = 0.01
EVAL_STEPS = 500



def load_data(datasets, split="train", mark_target=False):
    """Load and process the WiC dataset."""
    all_data = []
    for dataset in datasets.split(","):
        with open(f"data/{dataset}.{split}.json", "r") as f:
            all_data.extend(json.load(f))

    processed_data = []
    for item in tqdm(all_data):
        w1, w2 = item["WORD_x"], item["WORD_y"]
        s1 = item["USAGE_x"]
        s2 = item["USAGE_y"]

        if mark_target:
            s1 = s1.replace(w1, f"{START_TARGET_TOKEN}{w1}{END_TARGET_TOKEN}")
            s2 = s2.replace(w2, f"{START_TARGET_TOKEN}{w2}{END_TARGET_TOKEN}")

        processed_entry = {
            "sentence1": s1,
            "sentence2": s2,
            "labels": int(item["LABEL"] == "identical"),
        }

        processed_data.append(processed_entry)

    return Dataset.from_list(processed_data)


def mask_tokens(inputs, tokenizer, mlm_probability=MLM_PROBABILITY):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    mask_replace_prob = 0.8
    random_replace_prob = 0.1
    # We sample a few tokens in each sequence for MLM training (with probability self.mlm_probability)
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # mask_replace_prob% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = (
        torch.bernoulli(torch.full(labels.shape, mask_replace_prob)).bool()
        & masked_indices
    )
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    if mask_replace_prob == 1 or random_replace_prob == 0:
        return inputs, labels

    remaining_prob = 1 - mask_replace_prob
    # scaling the random_replace_prob to the remaining probability for example if
    # mask_replace_prob = 0.8 and random_replace_prob = 0.1,
    # then random_replace_prob_scaled = 0.1 / 0.2 = 0.5
    random_replace_prob_scaled = random_replace_prob / remaining_prob

    # random_replace_prob% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, random_replace_prob_scaled)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time ((1-random_replace_prob-mask_replace_prob)% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def align(examples, tokenizer, supersense=False, mode="train"):
    inputs = tokenizer(
        examples["sentences"],
        truncation=True,
        return_offsets_mapping=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    if supersense:
        mask = [0] * NUM_SUPERSENSE_CLASSES
        labels = []
        for i, offsets in enumerate(inputs["offset_mapping"]):
            text = examples["sentences"][i]
            words = word_tokenize(text)
            word_labels = encode_supersenses(words)
            pointer = 0
            # Create a char-to-word index
            char_to_word = {}
            for word_idx, word in enumerate(words):
                for c in range(pointer, pointer + len(word)):
                    char_to_word[c] = word_idx
                pointer += len(word) + 1  # account for space
            label_ids = []
            for offset in offsets:
                start, end = offset.tolist()
                if start == end:
                    label_ids.append(mask)
                elif start in char_to_word:
                    word_idx = char_to_word[start]
                    label_ids.append(word_labels[word_idx])
                else:
                    label_ids.append(mask)
            labels.append(label_ids)
        inputs["token_labels"] = torch.tensor(labels)
    inputs["seq_labels"] = examples["labels"]
    if mode == "train":
        inputs["input_ids"], inputs["mlm_labels"] = mask_tokens(
            inputs["input_ids"], tokenizer
        )
    return inputs


def preprocess_function(examples, tokenizer):
    """Tokenize input sentences and optionally process supersense labels."""
    examples["sentences"] = (
        f"{examples['sentence1']} {tokenizer.sep_token} {examples['sentence2']}"
    )
    return examples


@dataclass
class MultiTaskModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    mlm_loss: Optional[torch.Tensor] = None
    token_loss: Optional[torch.Tensor] = None
    sequence_loss: Optional[torch.Tensor] = None
    mlm_logits: Optional[torch.FloatTensor] = None
    sequence_logits: Optional[torch.FloatTensor] = None
    token_logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class CustomMultiTaskModel(ModernBertPreTrainedModel):
    _tied_weights_keys = ["decoder.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.model = ModernBertModel(config)  # Or your specific base model
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)
        if config.num_token_labels > 0:
            self.sense_embeddings = nn.Parameter(
                torch.randn(config.num_token_labels, config.hidden_size)
            )
            nn.init.xavier_uniform_(self.sense_embeddings)
        #
        self.drop = nn.Dropout(config.classifier_dropout)
        self.sequence_classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.token_classifier = nn.Linear(config.hidden_size, config.num_token_labels)
        self.loss_seq = nn.CrossEntropyLoss()
        self.loss_tok = nn.BCEWithLogitsLoss()
        self.loss_mlm = nn.CrossEntropyLoss()

        self.post_init()

    @torch.compile(dynamic=False)
    def compiled_head(self, output: torch.Tensor) -> torch.Tensor:
        return self.head(output)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        seq_labels: Optional[torch.Tensor] = None,  # Labels for sequence classification
        mlm_labels: Optional[torch.Tensor] = None,  # Labels for MLM
        token_labels: Optional[torch.Tensor] = None,  # Labels for token classification
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 1. Get standard word embeddings
        embed = self.model.embeddings
        word_embeds = embed.tok_embeddings(input_ids)

        # 2. Get sense embeddings
        mask = (mlm_labels == -100) if mlm_labels is not None else None
        if token_labels is not None and mlm_labels is not None:
            # Only provide embeddings for unmasked tokens
            reshape_mask = ~mask.unsqueeze(-1).expand(-1, -1, self.config.num_token_labels)
            masked_token_labels = token_labels * reshape_mask
            sense_embeds = masked_token_labels.float() @ self.sense_embeddings
            word_embeds += sense_embeds

        inputs_embeds = embed.drop(embed.norm(word_embeds))

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]

        last_hidden_state = self.compiled_head(last_hidden_state)
        last_hidden_state = self.drop(last_hidden_state)

        if self.config.classifier_pooling == "cls":
            pooled_output = last_hidden_state[:, 0]
        else:
            pooled_output = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                dim=1, keepdim=True
            )

        sequence_logits = self.sequence_classifier(pooled_output)
        token_logits = self.token_classifier(last_hidden_state)
        mlm_logits = self.decoder(last_hidden_state)

        # --- Calculate Losses ---
        loss = torch.tensor(0.0, device=last_hidden_state.device)
        mlm_loss = None
        sequence_loss = None
        token_loss = None

        if mlm_labels is not None:
            mlm_labels = mlm_labels[mask]
            masked_mlm_logits = mlm_logits[mask]
            mlm_loss = self.loss_mlm(masked_mlm_logits.view(mlm_labels.shape[0], -1), mlm_labels.view(-1))
            loss += mlm_loss
        if token_labels is not None and mlm_labels is not None:
            # Create mask for valid positions (masked tokens and non -100 labels)
            reshape_mask = mask.unsqueeze(-1).expand(-1, -1, self.sense_embeddings.size(0))
            # Apply the mask
            token_labels = token_labels[reshape_mask].float()  # match logits' shape
            masked_token_logits = token_logits[reshape_mask]
            # Compute loss
            token_loss = self.loss_tok(masked_token_logits, token_labels)
            uniform_loss = (
                masked_token_logits.softmax(dim=-1).sum() / token_labels.sum()
            )
            loss += token_loss + uniform_loss
        if seq_labels is not None:
            sequence_loss = self.loss_seq(sequence_logits.view(-1, self.num_labels), seq_labels.view(-1))
            loss += sequence_loss

        if not return_dict:
            output = (sequence_logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MultiTaskModelOutput(
            loss=loss,
            sequence_loss=sequence_loss,
            token_loss=token_loss,
            mlm_loss=mlm_loss,
            sequence_logits=sequence_logits,
            token_logits=token_logits,
            mlm_logits=mlm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def compute_metrics(pred):
    """Compute metrics for both sequence and token classification."""
    # Unpack predictions and labels
    seq_preds = pred.predictions["sequence"]
    seq_labels = pred.label_ids["sequence"]

    token_preds = pred.predictions.get("token")
    token_labels = pred.label_ids.get("token")

    # Sequence Classification (e.g., sentiment classification)
    seq_preds_argmax = seq_preds.argmax(-1)
    seq_precision, seq_recall, seq_f1, _ = precision_recall_fscore_support(
        seq_labels, seq_preds_argmax, average="weighted"
    )
    seq_acc = accuracy_score(seq_labels, seq_preds_argmax)

    metrics = {
        # Get loss
        "loss": pred.predictions.get("loss").mean(),
        # Sequence classification
        "seq_accuracy": seq_acc,
        "seq_f1": seq_f1,
        "seq_precision": seq_precision,
        "seq_recall": seq_recall,
    }
    # Token Classification (e.g., NER)
    if token_labels is not None:
        token_preds_argmax = token_preds > 0.5

        # Flatten inputs, ignore special tokens (commonly labeled -100)
        true_token_labels = token_labels.flatten()
        pred_token_labels = token_preds_argmax.flatten()

        mask = true_token_labels != 100
        true_token_labels = true_token_labels[mask]
        pred_token_labels = pred_token_labels[mask]

        token_precision, token_recall, token_f1, _ = precision_recall_fscore_support(
            true_token_labels, pred_token_labels, average="weighted"
        )
        token_acc = accuracy_score(true_token_labels, pred_token_labels)

        # Token classification
        metrics.update(
            {
                "token_accuracy": token_acc,
                "token_f1": token_f1,
                "token_precision": token_precision,
                "token_recall": token_recall,
            }
        )
    return metrics


class MultiTaskTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = inputs.copy()
        label_ids = {
            "sequence": inputs["seq_labels"],
        }
        if "token_labels" in inputs:
            label_ids["token"] = inputs["token_labels"]

        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)

        predictions = {
            "loss": outputs.loss,
            "sequence": outputs.sequence_logits,
            "token": outputs["token_logits"].sigmoid()
        }

        return None, predictions, label_ids


def main():
    parser = argparse.ArgumentParser(
        description="Train a MLM model for difference classification"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="FacebookAI/roberta-base",
        help="Pre-trained model to use",
    )
    parser.add_argument(
        "--dataset", type=str, default="wic", help="Path to the dataset file"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train"
    )
    parser.add_argument(
        "--mark_target",
        action="store_true",
        default=False,
        help="Mark the target word in the sentences",
    )
    parser.add_argument(
        "--supersense",
        action="store_true",
        default=False,
        help="Use supersense classification",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/bert-classifier",
        help="Directory to save the model",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="semantic-difference",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default=None, help="Weights & Biases run name"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--fp16", action="store_true", default=True, help="Use FP16 precision"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Initialize wandb
    if args.wandb_run_name is None:
        args.wandb_run_name = f"{args.model.split('/')[-1]}-{args.dataset}-classifier"
        args.wandb_run_name += "-supersense" if args.supersense else ""

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # Load dataset
    train_dataset = load_data(args.dataset, split="train", mark_target=args.mark_target)
    test_dataset = load_data("wic", split="test", mark_target=args.mark_target)

    datasets = DatasetDict({"train": train_dataset, "test": test_dataset})

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    datasets = datasets.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer},
        num_proc=4,
    )

    datasets["train"] = datasets["train"].map(
        align,
        fn_kwargs={"tokenizer": tokenizer, "supersense": args.supersense},
        batched=True,
        remove_columns=datasets["train"].column_names,
        num_proc=4,
    )
    # During test we show all the tokens however we don't give supersense embeddings
    datasets["test"] = datasets["test"].map(
        align,
        fn_kwargs={
            "tokenizer": tokenizer,
            "supersense": args.supersense,
            "mode": "test",
        },
        batched=True,
        remove_columns=datasets["test"].column_names,
        num_proc=4,
    )

    # Initialize model
    config = AutoConfig.from_pretrained(args.model, num_labels=2)
    config.num_token_labels = NUM_SUPERSENSE_CLASSES if args.supersense else 0
    config.embedding_dropout = 0.1
    config.classifier_dropout = 0.1
    model = CustomMultiTaskModel(config)
    # model = torch.compile(model, mode="max-autotune")

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
        gradient_accumulation_steps=32//args.batch_size,
        dataloader_num_workers=4,
        dataloader_pin_memory=True if device.type == "cuda" else False,
        report_to="wandb",
        run_name=args.wandb_run_name,
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
