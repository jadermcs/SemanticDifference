from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from transformers import PreTrainedModel, AutoConfig, AutoModelForSequenceClassification
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DataCollatorForJointMLMClassification:
    """
    Data collator that prepares batches for multitask learning:
    - Applies MLM masking to input_ids if mlm=True.
    - Pads inputs ('input_ids', 'attention_mask', 'token_type_ids').
    - Pads MLM 'labels'.
    - Preserves and stacks classification labels ('label_diff', 'label_supersense').
    """
    tokenizer: PreTrainedTokenizerBase
    mlm: bool = True  # Whether to apply MLM
    mlm_probability: float = 0.15
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        if self.mlm and self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is required for masked language modeling. "
                "You should pass `mlm=False` to train on causal language modeling instead."
            )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract classification labels before padding specific keys
        label_diff_list = [feature["label_diff"] for feature in features]
        label_supersense_list = [feature["supersenses"] for feature in features]

        # Prepare features for padding (remove scalar classification labels first)
        keys_to_pad = ['input_ids', 'attention_mask', 'token_type_ids'] # Add others if needed
        padding_features = [{k: v for k, v in feature.items() if k in keys_to_pad} for feature in features]

        # Pad the input features
        batch = self.tokenizer.pad(
            padding_features,
            padding=True, # Pad to longest in batch
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt", # Return PyTorch tensors
        )

        # --- Apply MLM ---
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"]
            )
        else:
            # If not applying MLM, MLM labels are ignored
            batch["labels"] = torch.full_like(batch["input_ids"], -100)

        # Add back the classification labels as tensors
        batch["sequence_labels"] = torch.tensor(label_diff_list, dtype=torch.long)
        batch["token_labels"] = torch.tensor(label_supersense_list, dtype=torch.long)

        return batch

    # Utility function borrowed & adapted from DataCollatorForLanguageModeling
    # Handles the actual masking procedure
    def torch_mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability self.mlm_probability)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels





import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForMaskedLM
from transformers.modeling_outputs import ModelOutput

# Define a custom output class to hold outputs for all tasks
@dataclass
class MultiTaskModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    mlm_logits: torch.FloatTensor = None
    sequence_logits: torch.FloatTensor = None
    token_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class CustomMultiTaskModel(nn.Module):
    def __init__(self, model_name_or_path, num_sequence_labels, num_token_labels, loss_weights=None):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name_or_path, config=self.config) # Or your specific base model

        # MLM Head (often part of the base model architecture for BERT-like models, e.g., via BertLMPredictionHead)
        # If using AutoModel, you might need to add the MLM head manually or use BertForMaskedLM as the base and access its components.
        # For simplicity, let's assume we might need to re-implement or fetch it if not using BertForMaskedLM directly.
        # We'll assume self.bert has or we add an MLM prediction capability later if needed.
        # A dedicated MLM head might look like this if needed:

        # Sequence Classification Head
        self.sequence_classifier = nn.Linear(self.config.hidden_size, num_sequence_labels)

        # Token Classification Head
        self.token_classifier = nn.Linear(self.config.hidden_size, num_token_labels)

        self.num_sequence_labels = num_sequence_labels
        self.num_token_labels = num_token_labels

        # Loss weights (optional, for combining losses)
        self.loss_weights = loss_weights if loss_weights else {"mlm": 1.0, "seq": 1.0, "token": 1.0}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,        # Labels for MLM
        sequence_labels=None, # Labels for sequence classification
        token_labels=None,    # Labels for token classification
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        sequence_output = outputs.hidden_states[-1] # Last hidden state (batch_size, sequence_length, hidden_size)
        
        # --- Calculate Logits ---
        # MLM Logits (predicting masked tokens)
        # If using BertForMaskedLM, this would be handled differently.
        # If using AutoModel, apply the head. Note: this is a simplification.
        # A proper MLM head often involves transformations + LayerNorm + bias.
        mlm_logits = outputs.logits

        # Sequence Classification Logits
        sequence_logits = self.sequence_classifier(sequence_output[:,0,:]) # (batch_size, num_sequence_labels)

        # Token Classification Logits
        token_logits = self.token_classifier(sequence_output) # (batch_size, sequence_length, num_token_labels)

        # --- Calculate Losses ---
        total_loss = None
        if labels is not None and sequence_labels is not None and token_labels is not None:
            loss_fct = CrossEntropyLoss() # Common loss function

            # MLM Loss
            mlm_loss = loss_fct(mlm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            # Sequence Classification Loss
            seq_loss = loss_fct(sequence_logits.view(-1, self.num_sequence_labels), sequence_labels.view(-1))

            # Token Classification Loss (ignoring padding, typically -100)
            # Only compute loss for non-special tokens if needed
            active_loss = attention_mask.view(-1) == 1 # Or based on token_labels != -100
            active_logits = token_logits.view(-1, self.num_token_labels)[active_loss]
            active_labels = token_labels.view(-1)[active_loss]
            if active_logits.shape[0] > 0: # Ensure there are valid tokens to compute loss on
                 token_loss = loss_fct(active_logits, active_labels)
            else:
                 # Handle cases where the batch might only contain padding after filtering
                 # Or if no token labels are present for the active parts
                 token_loss = torch.tensor(0.0, device=sequence_logits.device) # Ensure loss is on the correct device


            # Combine losses (e.g., weighted sum)
            total_loss = (self.loss_weights["mlm"] * mlm_loss +
                          self.loss_weights["seq"] * seq_loss +
                          self.loss_weights["token"] * token_loss)

        if not return_dict:
            output = (mlm_logits, sequence_logits, token_logits) + outputs[1:] # Add hidden states and attentions if requested
            return ((total_loss,) + output) if total_loss is not None else output

        return MultiTaskModelOutput(
            loss=total_loss,
            mlm_logits=mlm_logits,
            sequence_logits=sequence_logits,
            token_logits=token_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

from transformers import Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss

class MultitaskTrainerJoint(Trainer): # Renamed for clarity
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        print("compute_loss")
        # Extract all labels. The collator ensures these keys exist.
        sequence_labels = inputs.pop("sequence_labels", None)
        token_labels = inputs.pop("token_labels", None)
        labels = inputs.pop("labels", None)

        # Forward pass - Model gets potentially masked input_ids
        outputs = model(**inputs, return_dict=True)

        # Extract logits (ensure keys match your MultitaskOutput and model forward pass)
        sequence_logits = outputs.sequence_logits
        token_logits = outputs.token_logits
        mlm_logits = outputs.mlm_logits

        total_loss = 0.0
        loss_fct = CrossEntropyLoss() # Use appropriate loss, potentially different ones per task

        # --- Calculate Loss for Each Task ---
        # Sequence Task Loss (calculated if sequence_labels is provided and not -100)
        if sequence_labels is not None and sequence_logits is not None:
            # Filter out samples with ignore_index if used (e.g., -100)
            active_loss_diff = sequence_labels.view(-1) != -100
            if active_loss_diff.sum() > 0 :
                active_logits = sequence_logits.view(-1, self.model.num_sequence_labels)[active_loss_diff]
                active_labels = sequence_labels.view(-1)[active_loss_diff]
                # Ensure active_labels are within the valid range [0, num_labels_diff-1]
                if torch.all(active_labels >= 0) and torch.all(active_labels < self.model.num_sequence_labels):
                     total_loss += loss_fct(active_logits, active_labels) * 0.5 # Example weighting
                # else: print warning or handle invalid labels
            elif sequence_labels.numel() > 0 and torch.all(sequence_labels == -100):
                 total_loss += 0.0 * sequence_logits.sum() # Handle batches with only ignored labels

        # Token Task Loss (calculated if token_labels is provided and not -100)
        if token_labels is not None and token_logits is not None:
            active_loss_ss = token_labels.view(-1) != -100
            if active_loss_ss.sum() > 0:
                active_logits = token_logits.view(-1, self.model.num_token_labels)[active_loss_ss]
                active_labels = token_labels.view(-1)[active_loss_ss]
                # Ensure active_labels are within the valid range [0, num_labels_supersense-1]
                if torch.all(active_labels >= 0) and torch.all(active_labels < self.model.num_token_labels):
                    total_loss += loss_fct(active_logits, active_labels) * 0.5 # Example weighting
                # else: print warning or handle invalid labels
            elif token_labels.numel() > 0 and torch.all(token_labels == -100):
                 total_loss += 0.0 * token_logits.sum() # Handle batches with only ignored labels

        # MLM Task Loss (calculated only for masked tokens where labels != -100)
        if labels is not None and mlm_logits is not None:
            active_loss_mlm = labels.view(-1) != -100
            if active_loss_mlm.sum() > 0:
                active_logits = mlm_logits.view(-
                1, self.model.config.vocab_size)[active_loss_mlm]
                active_labels = labels.view(-1)[active_loss_mlm]
                total_loss += loss_fct(active_logits, active_labels) * 1.0 # Example weighting
            elif labels.numel() > 0 and torch.all(labels == -100):
                 total_loss += 0.0 * mlm_logits.sum() # Ensure graph connectivity

        return (total_loss, outputs) if return_outputs else total_loss