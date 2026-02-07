"""Custom RoBERTa model for sequence classification."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from .heads import (
    RobertaClassificationHead,
    RobertaClassificationHeadLength,
    RobertaClassificationHeadMatchings,
    RobertaClassificationHeadJoined,
    RobertaClassificationHeadJoinedLength,
    RobertaClassificationHeadJoinedLengthMatchings,
)


class CustomRobertaForSequenceClassification(nn.Module):
    def __init__(
        self,
        config: AutoConfig,
        use_length: bool = False,
        use_matchings: bool = False,
        join_embeddings: bool = False,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        # mirror modeling_roberta: flags are in config
        config.use_length = use_length
        config.use_matchings = use_matchings
        config.join_embeddings = join_embeddings
        # Create a backbone compatible with `config`.
        # We default to from_config here and let `main()` optionally overwrite
        # `self.roberta` with a pretrained / checkpoint-initialized instance.
        self.roberta = AutoModel.from_config(config)
        self.num_labels = config.num_labels
        hidden = config.hidden_size
        dropout_prob = getattr(config, 'hidden_dropout_prob', 0.1)

        if config.use_length and config.use_matchings and config.join_embeddings:
            self.classifier = RobertaClassificationHeadJoinedLengthMatchings(hidden, self.num_labels, dropout_prob)
        elif config.use_length:
            if config.join_embeddings:
                self.classifier = RobertaClassificationHeadJoinedLength(hidden, self.num_labels, dropout_prob)
            else:
                self.classifier = RobertaClassificationHeadLength(hidden, self.num_labels, dropout_prob)
        elif config.join_embeddings:
            self.classifier = RobertaClassificationHeadJoined(hidden, self.num_labels, dropout_prob)
        elif config.use_matchings:
            self.classifier = RobertaClassificationHeadMatchings(hidden, self.num_labels, dropout_prob)
        else:
            self.classifier = RobertaClassificationHead(hidden, self.num_labels, dropout_prob)

        # Optional class weights for imbalanced classes
        if class_weights is not None:
            # ensure on correct dtype; will be moved to DEVICE by caller or in forward
            self.class_weights = class_weights.float()
        else:
            self.class_weights = None
        self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        lengths=None,
        matchings=None,
        lexicon=None,
        labels=None,
    ):
        # Some backbones (model-dependent) do not accept token_type_ids.
        try:
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        except TypeError:
            outputs = self.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        sequence_output = outputs[0] if isinstance(outputs, (tuple, list)) else outputs.last_hidden_state
        logits = None
        # pass sequence_output and feature tensors into classifier similar to modeling_roberta
        if isinstance(self.classifier, RobertaClassificationHeadJoinedLengthMatchings):
            logits = self.classifier(sequence_output, sent_a_length=lengths, matchings=matchings, lexicon=lexicon, attention_mask=attention_mask)
        elif isinstance(self.classifier, RobertaClassificationHeadLength):
            logits = self.classifier(sequence_output, sent_a_length=lengths)
        elif isinstance(self.classifier, RobertaClassificationHeadMatchings):
            logits = self.classifier(sequence_output, matchings=matchings)
        elif isinstance(self.classifier, RobertaClassificationHeadJoinedLength):
            logits = self.classifier(sequence_output, sent_a_length=lengths, attention_mask=attention_mask)
        elif isinstance(self.classifier, RobertaClassificationHeadJoined):
            logits = self.classifier(sequence_output, attention_mask=attention_mask)
        else:
            logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # If weights exist but are not on the same device, move them
            if self.class_weights is not None and self.class_weights.device != logits.device:
                self.loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = self.loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return type('Output', (), {'loss': loss, 'logits': logits})
