from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class RobertaLargeBaseline(nn.Module):
    """
    Baseline model for token classification:
    - Roberta-large encoder
    - Linear classification layer
    - Optional dropout
    """

    def __init__(
        self,
        model_name: str = "roberta-large",
        num_labels: int = 3,  # O, B, I
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
        class_weights: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
        self.roberta = AutoModel.from_pretrained(model_name, config=config)
        hidden_size = config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing

        if class_weights is not None:
            if len(class_weights) != num_labels:
                raise ValueError("class_weights must match num_labels")
            # register as buffer to move with .to(device)
            self.register_buffer("class_weights", torch.tensor(class_weights, dtype=torch.float))
        else:
            self.class_weights = None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **_: dict,
    ):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)
        logits = self.classifier(hidden_states)

        result = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.num_labels),
                labels.view(-1),
                ignore_index=-100,
                reduction="mean",
                label_smoothing=self.label_smoothing,
                weight=self.class_weights,
            )
            result["loss"] = loss

        predictions = torch.argmax(logits, dim=-1)
        result["predictions"] = predictions
        return result
