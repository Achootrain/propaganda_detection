from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from TorchCRF import CRF

class TokenCRFModel(nn.Module):
    """RoBERTa encoder followed by a CRF for BIO tagging."""

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        aux_ce_weight: float = 0.0,
        label_smoothing: float = 0.0,
        class_weights: Optional[Tuple[float, ...]] = None,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.crf = CRF(num_labels, pad_idx=None, use_gpu=torch.cuda.is_available())
        self.num_labels = num_labels
        self.aux_ce_weight = aux_ce_weight
        self.label_smoothing = label_smoothing
        if class_weights is not None:
            if len(class_weights) != num_labels:
                raise ValueError("class_weights must match num_labels")
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float),
            )
        else:
            self.class_weights = None

        # Add BIO constraints to the CRF layer to prevent malformed sequences.
        # 0: O, 1: B, 2: I
        o_id, b_id, i_id = 0, 1, 2
        self.crf.start_trans.data[i_id] = -10000.0      # Can't start with I
        self.crf.trans_matrix.data[o_id, i_id] = -10000.0  # O -> I is forbidden
        self.crf.trans_matrix.data[b_id, b_id] = -10000.0  # B -> B is forbidden
        self.crf.trans_matrix.data[i_id, b_id] = -10000.0  # I -> B is forbidden

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs, # to ignore other arguments
    ) -> Dict[str, torch.Tensor]:
        """Run the encoder and CRF, returning loss, predictions, and confidence scores."""

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = self.dropout(outputs.last_hidden_state)
        emissions = self.classifier(sequence_output)
        mask = attention_mask.bool()
        result: Dict[str, torch.Tensor] = {}

        if labels is not None:
            label_mask = labels != -100
            effective_mask = mask & label_mask
            crf_labels = labels.masked_fill(~label_mask, 0)
            log_likelihood = self.crf(emissions, crf_labels, mask=effective_mask)
            token_counts = effective_mask.float().sum(dim=1).clamp(min=1.0)
            crf_loss = (-log_likelihood / token_counts).mean()
            loss = crf_loss
            if self.aux_ce_weight > 0:
                ce_loss = F.cross_entropy(
                    emissions.view(-1, self.num_labels),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                    label_smoothing=self.label_smoothing,
                    weight=self.class_weights,
                )
                loss = (1 - self.aux_ce_weight) * crf_loss + self.aux_ce_weight * ce_loss
            result["loss"] = loss
            decode_mask = effective_mask
        else:
            decode_mask = mask

        predictions = self.crf.viterbi_decode(emissions, mask=decode_mask)
        
        # Also return confidence scores (average log-probability of the predicted sequence)
        log_probs = F.log_softmax(emissions, dim=-1)
        confidence_scores = []
        for i, seq in enumerate(predictions):
            seq_len = int(decode_mask[i].sum())
            if seq_len == 0:
                confidence_scores.append(torch.tensor(0.0, device=emissions.device))
                continue
            
            # Gather the log-probabilities of the predicted tags for the actual sequence length
            seq_log_probs = log_probs[i, :seq_len].gather(1, torch.tensor(seq[:seq_len], device=log_probs.device).unsqueeze(1)).squeeze(1)
            
            # Calculate the average log-probability
            avg_log_prob = seq_log_probs.mean()
            confidence_scores.append(avg_log_prob)

        result["predictions"] = predictions
        result["confidence_scores"] = torch.stack(confidence_scores)
        return result