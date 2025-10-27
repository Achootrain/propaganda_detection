from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel
from TorchCRF import CRF


def focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Focal Loss - automatically focus on hard examples
    """
    num_classes = logits.size(-1)
    
    # Reshape
    logits_flat = logits.view(-1, num_classes)
    labels_flat = labels.view(-1)
    
    # Filter ignored labels
    valid_mask = (labels_flat != ignore_index)
    logits_valid = logits_flat[valid_mask]
    labels_valid = labels_flat[valid_mask]
    
    # Check empty
    if labels_valid.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    # Cross entropy loss
    ce_loss = F.cross_entropy(
        logits_valid,
        labels_valid,
        reduction="none",
    )
    
    # Compute p_t
    p = F.softmax(logits_valid, dim=-1)
    labels_one_hot = F.one_hot(labels_valid, num_classes=num_classes).float()
    p_t = (p * labels_one_hot).sum(dim=-1)
    
    # Focal weight
    focal_weight = (1 - p_t) ** gamma
    
    # Final loss
    loss = (alpha * focal_weight * ce_loss).mean()
    
    return loss


class MultiGranularityTokenCRFModel(nn.Module):
    """
    Multi-granularity model with:
    - Sentence-level gating
    - Token-level classification
    - CRF layer for structured prediction
    - Scheduled focal loss with warm-up
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        dropout: float = 0.1,
        sentence_loss_weight: float = 0.8,
        label_smoothing: float = 0.0,
        class_weights: Optional[Tuple[float, ...]] = None,
        gate_function: str = "sigmoid",
        apply_gate_in_inference: bool = True,
        # CRF parameters
        use_crf: bool = True,
        crf_loss_weight: float = 0.5,  # Balance between CRF and CE/Focal loss
        enforce_bio_constraints: bool = True,  # Add BIO constraints to CRF
        # Focal loss parameters
        use_focal_loss: bool = True,
        focal_alpha: float = 0.75,
        focal_gamma: float = 3.0,
        # Warm-up parameters
        focal_warmup_steps: int = 3,
        focal_warmup_method: str = "linear",
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)

        self.sentence_classifier = nn.Linear(hidden_size, 1)
        self.token_classifier = nn.Linear(hidden_size, num_labels)

        self.num_labels = num_labels
        self.sentence_loss_weight = sentence_loss_weight
        self.label_smoothing = label_smoothing
        self.gate_function = gate_function
        self.apply_gate_in_inference = apply_gate_in_inference

        # CRF layer
        self.use_crf = use_crf
        self.crf_loss_weight = crf_loss_weight
        if use_crf:
            self.crf = CRF(num_labels, pad_idx=None, use_gpu=torch.cuda.is_available())
            
            # Add BIO constraints (O=0, B=1, I=2)
            if enforce_bio_constraints and num_labels == 3:
                o_id, b_id, i_id = 0, 1, 2
                self.crf.start_trans.data[i_id] = -10000.0      # Can't start with I
                self.crf.trans_matrix.data[o_id, i_id] = -10000.0  # O -> I is forbidden
                self.crf.trans_matrix.data[b_id, b_id] = -10000.0  # B -> B is forbidden
                self.crf.trans_matrix.data[i_id, b_id] = -10000.0  # I -> B is forbidden

        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.focal_warmup_steps = focal_warmup_steps
        self.focal_warmup_method = focal_warmup_method
        
        self.register_buffer("_training_step", torch.tensor(0))

        if class_weights is not None:
            if len(class_weights) != num_labels:
                raise ValueError("class_weights must match num_labels")
            self.register_buffer(
                "class_weights",
                torch.tensor(class_weights, dtype=torch.float),
            )
        else:
            self.class_weights = None

    def _get_current_focal_gamma(self) -> float:
        """Compute current gamma based on warm-up schedule"""
        if not self.training or not self.use_focal_loss:
            return self.focal_gamma
        
        current_step = self._training_step.item()
        
        if self.focal_warmup_method == "step":
            if current_step < self.focal_warmup_steps:
                return 0.0
            else:
                return self.focal_gamma
        
        elif self.focal_warmup_method == "linear":
            if current_step < self.focal_warmup_steps:
                return self.focal_gamma * (current_step / self.focal_warmup_steps)
            else:
                return self.focal_gamma
        
        elif self.focal_warmup_method == "cosine":
            if current_step < self.focal_warmup_steps:
                progress = current_step / self.focal_warmup_steps
                return self.focal_gamma * (1 - math.cos(progress * math.pi)) / 2
            else:
                return self.focal_gamma
        
        return self.focal_gamma

    def _compute_sentence_gates(
        self,
        hidden_states: torch.Tensor,
        sentence_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Compute sentence-level gates"""
        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)

        gates = torch.ones(batch_size, seq_len, device=hidden_states.device)

        sentence_logits_list = []
        sentence_labels_list = []

        for i in range(batch_size):
            sample_hidden_states = hidden_states[i]
            sample_sentence_ids = sentence_ids[i]

            unique_sentence_ids = torch.unique(sample_sentence_ids[sample_sentence_ids != -100])
            if unique_sentence_ids.numel() == 0:
                continue

            # Compute sentence representations
            sentence_representations = []
            for sentence_id in unique_sentence_ids:
                token_indices = (sample_sentence_ids == sentence_id).nonzero(as_tuple=True)[0]
                sentence_representation = sample_hidden_states[token_indices].mean(dim=0)
                sentence_representations.append(sentence_representation)

            sentence_representations = torch.stack(sentence_representations)
            sentence_logits = self.sentence_classifier(sentence_representations).squeeze(-1)

            # Compute gates
            if self.gate_function == "relu":
                sentence_gates = torch.relu(sentence_logits)
            else:
                sentence_gates = torch.sigmoid(sentence_logits)

            # Assign gates to tokens
            for j, sentence_id in enumerate(unique_sentence_ids):
                token_indices = (sample_sentence_ids == sentence_id).nonzero(as_tuple=True)[0]
                gates[i, token_indices] = sentence_gates[j]

            # Collect for loss computation
            if labels is not None:
                sentence_logits_list.append(sentence_logits)

                sample_labels = labels[i]
                sentence_labels = []
                for sentence_id in unique_sentence_ids:
                    token_indices = (sample_sentence_ids == sentence_id).nonzero(as_tuple=True)[0]
                    sentence_label = (sample_labels[token_indices] > 0).any().float()
                    sentence_labels.append(sentence_label)
                sentence_labels = torch.stack(sentence_labels)
                sentence_labels_list.append(sentence_labels)

        sentence_logits_cat = torch.cat(sentence_logits_list) if sentence_logits_list else None
        sentence_labels_cat = torch.cat(sentence_labels_list) if sentence_labels_list else None

        return gates, sentence_logits_cat, sentence_labels_cat

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        sentence_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # Increment step counter
        if self.training and labels is not None:
            self._training_step += 1
        
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states[-1]
        hidden_states = self.dropout(hidden_states)

        result: Dict[str, torch.Tensor] = {}
        emissions = self.token_classifier(hidden_states)  # Token logits/emissions

        # Without sentence-level gating
        if sentence_ids is None:
            if labels is not None:
                # Prepare masks for CRF
                mask = attention_mask.bool()
                label_mask = labels != -100
                effective_mask = mask & label_mask
                
                # Initialize total loss
                total_loss = 0.0
                loss_components = {}
                
                # CRF loss
                if self.use_crf:
                    crf_labels = labels.masked_fill(~label_mask, 0)
                    log_likelihood = self.crf(emissions, crf_labels, mask=effective_mask)
                    token_counts = effective_mask.float().sum(dim=1).clamp(min=1.0)
                    crf_loss = (-log_likelihood / token_counts).mean()
                    total_loss += self.crf_loss_weight * crf_loss
                    loss_components["crf_loss"] = crf_loss.item()
                
                # CE or Focal loss
                current_gamma = self._get_current_focal_gamma()
                
                if self.use_focal_loss and current_gamma > 0:
                    token_loss = focal_loss(
                        emissions,
                        labels,
                        alpha=self.focal_alpha,
                        gamma=current_gamma,
                        ignore_index=-100,
                    )
                else:
                    token_loss = F.cross_entropy(
                        emissions.view(-1, self.num_labels),
                        labels.view(-1),
                        ignore_index=-100,
                        reduction="mean",
                        weight=self.class_weights,
                    )
                
                if self.use_crf:
                    total_loss += (1 - self.crf_loss_weight) * token_loss
                else:
                    total_loss = token_loss
                
                loss_components["token_loss"] = token_loss.item()
                loss_components["current_focal_gamma"] = current_gamma
                
                result["loss"] = total_loss
                result.update(loss_components)
            
            # Predictions
            if self.use_crf:
                mask = attention_mask.bool()
                if labels is not None:
                    decode_mask = effective_mask
                else:
                    decode_mask = mask
                predictions = self.crf.viterbi_decode(emissions, mask=decode_mask)
                result["predictions"] = predictions
            else:
                result["predictions"] = torch.argmax(emissions, dim=-1)
            
            return result

        # With sentence-level gating
        gates, sentence_logits_cat, sentence_labels_cat = self._compute_sentence_gates(
            hidden_states, sentence_ids, labels
        )

        result["gates"] = gates

        # Apply gating
        gates_expanded = gates.unsqueeze(-1)
        emissions_gated = emissions * gates_expanded

        if labels is not None and sentence_logits_cat is not None:
            # Sentence-level loss
            num_positives = sentence_labels_cat.sum()
            num_negatives = len(sentence_labels_cat) - num_positives
            pos_weight = num_negatives / num_positives if num_positives > 0 else torch.tensor(1.0, device=hidden_states.device)

            sentence_loss = F.binary_cross_entropy_with_logits(
                sentence_logits_cat,
                sentence_labels_cat,
                reduction="mean",
                pos_weight=pos_weight
            )

            # Token-level loss with CRF
            mask = attention_mask.bool()
            label_mask = labels != -100
            effective_mask = mask & label_mask
            
            total_token_loss = 0.0
            
            # CRF loss on gated emissions
            if self.use_crf:
                crf_labels = labels.masked_fill(~label_mask, 0)
                log_likelihood = self.crf(emissions_gated, crf_labels, mask=effective_mask)
                token_counts = effective_mask.float().sum(dim=1).clamp(min=1.0)
                crf_loss = (-log_likelihood / token_counts).mean()
                total_token_loss += self.crf_loss_weight * crf_loss
                result["crf_loss"] = crf_loss.item()
            
            # CE or Focal loss
            current_gamma = self._get_current_focal_gamma()
            
            if self.use_focal_loss and current_gamma > 0:
                token_loss = focal_loss(
                    emissions_gated,
                    labels,
                    alpha=self.focal_alpha,
                    gamma=current_gamma,
                    ignore_index=-100,
                )
            else:
                token_loss = F.cross_entropy(
                    emissions_gated.view(-1, self.num_labels),
                    labels.view(-1),
                    ignore_index=-100,
                    reduction="mean",
                    label_smoothing=self.label_smoothing,
                    weight=self.class_weights,
                )
            
            if self.use_crf:
                total_token_loss += (1 - self.crf_loss_weight) * token_loss
            else:
                total_token_loss = token_loss

            # Combined loss
            total_loss = self.sentence_loss_weight * sentence_loss + (1 - self.sentence_loss_weight) * total_token_loss
            
            result["loss"] = total_loss
            result["sentence_loss"] = sentence_loss.item()
            result["token_loss"] = token_loss.item()
            result["current_focal_gamma"] = current_gamma

        # Predictions
        if self.use_crf:
            mask = attention_mask.bool()
            if labels is not None:
                decode_mask = effective_mask
            else:
                decode_mask = mask
            
            # Use gated or non-gated emissions based on config
            if self.apply_gate_in_inference:
                predictions = self.crf.viterbi_decode(emissions_gated, mask=decode_mask)
            else:
                predictions = self.crf.viterbi_decode(emissions, mask=decode_mask)
            result["predictions"] = predictions
        else:
            if self.apply_gate_in_inference:
                result["predictions"] = torch.argmax(emissions_gated, dim=-1)
            else:
                result["predictions"] = torch.argmax(emissions, dim=-1)

        return result
