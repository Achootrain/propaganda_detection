from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

from .crf import ConditionalRandomField, allowed_transitions


class ScalarMix(nn.Module):
    r"""Parameterised scalar mixture of N tensors.

    $mixture = \gamma * \sum(s_k * tensor_k)$ where $s = softmax(w)$.
    """

    def __init__(
        self,
        mixture_size: int,
        do_layer_norm: bool = False,
        initial_scalar_parameters: Optional[Sequence[float]] = None,
        trainable: bool = True,
    ) -> None:
        super().__init__()
        self.mixture_size = mixture_size
        self.do_layer_norm = do_layer_norm

        if initial_scalar_parameters is None:
            initial_scalar_parameters = [0.0] * mixture_size
        elif len(initial_scalar_parameters) != mixture_size:
            raise ValueError(
                f"Length of initial_scalar_parameters {len(initial_scalar_parameters)} differs from mixture_size {mixture_size}"
            )

        self.scalar_parameters = nn.Parameter(
            torch.tensor(initial_scalar_parameters, dtype=torch.float32),
            requires_grad=trainable,
        )
        self.gamma = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=trainable)

        if self.do_layer_norm:
            # torch.nn.LayerNorm requires normalized_shape; this script historically didn't use it.
            pass

    def forward(self, tensors: Sequence[torch.Tensor], mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        if len(tensors) != self.mixture_size:
            raise ValueError(f"ScalarMix expects {self.mixture_size} tensors but got {len(tensors)}")

        normed_weights = torch.softmax(self.scalar_parameters, dim=0)
        mixture = torch.zeros_like(tensors[0])
        for i, tensor in enumerate(tensors):
            mixture = mixture + normed_weights[i] * tensor
        return self.gamma * mixture


class SentenceAuxiliaryTask(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_plm_layers: int,
        pos_embedding_dim: int = 0,
        ner_embedding_dim: int = 0,
        dep_embedding_dim: int = 0,
        lexicon_feature_dim: int = 0,
        rnn_layers: int = 1,
        rnn_hidden_size: int = 256,
        rnn_dropout: float = 0.1,
        output_dropout: float = 0.1,
        structural_feature_dim: int = 4,
    ) -> None:
        super().__init__()

        self.scalar_mix_sent = ScalarMix(num_plm_layers)
        self.embedding_dim = (
            hidden_size + pos_embedding_dim + ner_embedding_dim + dep_embedding_dim + lexicon_feature_dim
        )

        self.rnn_hidden_size = rnn_hidden_size if rnn_hidden_size > 0 else self.embedding_dim // 2
        self.sent_lstm = nn.LSTM(
            self.embedding_dim,
            self.rnn_hidden_size,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=True,
            dropout=rnn_dropout if rnn_layers > 1 else 0.0,
        )

        lstm_output_dim = self.rnn_hidden_size * 2
        self.structural_feature_dim = structural_feature_dim

        ffn_input_dim = lstm_output_dim + structural_feature_dim
        self.sent_ffn = nn.Sequential(
            nn.Linear(ffn_input_dim, lstm_output_dim),
            nn.ReLU(),
            nn.Dropout(output_dropout),
        )

        self.sent_classifier = nn.Linear(lstm_output_dim, 1)
        self.dropout = nn.Dropout(output_dropout)

    def forward(
        self,
        plm_hidden_states: Tuple[torch.Tensor, ...],
        pos_embeddings: Optional[torch.Tensor] = None,
        ner_embeddings: Optional[torch.Tensor] = None,
        dep_embeddings: Optional[torch.Tensor] = None,
        lexicon_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        pad_token_label_id: int = -100,
        o_label_index: int = 0,
        sentence_lengths: Optional[torch.Tensor] = None,
        sentence_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B = plm_hidden_states[0].shape[0]
        T = plm_hidden_states[0].shape[1]
        device = plm_hidden_states[0].device

        if attention_mask is None:
            attention_mask = torch.ones(B, T, device=device, dtype=torch.long)

        sent_plm_output = self.scalar_mix_sent(plm_hidden_states)
        sent_plm_output = self.dropout(sent_plm_output)

        features_to_concat = [sent_plm_output]
        if pos_embeddings is not None:
            features_to_concat.append(pos_embeddings)
        if ner_embeddings is not None:
            features_to_concat.append(ner_embeddings)
        if dep_embeddings is not None:
            features_to_concat.append(dep_embeddings)
        if lexicon_features is not None:
            features_to_concat.append(lexicon_features)

        sent_input = torch.cat(features_to_concat, dim=-1) if len(features_to_concat) > 1 else sent_plm_output

        sent_lstm_out, _ = self.sent_lstm(sent_input)
        bos_hidden = sent_lstm_out[:, 0, :]

        if sentence_lengths is not None:
            norm_lengths = sentence_lengths.float().unsqueeze(-1) / 512.0
        else:
            norm_lengths = attention_mask.sum(dim=1, keepdim=True).float() / 512.0

        if sentence_positions is not None:
            pos_features = sentence_positions.float()
        else:
            pos_features = torch.zeros(B, 3, device=device)
            pos_features[:, 1] = 1.0

        structural_features = torch.cat([norm_lengths, pos_features], dim=-1)
        combined = torch.cat([bos_hidden, structural_features], dim=-1)

        ffn_out = self.sent_ffn(combined)
        sentence_logits = self.sent_classifier(ffn_out).squeeze(-1)
        sentence_probs = torch.sigmoid(sentence_logits)

        sentence_labels = None
        if labels is not None:
            valid_mask = labels != pad_token_label_id
            non_o_mask = (labels != o_label_index) & valid_mask
            sentence_labels = non_o_mask.any(dim=1).float()

        if sentence_labels is not None:
            hard_gates = sentence_labels
        else:
            hard_gates = (sentence_probs > 0.5).float()

        gates = hard_gates.unsqueeze(1).expand(B, T)

        return {
            "sentence_logits": sentence_logits,
            "sentence_probs": sentence_probs,
            "sentence_labels": sentence_labels,
            "gates": gates,
            "hard_gates": hard_gates,
        }


class TechniqueAuxiliaryTask(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        num_techniques: int,
        ffn_hidden_dim: int,
        output_dropout: float,
    ) -> None:
        super().__init__()
        if num_techniques <= 0:
            raise ValueError("num_techniques must be > 0")

        hidden = ffn_hidden_dim if ffn_hidden_dim > 0 else input_dim
        self.num_techniques = int(num_techniques)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(output_dropout),
            nn.Linear(hidden, self.num_techniques),
        )

    def forward(
        self,
        token_features: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        logits = self.ffn(token_features)
        out: Dict[str, torch.Tensor] = {"tech_logits": logits}

        if targets is None:
            out["tech_loss"] = torch.tensor(0.0, device=logits.device)
            return out

        target = targets.float()
        if mask is None:
            mask = torch.ones(target.shape[:2], device=logits.device, dtype=torch.float)
        else:
            mask = mask.float()

        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        token_bce = bce.mean(dim=-1)
        masked = token_bce * mask
        denom = mask.sum().clamp_min(1.0)
        out["tech_loss"] = masked.sum() / denom
        return out


class BertLstmCrf(nn.Module):
    def __init__(
        self,
        bert_model: AutoModel,
        label_list: Sequence[str],
        pad_token_label_id: int,
        rnn_layers: int = 0,
        rnn_dropout: float = 0.1,
        output_dropout: float = 0.1,
        gate_enabled: bool = False,
        gate_function: str = "sigmoid",
        sentence_aux_weight: float = 1.0,
        token_loss_weight: float = 1.0,
        pos_vocab_size: int = 0,
        ner_vocab_size: int = 0,
        dep_vocab_size: int = 0,
        pos_embedding_dim: int = 50,
        ner_embedding_dim: int = 50,
        dep_embedding_dim: int = 50,
        lexicon_feature_dim: int = 0,
        use_scalar_mix: bool = True,
        rnn_hidden_size: int = 0,
        ffn_hidden_dim: int = 0,
        num_technique_classes: int = 0,
        tech_loss_weight: float = 0.5,
        use_mean_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.bert_encoder = bert_model
        print(f"Rnn layers: {rnn_layers}")
        self.label_list = list(label_list)
        self.num_labels = len(label_list)
        self.pad_token_label_id = pad_token_label_id
        self.hidden_size = bert_model.config.hidden_size
        self.gate_enabled = gate_enabled

        self.sentence_aux_weight = float(sentence_aux_weight)
        self.token_loss_weight = float(token_loss_weight)
        self.last_sentence_loss = 0.0

        self.use_scalar_mix = use_scalar_mix
        self.dropout = nn.Dropout(output_dropout)

        self.pos_vocab_size = pos_vocab_size
        self.ner_vocab_size = ner_vocab_size
        self.dep_vocab_size = dep_vocab_size
        self.lexicon_feature_dim = lexicon_feature_dim

        if pos_vocab_size > 0:
            self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_dim, padding_idx=0)
        else:
            self.pos_embedding = None
            pos_embedding_dim = 0

        if ner_vocab_size > 0:
            self.ner_embedding = nn.Embedding(ner_vocab_size, ner_embedding_dim, padding_idx=0)
        else:
            self.ner_embedding = None
            ner_embedding_dim = 0

        if dep_vocab_size > 0:
            self.dep_embedding = nn.Embedding(dep_vocab_size, dep_embedding_dim, padding_idx=0)
        else:
            self.dep_embedding = None
            dep_embedding_dim = 0

        self._pos_embedding_dim = pos_embedding_dim
        self._ner_embedding_dim = ner_embedding_dim
        self._dep_embedding_dim = dep_embedding_dim

        num_plm_layers = getattr(bert_model.config, "num_hidden_layers", 12) + 1
        self.num_plm_layers = num_plm_layers

        if gate_enabled:
            self.sentence_aux = SentenceAuxiliaryTask(
                hidden_size=self.hidden_size,
                num_plm_layers=num_plm_layers,
                pos_embedding_dim=pos_embedding_dim,
                ner_embedding_dim=ner_embedding_dim,
                dep_embedding_dim=dep_embedding_dim,
                lexicon_feature_dim=self.lexicon_feature_dim,
                rnn_layers=1,
                rnn_hidden_size=256,
                rnn_dropout=rnn_dropout,
                output_dropout=output_dropout,
                structural_feature_dim=4,
            )
        else:
            self.sentence_aux = None

        if self.use_scalar_mix:
            self.scalar_mix = ScalarMix(num_plm_layers)
        else:
            self.scalar_mix = None

        self.o_label_index = int(self.label_list.index("O")) if "O" in self.label_list else 0
        self.non_o_indices = [i for i, l in enumerate(self.label_list) if l != "O"]

        self.rnn_layers = rnn_layers
        self.embedding_dim = (
            self.hidden_size + pos_embedding_dim + ner_embedding_dim + dep_embedding_dim + self.lexicon_feature_dim
        )

        if rnn_hidden_size > 0:
            self.hidden_dim = rnn_hidden_size
        else:
            self.hidden_dim = max(1, self.embedding_dim // 2)

        if rnn_layers > 0:
            self.lstm = nn.LSTM(
                self.embedding_dim,
                self.hidden_dim,
                num_layers=rnn_layers,
                bidirectional=True,
                batch_first=True,
                dropout=rnn_dropout if rnn_layers > 1 else 0.0,
            )
            classifier_input = self.hidden_dim * 2
        else:
            self.lstm = None
            classifier_input = self.embedding_dim

        effective_ffn_dim = ffn_hidden_dim if ffn_hidden_dim > 0 else classifier_input
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, effective_ffn_dim),
            nn.ReLU(),
            nn.Dropout(output_dropout),
            nn.Linear(effective_ffn_dim, self.num_labels),
        )

        crf_labels = {
            idx: ("O" if label == "O" else label.split("-", 1)[0])
            for idx, label in enumerate(self.label_list)
        }
        constraints = allowed_transitions("BIO", crf_labels)
        self.crf = ConditionalRandomField(self.num_labels, constraints, include_start_end_transitions=True)

        self.use_mean_pooling = use_mean_pooling

        self.num_technique_classes = int(num_technique_classes)
        self.tech_loss_weight = float(tech_loss_weight)
        self.last_tech_loss = 0.0

        if self.num_technique_classes > 0:
            self.tech_aux = TechniqueAuxiliaryTask(
                input_dim=self.embedding_dim,
                num_techniques=self.num_technique_classes,
                ffn_hidden_dim=effective_ffn_dim,
                output_dropout=output_dropout,
            )
        else:
            self.tech_aux = None

    def mean_pool_subtokens(
        self,
        subtoken_embeddings: torch.Tensor,
        word_ids: torch.Tensor,
        max_words: Optional[int] = None,
    ) -> torch.Tensor:
        B, seq_len, hidden_dim = subtoken_embeddings.shape
        device = subtoken_embeddings.device

        if max_words is None:
            max_words = int((word_ids.max() + 1).item())

        if max_words == 0:
            return subtoken_embeddings

        word_embeddings = torch.zeros(B, max_words, hidden_dim, device=device)
        word_counts = torch.zeros(B, max_words, device=device)

        for b in range(B):
            for s in range(seq_len):
                w = word_ids[b, s].item()
                if w >= 0 and w < max_words:
                    word_embeddings[b, w] += subtoken_embeddings[b, s]
                    word_counts[b, w] += 1

        word_counts = word_counts.clamp_min(1.0)
        word_embeddings = word_embeddings / word_counts.unsqueeze(-1)
        return word_embeddings

    def _mean_pool_feature_tensor(self, feature_tensor: torch.Tensor, word_ids: torch.Tensor) -> torch.Tensor:
        if feature_tensor is None:
            return None
        B, seq_len, dim = feature_tensor.shape
        device = feature_tensor.device
        max_words = int((word_ids.max() + 1).item()) if word_ids.numel() > 0 else 0
        if max_words == 0:
            return feature_tensor
        pooled = torch.zeros(B, max_words, dim, device=device)
        counts = torch.zeros(B, max_words, device=device)
        for b in range(B):
            for s in range(seq_len):
                w = word_ids[b, s].item()
                if w >= 0 and w < max_words:
                    pooled[b, w] += feature_tensor[b, s]
                    counts[b, w] += 1
        counts = counts.clamp_min(1.0)
        pooled = pooled / counts.unsqueeze(-1)
        return pooled

    def forward(
        self,
        input_ids,
        attention_mask,
        pos_ids=None,
        ner_ids=None,
        labels=None,
        sentence_lengths=None,
        sentence_positions=None,
        word_ids=None,
        tech_multi_labels=None,
        tech_label_mask=None,
        dep_ids=None,
        lexicon_features=None,
    ):
        outputs = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else outputs[2]

        if self.use_scalar_mix:
            sequence_output = self.scalar_mix(hidden_states)
        else:
            sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        pos_embed_sentence = (
            self.pos_embedding(pos_ids) if (self.pos_embedding is not None and pos_ids is not None) else None
        )
        ner_embed_sentence = (
            self.ner_embedding(ner_ids) if (self.ner_embedding is not None and ner_ids is not None) else None
        )
        dep_embed_sentence = (
            self.dep_embedding(dep_ids) if (self.dep_embedding is not None and dep_ids is not None) else None
        )

        lex_sentence = None
        if self.lexicon_feature_dim > 0:
            if lexicon_features is None:
                B, L, _ = sequence_output.shape
                lex_sentence = torch.zeros(B, L, self.lexicon_feature_dim, device=sequence_output.device)
            else:
                lex_sentence = lexicon_features

        if self.use_mean_pooling and word_ids is not None:
            pooled_output = self.mean_pool_subtokens(sequence_output, word_ids)
            sequence_output = pooled_output

            if self.pos_embedding is not None and pos_ids is not None:
                word_pos = self._get_word_level_features(pos_ids, word_ids)
                pos_embed = self.pos_embedding(word_pos)
                sequence_output = torch.cat([sequence_output, pos_embed], dim=-1)

            if self.ner_embedding is not None and ner_ids is not None:
                word_ner = self._get_word_level_features(ner_ids, word_ids)
                ner_embed = self.ner_embedding(word_ner)
                sequence_output = torch.cat([sequence_output, ner_embed], dim=-1)

            if self.dep_embedding is not None:
                if dep_ids is not None:
                    word_dep = self._get_word_level_features(dep_ids, word_ids)
                else:
                    B, W, _ = sequence_output.shape
                    word_dep = torch.zeros(B, W, dtype=torch.long, device=sequence_output.device)
                dep_embed = self.dep_embedding(word_dep)
                sequence_output = torch.cat([sequence_output, dep_embed], dim=-1)

            if self.lexicon_feature_dim > 0:
                if lexicon_features is None:
                    B, W, _ = sequence_output.shape
                    lexicon_features = torch.zeros(B, W, self.lexicon_feature_dim, device=sequence_output.device)
                else:
                    lexicon_features = self._mean_pool_feature_tensor(lexicon_features, word_ids)
                sequence_output = torch.cat([sequence_output, lexicon_features], dim=-1)
        else:
            features_to_concat = [sequence_output]

            if self.pos_embedding is not None and pos_ids is not None:
                features_to_concat.append(self.pos_embedding(pos_ids))

            if self.ner_embedding is not None and ner_ids is not None:
                features_to_concat.append(self.ner_embedding(ner_ids))

            if self.dep_embedding is not None:
                if dep_ids is not None:
                    dep_embed = self.dep_embedding(dep_ids)
                else:
                    B, L, _ = sequence_output.shape
                    dep_ids_zeros = torch.zeros(B, L, dtype=torch.long, device=sequence_output.device)
                    dep_embed = self.dep_embedding(dep_ids_zeros)
                features_to_concat.append(dep_embed)

            if self.lexicon_feature_dim > 0:
                if lexicon_features is None:
                    B, L, _ = sequence_output.shape
                    lexicon_features = torch.zeros(B, L, self.lexicon_feature_dim, device=sequence_output.device)
                features_to_concat.append(lexicon_features)

            if len(features_to_concat) > 1:
                sequence_output = torch.cat(features_to_concat, dim=-1)

        tech_loss = torch.tensor(0.0, device=sequence_output.device)
        if self.tech_aux is not None and tech_multi_labels is not None:
            aux_targets = tech_multi_labels
            aux_mask = tech_label_mask

            if self.use_mean_pooling and word_ids is not None:
                B = word_ids.size(0)
                max_words = int((word_ids.max() + 1).item()) if word_ids.numel() > 0 else 0
                if max_words > 0:
                    word_targets = torch.zeros(B, max_words, self.num_technique_classes, device=sequence_output.device)
                    word_mask = torch.zeros(B, max_words, device=sequence_output.device)
                    for b in range(B):
                        seen_words = set()
                        for s in range(word_ids.size(1)):
                            w = int(word_ids[b, s].item())
                            if w >= 0 and w < max_words and w not in seen_words:
                                seen_words.add(w)
                                word_targets[b, w] = aux_targets[b, s]
                                if aux_mask is not None:
                                    word_mask[b, w] = aux_mask[b, s]
                                else:
                                    word_mask[b, w] = 1.0
                    aux_targets = word_targets
                    aux_mask = word_mask

            tech_out = self.tech_aux(sequence_output, targets=aux_targets, mask=aux_mask)
            tech_loss = tech_out["tech_loss"]
            self.last_tech_loss = float(tech_loss.detach().item())

        if self.lstm is not None:
            sequence_output, _ = self.lstm(sequence_output)

        logits = self.classifier(sequence_output)

        if self.sentence_aux is None:
            if self.use_mean_pooling and word_ids is not None:
                word_labels = self._get_word_level_features(labels, word_ids)
                max_words = logits.size(1)
                word_mask = torch.zeros(labels.size(0), max_words, device=labels.device, dtype=attention_mask.dtype)
                for b in range(labels.size(0)):
                    for w in range(max_words):
                        if (word_ids[b] == w).any():
                            word_mask[b, w] = 1
                base_loss, _, predicted_tags = self._decode_without_gating(logits, word_labels, word_mask)
            else:
                base_loss, _, predicted_tags = self._decode_without_gating(logits, labels, attention_mask)

            total_loss = base_loss + self.tech_loss_weight * tech_loss
            return total_loss, logits, predicted_tags

        aux_out = self.sentence_aux(
            plm_hidden_states=hidden_states,
            pos_embeddings=pos_embed_sentence,
            ner_embeddings=ner_embed_sentence,
            dep_embeddings=dep_embed_sentence,
            lexicon_features=lex_sentence,
            attention_mask=attention_mask,
            labels=labels,
            pad_token_label_id=self.pad_token_label_id,
            o_label_index=self.o_label_index,
            sentence_lengths=sentence_lengths,
            sentence_positions=sentence_positions,
        )

        if labels is None:
            raise ValueError("labels tensor is required to align sub-tokens in this condensed pipeline")

        clear_logits, clear_labels, clear_mask = self._clear_subtokens(logits, labels, attention_mask)

        best_paths = self.crf.viterbi_tags(clear_logits, clear_mask.long(), top_k=1)
        predicted_tags = [path_scores[0][0] for path_scores in best_paths]

        sentence_loss = torch.tensor(0.0, device=logits.device)
        if aux_out["sentence_labels"] is not None:
            bce = nn.BCEWithLogitsLoss()
            sentence_loss = bce(aux_out["sentence_logits"], aux_out["sentence_labels"])

        hard_gates = aux_out["hard_gates"]

        batch_size = clear_logits.size(0)
        token_losses = []

        for i in range(batch_size):
            sample_logits = clear_logits[i : i + 1]
            sample_labels = clear_labels[i : i + 1]
            sample_mask = clear_mask[i : i + 1]

            if sample_mask.sum() == 0:
                token_losses.append(torch.tensor(0.0, device=logits.device))
                continue

            sample_ll = self.crf(sample_logits, sample_labels, sample_mask.long())
            token_losses.append(-sample_ll)

        token_losses = torch.stack(token_losses)

        gated_token_losses = token_losses * hard_gates
        num_prop_sentences = hard_gates.sum().clamp_min(1.0)
        mean_token_loss = gated_token_losses.sum() / num_prop_sentences

        has_propaganda = hard_gates.sum() > 0
        gated_tech_loss = tech_loss if has_propaganda else torch.tensor(0.0, device=logits.device)

        loss = (
            self.sentence_aux_weight * sentence_loss
            + self.token_loss_weight * mean_token_loss
            + self.tech_loss_weight * gated_tech_loss
        )

        self.last_sentence_loss = float(sentence_loss.detach().item())

        return loss, logits, predicted_tags

    def _get_word_level_features(self, feature_ids: torch.Tensor, word_ids: torch.Tensor) -> torch.Tensor:
        B, seq_len = feature_ids.shape
        device = feature_ids.device
        max_words = int((word_ids.max() + 1).item())

        if max_words == 0:
            return feature_ids

        word_features = torch.zeros(B, max_words, dtype=feature_ids.dtype, device=device)

        for b in range(B):
            seen_words = set()
            for s in range(seq_len):
                w = word_ids[b, s].item()
                if w >= 0 and w < max_words and w not in seen_words:
                    word_features[b, w] = feature_ids[b, s]
                    seen_words.add(w)

        return word_features

    def _decode_without_gating(self, logits, labels, attention_mask):
        if labels is None:
            raise ValueError("labels tensor is required to align sub-tokens in this condensed pipeline")

        clear_logits, clear_labels, clear_mask = self._clear_subtokens(logits, labels, attention_mask)
        best_paths = self.crf.viterbi_tags(clear_logits, clear_mask.long(), top_k=1)
        predicted_tags = [p[0][0] for p in best_paths]

        log_likelihood = self.crf(clear_logits, clear_labels, clear_mask.long())
        loss = -log_likelihood

        return loss, logits, predicted_tags

    def _clear_subtokens(self, logits, labels, mask):
        batch_size, seq_length, num_labels = logits.size()
        clear_logits = logits.new_zeros((batch_size, seq_length, num_labels))
        clear_labels = labels.new_full((batch_size, seq_length), fill_value=self.pad_token_label_id)
        clear_mask = mask.new_zeros((batch_size, seq_length))

        for row in range(batch_size):
            valid_positions = labels[row] != self.pad_token_label_id
            trimmed_logits = logits[row][valid_positions]
            trimmed_labels = labels[row][valid_positions]
            length = trimmed_labels.size(0)
            if length == 0:
                continue
            clear_logits[row, :length] = trimmed_logits
            clear_labels[row, :length] = trimmed_labels
            clear_mask[row, :length] = 1

        clear_labels = clear_labels.masked_fill(clear_mask == 0, 0)
        return clear_logits, clear_labels, clear_mask
