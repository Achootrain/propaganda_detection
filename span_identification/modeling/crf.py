from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


VITERBI_DECODING = Tuple[List[int], float]


def logsumexp(tensor: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    max_score, _ = tensor.max(dim, keepdim=keepdim)
    if keepdim:
        stable_vec = tensor - max_score
    else:
        stable_vec = tensor - max_score.unsqueeze(dim)
    return max_score + (stable_vec.exp().sum(dim, keepdim=keepdim)).log()


class ConditionalRandomField(torch.nn.Module):
    def __init__(
        self,
        num_tags: int,
        constraints: Optional[List[Tuple[int, int]]] = None,
        include_start_end_transitions: bool = True,
    ) -> None:
        super().__init__()
        self.num_tags = num_tags
        self.transitions = torch.nn.Parameter(torch.empty(num_tags, num_tags))
        if constraints is None:
            constraint_mask = torch.ones(num_tags + 2, num_tags + 2)
        else:
            constraint_mask = torch.zeros(num_tags + 2, num_tags + 2)
            for i, j in constraints:
                constraint_mask[i, j] = 1.0
        self._constraint_mask = torch.nn.Parameter(constraint_mask, requires_grad=False)
        self.include_start_end_transitions = include_start_end_transitions
        if include_start_end_transitions:
            self.start_transitions = torch.nn.Parameter(torch.empty(num_tags))
            self.end_transitions = torch.nn.Parameter(torch.empty(num_tags))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.xavier_normal_(self.transitions)
        if self.include_start_end_transitions:
            torch.nn.init.normal_(self.start_transitions)
            torch.nn.init.normal_(self.end_transitions)

    def _input_likelihood(self, logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, num_tags = logits.size()
        mask = mask.float().transpose(0, 1).contiguous()
        logits = logits.transpose(0, 1).contiguous()
        if self.include_start_end_transitions:
            alpha = self.start_transitions.view(1, num_tags) + logits[0]
        else:
            alpha = logits[0]
        for index in range(1, sequence_length):
            emit_scores = logits[index].view(batch_size, 1, num_tags)
            transition_scores = self.transitions.view(1, num_tags, num_tags)
            broadcast_alpha = alpha.view(batch_size, num_tags, 1)
            inner = broadcast_alpha + emit_scores + transition_scores
            alpha = logsumexp(inner, 1) * mask[index].view(batch_size, 1) + alpha * (
                1 - mask[index]
            ).view(batch_size, 1)
        if self.include_start_end_transitions:
            stops = alpha + self.end_transitions.view(1, num_tags)
        else:
            stops = alpha
        return logsumexp(stops)

    def _joint_likelihood(self, logits: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = logits.data.shape
        logits = logits.transpose(0, 1).contiguous()
        mask = mask.float().transpose(0, 1).contiguous()
        tags_ = tags.transpose(0, 1).contiguous()
        if self.include_start_end_transitions:
            score = self.start_transitions.index_select(0, tags_[0])
        else:
            score = 0.0
        for i in range(sequence_length - 1):
            current_tag, next_tag = tags_[i], tags_[i + 1]
            transition_score = self.transitions[current_tag.view(-1), next_tag.view(-1)]
            emit_score = logits[i].gather(1, current_tag.view(batch_size, 1)).squeeze(1)
            score = score + transition_score * mask[i + 1] + emit_score * mask[i]
        last_tag_index = mask.sum(0).long() - 1
        last_tags = tags_.gather(0, last_tag_index.view(1, batch_size)).squeeze(0)
        if self.include_start_end_transitions:
            last_transition_score = self.end_transitions.index_select(0, last_tags)
        else:
            last_transition_score = 0.0
        last_inputs = logits[-1]
        last_input_score = last_inputs.gather(1, last_tags.view(-1, 1)).squeeze()
        score = score + last_transition_score + last_input_score * mask[-1]
        return score

    def forward(self, inputs: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.long)
        log_denominator = self._input_likelihood(inputs, mask)
        log_numerator = self._joint_likelihood(inputs, tags, mask)
        return torch.sum(log_numerator - log_denominator)

    def viterbi_tags(self, logits: torch.Tensor, mask: torch.Tensor, top_k: int = 1) -> List[List[VITERBI_DECODING]]:
        if top_k is None or top_k < 1:
            raise ValueError("top_k must be >= 1")
        flatten_output = top_k == 1
        _, max_seq_length, num_tags = logits.size()
        logits, mask = logits.data, mask.data
        start_tag = num_tags
        end_tag = num_tags + 1
        transitions = torch.full((num_tags + 2, num_tags + 2), -10000.0)
        constrained_transitions = self.transitions * self._constraint_mask[:num_tags, :num_tags] + -10000.0 * (
            1 - self._constraint_mask[:num_tags, :num_tags]
        )
        transitions[:num_tags, :num_tags] = constrained_transitions.data
        if self.include_start_end_transitions:
            transitions[start_tag, :num_tags] = self.start_transitions * self._constraint_mask[start_tag, :num_tags]
            transitions[:num_tags, end_tag] = self.end_transitions * self._constraint_mask[:num_tags, end_tag]
        else:
            transitions[start_tag, :num_tags] = -10000.0 * (1 - self._constraint_mask[start_tag, :num_tags])
            transitions[:num_tags, end_tag] = -10000.0 * (1 - self._constraint_mask[:num_tags, end_tag])

        best_paths: List[List[VITERBI_DECODING]] = []
        tag_sequence = torch.zeros(max_seq_length + 2, num_tags + 2)
        for prediction, prediction_mask in zip(logits, mask):
            sequence_length = int(torch.sum(prediction_mask))
            tag_sequence.fill_(-10000.0)
            tag_sequence[0, start_tag] = 0.0
            tag_sequence[1 : sequence_length + 1, :num_tags] = prediction[:sequence_length]
            tag_sequence[sequence_length + 1, end_tag] = 0.0
            paths, scores = viterbi_decode(tag_sequence[: sequence_length + 2], transitions, top_k=top_k)
            cleaned_paths: List[VITERBI_DECODING] = []
            for path, score in zip(paths, scores):
                cleaned_paths.append((path[1:-1], float(score.item())))
            best_paths.append(cleaned_paths)
        if flatten_output:
            return [[paths[0]] for paths in best_paths]
        return best_paths


def is_transition_allowed(constraint_type: str, from_tag: str, from_entity: str, to_tag: str, to_entity: str) -> bool:
    if to_tag == "START" or from_tag == "END":
        return False
    if constraint_type == "BIO":
        if from_tag == "START":
            return to_tag in ("O", "B")
        if to_tag == "END":
            return from_tag in ("O", "B", "I")
        if to_tag in ("O", "B"):
            return True
        return to_tag == "I" and from_tag in ("B", "I") and from_entity == to_entity
    raise ValueError(f"Unsupported constraint type: {constraint_type}")


def allowed_transitions(constraint_type: str, labels: Dict[int, str]) -> List[Tuple[int, int]]:
    num_labels = len(labels)
    start_tag = num_labels
    end_tag = num_labels + 1
    labels_with_boundaries = list(labels.items()) + [(start_tag, "START"), (end_tag, "END")]
    allowed: List[Tuple[int, int]] = []
    for from_label_index, from_label in labels_with_boundaries:
        from_tag = from_label if from_label in ("START", "END") else from_label[0]
        from_entity = "" if from_label in ("START", "END") else from_label[1:]
        for to_label_index, to_label in labels_with_boundaries:
            to_tag = to_label if to_label in ("START", "END") else to_label[0]
            to_entity = "" if to_label in ("START", "END") else to_label[1:]
            if is_transition_allowed(constraint_type, from_tag, from_entity, to_tag, to_entity):
                allowed.append((from_label_index, to_label_index))
    return allowed


def viterbi_decode(
    tag_sequence: torch.Tensor,
    transition_matrix: torch.Tensor,
    top_k: int = 1,
) -> Tuple[List[List[int]], torch.Tensor]:
    sequence_length, num_tags = tag_sequence.size()
    path_scores: List[torch.Tensor] = []
    path_indices: List[torch.Tensor] = []
    path_scores.append(tag_sequence[0].unsqueeze(0))
    for timestep in range(1, sequence_length):
        summed_potentials = path_scores[timestep - 1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)
        scores, paths = torch.topk(summed_potentials, k=min(summed_potentials.size(0), top_k), dim=0)
        path_scores.append(tag_sequence[timestep].unsqueeze(0) + scores)
        path_indices.append(paths.squeeze())
    path_scores_v = path_scores[-1].view(-1)
    max_k = min(path_scores_v.size(0), top_k)
    viterbi_scores, best_paths = torch.topk(path_scores_v, k=max_k, dim=0)
    viterbi_paths: List[List[int]] = []
    for i in range(max_k):
        viterbi_path = [int(best_paths[i])]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        viterbi_path.reverse()
        viterbi_paths.append(viterbi_path)
    return viterbi_paths, viterbi_scores
