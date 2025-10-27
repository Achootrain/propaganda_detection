from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoTokenizer

from ..process_data.utils import _collect_spans, LABEL_BEGIN, LABEL_INSIDE


def _span_overlap(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> int:
    start = max(span_a[0], span_b[0])
    end = min(span_a[1], span_b[1])
    return max(0, end - start)


def calculate_span_metrics(
    predictions,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """Compute cumulative precision/recall numerators and counts for span-level metrics.

    predictions: list[list[int]] from CRF.viterbi_decode or tensor [B, T]
    labels: tensor [B, T]
    mask: tensor [B, T] (True for valid label positions)
    """
    # Normalize predictions to list of lists
    if isinstance(predictions, torch.Tensor):
        preds_list = [pred[: int(m.sum().item())].tolist() for pred, m in zip(predictions, mask)]
    else:
        preds_list = [list(map(int, seq)) for seq in predictions]

    labels_list = [lab[m].tolist() for lab, m in zip(labels, mask)]

    cumulative_prec = 0.0
    cumulative_rec = 0.0
    pred_span_count = 0
    gold_span_count = 0
    pred_malformed = 0
    gold_malformed = 0

    def spans_from_labels(seq: Sequence[int]) -> List[Tuple[int, int]]:
        return _collect_spans(seq)

    def malformed_count(seq: Sequence[int]) -> int:
        # Count tokens labeled I that are not inside a span started by B
        inside = False
        count = 0
        for tag in seq:
            if tag == LABEL_BEGIN:  # B
                inside = True
            elif tag == LABEL_INSIDE:  # I
                if not inside:
                    count += 1
            else:  # O
                inside = False
        return count

    for pred_seq, gold_seq in zip(preds_list, labels_list):
        pred_spans = spans_from_labels(pred_seq)
        gold_spans = spans_from_labels(gold_seq)

        pred_span_count += len(pred_spans)
        gold_span_count += len(gold_spans)
        pred_malformed += malformed_count(pred_seq)
        gold_malformed += malformed_count(gold_seq)

        # Precision-like sum over predicted spans
        for p_start, p_end in pred_spans:
            p_len = max(1, p_end - p_start)
            best_overlap = 0
            for g_start, g_end in gold_spans:
                overlap = _span_overlap((p_start, p_end), (g_start, g_end))
                if overlap > best_overlap:
                    best_overlap = overlap
            cumulative_prec += best_overlap / p_len

        # Recall-like sum over gold spans
        for g_start, g_end in gold_spans:
            g_len = max(1, g_end - g_start)
            best_overlap = 0
            for p_start, p_end in pred_spans:
                overlap = _span_overlap((p_start, p_end), (g_start, g_end))
                if overlap > best_overlap:
                    best_overlap = overlap
            cumulative_rec += best_overlap / g_len

    return {
        "cumulative_prec": cumulative_prec,
        "cumulative_rec": cumulative_rec,
        "pred_span_count": pred_span_count,
        "gold_span_count": gold_span_count,
        "pred_malformed": pred_malformed,
        "gold_malformed": gold_malformed,
    }


def generate_evaluation_debug_report(
    step: int,
    batch_on_device: Dict[str, torch.Tensor],
    outputs: Dict[str, Any],
    labels: torch.Tensor,
    mask: torch.Tensor,
    predictions,
    tokenizer: AutoTokenizer,
) -> List[str]:
    """Generate a human-readable debug report for a batch.

    Includes token text, gold tags, and predicted tags for the first few examples.
    """
    report = [f"Batch {step} debug report:"]
    input_ids = batch_on_device["input_ids"].detach().cpu()
    attention_mask = batch_on_device["attention_mask"].detach().cpu()
    labels_cpu = labels.detach().cpu()
    mask_cpu = mask.detach().cpu()

    if isinstance(predictions, torch.Tensor):
        pred_cpu = predictions.detach().cpu()
        pred_seqs = [pred[: int(m.sum().item())].tolist() for pred, m in zip(pred_cpu, mask_cpu)]
    else:
        pred_seqs = [list(map(int, seq)) for seq in predictions]

    for i in range(input_ids.size(0)):
        seq_len = int(attention_mask[i].sum().item())
        tokens = tokenizer.convert_ids_to_tokens(input_ids[i, :seq_len].tolist())
        gold_seq = labels_cpu[i, :seq_len].tolist()
        gold_seq = [x if x != -100 else 0 for x in gold_seq]
        pred_seq = pred_seqs[i][:seq_len]
        report.append("TOK:\t" + " ".join(tokens))
        report.append("GOLD:\t" + " ".join(map(str, gold_seq)))
        report.append("PRED:\t" + " ".join(map(str, pred_seq)))
        report.append("")
    return report
