from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple


Span = Tuple[int, int]
SpanMap = Dict[str, List[Span]]


def load_labels_file(path: Path | str) -> SpanMap:
    """Load Task SI .labels file (article_id<TAB>start<TAB>end per line)."""
    p = Path(path)
    mapping: SpanMap = {}
    if not p.exists():
        return mapping
    with p.open("r", encoding="utf8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            aid, s_str, e_str = parts[0], parts[1], parts[2]
            try:
                s, e = int(s_str), int(e_str)
            except ValueError:
                continue
            if e <= s:
                continue
            mapping.setdefault(aid, []).append((s, e))
    return mapping


def _overlap(a: Span, b: Span) -> int:
    s = max(a[0], b[0])
    e = min(a[1], b[1])
    return max(0, e - s)


def _best_overlap_length(span: Span, others: List[Span]) -> int:
    best = 0
    for o in others:
        ov = _overlap(span, o)
        if ov > best:
            best = ov
    return best


def compute_span_prf(pred: SpanMap, gold: SpanMap) -> Tuple[float, float, float]:
    """Compute span-level precision/recall/F1 using overlap-based scoring similar to the official scorer:
    - Precision: average over predicted spans of best_overlap_len / predicted_span_len
    - Recall:    average over gold spans of best_overlap_len / gold_span_len
    """
    cumulative_prec = 0.0
    cumulative_rec = 0.0
    pred_count = 0
    gold_count = 0

    article_ids = set(pred.keys()) | set(gold.keys())
    for aid in article_ids:
        p_spans = pred.get(aid, [])
        g_spans = gold.get(aid, [])

        for ps in p_spans:
            p_len = max(1, ps[1] - ps[0])
            bo = _best_overlap_length(ps, g_spans)
            cumulative_prec += bo / p_len
            pred_count += 1

        for gs in g_spans:
            g_len = max(1, gs[1] - gs[0])
            bo = _best_overlap_length(gs, p_spans)
            cumulative_rec += bo / g_len
            gold_count += 1

    precision = cumulative_prec / max(1, pred_count)
    recall = cumulative_rec / max(1, gold_count)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1
