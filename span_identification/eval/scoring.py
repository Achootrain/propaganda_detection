from __future__ import annotations

from typing import Dict, List, Tuple


def load_span_annotations(filename: str) -> Dict[str, List[Tuple[int, int]]]:
    annotations: Dict[str, List[Tuple[int, int]]] = {}
    with open(filename, "r", encoding="utf-8") as handle:
        for line in handle:
            article_id, start, end = line.rstrip("\n").split("\t")
            annotations.setdefault(article_id, []).append((int(start), int(end)))
    return annotations


def convert_to_position_sets(annotations: Dict[str, List[Tuple[int, int]]]) -> Dict[str, List[set]]:
    converted: Dict[str, List[set]] = {}
    for article_id, spans in annotations.items():
        converted[article_id] = [set(range(start, end)) for start, end in spans if end > start]
    return converted


def compute_precision_recall_f1(
    predicted: Dict[str, List[Tuple[int, int]]],
    gold: Dict[str, List[Tuple[int, int]]],
) -> Dict[str, float]:
    pred_sets = convert_to_position_sets(predicted)
    gold_sets = convert_to_position_sets(gold)
    prec_den = sum(len(spans) for spans in pred_sets.values())
    rec_den = sum(len(spans) for spans in gold_sets.values())
    prec_num = 0.0
    rec_num = 0.0
    for article_id, pred_spans in pred_sets.items():
        gold_spans = gold_sets.get(article_id, [])
        for pred_span in pred_spans:
            pred_len = len(pred_span)
            if pred_len == 0:
                continue
            for gold_span in gold_spans:
                intersection = len(pred_span & gold_span)
                if intersection:
                    prec_num += intersection / pred_len
        for gold_span in gold_spans:
            gold_len = len(gold_span)
            if gold_len == 0:
                continue
            for pred_span in pred_spans:
                intersection = len(pred_span & gold_span)
                if intersection:
                    rec_num += intersection / gold_len
    precision = prec_num / prec_den if prec_den else 0.0
    recall = rec_num / rec_den if rec_den else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision and recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}
