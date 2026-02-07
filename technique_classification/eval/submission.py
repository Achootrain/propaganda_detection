"""Submission file creation and evaluation."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from ..utils.runtime import LOGGER
from .postprocess import postprocess_predictions_local


def create_submission_file(
    test_df: pd.DataFrame,
    logits: np.ndarray,
    label_list: List[str],
    output_path: str,
    *,
    postprocess: bool = True,
    train_df_for_postprocess: Optional[pd.DataFrame] = None,
) -> None:
    """Given test DataFrame, logits and label names, write submission TSV.

    If `postprocess=True` and repo helpers are available, applies the same
    heuristics described in the paper (Repetition matching, train-span boosting,
    local consistency, and multi-label top-n via duplicate spans).
    """
    # logits -> probabilities (post-processing thresholds assume probabilities)
    try:
        exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
    except Exception:
        probs = logits

    if postprocess and train_df_for_postprocess is not None:
        try:
            pred_labels = postprocess_predictions_local(probs, test_df.copy(), train_df=train_df_for_postprocess)
            label2id = {lab: i for i, lab in enumerate(label_list)}
            preds = np.array([label2id.get(str(lab), int(np.argmax(p))) for lab, p in zip(pred_labels, probs)], dtype=int)
        except Exception as e:
            LOGGER.warning("Post-processing failed; falling back to argmax. Error: %s", e)
            preds = np.argmax(logits, axis=1)
    else:
        preds = np.argmax(logits, axis=1)
    with open(output_path, "w", encoding="utf-8") as handle:
        for (_, row), pred_idx in zip(test_df.iterrows(), preds):
            handle.write(
                f"{row['article_id']}\t{label_list[pred_idx]}\t{row['span_start']}\t{row['span_end']}\n"
            )


def eval_submission_file(
    submission_path: str,
    gold_labels_path: str,
    label_list: List[str],
) -> Dict[str, float]:
    """Compute simple accuracy and per-class F1 for a submission vs gold.

    gold_labels_path is the official labels file; we use only the label column in order.
    """
    preds: List[str] = []
    with open(submission_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            preds.append(parts[1])

    gold: List[str] = []
    with open(gold_labels_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            gold.append(parts[1])

    if len(preds) != len(gold):
        LOGGER.warning("Prediction/gold size mismatch: %d vs %d", len(preds), len(gold))

    # Align lengths if necessary
    n = min(len(preds), len(gold))
    preds = preds[:n]
    gold = gold[:n]

    # Map to indices for consistency
    label2id = {lab: i for i, lab in enumerate(label_list)}
    y_pred = [label2id.get(x, -1) for x in preds if x in label2id]
    y_true = [label2id.get(x, -1) for x in gold if x in label2id]

    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0) if y_true else 0.0
    f1_macro = f1_score(y_true, y_pred, average="macro") if y_true else 0.0
    f1_micro = f1_score(y_true, y_pred, average="micro") if y_true else 0.0
    # Per-class F1 for all techniques (length == len(label_list))
    f1_per_class = f1_score(y_true, y_pred, labels=list(range(len(label_list))), average=None, zero_division=0) if y_true else [0.0]*len(label_list)
    return {
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "f1_per_class": {lab: float(f1_per_class[i]) for i, lab in enumerate(label_list)},
    }
