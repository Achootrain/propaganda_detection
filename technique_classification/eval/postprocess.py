"""Post-processing for predictions."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..utils.types import LABEL_ORDER
from ..utils.text_utils import (
    _cheap_stem,
    _normalize_span_paper,
    _normalize_for_train_lookup,
    get_stopwords,
)


def build_train_instances_for_postprocess(train_df: pd.DataFrame) -> Dict[str, set]:
    """Map normalized span -> set(labels) from training (excluding Repetition like the paper code)."""
    instances: Dict[str, set] = {}
    for _, r in train_df.iterrows():
        lab = str(r.get("label", ""))
        if lab == "Repetition":
            continue
        span = _normalize_for_train_lookup(str(r.get("span", "")))
        if not span:
            continue
        instances.setdefault(span, set()).add(lab)
    return instances


def build_insides_from_train(train_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Collect span-subspan label co-occurrence counts from training."""
    insides: Dict[str, Dict[str, int]] = {}
    spans_coords = list(zip(train_df["span_start"].values.tolist(), train_df["span_end"].values.tolist()))
    labels = train_df["label"].astype(str).values.tolist()
    article_ids = train_df["article_id"].astype(str).values.tolist()
    for i in range(len(spans_coords)):
        for j in range(i):
            if article_ids[i] != article_ids[j]:
                continue
            (s1, e1) = spans_coords[i]
            (s2, e2) = spans_coords[j]
            if s1 >= s2 and e1 <= e2 and (s1 != s2 or e1 != e2):
                insides.setdefault(labels[i], {})
                insides[labels[i]][labels[j]] = insides[labels[i]].get(labels[j], 0) + 1
            if s2 >= s1 and e2 <= e1 and (s1 != s2 or e1 != e2):
                insides.setdefault(labels[j], {})
                insides[labels[j]][labels[i]] = insides[labels[j]].get(labels[i], 0) + 1
    return insides


def postprocess_predictions_local(
    probs: np.ndarray,
    data: pd.DataFrame,
    *,
    train_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Paper-faithful post-processing.

    Implements (as described):
    - Repetition heuristic using punctuation removal + stopwords + Porter stemmer.
    - Train-span boosting: +0.5 to probabilities for labels observed with the span in training.
    - Local consistency for nested spans using observed span-subspan label combinations.
    - Multi-label recovery: for duplicated spans, assign top-n labels (n = number of duplicates).

    Returns label strings for each row in `data`.
    """
    if probs.ndim != 2:
        raise ValueError("probs must be 2D: [N, num_labels]")
    n, k = probs.shape
    if k != len(LABEL_ORDER):
        raise ValueError(f"Expected {len(LABEL_ORDER)} labels, got {k}")

    inv = {lab: i for i, lab in enumerate(LABEL_ORDER)}
    rep_idx = inv["Repetition"]

    # Stopwords (keep same list as earlier, but enforce paper-style stemming)
    stopwords = get_stopwords()
    try:
        from nltk.stem import PorterStemmer
        stemmer = PorterStemmer()
    except Exception:
        # If NLTK isn't available, fall back to a minimal stemmer.
        class _FallbackStemmer:
            def stem(self, t: str) -> str:
                return _cheap_stem(t)
        stemmer = _FallbackStemmer()

    # Build allowed nested label pairs from training: outer -> set(inner)
    allowed_pairs: Dict[str, set] = {}
    train_span_to_labels: Dict[str, set] = {}
    if train_df is not None and len(train_df) > 0:
        # span -> set(labels)
        for _, r in train_df.iterrows():
            lab = str(r.get("label", ""))
            key = _normalize_span_paper(str(r.get("span", "")), stemmer=stemmer, stopwords=stopwords)
            if key:
                train_span_to_labels.setdefault(key, set()).add(lab)

        # nested label pairs
        spans_coords = list(zip(train_df["span_start"].astype(int).values.tolist(), train_df["span_end"].astype(int).values.tolist()))
        labels = train_df["label"].astype(str).values.tolist()
        article_ids = train_df["article_id"].astype(str).values.tolist()
        for i in range(len(spans_coords)):
            for j in range(len(spans_coords)):
                if i == j:
                    continue
                if article_ids[i] != article_ids[j]:
                    continue
                (s1, e1) = spans_coords[i]
                (s2, e2) = spans_coords[j]
                # i is outer, j is inner
                if s2 >= s1 and e2 <= e1 and (s1 != s2 or e1 != e2):
                    allowed_pairs.setdefault(labels[i], set()).add(labels[j])

    out_preds: List[str] = [""] * n

    # group by article_id
    for _, group in data.reset_index(drop=False).groupby("article_id", sort=False):
        row_indices = group["index"].astype(int).values.tolist()
        spans_source = group["span"].astype(str).values.tolist()
        spans_coords = list(
            zip(
                group["span_start"].astype(int).values.tolist(),
                group["span_end"].astype(int).values.tolist(),
            )
        )

        # Work on a local copy of probabilities for this article
        p = probs[row_indices, :].copy()

        # Precompute normalized keys for matching
        norm_keys = [
            _normalize_span_paper(s, stemmer=stemmer, stopwords=stopwords)
            for s in spans_source
        ]

        # Repetition counts: number of spans in this article sharing the same normalized key
        rep_counts: Dict[str, int] = {}
        for k_ in norm_keys:
            rep_counts[k_] = rep_counts.get(k_, 0) + 1

        # --- Step 1: Repetition probability adjustment (paper thresholds) ---
        for i in range(len(row_indices)):
            key = norm_keys[i]
            c = rep_counts.get(key, 0)
            rep_p = float(p[i, rep_idx])
            if c >= 3:
                # matched at least two other spans
                p[i, :] = 0.0
                p[i, rep_idx] = 1.0
            elif c == 2:
                # matched only one other span
                if rep_p > 0.001:
                    p[i, :] = 0.0
                    p[i, rep_idx] = 1.0
            else:
                # matched no other spans
                if rep_p < 0.99:
                    p[i, rep_idx] = 0.0
                    s = float(p[i, :].sum())
                    if s > 0:
                        p[i, :] = p[i, :] / s

        # --- Step 2: Train-span boost (+0.5 to probabilities) ---
        if train_span_to_labels:
            for i in range(len(row_indices)):
                labs = train_span_to_labels.get(norm_keys[i], set())
                if not labs:
                    continue
                for lab in labs:
                    j = inv.get(lab)
                    if j is not None:
                        p[i, j] = float(p[i, j]) + 0.5
                # renormalize
                s = float(p[i, :].sum())
                if s > 0:
                    p[i, :] = p[i, :] / s

        # --- Step 3: Local consistency for nested spans ---
        # Iterate a few times to stabilize.
        for _ in range(2):
            pred_idx = np.argmax(p, axis=1)
            pred_lab = [LABEL_ORDER[int(ix)] for ix in pred_idx.tolist()]
            pred_conf = [float(p[i, int(pred_idx[i])]) for i in range(len(pred_idx))]

            for i in range(len(spans_coords)):
                for j in range(len(spans_coords)):
                    if i == j:
                        continue
                    (si, ei) = spans_coords[i]
                    (sj, ej) = spans_coords[j]
                    if sj >= si and ej <= ei and (sj != si or ej != ei):
                        outer = pred_lab[i]
                        inner = pred_lab[j]
                        if inner in allowed_pairs.get(outer, set()):
                            continue

                        # Invalid combination. Set to 0 the smaller of the two probabilities,
                        # unless the new maximum for that span would drop more than twice.
                        if pred_conf[i] <= pred_conf[j]:
                            target = i
                        else:
                            target = j

                        old_max = float(p[target, :].max())
                        old_label_ix = int(np.argmax(p[target, :]))
                        p2 = p[target, :].copy()
                        p2[old_label_ix] = 0.0
                        new_max = float(p2.max())
                        if old_max > 0 and new_max >= (old_max / 2.0):
                            p[target, :] = p2
                            s = float(p[target, :].sum())
                            if s > 0:
                                p[target, :] = p[target, :] / s

        # --- Step 4: Multi-label recovery for duplicated spans (top-n) ---
        # Group by exact coordinates within this article.
        coord_to_rows: Dict[Tuple[int, int], List[int]] = {}
        for i, (s, e) in enumerate(spans_coords):
            coord_to_rows.setdefault((s, e), []).append(i)

        pred_lab_final: List[str] = [""] * len(spans_coords)
        for (s, e), rows in coord_to_rows.items():
            n_dups = len(rows)
            # Use the first occurrence's probabilities as representative.
            # (All duplicates share the same text span.)
            base = rows[0]
            order = np.argsort(-p[base, :])
            topn = [LABEL_ORDER[int(ix)] for ix in order[:n_dups]]
            # assign in order of duplicates
            for rpos, row_local in enumerate(rows):
                pred_lab_final[row_local] = topn[min(rpos, len(topn) - 1)]

        for local_pos, ridx in enumerate(row_indices):
            out_preds[ridx] = pred_lab_final[local_pos]

    return np.array(out_preds, dtype=object)
