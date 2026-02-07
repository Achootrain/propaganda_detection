from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..utils.types import TokenSpan


def tokenize_like_bio(text: str, nlp) -> List[TokenSpan]:
    spans: List[TokenSpan] = []
    if nlp is None:
        pos = 0
        for line in text.splitlines(True):
            for m in re.finditer(r"\S+", line):
                token_text = m.group()
                cleaned = token_text.strip().replace("\n", " ").replace("\t", " ")
                if len(cleaned) == 0:
                    continue
                if repr(cleaned) in (repr("\ufeff"), repr("\u200f")):
                    continue
                start = pos + m.start()
                end = pos + m.end()
                spans.append(TokenSpan(cleaned, start, end))
            pos += len(line)
        return spans

    for token in nlp(text):
        cleaned = token.text.strip().replace("\n", " ").replace("\t", " ")
        if len(cleaned) == 0:
            continue
        if repr(cleaned) in (repr("\ufeff"), repr("\u200f")):
            continue
        spans.append(TokenSpan(cleaned, int(token.idx), int(token.idx) + len(token.text)))
    return spans


def labels_to_spans(tokens: Sequence[TokenSpan], labels: Sequence[str]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    active_start: Optional[int] = None
    active_end: Optional[int] = None
    for token, label in zip(tokens, labels):
        if label.startswith("B-"):
            if active_start is not None:
                spans.append((active_start, active_end if active_end is not None else token.end))
            active_start = token.start
            active_end = token.end
        elif label.startswith("I-") and active_start is not None:
            active_end = token.end
        else:
            if active_start is not None:
                spans.append((active_start, active_end if active_end is not None else token.end))
            active_start = None
            active_end = None
    if active_start is not None:
        spans.append((active_start, active_end if active_end is not None else active_start))

    merged: List[Tuple[int, int]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def write_token_predictions(
    bio_source: str,
    predictions: Sequence[Sequence[str]],
    output_path: str | Path,
) -> None:
    example_index = 0
    token_index = 0
    output_path = str(output_path)
    with open(bio_source, "r", encoding="utf-8") as reader, open(
        output_path, "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            stripped = line.rstrip("\n")
            if stripped == "":
                writer.write("\n")
                if token_index != 0:
                    example_index += 1
                token_index = 0
                continue
            token = stripped.split("\t")[0]
            label = "O"
            if example_index < len(predictions) and token_index < len(predictions[example_index]):
                label = predictions[example_index][token_index]
            writer.write(f"{token}\t{label}\n")
            token_index += 1


def write_submission_file(spans: Dict[str, List[Tuple[int, int]]], output_path: str | Path) -> None:
    output_path = str(output_path)
    with open(output_path, "w", encoding="utf-8") as handle:
        for article_id in sorted(spans.keys()):
            for start, end in sorted(spans[article_id]):
                handle.write(f"{article_id}\t{start}\t{end}\n")


def aggregate_article_spans(
    sentence_level_labels: Sequence[Sequence[str]],
    article_order: Sequence[str],
    token_cache: Dict[str, List[TokenSpan]],
) -> Dict[str, List[Tuple[int, int]]]:
    flat_labels: List[str] = [label for sequence in sentence_level_labels for label in sequence]
    pointer = 0
    spans: Dict[str, List[Tuple[int, int]]] = {}
    for article_id in article_order:
        tokens = token_cache.get(article_id, [])
        token_count = len(tokens)
        if token_count == 0:
            spans[article_id] = []
            continue
        remaining_labels = len(flat_labels) - pointer
        take = min(token_count, max(0, remaining_labels))
        article_labels = flat_labels[pointer : pointer + take]
        pointer += take
        if take < token_count:
            article_labels.extend(["O"] * (token_count - take))
        spans[article_id] = labels_to_spans(tokens[: len(article_labels)], article_labels)
    if pointer != len(flat_labels):
        print("Unused label predictions detected: %d", len(flat_labels) - pointer)
    return spans
