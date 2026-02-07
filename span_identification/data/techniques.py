from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..utils.text_utils import clean_token_text
from ..utils.types import InputExample, TokenSpan


def get_technique_labels(path: Optional[str] = None) -> List[str]:
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            techniques = [line.strip() for line in handle if line.strip()]
        return ["O"] + techniques

    return [
        "O",
        "Appeal_to_Authority",
        "Appeal_to_fear-prejudice",
        "Bandwagon,Reductio_ad_hitlerum",
        "Black-and-White_Fallacy",
        "Causal_Oversimplification",
        "Doubt",
        "Exaggeration,Minimisation",
        "Flag-Waving",
        "Loaded_Language",
        "Name_Calling,Labeling",
        "Repetition",
        "Slogans",
        "Thought-terminating_Cliches",
        "Whataboutism,Straw_Men,Red_Herring",
    ]


def read_technique_spans(filename: str) -> Dict[str, List[Tuple[int, int, str]]]:
    spans: Dict[str, List[Tuple[int, int, str]]] = {}
    with open(filename, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                article_id = parts[0]
                technique = parts[1]
                start = int(parts[2])
                end = int(parts[3])
                spans.setdefault(article_id, []).append((start, end, technique))
    return spans


def get_techniques_for_token(
    token_start: int,
    token_end: int,
    technique_spans: List[Tuple[int, int, str]],
) -> List[str]:
    overlaps: List[str] = []
    for span_start, span_end, technique in technique_spans:
        if token_start < span_end and token_end > span_start:
            overlaps.append(technique)

    if not overlaps:
        return []

    return list(dict.fromkeys(overlaps))


def get_technique_map(technique_labels: List[str]) -> Dict[str, int]:
    return {label: i for i, label in enumerate(technique_labels)}


def get_multilabel_technique_map(technique_labels: List[str]) -> Dict[str, int]:
    labels = list(technique_labels)
    if labels and labels[0] == "O":
        labels = labels[1:]
    return {label: i for i, label in enumerate(labels)}


def attach_technique_labels_to_examples(
    examples: List[InputExample],
    *,
    article_ids: Sequence[str],
    token_cache: Dict[str, List[TokenSpan]],
    technique_spans_by_article: Dict[str, List[Tuple[int, int, str]]],
) -> None:
    def _norm_token(text: str) -> str:
        return clean_token_text(text)

    example_index = 0
    for article_id in article_ids:
        tokens = token_cache.get(article_id, [])
        technique_spans = technique_spans_by_article.get(article_id, [])
        token_index = 0

        while token_index < len(tokens) and example_index < len(examples):
            example = examples[example_index]
            tech_labels: List[List[str]] = []

            remaining = len(tokens) - token_index
            overrun = len(example.words) > remaining

            for word in example.words:
                if token_index >= len(tokens):
                    tech_labels.append([])
                    continue

                token_span = tokens[token_index]

                if _norm_token(token_span.token) != _norm_token(word):
                    lookahead_limit = min(len(tokens), token_index + 6)
                    found_at = None
                    for j in range(token_index + 1, lookahead_limit):
                        if _norm_token(tokens[j].token) == _norm_token(word):
                            found_at = j
                            break
                    if found_at is not None:
                        token_index = found_at
                        token_span = tokens[token_index]

                tech_labels.append(
                    get_techniques_for_token(token_span.start, token_span.end, technique_spans)
                )
                token_index += 1

            example.tech_labels = tech_labels
            example_index += 1

            if overrun:
                break

        if token_index != len(tokens):
            print(
                f"[WARN] Technique-label alignment: article {article_id} token mismatch: "
                f"consumed={token_index} cache_len={len(tokens)}"
            )

    if example_index != len(examples):
        print(
            f"[WARN] Technique-label alignment: unused BIO examples: {len(examples) - example_index}"
        )
