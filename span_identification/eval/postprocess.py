from __future__ import annotations

import string
from typing import Dict, List, Optional, Tuple


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _is_span_boundary_char(ch: str) -> bool:
    KEEP_PUNCT = {"'", "'", '"', '"', '"', "-", "–", "—", "!", "?", "."}
    return ch.isalnum() or ch == "_" or ch in KEEP_PUNCT


def _adjust_span_to_words(
    text: str, s: int, e: int, keep_punct: bool = False
) -> Optional[Tuple[int, int]]:
    n = len(text)
    s = max(0, min(s, n))
    e = max(0, min(e, n))
    if e <= s:
        return None

    _ = _is_span_boundary_char if keep_punct else _is_word_char

    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1

    if not keep_punct:
        while s < e and not _is_word_char(text[s]):
            s += 1
        while e > s and not _is_word_char(text[e - 1]):
            e -= 1

    if e <= s:
        return None

    if s > 0 and _is_word_char(text[s]) and _is_word_char(text[s - 1]):
        while s > 0 and _is_word_char(text[s - 1]):
            s -= 1

    if e < n and _is_word_char(text[e - 1]) and _is_word_char(text[e]):
        while e < n and _is_word_char(text[e]):
            e += 1

    if e <= s:
        return None

    return s, e


def postprocess_spans(
    spans: Dict[str, List[Tuple[int, int]]],
    article_texts: Dict[str, str],
    keep_punct: bool = True,
) -> Dict[str, List[Tuple[int, int]]]:
    SINGLE_WORD_STOP = {
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "some",
        "any",
        "all",
        "each",
        "every",
        "i",
        "me",
        "my",
        "mine",
        "you",
        "your",
        "yours",
        "we",
        "us",
        "our",
        "ours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "it",
        "its",
        "they",
        "them",
        "their",
        "theirs",
        "who",
        "whom",
        "whose",
        "which",
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "from",
        "by",
        "about",
        "as",
        "into",
        "like",
        "through",
        "after",
        "over",
        "between",
        "out",
        "against",
        "during",
        "without",
        "before",
        "under",
        "around",
        "among",
        "via",
        "and",
        "but",
        "or",
        "nor",
        "so",
        "yet",
        "if",
        "because",
        "while",
        "when",
        "where",
        "although",
        "unless",
        "is",
        "am",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "its",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "can",
        "could",
        "may",
        "might",
        "must",
        "not",
        "no",
        "yes",
        "just",
        "very",
        "really",
        "now",
        "then",
        "here",
        "there",
        "also",
        "too",
    }

    adjusted: Dict[str, List[Tuple[int, int]]] = {}
    for aid, span_list in spans.items():
        text = article_texts.get(aid)
        if text is None:
            adjusted[aid] = list(span_list)
            continue
        new_spans: List[Tuple[int, int]] = []
        for s, e in span_list:
            adj = _adjust_span_to_words(text, s, e, keep_punct=keep_punct)
            if adj is not None:
                ss, ee = adj
                frag = text[ss:ee]
                words = [w.lower() for w in __import__("re").findall(r"[A-Za-z0-9_]+", frag)]
                if len(words) == 1 and words[0] in SINGLE_WORD_STOP:
                    continue
                new_spans.append(adj)

        unique_sorted = sorted(set(new_spans))

        merged: List[Tuple[int, int]] = []
        for s, e in unique_sorted:
            if merged:
                ps, pe = merged[-1]
                if pe <= s:
                    gap = text[pe:s]
                    if gap and all(
                        ch in string.whitespace or ch in ',:;.!?"\'()-[]' for ch in gap
                    ):
                        merged[-1] = (ps, e)
                        continue
                    stripped = gap.strip()
                    if stripped and all(_is_word_char(ch) for ch in stripped):
                        merged[-1] = (ps, e)
                        continue
            merged.append((s, e))

        adjusted[aid] = merged

    return adjusted
