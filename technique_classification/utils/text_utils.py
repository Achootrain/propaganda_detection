"""Text normalization and utility functions."""

from __future__ import annotations

import re
from typing import Dict, List, Set, Tuple

_EN_STOPWORDS = {
    # Determiners / articles
    "a", "an", "the", "this", "that", "these", "those",
    # Pronouns
    "i", "me", "my", "mine", "you", "your", "yours", "we", "us", "our", "ours",
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them", "their", "theirs",
    # Aux/verbs (common)
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    # Prepositions / conjunctions
    "of", "to", "in", "on", "for", "with", "as", "at", "by", "from", "about", "into", "over", "after",
    "and", "or", "but", "if", "then", "than", "because", "while", "when", "where",
    # Negation / misc
    "not", "no", "yes", "just", "very", "really", "now", "here", "there", "also", "too", "such",
}


def _strip_accents(text: str) -> str:
    """Best-effort ASCII folding without external deps."""
    try:
        import unicodedata
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    except Exception:
        return text


def _cheap_stem(token: str) -> str:
    """A tiny, dependency-free stemmer.

    Not a full Porter stemmer, but good enough for repetition/matching heuristics.
    """
    t = token
    for suf in ("'s", "'s"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[: -len(suf)]
    for suf in ("ing", "edly", "edly", "edly", "ed", "ly", "es", "s"):
        if t.endswith(suf) and len(t) > len(suf) + 2:
            t = t[: -len(suf)]
            break
    return t


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def _normalize_for_repetition(text: str) -> str:
    """Normalize span for repetition matching: lowercase, strip accents, remove stopwords/punct, cheap-stem."""
    text = _strip_accents(text)
    toks = []
    for tok in _tokenize_words(text):
        if tok in _EN_STOPWORDS:
            continue
        toks.append(_cheap_stem(tok))
    return " ".join([t for t in toks if t])


def _normalize_for_train_lookup(text: str) -> str:
    """Normalize span for train-instance lookup: keep all word tokens, cheap-stem."""
    text = _strip_accents(text)
    toks = [_cheap_stem(t) for t in _tokenize_words(text)]
    return " ".join([t for t in toks if t])


def _normalize_span_paper(text: str, *, stemmer, stopwords: set) -> str:
    """Paper-style normalization for repetition/train-span matching.

    - lowercase
    - remove punctuation by extracting word tokens
    - filter stopwords
    - apply Porter stemming
    """
    toks = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    out: List[str] = []
    for t in toks:
        if t in stopwords:
            continue
        out.append(stemmer.stem(t))
    return " ".join(out)


def _simple_stem(text: str) -> str:
    # Lightweight stemmer: lowercase + strip non-alpha + collapse spaces
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    return " ".join(tokens)


def _sentence_bounds(text: str) -> List[Tuple[int, int]]:
    """Return sentence spans (start,end) for `text`.

    Uses NLTK PunktSentenceTokenizer when available; otherwise falls back to a
    simple regex-based splitter (no external downloads).
    """
    if not text:
        return [(0, 0)]
    try:
        from nltk.tokenize.punkt import PunktSentenceTokenizer
        return list(PunktSentenceTokenizer().span_tokenize(text))
    except Exception:
        spans: List[Tuple[int, int]] = []
        start = 0
        for m in re.finditer(r"[.!?]\s+|\n+", text):
            end = m.end()
            if end > start:
                spans.append((start, end))
            start = end
        if start < len(text):
            spans.append((start, len(text)))
        return spans or [(0, len(text))]


def _get_sentence_context(article_text: str, span_start: int, span_end: int) -> str:
    """Extract the sentence containing the span (paper-style context)."""
    bounds = _sentence_bounds(article_text)
    for s, e in bounds:
        if s <= span_start and span_end <= e:
            return article_text[s:e].replace("\t", " ").replace("\n", " ").strip()
    # nearest fallback
    if bounds:
        idx = max(0, min(len(bounds) - 1, next((i for i, (s, _) in enumerate(bounds) if s > span_start), len(bounds)) - 1))
        s, e = bounds[idx]
        return article_text[s:e].replace("\t", " ").replace("\n", " ").strip()
    return article_text.replace("\t", " ").replace("\n", " ").strip()


def get_stopwords() -> Set[str]:
    return set(_EN_STOPWORDS)
