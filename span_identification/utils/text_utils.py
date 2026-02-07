"""Text cleaning and simple lexicon features."""

from __future__ import annotations

from typing import Set


HITLER_KEYWORDS: Set[str] = {
    "hitler",
    "nazi",
    "nazis",
    "nazism",
    "fascist",
    "fascism",
    "dictator",
    "totalitarian",
    "ss",
    "holocaust",
    "genocide",
    "reich",
    "gestapo",
}


def clean_token_text(text: str) -> str:
    return text.strip().replace("\n", " ").replace("\t", " ")


def has_hitler_keyword(word: str) -> int:
    """Return 1 if `word` contains any Hitler-related keyword, else 0."""
    normalized = clean_token_text(word).lower()
    return int(any(keyword in normalized for keyword in HITLER_KEYWORDS))
