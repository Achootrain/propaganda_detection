from __future__ import annotations


import sys
from collections import  defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import spacy
import torch
from transformers import AutoModel, AutoTokenizer, logging
from .utils import (
    LABEL_BEGIN,
    LABEL_INSIDE,
    LABEL_OUTSIDE,
    load_article_text,
    _adjust_first_label_in_chunk,   
)



logging.set_verbosity_error()


def tokenize_with_offsets(tokenizer: AutoTokenizer, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Tokenize full article text to get tokens and character offsets without triggering
    model max-length warnings. We chunk manually later, so we temporarily raise
    the tokenizer.model_max_length to avoid the standard HF warning
    (e.g., "704 > 512").

    This does NOT truncate; chunking is handled by chunk_tokens_with_stride.
    """
    old_max_len = getattr(tokenizer, "model_max_length", None)
    try:
        # If model_max_length is small (e.g., 512), temporarily raise it to a large value
        # so HF doesn't warn when we tokenize long documents.
        if isinstance(old_max_len, int) and old_max_len <= 2048:
            tokenizer.model_max_length = 1_000_000  # effectively "no limit" for our purposes
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,  # explicit: we do not want truncation here
        )
    finally:
        if old_max_len is not None:
            tokenizer.model_max_length = old_max_len
    return encoding.tokens(), list(encoding["offset_mapping"])


def label_token(start: int, end: int, spans: Iterable[Tuple[int, int]]) -> str:
    candidates = [(s, e) for (s, e) in spans if start >= s and end <= e]
    if not candidates:
        return str(LABEL_OUTSIDE)
    span_start, span_end = min(candidates, key=lambda x: x[1] - x[0])
    if start == span_start:
        return str(LABEL_BEGIN)
    return str(LABEL_INSIDE)


def chunk_tokens_with_stride(
    tokens: List[str], 
    offsets: List[Tuple[int, int]], 
    labels: List[str], 
    sentence_ids: List[int], 
    max_tokens: int, 
    stride: int
) -> List[Tuple[List[str], List[Tuple[int, int]], List[str], List[int]]]:
    chunks: List[Tuple[List[str], List[Tuple[int, int]], List[str], List[int]]] = []
    start_idx = 0
    while start_idx < len(tokens):
        end_idx = min(start_idx + max_tokens, len(tokens))
        chunks.append((
            tokens[start_idx:end_idx], 
            offsets[start_idx:end_idx], 
            labels[start_idx:end_idx],
            sentence_ids[start_idx:end_idx]
        ))
        if end_idx == len(tokens):
            break
        start_idx += max_tokens - stride
    return chunks




def create_token_labels(
    *,
    articles_dir: Path,
    tokens_output: Path,
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    max_tokens: int,
    stride: int,
    nlp=None,
    spans_file: Path | None = None,
) -> None:
    # Lazy-load spaCy model if not provided
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            # Fallback to a simple rule-based sentencizer if the model isn't available
            nlp = spacy.blank("en")
            if "sentencizer" not in nlp.pipe_names:
                nlp.add_pipe("sentencizer")

    span_map = load_spans(spans_file) if spans_file else defaultdict(list)

    tokens_output.parent.mkdir(parents=True, exist_ok=True)
    with tokens_output.open("w", encoding="utf8") as tokens_f:
        article_files = sorted([p for p in articles_dir.glob("*.txt") if p.is_file()])
        for article_path in article_files:
            article_id = article_path.stem.replace("article", "")
            spans = span_map.get(article_id, [])

            try:
                article_text = load_article_text(article_path)
            except FileNotFoundError as err:
                print(f"Skipping {article_id}: {err}", file=sys.stderr)
                continue

            # Perform sentence segmentation
            doc = nlp(article_text)
            sentence_map = {}
            for i, sent in enumerate(doc.sents):
                for char_idx in range(sent.start_char, sent.end_char):
                    sentence_map[char_idx] = i

            tokens, offsets = tokenize_with_offsets(tokenizer, article_text)
            token_labels = [label_token(start, end, spans) for (start, end) in offsets]
            token_sentence_ids = [sentence_map.get(start, -1) for start, end in offsets]

            for chunk_idx, (tokens_chunk, offsets_chunk, labels_chunk, sentence_ids_chunk) in enumerate(
                chunk_tokens_with_stride(tokens, offsets, token_labels, token_sentence_ids, max_tokens, stride)
            ):
                _adjust_first_label_in_chunk(labels_chunk)

                for (start, end), token, label, sentence_id in zip(offsets_chunk, tokens_chunk, labels_chunk, sentence_ids_chunk):
                    tokens_f.write(f"{article_id}\t{chunk_idx}\t{sentence_id}\t{start}:{end}\t{token}\t{label}\n")


def load_spans(labels_path: Path) -> Dict[str, List[Tuple[int, int]]]:
    """Load gold spans from a Task SI labels file.

    Expected format per line (TAB-separated):
        article_id    start_offset    end_offset

    Returns a mapping from article_id (str) -> list of (start, end) tuples.
    """
    spans: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    if labels_path is None or not labels_path.exists():
        return spans

    with labels_path.open("r", encoding="utf8") as handle:
        for line in handle:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            article_id, start_str, end_str = parts[0], parts[1], parts[2]
            try:
                start, end = int(start_str), int(end_str)
            except ValueError:
                continue
            spans[article_id].append((start, end))

    return spans
