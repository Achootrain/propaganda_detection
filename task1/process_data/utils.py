import csv
import json
import string
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from transformers import AutoTokenizer


@dataclass(frozen=True)
class TokenRecord:
    article_id: str
    chunk_index: int
    sentence_id: int
    span: str
    token: str
    label: int


# Numeric label mapping: 0 -> Outside, 1 -> Begin, 2 -> Inside
LABEL_OUTSIDE = 0
LABEL_BEGIN = 1
LABEL_INSIDE = 2


def resolve_article_path(article_id: str, articles_dir: Path) -> Path:
    file_name = article_id
    if not file_name.startswith("article"):
        file_name = f"article{file_name}"
    path = articles_dir / f"{file_name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Article file not found: {path}")
    return path


def load_article_text(article_path: Path) -> str:
    return article_path.read_text(encoding="utf8")


def read_token_rows(path: Path) -> Iterator[TokenRecord]:
    """Stream token rows from a tokenizer TSV output."""
    with path.open("r", encoding="utf8") as handle:
        for line_num, raw_line in enumerate(handle, 1):
            parts = raw_line.rstrip("\n").split("\t")
            sentence_id_str = "-1"  # Default to -1 if not present

            if len(parts) == 5:
                article_id, chunk_idx_str, span, token, label_str = parts
            elif len(parts) == 6:
                article_id, chunk_idx_str, sentence_id_str, span, token, label_str = parts
            else:
                continue

            try:
                chunk_index = int(chunk_idx_str)
                sentence_id = int(sentence_id_str)
                label_id = int(label_str)
            except ValueError:
                continue

            yield TokenRecord(
                article_id=article_id,
                chunk_index=chunk_index,
                sentence_id=sentence_id,
                span=span,
                token=token,
                label=label_id,
            )


def group_token_records(records: Iterable[TokenRecord]) -> Iterator[List[TokenRecord]]:
    """Group token records by article and chunk index."""
    keyfunc = lambda rec: (rec.article_id, rec.chunk_index)
    for _, group in groupby(sorted(records, key=keyfunc), key=keyfunc):
        grouped = list(group)
        if grouped:
            yield grouped


def parse_span(span: str) -> Optional[Tuple[int, int]]:
    """Convert a ``start:end`` span string into integer offsets."""
    if not span:
        return None
    try:
        start_str, end_str = span.split(":", 1)
        return int(start_str), int(end_str)
    except (ValueError, AttributeError):
        return None


def load_vocab_mapping(path: Path) -> Dict[str, int]:
    """Load a token-to-id mapping from a JSON vocabulary file."""
    with path.open("r", encoding="utf8") as handle:
        data = json.load(handle)
    return {str(token): int(token_id) for token, token_id in data.items()}


def _adjust_first_label_in_chunk(labels_chunk: List[str]):
    first_label_idx = -1
    for i, label in enumerate(labels_chunk):
        if label.strip() != str(LABEL_OUTSIDE):
            first_label_idx = i
            break
    if first_label_idx != -1 and labels_chunk[first_label_idx].strip() == str(LABEL_INSIDE):
        labels_chunk[first_label_idx] = str(LABEL_BEGIN)


def read_tokens(
    path: Path,
    *,
    delimiter: str = "\t",
    token_column: int = 3,
) -> Iterable[str]:
    """Stream tokens from a tokenized TSV file."""
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf8", newline="") as handle:
        reader = csv.reader(handle, delimiter=delimiter)
        for row in reader:
            if len(row) <= token_column:
                continue
            token = row[token_column]
            if not token:
                continue
            yield token


def should_keep_token(token: str) -> bool:
    """Return True if the token should remain in the vocabulary output."""
    stripped_token = token.lstrip("ĠĊ")
    if not stripped_token:
        return True

    if (
        (len(set(stripped_token)) == 1 and stripped_token[0] in string.punctuation and len(stripped_token) > 1)
        or (not stripped_token.isascii() or any(ch not in string.printable for ch in stripped_token))
        or (sum(ch in string.punctuation or ch.isdigit() for ch in stripped_token) / len(stripped_token) >= 0.66)
        or (stripped_token in {"),;", "?!", "?!?", "!?", "??", "..)"})
    ):
        return False

    return True


def build_vocab(
    tokens: Iterable[str],
    *,
    tokenizer,
    special_tokens: Iterable[str] = (),
    min_frequency: int = 1,
    max_size: Optional[int] = None,
) -> List[Tuple[str, int]]:
    """Select tokens present in a pretrained tokenizer and map them to IDs."""
    frequency = Counter(tokens)
    vocab: List[Tuple[str, int]] = []
    seen: set[str] = set()

    def convert(token: str) -> int:
        token_id = tokenizer.convert_tokens_to_ids(token)
        return int(token_id)

    for token in special_tokens:
        if token in seen:
            continue
        vocab.append((token, convert(token)))
        frequency.pop(token, None)
        seen.add(token)

    sorted_candidates = sorted(
        (
            (token, count)
            for token, count in frequency.items()
            if count >= max(1, min_frequency) and token not in seen and should_keep_token(token)
        ),
        key=lambda item: (-item[1], item[0]),
    )

    unk_id = getattr(tokenizer, "unk_token_id", None)

    for token, _ in sorted_candidates:
        token_id = convert(token)
        if unk_id is not None and token_id == unk_id:
            continue
        vocab.append((token, token_id))
        seen.add(token)
        if max_size is not None and len(vocab) >= max_size:
            break

    return vocab


def write_vocab(entries: List[Tuple[str, int]], path: Path) -> None:
    """Persist the vocabulary as a JSON token-to-id mapping."""
    mapping = {token: token_id for token, token_id in entries}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf8") as handle:
        json.dump(mapping, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _collect_spans(sequence: Sequence[int]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for idx, label in enumerate(sequence):
        if label == LABEL_BEGIN:
            if start is not None and start < idx:
                spans.append((start, idx))
            start = idx
        elif label == LABEL_INSIDE:
            pass
        else:
            if start is not None and start < idx:
                spans.append((start, idx))
            start = None
    if start is not None and start < len(sequence):
        spans.append((start, len(sequence)))
    return spans


def _get_span_texts(spans: List[Tuple[int, int]], tokens: List[str], tokenizer) -> List[List[str]]:
    span_texts = []
    for start, end in spans:
        span_tokens = tokens[start:end]
        span_texts.append(span_tokens)
    return span_texts


def _span_overlap(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> int:
    start = max(span_a[0], span_b[0])
    end = min(span_a[1], span_b[1])
    return max(0, end - start)

def load_tokenizer(model_name: str):
    """Load a tokenizer configured for byte-level BPE with prefix spacing."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if hasattr(tokenizer, "add_prefix_space"):
        tokenizer.add_prefix_space = True
    return tokenizer

