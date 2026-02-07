"""Data loading functions for technique classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from ..utils.text_utils import _get_sentence_context


def read_articles_from_folder(folder: str) -> Dict[str, str]:
    """Read all *.txt files and return {article_id: text}.

    Expects filenames like 'article123456789.txt' but only uses the numeric suffix as id.
    """
    folder_path = Path(folder)
    articles: Dict[str, str] = {}
    for file_path in sorted(folder_path.glob("*.txt")):
        name = file_path.stem  # e.g. 'article730081389'
        # Be robust: take everything after 'article' prefix if present, else full stem
        if name.startswith("article"):
            article_id = name[len("article") :]
        else:
            article_id = name
        with file_path.open("r", encoding="utf-8") as handle:
            articles[article_id] = handle.read()
    return articles


def read_task2_labels(path: str) -> Tuple[List[str], List[int], List[int], List[str]]:
    """Read Task 2 (technique classification) labels file.

    Format per line:
        article_id <TAB> technique_label <TAB> span_start <TAB> span_end
    """
    article_ids: List[str] = []
    labels: List[str] = []
    span_starts: List[int] = []
    span_ends: List[int] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 4:
                continue
            aid, label, start, end = parts
            article_ids.append(aid)
            labels.append(label)
            span_starts.append(int(start))
            span_ends.append(int(end))
    return article_ids, span_starts, span_ends, labels


def load_tc_data(
    articles_folder: str,
    labels_file: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load articles and Task 2 labels and build a span-level DataFrame.

    Returns a DataFrame with:
        article_id, span_start, span_end, span_text, context, label
    and the mapping article_id -> full article text.
    """
    articles = read_articles_from_folder(articles_folder)
    article_ids, span_starts, span_ends, labels = read_task2_labels(labels_file)

    rows = []
    for aid, s, e, lab in zip(article_ids, span_starts, span_ends, labels):
        text = articles.get(aid, "")
        s = max(0, min(s, len(text)))
        e = max(0, min(e, len(text)))
        span_text = text[s:e]
        # Mirror paper: use the sentence from which the span was extracted
        context = _get_sentence_context(text, s, e)
        rows.append(
            {
                "article_id": aid,
                "span_start": s,
                "span_end": e,
                "span": span_text.replace("\t", " ").replace("\n", " "),
                "context": context,
                "label": lab,
            }
        )
    df = pd.DataFrame(rows)
    return df, articles


def load_tc_test_template(
    articles_folder: str,
    template_labels_file: str,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Load test articles and template spans.

    template_labels_file has the same format as train labels but label column is a placeholder.
    """
    articles = read_articles_from_folder(articles_folder)
    article_ids, span_starts, span_ends, labels = read_task2_labels(template_labels_file)
    rows = []
    for aid, s, e, lab in zip(article_ids, span_starts, span_ends, labels):
        text = articles.get(aid, "")
        s = max(0, min(s, len(text)))
        e = max(0, min(e, len(text)))
        span_text = text[s:e]
        context = _get_sentence_context(text, s, e)
        rows.append(
            {
                "article_id": aid,
                "span_start": s,
                "span_end": e,
                "span": span_text.replace("\t", " ").replace("\n", " "),
                "context": context,
                "label": lab,  # placeholder
            }
        )
    df = pd.DataFrame(rows)
    return df, articles
