from __future__ import annotations

import os
from typing import Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm

from ..utils.text_utils import clean_token_text


def load_data(
    data_folder: str, propaganda_techniques_file: str
) -> Tuple[List[str], List[str], List[str]]:
    file_list = sorted(
        os.path.join(data_folder, name)
        for name in os.listdir(data_folder)
        if name.endswith(".txt")
    )
    articles_content: List[str] = []
    articles_id: List[str] = []
    for filename in file_list:
        with open(filename, "r", encoding="utf-8") as handle:
            articles_content.append(handle.read())
            articles_id.append(os.path.basename(filename).split(".")[0][7:])

    propaganda_techniques_names: List[str] = []
    if propaganda_techniques_file and os.path.exists(propaganda_techniques_file):
        with open(propaganda_techniques_file, "r", encoding="utf-8") as handle:
            propaganda_techniques_names = [line.rstrip("\n") for line in handle]

    return articles_content, articles_id, propaganda_techniques_names


def read_predictions_from_file(filename: str) -> Tuple[List[str], List[Tuple[int, int]]]:
    articles_id: List[str] = []
    gold_spans: List[Tuple[int, int]] = []
    with open(filename, "r", encoding="utf-8") as handle:
        for row in handle.readlines():
            article_id, gold_span_start, gold_span_end = row.rstrip("\n").split("\t")
            articles_id.append(article_id)
            gold_spans.append((int(gold_span_start), int(gold_span_end)))
    return articles_id, gold_spans


def group_spans_by_article_ids(
    span_list: Iterable[Tuple[str, Tuple[int, int]]],
) -> Dict[str, List[Tuple[int, int]]]:
    data: Dict[str, List[Tuple[int, int]]] = {}
    for article_id, span in span_list:
        data.setdefault(article_id, []).append(span)
    return data


def token_label_from_spans(pos: int, spans: Sequence[Tuple[int, int]]) -> str:
    for start, end in spans:
        if start <= int(pos) < end:
            return "PROP"
    return "O"


def create_bio_labeled(
    file_path: str,
    data: Sequence[Tuple[str, List[Tuple[int, int]]]],
    articles_content: Dict[str, str],
    nlp,
) -> None:
    with open(file_path, "w", encoding="utf-8") as handle:
        for article_id, spans in tqdm(
            data, desc=f"Writing {os.path.basename(file_path)}", leave=False
        ):
            text = articles_content[article_id]
            if nlp is None:
                raise ValueError("spaCy model is required for extracting POS/NER features")

            lines = text.splitlines()
            current_offset = 0
            wrote_sentence = False

            for line in lines:
                line_start = current_offset
                line_end = current_offset + len(line)

                if not line.strip():
                    current_offset = line_end + 1
                    continue

                doc = nlp(line)
                prev_label = "O"
                sentence_has_tokens = False

                for token in doc:
                    cleaned = clean_token_text(token.text)
                    if not cleaned or repr(cleaned) in (repr("\ufeff"), repr("\u200f")):
                        prev_label = "O"
                        continue

                    token_start = line_start + token.idx
                    label = token_label_from_spans(token_start, spans)
                    if label != "O":
                        label = ("I-" if prev_label != "O" else "B-") + "PROP"

                    pos_tag = token.pos_ if token.pos_ else "X"
                    ent_type = token.ent_type_ if token.ent_type_ else "O"
                    dep_rel = token.dep_ if token.dep_ else "dep"

                    handle.write(f"{cleaned}\t{label}\t{pos_tag}\t{ent_type}\t{dep_rel}\n")
                    prev_label = label
                    sentence_has_tokens = True

                if sentence_has_tokens:
                    handle.write("\n")
                    wrote_sentence = True

                current_offset = line_end + 1

            if not wrote_sentence:
                handle.write("\n")


def create_bio_unlabeled(
    file_path: str,
    articles_id: Sequence[str],
    articles_content: Sequence[str],
    nlp,
) -> None:
    with open(file_path, "w", encoding="utf-8") as handle:
        for article_id, text in tqdm(
            zip(articles_id, articles_content),
            total=len(articles_id),
            desc=f"Writing {os.path.basename(file_path)}",
            leave=False,
        ):
            if nlp is None:
                raise ValueError("spaCy model is required for extracting POS/NER features")

            lines = text.splitlines()
            wrote_sentence = False

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                doc = nlp(line)

                sentence_has_tokens = False
                for token in doc:
                    cleaned = clean_token_text(token.text)
                    if not cleaned or repr(cleaned) in (repr("\ufeff"), repr("\u200f")):
                        continue

                    pos_tag = token.pos_ if token.pos_ else "X"
                    ent_type = token.ent_type_ if token.ent_type_ else "O"
                    dep_rel = token.dep_ if token.dep_ else "dep"

                    handle.write(f"{cleaned}\tO\t{pos_tag}\t{ent_type}\t{dep_rel}\n")
                    sentence_has_tokens = True

                if sentence_has_tokens:
                    handle.write("\n")
                    wrote_sentence = True

            if not wrote_sentence:
                handle.write("\n")
