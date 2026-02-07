from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from ..utils.text_utils import has_hitler_keyword
from ..utils.types import InputExample, InputFeatures


def read_examples_from_file(
    file_path: str, mode: str, expected_count: Optional[int] = None
) -> List[InputExample]:
    guid_index = 1
    examples: List[InputExample] = []
    with open(file_path, encoding="utf-8") as handle:
        words: List[str] = []
        labels: List[str] = []
        current_pos: List[str] = []
        current_ner: List[str] = []
        current_dep: List[str] = []
        for line in handle:
            if line.startswith("-DOCSTART-") or line in ("", "\n"):
                if words:
                    examples.append(
                        InputExample(
                            guid=f"{mode}-{guid_index}",
                            words=words,
                            labels=labels,
                            pos_labels=current_pos,
                            ner_labels=current_ner,
                            dep_labels=current_dep,
                        )
                    )
                    guid_index += 1
                    words = []
                    labels = []
                    current_pos = []
                    current_ner = []
                    current_dep = []
            else:
                splits = line.split("\t")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[1].rstrip("\n"))
                else:
                    labels.append("O")

                if len(splits) > 2:
                    current_pos.append(splits[2])
                else:
                    current_pos.append("X")

                if len(splits) > 3:
                    current_ner.append(splits[3].rstrip("\n") if len(splits) == 4 else splits[3])
                else:
                    current_ner.append("O")

                if len(splits) > 4:
                    current_dep.append(splits[4].rstrip("\n"))
                else:
                    current_dep.append("dep")

        if words:
            examples.append(
                InputExample(
                    guid=f"{mode}-{guid_index}",
                    words=words,
                    labels=labels,
                    pos_labels=current_pos,
                    ner_labels=current_ner,
                    dep_labels=current_dep,
                )
            )

    if expected_count is not None and len(examples) != expected_count:
        raise ValueError(
            f"Expected {expected_count} {mode} examples, found {len(examples)} in {file_path}"
        )
    return examples


def convert_examples_to_features(
    examples: Sequence[InputExample],
    label_list: Sequence[str],
    max_seq_length: int,
    tokenizer,
    pad_token_label_id: int,
    pos_label_map: Optional[Dict[str, int]] = None,
    ner_label_map: Optional[Dict[str, int]] = None,
    dep_label_map: Optional[Dict[str, int]] = None,
    tech_label_map: Optional[Dict[str, int]] = None,
) -> List[InputFeatures]:
    label_map = {label: i for i, label in enumerate(label_list)}
    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token
    pad_token = tokenizer.pad_token_id

    pos_map = pos_label_map if pos_label_map else {}
    ner_map = ner_label_map if ner_label_map else {}
    dep_map = dep_label_map if dep_label_map else {}
    tech_map = tech_label_map if tech_label_map else {}
    num_techniques = len(tech_map)

    features: List[InputFeatures] = []
    for example in examples:
        tokens: List[str] = []
        label_ids: List[int] = []
        pos_ids: List[int] = []
        ner_ids: List[int] = []
        dep_ids: List[int] = []
        word_ids: List[int] = []
        tech_multi_labels: List[List[int]] = []
        tech_label_mask: List[int] = []
        lexicon_features: List[List[int]] = []

        ex_pos_labels = example.pos_labels if example.pos_labels else ["X"] * len(example.words)
        ex_ner_labels = example.ner_labels if example.ner_labels else ["O"] * len(example.words)
        ex_dep_labels = example.dep_labels if example.dep_labels else ["dep"] * len(example.words)
        ex_tech_labels = example.tech_labels if example.tech_labels else [[] for _ in range(len(example.words))]

        for word_idx, (word, label, pos, ner, dep, tech_list) in enumerate(
            zip(example.words, example.labels, ex_pos_labels, ex_ner_labels, ex_dep_labels, ex_tech_labels)
        ):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [tokenizer.unk_token]
            tokens.extend(word_tokens)
            lex_flag = has_hitler_keyword(word)

            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

            p_id = pos_map.get(pos, 0)
            pos_ids.extend([p_id] + [0] * (len(word_tokens) - 1))

            n_id = ner_map.get(ner, 0)
            ner_ids.extend([n_id] + [0] * (len(word_tokens) - 1))

            d_id = dep_map.get(dep, 0)
            dep_ids.extend([d_id] + [0] * (len(word_tokens) - 1))

            word_ids.extend([word_idx] * len(word_tokens))

            multi = [0] * num_techniques
            if isinstance(tech_list, str):
                tech_list = [tech_list] if tech_list and tech_list != "O" else []
            for tech in (tech_list or []):
                idx = tech_map.get(tech)
                if idx is not None:
                    multi[idx] = 1
            tech_multi_labels.extend([multi] + [[0] * num_techniques] * (len(word_tokens) - 1))
            tech_label_mask.extend([1] + [0] * (len(word_tokens) - 1))

            lexicon_features.extend([[lex_flag]] * len(word_tokens))

        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: max_seq_length - special_tokens_count]
            label_ids = label_ids[: max_seq_length - special_tokens_count]
            pos_ids = pos_ids[: max_seq_length - special_tokens_count]
            ner_ids = ner_ids[: max_seq_length - special_tokens_count]
            dep_ids = dep_ids[: max_seq_length - special_tokens_count]
            word_ids = word_ids[: max_seq_length - special_tokens_count]
            tech_multi_labels = tech_multi_labels[: max_seq_length - special_tokens_count]
            tech_label_mask = tech_label_mask[: max_seq_length - special_tokens_count]
            lexicon_features = lexicon_features[: max_seq_length - special_tokens_count]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        pos_ids += [0]
        ner_ids += [0]
        dep_ids += [0]
        word_ids += [-1]
        tech_multi_labels += [[0] * num_techniques]
        tech_label_mask += [0]
        lexicon_features += [[0]]
        segment_ids = [0] * len(tokens)

        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        pos_ids = [0] + pos_ids
        ner_ids = [0] + ner_ids
        dep_ids = [0] + dep_ids
        word_ids = [-1] + word_ids
        tech_multi_labels = [[0] * num_techniques] + tech_multi_labels
        tech_label_mask = [0] + tech_label_mask
        lexicon_features = [[0]] + lexicon_features
        segment_ids = [0] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)

        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        attention_mask += [0] * padding_length
        segment_ids += [0] * padding_length
        label_ids += [pad_token_label_id] * padding_length
        pos_ids += [0] * padding_length
        ner_ids += [0] * padding_length
        dep_ids += [0] * padding_length
        word_ids += [-1] * padding_length
        tech_multi_labels += [[0] * num_techniques] * padding_length
        tech_label_mask += [0] * padding_length
        lexicon_features += [[0]] * padding_length

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=segment_ids,
                label_ids=label_ids,
                pos_ids=pos_ids,
                ner_ids=ner_ids,
                dep_ids=dep_ids,
                word_ids=word_ids,
                tech_multi_labels=tech_multi_labels,
                tech_label_mask=tech_label_mask,
                lexicon_features=lexicon_features,
            )
        )

    return features


def get_labels(path: Optional[str] = None) -> List[str]:
    if path:
        with open(path, "r", encoding="utf-8") as handle:
            labels = handle.read().splitlines()
        if "O" not in labels:
            labels = ["O"] + labels
        return labels
    return ["O", "B-PROP", "I-PROP"]


def get_pos_ner_dep_maps(
    examples: List[InputExample],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    pos_types = set()
    ner_types = set()
    dep_types = set()
    for ex in examples:
        if ex.pos_labels:
            pos_types.update(ex.pos_labels)
        if ex.ner_labels:
            ner_types.update(ex.ner_labels)
        if ex.dep_labels:
            dep_types.update(ex.dep_labels)

    sorted_pos = sorted(list(pos_types))
    sorted_ner = sorted(list(ner_types))
    sorted_dep = sorted(list(dep_types))

    pos_map = {label: i + 1 for i, label in enumerate(sorted_pos)}
    ner_map = {label: i + 1 for i, label in enumerate(sorted_ner)}
    dep_map = {label: i + 1 for i, label in enumerate(sorted_dep)}

    return pos_map, ner_map, dep_map


def get_pos_ner_maps(examples: List[InputExample]) -> Tuple[Dict[str, int], Dict[str, int]]:
    pos_map, ner_map, _ = get_pos_ner_dep_maps(examples)
    return pos_map, ner_map
