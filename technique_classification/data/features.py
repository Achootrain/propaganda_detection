"""Feature encoding and dataset building."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import TensorDataset
from tqdm import tqdm
import pandas as pd

from ..utils.types import InputFeatures, LABEL_ORDER
from ..utils.text_utils import _simple_stem


def build_label_map(labels: Sequence[str]) -> Tuple[List[str], Dict[str, int]]:
    """Build label mapping using the task's fixed label order.

    This keeps logits->label indices compatible with the repo's reference
    post-processing in `submission.py`.
    """
    observed = set(labels)
    label_list = list(LABEL_ORDER)
    extras = sorted([x for x in observed if x not in set(label_list)])
    label_list.extend(extras)
    label2id = {lab: i for i, lab in enumerate(label_list)}
    return label_list, label2id


def build_train_instances(train_df: pd.DataFrame) -> Dict[str, set]:
    instances: Dict[str, set] = {}
    for _, r in train_df.iterrows():
        span = _simple_stem(str(r['span']))
        lab = str(r['label'])
        if span:
            instances.setdefault(span, set()).add(lab)
    return instances


def compute_matchings(span: str, train_instances: Dict[str, set]) -> List[float]:
    vec = [0.0] * len(LABEL_ORDER)
    stem = _simple_stem(span)
    labs = train_instances.get(stem, set())
    if not labs:
        return vec
    for i, name in enumerate(LABEL_ORDER):
        if name in labs:
            vec[i] = 1.0
    # normalize if any
    s = sum(vec)
    if s > 0:
        vec = [v / s for v in vec]
    return vec


def encode_examples(
    df: pd.DataFrame,
    tokenizer,
    label2id: Optional[Dict[str, int]],
    max_seq_length: int,
    is_train_or_eval: bool,
    use_length: bool = False,
    use_matchings: bool = False,
    join_embeddings: bool = False,  # kept for interface symmetry
    train_instances: Optional[Dict[str, set]] = None,
) -> Tuple[List[InputFeatures], Optional[List[str]]]:
    """Encode spans+context; optionally compute auxiliary features."""
    features: List[InputFeatures] = []
    used_labels: Optional[List[str]] = [] if is_train_or_eval else None

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Encoding examples", leave=False):
        span = str(row['span'])
        context = str(row['context'])
        encoded = tokenizer.encode_plus(
            span,
            context,
            add_special_tokens=True,
            max_length=max_seq_length,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True,
        )
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        token_type_ids = encoded.get('token_type_ids', [0] * len(input_ids))

        if is_train_or_eval:
            label_str = str(row['label'])
            label_id = label2id[label_str]
            used_labels.append(label_str)
        else:
            label_id = None

        length_feat = float(len(span.split())) if use_length else None
        matchings_vec = compute_matchings(span, train_instances) if use_matchings and train_instances else None

        features.append(InputFeatures(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label_id=label_id,
            length_feat=length_feat,
            matchings=matchings_vec,
        ))

    return features, used_labels


def features_to_dataset(features: Sequence[InputFeatures]) -> TensorDataset:
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    length_tensor = None
    matchings_tensor = None
    if any(f.length_feat is not None for f in features):
        length_tensor = torch.tensor([[f.length_feat or 0.0] for f in features], dtype=torch.float)
    if any(f.matchings is not None for f in features):
        matchings_tensor = torch.tensor([f.matchings or [0.0]*len(LABEL_ORDER) for f in features], dtype=torch.float)
    if features[0].label_id is not None:
        all_labels = torch.tensor([f.label_id for f in features], dtype=torch.long)
        tensors = [all_input_ids, all_attention_mask, all_token_type_ids, all_labels]
    else:
        tensors = [all_input_ids, all_attention_mask, all_token_type_ids]
    if length_tensor is not None:
        tensors.append(length_tensor)
    if matchings_tensor is not None:
        tensors.append(matchings_tensor)
    return TensorDataset(*tensors)
