from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from torch.utils.data import TensorDataset


@dataclass
class InputExample:
    guid: str
    words: List[str]
    labels: List[str]
    pos_labels: Optional[List[str]] = None
    ner_labels: Optional[List[str]] = None
    dep_labels: Optional[List[str]] = None
    tech_labels: Optional[List[List[str]]] = None


@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    label_ids: List[int]
    pos_ids: Optional[List[int]] = None
    ner_ids: Optional[List[int]] = None
    dep_ids: Optional[List[int]] = None
    word_ids: Optional[List[int]] = None
    tech_multi_labels: Optional[List[List[int]]] = None
    tech_label_mask: Optional[List[int]] = None
    lexicon_features: Optional[List[List[int]]] = None


@dataclass
class DatasetBundle:
    examples: List[InputExample]
    features: List[InputFeatures]
    dataset: "TensorDataset"
    article_ids: List[str]


@dataclass
class TokenSpan:
    token: str
    start: int
    end: int
