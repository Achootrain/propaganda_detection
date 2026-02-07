"""Data classes for technique classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class InputFeatures:
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: List[int]
    label_id: Optional[int]
    length_feat: Optional[float] = None  # scalar length feature
    matchings: Optional[List[float]] = None  # 14-d vector


LABEL_ORDER = [
    'Appeal_to_Authority', 'Doubt', 'Repetition', 'Appeal_to_fear-prejudice', 'Slogans',
    'Black-and-White_Fallacy', 'Loaded_Language', 'Flag-Waving', 'Name_Calling,Labeling',
    'Whataboutism,Straw_Men,Red_Herring', 'Causal_Oversimplification', 'Exaggeration,Minimisation',
    'Bandwagon,Reductio_ad_hitlerum', 'Thought-terminating_Cliches'
]
