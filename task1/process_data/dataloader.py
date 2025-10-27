from __future__ import annotations

from typing import Dict, List, Optional, Sequence
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from .utils import TokenRecord, read_token_rows, group_token_records

class PropagandaTokenDataset(Dataset):
    def __init__(
        self,
        chunks: Sequence[List[TokenRecord]],
        tokenizer,
        max_length: int = 256,
        vocab_map: Optional[Dict[str, int]] = None,
    ) -> None:
        self._chunks = list(chunks)
        self._tokenizer = tokenizer
        self._max_length = max_length
        self._vocab_map = dict(vocab_map or {})

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, index: int):
        chunk = self._chunks[index]
        tokens = [record.token for record in chunk]
        labels = [record.label for record in chunk]
        sentence_ids = [record.sentence_id for record in chunk]

        cls_id = getattr(self._tokenizer, "cls_token_id", None)
        sep_id = getattr(self._tokenizer, "sep_token_id", None)
        pad_id = getattr(self._tokenizer, "pad_token_id", 0)
        unk_id = getattr(self._tokenizer, "unk_token_id", None)
        unk_token = getattr(self._tokenizer, "unk_token", None)
        if unk_id is None and unk_token is not None:
            unk_id = self._tokenizer.convert_tokens_to_ids(unk_token)
        if unk_id is None:
            unk_id = 0

        available_length = self._max_length
        special_count = 0
        if cls_id is not None:
            special_count += 1
        if sep_id is not None:
            special_count += 1
        available_length = max(0, available_length - special_count)

        tokens = tokens[:available_length]
        labels = labels[:available_length]
        sentence_ids = sentence_ids[:available_length]

        token_ids = self._convert_tokens_to_ids(tokens, unk_id)

        input_ids = []
        label_ids = []
        attention_mask = []
        sentence_ids_padded = []

        self._add_special_tokens(cls_id, input_ids, label_ids, attention_mask, sentence_ids_padded)

        input_ids.extend(token_ids)
        label_ids.extend(labels)
        attention_mask.extend([1] * len(token_ids))
        sentence_ids_padded.extend(sentence_ids)

        self._add_special_tokens(sep_id, input_ids, label_ids, attention_mask, sentence_ids_padded)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
            "sentence_ids": torch.tensor(sentence_ids_padded, dtype=torch.long),
            "pad_token_id": pad_id,
        }

    def _convert_tokens_to_ids(self, tokens: List[str], unk_id: int) -> List[int]:
        token_ids = []
        for token in tokens:
            if token in self._vocab_map:
                token_id = self._vocab_map[token]
            else:
                token_id = self._tokenizer.convert_tokens_to_ids(token)
            if token_id is None:
                token_id = unk_id
            token_ids.append(token_id)
        return token_ids

    def _add_special_tokens(self, special_id: Optional[int], input_ids: List[int], label_ids: List[int], attention_mask: List[int], sentence_ids_padded: List[int]):
        if special_id is not None:
            input_ids.append(special_id)
            label_ids.append(-100)
            attention_mask.append(1)
            sentence_ids_padded.append(-100)

def build_collate_fn(pad_token_id: int):
    def _collate(batch):
        input_ids = pad_sequence(
            [item["input_ids"] for item in batch], batch_first=True, padding_value=pad_token_id
        )
        attention_mask = pad_sequence(
            [item["attention_mask"] for item in batch], batch_first=True, padding_value=0
        )
        labels = pad_sequence(
            [item["labels"] for item in batch], batch_first=True, padding_value=-100
        )
        sentence_ids = pad_sequence(
            [item["sentence_ids"] for item in batch], batch_first=True, padding_value=-100
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "sentence_ids": sentence_ids,
        }

    return _collate

def create_data_loader(
    data: Path,
    tokenizer,
    *,
    vocab_map: Optional[Dict[str, int]] = None,
    batch_size: int = 8,
    shuffle: bool = True,
    max_length: int = 256,
) -> DataLoader:
    """
    Create a DataLoader from either:
      - a TSV file path containing token data, or
      - preprocessed chunks of TokenRecord lists.
    """
   
    rows = read_token_rows(data)
    chunks = list(group_token_records(rows))
   

    dataset = PropagandaTokenDataset(chunks, tokenizer, max_length=max_length, vocab_map=vocab_map)
    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    collate_fn = build_collate_fn(pad_token_id)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)