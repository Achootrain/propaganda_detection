"""Custom RoBERTa classification heads."""

from __future__ import annotations

import torch
import torch.nn as nn


class RobertaClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaClassificationHeadLength(nn.Module):
    """Classification head with length feature concatenation."""

    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size + 1, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, sent_a_length=None):
        x = features[:, 0, :]
        x = torch.cat((x, sent_a_length), dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaClassificationHeadMatchings(nn.Module):
    """Classification head with matching features concatenation."""

    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size + 14, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, matchings=None):
        x = features[:, 0, :]
        x = torch.cat((x, matchings), dim=1)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaClassificationHeadJoined(nn.Module):
    """Classification head with CLS + pooled token embeddings."""

    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, attention_mask=None):
        x = features[:, 0, :]
        # mimic modeling_roberta: mask-based pooled embs over non-CLS tokens
        if attention_mask is not None:
            mask = attention_mask.reshape(features.shape[0], features.shape[1], 1).float()
            denom = mask.sum(dim=1).clamp(min=1.0)
            embs = (features * mask)[:, 1:, :].sum(dim=1) / denom
        else:
            embs = features[:, 1:, :].mean(dim=1)
        x = torch.cat((x, embs), dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaClassificationHeadJoinedLength(nn.Module):
    """Classification head with CLS + pooled embeddings + length."""

    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 2 + 1, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, sent_a_length=None, attention_mask=None):
        x = features[:, 0, :]
        # concat length first
        x = torch.cat((x, sent_a_length), dim=1)
        # pooled embs over non-CLS tokens (mean if no mask)
        if attention_mask is not None:
            mask = attention_mask.reshape(features.shape[0], features.shape[1], 1).float()
            embs = (features * mask)[:, 1:, :].sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            embs = features[:, 1:, :].mean(dim=1)
        x = torch.cat((x, embs), dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class RobertaClassificationHeadJoinedLengthMatchings(nn.Module):
    """CLS + pooled tokens + length scalar + matchings vector + lexicon scalar."""

    def __init__(self, hidden_size: int, num_labels: int, dropout_prob: float = 0.1):
        super().__init__()
        # hidden_size (CLS) + hidden_size (pooled non-CLS) + 1 (length) + 14 (matchings) + 1 (lexicon)
        self.dense = nn.Linear(hidden_size * 2 + 1 + 14 + 1, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features, sent_a_length=None, matchings=None, lexicon=None, attention_mask=None):
        x = features[:, 0, :]
        # Concatenate all scalar/vector features
        concat_list = [x]
        if sent_a_length is not None:
            concat_list.append(sent_a_length)
        else:
            concat_list.append(torch.zeros(x.size(0), 1, device=x.device))
        if matchings is not None:
            concat_list.append(matchings)
        else:
            concat_list.append(torch.zeros(x.size(0), 14, device=x.device))
        if lexicon is not None:
            concat_list.append(lexicon)
        else:
            concat_list.append(torch.zeros(x.size(0), 1, device=x.device))
        x = torch.cat(concat_list, dim=1)
        # pooled embs over non-CLS tokens (mean if no mask)
        if attention_mask is not None:
            mask = attention_mask.reshape(features.shape[0], features.shape[1], 1).float()
            embs = (features * mask)[:, 1:, :].sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            embs = features[:, 1:, :].mean(dim=1)
        x = torch.cat((x, embs), dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
