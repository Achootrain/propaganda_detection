from __future__ import annotations

from torch import nn
from torch.optim import SGD, AdamW
from transformers import get_linear_schedule_with_warmup

def prepare_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> SGD:
    """Create an SGD optimizer with weight decay."""

    return SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

def prepare_adamw_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Create an AdamW optimizer with weight decay."""

    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def create_scheduler(optimizer, total_steps: int, warmup_ratio: float = 0.0):
    """Build a linear warmup/decay scheduler tailored to the total training steps."""

    warmup_steps = int(total_steps * warmup_ratio)
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)