"""Runtime globals and device selection."""

from __future__ import annotations

import logging
import random

import numpy as np
import torch

LOGGER = logging.getLogger("technique_tc")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(device: torch.device) -> None:
    global DEVICE
    DEVICE = device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
