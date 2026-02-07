"""Runtime globals and device selection.

We keep these centralized so training/eval/inference modules can share them.
"""

from __future__ import annotations

import torch

try:
    import torch_xla  # noqa: F401
    import torch_xla.core.xla_model as xm  # noqa: F401
    import torch_xla.distributed.parallel_loader as pl  # noqa: F401

    TPU_AVAILABLE = True
except Exception:
    TPU_AVAILABLE = False


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_device(device: torch.device) -> None:
    global DEVICE
    DEVICE = device
