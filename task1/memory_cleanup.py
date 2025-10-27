from __future__ import annotations

import gc
from pathlib import Path
from typing import Iterable, Optional

import torch


def _to_cpu_and_drop_grads(model: torch.nn.Module) -> None:
    """Move model to CPU and drop all grad tensors in-place."""
    try:
        model.eval()
    except Exception:
        pass
    try:
        model.to("cpu")
    except Exception:
        pass
    try:
        for p in model.parameters():
            p.grad = None
    except Exception:
        pass


def release_model(model: Optional[torch.nn.Module]) -> None:
    """Release a PyTorch model's GPU memory as much as possible.

    Notes:
    - This function operates in-place on the module (moving params to CPU and
      clearing grads). The caller should also drop all references to the model
      afterwards (e.g., `model = None`) to ensure full collection.
    """
    if model is None:
        return
    _to_cpu_and_drop_grads(model)


def release_optimizer(optimizer: Optional[torch.optim.Optimizer]) -> None:
    """Clear optimizer gradients and state to free RAM/VRAM.

    Warning: After this call, the optimizer should not be used again.
    """
    if optimizer is None:
        return
    try:
        optimizer.zero_grad(set_to_none=True)
    except Exception:
        pass
    # Clear moments/state
    try:
        optimizer.state.clear()
    except Exception:
        pass


def release_scheduler(scheduler) -> None:
    """Detach references held by scheduler.

    Most schedulers are lightweight, but we still clear common attributes.
    """
    if scheduler is None:
        return
    for attr in ("optimizer", "_optimizer"):
        if hasattr(scheduler, attr):
            try:
                setattr(scheduler, attr, None)
            except Exception:
                pass


def shutdown_dataloaders(dataloaders: Optional[Iterable]) -> None:
    """Attempt to shut down DataLoader workers to free RAM.

    Works best if loaders were created with persistent_workers=False.
    """
    if not dataloaders:
        return
    for loader in dataloaders:
        # Best-effort: trigger iterator shutdown if present
        try:
            it = getattr(loader, "_iterator", None)
            if it is not None and hasattr(it, "_shutdown_workers"):
                it._shutdown_workers()  # type: ignore[attr-defined]
        except Exception:
            pass


def empty_cuda_cache(sync: bool = True, reset_stats: bool = True) -> None:
    if torch.cuda.is_available():
        try:
            if sync:
                torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        if reset_stats:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass


def cleanup_after_training(
    *,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    dataloaders: Optional[Iterable] = None,
    call_gc: bool = True,
    cuda_sync: bool = True,
    cuda_reset_stats: bool = True,
) -> None:
    """Centralized RAM/VRAM cleanup to call when training/evaluation is done.

    Example usage at end of training:
        cleanup_after_training(model=model, optimizer=optimizer, scheduler=scheduler, dataloaders=[train_loader, val_loader])
        model = None  # drop last reference in caller
    """
    release_model(model)
    release_optimizer(optimizer)
    release_scheduler(scheduler)
    shutdown_dataloaders(dataloaders)
    if call_gc:
        gc.collect()
    empty_cuda_cache(sync=cuda_sync, reset_stats=cuda_reset_stats)


def cleanup_after_inference(*, model: Optional[torch.nn.Module] = None, dataloaders: Optional[Iterable] = None) -> None:
    """Smaller variant for prediction-only scenarios."""
    release_model(model)
    shutdown_dataloaders(dataloaders)
    gc.collect()
    empty_cuda_cache()
