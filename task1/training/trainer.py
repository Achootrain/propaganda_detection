from __future__ import annotations
import math
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from transformers import AutoTokenizer
from ..process_data.utils import load_tokenizer
from .eval import calculate_span_metrics ,generate_evaluation_debug_report 

from ..model.utils import create_scheduler, prepare_optimizer, prepare_adamw_optimizer
from ..model.crf_model import TokenCRFModel
from ..model.baseline import RobertaLargeBaseline
from ..multigranularity.multi_granularity import MultiGranularityTokenCRFModel
"""
Span metrics and debug-report helpers are imported from process_data.utils.
"""
import os

@dataclass
class TrainConfig:
    # Paths are no longer required since loaders are provided externally.
    tokens_path: Optional[Path] = None
    vocab_path: Optional[Path] = None
    model_name: str ="roberta-large"
    model_type: str = "crf"
    dev_tokens_path: Optional[Path] = None
    checkpoint_path: Optional[Path] = None
    debug_output_path: Optional[Path] = None
    num_labels: int = 3
    batch_size: int = 8
    max_length: int = 256
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    momentum: float = 0.9
    optimizer_type: str = "adamw"
    epochs: int = 10
    validation_split: float = 0.1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_train_batches: Optional[int] = None
    max_eval_batches: Optional[int] = None
    dropout: float = 0.1
    aux_ce_weight: float = 0.3
    sentence_loss_weight: float = 0.5
    label_smoothing: float = 0.1
    early_stopping_patience: int = 2
    class_weights: Optional[Tuple[float, ...]] = None
    # Multi-granularity model specific knobs
    gate_function: str = "sigmoid"
    apply_gate_in_inference: bool = True
    use_crf: bool = True
    crf_loss_weight: float = 0.5
    enforce_bio_constraints: bool = True
    use_focal_loss: bool = True
    focal_alpha: float = 0.75
    focal_gamma: float = 3.0
    focal_warmup_steps: int = 3
    focal_warmup_method: str = "linear"

def train_one_epoch(
    model: torch.nn.Module,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    max_batches: Optional[int] = None,
    max_grad_norm: float = 1.0,
) -> float:
    """Run a single training epoch and return the average loss."""
    model.train()
    total_loss = 0.0
    steps = 0
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        # Filter inputs to match model signature unless it supports **kwargs
        fwd_sig = inspect.signature(model.forward)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in fwd_sig.parameters.values())
        if has_var_kw:
            model_inputs = batch
        else:
            allowed = set(fwd_sig.parameters.keys())
            model_inputs = {k: v for k, v in batch.items() if k in allowed}
        outputs = model(**model_inputs)
        loss = outputs["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item()
        steps += 1
    return total_loss / max(1, steps)

def evaluate_model(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    tokenizer: AutoTokenizer,
    max_batches: Optional[int] = None,
    generate_debug_report: bool = False,
) -> Dict[str, Any]:
    """Compute loss and span-level precision/recall scores mirroring the official scorer."""
    model.eval()
    total_loss = 0.0
    steps = 0
    total_cumulative_prec = 0.0
    total_cumulative_rec = 0.0
    total_pred_span_count = 0
    total_gold_span_count = 0
    total_pred_malformed = 0
    total_gold_malformed = 0
    debug_report: List[str] = []
    print("\n--- Starting Validation ---")
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_batches is not None and step >= max_batches:
                break
            batch_on_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            fwd_sig = inspect.signature(model.forward)
            has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in fwd_sig.parameters.values())
            if has_var_kw:
                model_inputs = batch_on_device
            else:
                allowed = set(fwd_sig.parameters.keys())
                model_inputs = {k: v for k, v in batch_on_device.items() if k in allowed}
            outputs = model(**model_inputs)
            loss = outputs.get("loss")
            if loss is not None:
                total_loss += loss.item()
            labels = batch_on_device["labels"]
            mask = labels != -100
            predictions = outputs["predictions"]
            metrics_batch = calculate_span_metrics(predictions, labels, mask)
            total_cumulative_prec += metrics_batch["cumulative_prec"]
            total_cumulative_rec += metrics_batch["cumulative_rec"]
            total_pred_span_count += metrics_batch["pred_span_count"]
            total_gold_span_count += metrics_batch["gold_span_count"]
            total_pred_malformed += metrics_batch["pred_malformed"]
            total_gold_malformed += metrics_batch["gold_malformed"]
            if generate_debug_report:
                debug_report.extend(generate_evaluation_debug_report(step, batch_on_device, outputs, labels, mask, predictions, tokenizer))
            steps += 1
    avg_loss = total_loss / max(1, steps)
    # Calculate overall precision, recall, and F1
    precision = total_cumulative_prec / max(1, total_pred_span_count)
    recall = total_cumulative_rec / max(1, total_gold_span_count)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"Validation complete. Loss: {avg_loss:.4f}, P: {precision:.4f}, R: {recall:.4f}, F1: {f1:.4f}")
    if total_pred_malformed > 0 or total_gold_malformed > 0:
        print(f"  Malformed predictions: {total_pred_malformed}, Malformed gold: {total_gold_malformed}")
    results = {
        "loss": avg_loss,
        "span_precision": precision,
        "span_recall": recall,
        "span_f1": f1,
    }
    if generate_debug_report:
        results["debug_report"] = debug_report
    return results

"""
DataLoader creation moved to process_data/dataloader.py (create_data_loaders).
"""

def _initialize_model(config: TrainConfig, device: torch.device) -> torch.nn.Module:
    """Instantiate the selected model type and move to device."""
    model_type = config.model_type.lower()
    if model_type in {"crf", "token_crf"}:
        model = TokenCRFModel(
            model_name=config.model_name,
            num_labels=config.num_labels,
            dropout=config.dropout,
            aux_ce_weight=config.aux_ce_weight,
            label_smoothing=config.label_smoothing,
            class_weights=config.class_weights,
        )
    elif model_type in {"multi_granularity", "multi-granularity"}:
        model = MultiGranularityTokenCRFModel(
            model_name=config.model_name,
            num_labels=config.num_labels,
            dropout=config.dropout,
            sentence_loss_weight=config.sentence_loss_weight,
            label_smoothing=config.label_smoothing,
            class_weights=config.class_weights,
            gate_function=config.gate_function,
            apply_gate_in_inference=config.apply_gate_in_inference,
            use_crf=config.use_crf,
            crf_loss_weight=config.crf_loss_weight,
            enforce_bio_constraints=config.enforce_bio_constraints,
            use_focal_loss=config.use_focal_loss,
            focal_alpha=config.focal_alpha,
            focal_gamma=config.focal_gamma,
            focal_warmup_steps=config.focal_warmup_steps,
            focal_warmup_method=config.focal_warmup_method,
        )
    elif model_type in {"baseline", "roberta_baseline", "roberta-baseline"}:
        model = RobertaLargeBaseline(
            model_name=config.model_name,
            num_labels=config.num_labels,
            dropout=config.dropout,
            label_smoothing=config.label_smoothing,
            class_weights=config.class_weights,
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    model.to(device)
    return model

def _setup_optimizer_and_scheduler(model, config: TrainConfig, total_steps: int):
    if config.optimizer_type.lower() == "adamw":
        optimizer = prepare_adamw_optimizer(model, learning_rate=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer_type.lower() == "sgd":
        optimizer = prepare_optimizer(model, learning_rate=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)
    else:
        raise ValueError(f"Unsupported optimizer_type: {config.optimizer_type}")
    scheduler = create_scheduler(optimizer, total_steps=total_steps, warmup_ratio=config.warmup_ratio)
    return optimizer, scheduler

def _load_checkpoint(
    config: TrainConfig,
    model: torch.nn.Module,
    optimizer,
    scheduler,
    device: torch.device,
    start_epoch: int,
    best_val_f1: float,
    best_val_loss: float,
):
    if config.checkpoint_path and isinstance(config.checkpoint_path, Path) and config.checkpoint_path.exists():
        state = torch.load(config.checkpoint_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", {}))
        if "optimizer_state_dict" in state:
            optimizer.load_state_dict(state["optimizer_state_dict"]) 
        if "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"]) 
        start_epoch = int(state.get("epoch", start_epoch)) + 1
        best_val_f1 = float(state.get("best_val_f1", best_val_f1))
        best_val_loss = float(state.get("best_val_loss", best_val_loss))
        print(f"Loaded checkpoint from {config.checkpoint_path}")
    return start_epoch, best_val_f1, best_val_loss

"""
Span metrics and debug-report helpers moved to training/utils.py.
"""

def train_model(
    config: TrainConfig,
    model_output_path: Path,
    *,
    train_loader=None,
    val_loader=None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> None:
    """Orchestrate dataloading, training, and evaluation for the token model."""
    device = torch.device(config.device)
    # Initialization: tokenizer (for debug reporting) if not provided
    if tokenizer is None:
        tokenizer = load_tokenizer(config.model_name)
    # Require externally provided loaders
    if train_loader is None or val_loader is None:
        raise ValueError("train_loader and val_loader must be provided to train_model.")
    model = _initialize_model(config, device)
    total_steps = len(train_loader) * config.epochs
    optimizer, scheduler = _setup_optimizer_and_scheduler(model, config, total_steps)
    start_epoch = 1
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch, best_val_f1, best_val_loss = _load_checkpoint(config, model, optimizer, scheduler, device, start_epoch, best_val_f1, best_val_loss)
    for epoch in range(start_epoch, config.epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            max_batches=config.max_train_batches,
            max_grad_norm=config.max_grad_norm,
        )
        metrics = evaluate_model(
            model,
            val_loader,
            device,
            tokenizer=tokenizer,
            max_batches=config.max_eval_batches,
            generate_debug_report=True,
        )
        print(
            f"Epoch {epoch}/{config.epochs} - "
            f"train_loss: {train_loss:.4f} - "
            f"val_loss: {metrics['loss']:.4f} - "
            f"span_p: {metrics['span_precision']:.4f} - "
            f"span_r: {metrics['span_recall']:.4f} - "
            f"span_f1: {metrics['span_f1']:.4f}"
        )
        if metrics["span_f1"] > best_val_f1:
            best_val_f1 = metrics["span_f1"]
            best_state = {
                "epoch": epoch,
                "model_state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_val_f1": best_val_f1,
                "best_val_loss": best_val_loss,
            }
            print(f"New best F1 score: {best_val_f1:.4f}. Saving model state.")
            if model_output_path:
                torch.save(best_state, model_output_path)
            if config.debug_output_path and "debug_report" in metrics:
                with open(config.debug_output_path, "w") as f:
                    for line in metrics["debug_report"]:
                        f.write(line + "\n")
                print(f"Saved debug report to {config.debug_output_path}")
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            epochs_without_improvement = 0
            best_epoch = epoch



def run_training(
    train_loader,
    val_loader,
    model_name: str,
    model_type: str = "crf",
    checkpoint_path: Optional[Path] = None,
    model_output_path: Optional[Path] = None,
    debug_output_path: Optional[Path] = None,
    # data & batching
    batch_size: int = 8,
    max_length: int = 256,
    num_labels: int = 3,
    # training schedule
    epochs: int = 60,
    learning_rate: float = 3e-5,
    optimizer_type: str = "adamw",
    weight_decay: float = 0.01,
    momentum: float = 0.9,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    max_train_batches: Optional[int] = None,
    max_eval_batches: Optional[int] = None,
    device: Optional[str] = None,
    # model regularization / losses
    dropout: float = 0.1,
    aux_ce_weight: float = 0.3,
    label_smoothing: float = 0.1,
    # multi-granularity model knobs
    sentence_loss_weight: float = 0.8,
    gate_function: str = "sigmoid",
    apply_gate_in_inference: bool = True,
    use_crf: bool = True,
    crf_loss_weight: float = 0.8,
    enforce_bio_constraints: bool = True,
    use_focal_loss: bool = True,
    focal_alpha: float = 0.35,
    focal_gamma: float = 2.0,
    focal_warmup_steps: int = 0,
    focal_warmup_method: str = "linear",
    # training controls
    early_stopping_patience: int = 2,
    class_weights: Optional[Tuple[float, ...]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
) -> None:
    """Entry point to configure and execute token-level training in one call."""
    config = TrainConfig(
        checkpoint_path=checkpoint_path,
        debug_output_path=debug_output_path,
        model_name=model_name,
        model_type=model_type,
        num_labels=num_labels,
        batch_size=batch_size,
        max_length=max_length,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        optimizer_type=optimizer_type,
        warmup_ratio=warmup_ratio,
        max_grad_norm=max_grad_norm,
        device=device or ("cuda" if torch.cuda.is_available() else "cpu"),
        max_train_batches=max_train_batches,
        max_eval_batches=max_eval_batches,
        dropout=dropout,
        aux_ce_weight=aux_ce_weight,
        sentence_loss_weight=sentence_loss_weight,
        label_smoothing=label_smoothing,
        early_stopping_patience=early_stopping_patience,
        class_weights=class_weights,
        gate_function=gate_function,
        apply_gate_in_inference=apply_gate_in_inference,
        use_crf=use_crf,
        crf_loss_weight=crf_loss_weight,
        enforce_bio_constraints=enforce_bio_constraints,
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        focal_warmup_steps=focal_warmup_steps,
        focal_warmup_method=focal_warmup_method,
    )
    train_model(
        config,
        model_output_path,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
    )