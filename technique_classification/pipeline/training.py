"""Training utilities for technique classification."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from ..utils.runtime import DEVICE, LOGGER


def train_classifier(
    model,
    train_dataset: TensorDataset,
    eval_dataset: Optional[TensorDataset],
    *,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.0,
    num_epochs: int = 3,
    train_batch_size: int = 16,
    eval_batch_size: int = 32,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.1,
    use_length: bool = False,
    use_matchings: bool = False,
    patience: int = 3,
) -> Dict[str, float]:
    """Simple training loop with optional dev evaluation and early best checkpoint in memory."""

    train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

    t_total = len(train_loader) * max(1, num_epochs)
    warmup_steps = int(t_total * warmup_ratio)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total
    )

    best_state = model.state_dict()
    best_eval_f1 = 0.0
    epochs_without_improvement = 0

    LOGGER.info("***** Training *****")
    print("***** Training *****")
    LOGGER.info("  Num epochs = %d", num_epochs)
    print(f"  Num epochs = {num_epochs}")
    LOGGER.info("  Train batch size = %d", train_batch_size)
    print(f"  Train batch size = {train_batch_size}")
    LOGGER.info("  Total optimization steps = %d", t_total)
    print(f"  Total optimization steps = {t_total}")
    LOGGER.info("  Early stopping patience = %d", patience)
    print(f"  Early stopping patience = {patience}")

    global_step = 0
    model.zero_grad()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            batch = tuple(t.to(DEVICE) for t in batch)
            # base positions
            input_ids, attention_mask, token_type_ids, labels = batch[:4]
            lengths = batch[4] if use_length and len(batch) > 4 else None
            matchings = batch[5] if use_matchings and len(batch) > 5 else (batch[4] if use_matchings and len(batch) > 4 and not use_length else None)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
                lengths=lengths,
                matchings=matchings,
            )
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            global_step += 1
            epoch_loss += loss.item()

        LOGGER.info("Epoch %d | loss %.4f", epoch, epoch_loss / max(1, len(train_loader)))
        print(f"Epoch {epoch} | loss {epoch_loss / max(1, len(train_loader)):.4f}")

        if eval_dataset is not None:
            eval_metrics = evaluate_classifier(
                model, eval_dataset, batch_size=eval_batch_size,
                use_length=use_length, use_matchings=use_matchings
            )
            LOGGER.info(
                "Eval epoch %d | precision %.4f | recall %.4f | f1-macro %.4f | f1-micro %.4f",
                epoch,
                eval_metrics["precision"],
                eval_metrics["recall"],
                eval_metrics["f1_macro"],
                eval_metrics["f1_micro"],
            )
            print(f"Eval epoch {epoch} | precision {eval_metrics['precision']:.4f} | recall {eval_metrics['recall']:.4f} | f1-macro {eval_metrics['f1_macro']:.4f} | f1-micro {eval_metrics['f1_micro']:.4f}")

            # Early stopping logic
            if eval_metrics["f1_macro"] > best_eval_f1:
                best_eval_f1 = eval_metrics["f1_macro"]
                best_state = model.state_dict()
                epochs_without_improvement = 0
                LOGGER.info("New best F1: %.4f", best_eval_f1)
                print(f"✓ New best F1: {best_eval_f1:.4f}")
            else:
                epochs_without_improvement += 1
                LOGGER.info("No improvement for %d epoch(s)", epochs_without_improvement)
                print(f"No improvement for {epochs_without_improvement} epoch(s)")

                if epochs_without_improvement >= patience:
                    LOGGER.info("Early stopping triggered after %d epochs", epoch)
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

    model.load_state_dict(best_state)
    return {"best_f1_macro": best_eval_f1}


def evaluate_classifier(
    model,
    dataset: TensorDataset,
    *,
    batch_size: int = 32,
    use_length: bool = False,
    use_matchings: bool = False,
) -> Dict[str, float]:
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    model.eval()
    preds: List[int] = []
    gold: List[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, token_type_ids, labels = batch[:4]
            lengths = batch[4] if use_length and len(batch) > 4 else None
            matchings = batch[5] if use_matchings and len(batch) > 5 else (batch[4] if use_matchings and len(batch) > 4 and not use_length else None)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                lengths=lengths,
                matchings=matchings,
            )
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1)
            preds.extend(batch_preds.cpu().tolist())
            gold.extend(labels.cpu().tolist())

    precision_macro = precision_score(gold, preds, average="macro", zero_division=0) if gold else 0.0
    recall_macro = recall_score(gold, preds, average="macro", zero_division=0) if gold else 0.0
    f1_macro = f1_score(gold, preds, average="macro") if gold else 0.0
    f1_micro = f1_score(gold, preds, average="micro") if gold else 0.0
    return {
        "precision": float(precision_macro),
        "recall": float(recall_macro),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
    }


def predict_classifier(
    model,
    dataset: TensorDataset,
    *,
    batch_size: int = 32,
    use_length: bool = False,
    use_matchings: bool = False,
) -> np.ndarray:
    sampler = SequentialSampler(dataset)
    loader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)

    model.eval()
    all_logits: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting", leave=False):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, attention_mask, token_type_ids = batch[:3]
            has_labels = len(batch) >= 4 and batch[3].dtype == torch.long and batch[3].dim() == 1
            base_idx = 4 if has_labels else 3
            lengths = batch[base_idx] if use_length and len(batch) > base_idx else None
            matchings_idx = base_idx + 1 if use_length else base_idx
            matchings = batch[matchings_idx] if use_matchings and len(batch) > matchings_idx else None
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                lengths=lengths,
                matchings=matchings,
            )
            logits = outputs.logits
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def predict_classifier_probs(
    model,
    dataset: TensorDataset,
    *,
    batch_size: int = 32,
    use_length: bool = False,
    use_matchings: bool = False,
) -> np.ndarray:
    """Return softmax probabilities [N, K] in LABEL_ORDER space."""
    logits = predict_classifier(
        model,
        dataset,
        batch_size=batch_size,
        use_length=use_length,
        use_matchings=use_matchings,
    )
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exps / np.clip(np.sum(exps, axis=1, keepdims=True), 1e-12, None)
