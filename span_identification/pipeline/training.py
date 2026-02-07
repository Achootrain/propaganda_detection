from __future__ import annotations

import copy
import logging
import math
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from ..utils import runtime
from ..data.bio_data import (
    create_bio_labeled,
    create_bio_unlabeled,
    group_spans_by_article_ids,
    read_predictions_from_file,
)
from ..data.features import convert_examples_to_features, read_examples_from_file
from ..modeling.model import BertLstmCrf
from ..eval.postprocess import postprocess_spans
from ..eval.scoring import compute_precision_recall_f1
from ..eval.submission import aggregate_article_spans, tokenize_like_bio
from ..data.techniques import attach_technique_labels_to_examples, get_multilabel_technique_map
from ..utils.types import DatasetBundle, TokenSpan

LOGGER = logging.getLogger("pipeline")

if runtime.TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm  # type: ignore
    import torch_xla.distributed.parallel_loader as pl  # type: ignore


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(
    file_path: str,
    mode: str,
    article_ids: List[str],
    tokenizer,
    label_list: Sequence[str],
    max_seq_length: int,
    pad_token_label_id: int,
    pos_label_map: Optional[Dict[str, int]] = None,
    ner_label_map: Optional[Dict[str, int]] = None,
    dep_label_map: Optional[Dict[str, int]] = None,
    token_cache: Optional[Dict[str, List[TokenSpan]]] = None,
    technique_spans_by_article: Optional[Dict[str, List[Tuple[int, int, str]]]] = None,
    technique_labels: Optional[List[str]] = None,
) -> DatasetBundle:
    examples = read_examples_from_file(file_path, mode)

    tech_label_map: Optional[Dict[str, int]] = None
    if technique_labels is not None:
        tech_label_map = get_multilabel_technique_map(technique_labels)

    if technique_spans_by_article is not None and token_cache is not None:
        attach_technique_labels_to_examples(
            examples,
            article_ids=article_ids,
            token_cache=token_cache,
            technique_spans_by_article=technique_spans_by_article,
        )

    features = convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        pad_token_label_id,
        pos_label_map=pos_label_map,
        ner_label_map=ner_label_map,
        dep_label_map=dep_label_map,
        tech_label_map=tech_label_map,
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
    all_ner_ids = torch.tensor([f.ner_ids for f in features], dtype=torch.long)
    all_dep_ids = torch.tensor(
        [f.dep_ids if f.dep_ids else [0] * max_seq_length for f in features],
        dtype=torch.long,
    )
    all_word_ids = torch.tensor([f.word_ids for f in features], dtype=torch.long)

    tech_dim = len(tech_label_map) if tech_label_map else 0
    all_tech_multi_labels = torch.tensor(
        [
            f.tech_multi_labels if f.tech_multi_labels is not None else [[0] * tech_dim] * max_seq_length
            for f in features
        ],
        dtype=torch.float,
    )
    all_tech_label_mask = torch.tensor(
        [f.tech_label_mask if f.tech_label_mask is not None else [0] * max_seq_length for f in features],
        dtype=torch.float,
    )
    all_lexicon_features = torch.tensor(
        [f.lexicon_features if f.lexicon_features is not None else [[0]] * max_seq_length for f in features],
        dtype=torch.float,
    )

    dataset = TensorDataset(
        all_input_ids,
        all_attention_mask,
        all_label_ids,
        all_pos_ids,
        all_ner_ids,
        all_dep_ids,
        all_word_ids,
        all_lexicon_features,
        all_tech_multi_labels,
        all_tech_label_mask,
    )

    return DatasetBundle(examples=examples, features=features, dataset=dataset, article_ids=list(article_ids))


def prepare_bio_files(
    work_dir: str,
    nlp,
    *,
    mode: str,
    articles: Sequence[str],
    article_ids: Sequence[str],
    labels_path: Optional[str] = None,
    token_cache: Optional[Dict[str, List[TokenSpan]]] = None,
    filename: Optional[str] = None,
) -> Tuple[str, List[str]]:
    os.makedirs(work_dir, exist_ok=True)
    if filename is None:
        filename = f"{mode}.bio"
    output_path = os.path.join(work_dir, filename)

    articles_content = dict(zip(article_ids, articles))

    if token_cache is not None:
        for article_id, text in articles_content.items():
            token_cache[article_id] = tokenize_like_bio(text, nlp)

    normalized_mode = mode.lower()
    labeled_modes = {"train", "dev"}
    prediction_modes = {"test", "prediction"}

    if normalized_mode in labeled_modes:
        if labels_path is None:
            raise ValueError("labels_path is required when mode is train or dev")
        article_id_labels, spans = read_predictions_from_file(labels_path)
        grouped_dict = group_spans_by_article_ids(zip(article_id_labels, spans))
        for article_id in article_ids:
            grouped_dict.setdefault(article_id, [])
        ordered = [(article_id, grouped_dict.get(article_id, [])) for article_id in article_ids]
        create_bio_labeled(output_path, ordered, articles_content, nlp)
    elif normalized_mode in prediction_modes:
        create_bio_unlabeled(output_path, article_ids, articles, nlp)
    else:
        raise ValueError(f"Unsupported prepare_bio_files mode: {mode}")

    return output_path, list(article_ids)


def evaluate_model(
    model: BertLstmCrf,
    dataloader: DataLoader,
    label_list: Sequence[str],
    pad_token_label_id: int,
) -> Tuple[Dict[str, float], List[List[str]]]:
    model.eval()
    preds: List[List[str]] = []
    gold: List[List[str]] = []
    losses: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(runtime.DEVICE) for t in batch)
            loss, _, predicted = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                labels=batch[2],
                pos_ids=batch[3],
                ner_ids=batch[4],
                dep_ids=batch[5],
                word_ids=batch[6],
                lexicon_features=batch[7],
                tech_multi_labels=batch[8],
                tech_label_mask=batch[9],
            )
            if loss is not None:
                losses.append(loss.item())
            for idx, sequence in enumerate(predicted):
                label_ids = [lid for lid in batch[2][idx].tolist() if lid != pad_token_label_id]
                gold_labels = [label_list[lid] for lid in label_ids]
                pred_labels = [label_list[tag_id] for tag_id in sequence[: len(label_ids)]]
                gold.append(gold_labels)
                preds.append(pred_labels)

    precision = precision_score(gold, preds) if gold else 0.0
    recall = recall_score(gold, preds) if gold else 0.0
    f1 = f1_score(gold, preds) if gold else 0.0
    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    return metrics, preds


def predict_model(
    model: BertLstmCrf,
    bundle: DatasetBundle,
    label_list: Sequence[str],
    pad_token_label_id: int,
    batch_size: int,
    num_workers: int = 0,
) -> List[List[str]]:
    dataloader = DataLoader(
        bundle.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(runtime.DEVICE) != "cpu" and not runtime.TPU_AVAILABLE,
    )
    model.eval()
    predictions: List[List[str]] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(runtime.DEVICE) for t in batch)
            _, _, predicted = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                labels=batch[2],
                pos_ids=batch[3],
                ner_ids=batch[4],
                dep_ids=batch[5],
                word_ids=batch[6],
                lexicon_features=batch[7],
                tech_multi_labels=batch[8],
                tech_label_mask=batch[9],
            )
            for idx, sequence in enumerate(predicted):
                label_ids = [lid for lid in batch[2][idx].tolist() if lid != pad_token_label_id]
                predictions.append([label_list[tag_id] for tag_id in sequence[: len(label_ids)]])
    return predictions


def train_model(
    model: BertLstmCrf,
    train_bundle: DatasetBundle,
    dev_bundle: DatasetBundle,
    label_list: Sequence[str],
    pad_token_label_id: int,
    train_batch_size: int,
    eval_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    num_epochs: int,
    warmup_steps: int,
    max_grad_norm: float,
    patience: int = 2,
    accum_steps: int = 1,
    *,
    freeze_encoder_epochs: int = 2,
    save_checkpoint_epochs: int = 0,
    checkpoint_dir: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
    dev_gold_spans: Optional[Dict[str, List[Tuple[int, int]]]] = None,
    token_cache: Optional[Dict[str, List[TokenSpan]]] = None,
    dev_article_texts: Optional[Dict[str, str]] = None,
    apply_postprocess: bool = True,
    num_workers: int = 0,
) -> Dict[str, float]:
    train_loader = DataLoader(
        train_bundle.dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=str(runtime.DEVICE) != "cpu" and not runtime.TPU_AVAILABLE,
    )
    dev_loader = DataLoader(
        dev_bundle.dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(runtime.DEVICE) != "cpu" and not runtime.TPU_AVAILABLE,
    )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, math.ceil(len(train_loader) * num_epochs / max(1, accum_steps)))

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    optimizer.zero_grad()
    wait = 0
    global_step = 0
    start_epoch = 1

    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=runtime.DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)
        best_metrics = checkpoint.get("best_metrics", best_metrics)
        best_state = checkpoint.get("best_state", best_state)
        wait = checkpoint.get("wait", 0)
        print(f"Resumed from epoch {checkpoint['epoch']}, best F1: {best_metrics['f1']:.4f}")

    if freeze_encoder_epochs > 0:
        for param in model.bert_encoder.parameters():
            param.requires_grad = False
        print(f"PLM (bert_encoder) frozen for first {freeze_encoder_epochs} epochs")

    for epoch in range(start_epoch, num_epochs + 1):
        if freeze_encoder_epochs > 0 and epoch == freeze_encoder_epochs + 1:
            for param in model.bert_encoder.parameters():
                param.requires_grad = True
            print(f"PLM (bert_encoder) unfrozen at epoch {epoch}")

        model.train()
        epoch_loss = 0.0

        iteration_loader = train_loader
        if runtime.TPU_AVAILABLE and runtime.DEVICE.type == "xla":
            iteration_loader = pl.ParallelLoader(train_loader, [runtime.DEVICE]).per_device_loader(runtime.DEVICE)

        remainder = len(train_loader) % accum_steps if accum_steps > 0 else 0

        for step, batch in enumerate(
            tqdm(
                iteration_loader,
                desc=f"Epoch {epoch}/{num_epochs}",
                mininterval=60,
                disable=runtime.TPU_AVAILABLE,
            )
        ):
            if not (runtime.TPU_AVAILABLE and runtime.DEVICE.type == "xla"):
                batch = tuple(t.to(runtime.DEVICE) for t in batch)

            loss, _, _ = model(
                input_ids=batch[0],
                attention_mask=batch[1],
                labels=batch[2],
                pos_ids=batch[3],
                ner_ids=batch[4],
                dep_ids=batch[5],
                word_ids=batch[6],
                lexicon_features=batch[7],
                tech_multi_labels=batch[8],
                tech_label_mask=batch[9],
            )
            if loss is None:
                continue

            if accum_steps <= 0:
                group_size = 1
            else:
                if remainder and step >= len(train_loader) - remainder:
                    group_size = remainder
                else:
                    group_size = accum_steps

            scaled_loss = loss / group_size
            scaled_loss.backward()
            epoch_loss += scaled_loss.item()

            should_step = False
            if accum_steps <= 1:
                should_step = True
            else:
                if (step + 1) % accum_steps == 0:
                    should_step = True
                if (step + 1) == len(train_loader):
                    should_step = True

            if should_step:
                if runtime.TPU_AVAILABLE and runtime.DEVICE.type == "xla":
                    xm.optimizer_step(optimizer)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

        print(f"Epoch {epoch} training loss: {epoch_loss / max(1, len(train_loader)):.4f}")

        metrics, dev_sentence_predictions = evaluate_model(model, dev_loader, label_list, pad_token_label_id)
        print(
            f"Dev token-level metrics after epoch {epoch} - F1: {metrics['f1']:.4f} | Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f}"
        )

        span_metrics = {"f1": 0.0, "precision": 0.0, "recall": 0.0}
        if dev_gold_spans is not None and token_cache is not None:
            try:
                dev_spans = aggregate_article_spans(dev_sentence_predictions, dev_bundle.article_ids, token_cache)
                if apply_postprocess and dev_article_texts is not None:
                    dev_spans = postprocess_spans(dev_spans, dev_article_texts)

                span_metrics = compute_precision_recall_f1(dev_spans, dev_gold_spans)
                print(
                    f"Dev span-level metrics after epoch {epoch} - F1: {span_metrics['f1']:.4f} | Precision: {span_metrics['precision']:.4f} | Recall: {span_metrics['recall']:.4f}"
                )
            except Exception:
                LOGGER.exception("Span-level evaluation failed for epoch %d", epoch)

        if span_metrics["f1"] > best_metrics["f1"]:
            best_metrics = span_metrics
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
            print(f"New best span F1: {span_metrics['f1']:.4f} at epoch {epoch}")
        else:
            wait += 1
            if wait == patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

        if save_checkpoint_epochs > 0 and epoch % save_checkpoint_epochs == 0:
            ckpt_dir = checkpoint_dir if checkpoint_dir else "."
            ckpt_path = os.path.join(ckpt_dir, f"checkpoint_epoch_{epoch}.pt")
            checkpoint_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "global_step": global_step,
                "best_metrics": best_metrics,
                "best_state": best_state,
                "wait": wait,
            }
            torch.save(checkpoint_data, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")

    model.load_state_dict(best_state)
    return best_metrics
