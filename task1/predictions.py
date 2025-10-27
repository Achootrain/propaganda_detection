from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer

from task1.process_data.labeling import create_token_labels
from task1.process_data.dataloader import PropagandaTokenDataset, build_collate_fn
from task1.process_data.utils import (
    TokenRecord,
    read_token_rows,
    group_token_records,
    parse_span,
    resolve_article_path,
    load_article_text,
)
from task1.model.crf_model import TokenCRFModel
from task1.multigranularity.multi_granularity import MultiGranularityTokenCRFModel
from task1.scorer import compute_span_prf, load_labels_file


def _init_model(
    model_type: str,
    model_name: str,
    num_labels: int = 3,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mt = model_type.lower()
    if mt in {"crf", "token_crf"}:
        model = TokenCRFModel(model_name=model_name, num_labels=num_labels)
    elif mt in {"multi_granularity", "multi-granularity"}:
        model = MultiGranularityTokenCRFModel(
            model_name=model_name,
            num_labels=num_labels,
            # keep defaults aligned with training
            use_crf=True,
            enforce_bio_constraints=True,
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    model.to(device)
    model.eval()
    return model


def _load_checkpoint_if_any(model: torch.nn.Module, checkpoint_path: Optional[Path], device: torch.device) -> None:
    if checkpoint_path is None:
        return
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state.get("model_state_dict", state))


def _batch_indices(total: int, batch_size: int) -> List[List[int]]:
    return [list(range(i, min(i + batch_size, total))) for i in range(0, total, batch_size)]


def _pred_labels_to_token_spans(pred_labels: List[int]) -> List[Tuple[int, int]]:
    """Convert BIO labels to (start_token_index, end_token_index) spans, end exclusive."""
    spans: List[Tuple[int, int]] = []
    start_idx: Optional[int] = None
    for i, tag in enumerate(pred_labels):
        if tag == 1:  # B
            if start_idx is not None and start_idx < i:
                spans.append((start_idx, i))
            start_idx = i
        elif tag == 2:  # I
            pass
        else:  # O
            if start_idx is not None and start_idx < i:
                spans.append((start_idx, i))
            start_idx = None
    if start_idx is not None and start_idx < len(pred_labels):
        spans.append((start_idx, len(pred_labels)))
    return spans


def _tokenspan_to_charspan(
    s_tok: int,
    e_tok: int,
    records: List[TokenRecord],
) -> Optional[Tuple[int, int]]:
    s_tok = max(0, min(s_tok, len(records) - 1))
    e_tok = max(s_tok + 1, min(e_tok, len(records)))
    start_span = parse_span(records[s_tok].span)
    end_span = parse_span(records[e_tok - 1].span)
    if not start_span or not end_span:
        return None
    start_char = start_span[0]
    end_char = end_span[1]
    if end_char <= start_char:
        return None
    return (start_char, end_char)


def _post_process_char_span(
    start_char: int,
    end_char: int,
    s_tok: int,
    e_tok: int,
    records: List[TokenRecord],
    tokenizer: AutoTokenizer,
    article_text: str,
) -> Tuple[int, int]:
    """Adjust span to better align with RoBERTa subword boundaries.
    - If the first token is a continuation subword (no 'Ġ' prefix), expand span backward by 1 char.
    - If the last character inside the span is not alphanumeric (e.g., whitespace/punct), shrink end by 1.
    Bounds are clamped and we ensure end > start.
    """
    try:
        first_token = records[s_tok].token
    except Exception:
        first_token = ""

    # RoBERTa BPE uses 'Ġ' to mark word starts; lack of it indicates a subword continuation
    if first_token and not first_token.startswith("Ġ") and start_char > 0:
        start_char = max(0, start_char - 1)

    # If end falls on a non-word char inside the span, pull it back by one
    if end_char > start_char:
        last_idx = end_char - 1
        if 0 <= last_idx < len(article_text):
            ch = article_text[last_idx]
            if not ch.isalnum():
                end_char = max(start_char + 1, end_char - 1)

    return (start_char, end_char)


def _merge_overlapping_spans(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge overlapping or touching character spans.
    Assumes spans are (start, end) with end exclusive.
    """
    if not spans:
        return []
    spans_sorted = sorted(spans)
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = spans_sorted[0]
    for s, e in spans_sorted[1:]:
        if s <= cur_e:  # overlap or touch
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def run_predictions(
    *,
    test_articles_dir: Path,
    test_gold_spans: Optional[Path],
    tokens_output: Path,
    predictions_output: Path,
    model_name: str = "roberta-large",
    model_type: str = "multi_granularity",
    checkpoint_path: Optional[Path] = None,
    batch_size: int = 8,
    max_length: int = 256,
) -> Dict[str, float]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # 1) Ensure tokens TSV exists for test (uses gold spans only to fill label column; model inference ignores them)
    create_token_labels(
        articles_dir=test_articles_dir,
        spans_file=test_gold_spans,
        tokens_output=tokens_output,
        tokenizer=tokenizer,
        model=None,  # not used inside create_token_labels
        device=device,
        max_tokens=512,
        stride=128,
    )

    # 2) Prepare chunks and dataset (keep the exact chunk list for offset mapping)
    rows_iter = read_token_rows(tokens_output)
    chunks: List[List[TokenRecord]] = list(group_token_records(rows_iter))
    dataset = PropagandaTokenDataset(chunks, tokenizer, max_length=max_length, vocab_map=None)
    collate_fn = build_collate_fn(getattr(tokenizer, "pad_token_id", 0))

    # 3) Initialize and load model
    model = _init_model(model_type, model_name, num_labels=3, device=device)
    _load_checkpoint_if_any(model, checkpoint_path, device)

    # 4) Inference in simple index-batches to preserve mapping to chunks
    has_cls = getattr(tokenizer, "cls_token_id", None) is not None
    has_sep = getattr(tokenizer, "sep_token_id", None) is not None
    special_count = (1 if has_cls else 0) + (1 if has_sep else 0)
    available_len = max_length - special_count

    predictions_by_article: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    article_text_cache: Dict[str, str] = {}

    for batch_indices in _batch_indices(len(dataset), batch_size):
        batch_items = [dataset[i] for i in batch_indices]
        batch = collate_fn(batch_items)
        batch_on_device = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            outputs = model(
                input_ids=batch_on_device["input_ids"],
                attention_mask=batch_on_device["attention_mask"],
                sentence_ids=batch_on_device.get("sentence_ids"),
            )
        preds = outputs["predictions"]  # List[List[int]] or tensor
        if isinstance(preds, torch.Tensor):
            preds = preds.cpu().tolist()

        for local_i, pred_seq in enumerate(preds):
            global_i = batch_indices[local_i]
            # Trim CLS/SEP predictions to align with records
            seq = list(map(int, pred_seq))
            if has_cls and len(seq) > 0:
                seq = seq[1:]
            if has_sep and len(seq) > 0:
                seq = seq[:-1]

            records = chunks[global_i]
            recs_trunc = records[:available_len] if available_len > 0 else []
            seq = seq[: len(recs_trunc)]
            if not seq or not recs_trunc:
                continue
            token_spans = _pred_labels_to_token_spans(seq)
            if not token_spans:
                continue
            article_id = recs_trunc[0].article_id
            # Load article text once
            if article_id not in article_text_cache:
                apath = resolve_article_path(article_id, test_articles_dir)
                article_text_cache[article_id] = load_article_text(apath)
            atext = article_text_cache[article_id]

            for s_tok, e_tok in token_spans:
                char_span = _tokenspan_to_charspan(s_tok, e_tok, recs_trunc)
                if not char_span:
                    continue
                s_char, e_char = char_span
                s_char, e_char = _post_process_char_span(s_char, e_char, s_tok, e_tok, recs_trunc, tokenizer, atext)
                if e_char > s_char:
                    predictions_by_article[article_id].append((s_char, e_char))

    # Optional: sort and deduplicate spans per article
    for aid in list(predictions_by_article.keys()):
        # Deduplicate and merge overlaps
        spans = list(set(predictions_by_article[aid]))
        predictions_by_article[aid] = _merge_overlapping_spans(spans)

    # 5) Write predictions in Task SI .labels format
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    with predictions_output.open("w", encoding="utf8") as out:
        def _aid_key(a: str) -> Tuple[int, str]:
            try:
                return (0, str(int(a)))
            except Exception:
                return (1, a)
        for aid in sorted(predictions_by_article.keys(), key=_aid_key):
            for s, e in sorted(predictions_by_article[aid]):
                out.write(f"{aid}\t{s}\t{e}\n")

    # 6) Evaluate using local scorer (logic aligned with official scorer)
    results = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    if test_gold_spans and Path(test_gold_spans).exists():
        try:
            pred_map = load_labels_file(predictions_output)
            gold_map = load_labels_file(Path(test_gold_spans))
            p, r, f1 = compute_span_prf(pred_map, gold_map)
            results = {"precision": p, "recall": r, "f1": f1}
        except Exception as e:
            print(f"Warning: scorer evaluation failed: {e}")

    return results


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    datasets_dir = root / "datasets"
    task1_data_dir = root / "task1" / "data"
    p = argparse.ArgumentParser(description="Run Task1 predictions on the test set and evaluate with the official scorer.")
    p.add_argument("--model-name", type=str, default="roberta-large", help="HF model name")
    p.add_argument("--model-type", type=str, default="multi_granularity", help="Model type: multi_granularity or crf")
    p.add_argument("--checkpoint", type=str, default=None, help="Path to a trained model checkpoint (state dict)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--test-articles-dir", type=str, default=str(datasets_dir / "test-articles"))
    p.add_argument("--test-gold-labels", type=str, default=str(datasets_dir / "test-task1-SI.labels"))
    p.add_argument("--tokens-output", type=str, default=str(task1_data_dir / "test-task1-tokens.tsv"))
    p.add_argument("--predictions-output", type=str, default=str(task1_data_dir / "test-task1-predictions.labels"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = run_predictions(
        test_articles_dir=Path(args.test_articles_dir),
        test_gold_spans=Path(args.test_gold_labels) if args.test_gold_labels else None,
        tokens_output=Path(args.tokens_output),
        predictions_output=Path(args.predictions_output),
        model_name=args.model_name,
        model_type=args.model_type,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    if results["f1"]:
        print(f"Test F1: {results['f1']:.4f}")


if __name__ == "__main__":
    main()
