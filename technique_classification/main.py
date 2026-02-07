"""Main entry point for technique classification pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .utils.runtime import DEVICE, LOGGER, set_seed
from .utils.types import LABEL_ORDER
from .data.loader import load_tc_data, load_tc_test_template
from .data.features import build_label_map, build_train_instances, encode_examples, features_to_dataset
from .modeling.model import CustomRobertaForSequenceClassification
from .pipeline.training import train_classifier, predict_classifier_probs
from .eval.postprocess import postprocess_predictions_local
from .eval.submission import eval_submission_file


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    LOGGER.info("Using device: %s", DEVICE)
    print(f"Using device: {DEVICE}")

    # ---- Config (paper-faithful defaults; override via env vars if needed) ----
    model_name = "roberta-large"
    max_seq_length = 256
    train_batch_size = 16
    eval_batch_size = 16
    learning_rate = 2e-5
    weight_decay = 0.01
    num_epochs = 15
    patience = 5
    seed = 22

    # Defaults for local repo usage; override via env vars.
    data_root = Path("/kaggle/input/propaganda-dataset/datasets")
    output_dir = Path("/kaggle/working")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_articles_dir = data_root / "train-articles"
    dev_articles_dir = data_root / "dev-articles"
    test_articles_dir = data_root / "test-articles"

    train_labels_path = data_root / "train-task2-TC.labels"
    dev_labels_path = data_root / "dev-task2-TC.labels"
    test_labels_path = data_root / "test-task2-TC.labels"
    set_seed(seed)

    # ---- Load data ----
    LOGGER.info("Loading training/dev data...")
    print("Loading training/dev data...")
    train_df, _ = load_tc_data(str(train_articles_dir), str(train_labels_path))
    dev_df, _ = load_tc_data(str(dev_articles_dir), str(dev_labels_path))
    test_df, _ = load_tc_test_template(str(test_articles_dir), str(test_labels_path))

    label_list, label2id = build_label_map(train_df["label"].tolist() + dev_df["label"].tolist())
    LOGGER.info("Num labels: %d", len(label_list))
    print(f"Num labels: {len(label_list)}")

    # ---- Compute class weights (inverse frequency) ----
    label_counts = train_df["label"].value_counts().to_dict()
    counts_vec = np.array([label_counts.get(lab, 0) for lab in label_list], dtype=np.float32)
    safe_counts = np.where(counts_vec == 0, 1.0, counts_vec)
    inv_freq = 1.0 / safe_counts
    class_weights_np = inv_freq / inv_freq.sum() * len(label_list)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float)

    # ---- Tokenizer & backbone init (supports SI->TC transfer learning) ----
    init_model_path = Path("/kaggle/input/deberta-v3-large-span-iden/transformers/default/1/best_model.pt")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_config = AutoConfig.from_pretrained(model_name, num_labels=len(label_list))

    def _init_backbone() -> torch.nn.Module:
        backbone = AutoModel.from_pretrained(model_name, config=base_config)
        if init_model_path and str(init_model_path) and init_model_path.exists() and init_model_path.is_file():
            print(f"Initializing backbone from checkpoint: {init_model_path}")
            ckpt = torch.load(str(init_model_path), map_location="cpu")
            state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
                state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

            try:
                if isinstance(state_dict, dict) and any(k.startswith("bert_encoder.") for k in state_dict.keys()):
                    filtered = {k[len("bert_encoder."):]: v for k, v in state_dict.items() if k.startswith("bert_encoder.")}
                    missing, unexpected = backbone.load_state_dict(filtered, strict=False)
                elif isinstance(state_dict, dict) and any(k.startswith("roberta.") for k in state_dict.keys()):
                    filtered = {k[len("roberta."):]: v for k, v in state_dict.items() if k.startswith("roberta.")}
                    missing, unexpected = backbone.load_state_dict(filtered, strict=False)
                else:
                    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
                print(f"Loaded checkpoint into backbone | missing: {len(missing)} | unexpected: {len(unexpected)}")
            except RuntimeError as exc:
                print(
                    "WARNING: checkpoint load failed for backbone (likely architecture/vocab mismatch). "
                    "Falling back to pretrained backbone. Error: %s" % exc
                )
        return backbone

    # Build span -> label matchings for auxiliary features
    train_instances = build_train_instances(train_df)

    # ---- Encode & build datasets ----
    train_features, _ = encode_examples(
        train_df, tokenizer, label2id, max_seq_length, True,
        use_length=True, use_matchings=True, join_embeddings=True,
        train_instances=train_instances,
    )
    dev_features, _ = encode_examples(
        dev_df, tokenizer, label2id, max_seq_length, True,
        use_length=True, use_matchings=True, join_embeddings=True,
        train_instances=train_instances,
    )
    test_features, _ = encode_examples(
        test_df, tokenizer, label2id, max_seq_length, False,
        use_length=True, use_matchings=True, join_embeddings=True,
        train_instances=train_instances,
    )

    train_dataset = features_to_dataset(train_features)
    dev_dataset = features_to_dataset(dev_features)
    test_dataset = features_to_dataset(test_features)

    # ---- Train model A ----
    print("Training Model A (CLS+avg+len+matchings)...")
    cfg_a = AutoConfig.from_pretrained(model_name, num_labels=len(label_list))
    model_a = CustomRobertaForSequenceClassification(
        cfg_a,
        use_length=True,
        use_matchings=True,
        join_embeddings=True,
        class_weights=class_weights.to(DEVICE),
    ).to(DEVICE)
    model_a.roberta = _init_backbone().to(DEVICE)
    _ = train_classifier(
        model_a,
        train_dataset,
        dev_dataset,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
        train_batch_size=train_batch_size,
        eval_batch_size=eval_batch_size,
        use_length=True,
        use_matchings=True,
        patience=patience,
    )

    print("Predicting probabilities...")
    probs_a_dev = predict_classifier_probs(model_a, dev_dataset, batch_size=eval_batch_size, use_length=True, use_matchings=True)
    probs_a_test = predict_classifier_probs(model_a, test_dataset, batch_size=eval_batch_size, use_length=True, use_matchings=True)

    # ---- Post-process labels ----
    dev_pred_a = postprocess_predictions_local(probs_a_dev, dev_df.copy(), train_df=train_df)
    test_pred_a = postprocess_predictions_local(probs_a_test, test_df.copy(), train_df=train_df)

    label2id = {lab: i for i, lab in enumerate(label_list)}
    best_name = "model_a"
    dev_pred_best, test_pred_best = dev_pred_a, test_pred_a
    dev_probs_best, test_probs_best = probs_a_dev, probs_a_test

    dev_submission_path = output_dir / "dev_submission_task2_best.tsv"
    test_submission_path = output_dir / "test_submission_task2_best.tsv"
    test_confidence_path = output_dir / "test_submission_task2_best_conf.tsv"

    # Write best submissions
    with open(dev_submission_path, "w", encoding="utf-8") as handle:
        for (_, row), lab in zip(dev_df.iterrows(), list(dev_pred_best)):
            handle.write(f"{row['article_id']}\t{lab}\t{row['span_start']}\t{row['span_end']}\n")
    with open(test_submission_path, "w", encoding="utf-8") as handle:
        for i, (_, row) in enumerate(test_df.iterrows()):
            lab = test_pred_best[i]
            handle.write(f"{row['article_id']}\t{lab}\t{row['span_start']}\t{row['span_end']}\n")

    # Confidence file (probability of emitted label)
    with open(test_confidence_path, "w", encoding="utf-8") as handle:
        handle.write("article_id\tlabel\tstart\tend\tconfidence\n")
        for i, (_, row) in enumerate(test_df.iterrows()):
            lab = str(test_pred_best[i])
            lab_idx = label2id.get(lab, None)
            conf = float(test_probs_best[i, lab_idx]) if lab_idx is not None else 0.0
            handle.write(f"{row['article_id']}\t{lab}\t{row['span_start']}\t{row['span_end']}\t{conf:.6f}\n")

    # ---- Evaluate on dev using gold labels (for the best pick) ----
    dev_scores = eval_submission_file(str(dev_submission_path), str(dev_labels_path), label_list)
    LOGGER.info(
        "Dev metrics - precision: %.4f | recall: %.4f | f1-macro: %.4f | f1-micro: %.4f",
        dev_scores["precision"],
        dev_scores["recall"],
        dev_scores["f1_macro"],
        dev_scores["f1_micro"],
    )
    print(f"Best model: {best_name} | Dev f1-micro: {dev_scores['f1_micro']:.4f} | Dev f1-macro: {dev_scores['f1_macro']:.4f}")

    # Optional: evaluate test vs gold template (if labels available)
    test_scores = eval_submission_file(str(test_submission_path), str(test_labels_path), label_list)
    LOGGER.info(
        "Test (best) metrics - precision: %.4f | recall: %.4f | f1-macro: %.4f | f1-micro: %.4f",
        test_scores["precision"],
        test_scores["recall"],
        test_scores["f1_macro"],
        test_scores["f1_micro"],
    )
    print(f"Dev submission written to {dev_submission_path}")
    print(f"Test submission written to {test_submission_path}")
    print(f"Confidence file written to {test_confidence_path}")

    # Save model (optional artifact)
    torch.save({"model_state_dict": model_a.state_dict(), "config": cfg_a, "label_list": label_list}, output_dir / "tc_model_a_cls.pt")
    tokenizer_save_path = output_dir / "tokenizer"
    tokenizer_save_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(tokenizer_save_path)


if __name__ == "__main__":
    main()
