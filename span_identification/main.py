from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModel, AutoTokenizer

from .utils import runtime
from .data.bio_data import load_data
from .data.features import get_labels, get_pos_ner_dep_maps, read_examples_from_file
from .pipeline.inference import predict_on_articles
from .modeling.model import BertLstmCrf
from .eval.postprocess import postprocess_spans
from .eval.scoring import compute_precision_recall_f1, load_span_annotations
from .nlp.spacy_utils import get_configured_spacy
from .eval.submission import aggregate_article_spans, write_submission_file, write_token_predictions
from .data.techniques import get_technique_labels, read_technique_spans
from .pipeline.training import build_dataset, prepare_bio_files, predict_model, set_seed, train_model
from .utils.types import TokenSpan


def main() -> int:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if runtime.TPU_AVAILABLE:
        import torch_xla  # type: ignore

    # Manual Configuration
    num_workers = 1
    use_tpu = True

    train_batch_size = 16
    eval_batch_size = 64
    learning_rate = 2e-5
    weight_decay = 0.01
    warmup_steps = 500
    max_grad_norm = 1.0
    num_epochs = 18
    freeze_encoder_epochs = 2
    seed = 22

    pos_embedding_dim = 50
    ner_embedding_dim = 50
    dep_embedding_dim = 50
    lexicon_feature_dim = 1
    rnn_hidden_size = 600
    rnn_layers = 2
    rnn_dropout = 0.1
    output_dropout = 0.1
    ffn_hidden_dim = 200

    sentence_aux_weight = 1.0
    token_loss_weight = 1.0
    tech_aux_weight = 0.5
    enable_tech_aux = True

    patience = 5
    accum_steps = 1

    save_checkpoint_epochs = 0
    resume_from_checkpoint = None

    model_name = "roberta-large"
    max_seq_length = 256

    data_root = (
        Path("/kaggle/input/propaganda-dataset/datasets")
        if os.path.exists("/kaggle/input/propaganda-dataset/datasets")
        else Path("/content/drive/MyDrive/ColabNotebooks/datasets")
    )
    work_dir = (
        Path("/kaggle/working/")
        if os.path.exists("/kaggle/working/")
        else Path("/content/drive/MyDrive/ColabNotebooks/output")
    )
    output_dir = work_dir

    if not data_root.exists():
        data_root = Path(r"D:\Project3\project\datasets")

    techniques_file = data_root.parent / "propaganda-techniques-names.txt"
    if not techniques_file.exists():
        techniques_file = data_root / "propaganda-techniques-names.txt"

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data from %s", data_root)

    if not (data_root / "train-articles").exists():
        print("Data directories not found at %s", data_root)
        return 1

    train_articles, train_ids, _ = load_data(str(data_root / "train-articles"), str(techniques_file))
    dev_articles, dev_ids, _ = load_data(str(data_root / "dev-articles"), str(techniques_file))
    test_articles, test_ids, _ = load_data(str(data_root / "test-articles"), str(techniques_file))

    train_labels_path = data_root / "train-task1-SI.labels"
    dev_labels_path = data_root / "dev-task1-SI.labels"
    test_labels_path = data_root / "test-task1-SI.labels"

    train_tc_labels_path = data_root / "train-task2-TC.labels"
    dev_tc_labels_path = data_root / "dev-task2-TC.labels"
    test_tc_labels_path = data_root / "test-task2-TC.labels"

    dev_prediction_labels = output_dir / "dev_predictions.bio"
    test_prediction_labels = output_dir / "test_predictions.bio"

    dev_submission_path = output_dir / "dev_submission.tsv"
    test_submission_path = output_dir / "test_submission.tsv"

    print("Initializing spaCy...")
    nlp = get_configured_spacy("en_core_web_sm")

    token_cache: Dict[str, List[TokenSpan]] = {}

    train_bio, train_order = prepare_bio_files(
        str(work_dir),
        nlp,
        mode="train",
        articles=train_articles,
        article_ids=train_ids,
        labels_path=str(train_labels_path),
        token_cache=token_cache,
        filename="train.bio",
    )
    dev_bio, dev_order = prepare_bio_files(
        str(work_dir),
        nlp,
        mode="dev",
        articles=dev_articles,
        article_ids=dev_ids,
        labels_path=str(dev_labels_path),
        token_cache=token_cache,
        filename="dev.bio",
    )
    test_bio, test_order = prepare_bio_files(
        str(work_dir),
        nlp,
        mode="test",
        articles=test_articles,
        article_ids=test_ids,
        token_cache=token_cache,
        filename="test.bio",
    )

    dev_gold_annotations = load_span_annotations(str(dev_labels_path))
    dev_texts_map = dict(zip(dev_ids, dev_articles))
    test_texts_map = dict(zip(test_ids, test_articles))

    technique_labels = get_technique_labels(str(techniques_file) if techniques_file.exists() else None)
    train_tc_spans = read_technique_spans(str(train_tc_labels_path)) if train_tc_labels_path.exists() else {}
    dev_tc_spans = read_technique_spans(str(dev_tc_labels_path)) if dev_tc_labels_path.exists() else {}
    test_tc_spans = read_technique_spans(str(test_tc_labels_path)) if test_tc_labels_path.exists() else {}

    if runtime.TPU_AVAILABLE and use_tpu:
        runtime.set_device(torch_xla.device())
        print("Using TPU device: %s (single-process mode)", runtime.DEVICE)
    else:
        runtime.set_device(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("Using device: %s", runtime.DEVICE)

    set_seed(seed)

    label_list = get_labels()
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    print("Building POS/NER/DEP maps from training data...")
    train_examples_temp = read_examples_from_file(train_bio, "train")
    pos_label_map, ner_label_map, dep_label_map = get_pos_ner_dep_maps(train_examples_temp)
    print(f"  POS labels: {len(pos_label_map)}, NER labels: {len(ner_label_map)}, DEP labels: {len(dep_label_map)}")

    model = BertLstmCrf(
        base_model,
        label_list,
        pad_token_label_id,
        rnn_layers=rnn_layers,
        rnn_dropout=rnn_dropout,
        output_dropout=getattr(base_model.config, "hidden_dropout_prob", output_dropout),
        pos_vocab_size=len(pos_label_map) + 1,
        ner_vocab_size=len(ner_label_map) + 1,
        dep_vocab_size=len(dep_label_map) + 1,
        pos_embedding_dim=pos_embedding_dim,
        ner_embedding_dim=ner_embedding_dim,
        dep_embedding_dim=dep_embedding_dim,
        lexicon_feature_dim=lexicon_feature_dim,
        use_scalar_mix=True,
        rnn_hidden_size=rnn_hidden_size,
        ffn_hidden_dim=ffn_hidden_dim,
        use_mean_pooling=True,
        num_technique_classes=(
            max(0, len(technique_labels) - (1 if technique_labels and technique_labels[0] == "O" else 0))
            if enable_tech_aux
            else 0
        ),
        tech_loss_weight=tech_aux_weight,
        sentence_aux_weight=sentence_aux_weight,
        token_loss_weight=token_loss_weight,
    ).to(runtime.DEVICE)

    train_bundle = build_dataset(
        train_bio,
        "train",
        train_order,
        tokenizer,
        label_list,
        max_seq_length,
        pad_token_label_id,
        pos_label_map=pos_label_map,
        ner_label_map=ner_label_map,
        dep_label_map=dep_label_map,
        token_cache=token_cache,
        technique_spans_by_article=train_tc_spans,
        technique_labels=technique_labels,
    )
    dev_bundle = build_dataset(
        dev_bio,
        "dev",
        dev_order,
        tokenizer,
        label_list,
        max_seq_length,
        pad_token_label_id,
        pos_label_map=pos_label_map,
        ner_label_map=ner_label_map,
        dep_label_map=dep_label_map,
        token_cache=token_cache,
        technique_spans_by_article=dev_tc_spans,
        technique_labels=technique_labels,
    )
    test_bundle = build_dataset(
        test_bio,
        "test",
        test_order,
        tokenizer,
        label_list,
        max_seq_length,
        pad_token_label_id,
        pos_label_map=pos_label_map,
        ner_label_map=ner_label_map,
        dep_label_map=dep_label_map,
        token_cache=token_cache,
        technique_spans_by_article=test_tc_spans,
        technique_labels=technique_labels,
    )

    best_metrics = train_model(
        model,
        train_bundle,
        dev_bundle,
        label_list,
        pad_token_label_id,
        train_batch_size,
        eval_batch_size,
        learning_rate,
        weight_decay,
        num_epochs,
        warmup_steps,
        max_grad_norm,
        patience,
        accum_steps,
        freeze_encoder_epochs=freeze_encoder_epochs,
        save_checkpoint_epochs=save_checkpoint_epochs,
        checkpoint_dir=str(output_dir),
        resume_from_checkpoint=resume_from_checkpoint,
        dev_gold_spans=dev_gold_annotations,
        token_cache=token_cache,
        dev_article_texts=dev_texts_map,
        apply_postprocess=True,
        num_workers=num_workers,
    )

    print("Best dev F1: %.4f", best_metrics["f1"])

    dev_predictions = predict_model(model, dev_bundle, label_list, pad_token_label_id, eval_batch_size, num_workers=num_workers)
    test_predictions = predict_model(model, test_bundle, label_list, pad_token_label_id, eval_batch_size, num_workers=num_workers)

    dev_token_spans = aggregate_article_spans(dev_predictions, dev_order, token_cache)
    test_token_spans = aggregate_article_spans(test_predictions, test_order, token_cache)

    dev_token_spans = postprocess_spans(dev_token_spans, dev_texts_map)
    test_token_spans = postprocess_spans(test_token_spans, test_texts_map)

    write_token_predictions(dev_bio, dev_predictions, dev_prediction_labels)
    write_token_predictions(test_bio, test_predictions, test_prediction_labels)
    write_submission_file(dev_token_spans, dev_submission_path)
    write_submission_file(test_token_spans, test_submission_path)

    dev_metrics = compute_precision_recall_f1(dev_token_spans, dev_gold_annotations)
    test_gold_annotations = load_span_annotations(str(test_labels_path))
    test_metrics = compute_precision_recall_f1(test_token_spans, test_gold_annotations)

    print("Dev submission metrics: %s", dev_metrics)
    print("Test submission metrics: %s", test_metrics)

    torch.save(model.state_dict(), Path(str(output_dir)) / "best_model.pt")

    return 0
