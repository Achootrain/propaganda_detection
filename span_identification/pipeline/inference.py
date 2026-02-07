from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from ..data.features import get_labels
from ..modeling.model import BertLstmCrf
from ..eval.postprocess import postprocess_spans
from ..eval.submission import aggregate_article_spans, write_submission_file, write_token_predictions
from ..data.techniques import get_technique_labels
from .training import build_dataset, prepare_bio_files, predict_model
from ..utils import runtime


def predict_on_articles(
    model_path: str,
    prediction_articles_path: str,
    *,
    model_name: str = "roberta-base",
    max_seq_length: int = 256,
    batch_size: int = 24,
    nlp=None,
    prediction_labels: str = "prediction_tokens.bio",
    prediction_submission_path: str = "prediction_submission.tsv",
    apply_postprocess: bool = True,
    pos_ner_maps_path: Optional[str] = None,
    pos_label_map: Optional[Dict[str, int]] = None,
    ner_label_map: Optional[Dict[str, int]] = None,
):
    if pos_ner_maps_path is not None and (pos_label_map is None or ner_label_map is None):
        with open(pos_ner_maps_path, "r", encoding="utf-8") as f:
            maps_data = json.load(f)
        pos_label_map = maps_data.get("pos_label_map", {})
        ner_label_map = maps_data.get("ner_label_map", {})
        print("Loaded POS/NER maps from %s", pos_ner_maps_path)

    if pos_label_map is None:
        pos_label_map = {}
    if ner_label_map is None:
        ner_label_map = {}

    prediction_labels_path = Path(prediction_labels)
    prediction_submission_path_obj = Path(prediction_submission_path)

    work_dir_path = prediction_labels_path.parent if prediction_labels_path.parent != Path("") else Path.cwd()
    work_dir_path.mkdir(parents=True, exist_ok=True)
    if prediction_submission_path_obj.parent != Path(""):
        prediction_submission_path_obj.parent.mkdir(parents=True, exist_ok=True)

    articles_path = Path(prediction_articles_path)
    if not articles_path.exists():
        raise FileNotFoundError(f"Prediction articles path does not exist: {prediction_articles_path}")

    prediction_articles = []
    prediction_article_ids = []

    file_list = sorted(p for p in articles_path.iterdir() if p.is_file() and p.suffix.lower() == ".txt")

    for article_file in file_list:
        with article_file.open("r", encoding="utf-8") as handle:
            prediction_articles.append(handle.read())

        stem = article_file.stem
        if stem.startswith("article"):
            prediction_article_ids.append(stem[len("article") :])
        else:
            prediction_article_ids.append(stem)

    if not prediction_articles:
        raise ValueError(f"No .txt articles found in {prediction_articles_path}")

    token_cache = {}
    bio_path, article_order = prepare_bio_files(
        str(work_dir_path),
        nlp,
        mode="prediction",
        articles=prediction_articles,
        article_ids=prediction_article_ids,
        token_cache=token_cache,
        filename="prediction.bio",
    )

    label_list = get_labels()
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name)

    enable_tech_aux = True
    technique_labels = get_technique_labels(None)
    num_technique_classes = max(
        0,
        len(technique_labels) - (1 if technique_labels and technique_labels[0] == "O" else 0),
    )

    pos_vocab_size = len(pos_label_map) + 1 if pos_label_map else 0
    ner_vocab_size = len(ner_label_map) + 1 if ner_label_map else 0
    dep_vocab_size = 0
    dep_embedding_dim = 50
    lexicon_feature_dim = 1

    model = BertLstmCrf(
        base_model,
        label_list,
        pad_token_label_id,
        rnn_layers=1,
        rnn_dropout=0.1,
        output_dropout=getattr(base_model.config, "hidden_dropout_prob", 0.1),
        pos_vocab_size=pos_vocab_size,
        ner_vocab_size=ner_vocab_size,
        pos_embedding_dim=25,
        ner_embedding_dim=25,
        dep_vocab_size=dep_vocab_size,
        dep_embedding_dim=dep_embedding_dim,
        lexicon_feature_dim=lexicon_feature_dim,
        num_technique_classes=(num_technique_classes if enable_tech_aux else 0),
        tech_loss_weight=0.5,
    ).to(runtime.DEVICE)

    state_dict = torch.load(model_path, map_location=runtime.DEVICE)
    model.load_state_dict(state_dict)

    dataset_bundle = build_dataset(
        bio_path,
        "prediction",
        article_order,
        tokenizer,
        label_list,
        max_seq_length,
        pad_token_label_id,
        pos_label_map=pos_label_map,
        ner_label_map=ner_label_map,
    )

    predictions = predict_model(model, dataset_bundle, label_list, pad_token_label_id, batch_size)
    spans = aggregate_article_spans(predictions, article_order, token_cache)
    if apply_postprocess:
        article_texts_map = dict(zip(prediction_article_ids, prediction_articles))
        spans = postprocess_spans(spans, article_texts_map)

    write_token_predictions(bio_path, predictions, prediction_labels_path)
    write_submission_file(spans, prediction_submission_path_obj)

    return {
        "bio": Path(bio_path),
        "token_predictions": prediction_labels_path,
        "submission": prediction_submission_path_obj,
    }
