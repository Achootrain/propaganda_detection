
from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    # Modules
    "runtime": (".utils.runtime", None),

    # spaCy
    "get_configured_spacy": (".nlp.spacy_utils", "get_configured_spacy"),

    # BIO/data
    "load_data": (".data.bio_data", "load_data"),
    "read_predictions_from_file": (".data.bio_data", "read_predictions_from_file"),
    "group_spans_by_article_ids": (".data.bio_data", "group_spans_by_article_ids"),
    "token_label_from_spans": (".data.bio_data", "token_label_from_spans"),
    "create_bio_labeled": (".data.bio_data", "create_bio_labeled"),
    "create_bio_unlabeled": (".data.bio_data", "create_bio_unlabeled"),

    # Types (keep these lightweight)
    "InputExample": (".utils.types", "InputExample"),
    "InputFeatures": (".utils.types", "InputFeatures"),
    "DatasetBundle": (".utils.types", "DatasetBundle"),
    "TokenSpan": (".utils.types", "TokenSpan"),

    # Features
    "read_examples_from_file": (".data.features", "read_examples_from_file"),
    "convert_examples_to_features": (".data.features", "convert_examples_to_features"),
    "get_labels": (".data.features", "get_labels"),
    "get_pos_ner_dep_maps": (".data.features", "get_pos_ner_dep_maps"),
    "get_pos_ner_maps": (".data.features", "get_pos_ner_maps"),

    # Techniques
    "get_technique_labels": (".data.techniques", "get_technique_labels"),
    "read_technique_spans": (".data.techniques", "read_technique_spans"),
    "get_techniques_for_token": (".data.techniques", "get_techniques_for_token"),
    "get_technique_map": (".data.techniques", "get_technique_map"),
    "get_multilabel_technique_map": (".data.techniques", "get_multilabel_technique_map"),
    "attach_technique_labels_to_examples": (".data.techniques", "attach_technique_labels_to_examples"),

    # CRF
    "ConditionalRandomField": (".modeling.crf", "ConditionalRandomField"),
    "allowed_transitions": (".modeling.crf", "allowed_transitions"),

    # Model
    "ScalarMix": (".modeling.model", "ScalarMix"),
    "SentenceAuxiliaryTask": (".modeling.model", "SentenceAuxiliaryTask"),
    "TechniqueAuxiliaryTask": (".modeling.model", "TechniqueAuxiliaryTask"),
    "BertLstmCrf": (".modeling.model", "BertLstmCrf"),

    # Submission/scoring/postprocess
    "tokenize_like_bio": (".eval.submission", "tokenize_like_bio"),
    "labels_to_spans": (".eval.submission", "labels_to_spans"),
    "write_token_predictions": (".eval.submission", "write_token_predictions"),
    "write_submission_file": (".eval.submission", "write_submission_file"),
    "aggregate_article_spans": (".eval.submission", "aggregate_article_spans"),
    "load_span_annotations": (".eval.scoring", "load_span_annotations"),
    "compute_precision_recall_f1": (".eval.scoring", "compute_precision_recall_f1"),
    "postprocess_spans": (".eval.postprocess", "postprocess_spans"),

    # Training/inference
    "set_seed": (".pipeline.training", "set_seed"),
    "build_dataset": (".pipeline.training", "build_dataset"),
    "prepare_bio_files": (".pipeline.training", "prepare_bio_files"),
    "evaluate_model": (".pipeline.training", "evaluate_model"),
    "predict_model": (".pipeline.training", "predict_model"),
    "train_model": (".pipeline.training", "train_model"),
    "predict_on_articles": (".pipeline.inference", "predict_on_articles"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _LAZY_EXPORTS[name]
    try:
        module = import_module(module_path, __name__)
    except ModuleNotFoundError as exc:
        missing = exc.name or "<unknown>"
        install_hint = ""
        if missing == "torch":
            install_hint = " Install it with: pip install torch"
        elif missing in {"numpy", "transformers", "spacy", "seqeval", "sklearn"}:
            install_hint = f" Install it with: pip install {missing}"

        raise AttributeError(
            f"Cannot import {name!r} because dependency {missing!r} is not installed." + install_hint
        ) from exc

    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_EXPORTS.keys()))

__all__ = [
    "runtime",
    "get_configured_spacy",
    "load_data",
    "read_predictions_from_file",
    "group_spans_by_article_ids",
    "token_label_from_spans",
    "create_bio_labeled",
    "create_bio_unlabeled",
    "InputExample",
    "InputFeatures",
    "read_examples_from_file",
    "convert_examples_to_features",
    "get_labels",
    "get_pos_ner_dep_maps",
    "get_pos_ner_maps",
    "get_technique_labels",
    "read_technique_spans",
    "get_techniques_for_token",
    "get_technique_map",
    "get_multilabel_technique_map",
    "attach_technique_labels_to_examples",
    "ConditionalRandomField",
    "allowed_transitions",
    "ScalarMix",
    "SentenceAuxiliaryTask",
    "TechniqueAuxiliaryTask",
    "BertLstmCrf",
    "TokenSpan",
    "tokenize_like_bio",
    "labels_to_spans",
    "write_token_predictions",
    "write_submission_file",
    "aggregate_article_spans",
    "load_span_annotations",
    "compute_precision_recall_f1",
    "postprocess_spans",
    "DatasetBundle",
    "set_seed",
    "build_dataset",
    "prepare_bio_files",
    "evaluate_model",
    "predict_model",
    "train_model",
    "predict_on_articles",
]
