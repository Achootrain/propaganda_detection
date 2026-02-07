"""Technique Classification

Modules:
    utils: Runtime configuration, types, text utilities
    data: Data loading, feature encoding
    modeling: Classification heads and custom model
    pipeline: Training, evaluation, prediction
    eval: Post-processing, submission, scoring
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS: dict[str, tuple[str, str | None]] = {
    # Runtime
    "DEVICE": (".utils.runtime", "DEVICE"),
    "LOGGER": (".utils.runtime", "LOGGER"),
    "set_seed": (".utils.runtime", "set_seed"),
    "set_device": (".utils.runtime", "set_device"),

    # Types
    "InputFeatures": (".utils.types", "InputFeatures"),
    "LABEL_ORDER": (".utils.types", "LABEL_ORDER"),

    # Text utilities
    "get_stopwords": (".utils.text_utils", "get_stopwords"),

    # Data loading
    "read_articles_from_folder": (".data.loader", "read_articles_from_folder"),
    "read_task2_labels": (".data.loader", "read_task2_labels"),
    "load_tc_data": (".data.loader", "load_tc_data"),
    "load_tc_test_template": (".data.loader", "load_tc_test_template"),

    # Feature encoding
    "build_label_map": (".data.features", "build_label_map"),
    "build_train_instances": (".data.features", "build_train_instances"),
    "compute_matchings": (".data.features", "compute_matchings"),
    "encode_examples": (".data.features", "encode_examples"),
    "features_to_dataset": (".data.features", "features_to_dataset"),

    # Modeling
    "RobertaClassificationHead": (".modeling.heads", "RobertaClassificationHead"),
    "RobertaClassificationHeadLength": (".modeling.heads", "RobertaClassificationHeadLength"),
    "RobertaClassificationHeadMatchings": (".modeling.heads", "RobertaClassificationHeadMatchings"),
    "RobertaClassificationHeadJoined": (".modeling.heads", "RobertaClassificationHeadJoined"),
    "RobertaClassificationHeadJoinedLength": (".modeling.heads", "RobertaClassificationHeadJoinedLength"),
    "RobertaClassificationHeadJoinedLengthMatchings": (".modeling.heads", "RobertaClassificationHeadJoinedLengthMatchings"),
    "CustomRobertaForSequenceClassification": (".modeling.model", "CustomRobertaForSequenceClassification"),

    # Pipeline
    "train_classifier": (".pipeline.training", "train_classifier"),
    "evaluate_classifier": (".pipeline.training", "evaluate_classifier"),
    "predict_classifier": (".pipeline.training", "predict_classifier"),
    "predict_classifier_probs": (".pipeline.training", "predict_classifier_probs"),

    # Eval
    "postprocess_predictions_local": (".eval.postprocess", "postprocess_predictions_local"),
    "build_train_instances_for_postprocess": (".eval.postprocess", "build_train_instances_for_postprocess"),
    "build_insides_from_train": (".eval.postprocess", "build_insides_from_train"),
    "create_submission_file": (".eval.submission", "create_submission_file"),
    "eval_submission_file": (".eval.submission", "eval_submission_file"),

    # Main
    "main": (".main", "main"),
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
        elif missing in {"numpy", "transformers", "sklearn", "pandas"}:
            install_hint = f" Install it with: pip install {missing}"

        raise AttributeError(
            f"Cannot import {name!r} because dependency {missing!r} is not installed." + install_hint
        ) from exc

    value = module if attr_name is None else getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_EXPORTS.keys()))


__all__ = list(_LAZY_EXPORTS.keys())