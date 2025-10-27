from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer, set_seed as hf_set_seed

from task1.training.trainer import run_training
from task1.process_data.labeling import create_token_labels
from task1.process_data.utils import (
    read_tokens,
    build_vocab,
    write_vocab,
    load_vocab_mapping,
)
from task1.process_data.dataloader import create_data_loader
from task1.multigranularity.sentence_splitter import create_sentence_labels


# ----------------------------
# Configuration
# ----------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT_DIR / "datasets"
TASK1_DATA_DIR = ROOT_DIR / "task1" / "data"

TRAIN_ARTICLES_DIR = DATASETS_DIR / "train-articles"
TRAIN_SPANS_FILE = DATASETS_DIR / "train-task1-SI.labels"
TRAIN_OUTPUT_LABELED_TOKENS = TASK1_DATA_DIR / "train-task1-tokens.tsv"
DEFAULT_VOCAB_OUTPUT = TASK1_DATA_DIR / "train-task1-vocab.json"

DEV_ARTICLES_DIR = DATASETS_DIR / "dev-articles"
DEV_SPANS_FILE = DATASETS_DIR / "dev-task1-SI.labels"
DEV_OUTPUT_LABELED_TOKENS = TASK1_DATA_DIR / "dev-task1-tokens.tsv"

TRAIN_OUTPUT_LABELED_SENTENCE = TASK1_DATA_DIR / "train-task1-sentences.tsv"
DEV_OUTPUT_LABELED_SENTENCE = TASK1_DATA_DIR / "dev-task1-sentences.tsv"

DEBUG_OUTPUT_PATH = TASK1_DATA_DIR / "debug_report.txt"

MODEL_NAME = "roberta-large"
MAX_TOKENS = 512
STRIDE = 128

MAX_SEQUENCE_LENGTH = 256
BATCH_SIZE = 8

# Where to save the best-performing model checkpoint for prediction
BEST_CHECKPOINT_PATH = ROOT_DIR / "task1" / "model" / "best.pt"


def main() -> None:
    """Main entrypoint for the pipeline."""

    # ----------------------------
    # Data Processing
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.config.output_hidden_states = True
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Process training data
    create_token_labels(
        articles_dir=TRAIN_ARTICLES_DIR,
        spans_file=TRAIN_SPANS_FILE,
        tokens_output=TRAIN_OUTPUT_LABELED_TOKENS,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_tokens=MAX_TOKENS,
        stride=STRIDE
    )

    # Process development data
    create_token_labels(
        articles_dir=DEV_ARTICLES_DIR,
        spans_file=DEV_SPANS_FILE,
        tokens_output=DEV_OUTPUT_LABELED_TOKENS,
        tokenizer=tokenizer,
        model=model,
        device=device,
        max_tokens=MAX_TOKENS,
        stride=STRIDE
    )

    # Build vocabulary
    tokens = read_tokens(TRAIN_OUTPUT_LABELED_TOKENS, delimiter="\t", token_column=4)
    vocab = build_vocab(
        tokens,
        tokenizer=tokenizer,
        special_tokens=[token for token in ["<pad>", "<unk>"]],
        min_frequency=1,
        max_size=None,
    )
    write_vocab(vocab, DEFAULT_VOCAB_OUTPUT)

    # ----------------------------
    # Build DataLoaders
    # ----------------------------
    vocab_map = load_vocab_mapping(DEFAULT_VOCAB_OUTPUT)

    train_loader = create_data_loader(
        TRAIN_OUTPUT_LABELED_TOKENS,
        tokenizer,
        batch_size=BATCH_SIZE,
        shuffle=True,
        max_length=MAX_SEQUENCE_LENGTH,
        vocab_map=vocab_map,
    )

    dev_loader = create_data_loader(
        DEV_OUTPUT_LABELED_TOKENS,
        tokenizer,
        batch_size=BATCH_SIZE,
        shuffle=False,
        max_length=MAX_SEQUENCE_LENGTH,
        vocab_map=None,
    )

    # ----------------------------
    # Training
    # ----------------------------
    run_training(
        train_loader=train_loader,
        val_loader=dev_loader,
        model_name=MODEL_NAME,
        model_type="multi_granularity",
        debug_output_path=DEBUG_OUTPUT_PATH,
        num_labels=3,
        batch_size=BATCH_SIZE,
        max_length=MAX_SEQUENCE_LENGTH,
        tokenizer=tokenizer,
        # Save the best model (by validation F1) here
        model_output_path=BEST_CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    hf_set_seed(42)
    main()
