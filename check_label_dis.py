#!/usr/bin/env python3
"""Summarize label distribution across SemEval Task 11 splits."""

import argparse
import csv
from collections import Counter
from typing import Iterable, Set, Tuple


def parse_args() -> argparse.Namespace:
    """Collect command-line arguments for label files."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare technique label distributions between train and dev "
            "files while reporting article counts."
        )
    )
    parser.add_argument(
        "--train",
        default="train-task2-TC.labels",
        help="Path to the train labels file (tab-delimited).",
    )
    parser.add_argument(
        "--dev",
        default="dev-task2-TC.labels",
        help="Path to the dev labels file (tab-delimited).",
    )
    return parser.parse_args()


def load_labels(path: str) -> Tuple[Counter, Set[str]]:
    """Count labels and collect article identifiers from a labels file."""
    counts: Counter = Counter()
    articles: Set[str] = set()
    # Open the labels file and accumulate counts grouped by label name.
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            # Skip the header row when present in the file.
            if row[0].lower() == "article_id":
                continue
            article_id, label = row[0], row[1]
            counts[label] += 1
            articles.add(article_id)
    return counts, articles


def compute_summary(
    train_counts: Counter, dev_counts: Counter
) -> Iterable[Tuple[str, int, int, float, float]]:
    """Yield summary rows containing counts and percentages per label."""
    all_labels = sorted(set(train_counts) | set(dev_counts))
    for label in all_labels:
        train_total = train_counts.get(label, 0)
        dev_total = dev_counts.get(label, 0)
        total = train_total + dev_total
        if total:
            # Percentages express the share of each split for current label.
            train_pct = (train_total / total) * 100.0
            dev_pct = (dev_total / total) * 100.0
        else:
            train_pct = dev_pct = 0.0
        yield label, train_total, dev_total, train_pct, dev_pct


def warn_missing_labels(train_counts: Counter, dev_counts: Counter) -> None:
    """Emit warnings for labels that only appear in one split."""
    train_only = sorted(set(train_counts) - set(dev_counts))
    dev_only = sorted(set(dev_counts) - set(train_counts))
    if train_only:
        print("Warning: labels only in train:", ", ".join(train_only))
    if dev_only:
        print("Warning: labels only in dev:", ", ".join(dev_only))


def main() -> None:
    """Execute the label distribution check and print the report."""
    args = parse_args()
    # Read both splits, collecting technique counts and article identifiers.
    train_counts, train_articles = load_labels(args.train)
    dev_counts, dev_articles = load_labels(args.dev)

    # Print an aligned table summarizing counts and percentages per label.
    header = f"{'Label':<35} {'Train':>10} {'Dev':>10} {'Train %':>10} {'Dev %':>10}"
    print(header)
    print("-" * len(header))
    for label, train_total, dev_total, train_pct, dev_pct in compute_summary(
        train_counts, dev_counts
    ):
        print(
            f"{label:<35} {train_total:>10d} {dev_total:>10d} {train_pct:>9.1f}% {dev_pct:>9.1f}%"
        )

    print()
    # Report the number of distinct articles included in each split.
    print(f"Total articles in train: {len(train_articles)}")
    print(f"Total articles in dev: {len(dev_articles)}")

    warn_missing_labels(train_counts, dev_counts)


if __name__ == "__main__":
    main()

