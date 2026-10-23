#!/usr/bin/env python3
"""Train-only report ablations for model inputs, context, and long messages."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import train_linear as strong

LABELS = strong.LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("reports/linear_ablations.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    return parser.parse_args()


def load_data(path: Path) -> tuple[list[dict], np.ndarray]:
    return strong.load_train(path)


def make_strata(targets: np.ndarray) -> list[str]:
    combinations = ["".join(str(int(x)) for x in row) for row in targets]
    counts = Counter(combinations)
    return [value if counts[value] >= 2 else "rare" for value in combinations]


def compact(text: str, policy: str, max_chars: int = 420) -> str:
    if len(text) <= max_chars or policy == "full":
        return text
    if policy == "head":
        return text[:max_chars]
    if policy == "tail":
        return text[-max_chars:]
    if policy == "head_tail":
        return f"{text[:140]}\n[... middle omitted ...]\n{text[-260:]}"
    raise ValueError(policy)


def context_text(row: dict, mode: str) -> str:
    messages = row.get("context", [])
    if mode == "none" or not messages:
        return ""
    if mode == "recent":
        messages = messages[-2:]
        limit = 1200
    elif mode == "full":
        limit = 1800
    else:
        raise ValueError(mode)
    joined = "\n".join(
        f"{message.get('role', '')}: {message.get('content', '')}"
        for message in messages
    )
    return joined[-limit:]


def make_features(
    train_rows: list[dict],
    valid_rows: list[dict],
    *,
    policy: str,
    use_word: bool,
    use_rules: bool,
    context_mode: str,
):
    train_texts = [compact(row["utterance"], policy) for row in train_rows]
    valid_texts = [compact(row["utterance"], policy) for row in valid_rows]
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=2,
        max_df=0.995,
        max_features=180_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_parts = [char_vectorizer.fit_transform(train_texts)]
    valid_parts = [char_vectorizer.transform(valid_texts)]
    if use_word:
        word_vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.995,
            max_features=100_000,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
        train_parts.append(word_vectorizer.fit_transform(train_texts))
        valid_parts.append(word_vectorizer.transform(valid_texts))
    if use_rules:
        train_parts.append(strong.rule_features(train_texts))
        valid_parts.append(strong.rule_features(valid_texts))
    if context_mode != "none":
        train_context = [context_text(row, context_mode) for row in train_rows]
        valid_context = [context_text(row, context_mode) for row in valid_rows]
        context_vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 4),
            min_df=3,
            max_features=80_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        train_parts.append(context_vectorizer.fit_transform(train_context) * 0.35)
        valid_parts.append(context_vectorizer.transform(valid_context) * 0.35)
    return (
        hstack(train_parts, format="csr"),
        hstack(valid_parts, format="csr"),
    )


def score_subset(
    targets: np.ndarray, predictions: np.ndarray, mask: np.ndarray
) -> dict | None:
    if int(mask.sum()) == 0:
        return None
    true = targets[mask]
    pred = predictions[mask]
    report = classification_report(
        true,
        pred,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "count": int(mask.sum()),
        "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(true, pred, average="micro", zero_division=0)),
        "per_label_f1": {
            label: float(report[label]["f1-score"]) for label in LABELS
        },
    }


def fit_variant(
    train_rows: list[dict],
    valid_rows: list[dict],
    y_train: np.ndarray,
    y_valid: np.ndarray,
    config: dict,
    seed: int,
) -> np.ndarray:
    x_train, x_valid = make_features(
        train_rows,
        valid_rows,
        policy=config["policy"],
        use_word=config["use_word"],
        use_rules=config["use_rules"],
        context_mode=config["context_mode"],
    )
    predictions = np.zeros_like(y_valid, dtype=np.int8)
    for label_idx in range(len(LABELS)):
        classifier = LinearSVC(
            C=1.7,
            class_weight="balanced",
            dual="auto",
            max_iter=10_000,
            random_state=seed,
        )
        classifier.fit(x_train, y_train[:, label_idx])
        predictions[:, label_idx] = (
            classifier.decision_function(x_valid) >= 0.0
        )
    del x_train, x_valid
    gc.collect()
    return predictions


def main() -> None:
    args = parse_args()
    rows, targets = load_data(args.train_file)
    indices = np.arange(len(rows))
    train_idx, valid_idx = train_test_split(
        indices,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=make_strata(targets),
    )
    train_rows = [rows[index] for index in train_idx]
    valid_rows = [rows[index] for index in valid_idx]
    y_train = targets[train_idx]
    y_valid = targets[valid_idx]

    variants = {
        "char_only": {
            "policy": "full", "use_word": False, "use_rules": False,
            "context_mode": "none",
        },
        "char_word": {
            "policy": "full", "use_word": True, "use_rules": False,
            "context_mode": "none",
        },
        "char_word_rules": {
            "policy": "full", "use_word": True, "use_rules": True,
            "context_mode": "none",
        },
        "full_context": {
            "policy": "full", "use_word": True, "use_rules": False,
            "context_mode": "full",
        },
        "recent_context": {
            "policy": "full", "use_word": True, "use_rules": False,
            "context_mode": "recent",
        },
        "long_head": {
            "policy": "head", "use_word": True, "use_rules": False,
            "context_mode": "none",
        },
        "long_tail": {
            "policy": "tail", "use_word": True, "use_rules": False,
            "context_mode": "none",
        },
        "long_head_tail": {
            "policy": "head_tail", "use_word": True, "use_rules": False,
            "context_mode": "none",
        },
    }

    valid_texts = [row["utterance"] for row in valid_rows]
    chinese = np.asarray([
        any("\u4e00" <= character <= "\u9fff" for character in text)
        for text in valid_texts
    ])
    long_mask = np.asarray([len(text) > 420 for text in valid_texts])
    context_mask = np.asarray([bool(row.get("context")) for row in valid_rows])
    masks = {
        "all": np.ones(len(valid_rows), dtype=bool),
        "chinese": chinese,
        "english": ~chinese,
        "long_gt_420_chars": long_mask,
        "short_le_420_chars": ~long_mask,
        "has_context": context_mask,
        "no_context": ~context_mask,
    }

    results = {
        "protocol": {
            "data": str(args.train_file),
            "seed": args.seed,
            "train_rows": len(train_idx),
            "valid_rows": len(valid_idx),
            "threshold": "fixed LinearSVC decision threshold 0.0",
            "note": "All comparisons use the same train-only split; no test labels.",
        },
        "variants": {},
    }
    for name, config in variants.items():
        print(f"Running {name}...", flush=True)
        predictions = fit_variant(
            train_rows, valid_rows, y_train, y_valid, config, args.seed
        )
        results["variants"][name] = {
            "config": config,
            "metrics": {
                mask_name: score_subset(y_valid, predictions, mask)
                for mask_name, mask in masks.items()
            },
        }
        metrics = results["variants"][name]["metrics"]["all"]
        print(
            f"{name}: macro={metrics['macro_f1']:.4f} "
            f"micro={metrics['micro_f1']:.4f}",
            flush=True,
        )

    base = results["variants"]["char_word"]["metrics"]["all"]["per_label_f1"]
    recent = results["variants"]["recent_context"]["metrics"]["all"]["per_label_f1"]
    results["context_per_label_delta_recent_minus_no_context"] = {
        label: float(recent[label] - base[label]) for label in LABELS
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Saved", args.output)


if __name__ == "__main__":
    main()
