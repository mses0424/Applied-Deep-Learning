#!/usr/bin/env python3
"""One-split Transformer comparison for the report, using train.jsonl only."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from accelerate import Accelerator
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

import train_transformer as trainer

LABELS = trainer.LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("train.jsonl"))
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--head-learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def strata_for_split(targets: np.ndarray) -> list[str]:
    combinations = ["".join(str(int(x)) for x in row) for row in targets]
    counts = Counter(combinations)
    return [value if counts[value] >= 2 else "rare" for value in combinations]


def summarize(
    targets: np.ndarray,
    predictions: np.ndarray,
    mask: np.ndarray,
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


def grouped_metrics(
    rows: list[dict],
    targets: np.ndarray,
    predictions: np.ndarray,
    valid_indices: np.ndarray,
) -> dict:
    valid_rows = [rows[index] for index in valid_indices]
    texts = [row["utterance"] for row in valid_rows]
    chinese = np.asarray([
        any("\u4e00" <= character <= "\u9fff" for character in text)
        for text in texts
    ])
    long_mask = np.asarray([len(text) > 420 for text in texts])
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
    valid_targets = targets[valid_indices]
    return {
        name: summarize(valid_targets, predictions, mask)
        for name, mask in masks.items()
    }


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
    )
    trainer.set_seed(args.seed + accelerator.process_index)
    rows, targets = trainer.load_train(args.train_file)
    indices = np.arange(len(rows))
    train_indices, valid_indices = train_test_split(
        indices,
        test_size=args.valid_size,
        random_state=args.seed,
        stratify=strata_for_split(targets),
    )
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model, optimizer, train_loader, valid_loader, scheduler = trainer.train_model(
        accelerator,
        args,
        tokenizer,
        rows,
        targets,
        train_indices,
        valid_indices,
        description=f"Report split: {args.model_name}",
    )
    probabilities = trainer.collect_validation(
        accelerator, model, valid_loader, len(rows)
    )
    if accelerator.is_main_process:
        valid_probabilities = probabilities[valid_indices]
        fixed_predictions = valid_probabilities >= 0.5
        tuned_thresholds = trainer.tune_thresholds(
            targets[valid_indices], valid_probabilities
        )
        tuned_predictions = valid_probabilities >= tuned_thresholds
        result = {
            "protocol": {
                "data": str(args.train_file),
                "model_name": args.model_name,
                "seed": args.seed,
                "train_rows": len(train_indices),
                "valid_rows": len(valid_indices),
                "epochs": args.epochs,
                "max_length": args.max_length,
                "views": "head-tail utterance and recent-context view averaged",
                "note": (
                    "Use fixed_0.5 for fair cross-model comparison. "
                    "tuned_on_same_validation is diagnostic and optimistic."
                ),
            },
            "fixed_0.5": grouped_metrics(
                rows, targets, fixed_predictions, valid_indices
            ),
            "tuned_on_same_validation": {
                "thresholds": {
                    label: float(value)
                    for label, value in zip(LABELS, tuned_thresholds)
                },
                "metrics": grouped_metrics(
                    rows, targets, tuned_predictions, valid_indices
                ),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        fixed = result["fixed_0.5"]["all"]
        print(
            f"{args.model_name}: fixed threshold macro={fixed['macro_f1']:.4f} "
            f"micro={fixed['micro_f1']:.4f}"
        )
        print("Saved", args.output)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
