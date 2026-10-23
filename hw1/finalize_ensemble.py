#!/usr/bin/env python3
"""Finalize train-only OOF calibration after all ensemble components exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from train_linear import LABELS, load_train, summarize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("train.jsonl"))
    parser.add_argument(
        "--model-dir", type=Path, default=Path("artifacts/xlmr_final_model")
    )
    return parser.parse_args()


def exact_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    order = np.argsort(-probability)
    ordered_target = target[order]
    true_positive = np.cumsum(ordered_target)
    false_positive = np.cumsum(1 - ordered_target)
    false_negative = ordered_target.sum() - true_positive
    f1 = 2 * true_positive / np.maximum(
        2 * true_positive + false_positive + false_negative, 1
    )
    return float(probability[order[int(np.argmax(f1))]])


def main():
    args = parse_args()
    rows, targets = load_train(args.train_file)
    model_dir = args.model_dir
    config_path = model_dir / "inference_config.json"
    metrics_path = model_dir / "training_metrics.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    transformer = np.load(model_dir / "oof_transformer.npz")["probabilities"]
    linear = np.load(model_dir / "oof_plain_linear.npz")["probabilities"]
    context_linear = np.load(model_dir / "oof_context_linear.npz")["probabilities"]
    stacker = np.load(model_dir / "oof_stacker.npz")["probabilities"]
    alpha = np.asarray(config["blend_alpha_transformer"])
    probabilities = alpha * transformer + (1.0 - alpha) * linear
    thresholds = np.asarray(config["thresholds"], dtype=np.float64)
    predictions = np.zeros_like(targets, dtype=np.int8)
    special_labels = {
        label for label, setting in config.get("context_linear_labels", {}).items()
        if setting.get("enabled", False)
    } | {
        label for label, setting in config.get("stacker_labels", {}).items()
        if setting.get("enabled", False)
    }
    for label_idx, label in enumerate(LABELS):
        if label not in special_labels:
            thresholds[label_idx] = exact_threshold(
                targets[:, label_idx], probabilities[:, label_idx]
            )
            predictions[:, label_idx] = probabilities[:, label_idx] >= thresholds[label_idx]

    chinese = np.asarray([
        any("\u4e00" <= char <= "\u9fff" for char in row["utterance"])
        for row in rows
    ])
    for label, setting in config.get("context_linear_labels", {}).items():
        if not setting.get("enabled", False):
            continue
        label_idx = LABELS.index(label)
        blended = (
            float(setting["transformer_weight"]) * transformer[:, label_idx]
            + float(setting["plain_linear_weight"]) * linear[:, label_idx]
            + float(setting["context_linear_weight"]) * context_linear[:, label_idx]
        )
        predictions[:, label_idx] = np.where(
            chinese,
            blended >= float(setting["threshold_zh"]),
            blended >= float(setting["threshold_en"]),
        )

    # The nested meta-model is the stronger Doubt source and deliberately runs
    # after the context blend when both have an entry for the same label.
    for label, setting in config.get("stacker_labels", {}).items():
        if not setting.get("enabled", False):
            continue
        label_idx = LABELS.index(label)
        predictions[:, label_idx] = (
            stacker[:, label_idx] >= float(setting["threshold"])
        )

    final_metrics = summarize(targets, predictions)
    config["thresholds"] = thresholds.tolist()
    config["final_oof_validation"] = final_metrics
    config["ensemble_order"] = [
        "dual_seed_transformer_mean",
        "per_label_plain_linear_blend",
        "context_linear_blend",
    ]
    if any(
        setting.get("enabled", False)
        for setting in config.get("stacker_labels", {}).values()
    ):
        config["ensemble_order"].append("nested_oof_stacker")
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["final"] = final_metrics
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
