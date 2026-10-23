#!/usr/bin/env python3
"""Tune bilingual OOF blends for Micro-F1 subject to Macro-F1 >= 0.82."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from train_linear import LABELS, load_train, summarize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("train.jsonl"))
    parser.add_argument(
        "--model-dir", type=Path, default=Path("artifacts/xlmr_final_model")
    )
    parser.add_argument("--macro-floor", type=float, default=0.82)
    return parser.parse_args()


def exact_threshold(target, probability, mask):
    indices = np.flatnonzero(mask)
    order = indices[np.argsort(-probability[indices])]
    ordered_target = target[order]
    true_positive = np.cumsum(ordered_target)
    false_positive = np.cumsum(1 - ordered_target)
    false_negative = ordered_target.sum() - true_positive
    score = 2 * true_positive / np.maximum(
        2 * true_positive + false_positive + false_negative, 1
    )
    return float(probability[order[int(np.argmax(score))]])


def counts(target, prediction):
    target_bool = target.astype(bool)
    true_positive = (prediction & target_bool).sum(axis=0)
    false_positive = (prediction & ~target_bool).sum(axis=0)
    false_negative = (~prediction & target_bool).sum(axis=0)
    per_label = 2 * true_positive / np.maximum(
        2 * true_positive + false_positive + false_negative, 1
    )
    macro = per_label.mean()
    micro = 2 * true_positive.sum() / (
        2 * true_positive.sum() + false_positive.sum() + false_negative.sum()
    )
    return true_positive, false_positive, false_negative, per_label, macro, micro


def main():
    args = parse_args()
    rows, targets = load_train(args.train_file)
    target_bool = targets.astype(bool)
    chinese = np.asarray([
        any("\u4e00" <= char <= "\u9fff" for char in row["utterance"])
        for row in rows
    ])
    model_dir = args.model_dir
    config_path = model_dir / "inference_config.json"
    metrics_path = model_dir / "training_metrics.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    transformer = np.load(model_dir / "oof_transformer.npz")["probabilities"]
    linear = np.load(model_dir / "oof_plain_linear.npz")["probabilities"]
    context = np.load(model_dir / "oof_context_linear.npz")["probabilities"]
    stacker = np.load(model_dir / "oof_stacker.npz")["probabilities"]

    # Disable the context replacement only when a validated stacker actually
    # supersedes it. XLM-R's stacker is a negative result, so its stronger
    # context-linear Doubt path must remain eligible.
    doubt_stacker = config.get("stacker_labels", {}).get("Doubt")
    if (
        doubt_stacker and doubt_stacker.get("enabled", False)
        and "Doubt" in config.get("context_linear_labels", {})
    ):
        config["context_linear_labels"]["Doubt"]["enabled"] = False
    context_labels = {
        label for label, setting in config.get("context_linear_labels", {}).items()
        if setting.get("enabled", False)
    }
    stacker_labels = {
        label for label, setting in config.get("stacker_labels", {}).items()
        if setting.get("enabled", False)
    }
    special_labels = context_labels | stacker_labels
    probabilities = np.zeros_like(transformer, dtype=np.float32)
    predictions = np.zeros_like(targets, dtype=bool)
    language_settings = {}
    thresholds = {}

    # Independently search Transformer/linear weights for each language, while
    # scoring the complete label so that neither language is optimized in isolation.
    for label_idx, label in enumerate(LABELS):
        if label in special_labels:
            continue
        best = (-1.0, None)
        for alpha_zh in np.linspace(0.0, 1.0, 21):
            probability_zh = (
                alpha_zh * transformer[:, label_idx]
                + (1.0 - alpha_zh) * linear[:, label_idx]
            )
            threshold_zh = exact_threshold(
                targets[:, label_idx], probability_zh, chinese
            )
            for alpha_en in np.linspace(0.0, 1.0, 21):
                probability_en = (
                    alpha_en * transformer[:, label_idx]
                    + (1.0 - alpha_en) * linear[:, label_idx]
                )
                threshold_en = exact_threshold(
                    targets[:, label_idx], probability_en, ~chinese
                )
                prediction = np.where(
                    chinese,
                    probability_zh >= threshold_zh,
                    probability_en >= threshold_en,
                )
                score = f1_score(targets[:, label_idx], prediction, zero_division=0)
                if score > best[0]:
                    best = (float(score), (
                        float(alpha_zh), float(alpha_en),
                        threshold_zh, threshold_en,
                    ))
        alpha_zh, alpha_en, threshold_zh, threshold_en = best[1]
        probabilities[:, label_idx] = np.where(
            chinese,
            alpha_zh * transformer[:, label_idx] + (1.0 - alpha_zh) * linear[:, label_idx],
            alpha_en * transformer[:, label_idx] + (1.0 - alpha_en) * linear[:, label_idx],
        )
        predictions[:, label_idx] = np.where(
            chinese,
            probabilities[:, label_idx] >= threshold_zh,
            probabilities[:, label_idx] >= threshold_en,
        )
        language_settings[label] = {
            "alpha_transformer_zh": alpha_zh,
            "alpha_transformer_en": alpha_en,
            "threshold_zh": threshold_zh,
            "threshold_en": threshold_en,
        }
        thresholds[label] = [threshold_zh, threshold_en]

    for label in context_labels:
        label_idx = LABELS.index(label)
        setting = config["context_linear_labels"][label]
        probabilities[:, label_idx] = (
            float(setting["transformer_weight"]) * transformer[:, label_idx]
            + float(setting["plain_linear_weight"]) * linear[:, label_idx]
            + float(setting["context_linear_weight"]) * context[:, label_idx]
        )
        thresholds[label] = [
            float(setting["threshold_zh"]), float(setting["threshold_en"])
        ]
        predictions[:, label_idx] = np.where(
            chinese,
            probabilities[:, label_idx] >= thresholds[label][0],
            probabilities[:, label_idx] >= thresholds[label][1],
        )

    for label in stacker_labels:
        label_idx = LABELS.index(label)
        probability = stacker[:, label_idx]
        probabilities[:, label_idx] = probability
        old_threshold = float(config["stacker_labels"][label]["threshold"])
        thresholds[label] = [old_threshold, old_threshold]
        predictions[:, label_idx] = probability >= old_threshold

    # Coordinate ascent trades a small amount of per-label Macro for Micro,
    # but never permits aggregate Macro-F1 to fall below the assignment target.
    for _cycle in range(8):
        changed = False
        for label_idx, label in enumerate(LABELS):
            for language_idx, mask in enumerate((chinese, ~chinese)):
                tp0, fp0, fn0, f10, _macro0, micro0 = counts(targets, predictions)
                indices = np.flatnonzero(mask)
                order = indices[np.argsort(-probabilities[indices, label_idx])]
                ordered_target = target_bool[order, label_idx].astype(np.int8)
                group_tp = np.cumsum(ordered_target)
                group_fp = np.cumsum(1 - ordered_target)
                group_fn = ordered_target.sum() - group_tp
                other = ~mask
                other_tp = (predictions[other, label_idx] & target_bool[other, label_idx]).sum()
                other_fp = (predictions[other, label_idx] & ~target_bool[other, label_idx]).sum()
                other_fn = (~predictions[other, label_idx] & target_bool[other, label_idx]).sum()
                candidate_tp = other_tp + group_tp
                candidate_fp = other_fp + group_fp
                candidate_fn = other_fn + group_fn
                candidate_f1 = 2 * candidate_tp / np.maximum(
                    2 * candidate_tp + candidate_fp + candidate_fn, 1
                )
                macro = (f10.sum() - f10[label_idx] + candidate_f1) / len(LABELS)
                total_tp = tp0.sum() - tp0[label_idx] + candidate_tp
                total_fp = fp0.sum() - fp0[label_idx] + candidate_fp
                total_fn = fn0.sum() - fn0[label_idx] + candidate_fn
                micro = 2 * total_tp / (2 * total_tp + total_fp + total_fn)
                valid = macro >= args.macro_floor
                if not valid.any():
                    continue
                best = int(np.argmax(np.where(valid, micro, -1.0)))
                candidate_threshold = float(probabilities[order[best], label_idx])
                old_column = predictions[:, label_idx].copy()
                predictions[mask, label_idx] = (
                    probabilities[mask, label_idx] >= candidate_threshold
                )
                new_micro = counts(targets, predictions)[5]
                if new_micro > micro0 + 1e-12:
                    thresholds[label][language_idx] = candidate_threshold
                    changed = True
                else:
                    predictions[:, label_idx] = old_column
        if not changed:
            break

    for label, setting in language_settings.items():
        setting["threshold_zh"], setting["threshold_en"] = thresholds[label]
    for label in context_labels:
        setting = config["context_linear_labels"][label]
        setting["threshold_zh"], setting["threshold_en"] = thresholds[label]
    for label in stacker_labels:
        setting = config["stacker_labels"][label]
        setting["threshold_zh"], setting["threshold_en"] = thresholds[label]

    final_metrics = summarize(targets, predictions.astype(np.int8))
    if final_metrics["macro_f1"] < args.macro_floor:
        raise RuntimeError("Constrained calibration violated its Macro-F1 floor")
    config["language_blend_labels"] = language_settings
    config["final_oof_validation"] = final_metrics
    config["calibration"] = "language_specific_blend_and_macro_constrained_micro"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["final"] = final_metrics
    metrics["language_blend_labels"] = language_settings
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
