#!/usr/bin/env python3
"""Evaluate a context-aware TF-IDF model as a third complementary OOF source."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

from train_linear import (
    LABELS, decision_matrix, fit_classifiers, fit_transform_features, load_train,
    make_strata, select_c_per_label, summarize, transform_features,
)

WEAK_LABELS = ["Compare_Competitor", "Doubt", "Too_Expensive"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("train.jsonl"))
    parser.add_argument(
        "--model-dir", type=Path, default=Path("artifacts/xlmr_final_model")
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def context_text(row: dict) -> str:
    parts = [f"[CURRENT_CUSTOMER]\n{row['utterance']}"]
    context = row.get("context", [])[-2:]
    if context:
        recent = "\n".join(
            f"[{message.get('role', '').upper()}]\n{message.get('content', '')}"
            for message in context
        )
        parts.append("[RECENT_DIALOGUE]\n" + recent[-1200:])
    return "\n".join(parts)


def oof_linear(texts, targets, splits, c_candidates):
    by_c = {c: np.zeros_like(targets, dtype=np.float32) for c in c_candidates}
    weak_indices = [LABELS.index(label) for label in WEAK_LABELS]
    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        train_texts = [texts[i] for i in train_idx]
        valid_texts = [texts[i] for i in valid_idx]
        cv, wv, train_x = fit_transform_features(train_texts)
        valid_x = transform_features(valid_texts, cv, wv)
        for c in c_candidates:
            for label_idx in weak_indices:
                classifier = LinearSVC(
                    C=c, class_weight="balanced", dual="auto", max_iter=10_000,
                    random_state=42,
                )
                classifier.fit(train_x, targets[train_idx, label_idx])
                by_c[c][valid_idx, label_idx] = classifier.decision_function(valid_x)
        print(f"Finished context-linear OOF fold {fold}/{len(splits)}", flush=True)
        del cv, wv, train_x, valid_x
        gc.collect()
    selected_c = [1.0] * len(LABELS)
    decisions = np.zeros_like(targets, dtype=np.float32)
    for label_idx in weak_indices:
        best = (-1.0, None)
        for c in c_candidates:
            probabilities = expit(by_c[c][:, label_idx])
            score = max(
                f1_score(targets[:, label_idx], probabilities >= threshold, zero_division=0)
                for threshold in np.linspace(0.05, 0.95, 91)
            )
            if score > best[0]:
                best = (score, c)
        selected_c[label_idx] = float(best[1])
        decisions[:, label_idx] = by_c[best[1]][:, label_idx]
    return selected_c, expit(decisions).astype(np.float32)


def reconstruct_plain_oof(texts, targets, splits, selected_c):
    decisions = np.zeros_like(targets, dtype=np.float32)
    for fold, (train_idx, valid_idx) in enumerate(splits, 1):
        cv, wv, train_x = fit_transform_features([texts[i] for i in train_idx])
        valid_x = transform_features([texts[i] for i in valid_idx], cv, wv)
        classifiers = fit_classifiers(train_x, targets[train_idx], selected_c)
        decisions[valid_idx] = decision_matrix(classifiers, valid_x)
        print(f"Reconstructed plain-linear OOF fold {fold}/{len(splits)}", flush=True)
        del cv, wv, train_x, valid_x, classifiers
        gc.collect()
    return expit(decisions).astype(np.float32)


def _exact_threshold(target, probability, mask):
    indices = np.flatnonzero(mask)
    order = indices[np.argsort(-probability[indices])]
    ordered_target = target[order]
    true_positive = np.cumsum(ordered_target)
    false_positive = np.cumsum(1 - ordered_target)
    false_negative = ordered_target.sum() - true_positive
    scores = 2 * true_positive / np.maximum(
        2 * true_positive + false_positive + false_negative, 1
    )
    best = int(np.argmax(scores))
    return float(probability[order[best]])


def tune_three(target, transformer, plain, context, chinese):
    best = (-1.0, None)
    weights = np.linspace(0.0, 1.0, 41)
    for wt in weights:
        for wc in weights:
            if wt + wc > 1.00001:
                continue
            wp = 1.0 - wt - wc
            probability = wt * transformer + wp * plain + wc * context
            threshold_zh = _exact_threshold(target, probability, chinese)
            threshold_en = _exact_threshold(target, probability, ~chinese)
            prediction = np.where(
                chinese, probability >= threshold_zh, probability >= threshold_en
            )
            score = f1_score(target, prediction, zero_division=0)
            if score > best[0]:
                best = (
                    float(score),
                    (float(wt), float(wp), float(wc), threshold_zh, threshold_en),
                )
    return best


def main():
    args = parse_args()
    rows, targets = load_train(args.train_file)
    plain_texts = [row["utterance"] for row in rows]
    context_texts = [context_text(row) for row in rows]
    chinese = np.asarray([
        any("\u4e00" <= char <= "\u9fff" for char in row["utterance"])
        for row in rows
    ])
    config_path = args.model_dir / "inference_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    transformer = np.load(args.model_dir / "oof_transformer.npz")["probabilities"].astype(np.float32)
    linear_artifact = joblib.load(args.model_dir / "linear_model.joblib")
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(np.arange(len(rows)), make_strata(targets, args.folds)))
    cache_path = args.model_dir / "oof_plain_linear.npz"
    if cache_path.is_file():
        cached = np.load(cache_path)
        plain = cached["probabilities"].astype(np.float32)
        if not np.array_equal(cached["targets"], targets):
            raise ValueError("Plain OOF cache target mismatch")
    else:
        plain = reconstruct_plain_oof(
            plain_texts, targets, splits, [float(x) for x in linear_artifact["selected_c"]]
        )
        np.savez_compressed(cache_path, probabilities=plain, targets=targets)
    selected_c, context = oof_linear(context_texts, targets, splits, [0.6, 1.2, 2.4, 4.0])
    alpha = np.asarray(config["blend_alpha_transformer"], dtype=np.float32)
    base_prob = alpha * transformer + (1.0 - alpha) * plain
    thresholds = np.asarray(config["thresholds"], dtype=np.float32)
    predictions = (base_prob >= thresholds).astype(np.int8)
    baseline = summarize(targets, predictions)
    choices = {}
    blend_weights = {}
    for label in WEAK_LABELS:
        idx = LABELS.index(label)
        score, values = tune_three(
            targets[:, idx], transformer[:, idx], plain[:, idx], context[:, idx], chinese
        )
        wt, wp, wc, threshold_zh, threshold_en = values
        enabled = score > baseline["per_label_f1"][label] + 0.002
        if enabled:
            probability = wt * transformer[:, idx] + wp * plain[:, idx] + wc * context[:, idx]
            predictions[:, idx] = np.where(
                chinese, probability >= threshold_zh, probability >= threshold_en
            )
        choices[label] = {
            "enabled": enabled,
            "baseline_f1": baseline["per_label_f1"][label],
            "context_blend_f1": score,
            "transformer_weight": wt,
            "plain_linear_weight": wp,
            "context_linear_weight": wc,
            "threshold_zh": threshold_zh,
            "threshold_en": threshold_en,
        }
        blend_weights[label] = values
        print(label, json.dumps(choices[label]), flush=True)
    enhanced = summarize(targets, predictions)
    result = {"baseline": baseline, "context_blend": enhanced, "labels": choices}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    np.savez_compressed(
        args.model_dir / "oof_context_linear.npz", probabilities=context, targets=targets
    )
    if args.apply:
        if not (enhanced["macro_f1"] >= baseline["macro_f1"] and enhanced["micro_f1"] >= baseline["micro_f1"]):
            raise RuntimeError("Context model did not improve both OOF aggregates")
        cv, wv, full_x = fit_transform_features(context_texts)
        classifiers = {}
        for label in WEAK_LABELS:
            label_idx = LABELS.index(label)
            classifier = LinearSVC(
                C=selected_c[label_idx], class_weight="balanced", dual="auto",
                max_iter=10_000, random_state=42,
            )
            classifier.fit(full_x, targets[:, label_idx])
            classifiers[label] = classifier
        artifact = {
            "labels": LABELS, "char_vectorizer": cv, "word_vectorizer": wv,
            "rule_patterns": linear_artifact["rule_patterns"], "rule_scale": 2.0,
            "selected_c": selected_c, "classifiers": classifiers,
            "input_format": "current_plus_recent_dialogue_v1",
        }
        artifact_path = args.model_dir / "context_linear_model.joblib"
        joblib.dump(artifact, artifact_path, compress=3)
        config["context_linear_model_file"] = artifact_path.name
        config["context_linear_labels"] = choices
        config["context_linear_oof_validation"] = enhanced
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        metrics_path = args.model_dir / "training_metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["context_blend"] = enhanced
        metrics["context_linear_labels"] = choices
        metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Applied", artifact_path)


if __name__ == "__main__":
    main()
