#!/usr/bin/env python3
"""Train a leakage-conscious OOF meta-classifier for the three weak labels."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.special import expit, logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from train_linear import (
    LABELS,
    decision_matrix,
    fit_classifiers,
    fit_transform_features,
    load_train,
    make_strata,
    rule_features,
    transform_features,
)

WEAK_LABELS = ["Compare_Competitor", "Doubt", "Too_Expensive"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("train.jsonl"))
    parser.add_argument(
        "--model-dir", type=Path, default=Path("artifacts/xlmr_final_model")
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--apply", action="store_true",
        help="Save the validated stacker and enable it in inference_config.json.",
    )
    return parser.parse_args()


def summarize(targets: np.ndarray, predictions: np.ndarray) -> dict:
    report = classification_report(
        targets, predictions, target_names=LABELS, output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": float(f1_score(targets, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(targets, predictions, average="micro", zero_division=0)),
        "per_label_f1": {
            label: float(report[label]["f1-score"]) for label in LABELS
        },
    }


def make_stack_features(
    transformer: np.ndarray, linear: np.ndarray, texts: list[str]
) -> np.ndarray:
    """Combine both model views, disagreement, and observable text cues."""
    eps = 1e-5
    transformer = np.clip(transformer, eps, 1.0 - eps)
    linear = np.clip(linear, eps, 1.0 - eps)
    rules = rule_features(texts).toarray().astype(np.float32)
    return np.column_stack(
        [
            transformer,
            linear,
            logit(transformer),
            logit(linear),
            transformer * linear,
            np.abs(transformer - linear),
            rules,
        ]
    ).astype(np.float32)


def best_threshold(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    best_score = -1.0
    best_value = 0.5
    for threshold in np.linspace(0.05, 0.95, 181):
        score = f1_score(target, probability >= threshold, zero_division=0)
        if score > best_score:
            best_score = float(score)
            best_value = float(threshold)
    return best_value, best_score


def new_classifier(c_value: float, class_weight: str | None):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight=class_weight,
            solver="liblinear",
            max_iter=3000,
            random_state=42,
        ),
    )


def reconstruct_linear_oof(
    texts: list[str], targets: np.ndarray, selected_c: list[float],
    folds: int, seed: int,
) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.arange(len(texts)), make_strata(targets, folds)))
    decisions = np.zeros_like(targets, dtype=np.float32)
    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        train_texts = [texts[index] for index in train_indices]
        valid_texts = [texts[index] for index in valid_indices]
        char_vectorizer, word_vectorizer, train_features = fit_transform_features(train_texts)
        valid_features = transform_features(valid_texts, char_vectorizer, word_vectorizer)
        classifiers = fit_classifiers(train_features, targets[train_indices], selected_c)
        decisions[valid_indices] = decision_matrix(classifiers, valid_features)
        print(f"Reconstructed linear OOF fold {fold}/{folds}", flush=True)
        del char_vectorizer, word_vectorizer, train_features, valid_features, classifiers
        gc.collect()
    return expit(decisions).astype(np.float32)


def main() -> None:
    args = parse_args()
    config_path = args.model_dir / "inference_config.json"
    metrics_path = args.model_dir / "training_metrics.json"
    oof_path = args.model_dir / "oof_transformer.npz"
    linear_path = args.model_dir / "linear_model.joblib"
    for path in (config_path, metrics_path, oof_path, linear_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    rows, targets = load_train(args.train_file)
    texts = [row["utterance"] for row in rows]
    transformer_data = np.load(oof_path)
    transformer = transformer_data["probabilities"].astype(np.float32)
    if not np.array_equal(targets, transformer_data["targets"].astype(np.int8)):
        raise ValueError("train.jsonl does not match oof_transformer.npz")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    linear_artifact = joblib.load(linear_path)
    plain_oof_path = args.model_dir / "oof_plain_linear.npz"
    if plain_oof_path.is_file():
        plain_data = np.load(plain_oof_path)
        linear = plain_data["probabilities"].astype(np.float32)
        if not np.array_equal(plain_data["targets"].astype(np.int8), targets):
            raise ValueError("Plain linear OOF cache target mismatch")
    else:
        linear = reconstruct_linear_oof(
            texts, targets, [float(x) for x in linear_artifact["selected_c"]],
            args.folds, args.seed,
        )

    alpha = np.asarray(config["blend_alpha_transformer"], dtype=np.float32)
    base_probability = alpha * transformer + (1.0 - alpha) * linear
    base_thresholds = np.asarray(config["thresholds"], dtype=np.float32)
    base_predictions = (base_probability >= base_thresholds).astype(np.int8)
    base_metrics = summarize(targets, base_predictions)
    saved_base = config["blended_oof_validation"]
    delta = max(
        abs(base_metrics["macro_f1"] - saved_base["macro_f1"]),
        abs(base_metrics["micro_f1"] - saved_base["micro_f1"]),
    )
    if delta > 1e-9:
        raise RuntimeError(
            f"Reconstructed baseline mismatch ({delta:.3g}); refusing to tune on a different base."
        )
    print(
        f"Verified baseline: Macro-F1={base_metrics['macro_f1']:.6f}, "
        f"Micro-F1={base_metrics['micro_f1']:.6f}", flush=True,
    )

    features = make_stack_features(transformer, linear, texts)
    meta_splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed + 3365
    )
    strata = make_strata(targets, args.folds)
    splits = list(meta_splitter.split(np.arange(len(rows)), strata))
    candidates = [
        (c_value, class_weight)
        for class_weight in (None, "balanced")
        for c_value in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
    ]
    selected: dict[str, dict] = {}
    final_classifiers: dict[str, object] = {}
    hybrid_predictions = base_predictions.copy()
    stacker_oof = np.zeros_like(targets, dtype=np.float32)

    for label in WEAK_LABELS:
        label_idx = LABELS.index(label)
        best = None
        for c_value, class_weight in candidates:
            oof_probability = np.zeros(len(rows), dtype=np.float32)
            for train_indices, valid_indices in splits:
                classifier = new_classifier(c_value, class_weight)
                classifier.fit(features[train_indices], targets[train_indices, label_idx])
                oof_probability[valid_indices] = classifier.predict_proba(
                    features[valid_indices]
                )[:, 1]
            threshold, score = best_threshold(targets[:, label_idx], oof_probability)
            candidate = (score, -abs(threshold - 0.5), -c_value, c_value, class_weight, threshold, oof_probability)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        score, _, _, c_value, class_weight, threshold, oof_probability = best
        stacker_oof[:, label_idx] = oof_probability
        old_score = base_metrics["per_label_f1"][label]
        use = score > old_score + 0.002
        if use:
            hybrid_predictions[:, label_idx] = (oof_probability >= threshold).astype(np.int8)
        final_classifier = new_classifier(c_value, class_weight)
        final_classifier.fit(features, targets[:, label_idx])
        final_classifiers[label] = final_classifier
        selected[label] = {
            "enabled": bool(use),
            "baseline_f1": float(old_score),
            "stacker_oof_f1": float(score),
            "threshold": float(threshold),
            "C": float(c_value),
            "class_weight": class_weight,
        }
        print(
            f"{label}: {old_score:.6f} -> {score:.6f}; "
            f"enabled={use}, C={c_value}, class_weight={class_weight}, threshold={threshold:.3f}",
            flush=True,
        )

    hybrid_metrics = summarize(targets, hybrid_predictions)
    improved = (
        any(item["enabled"] for item in selected.values())
        and hybrid_metrics["macro_f1"] >= base_metrics["macro_f1"]
        and hybrid_metrics["micro_f1"] >= base_metrics["micro_f1"]
    )
    result = {
        "baseline_blend": base_metrics,
        "stacked_hybrid": hybrid_metrics,
        "labels": selected,
        "validated_improvement": bool(improved),
    }
    print("\nTrain-only nested OOF result")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    np.savez_compressed(
        args.model_dir / "oof_stacker.npz",
        probabilities=stacker_oof,
        targets=targets,
    )

    if args.apply:
        if not improved:
            raise RuntimeError("Stacker did not improve both aggregate OOF metrics; not applying.")
        artifact = {
            "labels": LABELS,
            "weak_labels": WEAK_LABELS,
            "classifiers": final_classifiers,
            "selection": selected,
            "feature_version": 1,
            "sklearn_version": sklearn.__version__,
        }
        artifact_path = args.model_dir / "stacker_model.joblib"
        joblib.dump(artifact, artifact_path, compress=3)
        config["stacker_model_file"] = artifact_path.name
        config["stacker_labels"] = selected
        config["stacked_oof_validation"] = hybrid_metrics
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["stacked"] = hybrid_metrics
        metrics["stacker_labels"] = selected
        metrics_path.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Applied validated stacker: {artifact_path}")
    else:
        print("Dry run only. Re-run with --apply to enable this exact method.")


if __name__ == "__main__":
    main()
