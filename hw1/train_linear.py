#!/usr/bin/env python3
"""Train TF-IDF/SVM, tune an OOF ensemble, and update inference metadata."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.special import expit
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

LABELS = [
    "Ask_Price", "Ask_Service", "Ask_Spec", "Compare_Competitor",
    "Confirm_Order", "Doubt", "Need_More_Evidence", "Need_Time_To_Think",
    "State_Need", "Too_Expensive",
]

# Observable bilingual cues derived from the assignment label definitions.
# They are features learned by LinearSVC, not hard prediction overrides.
RULE_PATTERNS = {
    "Ask_Price": [
        "price", "how much", "cost", "fee", "premium", "discount", "deal",
        "\u50f9\u683c", "\u4ef7\u94b1", "\u591a\u5c11\u9322", "\u591a\u5c11\u94b1", "\u8cbb\u7528", "\u8d39\u7528", "\u4fdd\u8cbb", "\u4fdd\u8d39",
        "\u512a\u60e0", "\u4f18\u60e0", "\u6298\u6263", "\u4fc3\u92b7", "\u4fc3\u9500",
    ],
    "Ask_Service": [
        "delivery", "shipping", "arrive", "warranty", "return", "refund",
        "repair", "install", "support", "customer service", "claim process",
        "\u914d\u9001", "\u5230\u8ca8", "\u5230\u8d27", "\u7269\u6d41", "\u904b\u8cbb", "\u8fd0\u8d39", "\u4fdd\u56fa", "\u8cea\u4fdd", "\u8d28\u4fdd",
        "\u9000\u63db", "\u9000\u6362", "\u9000\u6b3e", "\u7dad\u4fee", "\u7ef4\u4fee", "\u5b89\u88dd", "\u5b89\u88c5", "\u5ba2\u670d",
        "\u552e\u5f8c", "\u552e\u540e", "\u7406\u8ce0", "\u7406\u8d54", "\u7db2\u9ede", "\u7f51\u70b9", "\u4e0a\u9580", "\u4e0a\u95e8",
    ],
    "Ask_Spec": [
        "feature", "spec", "material", "size", "capacity", "rating", "compatible",
        "support", "function", "how does", "can it", "does it", "is it",
        "\u529f\u80fd", "\u898f\u683c", "\u89c4\u683c", "\u6750\u8cea", "\u6750\u8d28", "\u5c3a\u5bf8", "\u5bb9\u91cf", "\u652f\u63f4",
        "\u652f\u6301", "\u9069\u5408", "\u9002\u5408", "\u80fd\u4e0d\u80fd", "\u53ef\u4e0d\u53ef\u4ee5", "\u600e\u9ebc\u7528", "\u600e\u4e48\u7528",
        "\u662f\u5426", "\u6709\u6c92\u6709", "\u6709\u6ca1\u6709",
    ],
    "Compare_Competitor": [
        "another brand", "other brand", "other shop", "other platform",
        "competitor", "compared with", "compare with", "versus", " vs ",
        "cheaper than", "better than", "\u5225\u5bb6", "\u522b\u5bb6", "\u53e6\u4e00\u5bb6", "\u5176\u4ed6\u54c1\u724c",
        "\u5176\u5b83\u54c1\u724c", "\u5176\u4ed6\u5e73\u53f0", "\u5225\u7684\u5e73\u53f0", "\u522b\u7684\u5e73\u53f0", "\u76f8\u6bd4", "\u6bd4\u8f03",
        "\u6bd4\u8f83", "\u5c0d\u6bd4", "\u540c\u985e", "\u540c\u7c7b",
    ],
    "Confirm_Order": [
        "i'll buy", "i will buy", "decided to buy", "place the order",
        "go ahead and order", "i'll take it", "\u6c7a\u5b9a\u8cb7", "\u51b3\u5b9a\u4e70", "\u6211\u8981\u8cb7",
        "\u6211\u8981\u4e70", "\u76f4\u63a5\u8cb7", "\u76f4\u63a5\u4e70", "\u53bb\u4e0b\u55ae", "\u53bb\u4e0b\u5355", "\u76f4\u63a5\u4e0b\u55ae",
        "\u76f4\u63a5\u4e0b\u5355", "\u5c31\u8cb7\u9019", "\u5c31\u4e70\u8fd9", "\u5c31\u8981\u9019", "\u5c31\u8981\u8fd9",
    ],
    "Doubt": [
        "really", "actually", "trust", "reliable", "suspicious", "not sure",
        "don't believe", "do not believe", "worried", "claim", "\u771f\u7684\u55ce",
        "\u771f\u7684\u5417", "\u771f\u7684\u5047\u7684", "\u9760\u8b5c", "\u9760\u8c31", "\u53ef\u4fe1", "\u4fe1\u4efb", "\u4e0d\u653e\u5fc3",
        "\u64d4\u5fc3", "\u62c5\u5fc3", "\u61f7\u7591", "\u6000\u7591", "\u6703\u4e0d\u6703", "\u4f1a\u4e0d\u4f1a",
    ],
    "Need_More_Evidence": [
        "review", "rating", "report", "proof", "evidence", "certificate",
        "certification", "test result", "case study", "screenshot",
        "\u8a55\u50f9", "\u8bc4\u4ef7", "\u6e2c\u8a55", "\u6d4b\u8bc4", "\u5831\u544a", "\u62a5\u544a", "\u8b49\u660e", "\u8bc1\u660e",
        "\u8a8d\u8b49", "\u8ba4\u8bc1", "\u6848\u4f8b", "\u622a\u5716", "\u622a\u56fe", "\u6578\u64da", "\u6570\u636e", "\u80cc\u66f8",
    ],
    "Need_Time_To_Think": [
        "think about", "need time", "consider", "later", "next time",
        "ask my", "talk to", "not buy yet", "hold off", "\u518d\u60f3", "\u8003\u616e",
        "\u8003\u8651", "\u4e4b\u5f8c\u518d", "\u4e4b\u540e\u518d", "\u4e0b\u6b21", "\u66ab\u6642", "\u6682\u65f6", "\u5148\u4e0d\u8cb7",
        "\u5148\u4e0d\u4e70", "\u4e0d\u6025", "\u5546\u91cf", "\u554f\u554f", "\u95ee\u95ee", "\u6bd4\u8f03\u4e00\u4e0b", "\u6bd4\u8f83\u4e00\u4e0b",
    ],
    "State_Need": [
        "looking for", "i need", "i want", "recommend", "suggestion",
        "suitable for", "help me choose", "\u60f3\u627e", "\u6211\u9700\u8981", "\u6211\u60f3\u8981",
        "\u60f3\u8cb7", "\u60f3\u4e70", "\u63a8\u85a6", "\u63a8\u8350", "\u9069\u5408\u6211\u7684", "\u9002\u5408\u6211\u7684", "\u5e6b\u6211\u6311",
        "\u5e2e\u6211\u6311", "\u9700\u6c42", "\u5e0c\u671b\u80fd",
    ],
    "Too_Expensive": [
        "too expensive", "overpriced", "not worth", "over budget", "pricey",
        "can't afford", "cannot afford", "\u592a\u8cb4", "\u592a\u8d35", "\u6709\u9ede\u8cb4", "\u6709\u70b9\u8d35",
        "\u8cb4\u4e86", "\u8d35\u4e86", "\u4e0d\u5212\u7b97", "\u4e0d\u503c\u5f97", "\u8d85\u51fa\u9810\u7b97", "\u8d85\u51fa\u9884\u7b97",
        "\u8ca0\u64d4\u4e0d\u8d77", "\u8d1f\u62c5\u4e0d\u8d77", "\u9810\u7b97\u4e0d\u5920", "\u9884\u7b97\u4e0d\u591f",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("train.jsonl"))
    parser.add_argument(
        "--model-dir", type=Path, default=Path("artifacts/xlmr_final_model")
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--c-values",
        default="0.6,1.2,2.4,4.0",
        help="OOF candidates; the best C is selected independently per label.",
    )
    return parser.parse_args()


def load_train(path: Path) -> tuple[list[dict], np.ndarray]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    label_to_id = {label: idx for idx, label in enumerate(LABELS)}
    targets = np.zeros((len(rows), len(LABELS)), dtype=np.int8)
    for row_idx, row in enumerate(rows):
        for label in row["labels"]:
            if label not in label_to_id:
                raise ValueError(f"Unknown label at row {row_idx}: {label}")
            targets[row_idx, label_to_id[label]] = 1
    return rows, targets


def make_strata(targets: np.ndarray, folds: int) -> list[str]:
    combinations = ["".join(str(int(x)) for x in row) for row in targets]
    counts = Counter(combinations)
    return [value if counts[value] >= folds else "rare" for value in combinations]


def new_vectorizers() -> tuple[TfidfVectorizer, TfidfVectorizer]:
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 5),
        min_df=2,
        max_df=0.995,
        max_features=240_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.995,
        max_features=120_000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    return char_vectorizer, word_vectorizer


def rule_features(
    texts: list[str], patterns: dict[str, list[str]] = RULE_PATTERNS
):
    values = np.zeros((len(texts), len(LABELS) + 4), dtype=np.float32)
    for row_idx, original in enumerate(texts):
        text = original.lower()
        for label_idx, label in enumerate(LABELS):
            count = sum(text.count(pattern) for pattern in patterns[label])
            values[row_idx, label_idx] = min(count, 4)
        values[row_idx, len(LABELS)] = min(text.count("?") + text.count("\uff1f"), 4)
        values[row_idx, len(LABELS) + 1] = min(text.count("not ") + text.count("\u4e0d"), 4)
        values[row_idx, len(LABELS) + 2] = min(len(original) / 200.0, 4)
        values[row_idx, len(LABELS) + 3] = float(any("\u4e00" <= ch <= "\u9fff" for ch in original))
    return csr_matrix(np.log1p(values) * 2.0)


def fit_transform_features(
    texts: list[str],
) -> tuple[TfidfVectorizer, TfidfVectorizer, object]:
    char_vectorizer, word_vectorizer = new_vectorizers()
    char_features = char_vectorizer.fit_transform(texts)
    word_features = word_vectorizer.fit_transform(texts)
    return (
        char_vectorizer,
        word_vectorizer,
        hstack([char_features, word_features, rule_features(texts)], format="csr"),
    )


def transform_features(
    texts: list[str],
    char_vectorizer: TfidfVectorizer,
    word_vectorizer: TfidfVectorizer,
):
    return hstack(
        [
            char_vectorizer.transform(texts),
            word_vectorizer.transform(texts),
            rule_features(texts),
        ],
        format="csr",
    )


def fit_classifiers(
    features, targets: np.ndarray, c_values: list[float]
) -> list[LinearSVC]:
    classifiers: list[LinearSVC] = []
    if len(c_values) != len(LABELS):
        raise ValueError("c_values must contain one value per label")
    for label_idx in range(len(LABELS)):
        classifier = LinearSVC(
            C=c_values[label_idx],
            class_weight="balanced",
            dual="auto",
            max_iter=10_000,
            random_state=42,
        )
        classifier.fit(features, targets[:, label_idx])
        classifiers.append(classifier)
    return classifiers


def decision_matrix(classifiers: list[LinearSVC], features) -> np.ndarray:
    return np.column_stack(
        [classifier.decision_function(features) for classifier in classifiers]
    ).astype(np.float32)


def select_c_per_label(
    targets: np.ndarray, decisions_by_c: dict[float, np.ndarray]
) -> tuple[list[float], np.ndarray]:
    selected_c: list[float] = []
    selected_decisions = np.zeros_like(next(iter(decisions_by_c.values())))
    for label_idx in range(len(LABELS)):
        best_f1 = -1.0
        best_c = None
        for c_value, decisions in decisions_by_c.items():
            probabilities = expit(decisions[:, label_idx])
            for threshold in np.linspace(0.05, 0.95, 91):
                score = f1_score(
                    targets[:, label_idx],
                    probabilities >= threshold,
                    zero_division=0,
                )
                if score > best_f1:
                    best_f1 = score
                    best_c = c_value
        selected_c.append(float(best_c))
        selected_decisions[:, label_idx] = decisions_by_c[best_c][:, label_idx]
    return selected_c, selected_decisions


def tune_single_source(
    targets: np.ndarray, probabilities: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)
    predictions = np.zeros_like(targets, dtype=np.int8)
    for label_idx in range(len(LABELS)):
        best_f1 = -1.0
        for threshold in np.linspace(0.05, 0.95, 181):
            candidate = probabilities[:, label_idx] >= threshold
            score = f1_score(targets[:, label_idx], candidate, zero_division=0)
            if score > best_f1:
                best_f1 = score
                thresholds[label_idx] = threshold
                predictions[:, label_idx] = candidate
    return thresholds, predictions


def tune_blend(
    targets: np.ndarray,
    transformer_probabilities: np.ndarray,
    linear_probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alphas = np.full(len(LABELS), 0.5, dtype=np.float32)
    thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)
    predictions = np.zeros_like(targets, dtype=np.int8)
    for label_idx in range(len(LABELS)):
        best_f1 = -1.0
        for alpha in np.linspace(0.0, 1.0, 21):
            blended = (
                alpha * transformer_probabilities[:, label_idx]
                + (1.0 - alpha) * linear_probabilities[:, label_idx]
            )
            for threshold in np.linspace(0.05, 0.95, 181):
                candidate = blended >= threshold
                score = f1_score(
                    targets[:, label_idx], candidate, zero_division=0
                )
                if score > best_f1:
                    best_f1 = score
                    alphas[label_idx] = alpha
                    thresholds[label_idx] = threshold
                    predictions[:, label_idx] = candidate
    return alphas, thresholds, predictions


def summarize(targets: np.ndarray, predictions: np.ndarray) -> dict:
    report = classification_report(
        targets,
        predictions,
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(targets, predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(targets, predictions, average="micro", zero_division=0)
        ),
        "per_label_f1": {
            label: float(report[label]["f1-score"]) for label in LABELS
        },
    }


def main() -> None:
    args = parse_args()
    oof_path = args.model_dir / "oof_transformer.npz"
    config_path = args.model_dir / "inference_config.json"
    if not oof_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(
            "Run train_transformer.py first; OOF predictions/model config are missing."
        )

    rows, targets = load_train(args.train_file)
    texts = [row["utterance"] for row in rows]
    transformer_data = np.load(oof_path)
    transformer_probabilities = transformer_data["probabilities"].astype(np.float32)
    saved_targets = transformer_data["targets"].astype(np.int8)
    if not np.array_equal(targets, saved_targets):
        raise ValueError("train.jsonl does not match the Transformer OOF artifact")

    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    splits = list(
        splitter.split(np.arange(len(rows)), make_strata(targets, args.folds))
    )
    c_candidates = [
        float(value.strip()) for value in args.c_values.split(",") if value.strip()
    ]
    if not c_candidates or any(value <= 0 for value in c_candidates):
        raise ValueError("--c-values must contain positive numbers")
    oof_by_c = {
        c_value: np.zeros_like(transformer_probabilities, dtype=np.float32)
        for c_value in c_candidates
    }
    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        train_texts = [texts[index] for index in train_indices]
        valid_texts = [texts[index] for index in valid_indices]
        char_vectorizer, word_vectorizer, train_features = fit_transform_features(
            train_texts
        )
        valid_features = transform_features(
            valid_texts, char_vectorizer, word_vectorizer
        )
        for c_value in c_candidates:
            classifiers = fit_classifiers(
                train_features,
                targets[train_indices],
                [c_value] * len(LABELS),
            )
            oof_by_c[c_value][valid_indices] = decision_matrix(
                classifiers, valid_features
            )
            del classifiers
        print(f"Finished linear OOF fold {fold}/{args.folds}")
        del char_vectorizer, word_vectorizer, train_features, valid_features
        gc.collect()

    selected_c, oof_decisions = select_c_per_label(targets, oof_by_c)
    linear_probabilities = expit(oof_decisions).astype(np.float32)
    _linear_thresholds, linear_predictions = tune_single_source(
        targets, linear_probabilities
    )
    alphas, thresholds, blended_predictions = tune_blend(
        targets, transformer_probabilities, linear_probabilities
    )
    linear_metrics = summarize(targets, linear_predictions)
    blended_metrics = summarize(targets, blended_predictions)

    char_vectorizer, word_vectorizer, full_features = fit_transform_features(
        texts
    )
    classifiers = fit_classifiers(full_features, targets, selected_c)
    artifact = {
        "labels": LABELS,
        "char_vectorizer": char_vectorizer,
        "word_vectorizer": word_vectorizer,
        "rule_patterns": RULE_PATTERNS,
        "rule_scale": 2.0,
        "selected_c": selected_c,
        "classifiers": classifiers,
        "sklearn_version": sklearn.__version__,
    }
    artifact_path = args.model_dir / "linear_model.joblib"
    joblib.dump(artifact, artifact_path, compress=3)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["labels"] != LABELS:
        raise ValueError("Transformer label order does not match")
    config.update(
        {
            "thresholds": thresholds.tolist(),
            "blend_alpha_transformer": alphas.tolist(),
            "linear_model_file": artifact_path.name,
            "linear_oof_validation": linear_metrics,
            "blended_oof_validation": blended_metrics,
        }
    )
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics = {
        "transformer": config["transformer_oof_validation"],
        "linear": linear_metrics,
        "blended": blended_metrics,
        "alpha_transformer": {
            label: round(float(value), 3)
            for label, value in zip(LABELS, alphas)
        },
        "thresholds": {
            label: round(float(value), 3)
            for label, value in zip(LABELS, thresholds)
        },
        "linear_c": {
            label: float(value) for label, value in zip(LABELS, selected_c)
        },
    }
    (args.model_dir / "training_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("\nFinal train-only OOF validation")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Saved linear model to", artifact_path)


if __name__ == "__main__":
    main()
