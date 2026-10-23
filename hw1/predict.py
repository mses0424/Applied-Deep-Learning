#!/usr/bin/env python3
"""Run the trained Transformer/linear ensemble and write the required CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from scipy.special import expit
from scipy.sparse import csr_matrix, hstack
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

LABELS = [
    "Ask_Price", "Ask_Service", "Ask_Spec", "Compare_Competitor",
    "Confirm_Order", "Doubt", "Need_More_Evidence", "Need_Time_To_Think",
    "State_Need", "Too_Expensive",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def compact_utterance(text: str, max_chars: int = 420) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:140]}\n[... middle omitted ...]\n{text[-260:]}"


def format_views(utterance: str, context: list[dict]) -> list[str]:
    current = f"Current customer message:\n{compact_utterance(utterance)}"
    if not context:
        return [current]
    history = "\n".join(
        f"{message.get('role', '')}: {message.get('content', '')}"
        for message in context[-2:]
    )
    contextual = f"{current}\n\nImmediately preceding dialogue:\n{history[-1200:]}"
    return [current, contextual]


class ViewDataset(Dataset):
    def __init__(
        self, texts: list[str], source_indices: list[int], tokenizer: object,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            texts, max_length=max_length, truncation=True, padding=False
        )
        self.source_indices = source_indices

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, index: int) -> dict:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["source_index"] = self.source_indices[index]
        return item


def make_collator(tokenizer: object):
    def collate(examples: list[dict]) -> dict[str, torch.Tensor]:
        source_indices = torch.tensor(
            [example.pop("source_index") for example in examples], dtype=torch.long
        )
        batch = tokenizer.pad(examples, padding=True, return_tensors="pt")
        batch["source_index"] = source_indices
        return batch
    return collate


@torch.inference_mode()
def transformer_probabilities(
    rows: list[dict],
    contexts: dict[str, list[dict]],
    model_dir: Path,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, use_fast=True
    )
    texts: list[str] = []
    source_indices: list[int] = []
    for row_idx, row in enumerate(rows):
        for text in format_views(row["utterance"], contexts.get(row["id"], [])):
            texts.append(text)
            source_indices.append(row_idx)
    dataset = ViewDataset(
        texts, source_indices, tokenizer, max_length
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=make_collator(tokenizer),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    ).to(device)
    model.eval()
    sums = np.zeros((len(rows), len(LABELS)), dtype=np.float64)
    counts = np.zeros(len(rows), dtype=np.int32)
    for batch in loader:
        indices = batch.pop("source_index").numpy()
        batch = {
            key: value.to(device, non_blocking=True)
            for key, value in batch.items()
        }
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(**batch).logits
        probabilities = torch.sigmoid(logits).float().cpu().numpy()
        np.add.at(sums, indices, probabilities)
        np.add.at(counts, indices, 1)
    sums /= counts[:, None]
    result = sums.astype(np.float32)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def rule_features(
    texts: list[str], patterns: dict[str, list[str]], scale: float
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
        values[row_idx, len(LABELS) + 3] = float(
            any("\u4e00" <= ch <= "\u9fff" for ch in original)
        )
    return csr_matrix(np.log1p(values) * scale)


def linear_probabilities(utterances: list[str], artifact_path: Path) -> np.ndarray:
    artifact = joblib.load(artifact_path)
    if artifact["labels"] != LABELS:
        raise ValueError("Linear model label order does not match CSV schema")
    features = hstack(
        [
            artifact["char_vectorizer"].transform(utterances),
            artifact["word_vectorizer"].transform(utterances),
            rule_features(
                utterances,
                artifact["rule_patterns"],
                float(artifact.get("rule_scale", 2.0)),
            ),
        ],
        format="csr",
    )
    decisions = np.column_stack(
        [
            classifier.decision_function(features)
            for classifier in artifact["classifiers"]
        ]
    )
    return expit(decisions).astype(np.float32)


def context_linear_probabilities(
    rows: list[dict], contexts: dict[str, list[dict]], artifact_path: Path
) -> np.ndarray:
    artifact = joblib.load(artifact_path)
    if artifact["labels"] != LABELS:
        raise ValueError("Context-linear model label order does not match CSV schema")
    texts = []
    for row in rows:
        parts = [f"[CURRENT_CUSTOMER]\n{row['utterance']}"]
        recent_context = contexts.get(row["id"], [])[-2:]
        if recent_context:
            recent = "\n".join(
                f"[{message.get('role', '').upper()}]\n{message.get('content', '')}"
                for message in recent_context
            )
            parts.append("[RECENT_DIALOGUE]\n" + recent[-1200:])
        texts.append("\n".join(parts))
    features = hstack(
        [
            artifact["char_vectorizer"].transform(texts),
            artifact["word_vectorizer"].transform(texts),
            rule_features(
                texts, artifact["rule_patterns"],
                float(artifact.get("rule_scale", 2.0)),
            ),
        ],
        format="csr",
    )
    probabilities = np.zeros((len(rows), len(LABELS)), dtype=np.float32)
    for label, classifier in artifact["classifiers"].items():
        probabilities[:, LABELS.index(label)] = expit(
            classifier.decision_function(features)
        )
    return probabilities


def stack_features(
    transformer: np.ndarray, linear: np.ndarray, utterances: list[str],
    patterns: dict[str, list[str]], rule_scale: float,
) -> np.ndarray:
    eps = 1e-5
    transformer_clipped = np.clip(transformer, eps, 1.0 - eps)
    linear_clipped = np.clip(linear, eps, 1.0 - eps)
    transformer_logit = np.log(transformer_clipped / (1.0 - transformer_clipped))
    linear_logit = np.log(linear_clipped / (1.0 - linear_clipped))
    rules = rule_features(utterances, patterns, rule_scale).toarray()
    return np.column_stack(
        [
            transformer_clipped,
            linear_clipped,
            transformer_logit,
            linear_logit,
            transformer_clipped * linear_clipped,
            np.abs(transformer_clipped - linear_clipped),
            rules,
        ]
    ).astype(np.float32)


def main() -> None:
    args = parse_args()
    config_path = args.model_dir / "inference_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model metadata: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["labels"] != LABELS:
        raise ValueError("Model label order does not match CSV schema")

    contexts = json.loads(args.context.read_text(encoding="utf-8"))
    rows = json.loads(args.test.read_text(encoding="utf-8"))
    if not isinstance(contexts, dict) or not isinstance(rows, list):
        raise ValueError("Unexpected context.json or test.json structure")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("test.json contains duplicate ids")

    model_names = config.get("transformer_models")
    model_dirs = (
        [args.model_dir / name for name in model_names]
        if model_names
        else [args.model_dir]
    )
    transformer_parts = [
        transformer_probabilities(
            rows,
            contexts,
            model_dir,
            int(config["max_length"]),
            args.batch_size,
        )
        for model_dir in model_dirs
    ]
    transformer = np.mean(transformer_parts, axis=0, dtype=np.float32)
    linear_file = config.get("linear_model_file")
    linear_artifact = None
    if linear_file:
        linear_artifact = joblib.load(args.model_dir / linear_file)
        linear = linear_probabilities(
            [row["utterance"] for row in rows],
            args.model_dir / linear_file,
        )
        alpha = np.asarray(
            config["blend_alpha_transformer"], dtype=np.float32
        )
        probabilities = alpha * transformer + (1.0 - alpha) * linear
    else:
        probabilities = transformer
        linear = None
    thresholds = np.asarray(config["thresholds"], dtype=np.float32).copy()
    chinese = np.asarray([
        any("\u4e00" <= char <= "\u9fff" for char in row["utterance"])
        for row in rows
    ])

    for label, setting in config.get("language_blend_labels", {}).items():
        if linear is None:
            raise ValueError("Language-specific blend requires the plain linear model")
        label_idx = LABELS.index(label)
        alpha_zh = float(setting["alpha_transformer_zh"])
        alpha_en = float(setting["alpha_transformer_en"])
        language_probability = np.where(
            chinese,
            alpha_zh * transformer[:, label_idx]
            + (1.0 - alpha_zh) * linear[:, label_idx],
            alpha_en * transformer[:, label_idx]
            + (1.0 - alpha_en) * linear[:, label_idx],
        )
        language_prediction = np.where(
            chinese,
            language_probability >= float(setting["threshold_zh"]),
            language_probability >= float(setting["threshold_en"]),
        )
        probabilities[:, label_idx] = language_prediction.astype(np.float32)
        thresholds[label_idx] = 0.5

    context_linear_file = config.get("context_linear_model_file")
    if context_linear_file:
        if linear is None:
            raise ValueError("Context blend requires the plain linear model")
        context_linear = context_linear_probabilities(
            rows, contexts, args.model_dir / context_linear_file
        )
        for label, setting in config["context_linear_labels"].items():
            if not setting.get("enabled", False):
                continue
            label_idx = LABELS.index(label)
            probabilities[:, label_idx] = (
                float(setting["transformer_weight"]) * transformer[:, label_idx]
                + float(setting["plain_linear_weight"]) * linear[:, label_idx]
                + float(setting["context_linear_weight"])
                * context_linear[:, label_idx]
            )
            thresholds[label_idx] = np.nan
            # Language-dependent calibration is applied directly below.
            label_probability = probabilities[:, label_idx]
            language_prediction = np.where(
                chinese,
                label_probability >= float(setting["threshold_zh"]),
                label_probability >= float(setting["threshold_en"]),
            )
            probabilities[:, label_idx] = language_prediction.astype(np.float32)
            thresholds[label_idx] = 0.5

    stacker_file = config.get("stacker_model_file")
    if stacker_file:
        if linear is None or linear_artifact is None:
            raise ValueError("Stacker requires the plain linear model")
        stacker = joblib.load(args.model_dir / stacker_file)
        features = stack_features(
            transformer,
            linear,
            [row["utterance"] for row in rows],
            linear_artifact["rule_patterns"],
            float(linear_artifact.get("rule_scale", 2.0)),
        )
        for label, setting in config["stacker_labels"].items():
            if not setting.get("enabled", False):
                continue
            label_idx = LABELS.index(label)
            stacker_probability = stacker["classifiers"][label].predict_proba(
                features
            )[:, 1]
            if "threshold_zh" in setting and "threshold_en" in setting:
                probabilities[:, label_idx] = np.where(
                    chinese,
                    stacker_probability >= float(setting["threshold_zh"]),
                    stacker_probability >= float(setting["threshold_en"]),
                ).astype(np.float32)
                thresholds[label_idx] = 0.5
            else:
                probabilities[:, label_idx] = stacker_probability
                thresholds[label_idx] = float(setting["threshold"])
    predictions = (probabilities >= thresholds).astype(np.int8)
    if predictions.shape != (len(rows), len(LABELS)):
        raise RuntimeError(f"Unexpected prediction shape: {predictions.shape}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["id", *LABELS])
        for row_id, values in zip(ids, predictions.tolist()):
            writer.writerow([row_id, *values])


if __name__ == "__main__":
    main()
