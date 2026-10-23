#!/usr/bin/env python3
"""Leakage-free cross-validation and full-data Transformer training for ADL HW1."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

LABELS = [
    "Ask_Price", "Ask_Service", "Ask_Spec", "Compare_Competitor",
    "Confirm_Order", "Doubt", "Need_More_Evidence", "Need_Time_To_Think",
    "State_Need", "Too_Expensive",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", type=Path, default=Path("train.jsonl"))
    parser.add_argument("--model-name", default="FacebookAI/xlm-roberta-base")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/xlmr_final_model")
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2, help="Per GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1.5e-5)
    parser.add_argument("--head-learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.08)
    parser.add_argument("--pos-weight-power", type=float, default=0.5)
    parser.add_argument("--label-smoothing", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--final-seeds",
        default="42,3407",
        help="Comma-separated CV/final ensemble seeds; two seeds fit under 4 GB.",
    )
    parser.add_argument("--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse completed fold/seed OOF checkpoints and final models.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_train(path: Path) -> tuple[list[dict], np.ndarray]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    label_to_id = {label: idx for idx, label in enumerate(LABELS)}
    targets = np.zeros((len(rows), len(LABELS)), dtype=np.float32)
    for row_idx, row in enumerate(rows):
        unknown = set(row["labels"]) - set(LABELS)
        if unknown:
            raise ValueError(f"Unknown labels in row {row_idx}: {sorted(unknown)}")
        for label in row["labels"]:
            targets[row_idx, label_to_id[label]] = 1.0
    return rows, targets


def compact_utterance(text: str, max_chars: int = 420) -> str:
    """Keep both the setup and intent-bearing tail of long customer messages."""
    if len(text) <= max_chars:
        return text
    return f"{text[:140]}\n[... middle omitted ...]\n{text[-260:]}"


def format_views(row: dict) -> list[str]:
    utterance = compact_utterance(row["utterance"])
    current = f"Current customer message:\n{utterance}"
    context = row.get("context", [])
    if not context:
        return [current]
    # Full history repeats old intents and harms precision. The last exchange is
    # enough to resolve short replies while keeping the current message dominant.
    recent_context = context[-2:]
    history = "\n".join(
        f"{message.get('role', '')}: {message.get('content', '')}"
        for message in recent_context
    )
    contextual = f"{current}\n\nImmediately preceding dialogue:\n{history[-1200:]}"
    return [current, contextual]


def make_strata(targets: np.ndarray, folds: int) -> list[str]:
    combinations = ["".join(str(int(x)) for x in row) for row in targets]
    counts = Counter(combinations)
    return [value if counts[value] >= folds else "rare" for value in combinations]


class ViewDataset(Dataset):
    def __init__(
        self,
        rows: list[dict],
        targets: np.ndarray,
        indices: np.ndarray,
        tokenizer: object,
        max_length: int,
    ) -> None:
        texts: list[str] = []
        labels: list[np.ndarray] = []
        source_indices: list[int] = []
        for source_idx in indices.tolist():
            for text in format_views(rows[source_idx]):
                texts.append(text)
                labels.append(targets[source_idx])
                source_indices.append(source_idx)
        self.encodings = tokenizer(
            texts, max_length=max_length, truncation=True, padding=False
        )
        self.labels = torch.tensor(np.asarray(labels), dtype=torch.float32)
        self.source_indices = torch.tensor(source_indices, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        item["source_index"] = self.source_indices[index]
        return item


def make_collator(tokenizer: object):
    def collate(examples: list[dict]) -> dict[str, torch.Tensor]:
        labels = torch.stack([example.pop("labels") for example in examples])
        source_indices = torch.stack([example.pop("source_index") for example in examples])
        batch = tokenizer.pad(examples, padding=True, return_tensors="pt")
        batch["labels"] = labels
        batch["source_index"] = source_indices
        return batch
    return collate


def create_model(model_name: str, gradient_checkpointing: bool):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification",
        id2label={idx: label for idx, label in enumerate(LABELS)},
        label2id={label: idx for idx, label in enumerate(LABELS)},
    )
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model


def train_model(
    accelerator: Accelerator,
    args: argparse.Namespace,
    tokenizer: object,
    rows: list[dict],
    targets: np.ndarray,
    train_indices: np.ndarray,
    valid_indices: np.ndarray | None,
    description: str,
):
    train_dataset = ViewDataset(
        rows, targets, train_indices, tokenizer, args.max_length
    )
    collator = make_collator(tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
    )
    valid_loader = None
    if valid_indices is not None:
        valid_dataset = ViewDataset(
            rows, targets, valid_indices, tokenizer, args.max_length
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=max(args.batch_size * 2, 8),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )

    with accelerator.main_process_first():
        model = create_model(args.model_name, args.gradient_checkpointing)
    parameter_groups: dict[tuple[bool, bool], dict] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_head = (
            name.startswith("classifier.")
            or ".classifier." in name
            or name.startswith("pooler.")
            or ".pooler." in name
        )
        no_decay = name.endswith(".bias") or "LayerNorm.weight" in name or "layer_norm.weight" in name
        key = (is_head, no_decay)
        if key not in parameter_groups:
            parameter_groups[key] = {
                "params": [],
                "lr": args.head_learning_rate if is_head else args.learning_rate,
                "weight_decay": 0.0 if no_decay else args.weight_decay,
            }
        parameter_groups[key]["params"].append(parameter)
    optimizer = torch.optim.AdamW(list(parameter_groups.values()))
    if valid_loader is None:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )
    else:
        model, optimizer, train_loader, valid_loader = accelerator.prepare(
            model, optimizer, train_loader, valid_loader
        )

    updates_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation
    )
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    positive = targets[train_indices].sum(axis=0)
    negative = len(train_indices) - positive
    pos_weight = torch.tensor(
        (negative / positive) ** args.pos_weight_power,
        dtype=torch.float32,
        device=accelerator.device,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model.train()
    progress = tqdm(
        total=len(train_loader) * args.epochs,
        desc=description,
        disable=not accelerator.is_local_main_process,
    )
    running_loss = 0.0
    seen_steps = 0
    for _epoch in range(args.epochs):
        for batch in train_loader:
            labels = batch.pop("labels")
            batch.pop("source_index")
            if args.label_smoothing:
                labels = labels * (1.0 - args.label_smoothing) + 0.5 * args.label_smoothing
            with accelerator.accumulate(model):
                logits = model(**batch).logits
                loss = criterion(logits, labels)
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach())
            seen_steps += 1
            progress.update(1)
            if seen_steps % 25 == 0:
                progress.set_postfix(loss=f"{running_loss / seen_steps:.4f}")
    progress.close()
    return model, optimizer, train_loader, valid_loader, scheduler


@torch.inference_mode()
def collect_validation(
    accelerator: Accelerator,
    model: torch.nn.Module,
    valid_loader: DataLoader,
    total_rows: int,
) -> np.ndarray | None:
    model.eval()
    gathered_probabilities: list[np.ndarray] = []
    gathered_indices: list[np.ndarray] = []
    for batch in valid_loader:
        batch.pop("labels")
        source_indices = batch.pop("source_index")
        logits = model(**batch).logits
        probabilities = torch.sigmoid(logits)
        probabilities, source_indices = accelerator.gather_for_metrics(
            (probabilities, source_indices)
        )
        if accelerator.is_main_process:
            gathered_probabilities.append(probabilities.float().cpu().numpy())
            gathered_indices.append(source_indices.cpu().numpy())
    if not accelerator.is_main_process:
        return None
    probabilities = np.concatenate(gathered_probabilities)
    indices = np.concatenate(gathered_indices)
    sums = np.zeros((total_rows, len(LABELS)), dtype=np.float64)
    counts = np.zeros(total_rows, dtype=np.int32)
    np.add.at(sums, indices, probabilities)
    np.add.at(counts, indices, 1)
    seen = counts > 0
    sums[seen] /= counts[seen, None]
    return sums.astype(np.float32)


def tune_thresholds(targets: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    thresholds = np.full(len(LABELS), 0.5, dtype=np.float32)
    for label_idx in range(len(LABELS)):
        best_f1 = -1.0
        for threshold in np.linspace(0.05, 0.90, 171):
            score = f1_score(
                targets[:, label_idx],
                probabilities[:, label_idx] >= threshold,
                zero_division=0,
            )
            if score > best_f1:
                best_f1 = score
                thresholds[label_idx] = threshold
    return thresholds


def clear_device_memory(accelerator: Accelerator) -> None:
    accelerator.free_memory()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()



def cv_checkpoint_path(output_dir: Path, fold: int, seed: int) -> Path:
    return output_dir / "checkpoints" / f"cv_fold_{fold}_seed_{seed}.npz"


def save_cv_checkpoint(
    path: Path,
    valid_indices: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Atomically save one completed fold/seed prediction checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary,
        valid_indices=valid_indices.astype(np.int64),
        probabilities=probabilities[valid_indices].astype(np.float32),
    )
    temporary.replace(path)


def load_cv_checkpoint(
    path: Path,
    expected_indices: np.ndarray,
) -> np.ndarray:
    data = np.load(path)
    saved_indices = data["valid_indices"].astype(np.int64)
    probabilities = data["probabilities"].astype(np.float32)
    if not np.array_equal(saved_indices, expected_indices):
        raise ValueError(f"Checkpoint split mismatch: {path}")
    if probabilities.shape != (len(expected_indices), len(LABELS)):
        raise ValueError(f"Checkpoint probability shape mismatch: {path}")
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Checkpoint contains non-finite values: {path}")
    return probabilities


def final_model_is_complete(model_dir: Path) -> bool:
    weights_exist = (
        (model_dir / "model.safetensors").is_file()
        or (model_dir / "pytorch_model.bin").is_file()
    )
    return (
        weights_exist
        and (model_dir / "config.json").is_file()
        and (model_dir / "tokenizer_config.json").is_file()
    )


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation,
        mixed_precision=args.mixed_precision,
    )
    set_seed(args.seed + accelerator.process_index)
    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

    rows, targets = load_train(args.train_file)
    strata = make_strata(targets, args.folds)
    splitter = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    splits = list(splitter.split(np.arange(len(rows)), strata))
    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    ensemble_seeds = [
        int(value.strip()) for value in args.final_seeds.split(",") if value.strip()
    ]
    if not ensemble_seeds:
        raise ValueError("--final-seeds must contain at least one integer")

    run_signature = {
        "checkpoint_version": 1,
        "model_name": args.model_name,
        "train_sha256": hashlib.sha256(args.train_file.read_bytes()).hexdigest(),
        "rows": len(rows),
        "folds": args.folds,
        "epochs": args.epochs,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "gradient_accumulation": args.gradient_accumulation,
        "learning_rate": args.learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "pos_weight_power": args.pos_weight_power,
        "label_smoothing": args.label_smoothing,
        "seed": args.seed,
        "ensemble_seeds": ensemble_seeds,
    }
    signature_path = args.output_dir / "checkpoints" / "run_signature.json"
    if accelerator.is_main_process:
        signature_path.parent.mkdir(parents=True, exist_ok=True)
        existing_cv = list(signature_path.parent.glob("cv_fold_*_seed_*.npz"))
        if args.resume and existing_cv and not signature_path.is_file():
            raise ValueError(
                "Found CV checkpoints without run_signature.json; refusing unsafe resume."
            )
        if args.resume and signature_path.is_file():
            saved_signature = json.loads(signature_path.read_text(encoding="utf-8"))
            if saved_signature != run_signature:
                raise ValueError(
                    "Checkpoint settings differ from this run. Use a new output "
                    "directory or remove old checkpoints instead of mixing runs."
                )
        signature_path.write_text(
            json.dumps(run_signature, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    accelerator.wait_for_everyone()

    oof = np.zeros_like(targets, dtype=np.float32) if accelerator.is_main_process else None
    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        fold_sum = (
            np.zeros_like(targets, dtype=np.float32)
            if accelerator.is_main_process
            else None
        )
        for cv_seed in ensemble_seeds:
            checkpoint_path = cv_checkpoint_path(args.output_dir, fold, cv_seed)
            if args.resume and checkpoint_path.is_file():
                if accelerator.is_main_process:
                    saved_probabilities = load_cv_checkpoint(
                        checkpoint_path, valid_indices
                    )
                    fold_sum[valid_indices] += saved_probabilities
                    print(
                        f"Resumed CV fold {fold}/{args.folds}, seed {cv_seed} "
                        f"from {checkpoint_path}"
                    )
                accelerator.wait_for_everyone()
                continue

            set_seed(cv_seed + fold * 1009 + accelerator.process_index)
            model, optimizer, train_loader, valid_loader, scheduler = train_model(
                accelerator,
                args,
                tokenizer,
                rows,
                targets,
                train_indices,
                valid_indices,
                description=f"CV fold {fold}/{args.folds}, seed {cv_seed}",
            )
            fold_probabilities = collect_validation(
                accelerator, model, valid_loader, len(rows)
            )
            if accelerator.is_main_process:
                fold_sum[valid_indices] += fold_probabilities[valid_indices]
                save_cv_checkpoint(
                    checkpoint_path, valid_indices, fold_probabilities
                )
                print("Saved resume checkpoint:", checkpoint_path)
            accelerator.wait_for_everyone()
            del model, optimizer, train_loader, valid_loader, scheduler
            clear_device_memory(accelerator)
        if accelerator.is_main_process:
            oof[valid_indices] = fold_sum[valid_indices] / len(ensemble_seeds)

    transformer_thresholds = None
    transformer_metrics = None
    if accelerator.is_main_process:
        transformer_thresholds = tune_thresholds(targets, oof)
        predictions = oof >= transformer_thresholds
        macro = f1_score(targets, predictions, average="macro", zero_division=0)
        micro = f1_score(targets, predictions, average="micro", zero_division=0)
        report = classification_report(
            targets, predictions, target_names=LABELS, digits=4, output_dict=True,
            zero_division=0,
        )
        transformer_metrics = {
            "macro_f1": float(macro),
            "micro_f1": float(micro),
            "per_label_f1": {label: float(report[label]["f1-score"]) for label in LABELS},
        }
        np.savez_compressed(
            args.output_dir / "oof_transformer.npz",
            probabilities=oof,
            targets=targets,
        )
    accelerator.wait_for_everyone()

    all_indices = np.arange(len(rows))
    final_model_names: list[str] = []
    for final_seed in ensemble_seeds:
        model_name = f"model_seed_{final_seed}"
        model_dir = args.output_dir / model_name
        final_model_names.append(model_name)
        if args.resume and final_model_is_complete(model_dir):
            if accelerator.is_main_process:
                print(f"Resumed: final model already complete at {model_dir}")
            accelerator.wait_for_everyone()
            continue

        set_seed(final_seed + accelerator.process_index)
        model, optimizer, train_loader, _valid_loader, scheduler = train_model(
            accelerator,
            args,
            tokenizer,
            rows,
            targets,
            all_indices,
            None,
            description=f"Full-data model seed {final_seed}",
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            unwrapped.save_pretrained(model_dir, safe_serialization=True)
            tokenizer.save_pretrained(model_dir)
        accelerator.wait_for_everyone()
        del model, optimizer, train_loader, _valid_loader, scheduler
        clear_device_memory(accelerator)

    if accelerator.is_main_process:
        inference_config = {
            "labels": LABELS,
            "thresholds": transformer_thresholds.tolist(),
            "max_length": args.max_length,
            "base_model": args.model_name,
            "views": "head_tail_utterance_and_recent_context",
            "folds": args.folds,
            "epochs": args.epochs,
            "seed": args.seed,
            "final_seeds": ensemble_seeds,
            "transformer_models": final_model_names,
            "transformer_oof_validation": transformer_metrics,
        }
        (args.output_dir / "inference_config.json").write_text(
            json.dumps(inference_config, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print("\nTransformer OOF validation (train.jsonl only)")
        print(json.dumps(transformer_metrics, ensure_ascii=False, indent=2))
        print("Saved final full-data models:", ", ".join(final_model_names))
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
