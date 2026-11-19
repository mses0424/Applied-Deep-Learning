#!/usr/bin/env python3
"""Loss-masked LoRA training for ADL HW3.

Only assistant-message tokens receive loss.  The Qwen3 instruct tokenizer supplies
the exact grading chat template while weights always start from
Qwen/Qwen3-0.6B-Base.  The 25 added chat/tool token rows are carried inside the
adapter through PEFT ``trainable_token_indices``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import types
from enum import Enum
from pathlib import Path
from typing import Any


BASE_MODEL = "Qwen/Qwen3-0.6B-Base"
TOKENIZER_MODEL = "Qwen/Qwen3-0.6B"
CHAT_TOKEN_IDS = list(range(151644, 151669))


def patch_obsolete_torchvision() -> None:
    """Allow text-only Transformers imports with this machine's torchvision 0.1.6."""
    try:
        import torchvision
        import torchvision.transforms
        try:
            from torchvision.transforms import InterpolationMode  # noqa: F401
        except ImportError:
            class InterpolationMode(Enum):
                NEAREST = "nearest"
                NEAREST_EXACT = "nearest-exact"
                BILINEAR = "bilinear"
                BICUBIC = "bicubic"
                BOX = "box"
                HAMMING = "hamming"
                LANCZOS = "lanczos"
            torchvision.transforms.InterpolationMode = InterpolationMode
        try:
            from torchvision import io as _torchvision_io  # noqa: F401
        except ImportError:
            io_module = types.ModuleType("torchvision.io")
            torchvision.io = io_module
            sys.modules["torchvision.io"] = io_module
    except ImportError:
        pass


def refuse_test_data(paths: list[Path]) -> None:
    for path in paths:
        normalized = str(path.resolve()).lower()
        if "public_test" in normalized or "/data/public_" in normalized:
            raise SystemExit(f"Refusing evaluation/test data as training input: {path}")


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("messages"), list) or not isinstance(row.get("tools"), list):
                    raise ValueError(f"{path}:{line_no}: expected messages and tools lists")
                rows.append(row)
    return rows


def assistant_examples(rows: list[dict[str, Any]]) -> list[tuple[list[dict[str, Any]], list[dict[str, Any]], str]]:
    examples = []
    for row in rows:
        messages = row["messages"]
        for i, message in enumerate(messages):
            if message.get("role") == "assistant":
                examples.append((messages[:i], [message], row.get("source", "unknown")))
    return examples


class TokenizedExamples:
    @classmethod
    def from_rows(cls, rows: list[dict[str, Any]], tokenizer: Any, max_length: int, seed: int,
                  structured_loss: bool = False) -> "TokenizedExamples":
        import torch

        obj = cls.__new__(cls)
        obj.items = []
        obj.skipped = 0
        obj.skipped_by_source = {}
        obj.by_source = {}
        raw: list[tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], str, bool, bool]] = []
        for row in rows:
            assistant_indices = [i for i, message in enumerate(row["messages"]) if message.get("role") == "assistant"]
            if row.get("loss_on_last_assistant_only"):
                assistant_indices = assistant_indices[-1:]
            for i in assistant_indices:
                message = row["messages"][i]
                raw.append((row["messages"][:i], message, row["tools"], row.get("source", "unknown"),
                            bool(row.get("truncate_history_to_last_user")),
                            bool(row.get("supervise_tool_name_only"))))
        random.Random(seed).shuffle(raw)
        for history, target, tools, source, truncate_history, tool_name_only in raw:
            kw = {"tools": tools or None, "enable_thinking": False, "tokenize": False}
            candidates = [history]
            if truncate_history:
                # Preserve as much recent context as possible, but allow old,
                # completed user tasks to be removed.  Every candidate starts
                # at a user boundary, so assistant/tool adjacency remains valid.
                user_starts = [i for i, message in enumerate(history) if message.get("role") == "user"]
                candidates.extend(history[start:] for start in user_starts[1:])
            ids: list[int] = []
            target_ids: list[int] = []
            for candidate_history in candidates:
                prompt = tokenizer.apply_chat_template(candidate_history, add_generation_prompt=True, **kw)
                full = tokenizer.apply_chat_template(candidate_history + [target], add_generation_prompt=False, **kw)
                if not full.startswith(prompt):
                    raise RuntimeError("Tokenizer template invariant failed: assistant prompt is not a prefix of the completed turn")
                target_text = full[len(prompt):]
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]
                ids = prompt_ids + target_ids
                if target_ids and len(ids) <= max_length:
                    break
            if not target_ids or len(ids) > max_length:
                obj.skipped += 1
                obj.skipped_by_source[source] = obj.skipped_by_source.get(source, 0) + 1
                continue
            target_labels = list(target_ids)
            offsets = None
            if tool_name_only:
                calls = target.get("tool_calls") or []
                if not calls:
                    raise ValueError(f"{source}: supervise_tool_name_only requires a tool-call target")
                encoded = tokenizer(target_text, add_special_tokens=False, return_offsets_mapping=True)
                if encoded["input_ids"] != target_ids:
                    raise RuntimeError("Tokenizer offsets do not match target tokenization")
                spans: list[tuple[int, int]] = []
                cursor = 0
                for call in calls:
                    name = str((call.get("function") or {}).get("name", ""))
                    marker = '"name": ' + json.dumps(name, ensure_ascii=False)
                    marker_at = target_text.find(marker, cursor)
                    if marker_at < 0:
                        raise RuntimeError(f"Could not locate rendered tool name {name!r} in target")
                    quoted = json.dumps(name, ensure_ascii=False)
                    start = marker_at + len('"name": ') + 1
                    end = start + len(quoted) - 2
                    spans.append((start, end))
                    cursor = end
                target_labels = [-100] * len(target_ids)
                for token_index, (start, end) in enumerate(encoded["offset_mapping"]):
                    if any(start < span_end and end > span_start for span_start, span_end in spans):
                        target_labels[token_index] = target_ids[token_index]
                if all(label == -100 for label in target_labels):
                    raise RuntimeError(f"{source}: tool-name auxiliary target selected no tokens")
                offsets = encoded["offset_mapping"]
            item = {
                # Compact CPU storage matters because every DDP rank holds the
                # tokenized curriculum. Cast back to int64 in the collator,
                # immediately before a batch is transferred to the model.
                "input_ids": torch.tensor(ids, dtype=torch.int32),
                "attention_mask": torch.ones(len(ids), dtype=torch.uint8),
                "labels": torch.tensor([-100] * len(prompt_ids) + target_labels, dtype=torch.int32),
            }
            if structured_loss:
                if offsets is None:
                    encoded = tokenizer(target_text, add_special_tokens=False, return_offsets_mapping=True)
                    if encoded["input_ids"] != target_ids:
                        raise RuntimeError("Tokenizer offsets do not match target tokenization")
                    offsets = encoded["offset_mapping"]
                previous_role = history[-1].get("role") if history else ""
                is_call = bool(target.get("tool_calls"))
                base_weight = 3.0 if is_call and previous_role == "tool" else 1.5 if is_call else 1.2 if previous_role == "tool" else 1.0
                weights = [base_weight if label != -100 else 0.0 for label in target_labels]
                name_spans = []
                arguments_spans = []
                for block in re.finditer(r"<tool_call>(.*?)</tool_call>", target_text, re.S):
                    body = block.group(1)
                    name = re.search(r'"name"\s*:\s*"(?:\\.|[^"\\])*"', body)
                    if name:
                        name_spans.append((block.start(1) + name.start(), block.start(1) + name.end()))
                    arguments = re.search(r'"arguments"\s*:', body)
                    if arguments:
                        arguments_spans.append((block.start(1) + arguments.start(), block.end(1)))
                for token_index, (start, end) in enumerate(offsets):
                    if weights[token_index] == 0 or start == end:
                        continue
                    if any(start < span_end and end > span_start for span_start, span_end in name_spans):
                        weights[token_index] *= 2.0
                    elif any(start < span_end and end > span_start for span_start, span_end in arguments_spans):
                        weights[token_index] *= 1.5
                item["loss_weights"] = torch.tensor([0.0] * len(prompt_ids) + weights, dtype=torch.float16)
            obj.items.append(item)
            obj.by_source[source] = obj.by_source.get(source, 0) + 1
        return obj

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


class PadCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        width = max(len(x["input_ids"]) for x in features)
        batch: dict[str, list[Any]] = {"input_ids": [], "attention_mask": [], "labels": []}
        if "loss_weights" in features[0]:
            batch["loss_weights"] = []
        for x in features:
            n = width - len(x["input_ids"])
            # Right padding is correct for training and keeps the supervised suffix contiguous.
            batch["input_ids"].append(torch.cat([
                x["input_ids"], torch.full((n,), self.pad_token_id, dtype=x["input_ids"].dtype)
            ]))
            batch["attention_mask"].append(torch.cat([
                x["attention_mask"], torch.zeros(n, dtype=x["attention_mask"].dtype)
            ]))
            batch["labels"].append(torch.cat([
                x["labels"], torch.full((n,), -100, dtype=x["labels"].dtype)
            ]))
            if "loss_weights" in batch:
                batch["loss_weights"].append(torch.cat([
                    x["loss_weights"], torch.zeros(n, dtype=x["loss_weights"].dtype)
                ]))
        result = {
            "input_ids": torch.stack(batch["input_ids"]).long(),
            "attention_mask": torch.stack(batch["attention_mask"]).long(),
            "labels": torch.stack(batch["labels"]).long(),
        }
        if "loss_weights" in batch:
            result["loss_weights"] = torch.stack(batch["loss_weights"]).float()
        return result


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, nargs="+", required=True)
    ap.add_argument("--validation", type=Path, nargs="*", default=[])
    ap.add_argument("--output-dir", type=Path, default=Path("runs/hw3_lora"))
    ap.add_argument("--final-adapter", type=Path, default=Path("adapter_checkpoint"))
    ap.add_argument("--base-model", default=BASE_MODEL)
    ap.add_argument("--init-adapter", type=Path, help="Continue training from an existing LoRA adapter on a new curriculum.")
    ap.add_argument("--tokenizer", default=TOKENIZER_MODEL)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--learning-rate", type=float, default=1.5e-4)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--gradient-accumulation", type=int, default=16)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--save-steps", type=int, default=250)
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--save-only-model", action="store_true",
                    help="Checkpoints hold the adapter only (no optimizer state); about a third of the disk.")
    ap.add_argument("--load-best-model-at-end", action="store_true",
                    help="Restore the checkpoint with the lowest held-out validation loss before saving the adapter.")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--tokenize-only", action="store_true",
                    help="Audit rendered lengths and exit before loading model weights.")
    ap.add_argument("--limit-train-rows", type=int, default=0,
                    help="Use the first N training rows for a short hardware smoke test (0 = all).")
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Optional optimizer-step limit for a hardware smoke test (-1 = normal epochs).")
    ap.add_argument("--structured-loss", action="store_true",
                    help="Weight post-tool calls, tool names, and argument tokens in the loss.")
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--load-in-4bit", action="store_true", help="Use NF4 QLoRA (requires bitsandbytes).")
    ap.add_argument("--resume-from-checkpoint", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    refuse_test_data(args.train + args.validation)
    if not 1 <= args.rank <= 512:
        raise SystemExit("LoRA rank must be between 1 and the assignment limit of 512")
    if args.base_model != BASE_MODEL:
        raise SystemExit(f"The assignment requires --base-model {BASE_MODEL}")

    patch_obsolete_torchvision()
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.convert_tokens_to_ids("<|im_start|>") != 151644 or tokenizer.convert_tokens_to_ids("<|im_end|>") != 151645:
        raise RuntimeError("Unexpected Qwen3 chat-token ids; use the official Qwen/Qwen3-0.6B tokenizer")

    train_rows = read_rows(args.train)
    if args.limit_train_rows > 0:
        train_rows = train_rows[:args.limit_train_rows]
    val_rows = read_rows(args.validation) if args.validation else []
    train_set = TokenizedExamples.from_rows(train_rows, tokenizer, args.max_length, args.seed, args.structured_loss)
    val_set = TokenizedExamples.from_rows(val_rows, tokenizer, args.max_length, args.seed + 1, args.structured_loss) if val_rows else None
    if not train_set.items:
        raise SystemExit("No usable assistant examples after tokenization")
    print(json.dumps({"train_examples": len(train_set), "train_skipped_overlength": train_set.skipped,
                      "train_by_source": train_set.by_source,
                      "train_skipped_by_source": train_set.skipped_by_source,
                      "validation_examples": len(val_set) if val_set else 0,
                      "validation_skipped_overlength": val_set.skipped if val_set else 0,
                      "validation_skipped_by_source": val_set.skipped_by_source if val_set else {}}, indent=2), flush=True)
    if args.tokenize_only:
        return

    model_kwargs: dict[str, Any] = {"dtype": torch.float16, "low_cpu_mem_usage": True}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        model_kwargs["device_map"] = {"": int(os.environ.get("LOCAL_RANK", "0"))}
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
    else:
        config = LoraConfig(
            task_type="CAUSAL_LM",
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules="all-linear",
            trainable_token_indices=CHAT_TOKEN_IDS,
        )
        model = get_peft_model(model, config)
    model.print_trainable_parameters()

    world = max(1, int(os.environ.get("WORLD_SIZE", "1")))
    total_batch = args.batch_size * args.gradient_accumulation * world
    steps_per_epoch = math.ceil(len(train_set) / total_batch)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        # Transformers 5.x accepts either an absolute count (>=1) or a
        # fractional schedule ratio (<1) through warmup_steps.
        warmup_steps=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_only_model=args.save_only_model,
        eval_strategy="steps" if val_set else "no",
        eval_steps=args.save_steps if val_set else None,
        load_best_model_at_end=args.load_best_model_at_end and bool(val_set),
        metric_for_best_model="eval_loss" if args.load_best_model_at_end and val_set else None,
        greater_is_better=False if args.load_best_model_at_end and val_set else None,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        seed=args.seed,
        data_seed=args.seed,
        optim="adamw_torch",
        ddp_find_unused_parameters=False if world > 1 else None,
    )
    print(f"effective batch={total_batch}; estimated optimizer steps/epoch={steps_per_epoch}", flush=True)
    class StructuredTrainer(Trainer):
        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False,
                         num_items_in_batch: Any = None) -> Any:
            import torch.nn.functional as F
            from torch.utils.checkpoint import checkpoint
            weights = inputs.pop("loss_weights")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            # Shift once, as CausalLM's normal loss does.  Normalize by the
            # count of supervised tokens so example weights remain effective
            # with micro-batch size one.  FP16 softmax over the large Qwen
            # vocabulary can overflow, so do the CE in float32.  Checkpoint
            # chunks to avoid retaining another full-vocabulary FP32 tensor.
            def chunk_loss(logits: Any, gold: Any, scale: Any) -> Any:
                ce = F.cross_entropy(logits.float().transpose(1, 2), gold,
                                     ignore_index=-100, reduction="none")
                return (ce * scale).sum()

            shifted_logits = outputs.logits[:, :-1, :]
            shifted_labels = labels[:, 1:]
            shifted_weights = weights[:, 1:]
            loss_sum = shifted_logits.new_zeros((), dtype=torch.float32)
            for start in range(0, shifted_labels.shape[1], 128):
                stop = start + 128
                gold = shifted_labels[:, start:stop]
                if (gold != -100).any():
                    loss_sum = loss_sum + checkpoint(
                        chunk_loss, shifted_logits[:, start:stop, :], gold,
                        shifted_weights[:, start:stop], use_reentrant=False,
                    )
            valid = (labels[:, 1:] != -100).sum().clamp(min=1)
            loss = loss_sum / valid
            return (loss, outputs) if return_outputs else loss

    trainer_cls = StructuredTrainer if args.structured_loss else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )
    if args.structured_loss:
        trainer.model_accepts_loss_kwargs = False
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    summary = {
        "base_model": args.base_model,
        "tokenizer": args.tokenizer,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "load_best_model_at_end": args.load_best_model_at_end,
        "effective_batch_size": total_batch,
        "max_length": args.max_length,
        "train_examples": len(train_set),
        "skipped_overlength": train_set.skipped,
        "chat_token_ids": CHAT_TOKEN_IDS,
        "init_adapter": str(args.init_adapter) if args.init_adapter else None,
        "structured_loss": args.structured_loss,
    }
    # Multiple DDP ranks writing the same safetensors file can corrupt the
    # final adapter. Save once on rank 0, then synchronize before shutdown.
    if trainer.is_world_process_zero():
        args.final_adapter.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.final_adapter, safe_serialization=True)
        tokenizer.save_pretrained(args.final_adapter)
        (args.final_adapter / "training_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Saved final adapter to {args.final_adapter}")
    if torch.distributed.is_initialized():
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
