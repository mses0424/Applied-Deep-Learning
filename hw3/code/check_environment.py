#!/usr/bin/env python3
"""Fail-fast checks for the HW3 training/evaluation environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from pathlib import Path


MINIMUM = {
    "torch": (2, 6),
    "transformers": (5, 13),
    "peft": (0, 19),
    "accelerate": (1, 2),
}

TA_EXACT = {
    "torch": "2.11.0",
    "transformers": "5.13.1",
    "peft": "0.19.1",
    "vllm": "0.25.1",
    "jinja2": "3.1.6",
}


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(piece) for piece in re.findall(r"\d+", value)[:3])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--allow-version-mismatch", action="store_true", help="development only; never use for the final course run")
    ap.add_argument("--require-bitsandbytes", action="store_true")
    ap.add_argument("--require-data", action="store_true", help="also check prepared training files")
    ap.add_argument("--require-ta-exact", action="store_true", help="require the exact Python/package versions listed in the slides")
    args = ap.parse_args()
    problems: list[str] = []
    versions: dict[str, str] = {}
    if sys.version_info < (3, 10):
        problems.append(f"Python >=3.10 is required, found {sys.version.split()[0]}")
    for package, minimum in MINIMUM.items():
        try:
            installed = importlib.metadata.version(package)
            versions[package] = installed
            if version_tuple(installed) < minimum:
                problems.append(f"{package}>={'.'.join(map(str, minimum))} is required, found {installed}")
        except importlib.metadata.PackageNotFoundError:
            problems.append(f"missing package: {package}")
    if args.require_bitsandbytes:
        try:
            versions["bitsandbytes"] = importlib.metadata.version("bitsandbytes")
        except importlib.metadata.PackageNotFoundError:
            problems.append("bitsandbytes is required for the single-GPU QLoRA path")
    if args.require_ta_exact:
        if sys.version_info[:3] != (3, 11, 15):
            problems.append(f"TA environment requires Python 3.11.15, found {sys.version.split()[0]}")
        for package, required in TA_EXACT.items():
            try:
                installed = importlib.metadata.version(package)
                versions[package] = installed
                if installed.split("+", 1)[0] != required:
                    problems.append(f"TA environment requires {package}=={required}, found {installed}")
            except importlib.metadata.PackageNotFoundError:
                problems.append(f"missing TA package: {package}=={required}")

    from train import patch_obsolete_torchvision
    patch_obsolete_torchvision()
    import torch
    from transformers import AutoConfig, AutoTokenizer

    if not torch.cuda.is_available():
        problems.append("CUDA GPU is not available")
    config = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B-Base")
    if config.model_type != "qwen3":
        problems.append(f"unexpected base model type: {config.model_type}")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    expected = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<tool_call>": 151657}
    actual = {token: tokenizer.convert_tokens_to_ids(token) for token in expected}
    for token, token_id in expected.items():
        if actual[token] != token_id:
            problems.append(f"unexpected token id for {token}: {actual[token]} != {token_id}")
    if args.require_data:
        required_data = [
            Path("training_data_fresh_v12/train.jsonl"),
            Path("training_data_fresh_v12/validation.jsonl"),
            Path("training_data_fresh_v12/manifest.json"),
        ]
        missing = [str(path) for path in required_data if not path.is_file()]
        if missing:
            problems.append("missing prepared v12 training data: " + ", ".join(missing))
    report = {
        "python": sys.version.split()[0],
        "packages": versions,
        "cuda": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "base_model_type": config.model_type,
        "chat_token_ids": actual,
        "problems": problems,
        "ta_exact_required": args.require_ta_exact,
    }
    print(json.dumps(report, indent=2))
    if problems and not args.allow_version_mismatch:
        raise SystemExit("Environment check failed. Install the assignment requirements before training.")


if __name__ == "__main__":
    main()
