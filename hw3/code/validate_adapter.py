#!/usr/bin/env python3
"""Fail fast on common HW3 adapter-submission mistakes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


BASE = "Qwen/Qwen3-0.6B-Base"
TOKENS = set(range(151644, 151669))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("adapter", type=Path)
    args = ap.parse_args()
    root = args.adapter
    problems = []
    config_path = root / "adapter_config.json"
    weights = root / "adapter_model.safetensors"
    if not config_path.is_file():
        problems.append("adapter_config.json is missing")
        config = {}
    else:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    if not weights.is_file():
        problems.append("adapter_model.safetensors is missing")
    if config.get("base_model_name_or_path") != BASE:
        problems.append(f"base_model_name_or_path must be {BASE!r}")
    rank = config.get("r")
    if not isinstance(rank, int) or not 1 <= rank <= 512:
        problems.append("LoRA rank r must be an integer in [1, 512]")
    indices = config.get("trainable_token_indices")
    if isinstance(indices, dict):
        flat = {x for values in indices.values() for x in values}
    else:
        flat = set(indices or [])
    missing_tokens = sorted(TOKENS - flat)
    if missing_tokens:
        problems.append(f"adapter does not carry all chat-token rows; missing {missing_tokens}")
    forbidden = [p.name for p in root.iterdir() if p.name in {"model.safetensors", "pytorch_model.bin"}]
    if forbidden:
        problems.append(f"full model weights must not be submitted: {forbidden}")
    if problems:
        print("INVALID ADAPTER")
        for problem in problems:
            print(" -", problem)
        raise SystemExit(1)
    print(json.dumps({"status": "ok", "adapter": str(root), "rank": rank,
                      "weights_bytes": weights.stat().st_size,
                      "trainable_chat_tokens": len(flat & TOKENS)}, indent=2))


if __name__ == "__main__":
    main()
