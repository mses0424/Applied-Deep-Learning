#!/usr/bin/env python3
"""Print final train-only OOF metrics without accessing test labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", type=Path, default=Path("artifacts/xlmr_final_model")
    )
    args = parser.parse_args()
    path = args.model_dir / "training_metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    final = metrics["final"]
    macro = float(final["macro_f1"])
    micro = float(final["micro_f1"])
    print("Train-only 5-fold OOF validation (not test-set score)")
    print(f"Macro-F1: {macro:.6f}  {'PASS' if macro >= 0.82 else 'BELOW'} challenge 0.82")
    print(f"Micro-F1: {micro:.6f}  {'PASS' if micro >= 0.83 else 'BELOW'} challenge 0.83")
    print("\nPer-label F1")
    for label, score in final["per_label_f1"].items():
        print(f"  {label:24s} {float(score):.6f}")


if __name__ == "__main__":
    main()
