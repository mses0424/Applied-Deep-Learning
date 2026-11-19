#!/usr/bin/env python3
"""Export recorded v12 validation losses for the report's full-range plot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-state", type=Path, default=Path(
        "runs/fresh_v12_20261026_124322/train/trainer_state.json"))
    parser.add_argument("--out", type=Path, default=Path("code/evidence/v12_validation_loss.csv"))
    args = parser.parse_args()
    state = json.loads(args.trainer_state.read_text(encoding="utf-8"))
    rows = [
        {"step": int(event["step"]), "validation_loss": float(event["eval_loss"])}
        for event in state["log_history"] if "eval_loss" in event
    ]
    if not rows or any(a["step"] >= b["step"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Expected nonempty validation-loss history with increasing steps")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("step", "validation_loss"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} recorded validation losses to {args.out}")


if __name__ == "__main__":
    main()
