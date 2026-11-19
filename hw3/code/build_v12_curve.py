#!/usr/bin/env python3
"""Export v12 checkpoint scores with their recorded or user-reported sources."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


RUN = "fresh_v12_20261026_124322"
STEPS = (10800, 11400, 12000, 12284)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "evidence",
                    help="directory containing the four official v12 subset score JSON files")
    ap.add_argument("--out", type=Path, default=Path("results/v12_learning_curve.csv"))
    ap.add_argument("--reported", type=Path,
                    default=Path(__file__).resolve().parent / "evidence/v12_user_reported_accuracy.csv")
    args = ap.parse_args()
    rows = []
    for step in STEPS:
        path = args.results / f"{RUN}_checkpoint-{step}_subset.json"
        score = json.loads(path.read_text(encoding="utf-8"))
        measured = score["headline"]["with_first_step_rule"]["all"]
        if measured["n"] != 600:
            raise ValueError(f"Expected 600 fixed PUBLIC records in {path}")
        rows.append({
            "step": step,
            "n": measured["n"],
            "step_accuracy": measured["step_acc"],
            "weighted_step_accuracy": measured["weighted_step_acc"],
            "source": "official_score_json",
            "source_file": str(path),
        })
    with args.reported.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            step = int(row["step"])
            accuracy = float(row["accuracy_percent"]) / 100
            if not 0 <= accuracy <= 1 or step in {r["step"] for r in rows}:
                raise ValueError(f"Invalid or duplicate reported checkpoint: {row}")
            rows.append({
                "step": step, "n": "", "step_accuracy": "",
                "weighted_step_accuracy": accuracy, "source": "user_reported",
                "source_file": str(args.reported),
            })
    rows.sort(key=lambda row: row["step"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.out} with {len(rows)} checkpoints and explicit provenance")

if __name__ == "__main__":
    main()
