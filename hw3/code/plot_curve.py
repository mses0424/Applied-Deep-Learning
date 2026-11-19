#!/usr/bin/env python3
"""Plot recorded and user-reported checkpoint scores with source labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", type=Path)
    ap.add_argument("--history", type=Path, default=Path("code/evidence/v12_validation_loss.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/learning_curve.svg"))
    args = ap.parse_args()
    import matplotlib.pyplot as plt

    with args.csv.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("Accuracy CSV has no measured rows")
    accuracy_steps = [int(row["step"]) for row in rows]
    accuracy = [100 * float(row["weighted_step_accuracy"]) for row in rows]
    with args.history.open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    if not history:
        raise SystemExit("Validation history CSV has no recorded rows")
    history_steps = [int(row["step"]) for row in history]
    validation_loss = [float(row["validation_loss"]) for row in history]

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.2, 6.2), constrained_layout=True)
    top.plot(accuracy_steps, accuracy, "o-", color="#17549b", linewidth=2)
    top.set_xlim(min(accuracy_steps) - 250, max(accuracy_steps) + 250)
    top.set_ylim(min(accuracy) - 2, max(accuracy) + 2)
    top.set_ylabel("Checkpoint accuracy (%)")
    top.set_title("v12 checkpoint accuracy")
    top.set_xticks(accuracy_steps)
    top.tick_params(axis="x", labelrotation=55, labelsize=8)
    top.grid(alpha=0.3)
    for step, value in zip(accuracy_steps, accuracy):
        top.annotate(f"{value:.2f}%", (step, value), xytext=(0, -15 if step == 12000 else 8),
                     textcoords="offset points", ha="center", fontsize=8)

    bottom.plot(history_steps, validation_loss, "o-", color="#ba5a28", markersize=3)
    bottom.set_xlim(0, max(history_steps) + 250)
    bottom.set_ylabel("Held-out validation loss")
    bottom.set_xlabel("Optimizer step")
    bottom.set_title("Recorded validation loss across the full v12 training run")
    bottom.grid(alpha=0.3)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="svg")
    plt.close(fig)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
