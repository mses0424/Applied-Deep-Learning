#!/usr/bin/env python3
"""Render the final Markdown report to a readable PDF using matplotlib."""

from __future__ import annotations

import argparse
import csv
import re
import textwrap
from pathlib import Path


def clean_inline(text: str) -> str:
    text = re.sub(r"!\[[^]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    # Matplotlib treats pairs of dollar signs as math markup. In the buyer
    # dialogue they denote literal currency, so escape them before rendering.
    return text.replace("**", "").replace("`", "").replace("$", r"\$")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("markdown", type=Path, default=Path("report/final_report.md"), nargs="?")
    ap.add_argument("--curve", type=Path, default=Path("results/learning_curve.csv"))
    ap.add_argument("--history", type=Path, default=Path("code/evidence/v12_validation_loss.csv"))
    ap.add_argument("--out", type=Path, default=Path("report.pdf"))
    args = ap.parse_args()
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "DejaVu Sans"]

    lines = args.markdown.read_text(encoding="utf-8").splitlines()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        fig = None
        y = 0.95
        page_no = 0

        def new_page() -> None:
            nonlocal fig, y, page_no
            if fig is not None:
                fig.text(0.5, 0.025, str(page_no), ha="center", fontsize=8, color="#666666")
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
            page_no += 1
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            y = 0.95

        def put(text: str, size: float = 10, weight: str = "normal", family: str = "sans-serif",
                indent: float = 0, color: str = "black", gap: float = 0.005) -> None:
            nonlocal y
            width = 94 if family != "monospace" else 88
            wrapped = textwrap.wrap(clean_inline(text), width=max(35, width - int(indent * 100)),
                                    replace_whitespace=False, drop_whitespace=True) or [""]
            line_height = 0.0185 * (size / 10)
            need = line_height * len(wrapped) + gap
            if y - need < 0.065:
                new_page()
            for part in wrapped:
                fig.text(0.08 + indent, y, part, fontsize=size, fontweight=weight, family=family,
                         va="top", color=color)
                y -= line_height
            y -= gap

        new_page()
        in_code = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("!["):
                if args.curve.is_file():
                    if y < 0.57:
                        new_page()
                    rows = list(csv.DictReader(args.curve.open(encoding="utf-8")))
                    xs = [int(r["step"]) for r in rows]
                    ys = [100 * float(r["weighted_step_accuracy"]) for r in rows]
                    top = fig.add_axes([0.16, y - 0.23, 0.70, 0.15])
                    top.plot(xs, ys, marker="o", linewidth=2, color="#17549b")
                    top.set_xlim(min(xs) - 250, max(xs) + 250)
                    top.set_xticks(xs)
                    top.tick_params(axis="x", labelrotation=55)
                    top.set_ylim(max(0, min(ys) - 2), min(100, max(ys) + 2))
                    for step, value in zip(xs, ys):
                        top.annotate(f"{value:.2f}%", (step, value), xytext=(0, -12 if step == 12000 else 6),
                                    textcoords="offset points", ha="center", fontsize=7)
                    top.set_title("v12 checkpoint accuracy", fontsize=8)
                    top.set_ylabel("Accuracy (%)", fontsize=7)
                    top.tick_params(labelsize=7)
                    top.grid(alpha=0.3)
                    if args.history.is_file():
                        history = list(csv.DictReader(args.history.open(encoding="utf-8")))
                        h_steps = [int(row["step"]) for row in history]
                        h_loss = [float(row["validation_loss"]) for row in history]
                        bottom = fig.add_axes([0.16, y - 0.47, 0.70, 0.15])
                        bottom.plot(h_steps, h_loss, marker="o", markersize=2,
                                    linewidth=1.3, color="#ba5a28")
                        bottom.set_xlim(0, max(h_steps) + 250)
                        bottom.set_xticks([0, 3000, 6000, 9000, 12000])
                        bottom.set_title("Recorded held-out validation loss (full v12 run)", fontsize=8)
                        bottom.set_xlabel("Optimizer step", fontsize=7)
                        bottom.set_ylabel("Validation loss", fontsize=7)
                        bottom.tick_params(labelsize=7)
                        bottom.grid(alpha=0.3)
                        y -= 0.53
                    else:
                        top.set_xlabel("Optimizer step", fontsize=7)
                        y -= 0.29
                continue
            if stripped.startswith("```"):
                in_code = not in_code
                y -= 0.006
                continue
            if not stripped:
                y -= 0.009
                continue
            if stripped.startswith("# "):
                put(stripped[2:], size=18, weight="bold", gap=0.018)
            elif stripped.startswith("## "):
                if y < 0.42:
                    new_page()
                put(stripped[3:], size=14, weight="bold", color="#17365d", gap=0.012)
            elif stripped.startswith("### "):
                put(stripped[4:], size=11.5, weight="bold")
            elif in_code:
                put(line, size=7.6, family="monospace", indent=0.02, color="#222222", gap=0.002)
            elif stripped.startswith("|"):
                put(stripped, size=7.2, family="monospace", gap=0.002)
            elif stripped.startswith(('- ', '* ')):
                put("• " + stripped[2:], indent=0.02)
            else:
                put(stripped)
        if fig is not None:
            fig.text(0.5, 0.025, str(page_no), ha="center", fontsize=8, color="#666666")
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        info = pdf.infodict()
        info["Title"] = "ADL 2026 HW3 — Tool-Calling Training"
        info["Subject"] = "Tool-calling and buyer-negotiation LoRA report"
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
