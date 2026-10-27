#!/usr/bin/env python3
"""Build editable report.md and a submission-ready text PDF."""

from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", ".matplotlib-cache")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="report_source.md")
    parser.add_argument("--generations", default="results/generation_analysis.json")
    parser.add_argument("--markdown-output", default="report.md")
    parser.add_argument("--pdf-output", default="report.pdf")
    return parser.parse_args()


def generation_appendix(path: str) -> str:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    blocks = []
    for sample_number, sample in enumerate(payload["samples"], 1):
        title = sample["product"].splitlines()[0]
        blocks.extend(
            [
                f"### Sample {sample_number}: {sample['qid']}",
                "",
                f"Retrieved product: {title}",
                f"Exact target: {sample['retrieval_is_gold']}",
                "",
            ]
        )
        for prompt_name, output in sample["generations"].items():
            blocks.extend([f"**{prompt_name}**", "", output, ""])
    return "\n".join(blocks)


def markdown_to_lines(text: str, width: int = 102) -> list[str]:
    lines: list[str] = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            lines.append("")
            continue
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            lines.extend([heading, ""])
            continue
        # Keep tables readable as plain text and strip simple Markdown markup.
        clean = re.sub(r"[*`]", "", stripped)
        indent = "  " if clean.startswith(("- ", "1. ", "2. ", "3. ")) else ""
        lines.extend(
            textwrap.wrap(
                clean,
                width=width,
                initial_indent=indent,
                subsequent_indent=indent + "  ",
                break_long_words=False,
                break_on_hyphens=False,
            ) or [""]
        )
    return lines


def write_pdf(text: str, path: str) -> None:
    lines = markdown_to_lines(text)
    lines_per_page = 52
    with PdfPages(path) as pdf:
        for start in range(0, len(lines), lines_per_page):
            page = lines[start : start + lines_per_page]
            figure = plt.figure(figsize=(8.27, 11.69))
            figure.patch.set_facecolor("white")
            figure.text(
                0.07,
                0.95,
                "\n".join(page),
                va="top",
                ha="left",
                family="DejaVu Sans",
                fontsize=9.2,
                linespacing=1.35,
            )
            figure.text(
                0.5,
                0.025,
                f"Page {start // lines_per_page + 1}",
                ha="center",
                fontsize=8,
                color="0.35",
            )
            plt.axis("off")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)


def main() -> None:
    args = parse_args()
    source = Path(args.source).read_text(encoding="utf-8")
    report = source.replace("{{GENERATIONS}}", generation_appendix(args.generations))
    Path(args.markdown_output).write_text(report, encoding="utf-8")
    write_pdf(report, args.pdf_output)
    print(f"Saved {args.markdown_output} and {args.pdf_output}")


if __name__ == "__main__":
    main()
