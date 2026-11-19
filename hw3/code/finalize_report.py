#!/usr/bin/env python3
"""Fill report metadata and measured metrics into a final Markdown report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def pct(value: object) -> str:
    return "not computed" if not isinstance(value, (int, float)) else f"{100 * value:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--student-id", required=True)
    ap.add_argument("--name", default="", help="optional report name; student ID is required")
    ap.add_argument("--tools", type=Path, default=Path("results/tools.json"))
    ap.add_argument("--negotiation", type=Path, default=Path("results/negotiation.json"))
    ap.add_argument("--template", type=Path, default=Path(__file__).resolve().parent / "report_template.md")
    ap.add_argument("--out", type=Path, default=Path("report/final_report.md"))
    args = ap.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9]+", args.student_id.lower()):
        ap.error("student ID must be lower-case alphanumeric and begin with a letter")
    tools = json.loads(args.tools.read_text(encoding="utf-8"))
    negotiation = json.loads(args.negotiation.read_text(encoding="utf-8"))
    score = tools["headline"]["with_first_step_rule"]["all"]["weighted_step_acc"]
    metrics = negotiation["metrics"]
    values = {
        "{{STUDENT_ID}}": args.student_id.lower(),
        "{{STUDENT_NAME_LINE}}": f"Name: **{args.name}**" if args.name else "",
        "{{TOOL_SCORE}}": f"{100 * score:.2f}%",
        "{{TOOL_SCORE_PRECISE}}": f"{100 * score:.6f}%",
        "{{ORDER_ACCURACY}}": pct(metrics.get("order_accuracy")),
        "{{DEAL_VOLUME}}": pct(metrics.get("fill_rate")),
        "{{DEAL_PRICE}}": (f"{metrics['relative_price']:.5f}" if isinstance(metrics.get("relative_price"), (int, float)) else "n/a"),
        "{{RESPONSE_ACCURACY}}": pct(metrics.get("text_accuracy")),
    }
    text = args.template.read_text(encoding="utf-8")
    for key, value in values.items():
        text = text.replace(key, value)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(json.dumps(values, indent=2))


if __name__ == "__main__":
    main()
