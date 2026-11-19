#!/usr/bin/env python3
"""Generate original multi-step continuation curricula from existing synthetic rows."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def first_call(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    call = row["messages"][1]["tool_calls"][0]["function"]
    args = json.loads(call["arguments"]) if isinstance(call["arguments"], str) else call["arguments"]
    return call["name"], args


def call_message(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {
        "name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}


def tool_result(name: str, index: int, stage: int) -> dict[str, Any]:
    return {"role": "tool", "name": name, "content": json.dumps({
        "status": "success", "result_id": f"chain-{index}-{stage}", "continue": True})}


def dedupe_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list({tool["function"]["name"]: tool for tool in tools}.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("training_data_bfcl_synthetic_v3/train.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("training_data_multistep_v2"))
    ap.add_argument("--conversations", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    if "public_test" in str(args.source.resolve()).lower():
        raise SystemExit("Refusing test data as a curriculum source")
    rows = [json.loads(line) for line in args.source.open(encoding="utf-8") if line.strip()]
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        name, arguments = first_call(row)
        key = (row.get("domain", "unknown"), name, json.dumps(arguments, sort_keys=True))
        unique.setdefault(key, row)
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in unique.values():
        by_domain[row.get("domain", "unknown")].append(row)
    eligible = [domain for domain, domain_rows in by_domain.items() if len(domain_rows) >= 2]
    rng = random.Random(args.seed)
    out = []
    for i in range(args.conversations):
        domain = rng.choice(eligible)
        first, second = rng.sample(by_domain[domain], 2)
        name1, args1 = first_call(first)
        name2, args2 = first_call(second)
        request1 = first["messages"][0]["content"].rstrip(".?! ")
        request2 = second["messages"][0]["content"].rstrip(".?! ")
        user = f"First, {request1[0].lower() + request1[1:]}. After that succeeds, {request2[0].lower() + request2[1:]}."
        all_tools = dedupe_tools(first["tools"] + second["tools"])
        target_names = {name1, name2}
        targets = [t for t in all_tools if t["function"]["name"] in target_names]
        distractors = [t for t in all_tools if t["function"]["name"] not in target_names]
        tools = targets + rng.sample(distractors, min(8, len(distractors)))
        rng.shuffle(tools)
        messages = [
            {"role": "user", "content": user},
            call_message(name1, args1),
            tool_result(name1, i, 1),
            call_message(name2, args2),
            tool_result(name2, i, 2),
            {"role": "assistant", "content": "Done — both requested actions completed successfully."},
        ]
        out.append({"source": "original_multistep_curriculum_v1", "domain": domain, "tools": tools, "messages": messages})
    rng.shuffle(out)
    n_val = max(1, round(len(out) * 0.02))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("train.jsonl", out[n_val:]), ("validation.jsonl", out[:n_val])]:
        with (args.out_dir / name).open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {"source": str(args.source), "seed": args.seed, "conversations": len(out),
                "train_conversations": len(out) - n_val, "validation_conversations": n_val,
                "note": "Pairs only newly synthesized BFCL-doc requests; no benchmark question/answer data."}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
