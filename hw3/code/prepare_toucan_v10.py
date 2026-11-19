#!/usr/bin/env python3
"""Convert a pinned subset of official Toucan SFT trajectories to HW3 messages.

The converter is original to this project.  It validates every function call,
serializes parallel calls into the one-call/one-result sequence used by the
homework, drops action-announcement preambles immediately before calls, caps
very long tool results, and splits only by the source UUID.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from public_guard_v10 import PublicOverlapGuard


PROMISE = re.compile(
    r"\b(let me|i(?:'ll| will)|allow me|one moment|please wait|i(?:'m| am) going to)\b.{0,120}"
    r"\b(check|look|search|find|get|fetch|verify|review|retrieve|call|use)\b",
    re.I | re.S,
)
ALLOWED_SUBSETS = {"single-turn-original", "single-turn-diversify", "multi-turn"}


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def parse_call(value: Any) -> tuple[str, dict[str, Any]]:
    parsed = parse_jsonish(value)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("name"), str):
        raise ValueError("invalid tool call")
    arguments = parse_jsonish(parsed.get("arguments", {}))
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be an object")
    return parsed["name"], arguments


def assistant_call(name: str, arguments: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": arguments},
    }]}


def tool_result(name: str, content: str, call_id: str, cap: int) -> dict[str, Any]:
    clean = content if len(content) <= cap else content[:cap] + " ...[truncated]"
    return {"role": "tool", "name": name, "tool_call_id": call_id, "content": clean}


def convert(raw: dict[str, Any], cap: int) -> dict[str, Any] | None:
    if raw.get("subset_name") not in ALLOWED_SUBSETS:
        return None
    tools = parse_jsonish(raw.get("tools"))
    source_messages = parse_jsonish(raw.get("messages"))
    if not isinstance(tools, list) or not isinstance(source_messages, list) or not 1 <= len(tools) <= 40:
        return None
    names = set()
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            return None
        if not isinstance(function.get("parameters", {}), dict):
            return None
        if function["name"] in names:
            return None
        names.add(function["name"])

    output: list[dict[str, Any]] = []
    pending: list[tuple[str, dict[str, Any], str]] = []
    calls = 0
    for index, message in enumerate(source_messages):
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            return None
        role = message["role"]
        content = message.get("content")
        if role in {"system", "user"}:
            if pending:
                return None
            if not isinstance(content, str) or not content.strip():
                return None
            output.append({"role": role, "content": content.strip()})
        elif role == "assistant":
            if pending or not isinstance(content, str) or not content.strip():
                return None
            output.append({"role": "assistant", "content": content.strip()})
        elif role == "tool_call":
            name, arguments = parse_call(content)
            if name not in names:
                return None
            # Toucan often emits a prose promise directly before its calls.
            # HW3 expects the call at that decision, so remove that preamble.
            if output and output[-1].get("role") == "assistant" and not output[-1].get("tool_calls"):
                prior = str(output[-1].get("content") or "")
                if PROMISE.search(prior) or index + 1 < len(source_messages):
                    output.pop()
            call_id = f"tc_{raw.get('uuid', 'row')}_{calls}"
            pending.append((name, arguments, call_id))
            calls += 1
        elif role == "tool_response":
            if not pending or not isinstance(content, str):
                return None
            name, arguments, call_id = pending.pop(0)
            output.append(assistant_call(name, arguments, call_id))
            output.append(tool_result(name, content, call_id, cap))
        else:
            return None
    if pending or calls < 2 or not output:
        return None
    if output[-1].get("role") != "assistant" or output[-1].get("tool_calls"):
        return None
    if output[0].get("role") not in {"system", "user"}:
        return None
    return {
        "source": f"toucan_v10_{raw['subset_name']}",
        "source_group": f"toucan-{raw['uuid']}",
        "truncate_history_to_last_user": True,
        "tools": tools,
        "messages": output,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("external_data/toucan_sft/SFT"))
    parser.add_argument("--out-dir", type=Path, default=Path("training_data_toucan_v10"))
    parser.add_argument("--train", type=int, default=6000)
    parser.add_argument("--validation", type=int, default=300)
    parser.add_argument("--result-cap", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args()
    guard = PublicOverlapGuard([
        Path("data/public_test.jsonl.gz"), Path("data/public_test_negotiation.jsonl.gz")
    ])

    candidates: list[tuple[str, dict[str, Any]]] = []
    stats = {"source_rows": 0, "hash_sampled": 0, "unconvertible": 0, "public_overlap": 0}
    for path in sorted(args.input.glob("*.parquet")):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=512):
            for raw in batch.to_pylist():
                stats["source_rows"] += 1
                uuid = str(raw.get("uuid", ""))
                key = hashlib.sha256(f"{args.seed}:{uuid}".encode()).hexdigest()
                # Deterministic pre-sampling bounds memory while leaving far
                # more candidates than the requested final subset.
                if int(key[:8], 16) % 1000 >= 180:
                    continue
                stats["hash_sampled"] += 1
                try:
                    row = convert(raw, args.result_cap)
                except (ValueError, TypeError, SyntaxError, json.JSONDecodeError):
                    row = None
                if row is None:
                    stats["unconvertible"] += 1
                    continue
                if guard.row_overlaps(row["messages"]):
                    stats["public_overlap"] += 1
                    continue
                candidates.append((key, row))
    candidates.sort(key=lambda item: item[0])
    needed = args.train + args.validation
    if len(candidates) < needed:
        raise SystemExit(f"Only {len(candidates)} valid Toucan candidates; need {needed}")
    chosen = candidates[:needed]
    validation = [row for _, row in chosen[: args.validation]]
    train = [row for _, row in chosen[args.validation:]]
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "validation.jsonl", validation)
    manifest = {
        "generator": "code/prepare_toucan_v10.py",
        "source": "Agent-Ark/Toucan-1.5M SFT",
        "revision": "0df3cf37f2abefb380370cfb02eabea2a35ae782",
        "license": "Apache-2.0",
        "seed": args.seed,
        "policy": "Official public SFT data; PUBLIC is a discard-only exact/near-overlap guard.",
        **guard.stats(),
        **stats,
        "train_conversations": len(train),
        "validation_conversations": len(validation),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
