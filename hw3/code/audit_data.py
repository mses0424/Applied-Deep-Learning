#!/usr/bin/env python3
"""Validate generated conversations and print auditable dataset statistics."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def call_name(call: dict[str, Any]) -> str:
    fn = call.get("function", call)
    return fn.get("name", "")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("files", type=Path, nargs="+")
    args = ap.parse_args()
    stats = collections.Counter()
    sources = collections.Counter()
    tools_seen = set()
    for path in args.files:
        if "public_test" in str(path.resolve()).lower():
            raise SystemExit(f"Refusing to audit a test file as training data: {path}")
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                stats["conversations"] += 1
                sources[row.get("source", "unknown")] += 1
                defined = {t["function"]["name"] for t in row["tools"]}
                if len(defined) != len(row["tools"]):
                    raise ValueError(f"{path}:{line_no}: duplicate tool name")
                tools_seen.update(defined)
                assistant_positions = [i for i, msg in enumerate(row["messages"]) if msg.get("role") == "assistant"]
                supervised_positions = set(assistant_positions[-1:] if row.get("loss_on_last_assistant_only") else assistant_positions)
                previous = None
                for message_index, msg in enumerate(row["messages"]):
                    role = msg.get("role")
                    if role not in {"system", "user", "assistant", "tool"}:
                        raise ValueError(f"{path}:{line_no}: invalid role {role!r}")
                    if role == "assistant":
                        stats["assistant_examples"] += 1
                        if message_index in supervised_positions:
                            stats["supervised_assistant_examples"] += 1
                        calls = msg.get("tool_calls") or []
                        stats["assistant_with_call" if calls else "assistant_text_only"] += 1
                        if (msg.get("content") or "").strip() and calls:
                            stats["assistant_text_plus_call"] += 1
                        for call in calls:
                            name = call_name(call)
                            if name not in defined:
                                raise ValueError(f"{path}:{line_no}: call to undefined tool {name!r}")
                            raw = call.get("function", call).get("arguments", "{}")
                            args_obj = json.loads(raw) if isinstance(raw, str) else raw
                            if not isinstance(args_obj, dict):
                                raise ValueError(f"{path}:{line_no}: call arguments are not an object")
                    previous = role
    result = dict(stats)
    result["unique_tool_names"] = len(tools_seen)
    result["sources"] = dict(sources)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
