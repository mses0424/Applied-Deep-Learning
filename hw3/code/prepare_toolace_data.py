#!/usr/bin/env python3
"""Convert official ToolACE training dialogs to native Qwen/OpenAI tool turns.

Parallel ToolACE calls are deliberately sequentialized as call -> result ->
next call.  This matches HW3's teacher-forced per-step evaluation, where only
the first call is consumed before the next gold tool-result context is shown.
Homework PUBLIC data is used only by a discard-only contamination guard.
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import json
import re
from pathlib import Path
from typing import Any

from prepare_corrective_data import PUBLIC_FILES, ROOT, Decontaminator, sha256, split_rows, write_jsonl


TYPE_MAP = {"dict": "object", "list": "array", "str": "string", "int": "integer",
            "float": "number", "bool": "boolean"}


def normalize_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {key: normalize_schema(item) for key, item in value.items() if not (key == "required" and item is None)}
    if isinstance(out.get("type"), str):
        out["type"] = TYPE_MAP.get(out["type"], out["type"])
    return out


def parse_tools(system: str) -> list[dict[str, Any]]:
    marker = "Here is a list of functions in JSON format that you can invoke:"
    start = system.find(marker)
    if start < 0:
        raise ValueError("tool marker not found")
    start = system.find("[", start + len(marker))
    end_marker = "\nShould you decide"
    end = system.find(end_marker, start)
    if start < 0 or end < 0:
        raise ValueError("tool JSON bounds not found")
    text = system[start:end].strip()
    if text.endswith("."):
        text = text[:-1]
    raw_tools = json.loads(text)
    tools = []
    for raw in raw_tools:
        function = normalize_schema(raw)
        parameters = function.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        parameters["type"] = "object"
        parameters.setdefault("properties", {})
        function["parameters"] = parameters
        function.pop("required", None)
        tools.append({"type": "function", "function": function})
    return tools


def split_top_level(text: str) -> list[str]:
    pieces: list[str] = []
    start = 0
    depth = 0
    quote = ""
    escaped = False
    for index, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "," and depth == 0:
            pieces.append(text[start:index].strip())
            start = index + 1
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return pieces


def literal(text: str) -> Any:
    normalized = re.sub(r"\btrue\b", "True", text, flags=re.I)
    normalized = re.sub(r"\bfalse\b", "False", normalized, flags=re.I)
    normalized = re.sub(r"\bnull\b", "None", normalized, flags=re.I)
    try:
        return ast.literal_eval(normalized)
    except (ValueError, SyntaxError):
        # Bare identifiers occur rarely and are intended as string values.
        return text.strip()


def parse_calls(text: str) -> list[dict[str, Any]] | None:
    value = text.strip()
    if not (value.startswith("[") and value.endswith("]") and "(" in value):
        return None
    calls: list[dict[str, Any]] = []
    for expression in split_top_level(value[1:-1]):
        left = expression.find("(")
        if left <= 0 or not expression.endswith(")"):
            raise ValueError("invalid call expression")
        name = expression[:left].strip()
        arguments: dict[str, Any] = {}
        for argument in split_top_level(expression[left + 1:-1]):
            if not argument:
                continue
            if "=" not in argument:
                raise ValueError("positional argument")
            key, raw_value = argument.split("=", 1)
            key = key.strip()
            if not key or key in arguments:
                raise ValueError("invalid or duplicate argument")
            arguments[key] = literal(raw_value.strip())
        calls.append({
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))},
        })
    return calls or None


def parse_results(text: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(value, dict):
        value = [value]
    return value if isinstance(value, list) else []


def result_for(call: dict[str, Any], results: list[dict[str, Any]]) -> str:
    name = call["function"]["name"]
    for result in results:
        if isinstance(result, dict) and str(result.get("name", "")) == name:
            payload = result.get("results", result.get("content", result))
            return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return json.dumps({"status": "success", "completed_tool": name}, separators=(",", ":"))


def convert_conversation(raw: dict[str, Any], index: int) -> tuple[dict[str, Any], collections.Counter[str]]:
    tools = parse_tools(str(raw.get("system", "")))
    defined = {tool["function"]["name"] for tool in tools}
    original = raw.get("conversations") or []
    messages: list[dict[str, Any]] = []
    stats: collections.Counter[str] = collections.Counter()
    i = 0
    while i < len(original):
        item = original[i]
        role = item.get("from")
        value = str(item.get("value", ""))
        if role == "user":
            messages.append({"role": "user", "content": value})
        elif role == "assistant":
            calls = parse_calls(value)
            if not calls:
                messages.append({"role": "assistant", "content": value})
                stats["text_targets"] += 1
            else:
                if any(call["function"]["name"] not in defined for call in calls):
                    raise ValueError("call to undefined tool")
                next_is_tool = i + 1 < len(original) and original[i + 1].get("from") == "tool"
                results = parse_results(str(original[i + 1].get("value", ""))) if next_is_tool else []
                for call_index, call in enumerate(calls):
                    previous_role = messages[-1]["role"] if messages else ""
                    messages.append({"role": "assistant", "content": None, "tool_calls": [call]})
                    stats["must_call_targets" if previous_role == "tool" else "eligible_call_targets"] += 1
                    # Preserve real results when available.  For parallel calls
                    # without recorded results, synthesize only the intermediate
                    # success needed to supervise the next call.
                    if next_is_tool or call_index + 1 < len(calls):
                        messages.append({"role": "tool", "name": call["function"]["name"],
                                         "content": result_for(call, results)})
                stats["parallel_calls_sequentialized"] += max(0, len(calls) - 1)
                if next_is_tool:
                    i += 1
        elif role == "tool":
            # Normally consumed with its preceding assistant call.  A stray
            # source result cannot be attached safely and invalidates the row.
            raise ValueError("stray tool result")
        else:
            raise ValueError(f"unsupported role: {role!r}")
        i += 1
    if not messages or not any(message["role"] == "assistant" for message in messages):
        raise ValueError("no assistant targets")
    return ({
        "source": "team_ace_toolace_sequential_v1",
        "source_group": f"toolace-{index}",
        "truncate_history_to_last_user": True,
        "tools": tools,
        "messages": messages,
    }, stats)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "external_data/toolace/data.json")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training_data_toolace_v6")
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    if "public_test" in str(args.input.resolve()).lower():
        raise SystemExit("Refusing homework test data as a training source")
    raw_rows = json.loads(args.input.read_text(encoding="utf-8"))
    guard = Decontaminator(PUBLIC_FILES)
    rows: list[dict[str, Any]] = []
    stats: collections.Counter[str] = collections.Counter()
    for index, raw in enumerate(raw_rows):
        try:
            row, row_stats = convert_conversation(raw, index)
        except (ValueError, TypeError, json.JSONDecodeError):
            stats["discarded_parse_or_schema"] += 1
            continue
        if guard.row_overlaps(row["messages"]):
            stats["discarded_public_overlap"] += 1
            continue
        rows.append(row)
        stats.update(row_stats)
    train, validation = split_rows(rows, args.seed)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "validation.jsonl", validation)
    manifest = {
        "generator": "code/prepare_toolace_data.py",
        "seed": args.seed,
        "policy": "PUBLIC is a discard-only user-text guard; no PUBLIC content is written",
        "source": {
            "name": "Team-ACE/ToolACE",
            "url": "https://huggingface.co/datasets/Team-ACE/ToolACE",
            "revision": "6bda777c88d21e5a204703c1ee45597a8fa4f734",
            "license": "Apache-2.0",
            "local_file": str(args.input.relative_to(ROOT)),
            "sha256": sha256(args.input),
        },
        "raw_conversations": len(raw_rows),
        "converted_conversations": len(rows),
        "train_conversations": len(train),
        "validation_conversations": len(validation),
        "conversion": dict(sorted(stats.items())),
        "decontamination": {
            "public_user_texts_held_out": len(guard.exact),
            "texts_checked": guard.checked,
            "exact_overlaps_discarded": guard.exact_hits,
            "near_overlaps_discarded": guard.near_hits,
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
