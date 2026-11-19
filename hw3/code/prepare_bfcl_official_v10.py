#!/usr/bin/env python3
"""Convert pinned official BFCL v3 multi-turn tasks to HW3 step rows.

Calls come from BFCL's Apache-2.0 possible-answer files.  This converter does
not execute or copy BFCL code.  It creates deterministic, internally
consistent JSON tool results that expose values needed by the following call,
then emits one loss-masked row for every assistant decision.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from public_guard_v10 import PublicOverlapGuard


CLASS_FILES = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}
CATEGORIES = ("base", "miss_param", "miss_func", "long_context")


def safe_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [safe_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    output = {key: safe_schema(item) for key, item in value.items()}
    kind = output.get("type")
    if isinstance(kind, str):
        output["type"] = {
            "dict": "object", "list": "array", "tuple": "array", "str": "string",
            "int": "integer", "float": "number", "bool": "boolean",
        }.get(kind, kind)
    elif kind is not None:
        output.pop("type", None)
    return output


def openai_tool(document: dict[str, Any]) -> dict[str, Any]:
    parameters = safe_schema(document.get("parameters") or {})
    parameters["type"] = "object"
    parameters.setdefault("properties", {})
    return {"type": "function", "function": {
        "name": document["name"], "description": document.get("description", ""),
        "parameters": parameters,
    }}


def parse_call(expression: str, documents: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    node = ast.parse(expression, mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError("ground truth is not a call")
    if isinstance(node.func, ast.Name):
        name = node.func.id
    elif isinstance(node.func, ast.Attribute):
        name = node.func.attr
    else:
        raise ValueError("unsupported function expression")
    if name not in documents:
        raise ValueError(f"unknown BFCL function {name}")
    properties = list(((documents[name].get("parameters") or {}).get("properties") or {}).keys())
    if len(node.args) > len(properties):
        raise ValueError("too many positional arguments")
    arguments = {properties[index]: ast.literal_eval(value) for index, value in enumerate(node.args)}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ValueError("expanded keyword arguments are unsupported")
        arguments[keyword.arg] = ast.literal_eval(keyword.value)
    return name, arguments


def call_message(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [{
        "type": "function", "function": {"name": name, "arguments": arguments}
    }]}


def synthetic_result(name: str, arguments: dict[str, Any], next_arguments: dict[str, Any] | None,
                     counter: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "success", "function": name, "operation_id": f"bfcl-op-{counter:07d}",
        "received": arguments,
    }
    if next_arguments:
        result["values_for_next_step"] = next_arguments
    return result


def step_row(source: str, group: str, tools: list[dict[str, Any]],
             messages: list[dict[str, Any]]) -> dict[str, Any]:
    target = messages[-1]
    previous = messages[-2].get("role") if len(messages) > 1 else ""
    if target.get("tool_calls"):
        suffix = "must" if previous == "tool" else "first"
    else:
        suffix = "stop" if previous == "tool" else "text"
    return {
        "source": f"bfcl_official_v10_{source}_{suffix}",
        "source_group": group,
        "loss_on_last_assistant_only": True,
        "truncate_history_to_last_user": True,
        "tools": tools,
        "messages": list(messages),
    }


def load_lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("external_data/bfcl_v3_official"))
    parser.add_argument("--out-dir", type=Path, default=Path("training_data_bfcl_official_v10"))
    parser.add_argument("--validation-percent", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    guard = PublicOverlapGuard([
        Path("data/public_test.jsonl.gz"), Path("data/public_test_negotiation.jsonl.gz")
    ])

    docs_by_class: dict[str, list[dict[str, Any]]] = {}
    all_docs: dict[str, dict[str, Any]] = {}
    for class_name, filename in CLASS_FILES.items():
        docs = load_lines(args.input / "multi_turn_func_doc" / filename)
        docs_by_class[class_name] = docs
        for document in docs:
            all_docs[document["name"]] = document

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    stats = {"episodes": 0, "public_overlap_discarded": 0, "parse_errors": 0, "rows": 0}
    operation_counter = 0
    for category in CATEGORIES:
        questions = load_lines(args.input / f"BFCL_v3_multi_turn_{category}.json")
        answers = {row["id"]: row["ground_truth"] for row in load_lines(
            args.input / "possible_answer" / f"BFCL_v3_multi_turn_{category}.json"
        )}
        for question in questions:
            stats["episodes"] += 1
            user_messages = [message for turn in question["question"] for message in turn
                             if message.get("role") == "user"]
            if guard.row_overlaps(user_messages):
                stats["public_overlap_discarded"] += 1
                continue
            available_docs = [doc for cls in question["involved_classes"] for doc in docs_by_class[cls]]
            documents = {doc["name"]: doc for doc in available_docs}
            hidden_by_turn = {int(index): set(names) for index, names in question.get("missed_function", {}).items()}
            hidden = set().union(*hidden_by_turn.values()) if hidden_by_turn else set()
            messages: list[dict[str, Any]] = []
            episode_rows: list[dict[str, Any]] = []
            try:
                for turn_index, (user_messages, call_strings) in enumerate(zip(question["question"], answers[question["id"]])):
                    if turn_index in hidden_by_turn:
                        hidden -= hidden_by_turn[turn_index]
                    if user_messages:
                        messages.extend({"role": m["role"], "content": m["content"]} for m in user_messages)
                    else:
                        messages.append({"role": "user", "content":
                                         "The previously unavailable functions have now been added. Please continue."})
                    visible_tools = [openai_tool(doc) for doc in available_docs if doc["name"] not in hidden]
                    parsed = [parse_call(expression, documents) for expression in call_strings]
                    if not parsed:
                        reply = ("I cannot perform that action with the currently available functions."
                                 if category == "miss_func" else
                                 "I need the missing required detail before I can continue. Please provide it.")
                        messages.append({"role": "assistant", "content": reply})
                        episode_rows.append(step_row(category, question["id"], visible_tools, messages))
                        continue
                    for call_index, (name, arguments) in enumerate(parsed):
                        if name in hidden:
                            raise ValueError(f"hidden function {name} used before release")
                        messages.append(call_message(name, arguments))
                        episode_rows.append(step_row(category, question["id"], visible_tools, messages))
                        next_arguments = parsed[call_index + 1][1] if call_index + 1 < len(parsed) else None
                        operation_counter += 1
                        payload = synthetic_result(name, arguments, next_arguments, operation_counter)
                        messages.append({"role": "tool", "name": name,
                                         "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))})
                    called = ", ".join(name for name, _ in parsed)
                    messages.append({"role": "assistant", "content":
                                     f"Completed the requested operations successfully using {called}."})
                    episode_rows.append(step_row(category, question["id"], visible_tools, messages))
            except (ValueError, TypeError, SyntaxError, KeyError):
                stats["parse_errors"] += 1
                continue
            split_key = int(hashlib.sha256(f"{args.seed}:{question['id']}".encode()).hexdigest()[:8], 16) % 100
            destination = validation if split_key < args.validation_percent else train
            destination.extend(episode_rows)
            stats["rows"] += len(episode_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("train.jsonl", train), ("validation.jsonl", validation)):
        with (args.out_dir / filename).open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "generator": "code/prepare_bfcl_official_v10.py",
        "source": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "revision": "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
        "license": "Apache-2.0",
        "seed": args.seed,
        "policy": "Official public multi-turn questions/answers; PUBLIC is a discard-only exact/near-overlap guard; no BFCL code executed.",
        **guard.stats(),
        **stats,
        "train_rows": len(train), "validation_rows": len(validation),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
