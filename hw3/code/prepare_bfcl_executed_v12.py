#!/usr/bin/env python3
"""Replay the public BFCL multi-turn answers against its official simulators.

The conversation and possible-answer files are public training sources.  The
homework PUBLIC set is used only to reject overlapping source questions.  This
module dispatches parsed, literal arguments directly to known BFCL classes;
it never evaluates a model-produced Python expression.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

from prepare_bfcl_official_v10 import CLASS_FILES, CATEGORIES, load_lines, openai_tool, parse_call
from public_guard_v10 import PublicOverlapGuard


BACKEND_PACKAGE = "bfcl_eval.eval_checker.multi_turn_eval.func_source_code"


def initialize(question: dict[str, Any], category: str) -> dict[str, Any]:
    instances = {}
    for class_name in question["involved_classes"]:
        module = importlib.import_module(f"{BACKEND_PACKAGE}.{CLASS_FILES[class_name].removesuffix('.json')}")
        instance = getattr(module, class_name)()
        if class_name != "MathAPI":
            instance._load_scenario(
                copy.deepcopy(question["initial_config"].get(class_name, {})),
                long_context=category == "long_context",
            )
        instances[class_name] = instance
    return instances


def result_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def replay_call(name: str, args: dict[str, Any], owner: dict[str, str], instances: dict[str, Any]) -> str:
    class_name = owner[name]
    if class_name not in instances or name.startswith("_"):
        raise ValueError(f"Unavailable BFCL method: {class_name}.{name}")
    method = getattr(instances[class_name], name)
    if not callable(method):
        raise ValueError(f"Non-callable BFCL method: {class_name}.{name}")
    return result_text(method(**args))


def add_step(rows: list[dict[str, Any]], category: str, group: str,
             tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> None:
    target = messages[-1]
    suffix = ("must" if messages[-2]["role"] == "tool" else "first") if target.get("tool_calls") else (
        "stop" if messages[-2]["role"] == "tool" else "text")
    rows.append({
        "source": f"bfcl_executed_v12_{category}_{suffix}",
        "source_group": group,
        "tools": tools,
        "messages": copy.deepcopy(messages),
        "loss_on_last_assistant_only": True,
        "truncate_history_to_last_user": True,
    })


def convert_episode(question: dict[str, Any], answers: list[list[str]], category: str,
                    docs_by_class: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if len(question["question"]) != len(answers):
        raise ValueError("Question/answer turn count differs")
    documents = {doc["name"]: doc for class_name in question["involved_classes"]
                 for doc in docs_by_class[class_name]}
    owner = {doc["name"]: class_name for class_name in question["involved_classes"]
             for doc in docs_by_class[class_name]}
    if len(owner) != sum(len(docs_by_class[name]) for name in question["involved_classes"]):
        raise ValueError("Ambiguous BFCL function owner")
    instances = initialize(question, category)
    hidden_by_turn = {int(i): set(names) for i, names in question.get("missed_function", {}).items()}
    hidden = set().union(*hidden_by_turn.values()) if hidden_by_turn else set()
    messages: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for turn_index, (turn_messages, call_strings) in enumerate(zip(question["question"], answers)):
        hidden.difference_update(hidden_by_turn.get(turn_index, set()))
        if turn_messages:
            messages.extend({"role": m["role"], "content": m["content"]} for m in turn_messages)
        else:
            messages.append({"role": "user", "content":
                             "The previously unavailable functions have now been added. Please continue."})
        visible_tools = [openai_tool(documents[name]) for name in documents if name not in hidden]
        parsed = [parse_call(expression, documents) for expression in call_strings]
        if any(name in hidden for name, _ in parsed):
            raise ValueError("Ground truth calls a hidden function")
        if not parsed:
            body = ("I cannot perform that action with the available functions."
                    if category == "miss_func" else
                    "Please provide the missing detail before I continue.")
            messages.append({"role": "assistant", "content": body})
            add_step(rows, category, question["id"], visible_tools, messages)
            continue
        for name, arguments in parsed:
            messages.append({"role": "assistant", "content": "", "tool_calls": [{
                "type": "function", "function": {"name": name, "arguments": arguments},
            }]})
            add_step(rows, category, question["id"], visible_tools, messages)
            observed = replay_call(name, arguments, owner, instances)
            messages.append({"role": "tool", "name": name, "content": observed})
        messages.append({"role": "assistant", "content": "I have completed the requested actions."})
        add_step(rows, category, question["id"], visible_tools, messages)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("external_data/bfcl_v3_official"))
    parser.add_argument("--out-dir", type=Path, default=Path("training_data_bfcl_executed_v12"))
    parser.add_argument("--validation-percent", type=int, default=8)
    parser.add_argument("--limit-episodes", type=int, default=0, help="Small source smoke test; 0 means all")
    parser.add_argument("--seed", type=int, default=20261104)
    args = parser.parse_args()
    guard = PublicOverlapGuard([
        Path("data/public_test.jsonl.gz"), Path("data/public_test_negotiation.jsonl.gz")
    ])
    docs_by_class = {class_name: load_lines(args.input / "multi_turn_func_doc" / filename)
                     for class_name, filename in CLASS_FILES.items()}
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    stats = collections.Counter()
    errors = collections.Counter()
    for category in CATEGORIES:
        questions = load_lines(args.input / f"BFCL_v3_multi_turn_{category}.json")
        answers = {row["id"]: row["ground_truth"] for row in load_lines(
            args.input / "possible_answer" / f"BFCL_v3_multi_turn_{category}.json")}
        for question in questions[:args.limit_episodes or None]:
            stats["episodes_seen"] += 1
            user_messages = [m for turn in question["question"] for m in turn if m.get("role") == "user"]
            if guard.row_overlaps(user_messages):
                stats["public_overlap_discarded"] += 1
                continue
            try:
                rows = convert_episode(question, answers[question["id"]], category, docs_by_class)
            except (KeyError, ValueError, TypeError, SyntaxError, AttributeError, IndexError, RuntimeError) as exc:
                stats["replay_failed"] += 1
                errors[type(exc).__name__ + ": " + str(exc)[:100]] += 1
                continue
            key = int(hashlib.sha256(f"{args.seed}:{question['id']}".encode()).hexdigest()[:8], 16) % 100
            destination = validation if key < args.validation_percent else train
            destination.extend(rows)
            stats["episodes_kept"] += 1
            stats["tool_results"] += sum(m.get("role") == "tool" for row in rows[-1:] for m in row["messages"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("validation", validation)):
        with (args.out_dir / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "generator": "code/prepare_bfcl_executed_v12.py",
        "public_source": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "source_revision": "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
        "executor": "bfcl-eval==2025.6.8 official simulator; direct literal-argument dispatch",
        "seed": args.seed,
        "train_rows": len(train), "validation_rows": len(validation),
        "stats": dict(stats), "replay_errors": dict(errors.most_common(12)),
        "public_guard": guard.stats(),
        "public_test_usage": "discard-only user-text overlap check",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
