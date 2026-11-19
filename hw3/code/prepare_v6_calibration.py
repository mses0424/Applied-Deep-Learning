#!/usr/bin/env python3
"""Build a small post-ToolACE calibration set without homework test data."""

from __future__ import annotations

import argparse
import collections
import copy
import json
import random
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_sample(rows: list[dict[str, Any]], source: str, limit: int,
                  rng: random.Random) -> list[dict[str, Any]]:
    selected = [copy.deepcopy(row) for row in rows if str(row.get("source", "")) == source]
    rng.shuffle(selected)
    return selected[:limit]


def must_call_boost(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boosted: list[dict[str, Any]] = []
    for row in rows:
        messages = row["messages"]
        for index, message in enumerate(messages):
            if (message.get("role") == "assistant" and message.get("tool_calls")
                    and index and messages[index - 1].get("role") == "tool"):
                boosted.append({
                    "source": "team_ace_toolace_must_call_boost_v1",
                    "source_group": str(row["source_group"]),
                    "loss_on_last_assistant_only": True,
                    "truncate_history_to_last_user": True,
                    "tools": copy.deepcopy(row["tools"]),
                    "messages": copy.deepcopy(messages[:index + 1]),
                })
    return boosted


def tool_name(tool: dict[str, Any]) -> str:
    return str((tool.get("function") or {}).get("name", ""))


def add_tool_distractors(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]],
                         limit: int, target_tools: int, rng: random.Random) -> list[dict[str, Any]]:
    """Make real must-call targets harder without changing their gold reply."""
    pool: list[dict[str, Any]] = []
    for row in source_rows:
        pool.extend(copy.deepcopy(row.get("tools") or []))
    selected = list(rows)
    rng.shuffle(selected)
    augmented: list[dict[str, Any]] = []
    for original in selected[:limit]:
        row = copy.deepcopy(original)
        names = {tool_name(tool) for tool in row["tools"]}
        attempts = 0
        while len(row["tools"]) < target_tools and attempts < target_tools * 50:
            candidate = copy.deepcopy(rng.choice(pool))
            name = tool_name(candidate)
            attempts += 1
            if name and name not in names:
                row["tools"].append(candidate)
                names.add(name)
        if len(row["tools"]) >= target_tools:
            rng.shuffle(row["tools"])
            row["source"] = "team_ace_toolace_must_call_12_tools_v1"
            augmented.append(row)
    return augmented


def dependency_last_only(rows: list[dict[str, Any]], limit: int,
                         rng: random.Random) -> list[dict[str, Any]]:
    selected = [copy.deepcopy(row) for row in rows
                if str(row.get("source", "")) == "bfcl_docs_dependency_curriculum_v1"]
    rng.shuffle(selected)
    out: list[dict[str, Any]] = []
    for row in selected[:limit]:
        row["source"] = "bfcl_docs_dependency_last_call_v1"
        row["loss_on_last_assistant_only"] = True
        row["truncate_history_to_last_user"] = True
        out.append(row)
    return out


def build_split(toolace_rows: list[dict[str, Any]], v5_rows: list[dict[str, Any]],
                dependency_rows: list[dict[str, Any]], rng: random.Random,
                scale: float) -> list[dict[str, Any]]:
    boosted = must_call_boost(toolace_rows)
    rows = list(boosted)
    rows.extend(add_tool_distractors(
        boosted, toolace_rows, round(4000 * scale), 12, rng
    ))
    rows.extend(dependency_last_only(dependency_rows, round(2500 * scale), rng))
    quotas = {
        "nvidia_when2call_hard_no_call_v1": round(3000 * scale),
        "argilla_synth_apigen_call_v1": round(2500 * scale),
        "targeted_apigen_stop_compact_v1": round(1500 * scale),
        "targeted_apigen_must_compact_11_v1": round(1500 * scale),
        "negotiation_balanced_order_v3": round(4000 * scale),
        "negotiation_balanced_wait_high_v3": round(1500 * scale),
        "negotiation_balanced_wait_wrong_v3": round(1500 * scale),
        "retention_bfcl_docs_dependency_curriculum_v1": round(800 * scale),
    }
    for source, limit in quotas.items():
        rows.extend(source_sample(v5_rows, source, limit, rng))
    rng.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toolace-dir", type=Path, default=ROOT / "training_data_toolace_v6")
    parser.add_argument("--v5-dir", type=Path, default=ROOT / "training_data_targeted_v5")
    parser.add_argument("--dependency-dir", type=Path, default=ROOT / "training_data_dependent")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training_data_v6_calibration")
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    for directory in (args.toolace_dir, args.v5_dir, args.dependency_dir):
        if "public_test" in str(directory.resolve()).lower():
            raise SystemExit("Refusing homework test data as a training source")
    rng = random.Random(args.seed)
    toolace_train = read_rows(args.toolace_dir / "train.jsonl")
    toolace_val = read_rows(args.toolace_dir / "validation.jsonl")
    v5_train = read_rows(args.v5_dir / "train.jsonl")
    v5_val = read_rows(args.v5_dir / "validation.jsonl")
    dependency_train = read_rows(args.dependency_dir / "train.jsonl")
    dependency_val = read_rows(args.dependency_dir / "validation.jsonl")
    train = build_split(toolace_train, v5_train, dependency_train, rng, 1.0)
    # Validation keeps the original held-out source groups and needs no large quota.
    validation = build_split(
        toolace_val, v5_val, dependency_val, random.Random(args.seed + 1), 0.04
    )
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "validation.jsonl", validation)
    train_sources = collections.Counter(str(row["source"]) for row in train)
    val_sources = collections.Counter(str(row["source"]) for row in validation)
    manifest = {
        "generator": "code/prepare_v6_calibration.py",
        "seed": args.seed,
        "policy": "Contains only prior compliant training rows and ToolACE train-derived rows; no homework PUBLIC content",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "train_by_source": dict(sorted(train_sources.items())),
        "validation_by_source": dict(sorted(val_sources.items())),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
