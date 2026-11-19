#!/usr/bin/env python3
"""Build a fresh Base-model curriculum from independent public and local data.

No peer artifact is read.  Original source conversations stay in one split;
PUBLIC is used only by a discard-only overlap guard.  BFCL function lists are
shortened using its public source documentation, while all visible history and
target tool names remain available.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from public_guard_v10 import PublicOverlapGuard


ROOT = Path(__file__).resolve().parents[1]
SOURCES: dict[str, tuple[str, dict[str, int]]] = {
    "training_data_policy_v11": ("policy", {"policy_v11_": 999999}),
    "training_data_policy_v10": ("policy", {"policy_sim_v10_": 2800}),
    "training_data_bfcl_official_v10": ("bfcl", {"bfcl_official_v10_": 999999}),
    "training_data_bfcl_v9": ("bfcl", {
        "bfcl_v9_dependent_first": 650, "bfcl_v9_dependent_must": 1200,
        "bfcl_v9_dependent_stop": 650, "bfcl_v9_independent_first": 650,
        "bfcl_v9_independent_must": 1200, "bfcl_v9_independent_stop": 650,
    }),
    "training_data_toucan_v10": ("toucan", {
        "toucan_v10_multi-turn": 2600, "toucan_v10_single-turn-original": 700,
        "toucan_v10_single-turn-diversify": 350,
    }),
    "training_data_targeted_v5": ("general", {
        "targeted_apigen_must_compact_11_v1": 4100,
        "targeted_apigen_eligible_compact_v1": 2000,
        "targeted_apigen_stop_compact_v1": 1800,
        "argilla_synth_apigen_call_v1": 2600,
        "nvidia_when2call_hard_no_call_v1": 3000,
        "negotiation_balanced_order_v3": 3700,
        "negotiation_balanced_wait_high_v3": 1100,
        "negotiation_balanced_wait_wrong_v3": 1100,
    }),
    "training_data_v6_calibration": ("general", {
        "team_ace_toolace_must_call_boost_v1": 800,
    }),
}
ANNOUNCEMENT = re.compile(
    r"\b(let me|i will|i'll|allow me|i am going to|i'm going to)\b.{0,110}"
    r"\b(check|look up|search|find|fetch|verify|retrieve|call|use the tool)\b",
    re.I | re.S,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if "public_test" in str(path).lower():
        raise ValueError("A PUBLIC evaluation path cannot be a training source")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def group(row: dict[str, Any]) -> str:
    if row.get("source_group"):
        return str(row["source_group"])
    raw = json.dumps(row.get("messages", []), sort_keys=True, ensure_ascii=False)
    return "row-" + hashlib.sha256(raw.encode()).hexdigest()


def call_names(row: dict[str, Any]) -> set[str]:
    return {
        str(call.get("function", {}).get("name", ""))
        for message in row.get("messages", []) if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    } - {""}


def compact_bfcl_tools(row: dict[str, Any], rng: random.Random, limit: int = 16) -> dict[str, Any]:
    """Preserve used functions; sample local semantic distractors from visible docs."""
    tools = row.get("tools") or []
    if len(tools) <= limit:
        return row
    used = call_names(row)
    must_keep = [tool for tool in tools if tool.get("function", {}).get("name") in used]
    if len(must_keep) > limit:
        return row
    candidate = [tool for tool in tools if tool not in must_keep]
    def keywords(tool: dict[str, Any]) -> set[str]:
        function = tool.get("function") or {}
        blob = str(function.get("name", "")).replace("_", " ") + " " + str(function.get("description", ""))
        return set(re.findall(r"[a-z]{4,}", blob.lower())) - {"with", "this", "that", "from", "function"}
    wanted = set().union(*(keywords(tool) for tool in must_keep)) if must_keep else set()
    rng.shuffle(candidate)
    candidate.sort(key=lambda tool: len(keywords(tool) & wanted), reverse=True)
    selected = must_keep + candidate[:limit - len(must_keep)]
    rng.shuffle(selected)
    output = copy.copy(row)
    output["tools"] = selected
    return output


def remove_bfcl_hindsight(row: dict[str, Any]) -> None:
    """Delete artificial fields that disclose the next gold tool arguments."""
    for message in row.get("messages") or []:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content") or "")
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and "values_for_next_step" in payload:
            del payload["values_for_next_step"]
            message["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def action_announcement(row: dict[str, Any]) -> bool:
    messages = row.get("messages") or []
    if not messages or messages[-1].get("role") != "assistant":
        return False
    if messages[-1].get("tool_calls"):
        return False
    body = messages[-1].get("content") or ""
    return bool(isinstance(body, str) and "?" not in body and ANNOUNCEMENT.search(body))


def semantic_user_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Exclude protocol-only counters; retain all substantive seller speech."""
    kept = []
    for message in row.get("messages") or []:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            payload = None
        if (isinstance(payload, dict) and set(payload) == {"phase", "turns_remaining_in_phase"}
                and payload.get("phase") == "negotiation"
                and isinstance(payload.get("turns_remaining_in_phase"), int)):
            continue
        kept.append(message)
    return kept


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training_data_fresh_v11")
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--bfcl-executed", action="store_true",
                        help="Use the independently replayed official BFCL simulator results in place of synthetic BFCL results.")
    parser.add_argument("--policy-branches", action="store_true",
                        help="Include original v12 policy-branch examples with observable prerequisites.")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    sources = dict(SOURCES)
    if args.bfcl_executed:
        del sources["training_data_bfcl_official_v10"]
        sources["training_data_bfcl_executed_v12"] = ("bfcl", {"bfcl_executed_v12_": 999999})
    if args.policy_branches:
        sources["training_data_policy_branch_v12"] = ("policy", {"policy_branch_v12_": 999999})
    guard = PublicOverlapGuard([
        ROOT / "data/public_test.jsonl.gz", ROOT / "data/public_test_negotiation.jsonl.gz"
    ])
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    manifest_sources: dict[str, Any] = {}
    dropped = collections.Counter()
    for directory, (family, quotas) in sources.items():
        base = ROOT / directory
        source_rows = {}
        for split in ("train", "validation"):
            grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
            for row in read_jsonl(base / f"{split}.jsonl"):
                name = str(row.get("source", ""))
                if any(name == prefix or name.startswith(prefix) for prefix in quotas):
                    grouped[name].append(row)
            selected: list[dict[str, Any]] = []
            for prefix, quota in quotas.items():
                candidates = [row for name, rows in grouped.items()
                              if name == prefix or name.startswith(prefix) for row in rows]
                if not candidates:
                    raise SystemExit(f"Missing required {directory}:{split}:{prefix}")
                rng.shuffle(candidates)
                # Validation comes from an existing held-out split.  Keep a
                # modest sample that tracks each chosen training source.
                wanted = min(len(candidates), quota if split == "train" else max(12, round(quota * 0.025)))
                for original in candidates[:wanted]:
                    row = copy.deepcopy(original)
                    if family == "bfcl" and directory in {"training_data_bfcl_official_v10", "training_data_bfcl_executed_v12"}:
                        if directory.endswith("official_v10"):
                            remove_bfcl_hindsight(row)
                        row = compact_bfcl_tools(row, rng)
                    if action_announcement(row):
                        dropped["action_announcement"] += 1
                        continue
                    if guard.row_overlaps(semantic_user_messages(row)):
                        dropped["public_user_overlap"] += 1
                        continue
                    selected.append(row)
            source_rows[split] = selected
        train.extend(source_rows["train"])
        validation.extend(source_rows["validation"])
        manifest_sources[directory] = {
            "source_manifest_sha256": hashlib.sha256((base / "manifest.json").read_bytes()).hexdigest(),
            "train_rows": len(source_rows["train"]),
            "validation_rows": len(source_rows["validation"]),
        }

    # Multiple curricula can contain the same original conversation.  Its
    # source_group is a stable identifier, so all such rows stay on one side.
    training_groups = {group(row) for row in train}
    validation_before = len(validation)
    validation = [row for row in validation if group(row) not in training_groups]
    dropped["validation_group_collision"] = validation_before - len(validation)
    if not validation or training_groups & {group(row) for row in validation}:
        raise SystemExit("Training/validation group isolation failed")
    rng.shuffle(train)
    rng.shuffle(validation)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "train.jsonl", train)
    write_rows(args.out_dir / "validation.jsonl", validation)
    manifest = {
        "generator": "code/prepare_fresh_v11.py", "seed": args.seed,
        "design": ("Fresh Base-model training with independent stateful policy data, "
                   "BFCL, APIGen, Toucan, no-call, and buyer trajectories"),
        "bfcl_executed": args.bfcl_executed,
        "policy_branches": args.policy_branches,
        "sources": manifest_sources, "train_rows": len(train),
        "validation_rows": len(validation),
        "train_by_source": dict(sorted(collections.Counter(row["source"] for row in train).items())),
        "validation_by_source": dict(sorted(collections.Counter(row["source"] for row in validation).items())),
        "discarded": dict(sorted(dropped.items())),
        "public_guard": guard.stats(),
        "public_test_usage": "discard-only user text overlap check; no test record output",
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("train_rows", "validation_rows", "discarded", "public_guard")}, indent=2))


if __name__ == "__main__":
    main()
