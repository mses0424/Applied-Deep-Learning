#!/usr/bin/env python3
"""Build the v5 error-targeted, decontaminated curriculum.

PUBLIC homework records are used only by ``Decontaminator`` as a discard-only
guard.  No PUBLIC prompt, schema, answer, or model output is written here.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import random
from pathlib import Path
from typing import Any

from generate_data import NEGOTIATION_TOOLS, PRODUCTS, buyer_system, call_message, seller_message
from prepare_corrective_data import (
    PUBLIC_FILES,
    ROOT,
    Decontaminator,
    extract_apigen_mt,
    extract_synth_apigen,
    load_retention,
    make_broad_multistep,
    sha256,
    split_rows,
    tool_name,
    write_jsonl,
)


def called_names(messages: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for message in messages:
        for call in message.get("tool_calls") or []:
            name = tool_name(call)
            if name:
                names.add(name)
        if message.get("role") == "tool" and message.get("name"):
            names.add(str(message["name"]))
    return names


def compact_tools(row: dict[str, Any], pool: list[dict[str, Any]], rng: random.Random,
                  desired: int) -> dict[str, Any]:
    """Keep every trajectory tool and a reproducible set of hard distractors."""
    row = copy.deepcopy(row)
    required = called_names(row["messages"])
    selected = {tool_name(tool): tool for tool in row["tools"] if tool_name(tool) in required}
    local = [tool for tool in row["tools"] if tool_name(tool) not in selected]
    rng.shuffle(local)
    for tool in local:
        selected.setdefault(tool_name(tool), tool)
        if len(selected) >= desired:
            break
    attempts = 0
    while len(selected) < desired and attempts < desired * 20:
        tool = rng.choice(pool)
        selected.setdefault(tool_name(tool), tool)
        attempts += 1
    row["tools"] = list(selected.values())
    rng.shuffle(row["tools"])
    return row


def make_compact_apigen(rows: list[dict[str, Any]], synth_rows: list[dict[str, Any]],
                        rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, int]]:
    pool_by_name: dict[str, dict[str, Any]] = {}
    for row in rows + synth_rows:
        for tool in row["tools"]:
            pool_by_name.setdefault(tool_name(tool), tool)
    pool = list(pool_by_name.values())
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        buckets[str(row["source"])].append(row)

    output: list[dict[str, Any]] = []
    # Two schema layouts expose every real post-tool must-call target while
    # matching the 11--20-tool evaluation slice that was weakest in v4.
    must = buckets["apigen_mt_must_call_v1"]
    for variant, desired in enumerate((11, 12)):
        for row in must:
            item = compact_tools(row, pool, rng, desired)
            item["source"] = f"targeted_apigen_must_compact_{desired}_v1"
            # Keep variants of one source trajectory in the same split.
            item["source_group"] = str(row["source_group"])
            item["schema_variant"] = variant
            output.append(item)

    selections = (
        ("apigen_mt_stop_after_tool_v1", 3500, 10, "targeted_apigen_stop_compact_v1"),
        ("apigen_mt_eligible_call_v1", 3000, 12, "targeted_apigen_eligible_compact_v1"),
        ("apigen_mt_text_response_v1", 1000, 10, "targeted_apigen_text_compact_v1"),
    )
    for source, limit, desired, new_source in selections:
        chosen = buckets[source][:limit]
        for row in chosen:
            item = compact_tools(row, pool, rng, desired)
            item["source"] = new_source
            output.append(item)
    return output, {
        "unique_tool_pool": len(pool),
        "must_source_rows": len(must),
        "must_compact_variants": len(must) * 2,
        "all_compact_rows": len(output),
    }


TYPE_MAP = {"dict": "object", "list": "array", "str": "string", "int": "integer",
            "float": "number", "bool": "boolean"}


def normalize_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [normalize_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {key: normalize_schema(item) for key, item in value.items()}
    if isinstance(out.get("type"), str):
        out["type"] = TYPE_MAP.get(out["type"], out["type"])
    return out


def normalize_when_tool(raw: str) -> dict[str, Any] | None:
    try:
        function = normalize_schema(json.loads(raw))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
        return None
    parameters = function.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    parameters["type"] = "object"
    if "required" in function and "required" not in parameters:
        parameters["required"] = function.pop("required")
    parameters.setdefault("properties", {})
    function["parameters"] = parameters
    return {"type": "function", "function": function}


def load_when2call(path: Path, guard: Decontaminator, rng: random.Random,
                   limit: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    parsed: list[tuple[int, list[dict[str, Any]], list[dict[str, Any]]]] = []
    pool_by_name: dict[str, dict[str, Any]] = {}
    stats = collections.Counter()
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            raw = json.loads(line)
            messages = raw.get("messages") or []
            tools = [tool for text in raw.get("tools") or [] if (tool := normalize_when_tool(text))]
            if not tools or not messages or messages[-1].get("role") != "assistant":
                stats["discarded_invalid"] += 1
                continue
            if messages[-1].get("tool_calls"):
                stats["discarded_call_target"] += 1
                continue
            if guard.row_overlaps(messages):
                stats["discarded_public_overlap"] += 1
                continue
            for tool in tools:
                pool_by_name.setdefault(tool_name(tool), tool)
            parsed.append((index, copy.deepcopy(messages), tools))
    rng.shuffle(parsed)
    pool = list(pool_by_name.values())
    rows: list[dict[str, Any]] = []
    for position, (index, messages, tools) in enumerate(parsed[:limit]):
        selected = {tool_name(tool): tool for tool in tools}
        desired = 12 if position % 2 else 18
        attempts = 0
        while len(selected) < desired and attempts < desired * 20:
            tool = rng.choice(pool)
            selected.setdefault(tool_name(tool), tool)
            attempts += 1
        tool_list = list(selected.values())
        rng.shuffle(tool_list)
        rows.append({
            "source": "nvidia_when2call_hard_no_call_v1",
            "source_group": f"when2call-{index}",
            "loss_on_last_assistant_only": True,
            "tools": tool_list,
            "messages": messages,
        })
    stats["eligible"] = len(parsed)
    stats["selected"] = len(rows)
    stats["unique_tool_pool"] = len(pool)
    return rows, dict(stats)


def make_balanced_negotiations(rng: random.Random, scenarios: int) -> list[dict[str, Any]]:
    """Four exact order decisions and two wait decisions per scenario."""
    rows: list[dict[str, Any]] = []
    for index in range(scenarios):
        pid, item = rng.choice(PRODUCTS)
        qty = rng.choice([1, 1, 2, 3])
        target = round(rng.uniform(4.0, 45.0), 2)
        budget = round(target * qty * rng.uniform(1.25, 1.55), 2)
        system = buyer_system(pid, item, qty, budget, target)
        state = {"role": "user", "content": json.dumps({"phase": "negotiation", "turns_remaining_in_phase": 15})}
        opening = call_message("wait", {}, f"Could you offer {qty} of {pid} near ${target * .82:.2f} each?")
        base = [{"role": "system", "content": system}, state, opening]
        group = f"negotiation-balanced-{index}"

        for order_index, factor in enumerate((0.84, 0.93, 1.00, 1.04)):
            price = round(min(target * factor, budget / qty * 0.98), 2)
            mid = f"deal-{index}-{order_index}"
            quote = f"Offer: {pid} at ${price:.2f} per unit, qty {qty}."
            messages = copy.deepcopy(base)
            messages.append({"role": "user", "content": seller_message(
                mid, pid, item, price, qty, 14 - order_index, "This is my current formal offer.")})
            messages.append(call_message("order", {
                "product_id": pid, "agree_price": price, "qty": qty,
                "evidence_message_ids": [mid], "evidence_quote": quote,
            }, "Agreed. I accept this current offer."))
            rows.append({"source": "negotiation_balanced_order_v3", "source_group": group,
                         "loss_on_last_assistant_only": True,
                         "tools": copy.deepcopy(NEGOTIATION_TOOLS), "messages": messages})

        high = round(min(target * 1.18, budget / qty * 0.98), 2)
        high_mid = f"high-{index}"
        high_messages = copy.deepcopy(base)
        high_messages.append({"role": "user", "content": seller_message(
            high_mid, pid, item, high, qty, 13, "This is my current formal offer.")})
        high_messages.append(call_message("wait", {},
            f"${high:.2f} is above my goal. Could you get closer to ${target:.2f}?"))
        rows.append({"source": "negotiation_balanced_wait_high_v3", "source_group": group,
                     "loss_on_last_assistant_only": True,
                     "tools": copy.deepcopy(NEGOTIATION_TOOLS), "messages": high_messages})

        wrong_qty = qty + 1
        wrong_messages = copy.deepcopy(base)
        wrong_messages.append({"role": "user", "content": seller_message(
            f"wrong-{index}", pid, item, round(target * .90, 2), wrong_qty, 13,
            "This is my current formal offer.")})
        wrong_messages.append(call_message("wait", {},
            f"I need exactly {qty}, not {wrong_qty}. Please quote the requested quantity."))
        rows.append({"source": "negotiation_balanced_wait_wrong_v3", "source_group": group,
                     "loss_on_last_assistant_only": True,
                     "tools": copy.deepcopy(NEGOTIATION_TOOLS), "messages": wrong_messages})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apigen-mt", type=Path,
                        default=ROOT / "external_data/apigen_mt_parsed/apigen-mt-5k.parquet")
    parser.add_argument("--synth-apigen", type=Path,
                        default=ROOT / "external_data/synth_apigen/data/train-00000-of-00001.parquet")
    parser.add_argument("--when2call", type=Path,
                        default=ROOT / "external_data/when2call/train/when2call_train_sft.jsonl")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training_data_targeted_v5")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--single-turn", type=int, default=12000)
    parser.add_argument("--broad-multistep", type=int, default=12000)
    parser.add_argument("--when2call-limit", type=int, default=10000)
    parser.add_argument("--negotiations", type=int, default=2000)
    args = parser.parse_args()
    for path in (args.apigen_mt, args.synth_apigen, args.when2call):
        if "public_test" in str(path.resolve()).lower():
            raise SystemExit(f"Refusing test data as a training source: {path}")
        if not path.is_file():
            raise SystemExit(f"Missing training source: {path}")

    rng = random.Random(args.seed)
    guard = Decontaminator(PUBLIC_FILES)
    mt_raw, mt_stats = extract_apigen_mt(args.apigen_mt, guard, rng)
    single_rows, single_stats = extract_synth_apigen(args.synth_apigen, guard, rng, args.single_turn)
    compact_mt, compact_stats = make_compact_apigen(mt_raw, single_rows, rng)
    broad_rows = make_broad_multistep(single_rows, rng, args.broad_multistep)
    for row in broad_rows:
        row["source"] = "targeted_argilla_multistep_v2"
    no_call_rows, when_stats = load_when2call(args.when2call, guard, rng, args.when2call_limit)
    negotiation_rows = make_balanced_negotiations(rng, args.negotiations)

    retention: list[dict[str, Any]] = []
    retention += load_retention(ROOT / "training_data_dependent/train.jsonl", rng, 1500, "bfcl_docs_dependency")
    retention += load_retention(ROOT / "training_data_v2/train.jsonl", rng, 2000, "original_synthetic_tools")
    retention += load_retention(ROOT / "training_data_v2/train.jsonl", rng, 1000, "original_synthetic_negotiation")

    rows = compact_mt + single_rows + broad_rows + no_call_rows + negotiation_rows + retention
    train, validation = split_rows(rows, args.seed)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "validation.jsonl", validation)
    sources = collections.Counter(str(row["source"]) for row in rows)
    manifest = {
        "generator": "code/prepare_targeted_data.py",
        "seed": args.seed,
        "policy": "PUBLIC is used only as a discard-only exact/near-duplicate guard; no PUBLIC content is written",
        "public_user_texts_held_out": len(guard.exact),
        "decontamination": {
            "texts_checked": guard.checked,
            "exact_overlaps_discarded": guard.exact_hits,
            "near_overlaps_discarded": guard.near_hits,
        },
        "sources": [
            {"name": "Salesforce APIGen-MT-5k", "url": "https://huggingface.co/datasets/Salesforce/APIGen-MT-5k",
             "license": "CC-BY-NC-4.0", "sha256": sha256(args.apigen_mt)},
            {"name": "Argilla Synth-APIGen-v0.1", "url": "https://huggingface.co/datasets/argilla/Synth-APIGen-v0.1",
             "license": "Apache-2.0", "sha256": sha256(args.synth_apigen)},
            {"name": "NVIDIA When2Call", "url": "https://huggingface.co/datasets/nvidia/When2Call",
             "license": "CC-BY-4.0", "sha256": sha256(args.when2call),
             "scope": "train_sft split only; no When2Call test records"},
            {"name": "Original balanced negotiation synthesis and prior compliant retention data"},
        ],
        "apigen_mt": mt_stats,
        "compact_apigen": compact_stats,
        "synth_apigen": single_stats,
        "when2call": when_stats,
        "broad_multistep_rows": len(broad_rows),
        "negotiation_scenarios": args.negotiations,
        "negotiation_order_rows": sum(row["source"].endswith("order_v3") for row in negotiation_rows),
        "negotiation_wait_rows": sum("wait" in row["source"] for row in negotiation_rows),
        "retention_rows": len(retention),
        "all_rows": len(rows), "train_rows": len(train), "validation_rows": len(validation),
        "rows_by_source": dict(sorted(sources.items())),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
