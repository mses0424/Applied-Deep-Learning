#!/usr/bin/env python3
"""Generate original, dependency-aware multi-tool conversations from function docs.

Only function documentation is read.  A chain is eligible when a field returned
by its first function is a required argument of its second function.  The tool
result supplies a newly sampled value and the second gold call must copy it.
"""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
from pathlib import Path
from typing import Any

from synthesize_bfcl_docs import normalize_schema, openai_tool, sample_arguments, sample_value


DEPENDENCY_FIELDS = {
    "access_token", "account_id", "booking_id", "card_id", "order_id",
    "symbol", "distance", "gallon", "liter", "user_id", "username",
}


def action(doc: dict[str, Any]) -> str:
    text = doc.get("description", "perform the requested action").strip()
    if "Tool description:" in text:
        text = text.split("Tool description:", 1)[1].strip()
    return text.rstrip(". ")


def friendly(name: str) -> str:
    return name.replace("_", " ")


def details(arguments: dict[str, Any]) -> str:
    if not arguments:
        return ""
    rendered = ", ".join(f"{friendly(k)} = {json.dumps(v, ensure_ascii=False)}" for k, v in arguments.items())
    return f" Details: {rendered}."


def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments, ensure_ascii=False),
        }}],
    }


def sample_response(
    doc: dict[str, Any], rng: random.Random, index: int, inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema = normalize_schema(doc.get("response") or {})
    props = schema.get("properties") or {}
    result = {name: sample_value(name, prop, rng, index) for name, prop in props.items()}
    # APIs commonly echo identifiers and attributes from their inputs.  Keeping
    # those values consistent makes a chained conversation semantically sound.
    for name, value in (inputs or {}).items():
        if name in result:
            result[name] = value
    return result


def eligible_pairs(docs: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any], list[str]]]:
    pairs = []
    for first in docs:
        returned = set((first.get("response") or {}).get("properties") or {})
        for second in docs:
            if first is second:
                continue
            required = set((second.get("parameters") or {}).get("required") or [])
            shared = sorted(returned & required & DEPENDENCY_FIELDS)
            if shared:
                pairs.append((first, second, shared))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=Path, default=Path("external_data/bfcl_docs/multi_turn_func_doc"))
    ap.add_argument("--negotiation", type=Path, default=Path("training_data_v2/train.jsonl"))
    ap.add_argument("--out-dir", type=Path, default=Path("training_data_dependent"))
    ap.add_argument("--chains", type=int, default=5000)
    ap.add_argument("--negotiations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=982451653)
    args = ap.parse_args()
    for source in (args.docs, args.negotiation):
        if "public_test" in str(source.resolve()).lower():
            raise SystemExit("Refusing public-test data as a training source")

    rng = random.Random(args.seed)
    domains: list[tuple[str, list[dict[str, Any]], list[tuple[dict[str, Any], dict[str, Any], list[str]]]]] = []
    for filename in sorted(glob.glob(str(args.docs / "*.json"))):
        path = Path(filename)
        docs = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        pairs = eligible_pairs(docs)
        if pairs:
            domains.append((path.stem, docs, pairs))
    if not domains:
        raise SystemExit("No dependency-bearing function pairs found")

    rows: list[dict[str, Any]] = []
    pair_counts: dict[str, int] = {}
    for i in range(args.chains):
        domain, docs, pairs = domains[i % len(domains)]
        first, second, shared = pairs[(i // len(domains)) % len(pairs)]
        args1 = sample_arguments(first, rng, i * 2)
        args2 = sample_arguments(second, rng, i * 2 + 1)
        result1 = sample_response(first, rng, i * 3, args1)
        second_props = (second.get("parameters") or {}).get("properties") or {}
        for field in shared:
            # Sample against the consumer's schema.  Some public docs have a
            # loose or inaccurate response type, while the required input type
            # is the contract the next call must satisfy.
            dependency = args1.get(field, sample_value(field, second_props[field], rng, i * 3 + 2))
            result1[field] = dependency
            args2[field] = dependency
        nondependent = {k: v for k, v in args2.items() if k not in shared}
        request = (
            f"First, {action(first)[0].lower() + action(first)[1:]}." + details(args1) +
            f" After that succeeds, use the returned {', '.join(friendly(x) for x in shared)} to "
            f"{action(second)[0].lower() + action(second)[1:]}." + details(nondependent)
        )
        selected = [first, second]
        distractors = [doc for doc in docs if doc["name"] not in {first["name"], second["name"]}]
        selected += rng.sample(distractors, min(8, len(distractors)))
        rng.shuffle(selected)
        result2 = sample_response(second, rng, i * 3 + 1, args2) or {"status": "success"}
        messages = [
            {"role": "user", "content": request},
            call(first["name"], args1),
            {"role": "tool", "name": first["name"], "content": json.dumps(result1, ensure_ascii=False)},
            call(second["name"], args2),
            {"role": "tool", "name": second["name"], "content": json.dumps(result2, ensure_ascii=False)},
            {"role": "assistant", "content": "Done — both requested actions completed successfully."},
        ]
        rows.append({
            "source": "bfcl_docs_dependency_curriculum_v1",
            "domain": domain,
            "dependency_fields": shared,
            "tools": [openai_tool(doc) for doc in selected],
            "messages": messages,
        })
        key = f"{domain}:{first['name']}->{second['name']}"
        pair_counts[key] = pair_counts.get(key, 0) + 1

    negotiation_rows = []
    for line in args.negotiation.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("source") == "original_synthetic_negotiation_v1":
            negotiation_rows.append(row)
    rng.shuffle(negotiation_rows)
    preserved = negotiation_rows[: min(args.negotiations, len(negotiation_rows))]
    rows.extend(preserved)
    rng.shuffle(rows)

    validation_size = max(1, round(args.chains * 0.02))
    validation = [row for row in rows if row.get("source") == "bfcl_docs_dependency_curriculum_v1"][:validation_size]
    validation_ids = {id(row) for row in validation}
    train = [row for row in rows if id(row) not in validation_ids]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename, split in (("train.jsonl", train), ("validation.jsonl", validation)):
        with (args.out_dir / filename).open("w", encoding="utf-8") as handle:
            for row in split:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {
        "source_scope": "BFCL multi_turn_func_doc only plus original synthetic negotiation preservation rows",
        "seed": args.seed,
        "dependency_chains": args.chains,
        "negotiation_rows_preserved": len(preserved),
        "train_conversations": len(train),
        "validation_conversations": len(validation),
        "unique_dependency_pairs": len(pair_counts),
        "pair_counts": pair_counts,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "pair_counts"}, indent=2))


if __name__ == "__main__":
    main()
