#!/usr/bin/env python3
"""Synthesize new stateful tool tasks from BFCL function documentation only.

No BFCL question, answer, trajectory, or HW3 PUBLIC record is read.  Targets
are derived from sampled JSON Schema arguments and simulated tool results.
Each trajectory contributes balanced first-call, post-result call, and stop
decisions, with related same-domain distractor functions.
"""

from __future__ import annotations

import argparse
import collections
import copy
import glob
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from generate_dependent import eligible_pairs
from synthesize_bfcl_docs import sample_value


ROOT = Path(__file__).resolve().parents[1]
STOP_WORDS = {"the", "and", "from", "with", "that", "this", "your", "into", "using", "tool", "api", "for", "get", "set"}


def words(doc: dict[str, Any]) -> set[str]:
    function = doc["name"].replace("_", " ") + " " + str(doc.get("description", ""))
    function += " " + " ".join((doc.get("parameters") or {}).get("properties") or {})
    return {word for word in re.findall(r"[a-z]{3,}", function.lower()) if word not in STOP_WORDS}


def safe_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [safe_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {name: safe_schema(item) for name, item in value.items()}
    kind = out.get("type")
    if isinstance(kind, str):
        out["type"] = {"dict": "object", "list": "array", "tuple": "array",
                       "str": "string", "int": "integer", "float": "number",
                       "bool": "boolean"}.get(kind, kind)
    elif kind is not None:
        out.pop("type")
    return out


def safe_openai_tool(doc: dict[str, Any]) -> dict[str, Any]:
    params = safe_schema(doc.get("parameters") or {})
    if not isinstance(params, dict):
        params = {}
    params["type"] = "object"
    if not isinstance(params.get("properties"), dict):
        params["properties"] = {}
    return {"type": "function", "function": {"name": doc["name"],
                                                "description": doc.get("description", ""),
                                                "parameters": params}}


def same_domain_distractors(docs: list[dict[str, Any]], targets: list[dict[str, Any]],
                            all_docs: list[dict[str, Any]], rng: random.Random,
                            desired: int) -> list[dict[str, Any]]:
    names = {item["name"] for item in targets}
    target_words = [words(item) for item in targets]
    candidates = [item for item in docs if item["name"] not in names]
    rng.shuffle(candidates)
    candidates.sort(key=lambda item: max(len(words(item) & target) for target in target_words), reverse=True)
    selected = list(targets)
    for item in candidates:
        if len(selected) >= desired:
            break
        if item["name"] not in names:
            selected.append(item)
            names.add(item["name"])
    if len(selected) < desired:
        extras = list(all_docs)
        rng.shuffle(extras)
        for item in extras:
            if len(selected) >= desired:
                break
            if item["name"] not in names:
                selected.append(item)
                names.add(item["name"])
    rng.shuffle(selected)
    return [safe_openai_tool(doc) for doc in selected]


def action(doc: dict[str, Any]) -> str:
    value = str(doc.get("description") or doc["name"].replace("_", " ")).strip()
    value = value.split("Tool description:")[-1].strip().rstrip(". ")
    return value[0].lower() + value[1:] if value else doc["name"].replace("_", " ")


def details(arguments: dict[str, Any]) -> str:
    if not arguments:
        return ""
    return " Specify " + ", ".join(
        f"{name.replace('_', ' ')} as {json.dumps(value, ensure_ascii=False)}"
        for name, value in arguments.items()
    ) + "."


def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {
        "name": name, "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
    }}]}


def safe_value(name: str, prop: Any, rng: random.Random, index: int) -> Any:
    if not isinstance(prop, dict):
        return f"generated_{name}_{index}"
    try:
        return sample_value(name, prop, rng, index)
    except (TypeError, ValueError, KeyError):
        # Some BFCL documentation has non-JSON-Schema type descriptors.
        # Keep a simple string rather than fabricate a nested typed value.
        return f"generated_{name}_{index}"


def safe_arguments(doc: dict[str, Any], rng: random.Random, index: int) -> dict[str, Any]:
    params = doc.get("parameters") or {}
    props = params.get("properties") or {}
    if not isinstance(props, dict):
        props = {}
    required = set(params.get("required") or [])
    return {name: safe_value(name, prop, rng, index) for name, prop in props.items()
            if name in required or rng.random() < 0.30}


def result(doc: dict[str, Any], rng: random.Random, sample_index: int,
           inputs: dict[str, Any]) -> dict[str, Any]:
    schema = doc.get("response") or {}
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        props = {}
    output = {name: safe_value(name, prop, rng, sample_index)
              for name, prop in list(props.items())[:6]}
    for name, value in inputs.items():
        if name in output:
            output[name] = value
    return output or {"status": "success", "request_id": f"generated-{sample_index}"}


def step_rows(domain: str, group: str, source: str,
              tools: list[dict[str, Any]], messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        output.append({
            "source": source + ("_must" if messages[index - 1].get("role") == "tool" and message.get("tool_calls")
                               else "_stop" if not message.get("tool_calls") else "_first"),
            "source_group": group,
            "domain": domain,
            "loss_on_last_assistant_only": True,
            "tools": copy.deepcopy(tools),
            "messages": copy.deepcopy(messages[:index + 1]),
        })
    return output


def build_dependent(domain: str, docs: list[dict[str, Any]], all_docs: list[dict[str, Any]],
                    pairs: list[tuple[dict[str, Any], dict[str, Any], list[str]]],
                    rng: random.Random, index: int) -> list[dict[str, Any]]:
    first, second, shared = rng.choice(pairs)
    args1 = safe_arguments(first, rng, index * 11)
    args2 = safe_arguments(second, rng, index * 11 + 1)
    output1 = result(first, rng, index * 11 + 2, args1)
    second_props = (second.get("parameters") or {}).get("properties") or {}
    for field in shared:
        value = safe_value(field, second_props[field], rng, index * 11 + 3)
        output1[field] = value
        args2[field] = value
    independent_args = {name: value for name, value in args2.items() if name not in shared}
    language = (
        f"First, {action(first)}." + details(args1) +
        f" After its result arrives, {action(second)} using the returned " +
        ", ".join(x.replace("_", " ") for x in shared) + "." + details(independent_args)
    )
    tools = same_domain_distractors(docs, [first, second], all_docs, rng, rng.choice((12, 16, 20)))
    output2 = result(second, rng, index * 11 + 4, args2)
    messages = [
        {"role": "user", "content": language},
        call(first["name"], args1),
        {"role": "tool", "name": first["name"], "content": json.dumps(output1, ensure_ascii=False)},
        call(second["name"], args2),
        {"role": "tool", "name": second["name"], "content": json.dumps(output2, ensure_ascii=False)},
        {"role": "assistant", "content": "Both requested actions are complete."},
    ]
    group = f"bfcl-v9-dependent-{domain}-{first['name']}-{second['name']}"
    return step_rows(domain, group, "bfcl_v9_dependent", tools, messages)


def build_independent(domain: str, docs: list[dict[str, Any]], all_docs: list[dict[str, Any]],
                      rng: random.Random, index: int) -> list[dict[str, Any]]:
    first, second = rng.sample(docs, 2)
    args1 = safe_arguments(first, rng, index * 13)
    args2 = safe_arguments(second, rng, index * 13 + 1)
    language = f"Please do these in order. First, {action(first)}." + details(args1)
    language += f" Once that finishes, {action(second)}." + details(args2)
    tools = same_domain_distractors(docs, [first, second], all_docs, rng, rng.choice((12, 16, 20)))
    output1 = result(first, rng, index * 13 + 2, args1)
    output2 = result(second, rng, index * 13 + 3, args2)
    messages = [
        {"role": "user", "content": language},
        call(first["name"], args1),
        {"role": "tool", "name": first["name"], "content": json.dumps(output1, ensure_ascii=False)},
        call(second["name"], args2),
        {"role": "tool", "name": second["name"], "content": json.dumps(output2, ensure_ascii=False)},
        {"role": "assistant", "content": "Both requested actions are complete."},
    ]
    group = f"bfcl-v9-independent-{domain}-{first['name']}-{second['name']}"
    return step_rows(domain, group, "bfcl_v9_independent", tools, messages)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docs", type=Path, default=ROOT / "external_data/bfcl_docs/multi_turn_func_doc")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training_data_bfcl_v9")
    parser.add_argument("--trajectories", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=20261030)
    args = parser.parse_args()
    if "multi_turn_func_doc" not in str(args.docs.resolve()):
        raise SystemExit("Only BFCL function documentation is permitted")
    rng = random.Random(args.seed)
    domains = []
    paths = sorted(Path(path) for path in glob.glob(str(args.docs / "*.json")))
    for path in paths:
        docs = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if not all(set(x) == {"name", "description", "parameters", "response"} for x in docs):
            raise ValueError(f"Unexpected non-document record in {path}")
        domains.append((path.stem, docs, eligible_pairs(docs)))
    all_docs = [item for _, docs, _ in domains for item in docs]
    rows = []
    for index in range(args.trajectories):
        domain, docs, pairs = domains[index % len(domains)]
        if pairs and index % 2 == 0:
            rows.extend(build_dependent(domain, docs, all_docs, pairs, rng, index))
        elif len(docs) >= 2:
            rows.extend(build_independent(domain, docs, all_docs, rng, index))
    # Hold out complete function pairs within each task type. This gives both
    # dependent and independent validation coverage without leaking variants.
    groups_by_type: dict[str, set[str]] = collections.defaultdict(set)
    for row in rows:
        kind = "independent" if "independent" in row["source"] else "dependent"
        groups_by_type[kind].add(row["source_group"])
    validation_groups = set()
    for groups in groups_by_type.values():
        ordered = sorted(groups, key=lambda group: hashlib.sha256(
            (str(args.seed) + ":" + group).encode()).digest())
        validation_groups.update(ordered[:max(1, round(len(ordered) * 0.04))])
    train, validation = [], []
    for row in rows:
        (validation if row["source_group"] in validation_groups else train).append(row)
    rng.shuffle(train)
    rng.shuffle(validation)
    write = lambda path, data: path.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in data))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write(args.out_dir / "train.jsonl", train)
    write(args.out_dir / "validation.jsonl", validation)
    manifest = {
        "generator": "code/prepare_bfcl_stateful_v9.py",
        "policy": "Only BFCL function documentation; no questions, answers, trajectories, or HW3 PUBLIC input",
        "doc_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "seed": args.seed,
        "trajectories": args.trajectories,
        "train_rows": len(train), "validation_rows": len(validation),
        "train_by_source": dict(sorted(collections.Counter(x["source"] for x in train).items())),
        "validation_by_source": dict(sorted(collections.Counter(x["source"] for x in validation).items())),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
