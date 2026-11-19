#!/usr/bin/env python3
"""Build a decontaminated corrective curriculum for both HW3 parts.

The output contains only public *training* datasets and original synthetic
examples.  Homework PUBLIC files are read only to reject exact/near duplicate
user text; no PUBLIC prompt, tool schema, answer, or model reply is copied to
the output.
"""

from __future__ import annotations

import argparse
import collections
import copy
import difflib
import gzip
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from generate_data import NEGOTIATION_TOOLS, PRODUCTS, buyer_system, call_message, seller_message


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = (ROOT / "data/public_test.jsonl.gz", ROOT / "data/public_test_negotiation.jsonl.gz")


def normalized(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def word_ngrams(text: str, n: int = 4) -> set[str]:
    words = normalized(text).split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


class Decontaminator:
    """Reject copied and near-copied user text without exporting test data."""

    def __init__(self, paths: Iterable[Path], threshold: float = 0.92) -> None:
        self.threshold = threshold
        texts: set[str] = set()
        for path in paths:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    for message in row.get("messages") or []:
                        if message.get("role") == "user" and isinstance(message.get("content"), str):
                            value = normalized(message["content"])
                            if value:
                                texts.add(value)
        self.texts = sorted(texts)
        self.exact = set(self.texts)
        self.grams = [word_ngrams(text) for text in self.texts]
        posting: dict[str, list[int]] = collections.defaultdict(list)
        for i, grams in enumerate(self.grams):
            for gram in grams:
                posting[gram].append(i)
        self.posting = posting
        self.checked = self.exact_hits = self.near_hits = 0

    def overlaps(self, text: str) -> bool:
        value = normalized(text)
        if not value:
            return False
        self.checked += 1
        if value in self.exact:
            self.exact_hits += 1
            return True
        words = value.split()
        if len(words) < 8:
            return False
        grams = word_ngrams(value)
        useful = sorted((g for g in grams if g in self.posting), key=lambda g: len(self.posting[g]))[:8]
        candidates: set[int] = set()
        for gram in useful:
            if len(self.posting[gram]) <= 100:
                candidates.update(self.posting[gram])
        for i in candidates:
            other_grams = self.grams[i]
            union = len(grams | other_grams)
            jaccard = len(grams & other_grams) / union if union else 0.0
            if jaccard >= self.threshold or difflib.SequenceMatcher(None, value, self.texts[i]).ratio() >= 0.95:
                self.near_hits += 1
                return True
        return False

    def row_overlaps(self, messages: list[dict[str, Any]]) -> bool:
        return any(
            self.overlaps(str(message.get("content", "")))
            for message in messages
            if message.get("role") == "user"
        )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tool_name(tool: dict[str, Any]) -> str:
    return str((tool.get("function") or tool).get("name", ""))


def as_openai_tool(tool: dict[str, Any]) -> dict[str, Any]:
    if tool.get("type") == "function" and isinstance(tool.get("function"), dict):
        return copy.deepcopy(tool)
    return {"type": "function", "function": copy.deepcopy(tool)}


def clean_call(call: dict[str, Any]) -> dict[str, Any] | None:
    function = call.get("function") if isinstance(call.get("function"), dict) else call
    name = function.get("name")
    if not isinstance(name, str) or not name or name == "think":
        return None
    arguments = function.get("arguments", function.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    else:
        parsed = arguments
    if not isinstance(parsed, dict):
        return None
    return {
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))},
    }


def clean_apigen_messages(messages: list[dict[str, Any]], defined: set[str]) -> list[dict[str, Any]]:
    """Remove source-specific system/think traces and normalize OpenAI calls."""
    out: list[dict[str, Any]] = []
    removed_ids: set[str] = set()
    for raw in messages:
        role = raw.get("role")
        if role == "system":
            continue
        if role == "assistant":
            calls = []
            for original in raw.get("tool_calls") or []:
                cleaned = clean_call(original)
                if cleaned and tool_name(cleaned) in defined:
                    calls.append(cleaned)
                elif original.get("id"):
                    removed_ids.add(str(original["id"]))
            content = raw.get("content")
            if not calls and not (isinstance(content, str) and content.strip()):
                continue
            message: dict[str, Any] = {"role": "assistant", "content": content if isinstance(content, str) else None}
            if calls:
                message["tool_calls"] = calls
            out.append(message)
        elif role == "tool":
            if raw.get("name") == "think" or str(raw.get("tool_call_id")) in removed_ids:
                continue
            if raw.get("name") not in defined:
                continue
            out.append({"role": "tool", "name": raw.get("name"), "content": str(raw.get("content", ""))})
        elif role in {"user"} and isinstance(raw.get("content"), str):
            out.append({"role": "user", "content": raw["content"]})
    return out


def extract_apigen_mt(path: Path, guard: Decontaminator, rng: random.Random) -> tuple[list[dict[str, Any]], dict[str, int]]:
    buckets: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    stats = collections.Counter()
    table = pq.read_table(path, columns=["messages", "tools"])
    for row_index, raw in enumerate(table.to_pylist()):
        tools = [as_openai_tool(tool) for tool in json.loads(raw["tools"]) if tool.get("name") != "think"]
        defined = {tool_name(tool) for tool in tools}
        messages = clean_apigen_messages(raw["messages"], defined)
        if guard.row_overlaps(messages):
            stats["discarded_public_overlap"] += 1
            continue
        for i, target in enumerate(messages):
            if target.get("role") != "assistant":
                continue
            calls = target.get("tool_calls") or []
            previous_role = messages[i - 1].get("role") if i else None
            if calls and previous_role == "tool":
                category = "must_call"
            elif calls:
                category = "eligible_call"
            elif previous_role == "tool":
                category = "stop_after_tool"
            else:
                category = "text_response"
            prefix = copy.deepcopy(messages[:i + 1])
            if not prefix or prefix[-1].get("role") != "assistant":
                continue
            buckets[category].append({
                "source": f"apigen_mt_{category}_v1",
                "source_group": f"apigen-mt-{row_index}",
                "loss_on_last_assistant_only": True,
                "tools": tools,
                "messages": prefix,
            })
    quotas = {"must_call": 18000, "eligible_call": 6500, "stop_after_tool": 3500, "text_response": 1200}
    selected: list[dict[str, Any]] = []
    for category, rows in buckets.items():
        rng.shuffle(rows)
        chosen = rows[:quotas[category]]
        selected.extend(chosen)
        stats[f"available_{category}"] = len(rows)
        stats[f"selected_{category}"] = len(chosen)
    return selected, dict(stats)


def extract_synth_apigen(path: Path, guard: Decontaminator, rng: random.Random, limit: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    raw_rows = pq.read_table(path, columns=["tools", "query", "answers", "hash_id"]).to_pylist()
    pool: dict[str, dict[str, Any]] = {}
    parsed: list[tuple[int, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]] = []
    stats = collections.Counter()
    for index, raw in enumerate(raw_rows):
        try:
            tools = [as_openai_tool(tool) for tool in json.loads(raw["tools"])]
            answers = json.loads(raw["answers"])
        except (json.JSONDecodeError, TypeError):
            stats["discarded_unparseable"] += 1
            continue
        if not tools or not isinstance(answers, list) or not answers:
            stats["discarded_no_call"] += 1
            continue
        defined = {tool_name(tool) for tool in tools}
        calls = [call for answer in answers if (call := clean_call(answer)) and tool_name(call) in defined]
        if not calls:
            stats["discarded_bad_call"] += 1
            continue
        if guard.overlaps(str(raw["query"])):
            stats["discarded_public_overlap"] += 1
            continue
        for tool in tools:
            pool.setdefault(tool_name(tool), tool)
        parsed.append((index, raw, tools, calls))
    rng.shuffle(parsed)
    pool_values = list(pool.values())
    rows: list[dict[str, Any]] = []
    for index, raw, tools, calls in parsed[:limit]:
        selected = {tool_name(tool): tool for tool in tools}
        desired = 12 if len(rows) % 2 else 18
        attempts = 0
        while len(selected) < desired and attempts < desired * 5:
            tool = rng.choice(pool_values)
            selected.setdefault(tool_name(tool), tool)
            attempts += 1
        tool_list = list(selected.values())
        rng.shuffle(tool_list)
        rows.append({
            "source": "argilla_synth_apigen_call_v1",
            "source_group": f"synth-apigen-{raw['hash_id'] or index}",
            "loss_on_last_assistant_only": True,
            "tools": tool_list,
            "messages": [
                {"role": "user", "content": str(raw["query"])},
                {"role": "assistant", "content": None, "tool_calls": calls},
            ],
        })
    stats["eligible"] = len(parsed)
    stats["selected"] = len(rows)
    stats["unique_tool_pool"] = len(pool)
    return rows, dict(stats)


def make_broad_multistep(single_rows: list[dict[str, Any]], rng: random.Random, count: int) -> list[dict[str, Any]]:
    """Turn independent public-training calls into original two-action continuations.

    Only the second call is supervised.  This makes the critical post-tool
    must-call signal cover thousands of schemas instead of the 129 BFCL docs.
    """
    eligible = [
        row for row in single_rows
        if len((row["messages"][-1].get("tool_calls") or [])) == 1
    ]
    if len(eligible) < 2:
        return []
    order = list(range(len(eligible)))
    rng.shuffle(order)
    rows: list[dict[str, Any]] = []
    for i in range(count):
        first = eligible[order[i % len(order)]]
        second = eligible[order[(i * 7919 + 1) % len(order)]]
        first_call = copy.deepcopy(first["messages"][-1])
        second_call = copy.deepcopy(second["messages"][-1])
        first_name = tool_name(first_call["tool_calls"][0])
        second_name = tool_name(second_call["tool_calls"][0])
        if first_name == second_name:
            second = eligible[order[(i * 7919 + 17) % len(order)]]
            second_call = copy.deepcopy(second["messages"][-1])
            second_name = tool_name(second_call["tool_calls"][0])
        combined = {tool_name(tool): copy.deepcopy(tool) for tool in first["tools"] + second["tools"]}
        required = {first_name, second_name}
        distractors = [tool for name, tool in combined.items() if name not in required]
        rng.shuffle(distractors)
        tool_list = [combined[name] for name in sorted(required) if name in combined] + distractors[:22]
        rng.shuffle(tool_list)
        first_query = str(first["messages"][0]["content"])
        second_query = str(second["messages"][0]["content"])
        rows.append({
            "source": "argilla_synth_apigen_multistep_v1",
            "source_group": f"synth-multistep-{i}",
            "loss_on_last_assistant_only": True,
            "tools": tool_list,
            "messages": [
                {"role": "user", "content": (
                    "Complete both independent tasks in order. First task: " + first_query
                    + "\nAfter it succeeds, continue with the second task: " + second_query
                )},
                first_call,
                {"role": "tool", "name": first_name, "content": json.dumps({
                    "status": "success", "completed_tool": first_name,
                }, separators=(",", ":"))},
                second_call,
            ],
        })
    return rows


def emit_turn(rows: list[dict[str, Any]], history: list[dict[str, Any]], target: dict[str, Any],
              group: str, source: str) -> None:
    history.append(copy.deepcopy(target))
    rows.append({
        "source": source,
        "source_group": group,
        "loss_on_last_assistant_only": True,
        "tools": copy.deepcopy(NEGOTIATION_TOOLS),
        "messages": copy.deepcopy(history),
    })


def make_negotiation_curriculum(rng: random.Random, scenarios: int) -> list[dict[str, Any]]:
    """Create many high-offer negatives and one exact low-price order per scenario."""
    rows: list[dict[str, Any]] = []
    for index in range(scenarios):
        pid, item = rng.choice(PRODUCTS)
        qty = rng.choice([1, 1, 2, 3])
        target = round(rng.uniform(4.0, 45.0), 2)
        budget = round(target * qty * rng.uniform(1.30, 1.55), 2)
        system = buyer_system(pid, item, qty, budget, target) + (
            "\nContinue bargaining while a current valid offer is above the private target and turns remain. "
            "A merely budget-valid price is not automatically a good deal. Never order an offer for the wrong product or quantity."
        )
        group = f"negotiation-corrective-{index}"
        history: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"phase": "negotiation", "turns_remaining_in_phase": 16})},
        ]
        opening = round(target * rng.uniform(0.72, 0.84), 2)
        emit_turn(
            rows, history,
            call_message("wait", {}, f"I'm interested in {pid}. Could you do ${opening:.2f} per unit for {qty}?"),
            group, "negotiation_wait_before_offer_v2",
        )

        # A malformed or mismatched offer must never be ordered.
        if index % 3 == 0:
            wrong_qty = qty + 1
            mid = f"m-{index}-wrong"
            history.append({"role": "user", "content": seller_message(
                mid, pid, item, round(target * 0.92, 2), wrong_qty, 15,
                "This quantity is what I can offer right now.",
            )})
            emit_turn(
                rows, history,
                call_message("wait", {}, f"I need exactly {qty}, not {wrong_qty}. Please quote the requested quantity."),
                group, "negotiation_wait_wrong_terms_v2",
            )
        elif index % 3 == 1:
            history.append({"role": "user", "content": (
                json.dumps({"from_id": "seller", "message_id": f"m-{index}-chat"})
                + "\nI may be able to make an offer, but I have not quoted formal terms yet."
            )})
            emit_turn(
                rows, history,
                call_message("wait", {}, "Please send a formal offer with the product, unit price, and quantity."),
                group, "negotiation_wait_no_offer_v2",
            )

        # Two or three valid but intentionally expensive standing offers.
        high_count = 2 + (index % 2)
        for turn in range(high_count):
            multiplier = rng.uniform(1.10, 1.28) - 0.035 * turn
            price = round(min(budget / qty, target * multiplier), 2)
            mid = f"m-{index}-high-{turn}"
            history.append({"role": "user", "content": seller_message(
                mid, pid, item, price, qty, 14 - turn,
                "This replaces my previous offer and is my current price.",
            )})
            counter = round(target * rng.uniform(0.84, 0.96), 2)
            emit_turn(
                rows, history,
                call_message("wait", {}, f"Thanks, but ${price:.2f} is above my goal. Could you do ${counter:.2f} for {qty}?"),
                group, "negotiation_wait_high_offer_v2",
            )

        deal = round(target * rng.uniform(0.86, 0.99), 2)
        final_id = f"m-{index}-deal"
        quote = f"Offer: {pid} at ${deal:.2f} per unit, qty {qty}."
        history.append({"role": "user", "content": seller_message(
            final_id, pid, item, deal, qty, 11,
            "This is a new formal offer and replaces all earlier terms.",
        )})
        emit_turn(
            rows, history,
            call_message("order", {
                "product_id": pid,
                "agree_price": deal,
                "qty": qty,
                "evidence_message_ids": [final_id],
                "evidence_quote": quote,
            }, "Agreed. I accept this latest offer."),
            group, "negotiation_exact_low_order_v2",
        )
    return rows


def load_retention(path: Path, rng: random.Random, limit: int, source_prefix: str | None = None) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if source_prefix and not str(row.get("source", "")).startswith(source_prefix):
                continue
            row = copy.deepcopy(row)
            row["source"] = "retention_" + str(row.get("source", "unknown"))
            row["source_group"] = f"retention-{path.parent.name}-{index}"
            rows.append(row)
    rng.shuffle(rows)
    return rows[:limit]


def split_rows(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train, validation = [], []
    for row in rows:
        group = str(row.get("source_group", ""))
        value = int(hashlib.sha256((str(seed) + ":" + group).encode()).hexdigest()[:8], 16)
        (validation if value % 50 == 0 else train).append(row)
    random.Random(seed + 1).shuffle(train)
    random.Random(seed + 2).shuffle(validation)
    return train, validation


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apigen-mt", type=Path, default=ROOT / "external_data/apigen_mt_parsed/apigen-mt-5k.parquet")
    parser.add_argument("--synth-apigen", type=Path, default=ROOT / "external_data/synth_apigen/data/train-00000-of-00001.parquet")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "training_data_corrective_v4")
    parser.add_argument("--single-turn", type=int, default=14000)
    parser.add_argument("--broad-multistep", type=int, default=20000)
    parser.add_argument("--negotiations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261022)
    args = parser.parse_args()
    for path in (args.apigen_mt, args.synth_apigen):
        if "public_test" in str(path.resolve()).lower():
            raise SystemExit(f"Refusing test data as a training source: {path}")
        if not path.is_file():
            raise SystemExit(f"Missing public training source: {path}")

    rng = random.Random(args.seed)
    guard = Decontaminator(PUBLIC_FILES)
    mt_rows, mt_stats = extract_apigen_mt(args.apigen_mt, guard, rng)
    single_rows, single_stats = extract_synth_apigen(args.synth_apigen, guard, rng, args.single_turn)
    broad_multistep_rows = make_broad_multistep(single_rows, rng, args.broad_multistep)
    negotiation_rows = make_negotiation_curriculum(rng, args.negotiations)
    retention_rows = []
    retention_rows += load_retention(ROOT / "training_data_dependent/train.jsonl", rng, 3200, "bfcl_docs_dependency")
    retention_rows += load_retention(ROOT / "training_data_bfcl_synthetic_v3/train.jsonl", rng, 1800)
    retention_rows += load_retention(ROOT / "training_data_v2/train.jsonl", rng, 1800, "original_synthetic_tools")

    rows = mt_rows + single_rows + broad_multistep_rows + negotiation_rows + retention_rows
    train, validation = split_rows(rows, args.seed)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "validation.jsonl", validation)
    sources = collections.Counter(row["source"] for row in rows)
    manifest = {
        "generator": "code/prepare_corrective_data.py",
        "seed": args.seed,
        "policy": "PUBLIC is used only as a discard-only exact/near-duplicate guard; no PUBLIC content is written",
        "public_user_texts_held_out": len(guard.exact),
        "decontamination": {
            "threshold_4gram_jaccard": guard.threshold,
            "sequence_match_threshold": 0.95,
            "texts_checked": guard.checked,
            "exact_overlaps_discarded": guard.exact_hits,
            "near_overlaps_discarded": guard.near_hits,
        },
        "sources": [
            {
                "name": "Salesforce APIGen-MT-5k (parsed mirror)",
                "url": "https://huggingface.co/datasets/Salesforce/APIGen-MT-5k",
                "mirror_url": "https://huggingface.co/datasets/minpeter/apigen-mt-5k-parsed",
                "license": "CC-BY-NC-4.0; used only for this non-commercial academic homework",
                "local_file": str(args.apigen_mt.relative_to(ROOT)),
                "sha256": sha256(args.apigen_mt),
                "scope": "public multi-turn training trajectories; source-specific think traces removed",
            },
            {
                "name": "Argilla Synth-APIGen-v0.1",
                "url": "https://huggingface.co/datasets/argilla/Synth-APIGen-v0.1",
                "license": "Apache-2.0",
                "local_file": str(args.synth_apigen.relative_to(ROOT)),
                "sha256": sha256(args.synth_apigen),
                "scope": "independent synthetic training split only",
            },
            {
                "name": "Original corrective negotiation synthesis and prior compliant curricula",
                "scope": "generated locally; excludes prior negotiation rows with the weaker early-order policy",
            },
        ],
        "apigen_mt": mt_stats,
        "synth_apigen": single_stats,
        "broad_multistep_supervised_steps": len(broad_multistep_rows),
        "negotiation_scenarios": args.negotiations,
        "negotiation_supervised_steps": len(negotiation_rows),
        "retention_rows": len(retention_rows),
        "all_rows": len(rows),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "rows_by_source": dict(sorted(sources.items())),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
