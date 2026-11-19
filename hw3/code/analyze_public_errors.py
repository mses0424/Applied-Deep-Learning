#!/usr/bin/env python3
"""Aggregate official HW3 public-evaluation errors without exporting examples.

This is an evaluation-only diagnostic.  It never writes prompts, gold calls,
arguments, or model replies, so its output cannot be used as training data.
Part 1 decisions reuse eval.score_tools; Part 2 decisions reuse
eval.score_negotiation.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import format as F
from eval import score_negotiation as SN
from eval import score_tools as ST


def ratio(n: int, d: int) -> float | None:
    return n / d if d else None


def tool_bucket(n: int) -> str:
    if n <= 10:
        return "01-10"
    if n <= 20:
        return "11-20"
    if n <= 40:
        return "21-40"
    return "41+"


def history_bucket(messages: list[dict[str, Any]]) -> str:
    calls = sum(bool(m.get("tool_calls")) for m in messages if m.get("role") == "assistant")
    if calls == 0:
        return "0_previous_calls"
    if calls == 1:
        return "1_previous_call"
    if calls <= 3:
        return "2-3_previous_calls"
    return "4+_previous_calls"


def domain_maps(doc_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for path in sorted(doc_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row.get("name"), str):
                        out.setdefault(row["name"], path.stem)
    return out


def target_domain(item: dict[str, Any], target_name: str, fn_domain: dict[str, str]) -> str:
    if target_name == "no_call":
        return "no_call"
    if target_name in fn_domain:
        return fn_domain[target_name]
    description = ""
    for tool in item.get("tools") or []:
        fn = tool.get("function") or {}
        if fn.get("name") == target_name:
            description = str(fn.get("description", "")).lower()
            break
    for needle, domain in (
        ("file system", "gorilla_file_system"),
        ("math", "math_api"),
        ("message api", "message_api"),
        ("posting", "posting_api"),
        ("ticket", "ticket_api"),
        ("trading", "trading_bot"),
        ("travel", "travel_booking"),
        ("vehicle", "vehicle_control"),
    ):
        if needle in description:
            return domain
    return "unmapped_call"


def aggregate_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"n": 0}
    weight = F.settings()["must_call_weight"]
    den = sum(weight if r["must_call"] else 1 for r in records)
    return {
        "n": len(records),
        "must_call_n": sum(r["must_call"] for r in records),
        "official_weighted_step_acc": sum((weight if r["must_call"] else 1) * r["official_ok"] for r in records) / den,
        "exact_step_acc": sum(r["exact_ok"] for r in records) / len(records),
        "tool_name_acc": sum(r["tool_name_ok"] for r in records) / len(records),
    }


def grouped(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    names = sorted({str(r[key]) for r in records})
    return {name: aggregate_group([r for r in records if str(r[key]) == name]) for name in names}


def analyze_tools(data: Path, pred: Path, docs: Path) -> dict[str, Any]:
    gold = F.read_jsonl(data)
    preds, duplicates = F.load_predictions(str(pred))
    fn_domain = domain_maps(docs)
    records: list[dict[str, Any]] = []
    categories = collections.Counter()
    for item in gold:
        reply = F.read_reply(preds.get(item["id"]), "lenient")
        calls = reply["calls"] if reply and not reply["invalid"] else []
        is_call = bool(item["acceptable_next"])
        eligible = bool(item.get("first_step_eligible"))
        must_call = is_call and not eligible
        replied = ST.replied_text(reply)
        exact = bool(reply and not reply["invalid"] and ST.first_ok(item, calls))
        official = exact or bool(eligible and replied)
        acceptable_names = {c["name"] for c in item["acceptable_next"]}
        name_ok = (bool(calls) and calls[0]["name"] in acceptable_names) if is_call else bool(reply and not reply["invalid"] and not calls)

        if reply is None:
            category = "missing"
        elif reply["invalid"]:
            category = "invalid"
        elif not is_call and calls:
            category = "no_call_false_positive"
        elif not is_call:
            category = "no_call_correct"
        elif eligible and replied:
            category = "eligible_text_credit"
        elif not calls:
            category = "missed_required_call"
        elif calls[0]["name"] not in acceptable_names:
            category = "wrong_tool_name"
        elif exact:
            category = "exact_call"
        else:
            category = "right_tool_wrong_arguments"
        categories[category] += 1

        target_name = next(iter(acceptable_names), "no_call")
        records.append({
            "official_ok": official,
            "exact_ok": exact,
            "tool_name_ok": name_ok,
            "must_call": must_call,
            "slice": "must_call" if must_call else "eligible_call" if is_call else "no_call",
            "category": category,
            "domain": target_domain(item, target_name, fn_domain),
            "tool_count": tool_bucket(len(item.get("tools") or [])),
            "history": history_bucket(item.get("messages") or []),
        })

    return {
        "data": str(data),
        "prediction": str(pred),
        "n": len(records),
        "duplicate_predictions": duplicates,
        "overall": aggregate_group(records),
        "categories": {k: {"n": v, "fraction": v / len(records)} for k, v in sorted(categories.items())},
        "by_slice": grouped(records, "slice"),
        "by_slice_and_category": {
            name: dict(sorted(collections.Counter(r["category"] for r in records if r["slice"] == name).items()))
            for name in ("must_call", "eligible_call", "no_call")
        },
        "by_domain": grouped(records, "domain"),
        "by_tool_count": grouped(records, "tool_count"),
        "by_history": grouped(records, "history"),
    }


def order_failure(meta: dict[str, Any], args: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if args.get("product_id") not in (meta.get("acceptable_product_ids") or []):
        failures.append("unacceptable_product")
    if SN.cents(args.get("agree_price")) is None:
        failures.append("bad_price")
    if SN.qty_of(args.get("qty")) is None:
        failures.append("bad_quantity")
    if meta.get("budget") is not None and not SN.budget_ok(args, meta["budget"]):
        failures.append("over_budget")
    valid = meta.get("valid_order")
    if not valid:
        failures.append("no_current_valid_offer")
    elif not SN._terms_eq(args, valid):
        failures.append("terms_not_latest_offer")
    standing = meta.get("standing_offers") or {}
    if not any(mid in standing and SN._terms_eq(args, standing[mid]) for mid in SN.cited_ids(args)):
        failures.append("missing_or_wrong_citation")
    return failures


def analyze_negotiation(data: Path, pred: Path) -> dict[str, Any]:
    gold = F.read_jsonl(data)
    preds, duplicates = F.load_predictions(str(pred))
    failures = collections.Counter()
    confusion = collections.Counter()
    shapes = collections.Counter()
    orders = valid_orders = cited_orders = 0
    by_neg: dict[str, dict[str, bool]] = collections.defaultdict(lambda: {"any": False, "valid": False})
    rel: dict[str, float] = {}
    for item in gold:
        by_neg[item["negotiation"]]
        reply = SN.reply_of(preds.get(item["id"]), "lenient")
        call = SN.effective_call(reply)
        pred_name = call["name"] if call else "none"
        confusion[f"{item.get('gold_call', 'none')}->{pred_name}"] += 1
        shapes[f"{item.get('kind', 'unknown')}:{SN.shape_of(reply)}"] += 1
        if call and call["name"] == "order":
            orders += 1
            negotiation = item["negotiation"]
            by_neg[negotiation]["any"] = True
            args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            ok = SN.order_ok_hw(item["step_meta"], args, require_citation=False)
            cited = SN.order_ok_hw(item["step_meta"], args, require_citation=True)
            valid_orders += bool(ok)
            cited_orders += bool(cited)
            if ok:
                by_neg[negotiation]["valid"] = True
                rel.setdefault(negotiation, float(item["step_meta"]["valid_order"]["agree_price"]) / item["gold_price"])
            for reason in order_failure(item["step_meta"], args):
                failures[reason] += 1
    rel_values = sorted(rel.values())
    return {
        "data": str(data),
        "prediction": str(pred),
        "steps": len(gold),
        "negotiations": len(by_neg),
        "duplicate_predictions": duplicates,
        "orders": orders,
        "valid_orders": valid_orders,
        "cited_valid_orders": cited_orders,
        "order_accuracy": ratio(valid_orders, orders),
        "cited_order_accuracy": ratio(cited_orders, orders),
        "negotiations_with_any_order": sum(v["any"] for v in by_neg.values()),
        "negotiations_with_valid_order": sum(v["valid"] for v in by_neg.values()),
        "order_failure_reasons_nonexclusive": dict(sorted(failures.items())),
        "gold_to_pred_first_call": dict(sorted(confusion.items())),
        "kind_to_reply_shape": dict(sorted(shapes.items())),
        "relative_price": {
            "n": len(rel_values),
            "mean": sum(rel_values) / len(rel_values) if rel_values else None,
            "below_1": sum(x < 1 for x in rel_values),
            "equal_1": sum(math.isclose(x, 1.0) for x in rel_values),
            "above_1": sum(x > 1 for x in rel_values),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools-data", type=Path, default=Path("data/public_test.jsonl.gz"))
    parser.add_argument("--tools-pred", type=Path, required=True)
    parser.add_argument("--negotiation-data", type=Path, default=Path("data/public_test_negotiation.jsonl.gz"))
    parser.add_argument("--negotiation-pred", type=Path, required=True)
    parser.add_argument("--docs", type=Path, default=Path("external_data/bfcl_docs/multi_turn_func_doc"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "note": "Aggregate evaluation diagnostics only; no prompt, gold call, argument, or reply is exported.",
        "tools": analyze_tools(args.tools_data, args.tools_pred, args.docs),
        "negotiation": analyze_negotiation(args.negotiation_data, args.negotiation_pred),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
