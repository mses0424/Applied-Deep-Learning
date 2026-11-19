#!/usr/bin/env python3
"""Create new requests from BFCL function *documentation only*.

This script intentionally accepts only ``multi_turn_func_doc/*.json`` JSONL
files.  It never reads BFCL questions, ground truth, possible answers, model
outputs, or the homework public test.  A small local instruction model merely
paraphrases an authored tool description plus newly sampled arguments into a
natural request; the exact call target remains deterministic code.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


def patch_torchvision() -> None:
    from train import patch_obsolete_torchvision
    patch_obsolete_torchvision()


def normalize_schema(obj: Any) -> Any:
    if isinstance(obj, list):
        return [normalize_schema(x) for x in obj]
    if not isinstance(obj, dict):
        return obj
    out = {k: normalize_schema(v) for k, v in obj.items()}
    mapping = {"dict": "object", "float": "number", "tuple": "array", "list": "array"}
    if out.get("type") in mapping:
        out["type"] = mapping[out["type"]]
    if out.get("type") == "array" and "items" not in out:
        out["items"] = {"type": "string"}
    return out


def enum_values(prop: dict[str, Any]) -> list[Any]:
    if isinstance(prop.get("enum"), list):
        return prop["enum"]
    match = re.search(r"\[Enum\]:\s*(\[[^\n]+\])", str(prop.get("description", "")))
    if match:
        try:
            value = json.loads(match.group(1))
            return value if isinstance(value, list) else []
        except json.JSONDecodeError:
            return []
    description = str(prop.get("description", ""))
    match = re.search(r"\[Enum\]:\s*([^.;\n]+)", description, flags=re.I)
    if match:
        values = [value.strip(" `\"'") for value in match.group(1).split(",")]
        if len(values) > 1:
            return values
    match = re.search(r"\(([^()]+/[^()]+)\)", description)
    if match:
        values = [value.strip(" `\"'") for value in match.group(1).split("/")]
        if all(values):
            return values
    match = re.search(r"\b(deposit|buy)\s+or\s+(withdrawal|sell)\b", description, flags=re.I)
    if match:
        return [match.group(1), match.group(2)]
    return []


def string_value(name: str, prop: dict[str, Any], rng: random.Random, i: int) -> str:
    enums = enum_values(prop)
    if enums:
        return str(rng.choice(enums))
    low = name.lower()
    choices = {
        "city": ["Taipei", "Kyoto", "Berlin", "Toronto"], "username": ["alex_chen", "maya7", "river99"],
        "password": ["Cedar!47", "BlueSky#82", "River$19"], "email": ["alex@example.org", "maya@example.org"],
        "symbol": ["ACME", "NOVA", "ORBT"], "stock": ["ACME", "NOVA", "ORBT"],
        "date": ["2027-02-12", "2027-04-08", "2027-06-21"], "time": ["09:30", "14:00", "17:45"],
        "path": ["/workspace/notes.txt", "/tmp/report.csv", "/projects/demo/data.json"],
        "file": ["notes.txt", "report.csv", "data.json"], "directory": ["/workspace", "/tmp/reports"],
        "phone": ["+1-555-0102", "+1-555-0178"], "token": ["tok_A17", "tok_B42"],
        "id": [f"id_{i:04d}", f"item_{i:04d}"], "message": ["The review is complete.", "Please call me tomorrow."],
        "content": ["Quarterly update", "Meeting notes are ready."], "title": ["Login issue", "Project update"],
        "name": ["Alex Chen", "Maya Lin", "Jordan Lee"], "currency": ["USD", "EUR", "TWD"],
        "order_type": ["Buy", "Sell"], "travel_class": ["economy", "business", "first"],
        "sector": ["technology", "healthcare", "energy"], "status": ["open", "closed", "pending"],
        "resolution": ["Issue resolved after configuration update", "Replacement approved"],
        "passport": ["P4820193", "P7301946"], "airport": ["TPE", "NRT", "SFO"],
    }
    for key, values in choices.items():
        if key in low:
            return rng.choice(values)
    return f"sample_{name}_{i}"


def sample_value(name: str, prop: dict[str, Any], rng: random.Random, i: int) -> Any:
    enums = enum_values(prop)
    if enums:
        return rng.choice(enums)
    typ = normalize_schema(prop).get("type", "string")
    if typ == "string":
        return string_value(name, prop, rng, i)
    if typ == "integer":
        low = name.lower()
        return rng.choice([1, 2, 3]) if any(x in low for x in ["qty", "quantity", "count"]) else rng.randint(2, 90)
    if typ == "number":
        return round(rng.uniform(2.5, 95.0), 2)
    if typ == "boolean":
        return bool(rng.randrange(2))
    if typ == "array":
        item = prop.get("items", {"type": "string"})
        return [sample_value(name.rstrip("s") or "item", item, rng, i), sample_value(name.rstrip("s") or "item", item, rng, i + 1)]
    if typ == "object":
        return {k: sample_value(k, v, rng, i) for k, v in list(prop.get("properties", {}).items())[:3]}
    return string_value(name, prop, rng, i)


def sample_arguments(doc: dict[str, Any], rng: random.Random, i: int) -> dict[str, Any]:
    params = doc.get("parameters") or {}
    props = params.get("properties") or {}
    required = set(params.get("required") or [])
    args = {}
    for name, prop in props.items():
        if name in required or rng.random() < 0.30:
            args[name] = sample_value(name, prop, rng, i)
    return args


def openai_tool(doc: dict[str, Any]) -> dict[str, Any]:
    return {"type": "function", "function": {"name": doc["name"], "description": doc.get("description", ""),
                                                "parameters": normalize_schema(doc.get("parameters") or {"type": "object", "properties": {}})}}


def prompt_for(doc: dict[str, Any], args: dict[str, Any]) -> str:
    return (
        "Write one natural user request that unambiguously asks for the action below and explicitly supplies every value in ARGS. "
        "Do not mention a function name, API, JSON, parameter names, or that tools exist. Do not invent extra facts. "
        "Output only the request in one or two sentences.\n\nACTION DESCRIPTION:\n" + doc.get("description", "") +
        "\n\nARGS:\n" + json.dumps(args, ensure_ascii=False)
    )


def teacher_requests(items: list[tuple[dict[str, Any], dict[str, Any]]], model_name: str, batch_size: int) -> list[str]:
    patch_torchvision()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda().eval()
    prompts = [tok.apply_chat_template([{"role": "system", "content": "You write concise, realistic user requests."},
                                        {"role": "user", "content": prompt_for(doc, args)}],
                                       tokenize=False, add_generation_prompt=True) for doc, args in items]
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start:start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            generated = model.generate(**enc, do_sample=False, max_new_tokens=96, eos_token_id=tok.eos_token_id,
                                       pad_token_id=tok.pad_token_id)
        for row in generated[:, enc["input_ids"].shape[1]:]:
            text = tok.decode(row, skip_special_tokens=True).strip().replace("\n", " ")
            text = re.sub(r"^(Request|User request)\s*:\s*", "", text, flags=re.I).strip(' "')
            outputs.append(text)
        print(f"teacher generated {min(start + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    del model
    torch.cuda.empty_cache()
    return outputs


def call_message(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {
        "name": name, "arguments": json.dumps(args, ensure_ascii=False)}}]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=Path, default=Path("external_data/bfcl_docs/multi_turn_func_doc"))
    ap.add_argument("--out-dir", type=Path, default=Path("training_data_bfcl_synthetic"))
    ap.add_argument("--teacher", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--variants", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()
    if "multi_turn_func_doc" not in str(args.docs):
        raise SystemExit("--docs must point to BFCL multi_turn_func_doc only")
    paths = sorted(Path(p) for p in glob.glob(str(args.docs / "*.json")))
    if not paths:
        raise SystemExit(f"No function-doc JSONL files in {args.docs}")
    rng = random.Random(args.seed)
    domains: list[tuple[str, list[dict[str, Any]]]] = []
    for path in paths:
        docs = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
        if not all(set(d) == {"name", "description", "parameters", "response"} for d in docs):
            raise SystemExit(f"Unexpected non-doc record in {path}; refusing input")
        domains.append((path.stem, docs))
    items = [(doc, sample_arguments(doc, rng, variant * 1000 + j))
             for _, docs in domains for j, doc in enumerate(docs) for variant in range(args.variants)]
    requests = teacher_requests(items, args.teacher, args.batch_size)
    req_iter = iter(requests)
    rows = []
    for domain, docs in domains:
        for j, doc in enumerate(docs):
            for variant in range(args.variants):
                args_obj = items.pop(0)[1]
                request = next(req_iter)
                if not request:
                    request = f"Please perform this action: {doc.get('description', '')} Values: {json.dumps(args_obj)}"
                distractor_docs = [other for other in docs if other["name"] != doc["name"]]
                selected_docs = [doc] + rng.sample(distractor_docs, min(9, len(distractor_docs)))
                rng.shuffle(selected_docs)
                tools = [openai_tool(selected) for selected in selected_docs]
                messages = [{"role": "user", "content": request}, call_message(doc["name"], args_obj)]
                if variant % 3 == 0:
                    result = {"status": "success", "result_id": f"synthetic-{domain}-{j}-{variant}"}
                    messages += [{"role": "tool", "name": doc["name"], "content": json.dumps(result)},
                                 {"role": "assistant", "content": "Done — the requested action completed successfully."}]
                rows.append({"source": "bfcl_docs_original_requests_v1", "domain": domain, "tools": tools, "messages": messages})
    rng.shuffle(rows)
    val_n = max(1, round(len(rows) * 0.03))
    validation, base_train = rows[:val_n], rows[val_n:]
    train = [row for row in base_train for _ in range(args.repeat)]
    rng.shuffle(train)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in [("train.jsonl", train), ("validation.jsonl", validation)]:
        with (args.out_dir / name).open("w", encoding="utf-8") as f:
            for row in data:
                f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {"source_repo": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
                "source_scope": "multi_turn_func_doc/*.json only; no questions or answers",
                "doc_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "teacher": args.teacher, "seed": args.seed, "functions": sum(len(d) for _, d in domains),
                "variants_per_function": args.variants, "repeat_train": args.repeat,
                "unique_conversations": len(rows), "train_rows_after_repeat": len(train), "validation_rows": len(validation)}
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
