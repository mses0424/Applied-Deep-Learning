#!/usr/bin/env python3
"""Run the HW2 three-prompt generation comparison on five public examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SAMPLE_QIDS = [
    "public-test-beauty-000001",
    "public-test-beauty-000002",
    "public-test-beauty-000004",
    "public-test-beauty-000009",
    "public-test-beauty-000016",
]

PROMPTS = {
    "P1_direct": """You are a helpful sales assistant. Based on the conversation and product
specification below, recommend the product in one paragraph of at most 55 words.
Explain why it fits. Do not mention information that is not in the supplied text.

[CONVERSATION]\n{conversation}\n\n[PRODUCT]\n{product}\n\n[RESPONSE]""",
    "P2_constraint_grounded": """You are a careful product recommender. Identify the customer's
firm requirements, silently verify each one against the product specification, then write a
recommendation of at most 70 words. Mention only supported features. If a requested feature is
not stated in the specification, say that it should be verified rather than inventing it.

[CONVERSATION]\n{conversation}\n\n[PRODUCT SPECIFICATION]\n{product}\n\n[RESPONSE]""",
    "P3_consultative": """Act as a friendly store specialist. Recommend the supplied product for
this customer. Write exactly three compact bullet points and at most 75 words total: two points
that connect product evidence to needs, then one caveat/check. Never claim an absent feature.

[CUSTOMER DIALOGUE]\n{conversation}\n\n[RETRIEVED PRODUCT]\n{product}\n\n[ANSWER]""",
}


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def conversation_text(row: dict) -> str:
    return "\n".join(
        f"{turn['role'].title()}: {turn['content']}" for turn in row["conversation"]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogues", default="data/public/dialogues.jsonl")
    parser.add_argument("--products", default="data/public/products.jsonl")
    parser.add_argument("--retrieval-results", default="results/retrieval_results.json")
    parser.add_argument("--model", default="models/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output", default="results/generation_analysis.json")
    parser.add_argument("--max-new-tokens", type=int, default=112)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dialogues = {row["dialogue_id"]: row for row in read_jsonl(args.dialogues)}
    products = {row["parent_asin"]: row for row in read_jsonl(args.products)}
    with open(args.retrieval_results, encoding="utf-8") as handle:
        ranked = {
            row["qid"]: row["ranked_pids"] for row in json.load(handle)["results"]
        }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.eval()

    records = []
    for qid in SAMPLE_QIDS:
        dialogue = dialogues[qid]
        pid = ranked[qid][0]
        product_row = products[pid]
        product = "\n".join(
            value for value in (product_row.get("title"), product_row.get("description"))
            if value
        )
        conversation = conversation_text(dialogue)
        records.append(
            {
                "qid": qid,
                "retrieved_pid": pid,
                "gold_pid": dialogue["target"],
                "retrieval_is_gold": pid == dialogue["target"],
                "conversation": conversation,
                "product": product,
                "generations": {},
            }
        )

    # Decode one five-example batch per prompt. This is substantially faster
    # than 15 independent calls while preserving deterministic greedy output.
    for prompt_name, template in PROMPTS.items():
        rendered_prompts = []
        for record in records:
            user_text = template.format(
                conversation=record["conversation"],
                product=record["product"][:1800],
            )
            rendered_prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_text}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        inputs = tokenizer(rendered_prompts, padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for record, text in zip(records, decoded):
            record["generations"][prompt_name] = text.strip()
        print(f"Completed {prompt_name}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"model": args.model, "decoding": "greedy", "samples": records},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(records)} samples x {len(PROMPTS)} prompts -> {output}")


if __name__ == "__main__":
    main()
