#!/usr/bin/env python3
"""Report conservative exact-gold retrieval metrics without running the LLM judge."""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogues", default="data/public/dialogues.jsonl")
    parser.add_argument("--retrieval-results", default="results/retrieval_results.json")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.dialogues, encoding="utf-8") as handle:
        gold = {
            row["dialogue_id"]: row["target"]
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
        }
    with open(args.retrieval_results, encoding="utf-8") as handle:
        results = json.load(handle)["results"]
    ranks = []
    for row in results:
        target = gold[row["qid"]]
        pids = row["ranked_pids"][: args.top_k]
        ranks.append(
            pids.index(target) + 1
            if target in pids
            else 0
        )
    count = len(ranks)
    print(f"queries: {count}")
    print(f"exact Recall@{args.top_k}: {sum(rank > 0 for rank in ranks) / count:.6f}")
    print(f"exact MRR@{args.top_k}: {sum(1 / rank for rank in ranks if rank) / count:.6f}")


if __name__ == "__main__":
    main()
