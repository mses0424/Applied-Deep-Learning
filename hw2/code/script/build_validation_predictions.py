#!/usr/bin/env python3
"""Package a recorded train-validation ranking for the unchanged evaluator."""

import argparse

from common import ROOT, read_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    rows = read_jsonl(ROOT / "prepared/validation.jsonl")
    rankings = read_json(ROOT / args.rankings)
    expected = {row["dialogue_id"] for row in rows}
    if set(rankings) != expected:
        raise ValueError("Ranking qids do not exactly match the frozen validation split")
    results = []
    for row in rows:
        qid = row["dialogue_id"]
        pids = rankings[qid][: args.top_k]
        if len(pids) != args.top_k or len(set(pids)) != args.top_k:
            raise ValueError(f"{qid}: invalid top-{args.top_k} ranking")
        results.append({"qid": qid, "ranked_pids": pids})
    write_json(ROOT / args.output, {"top_k": args.top_k, "results": results})
    print(f"Wrote {len(results)} frozen validation predictions -> {args.output}")


if __name__ == "__main__":
    main()
