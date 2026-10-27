#!/usr/bin/env python3
"""Select the verifier/cross-rank blend on frozen train-validation exact labels."""

import argparse
import math

from common import ROOT, exact_metrics, read_json, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--evaluation", help="Optional unchanged-judge results for the same candidates")
    parser.add_argument("--output", default="results/v5/verifier_weight_study.json")
    parser.add_argument("--selected-config", default="configs/v5/verifier_selected.json")
    args = parser.parse_args()
    rows = read_jsonl(ROOT / "prepared/validation.jsonl")
    payload = read_json(ROOT / args.predictions)
    original = {row["qid"]: row["ranked_pids"] for row in payload["results"]}
    diagnostics = read_json(ROOT / args.diagnostics)
    judged = None
    if args.evaluation:
        evaluation = read_json(ROOT / args.evaluation)
        judged = {record["qid"]: {item["pid"]: float(item["gain"])
                                  for item in record["candidates"]}
                  for record in evaluation["records"]}
    weights = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    study = {}
    for weight in weights:
        rankings = {}
        for row in rows:
            qid = row["dialogue_id"]
            scored = diagnostics["scores"][qid]
            ordered = sorted(scored, key=lambda item: (
                -(float(item["logit_margin"]) - weight * math.log(int(item["cross_rank"]))),
                int(item["cross_rank"]), str(item["pid"])))
            prefix = [str(item["pid"]) for item in ordered]
            selected = set(prefix)
            rankings[qid] = prefix + [pid for pid in original[qid] if pid not in selected]
        record = {"exact": exact_metrics(rows, rankings)}
        if judged is not None:
            recalls, reciprocal = [], []
            for row in rows:
                qid = row["dialogue_id"]
                top = rankings[qid][:10]
                if any(pid not in judged[qid] for pid in top):
                    raise ValueError("Cannot reuse judge results after changing the candidate set")
                gains = [judged[qid][pid] for pid in top]
                recalls.append(max(gains, default=0.0))
                reciprocal.append(max((gain / rank for rank, gain in enumerate(gains, 1)), default=0.0))
            record["graded"] = {"queries": len(rows), "graded_recall@10": sum(recalls) / len(rows),
                                "graded_mrr@10": sum(reciprocal) / len(rows)}
        study[str(weight)] = record
    if judged is not None:
        winner = max(study, key=lambda key: (study[key]["graded"]["graded_mrr@10"],
                                              study[key]["graded"]["graded_recall@10"], -float(key)))
        criterion = "train-validation official graded MRR, Recall, then smaller prior"
    else:
        winner = max(study, key=lambda key: (study[key]["exact"]["exact_mrr@10"],
                                              study[key]["exact"]["exact_recall@10"], -float(key)))
        criterion = "train-validation exact MRR, Recall, then smaller prior"
    cfg = read_json(ROOT / "configs/v5/verifier.json")
    cfg["rank_prior_weight"] = float(winner)
    write_json(ROOT / args.output, {"source": "frozen train-validation only",
                                    "criterion": criterion, "weights": study, "selected": winner})
    write_json(ROOT / args.selected_config, cfg)
    print(f"Selected rank_prior_weight={winner}: {study[winner]}")


if __name__ == "__main__":
    main()
