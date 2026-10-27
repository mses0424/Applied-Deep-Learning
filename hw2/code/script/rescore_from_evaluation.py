#!/usr/bin/env python3
"""Rescore a reordered candidate set using already-recorded judge decisions."""

import argparse

from common import ROOT, read_json, write_json, sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    predictions = read_json(ROOT / args.predictions)
    evaluation = read_json(ROOT / args.evaluation)
    records = {row["qid"]: row for row in evaluation["records"]}
    prediction_qids = [row["qid"] for row in predictions["results"]]
    if predictions.get("top_k") != 10 or len(prediction_qids) != len(set(prediction_qids)):
        raise ValueError("Expected unique top-10 prediction rows")
    if set(prediction_qids) != set(records):
        raise ValueError("Prediction and evaluation qids differ")
    recalls, reciprocal = [], []
    output_records = []
    for row in predictions["results"]:
        qid = row["qid"]
        if qid not in records:
            raise ValueError(f"Unknown qid: {qid}")
        gains = {item["pid"]: item["gain"] for item in records[qid]["candidates"]}
        unknown = [pid for pid in row["ranked_pids"] if pid not in gains]
        if unknown:
            raise ValueError(f"{qid}: candidate was not judged previously: {unknown[:2]}")
        ordered = [float(gains[pid]) for pid in row["ranked_pids"]]
        recall = max(ordered, default=0.0)
        mrr = max((gain / rank for rank, gain in enumerate(ordered, 1)), default=0.0)
        recalls.append(recall)
        reciprocal.append(mrr)
        output_records.append({"qid": qid, "graded_recall@10": recall,
                               "graded_mrr@10": mrr})
    count = len(recalls)
    result = {
        "source": "reordered identical candidate sets; no new judge calls",
        "metrics": {"queries": count, "graded_recall@10": sum(recalls) / count,
                    "graded_mrr@10": sum(reciprocal) / count},
        "predictions_sha256": sha256(ROOT / args.predictions),
        "source_evaluation_sha256": sha256(ROOT / args.evaluation),
        "records": output_records,
    }
    write_json(ROOT / args.output, result)
    print(result["metrics"])


if __name__ == "__main__":
    main()
