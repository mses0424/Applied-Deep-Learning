"""Validate official metrics against public/private simple/strong/boss thresholds."""
import argparse
import math
from pathlib import Path
from common import ROOT, read_json, sha256

THRESHOLDS = {
    "public": {"simple": (0.860, 0.650), "strong": (0.885, 0.680), "boss": (0.950, 0.800)},
    "private": {"simple": (0.845, 0.635), "strong": (0.880, 0.675), "boss": (0.930, 0.780)},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default="results/v2/evaluation.json")
    parser.add_argument("--predictions", default="results/v2/public.json")
    parser.add_argument("--split", choices=THRESHOLDS, default="public")
    parser.add_argument("--target", choices=["simple", "strong", "boss"], default="boss")
    args = parser.parse_args()
    evaluation = read_json(args.evaluation)
    receipt = read_json(Path(args.evaluation).with_suffix(".receipt.json"))
    runtime = read_json(Path(args.predictions).with_suffix(".run.json"))
    prediction_hash = sha256(args.predictions)
    if receipt["smoke"] or receipt["evaluation_sha256"] != sha256(args.evaluation):
        raise SystemExit("INVALID: smoke/modified evaluation cannot establish passing")
    if receipt["predictions_sha256"] != prediction_hash or runtime["prediction_sha256"] != prediction_hash:
        raise SystemExit("INVALID: prediction/evaluation/runtime files come from different runs")
    if any(receipt[key] != runtime[key] for key in ("dialogues_sha256", "products_sha256")):
        raise SystemExit("INVALID: evaluation and retrieval datasets differ")
    expected_hash = read_json(ROOT / "configs/starter_sha256.json")["evaluate.py"]
    if receipt["evaluate_py_sha256"] != expected_hash:
        raise SystemExit("INVALID: evaluator differs from the supplied original")
    metrics = evaluation["metrics"]
    expected = 1000 if args.split == "public" else 2000
    if metrics["queries"] != expected or runtime["query_count"] != expected:
        raise SystemExit(f"INVALID: expected all {expected} {args.split} queries")
    recall, mrr = metrics["graded_recall@10"], metrics["graded_mrr@10"]
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (recall, mrr)):
        raise SystemExit("INVALID: metrics out of range")
    for name, (r, m) in THRESHOLDS[args.split].items():
        print(f'{name:6s}: Recall@10 {recall:.6f} / {r:.3f}; MRR@10 {mrr:.6f} / {m:.3f}: '
              + ('PASS' if recall >= r and mrr >= m else 'FAIL'))
    elapsed = runtime["wall_seconds"]
    print(f'Local retrieval wall time: {elapsed:.1f}s; cache hit: {runtime["cache_hit"]}')
    print(f'Judge parse failures: {evaluation["judge"]["parse_failures"]}')
    print('Public passing does not establish private passing; TA hardware timing remains separate.')
    if runtime["cache_hit"]:
        print('Warm-cache timing is not evidence of cold-start compliance. Repeat with a fresh cache directory.')
    r, m = THRESHOLDS[args.split][args.target]
    raise SystemExit(0 if recall >= r and mrr >= m and (args.split != "public" or elapsed <= 3600) else 1)


if __name__ == "__main__":
    main()
