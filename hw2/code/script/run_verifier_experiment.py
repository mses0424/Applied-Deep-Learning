#!/usr/bin/env python3
"""Apply the label-blind Qwen verifier to recorded retrieval predictions."""

import argparse
import os
from pathlib import Path
import time

from common import ROOT, model_fingerprint, read_json, write_json, sha256
from eval_utils import load_products, load_retrieval_queries
from llm_verifier import rerank_with_verifier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--dialogues", required=True)
    parser.add_argument("--products", required=True)
    parser.add_argument("--config", default="configs/v5/verifier.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = ROOT / args.output
    if output.exists():
        raise SystemExit(f"Output exists: {output}")
    cfg = read_json(ROOT / args.config)
    payload = read_json(ROOT / args.predictions)
    queries = load_retrieval_queries(str(ROOT / args.dialogues))
    passages = load_products(str(ROOT / args.products))
    by_qid = {row["qid"]: row["ranked_pids"] for row in payload["results"]}
    if set(by_qid) != {row["qid"] for row in queries}:
        raise ValueError("Prediction and dialogue qids differ")
    rankings = [by_qid[row["qid"]] for row in queries]
    if args.output_top_k < 1 or args.output_top_k > payload["top_k"]:
        raise ValueError("--output-top-k must fit inside the source prediction depth")
    started = time.perf_counter()
    reranked, diagnostics = rerank_with_verifier(
        queries, passages, rankings, model_path=ROOT / cfg["model"], device=args.device,
        candidate_count=cfg["candidate_count"], batch_size=cfg["batch_size"],
        query_tokens=cfg["query_tokens"], product_tokens=cfg["product_tokens"],
        rank_prior_weight=cfg["rank_prior_weight"],
    )
    result = {"top_k": args.output_top_k, "results": [
        {"qid": row["qid"], "ranked_pids": ranking[:args.output_top_k]}
        for row, ranking in zip(queries, reranked)
    ]}
    write_json(output, result)
    diagnostics.update({"runtime_seconds": time.perf_counter() - started,
                        "source_predictions_sha256": sha256(ROOT / args.predictions),
                        "config_sha256": sha256(ROOT / args.config),
                        "model_fingerprint": model_fingerprint(ROOT / cfg["model"]),
                        "dialogues_sha256": sha256(ROOT / args.dialogues),
                        "products_sha256": sha256(ROOT / args.products),
                        "public_labels_used": False})
    write_json(output.with_suffix(".verifier.json"), diagnostics)
    print(f"Saved label-blind verifier ranking -> {output}")


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
