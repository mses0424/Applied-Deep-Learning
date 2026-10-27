"""Controlled v4 retrieval selection, mining and reranker comparisons; train-only."""
import argparse
import csv
import os
from common import ROOT, read_json, write_json, sha256, verify_train_manifest, run_retrieval, exact_metrics
from v4_common import candidate_configs, signature, inference_config


def run_case(key, cfg, rows, k, folder):
    path = folder / key
    expected = signature(cfg)
    if (path / "result.json").exists():
        saved = read_json(path / "result.json")
        if saved["signature"] != expected or saved["rankings_sha256"] != sha256(path / "rankings.json"):
            raise ValueError(f"Stale experiment {key}; preserve it in a separate version")
        return saved
    write_json(path / "config.json", cfg)
    os.environ["HW2_CANDIDATES_OUTPUT"] = str(path / "candidates.json")
    for name in ("HW2_ENCODE_BATCH", "HW2_RERANK_BATCH", "HW2_CACHE_DIR"):
        os.environ.pop(name, None)
    predictions = run_retrieval(rows, path / "config.json", cfg["strategy"], k=k, metadata_path=path / "run.json")
    rankings = {r["qid"]: r["ranked_pids"] for r in predictions}
    write_json(path / "rankings.json", rankings)
    result = {"id": key, "signature": expected, "config": cfg,
              "metrics": exact_metrics(rows, rankings),
              "coverage": exact_metrics(rows, read_json(path / "candidates.json"), cfg["rerank_k"]),
              "run": read_json(path / "run.json"), "rankings_sha256": sha256(path / "rankings.json")}
    write_json(path / "result.json", result)
    return result


def choose_candidates(dev):
    results = {key: run_case(key, cfg, dev, cfg["retrieve_k"], ROOT / "results/v4/candidate_study")
               for key, cfg in candidate_configs().items()}
    best = max(sorted(results), key=lambda k: (results[k]["coverage"]["exact_recall@128"],
               results[k]["metrics"]["exact_mrr@10"], -results[k]["run"]["timings"]["total_seconds"]))
    write_json(ROOT / "configs/v4/candidates_selected.json", results[best]["config"])
    write_json(ROOT / "results/v4/candidate_selection.json", {
        "selected": best, "rule": "Train-validation Recall@128, then MRR@10, then time; same budget",
        "results": {k: {"coverage": r["coverage"], "metrics": r["metrics"]} for k, r in results.items()}})
    print("Candidate method selected:", best, results[best]["coverage"], flush=True)


def mine(train, dev):
    cfg = read_json(ROOT / "configs/v4/candidates_selected.json")
    folder = ROOT / "prepared/v4"
    expected = signature(cfg)
    if (folder / "mining.json").exists():
        old = read_json(folder / "mining.json")
        if old["signature"] != expected or old["candidates_sha256"] != sha256(folder / "candidates.json"):
            raise ValueError("Mining inputs changed")
        return
    os.environ.pop("HW2_CANDIDATES_OUTPUT", None)
    results = run_retrieval(train + dev, ROOT / "configs/v4/candidates_selected.json", "hybrid",
                            k=cfg["retrieve_k"], metadata_path=folder / "run.json")
    rankings = {r["qid"]: r["ranked_pids"] for r in results}
    write_json(folder / "candidates.json", rankings)
    write_json(folder / "mining.json", {"signature": expected, "candidates_sha256": sha256(folder / "candidates.json"),
               "manifest_sha256": sha256(ROOT / "prepared/manifest.json"),
               "coverage": exact_metrics(dev, rankings, cfg["rerank_k"]), "source": "official train only"})


def evaluate(dev):
    configs = {"L0_v3": read_json(ROOT / "archive/v3_20261007/configs/v3/selected.json"),
               "C0_pretrained": inference_config("models/bge-reranker-v2-m3"),
               "C1_v3_new_candidates": inference_config("models/v3-reranker-hard")}
    for mode in ("listwise", "pairwise"):
        from train_v4 import verify_complete
        verify_complete(mode)
        model_path = read_json(ROOT / f"runs/v4/{mode}/manifest.json")["checkpoint"]
        configs[f"T_{mode}"] = inference_config(model_path)
        configs[f"T_{mode}_blend"] = {**configs[f"T_{mode}"], "rerank_blend": 0.2}
    results = {k: run_case(k, c, dev, 10, ROOT / "results/v4/rerank_study") for k, c in configs.items()}
    # Include the exact previous method as a fallback; never force a new model
    # into the submission just because it required more training.
    best = max(sorted(results), key=lambda k: (results[k]["metrics"]["exact_mrr@10"],
               results[k]["metrics"]["exact_recall@10"], -results[k]["run"]["timings"]["total_seconds"]))
    write_json(ROOT / "configs/v4/selected.json", configs[best])
    write_json(ROOT / "results/v4/selection.json", {"selected": best, "metrics": results[best]["metrics"],
               "public_data_used": False, "criterion": "Train-validation MRR@10, then Recall@10, then time",
               "result_sha256": {k: sha256(ROOT / f"results/v4/rerank_study/{k}/result.json") for k in results}})
    print("Selected", best, results[best]["metrics"], "Official public score is still unknown.", flush=True)


def summarize():
    root = ROOT / "results/v4"
    rows = []
    for group in ("candidate_study", "rerank_study"):
        for path in sorted((root / group).glob("*/result.json")):
            r = read_json(path)
            rows.append([group, r["id"], r["metrics"]["exact_recall@10"], r["metrics"]["exact_mrr@10"],
                         r["config"]["rerank_k"], r["coverage"][f"exact_recall@{r['config']['rerank_k']}"],
                         r["run"]["timings"]["total_seconds"], r["run"]["cache_hit"]])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "comparisons.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["study", "method", "exact_recall10", "exact_mrr10", "budget", "candidate_recall", "seconds", "warm_cache"])
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["candidates", "mine", "evaluate", "summary"])
    args = parser.parse_args()
    train, dev, _ = verify_train_manifest()
    if args.stage == "candidates": choose_candidates(dev)
    elif args.stage == "mine": mine(train, dev)
    elif args.stage == "evaluate": evaluate(dev)
    summarize()
