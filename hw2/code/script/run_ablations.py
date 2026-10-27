"""Run controlled experiments ONLY on the held-out official training split."""
import argparse
import csv
import os
from pathlib import Path
import subprocess
import sys
from common import (ROOT, read_json, write_json, sha256, verify_train_manifest,
                    run_retrieval, exact_metrics, model_fingerprint)
from v3_experiments import experiments


def provenance(config):
    keys = [] if config["strategy"] == "bm25" else ["dense"]
    if config["strategy"] == "hybrid_rerank":
        keys.append("reranker")
    return {
        "config": config,
        "manifest_sha256": sha256(ROOT / "prepared/manifest.json"),
        "code": {name: sha256(ROOT / name) for name in
                 ["retrieval_utils.py", "code/script/run_ablations.py", "code/script/v3_experiments.py",
                  "code/script/common.py", "eval_utils.py"]},
        "models": {key: model_fingerprint(ROOT / config[key + "_model"]) for key in keys},
    }


def worker(key):
    _, dev, _ = verify_train_manifest()
    entry = experiments()[key]
    directory = ROOT / "results/v3/ablations" / key
    directory.mkdir(parents=True, exist_ok=True)
    config = entry["config"]
    if key in {"E09", "E10"}:
        from check_v3_training import check
        check("random" if key == "E09" else "hard")
    expected = provenance(config)
    record_path = directory / "result.json"
    if record_path.exists():
        old = read_json(record_path)
        if old["provenance"] != expected or old["predictions_sha256"] != sha256(directory / "rankings.json"):
            raise ValueError(f"{key} evidence changed; preserve it and run in a fresh project copy")
        print(f"{key}: verified previous completed experiment", flush=True)
        return
    write_json(directory / "config.json", config)
    # Prevent accidental shell overrides from corrupting a controlled comparison.
    for variable in ["HW2_STRATEGY", "HW2_ENCODE_BATCH", "HW2_RERANK_BATCH", "HW2_CACHE_DIR"]:
        os.environ.pop(variable, None)
    os.environ["HW2_CANDIDATES_OUTPUT"] = str(directory / "candidates.json")
    rows = run_retrieval(dev, directory / "config.json", config["strategy"],
                         metadata_path=directory / "run.json")
    ranks = {r["qid"]: r["ranked_pids"] for r in rows}
    write_json(directory / "rankings.json", ranks)
    run = read_json(directory / "run.json")
    candidate_metrics = None
    if config["strategy"] != "bm25":
        candidate_metrics = exact_metrics(dev, read_json(directory / "candidates.json"), config["rerank_k"])
    write_json(record_path, {"id": key, **entry, "provenance": expected,
                            "predictions_sha256": sha256(directory / "rankings.json"),
                            "metrics": exact_metrics(dev, ranks), "candidate_metrics": candidate_metrics,
                            "runtime": run})


def summarize(select=False):
    registry = experiments()
    records = {}
    table = []
    for key, entry in registry.items():
        path = ROOT / "results/v3/ablations" / key / "result.json"
        if path.exists():
            if key in {"E09", "E10"}:
                from check_v3_training import check
                check("random" if key == "E09" else "hard")
            record = read_json(path)
            if record["provenance"] != provenance(entry["config"]):
                raise ValueError(f"Stale evidence for {key}")
            if record["predictions_sha256"] != sha256(path.parent / "rankings.json"):
                raise ValueError(f"Changed predictions for {key}")
            records[key] = record
            t = record["runtime"]["timings"]
            table.append([key, entry["name"], entry["parent"], "DONE",
                          record["metrics"]["exact_recall@10"], record["metrics"]["exact_mrr@10"],
                          t["total_seconds"], t.get("peak_cuda_allocated_gib", 0),
                          record["runtime"]["cache_hit"]])
        else:
            table.append([key, entry["name"], entry["parent"], "NOT RUN", "", "", "", "", ""])
    output = ROOT / "results/v3"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "ablations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "method", "parent", "status", "exact_recall@10", "exact_mrr@10",
                         "seconds", "peak_allocated_gib", "cache_hit"])
        writer.writerows(table)
    if records:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(11, 8))
        keys = list(records)
        axes[0].plot(keys, [records[k]["metrics"]["exact_recall@10"] for k in keys], "o-", label="Exact Recall@10")
        axes[0].plot(keys, [records[k]["metrics"]["exact_mrr@10"] for k in keys], "o-", label="Exact MRR@10")
        axes[0].set_ylim(0, 1)
        axes[0].legend()
        axes[0].set_title("Train validation only; NOT official LLM-judged public scores")
        axes[1].bar(keys, [records[k]["runtime"]["timings"]["total_seconds"] for k in keys])
        axes[1].set_ylabel("Seconds (see CSV for cold/warm cache)")
        fig.tight_layout()
        fig.savefig(output / "ablations.png", dpi=160)
        plt.close(fig)
    if select:
        required = {f"E{i:02d}" for i in range(11)}
        if required - records.keys():
            raise ValueError(f"Complete main report experiments before selection: {sorted(required - records.keys())}")
        # Budget variants are supplementary, not added to the selection search.
        best = max(sorted(required), key=lambda k: (records[k]["metrics"]["exact_mrr@10"],
                   records[k]["metrics"]["exact_recall@10"], -records[k]["runtime"]["timings"]["total_seconds"]))
        write_json(ROOT / "configs/v3/selected.json", records[best]["config"])
        write_json(output / "selection.json", {"selected": best, "metrics": records[best]["metrics"],
                   "criterion": "Train-validation exact MRR@10, then Recall@10, then measured runtime",
                   "public_data_used": False, "ta_8gb_cold_runtime_verified": False,
                   "result_sha256": {k: sha256(output / "ablations" / k / "result.json") for k in sorted(required)}})
        print(f"Selected {best}. Public boss thresholds and TA runtime still require verification.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", default=",".join(f"E{i:02d}" for i in range(11)))
    parser.add_argument("--worker", choices=list(experiments()))
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--select", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        worker(args.worker)
        return
    verify_train_manifest()
    if not args.summary_only:
        ids = args.ids.split(",")
        if set(ids) - experiments().keys():
            parser.error("Unknown experiment ID")
        for key in ids:
            subprocess.run([sys.executable, __file__, "--worker", key], check=True)
    summarize(args.select)


if __name__ == "__main__":
    main()
