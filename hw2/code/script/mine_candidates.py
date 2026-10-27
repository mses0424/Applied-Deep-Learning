"""Generate label-blind hybrid candidates for train/validation ranking experiments."""
from common import ROOT, read_json, write_json, sha256, verify_train_manifest, run_retrieval, exact_metrics


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", action="store_true")
    args = parser.parse_args()
    directory = ROOT / ("prepared/v3" if args.v3 else "prepared")
    config_path = ROOT / ("configs/v3/inference.json" if args.v3 else "configs/inference.json")
    train, dev, manifest = verify_train_manifest()
    output = directory / "candidates.json"
    if output.exists():
        raise SystemExit("Candidate file exists; not overwriting recorded training inputs.")
    cfg = read_json(config_path)
    results = run_retrieval(train + dev, config_path, "hybrid",
                            k=cfg["retrieve_k"], metadata_path=directory / "mining.run.json")
    rankings = {row["qid"]: row["ranked_pids"] for row in results}
    write_json(output, rankings)
    write_json(directory / "mining.json", {
        "source": "official train only; targets excluded from retrieve_and_rerank inputs",
        "manifest_sha256": sha256(ROOT / "prepared/manifest.json"),
        "candidates_sha256": sha256(output), "config": cfg,
        "retrieval_source_sha256": sha256(ROOT / "retrieval_utils.py"),
        "validation_hybrid": exact_metrics(dev, rankings),
        "validation_candidate_recall": exact_metrics(dev, rankings, cfg["rerank_k"]),
    })
    baseline = run_retrieval(dev, config_path, "bm25", k=10)
    write_json(directory / "baselines.json", {
        "bm25": exact_metrics(dev, {r["qid"]: r["ranked_pids"] for r in baseline}),
        "hybrid": exact_metrics(dev, rankings),
    })
    print("Mined candidates and recorded validation baselines.")


if __name__ == "__main__":
    main()
