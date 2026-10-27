"""Freeze a submission folder; trained weights need user-owned HTTPS hosting.

No upload is performed. Report and result provenance must already be complete.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import zipfile
from urllib.parse import quote
from common import ROOT, read_json, write_json, sha256, model_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--config", default="configs/selected.json")
    parser.add_argument("--report", default="reports/report.pdf")
    parser.add_argument("--report-markdown", default="reports/report.md")
    parser.add_argument("--predictions", default="results/v2/public.json")
    parser.add_argument("--evaluation", default="results/v2/evaluation.json")
    parser.add_argument("--model-base-url", help="Your HTTPS directory containing trained checkpoint files")
    parser.add_argument("--v3", action="store_true", help="Package the stronger training/download pipeline")
    parser.add_argument("--v4", action="store_true")
    args = parser.parse_args()
    if args.v3 and args.v4: parser.error("Choose only one experiment version")
    if not re.fullmatch(r"[a-z][a-z0-9]+", args.student_id):
        raise ValueError("Use your lowercase student ID")
    for name, digest in read_json(ROOT / "configs/starter_sha256.json").items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f"Starter changed: {name}")
    report_text = Path(args.report_markdown).read_text(encoding="utf-8")
    if any(marker in report_text for marker in ["TODO_", "{{", "REPLACE_WITH", "NOT RUN"]):
        raise ValueError("Finish and rebuild the report before packaging")
    if args.student_id not in report_text:
        raise ValueError("Report student ID does not match the package")
    if not Path(args.report).read_bytes().startswith(b"%PDF-"):
        raise ValueError("A valid report.pdf is required")
    report_receipt = read_json(Path(args.report).with_suffix(".receipt.json"))
    if report_receipt["pdf_sha256"] != sha256(args.report) or report_receipt["markdown_sha256"] != sha256(args.report_markdown):
        raise ValueError("Report files changed since build; rebuild the report")
    if report_receipt["predictions_sha256"] != sha256(args.predictions) or report_receipt["evaluation_sha256"] != sha256(args.evaluation):
        raise ValueError("Report refers to a different retrieval/evaluation experiment")
    config = read_json(args.config)
    if args.v4:
        from train_v4 import verify_complete
        from v4_common import signature
        for mode in ("listwise", "pairwise"): verify_complete(mode)
        selection = read_json(ROOT / "results/v4/selection.json")
        if report_receipt["selection_sha256"] != sha256(ROOT / "results/v4/selection.json"):
            raise ValueError("Report selection evidence changed")
        for key, digest in selection["result_sha256"].items():
            path = ROOT / f"results/v4/rerank_study/{key}/result.json"
            record = read_json(path)
            if sha256(path) != digest or record["signature"] != signature(record["config"]):
                raise ValueError(f"Changed v4 evidence {key}")
            if sha256(path.parent / "rankings.json") != record["rankings_sha256"]:
                raise ValueError("Changed rankings")
        selected_result = read_json(ROOT / f"results/v4/rerank_study/{selection['selected']}/result.json")
        if selected_result["config"] != config: raise ValueError("Config differs from train-validation selection")
    if args.v3:
        from v3_experiments import experiments
        from run_ablations import provenance
        from check_v3_training import check
        selection = read_json(ROOT / "results/v3/selection.json")
        registry = experiments()
        required = {f"E{i:02d}" for i in range(11)}
        if set(selection["result_sha256"]) != required:
            raise ValueError("All eleven main report experiments must be complete")
        for key, expected_hash in selection["result_sha256"].items():
            result_path = ROOT / f"results/v3/ablations/{key}/result.json"
            record = read_json(result_path)
            if sha256(result_path) != expected_hash or record["provenance"] != provenance(registry[key]["config"]):
                raise ValueError(f"Stale selection evidence: {key}")
            if sha256(result_path.parent / "rankings.json") != record["predictions_sha256"]:
                raise ValueError(f"Changed validation rankings: {key}")
        check("random")
        check("hard")
        if config != registry[selection["selected"]]["config"]:
            raise ValueError("Submitted config must match the train-validation selection")
    runtime = read_json(Path(args.predictions).with_suffix(".run.json"))
    receipt = read_json(Path(args.evaluation).with_suffix(".receipt.json"))
    if receipt["smoke"] or receipt["evaluation_sha256"] != sha256(args.evaluation):
        raise ValueError("Need a complete unmodified official evaluation")
    if read_json(args.evaluation)["metrics"]["queries"] != 1000:
        raise ValueError("Package requires a complete 1000-query public evaluation")
    if receipt["predictions_sha256"] != sha256(args.predictions) or runtime["prediction_sha256"] != sha256(args.predictions):
        raise ValueError("Evaluation/prediction provenance mismatch")
    if runtime["retrieval_source_sha256"] != sha256(ROOT / "retrieval_utils.py"):
        raise ValueError("Inference code changed since the measured public run")
    if any(receipt[key] != runtime[key] for key in ("dialogues_sha256", "products_sha256")):
        raise ValueError("Evaluation/retrieval dataset mismatch")
    if receipt["evaluate_py_sha256"] != sha256(ROOT / "evaluate.py"):
        raise ValueError("Evaluation was produced with a different evaluator")
    needed = [] if config["strategy"] == "bm25" else ["dense"]
    if config["strategy"] == "hybrid_rerank":
        needed.append("reranker")
    for name in needed:
        if runtime["model_fingerprints"][name] != model_fingerprint(ROOT / config[name + "_model"]):
            raise ValueError(f"{name} model files changed after evaluation")
    operational = {"device", "cache_dir", "encode_batch_size", "rerank_batch_size"}
    semantic = lambda c: {k: v for k, v in c.items() if k not in operational}
    if semantic(config) != semantic(runtime["config"]) or runtime["strategy"] != config["strategy"]:
        raise ValueError("Submitted config differs from the evaluated configuration")
    config_dir = ROOT / ("configs/v4" if args.v4 else "configs/v3" if args.v3 else "configs")
    lock = read_json(config_dir / "models.lock.json")
    if any(not re.fullmatch(r"[0-9a-f]{40}", item["revision"]) for item in lock["models"]):
        raise ValueError("Model revisions must be locked to immutable commits")
    downloads = read_json(config_dir / "downloads.json")
    downloads["trained_files"] = []
    trained_path = config["reranker_model"]
    packaged_config = dict(config)
    trained_paths = {"models/reranker-trained", "models/reranker-reproduced"} | {
        f"models/v3-reranker-{mode}{suffix}" for mode in ("random", "hard") for suffix in ("", "-reproduced")}
    trained_paths |= {f"models/v4-reranker-{mode}{suffix}" for mode in ("listwise", "pairwise") for suffix in ("", "-reproduced")}
    if "reranker" in needed and trained_path in trained_paths:
        if not args.model_base_url or not args.model_base_url.startswith("https://"):
            raise ValueError("Selected checkpoint is trained: upload its files to your own HTTPS storage, "
                             "then provide --model-base-url. No model upload is performed by this script.")
        files = sorted((ROOT / trained_path).glob("*"))
        if not any(file.suffix == ".safetensors" for file in files):
            raise ValueError("Trained checkpoint weights are missing")
        packaged_trained_path = "models/selected-listwise-reranker" if args.v4 else trained_path
        packaged_config["reranker_model"] = packaged_trained_path
        for file in files:
            if file.is_file() and file.suffix in {".json", ".safetensors", ".txt", ".model"}:
                downloads["trained_files"].append({
                    "url": args.model_base_url.rstrip("/") + "/" + quote(file.name),
                    "destination": f"{packaged_trained_path}/{file.name}", "sha256": sha256(file),
                })
    elif "reranker" in needed and trained_path not in {"models/bge-reranker-base", "models/bge-reranker-v2-m3"}:
        raise ValueError("Unknown model artifact; define and audit its download contract first")
    destination = ROOT / "dist" / args.student_id
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        raise ValueError("Submission destination exists; refusing to overwrite a frozen artifact")
    destination.mkdir(parents=True)
    for filename in ["retrieval_utils.py", "requirements.txt", "download.sh", "run.sh", "README.md", "setup_env.sh"]:
        shutil.copy2(ROOT / filename, destination / filename)
    if args.v4:
        shutil.copy2(ROOT / "run_submission_v4.sh", destination / "run.sh")
        shutil.copy2(ROOT / "README_SUBMISSION.md", destination / "README.md")
        shutil.copy2(ROOT / "REPORT_EXPERIMENTS.md", destination / "REPORT_EXPERIMENTS.md")
        shutil.copytree(ROOT / "archive/v3_20261007", destination / "archive/v3_20261007",
            ignore=shutil.ignore_patterns("models", "data", "adapter", "__pycache__", "*.pyc", "candidates.json", "*.jsonl",
                                          "retrieval.py", "evaluate.py", "eval_utils.py"))
    elif args.v3:
        shutil.copy2(ROOT / "run.sh", destination / "run_background.sh")
        shutil.copy2(ROOT / "run_v3.sh", destination / "run.sh")
    else:
        shutil.copy2(ROOT / "run_v2.sh", destination / "run.sh")
    shutil.copy2(args.report, destination / "report.pdf")
    shutil.copytree(ROOT / "code" / "script", destination / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "configs", destination / "configs",
                    ignore=shutil.ignore_patterns("v5", "__pycache__"))
    shutil.copytree(ROOT / "reports", destination / "reports")
    # The TA invokes retrieval.py without our wrapper; make its default config
    # exactly the selected configuration whose public metrics were measured.
    write_json(destination / "configs/inference.json", packaged_config)
    write_json(destination / "configs/selected.json", packaged_config)
    write_json(destination / "configs/downloads.json", downloads)
    write_json(destination / "configs/models.lock.json", lock)
    evidence_dir = destination / "experiment_evidence"
    evidence_dir.mkdir()
    for folder in [ROOT / "prepared", ROOT / "runs/training", ROOT / "results/v2"]:
        if folder.exists():
            target = evidence_dir / folder.name
            target.mkdir(exist_ok=True)
            for file in folder.glob("*.json"):
                if file.name != "candidates.json":
                    shutil.copy2(file, target / file.name)
    if args.v3:
        for name in ["runs/v3", "results/v3"]:
            source = ROOT / name
            if source.exists():
                shutil.copytree(source, evidence_dir / name.replace("/", "_"),
                                ignore=shutil.ignore_patterns("adapter", "candidates.json", "__pycache__"))
    if args.v4:
        for name in ["runs/v4", "results/v4"]:
            if (ROOT / name).exists():
                shutil.copytree(ROOT / name, evidence_dir / name.replace("/", "_"),
                    ignore=shutil.ignore_patterns("adapter_epoch_*", "resume.*", "candidates.json", "__pycache__"))
    files = [p for p in destination.rglob("*") if p.is_file()]
    write_json(destination / "submission_sha256.json", {
        str(p.relative_to(destination)): sha256(p) for p in files
    })
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for file in sorted(destination.rglob("*")):
            if file.is_file():
                handle.write(file, str(file.relative_to(destination.parent)))
    if args.v4:
        from verify_submission_archive import verify_archive
        verify_archive(archive, args.student_id)
    print(f"Created {archive}")
    print("Verify download.sh in a separate clean extraction with the TA starter files.")
    print("Keep artifact URLs valid >=3 weeks after the deadline; freeze files before the deadline.")


if __name__ == "__main__":
    main()
