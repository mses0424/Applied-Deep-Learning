"""Build the course-layout code bundle with cloud-pinned model downloads.

The generated archive contains no model weights, datasets, caches, virtual
environment, or TA starter files.  The current report.pdf is copied verbatim;
replace it before submission when the written report is finalized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

from common import ROOT, model_fingerprint, read_json, sha256, write_json


STUDENT_FILES = ["retrieval_utils.py", "requirements.txt"]
MODEL_NAMES = {
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


def transform_script(source: Path, destination: Path) -> None:
    """Relocate root/scripts to root/code/script without changing behavior."""
    text = source.read_text(encoding="utf-8")
    text = text.replace("Path(__file__).resolve().parents[2]", "Path(__file__).resolve().parents[2]")
    text = text.replace('ROOT / "code" / "script"', 'ROOT / "code" / "script"')
    text = text.replace('"code/script/', '"code/script/')
    text = text.replace("'code/script/", "'code/script/")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    shutil.copymode(source, destination)


def copy_evidence(destination: Path) -> None:
    """Keep compact report evidence, excluding predictions/caches/weights."""
    skipped_names = {"candidates.json", "rankings.json", "resume.receipt.json"}
    allowed = {".json", ".csv", ".md", ".txt"}
    for base_name in ("results/v3", "results/v4", "runs/v3", "runs/v4"):
        base = ROOT / base_name
        if not base.exists():
            continue
        for source in base.rglob("*"):
            if (not source.is_file() or source.name in skipped_names or
                    source.suffix not in allowed or source.stat().st_size > 5 * 1024 * 1024):
                continue
            target = destination / base_name / source.relative_to(base)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--report", default="report.pdf")
    parser.add_argument("--output", help="Defaults to dist/<student-id>_code_ready.zip")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9]+", args.student_id):
        raise ValueError("Student ID must be lower-case and start with a letter")

    report = (ROOT / args.report).resolve()
    if not report.is_file() or not report.read_bytes().startswith(b"%PDF-"):
        raise ValueError("A valid report.pdf placeholder/final report is required")
    for name, expected in read_json(ROOT / "configs/starter_sha256.json").items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"TA starter changed: {name}")

    selected_model = ROOT / "models/v4-reranker-listwise"
    urls = read_json(ROOT / "configs/v4/trained_urls.json")
    files = {p.name: p for p in selected_model.iterdir() if p.is_file() and p.name in MODEL_NAMES}
    if set(files) != MODEL_NAMES or set(urls) != MODEL_NAMES:
        raise ValueError("Selected model files and cloud URL manifest must contain exactly six files")

    config = read_json(ROOT / "configs/v4/selected.json")
    config["reranker_model"] = "models/selected-listwise-reranker"
    downloads = read_json(ROOT / "configs/v4/downloads.json")
    downloads["trained_files"] = [
        {
            "url": urls[name],
            "destination": f"models/selected-listwise-reranker/{name}",
            "sha256": sha256(files[name]),
        }
        for name in sorted(MODEL_NAMES)
    ]
    lock = read_json(ROOT / "configs/v4/models.lock.json")
    if downloads["models"] != lock["models"]:
        raise ValueError("Pretrained download config and immutable lock differ")

    output = Path(args.output) if args.output else ROOT / "dist" / f"{args.student_id}_code_ready.zip"
    if not output.is_absolute():
        output = ROOT / output
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite frozen bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hw2_bundle_", dir=output.parent) as temporary:
        outer = Path(temporary) / args.student_id
        outer.mkdir()
        for name in STUDENT_FILES:
            shutil.copy2(ROOT / name, outer / name)
        shutil.copy2(ROOT / "submission_download.sh", outer / "download.sh")
        shutil.copy2(ROOT / "submission_run.sh", outer / "run.sh")
        shutil.copy2(report, outer / "report.pdf")
        shutil.copy2(ROOT / "README_SUBMISSION.md", outer / "README.md")
        for shell in (outer / "download.sh", outer / "run.sh"):
            shell.chmod(0o755)

        script_dir = outer / "code/script"
        for source in sorted((ROOT / "code" / "script").glob("*.py")):
            transform_script(source, script_dir / source.name)
        shutil.copy2(ROOT / "setup_env.sh", script_dir / "setup_env.sh")
        (script_dir / "setup_env.sh").chmod(0o755)

        readme = (outer / "README.md").read_text(encoding="utf-8")
        readme = readme.replace("code/script/", "code/script/")
        readme = readme.replace("`setup_env.sh`", "`code/script/setup_env.sh`")
        readme = readme.replace("`experiment_evidence/`", "`code/evidence/`")
        readme = readme.replace("`archive/v3_20261007/`", "`code/evidence/results/v3/`")
        (outer / "README.md").write_text(readme, encoding="utf-8")

        shutil.copytree(ROOT / "configs", outer / "configs", ignore=shutil.ignore_patterns("__pycache__", "v5"))
        write_json(outer / "configs/inference.json", config)
        write_json(outer / "configs/selected.json", config)
        write_json(outer / "configs/downloads.json", downloads)
        write_json(outer / "configs/models.lock.json", lock)
        if (ROOT / "REPORT_EXPERIMENTS.md").exists():
            target = outer / "code/report/REPORT_EXPERIMENTS.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "REPORT_EXPERIMENTS.md", target)
        copy_evidence(outer / "code/evidence")

        checksums = {}
        for file in sorted(p for p in outer.rglob("*") if p.is_file() and p.name != "report.pdf"):
            checksums[str(file.relative_to(outer))] = sha256(file)
        write_json(outer / "submission_sha256.json", checksums)

        temporary_archive = Path(temporary) / f"{args.student_id}.zip"
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
            for file in sorted(outer.rglob("*")):
                if file.is_file():
                    handle.write(file, str(file.relative_to(outer.parent)))
        from verify_submission_archive import verify_archive
        verify_archive(temporary_archive, args.student_id)
        shutil.move(temporary_archive, output)

    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(f"Created code-ready bundle: {output}")
    print(f"Replace {args.student_id}/report.pdf with the final report, then create {args.student_id}.zip for COOL.")


if __name__ == "__main__":
    main()
