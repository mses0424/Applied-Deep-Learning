"""Validate the final HW2 zip layout and its frozen download/inference contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import PurePosixPath, Path
import re
import zipfile


REQUIRED = {
    "download.sh",
    "run.sh",
    "retrieval_utils.py",
    "requirements.txt",
    "report.pdf",
    "README.md",
    "configs/inference.json",
    "configs/downloads.json",
    "configs/models.lock.json",
    "code/script/download_models.py",
    "code/script/train_v4.py",
    "code/script/v4_pipeline.py",
    "code/script/generate_report_examples.py",
    "submission_sha256.json",
}
FORBIDDEN_ROOT = {"retrieval.py", "evaluate.py", "eval_utils.py"}
FORBIDDEN_PARTS = {".git", ".venv", ".python", ".cache", "cache", "data", "models", "__pycache__"}
ALLOWED_REPOS = {
    "intfloat/e5-base-v2",
    "BAAI/bge-reranker-v2-m3",
}
REQUIRED_README_TEXT = {
    "## 1. Project layout",
    "## 2. Environment setup",
    "## 3. Train the submitted model",
    "### Complete training workflow",
    "## 4. Run retrieval",
    "## 5. Main file reference",
    "bash code/script/setup_env.sh",
    "bash download.sh",
    "CUDA_VISIBLE_DEVICES=0 bash run.sh",
    "python code/script/prepare_train.py",
    "python code/script/v4_pipeline.py mine",
    "python code/script/train_v4.py --mode listwise",
}


def load_json(handle: zipfile.ZipFile, name: str):
    return json.loads(handle.read(name).decode("utf-8"))


def verify_archive(archive: str | Path, student_id: str) -> None:
    if not re.fullmatch(r"[a-z][a-z0-9]+", student_id):
        raise ValueError("Student ID must be lower-case and start with a letter")
    archive = Path(archive)
    if archive.name != f"{student_id}.zip":
        raise ValueError(f"Archive must be named {student_id}.zip")
    prefix = f"{student_id}/"
    with zipfile.ZipFile(archive) as handle:
        files = sorted(name for name in handle.namelist() if not name.endswith("/"))
        if not files or any(not name.startswith(prefix) for name in files):
            raise ValueError("Every zip member must be inside the single student-ID folder")
        relative = {name[len(prefix):] for name in files}
        missing = REQUIRED - relative
        if missing:
            raise ValueError(f"Missing required submission files: {sorted(missing)}")
        unexpected_starter = FORBIDDEN_ROOT & relative
        if unexpected_starter:
            raise ValueError(f"TA-provided root files must not be submitted: {sorted(unexpected_starter)}")
        for name in relative:
            path = PurePosixPath(name)
            if FORBIDDEN_PARTS & set(path.parts):
                raise ValueError(f"Forbidden generated/private directory in zip: {name}")
            if path.suffix in {".pyc", ".pt", ".safetensors", ".bin"}:
                raise ValueError(f"Generated cache/model weight must be downloaded, not zipped: {name}")

        report = handle.read(prefix + "report.pdf")
        if not report.startswith(b"%PDF-"):
            raise ValueError("report.pdf is not a PDF")
        readme = handle.read(prefix + "README.md").decode("utf-8")
        missing_readme = sorted(text for text in REQUIRED_README_TEXT if text not in readme)
        if missing_readme:
            raise ValueError(f"README.md is missing required setup/training instructions: {missing_readme}")
        run_text = handle.read(prefix + "run.sh").decode("utf-8")
        if "background_run.py" in run_text or "--public" in run_text:
            raise ValueError("Submission run.sh must be the foreground train-only entrypoint")
        download_text = handle.read(prefix + "download.sh").decode("utf-8")
        if re.search(r"(^|\s)(pip|uv)\s+install", download_text):
            raise ValueError("download.sh must not install packages")

        inference = load_json(handle, prefix + "configs/inference.json")
        if inference.get("strategy") != "hybrid_rerank":
            raise ValueError("Final inference strategy changed")
        if inference.get("dense_model") != "models/e5-base-v2":
            raise ValueError("Final dense model path is not the frozen E5 model")
        if inference.get("reranker_model") != "models/selected-listwise-reranker":
            raise ValueError("Final reranker path must use the semantic submission name")

        downloads = load_json(handle, prefix + "configs/downloads.json")
        lock = load_json(handle, prefix + "configs/models.lock.json")
        if downloads.get("models") != lock.get("models"):
            raise ValueError("Download model list and immutable lock differ")
        if {item.get("repo_id") for item in downloads.get("models", [])} != ALLOWED_REPOS:
            raise ValueError("Download list contains a missing or unused pretrained model")
        trained = downloads.get("trained_files", [])
        if len(trained) != 6:
            raise ValueError("Exactly six selected-reranker files must be downloaded")
        expected_names = {
            "config.json", "model.safetensors", "sentencepiece.bpe.model",
            "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
        }
        if {PurePosixPath(item["destination"]).name for item in trained} != expected_names:
            raise ValueError("Selected-reranker download file set is incomplete")
        for item in trained:
            destination = PurePosixPath(item["destination"])
            if destination.parent.as_posix() != "models/selected-listwise-reranker":
                raise ValueError("Only the selected reranker may use uploaded trained files")
            if not item["url"].startswith("https://"):
                raise ValueError("Every trained file needs a direct HTTPS URL")
            if not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
                raise ValueError("Invalid trained-file SHA256")

        checksums = load_json(handle, prefix + "submission_sha256.json")
        for relative_name, expected in checksums.items():
            member = prefix + relative_name
            if member not in files:
                raise ValueError(f"Checksum manifest refers to a missing file: {relative_name}")
            actual = hashlib.sha256(handle.read(member)).hexdigest()
            if actual != expected:
                raise ValueError(f"Checksum mismatch inside zip: {relative_name}")
    print(f"Submission archive PASS: {archive} ({len(files)} files, one outer folder {student_id}/)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive")
    parser.add_argument("student_id")
    args = parser.parse_args()
    verify_archive(args.archive, args.student_id)


if __name__ == "__main__":
    main()
