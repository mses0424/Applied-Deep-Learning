"""Download-only entrypoint: immutable model snapshots / checksum-verified files.

No training, extraction, installation, public-data access, or inference here.
All upstream revisions must already be frozen in the submitted lock file.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[2]
ALLOW = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.txt", "vocab.json", "merges.txt", "added_tokens.json",
         "sentencepiece.bpe.model", "tokenizer.model", "model.safetensors", "model-*.safetensors",
         "model.safetensors.index.json"]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def destination_path(value):
    result = (ROOT / value).resolve()
    if not result.is_relative_to(ROOT / "models"):
        raise ValueError("Downloads may only write under the project models/ directory")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/downloads.json"))
    parser.add_argument("--lock", default=str(ROOT / "configs/models.lock.json"))
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    lock_path = Path(args.lock)
    if not lock_path.is_file():
        raise FileNotFoundError("Submitted immutable model lock is missing")
    lock = json.loads(lock_path.read_text())
    if lock.get("models") != config.get("models"):
        raise ValueError("Download config differs from the submitted immutable model lock")
    for item in lock["models"]:
        snapshot_download(repo_id=item["repo_id"], revision=item["revision"],
                          local_dir=destination_path(item["destination"]), allow_patterns=ALLOW)
    for item in config["trained_files"]:
        if not item["url"].startswith("https://"):
            raise ValueError("Trained artifact requires a real HTTPS download URL")
        destination = destination_path(item["destination"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and digest(destination) == item["sha256"]:
            continue
        temporary = destination.with_suffix(destination.suffix + ".partial")
        with urllib.request.urlopen(item["url"], timeout=120) as response, temporary.open("wb") as handle:
            while block := response.read(8 * 1024 * 1024):
                handle.write(block)
        if digest(temporary) != item["sha256"]:
            raise ValueError(f"Download checksum mismatch: {item['destination']}")
        temporary.replace(destination)
    print("Downloads complete; no package installs or model execution performed.")


if __name__ == "__main__":
    main()
