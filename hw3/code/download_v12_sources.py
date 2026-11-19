#!/usr/bin/env python3
"""Fetch the pinned public training files used by the selected v12 mix.

Only public *training* sources are downloaded.  The homework PUBLIC files are
provided by the staff and are never downloaded or used as training targets.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from download_public_v10 import SOURCES as BFCL_AND_TOUCAN


ROOT = Path(__file__).resolve().parents[1]
EXTRA_SOURCES = (
    (
        "minpeter/apigen-mt-5k-parsed",
        "2b343962e59b01eaa673c34b1d9610e08d5c6369",
        "apigen-mt-5k.parquet",
        "external_data/apigen_mt_parsed/apigen-mt-5k.parquet",
        "6ce47df317967cf2e909f65dca09b3fb44ec32b7089720dbe28ed7ef860fa297",
    ),
    (
        "argilla/Synth-APIGen-v0.1",
        "20107f6709aabd18c7f7b4afc96fe7bfe848b5bb",
        "data/train-00000-of-00001.parquet",
        "external_data/synth_apigen/data/train-00000-of-00001.parquet",
        "c09cec09e20f5a90a82358c38cbffd9b5b50ab43d3f2c58f678881167b233f96",
    ),
    (
        "nvidia/When2Call",
        "0582f7749df63a96fdc3070932e83e72396ace53",
        "train/when2call_train_sft.jsonl",
        "external_data/when2call/train/when2call_train_sft.jsonl",
        "3eb20258557513579995ff55c09fcc33fabf2cd2004dea49dc3a0ba9880e631c",
    ),
    (
        "Team-ACE/ToolACE",
        "6bda777c88d21e5a204703c1ee45597a8fa4f734",
        "data.json",
        "external_data/toolace/data.json",
        "ba12c083fca7e8da48c67ad5b895e495447da7c66e39a2e19742c082e6cb537e",
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    for repo, revision, filename, local_name, expected_hash in EXTRA_SOURCES:
        local = ROOT / local_name
        if not local.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=repo, repo_type="dataset", revision=revision,
                filename=filename, local_dir=local.parent.parent if "/" in filename else local.parent,
            )
        actual = sha256(local)
        if actual != expected_hash:
            raise SystemExit(f"Wrong source bytes at {local}: {actual} != {expected_hash}")
        print(f"verified {repo}@{revision}:{filename}", flush=True)

    missing = [
        (source, filename)
        for source in BFCL_AND_TOUCAN
        for filename in source["files"]
        if not (ROOT / source["out"] / filename).is_file()
    ]
    if missing:
        # The included downloader pins both repository revisions and records
        # a checksum manifest for every BFCL and Toucan file it retrieves.
        from download_public_v10 import main as download_bfcl_and_toucan
        download_bfcl_and_toucan()
    recorded = json.loads((Path(__file__).resolve().parent / "evidence/v10_public_manifest.json").read_text(encoding="utf-8"))
    for source in recorded["sources"]:
        for filename, metadata in source["files"].items():
            local = ROOT / next(item["out"] for item in BFCL_AND_TOUCAN if item["repo"] == source["repo"]) / filename
            actual = sha256(local)
            if actual != metadata["sha256"]:
                raise SystemExit(f"Wrong source bytes at {local}: {actual} != {metadata['sha256']}")
    print("verified pinned BFCL and Toucan source hashes", flush=True)

    docs = ROOT / "external_data/bfcl_v3_official/multi_turn_func_doc"
    legacy_docs = ROOT / "external_data/bfcl_docs/multi_turn_func_doc"
    legacy_docs.mkdir(parents=True, exist_ok=True)
    for source_file in sorted(docs.glob("*.json")):
        destination = legacy_docs / source_file.name
        if not destination.is_file() or sha256(destination) != sha256(source_file):
            shutil.copy2(source_file, destination)
    print("v12 public training sources ready", flush=True)


if __name__ == "__main__":
    main()
