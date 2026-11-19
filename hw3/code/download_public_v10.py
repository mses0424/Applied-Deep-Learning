#!/usr/bin/env python3
"""Download pinned official BFCL and Toucan files used by the v10 curriculum."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from huggingface_hub import hf_hub_download


SOURCES = [
    {
        "repo": "gorilla-llm/Berkeley-Function-Calling-Leaderboard",
        "revision": "61fc0608cfd831fcfbbaa676ebdfef0ed963eeda",
        "license": "Apache-2.0",
        "out": Path("external_data/bfcl_v3_official"),
        "files": [
            *(f"BFCL_v3_multi_turn_{name}.json" for name in ("base", "miss_param", "miss_func", "long_context")),
            *(f"possible_answer/BFCL_v3_multi_turn_{name}.json" for name in ("base", "miss_param", "miss_func", "long_context")),
            *(f"multi_turn_func_doc/{name}.json" for name in (
                "gorilla_file_system", "math_api", "message_api", "posting_api",
                "ticket_api", "trading_bot", "travel_booking", "vehicle_control",
            )),
        ],
    },
    {
        "repo": "Agent-Ark/Toucan-1.5M",
        "revision": "0df3cf37f2abefb380370cfb02eabea2a35ae782",
        "license": "Apache-2.0",
        "out": Path("external_data/toucan_sft"),
        "files": [f"SFT/train-{index:05d}-of-00003.parquet" for index in range(3)],
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = {"downloader": "code/download_public_v10.py", "sources": []}
    for source in SOURCES:
        output = source["out"]
        output.mkdir(parents=True, exist_ok=True)
        hashes = {}
        for filename in source["files"]:
            downloaded = Path(hf_hub_download(
                repo_id=source["repo"], repo_type="dataset", filename=filename,
                revision=source["revision"], local_dir=output,
            ))
            hashes[filename] = {"bytes": downloaded.stat().st_size, "sha256": sha256(downloaded)}
            print(f"downloaded {source['repo']}:{filename} -> {downloaded}", flush=True)
        manifest["sources"].append({
            "repo": source["repo"], "revision": source["revision"],
            "license": source["license"], "files": hashes,
        })
    Path("external_data/v10_public_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
