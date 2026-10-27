#!/usr/bin/env python3
"""Recoverably archive generated v4 artifacts before an explicit fresh rerun."""

from __future__ import annotations

from datetime import datetime
import fcntl
import json
from pathlib import Path

from common import ROOT


GENERATED = [
    "runs/v4",
    "results/v4",
    "prepared/v4",
    "reports/v4/report.md",
    "reports/v4/report.pdf",
    "reports/v4/report.receipt.json",
    "models/v4-reranker-listwise",
    "models/v4-reranker-pairwise",
    "models/v4-reranker-listwise-reproduced",
    "models/v4-reranker-pairwise-reproduced",
    "configs/v4/candidates_selected.json",
    "configs/v4/selected.json",
]


def main() -> None:
    lock_path = ROOT / "runs/launcher/run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SystemExit(
                "A training/public job is active. Wait for `bash run_background.sh --status` "
                "to report NOT RUNNING before archiving."
            ) from error

        existing = [relative for relative in GENERATED if (ROOT / relative).exists()]
        if not existing:
            raise SystemExit("No generated v4 artifacts found; run `bash run_background.sh` directly.")
        stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
        archive = ROOT / "archive" / f"manual_v4_{stamp}"
        archive.mkdir(parents=True, exist_ok=False)
        moved = []
        for relative in existing:
            source = ROOT / relative
            destination = archive / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            source.replace(destination)
            moved.append(relative)
        manifest = {
            "created_at": datetime.now().astimezone().isoformat(),
            "purpose": "recoverable archive before an explicitly requested fresh v4 run",
            "moved": moved,
        }
        (archive / "archive_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"Archived {len(moved)} paths -> {archive}")
        print("Nothing was deleted. Start the fresh background run with: bash run_background.sh")


if __name__ == "__main__":
    main()
