#!/usr/bin/env python3
"""Create the required student-id folder and zip without full model weights."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("student_id", help="lower-case NTU student id, e.g. r14000000")
    ap.add_argument("--report", type=Path, default=Path("report.pdf"))
    ap.add_argument("--adapter", type=Path, default=Path("adapter_checkpoint"))
    ap.add_argument("--out-dir", type=Path, default=Path("submission"))
    args = ap.parse_args()
    sid = args.student_id.lower()
    if not re.fullmatch(r"[a-z][a-z0-9]+", sid):
        ap.error("student_id must be lower-case alphanumeric and begin with a letter")
    root = Path(__file__).resolve().parents[1]
    report = root / args.report
    adapter = root / args.adapter
    for required in [report, adapter / "adapter_config.json", adapter / "adapter_model.safetensors", root / "README.md"]:
        if not required.is_file():
            raise SystemExit(f"Required file is missing: {required}")
    target = root / args.out_dir / sid
    archive = root / args.out_dir / f"{sid}.zip"
    if target.exists() or archive.exists():
        raise SystemExit(f"Refusing to overwrite existing {target} or {archive}")
    (target / "adapter_checkpoint").mkdir(parents=True)
    shutil.copy2(report, target / "report.pdf")
    shutil.copy2(root / "README.md", target / "README.md")
    for name in ["adapter_config.json", "adapter_model.safetensors"]:
        shutil.copy2(adapter / name, target / "adapter_checkpoint" / name)
    shutil.copytree(root / "code", target / "code",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "_vendor"))
    shutil.make_archive(str(archive.with_suffix("")), "zip", root_dir=target.parent, base_dir=target.name)
    print(f"Created {archive} ({archive.stat().st_size / (1024**2):.1f} MiB)")


if __name__ == "__main__":
    main()
