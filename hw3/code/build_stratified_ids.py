#!/usr/bin/env python3
"""Create an evaluation-only balanced ID list; never exports PUBLIC content."""

from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/public_test.jsonl.gz"))
    parser.add_argument("--per-slice", type=int, default=200)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--out", type=Path, default=Path("results/stratified_tools_600.ids"))
    args = parser.parse_args()
    groups = {"must_call": [], "eligible_call": [], "no_call": []}
    with gzip.open(args.data, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (
                "must_call" if row.get("acceptable_next") and not row.get("first_step_eligible")
                else "eligible_call" if row.get("acceptable_next") else "no_call"
            )
            groups[key].append(row["id"])
    rng = random.Random(args.seed)
    chosen = []
    for key in ("must_call", "eligible_call", "no_call"):
        chosen.extend(rng.sample(groups[key], min(args.per_slice, len(groups[key]))))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    print(json.dumps({key: min(args.per_slice, len(value)) for key, value in groups.items()}, indent=2))


if __name__ == "__main__":
    main()
