#!/usr/bin/env python3
"""Validate prediction.csv structure without reading any gold labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

LABELS = [
    "Ask_Price", "Ask_Service", "Ask_Spec", "Compare_Competitor",
    "Confirm_Order", "Doubt", "Need_More_Evidence", "Need_Time_To_Think",
    "State_Need", "Too_Expensive",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_json", type=Path)
    parser.add_argument("prediction_csv", type=Path)
    args = parser.parse_args()
    test_rows = json.loads(args.test_json.read_text(encoding="utf-8"))
    expected_ids = [row["id"] for row in test_rows]
    with args.prediction_csv.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        fieldnames = rows[0].keys() if rows else []
    expected_header = ["id", *LABELS]
    if list(fieldnames) != expected_header:
        raise ValueError(f"Wrong header: {list(fieldnames)}")
    actual_ids = [row["id"] for row in rows]
    if actual_ids != expected_ids:
        raise ValueError("Prediction ids/count/order do not match test.json")
    for row_number, row in enumerate(rows, start=2):
        for label in LABELS:
            if row[label] not in {"0", "1"}:
                raise ValueError(f"Non-binary value at row {row_number}, {label}")
    print(f"OK: {len(rows)} ordered rows, exact header, all predictions binary.")


if __name__ == "__main__":
    main()
