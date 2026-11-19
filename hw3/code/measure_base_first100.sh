#!/usr/bin/env bash
# Like-for-like report Q5 comparison on the first 100 PUBLIC tool steps.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-ta/bin/python}"
mkdir -p preds results runs

"$PYTHON_BIN" eval.py generate --data data/public_test.jsonl.gz \
  --out preds/untuned_base_first100_128.jsonl \
  --model Qwen/Qwen3-0.6B-Base --limit 100 --max-tokens 128 --gpu-util 0.70
"$PYTHON_BIN" eval.py score --data data/public_test.jsonl.gz \
  --pred preds/untuned_base_first100_128.jsonl --only-predicted \
  --out results/untuned_base_first100_128.json

"$PYTHON_BIN" eval.py generate --data data/public_test.jsonl.gz \
  --out preds/v12_first100_128.jsonl --adapter adapter_checkpoint \
  --merge-dir runs/v12_first100_128_merged --limit 100 --max-tokens 128 --gpu-util 0.70
"$PYTHON_BIN" eval.py score --data data/public_test.jsonl.gz \
  --pred preds/v12_first100_128.jsonl --only-predicted \
  --out results/v12_first100_128.json

"$PYTHON_BIN" - <<'PY'
import json
for name in ("untuned_base", "v12"):
    path = f"results/{name}_first100_128.json"
    data = json.load(open(path, encoding="utf-8"))
    print(name, data["headline"]["with_first_step_rule"]["all"]["weighted_step_acc"])
PY
