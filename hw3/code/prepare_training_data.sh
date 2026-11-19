#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-ta/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing TA venv: $PYTHON_BIN" >&2; exit 2; }
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
TEACHER_GPU="${TEACHER_GPU:-0}"

"$PYTHON_BIN" code/generate_data.py --out-dir training_data_v2 \
  --tool-conversations 12000 --negotiations 1000 --seed 3407

if ! compgen -G 'external_data/bfcl_docs/multi_turn_func_doc/*.json' >/dev/null; then
  hf download gorilla-llm/Berkeley-Function-Calling-Leaderboard \
    --repo-type dataset --include 'multi_turn_func_doc/*.json' \
    --local-dir external_data/bfcl_docs
fi

CUDA_VISIBLE_DEVICES="$TEACHER_GPU" "$PYTHON_BIN" code/synthesize_bfcl_docs.py \
  --docs external_data/bfcl_docs/multi_turn_func_doc \
  --out-dir training_data_bfcl_synthetic_v3 \
  --teacher Qwen/Qwen2.5-0.5B-Instruct --variants 8 --repeat 3 \
  --batch-size 16 --seed 3407

"$PYTHON_BIN" code/generate_multistep.py \
  --source training_data_bfcl_synthetic_v3/train.jsonl \
  --out-dir training_data_multistep_v2 --conversations 5000 --seed 3407

"$PYTHON_BIN" code/generate_dependent.py \
  --docs external_data/bfcl_docs/multi_turn_func_doc \
  --negotiation training_data_v2/train.jsonl \
  --out-dir training_data_dependent --chains 5000 --negotiations 900

"$PYTHON_BIN" code/audit_data.py \
  training_data_v2/train.jsonl training_data_v2/validation.jsonl \
  training_data_bfcl_synthetic_v3/train.jsonl training_data_bfcl_synthetic_v3/validation.jsonl \
  training_data_multistep_v2/train.jsonl training_data_multistep_v2/validation.jsonl \
  training_data_dependent/train.jsonl training_data_dependent/validation.jsonl

echo "Prepared and audited all training data."
