#!/usr/bin/env bash
# Rebuild the selected v12 training mix from the public training sources.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-ta/bin/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "Missing TA venv: $PYTHON_BIN" >&2; exit 2; }
"$PYTHON_BIN" code/download_v12_sources.py
for path in data/public_test.jsonl.gz data/public_test_negotiation.jsonl.gz \
            external_data/apigen_mt_parsed/apigen-mt-5k.parquet \
            external_data/synth_apigen/data/train-00000-of-00001.parquet \
            external_data/when2call/train/when2call_train_sft.jsonl \
            external_data/toolace/data.json; do
  [[ -f "$path" ]] || { echo "Missing source file: $path; see README.md" >&2; exit 2; }
done

if [[ ! -f training_data_dependent/manifest.json || ! -f training_data_v2/manifest.json ]]; then
  bash code/prepare_training_data.sh
fi
if [[ ! -f training_data_corrective_v4/manifest.json ]]; then
  "$PYTHON_BIN" code/prepare_corrective_data.py
fi
if [[ ! -f training_data_targeted_v5/manifest.json ]]; then
  "$PYTHON_BIN" code/prepare_targeted_data.py
fi
if [[ ! -f training_data_toolace_v6/manifest.json ]]; then
  "$PYTHON_BIN" code/prepare_toolace_data.py
fi
if [[ ! -f training_data_v6_calibration/manifest.json ]]; then
  "$PYTHON_BIN" code/prepare_v6_calibration.py
fi
if [[ ! -f training_data_bfcl_v9/manifest.json ]]; then
  "$PYTHON_BIN" code/prepare_bfcl_stateful_v9.py
fi
if [[ ! -f training_data_toucan_v10/manifest.json ]]; then
  "$PYTHON_BIN" code/prepare_toucan_v10.py
fi
if [[ ! -f training_data_policy_v10/manifest.json ]]; then
  "$PYTHON_BIN" code/generate_policy_v10.py
fi
if [[ ! -f training_data_policy_v11/manifest.json ]]; then
  "$PYTHON_BIN" code/generate_policy_v11.py
fi
if [[ ! -f training_data_bfcl_executed_v12/manifest.json ]]; then
  "$PYTHON_BIN" -c 'import bfcl_eval' || {
    echo "Install the public simulator: $PYTHON_BIN -m pip install --no-deps bfcl-eval==2025.6.8" >&2
    exit 2
  }
  "$PYTHON_BIN" code/prepare_bfcl_executed_v12.py
fi
if [[ ! -f training_data_policy_branch_v12/manifest.json ]]; then
  "$PYTHON_BIN" code/generate_policy_branch_v12.py
fi

"$PYTHON_BIN" code/prepare_fresh_v11.py --bfcl-executed --policy-branches \
  --out-dir training_data_fresh_v12 --seed 20261030
"$PYTHON_BIN" code/audit_data.py \
  training_data_fresh_v12/train.jsonl training_data_fresh_v12/validation.jsonl
echo "v12 data ready: training_data_fresh_v12/"
