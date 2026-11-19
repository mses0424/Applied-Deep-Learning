#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: bash code/finish_submission.sh STUDENT_ID ['Student Name']" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv-ta/bin/python}"
STUDENT_ID="$1"
STUDENT_NAME="${2:-}"
RUN_TAG="fresh_v12_20261026_124322"
SOURCE_ADAPTER="runs/$RUN_TAG/train/checkpoint-10800"
TOOLS_SCORE="code/evidence/${RUN_TAG}_tools_full.json"
BUYER_SCORE="code/evidence/${RUN_TAG}_negotiation_full.json"

[[ -x "$PYTHON_BIN" ]] || { echo "Missing TA venv: $PYTHON_BIN" >&2; exit 3; }
[[ -f "$TOOLS_SCORE" && -f "$BUYER_SCORE" ]] || { echo "Missing complete v12 PUBLIC scores" >&2; exit 3; }
"$PYTHON_BIN" code/validate_adapter.py adapter_checkpoint
if [[ -f "$SOURCE_ADAPTER/adapter_model.safetensors" ]]; then
  cmp -- "$SOURCE_ADAPTER/adapter_config.json" adapter_checkpoint/adapter_config.json
  cmp -- "$SOURCE_ADAPTER/adapter_model.safetensors" adapter_checkpoint/adapter_model.safetensors
fi

"$PYTHON_BIN" code/build_v12_curve.py
"$PYTHON_BIN" code/plot_curve.py results/v12_learning_curve.csv --out results/v12_learning_curve.svg
if [[ "${REBUILD_REPORT:-0}" == "1" || ! -f report.pdf ]]; then
  "$PYTHON_BIN" code/finalize_report.py --student-id "$STUDENT_ID" --name "$STUDENT_NAME" \
    --tools "$TOOLS_SCORE" --negotiation "$BUYER_SCORE"
  "$PYTHON_BIN" code/build_report_pdf.py report/final_report.md \
    --curve results/v12_learning_curve.csv --out report.pdf
fi
"$PYTHON_BIN" code/package_submission.py "$STUDENT_ID"

echo "Final report and submission/${STUDENT_ID,,}.zip are ready."
