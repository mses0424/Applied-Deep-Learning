#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${SCRIPT_DIR}/.venv" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42

"${PYTHON}" "${SCRIPT_DIR}/report_ablations.py"   --train-file "${SCRIPT_DIR}/train.jsonl"   --output "${SCRIPT_DIR}/reports/linear_ablations.json"

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2   "${SCRIPT_DIR}/experiment_transformer_split.py"   --train-file "${SCRIPT_DIR}/train.jsonl"   --model-name microsoft/mdeberta-v3-base   --output "${SCRIPT_DIR}/reports/mdeberta_split.json"   --epochs 4 --max-length 512 --batch-size 2   --gradient-accumulation 4 --mixed-precision fp16

"${PYTHON}" -m torch.distributed.run --standalone --nproc_per_node=2   "${SCRIPT_DIR}/experiment_transformer_split.py"   --train-file "${SCRIPT_DIR}/train.jsonl"   --model-name FacebookAI/xlm-roberta-base   --output "${SCRIPT_DIR}/reports/xlmr_split.json"   --epochs 4 --max-length 512 --batch-size 2   --gradient-accumulation 4 --mixed-precision fp16

echo "Experiments complete. Measured JSON files are under ${SCRIPT_DIR}/reports/."
