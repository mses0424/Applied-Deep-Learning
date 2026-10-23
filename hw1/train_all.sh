#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"
MODEL_DIR="${MODEL_DIR:-${SCRIPT_DIR}/artifacts/xlmr_final_model}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "Missing ${SCRIPT_DIR}/.venv; create it first with: python3 -m venv .venv" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONHASHSEED=42

"${PYTHON}" -m torch.distributed.run \
  --standalone \
  --nproc_per_node=2 \
  "${SCRIPT_DIR}/train_transformer.py" \
  --model-name FacebookAI/xlm-roberta-base \
  --train-file "${SCRIPT_DIR}/train.jsonl" \
  --output-dir "${MODEL_DIR}" \
  --folds 5 \
  --epochs 6 \
  --max-length 512 \
  --batch-size 2 \
  --gradient-accumulation 4 \
  --learning-rate 1.5e-5 \
  --head-learning-rate 8e-5 \
  --final-seeds 42,3407 \
  --mixed-precision fp16 \
  --resume

"${PYTHON}" "${SCRIPT_DIR}/train_linear.py" \
  --train-file "${SCRIPT_DIR}/train.jsonl" \
  --model-dir "${MODEL_DIR}" \
  --folds 5 \
  --seed 42

"${PYTHON}" "${SCRIPT_DIR}/train_context_linear.py" \
  --train-file "${SCRIPT_DIR}/train.jsonl" \
  --model-dir "${MODEL_DIR}" \
  --folds 5 --seed 42 --apply

"${PYTHON}" "${SCRIPT_DIR}/train_stacker.py" \
  --train-file "${SCRIPT_DIR}/train.jsonl" \
  --model-dir "${MODEL_DIR}" \
  --folds 5 --seed 42

"${PYTHON}" "${SCRIPT_DIR}/finalize_ensemble.py" \
  --train-file "${SCRIPT_DIR}/train.jsonl" \
  --model-dir "${MODEL_DIR}"

"${PYTHON}" "${SCRIPT_DIR}/optimize_language_calibration.py" \
  --train-file "${SCRIPT_DIR}/train.jsonl" \
  --model-dir "${MODEL_DIR}" \
  --macro-floor 0.82
