#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -x .venv/bin/python ]]; then
  export PATH="$PWD/.venv/bin:$PATH"
fi
export HF_HOME="$PWD/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HW2_DEVICE=cuda:0

# Reproduce only the selected training path using the released training split.
python code/script/check_sources.py
if [[ ! -f prepared/manifest.json ]]; then
  python code/script/prepare_train.py
fi
python code/script/v4_pipeline.py mine
python code/script/train_v4.py --mode listwise
# A completed second invocation verifies the frozen specification, checkpoint,
# training history, and fingerprint without retraining.
python code/script/train_v4.py --mode listwise
