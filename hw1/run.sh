#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: bash run.sh CONTEXT_JSON TEST_JSON OUTPUT_CSV" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model"
if [[ ! -f "${MODEL_DIR}/inference_config.json" ]]; then
  MODEL_DIR="${SCRIPT_DIR}/artifacts/xlmr_final_model"
fi
if [[ ! -f "${MODEL_DIR}/inference_config.json" ]]; then
  echo "Model not found. Run download.sh first." >&2
  exit 1
fi

PYTHON="python3"
if [[ -x "${SCRIPT_DIR}/.venv/bin/python" ]]; then
  PYTHON="${SCRIPT_DIR}/.venv/bin/python"
fi

EXPECTED_SKLEARN="1.7.2"
if ! SKLEARN_VERSION="$("${PYTHON}" -c 'import sklearn; print(sklearn.__version__)' 2>/dev/null)"; then
  echo "scikit-learn is missing from ${PYTHON}." >&2
  echo "Install the required environment with: ${PYTHON} -m pip install -r requirements.txt" >&2
  exit 1
fi
if [[ "${SKLEARN_VERSION}" != "${EXPECTED_SKLEARN}" ]]; then
  echo "Incompatible scikit-learn version: ${SKLEARN_VERSION} (expected ${EXPECTED_SKLEARN})." >&2
  echo "Create/use .venv and install: .venv/bin/python -m pip install --force-reinstall scikit-learn==${EXPECTED_SKLEARN}" >&2
  exit 1
fi

"${PYTHON}" "${SCRIPT_DIR}/predict.py" \
  --context "$1" \
  --test "$2" \
  --output "$3" \
  --model-dir "${MODEL_DIR}"
