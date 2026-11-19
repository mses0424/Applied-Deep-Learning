#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY_PREFIX="$ROOT/.python311"
VENV="$ROOT/.venv-ta"

if [[ ! -x "$VENV/bin/python" ]] || [[ "$($VENV/bin/python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)" != "3.11.15" ]]; then
  PYTHON311=""
  for candidate in python3.11 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && [[ "$("$candidate" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" == "3.11.15" ]]; then
      PYTHON311="$(command -v "$candidate")"
      break
    fi
  done
  if [[ -z "$PYTHON311" ]]; then
    if [[ ! -x "$PY_PREFIX/bin/python" ]] || [[ "$($PY_PREFIX/bin/python -c 'import sys; print(".".join(map(str, sys.version_info[:3])))' 2>/dev/null || true)" != "3.11.15" ]]; then
      command -v conda >/dev/null 2>&1 || { echo "Install Python 3.11.15 or conda before running this script" >&2; exit 2; }
      conda create --yes --prefix "$PY_PREFIX" python=3.11.15 pip
    fi
    PYTHON311="$PY_PREFIX/bin/python"
  fi
  "$PYTHON311" -m venv "$VENV"
fi

"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install \
  torch==2.11.0 transformers==5.13.1 peft==0.19.1 vllm==0.25.1 jinja2==3.1.6 \
  'accelerate>=1.2' 'bitsandbytes>=0.45' 'matplotlib>=3.8' 'safetensors>=0.4'
"$VENV/bin/python" -m pip install pyarrow==22.0.0 mpmath==1.3.0
"$VENV/bin/python" -m pip install --no-deps bfcl-eval==2025.6.8

cd "$ROOT"
"$VENV/bin/python" code/check_environment.py --require-ta-exact
echo "TA-compatible VENV ready: $VENV"
