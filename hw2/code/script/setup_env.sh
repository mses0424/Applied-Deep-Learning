#!/usr/bin/env bash
# User-run setup only. The grader must never call this from download.sh.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# In the development tree this file is at the project root. In the submitted
# archive it is relocated to code/script/, so resolve the root in either layout.
if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
  PROJECT_ROOT="$SCRIPT_DIR"
  CHECK_ENVIRONMENT="$SCRIPT_DIR/scripts/check_environment.py"
elif [[ -f "$SCRIPT_DIR/../../requirements.txt" ]]; then
  PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
  CHECK_ENVIRONMENT="$SCRIPT_DIR/check_environment.py"
else
  echo "Could not locate the project root from $SCRIPT_DIR" >&2
  exit 1
fi
cd -- "$PROJECT_ROOT"
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.python"
export UV_PYTHON_BIN_DIR="$PWD/.python/bin"
export PIP_CACHE_DIR="$PWD/.cache/pip"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3.12}"
if ! command -v "$PYTHON_BOOTSTRAP" >/dev/null 2>&1; then
  echo "Missing $PYTHON_BOOTSTRAP. Install Python 3.12 with the venv module first." >&2
  exit 1
fi
if [[ ! -x .bootstrap/bin/python ]]; then
  "$PYTHON_BOOTSTRAP" -m venv .bootstrap
fi
.bootstrap/bin/python -m pip install 'uv==0.8.22'
.bootstrap/bin/uv python install --no-bin 3.12.9
if [[ ! -x .venv/bin/python ]]; then
  .bootstrap/bin/uv venv --python 3.12.9 .venv
fi
.venv/bin/python -c 'import sys; assert sys.version_info[:3] == (3,12,9), sys.version'
.bootstrap/bin/uv pip install --python .venv/bin/python \
  'torch==2.8.0' --index-url https://download.pytorch.org/whl/cu128
.bootstrap/bin/uv pip install --python .venv/bin/python -r requirements.txt
.bootstrap/bin/uv pip check --python .venv/bin/python
.venv/bin/python "$CHECK_ENVIRONMENT"
