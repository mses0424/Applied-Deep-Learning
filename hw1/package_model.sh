#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_NAME="xlmr_final_model"
SOURCE_DIR="${SCRIPT_DIR}/artifacts/${SOURCE_NAME}"
ARCHIVE="${SCRIPT_DIR}/model.tar.gz"
if [[ ! -f "${SOURCE_DIR}/inference_config.json" ]]; then
  echo "Train the final model first." >&2
  exit 1
fi

tar -C "${SCRIPT_DIR}/artifacts" \
  --exclude="${SOURCE_NAME}/checkpoints" \
  --transform="s,^${SOURCE_NAME},model," \
  -czf "${ARCHIVE}" "${SOURCE_NAME}"
echo "Created ${ARCHIVE}"
echo "Archive root is model/ and is ready to upload."
