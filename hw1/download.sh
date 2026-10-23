#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/model"
ARCHIVE="${SCRIPT_DIR}/model.tar.gz"

if [[ -f "${MODEL_DIR}/inference_config.json" ]]; then
  echo "Model already present."
  exit 0
fi

# Google Drive file id for model.tar.gz, shared as "Anyone with the link: Viewer".
MODEL_GDRIVE_ID="${MODEL_GDRIVE_ID:-1y0mTF7dshl1FO_HqujKIIcVGv7QWMnZa}"
MODEL_SHA256="03a4832e8779f76016db63dbdae0c3c10ede5117eb3515841d01899d1f66344b"

gdown "${MODEL_GDRIVE_ID}" -O "${ARCHIVE}"
echo "${MODEL_SHA256}  ${ARCHIVE}" | sha256sum --check --status || {
  echo "Downloaded model archive failed SHA-256 verification." >&2
  exit 1
}
tar -xzf "${ARCHIVE}" -C "${SCRIPT_DIR}"
test -f "${MODEL_DIR}/inference_config.json"
