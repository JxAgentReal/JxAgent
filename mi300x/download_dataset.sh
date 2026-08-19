#!/usr/bin/env bash
# Download the PREPARED JxAgent dataset (built locally, uploaded beforehand via
# publish_dataset.py) into persistent storage. The MI300X never builds the
# dataset from raw sources.
set -euo pipefail

REPO="${JXAGENT_DATASET_REPO:-<YOUR_PRIVATE_HF_DATASET_REPO>}"
DEST="${JXAGENT_DATA_DIR:-$HOME/data/JxAgentData}"
mkdir -p "$DEST"

if [ "$REPO" = "<YOUR_PRIVATE_HF_DATASET_REPO>" ]; then
  echo "[dataset] set JXAGENT_DATASET_REPO (private HF dataset repo created by publish_dataset.py)" >&2
  exit 1
fi

echo "[dataset] downloading $REPO -> $DEST"
HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download "$REPO" \
  --repo-type dataset \
  --local-dir "$DEST" \
  --max-workers 8

bash "$(dirname "$0")/validate_dataset.sh"
