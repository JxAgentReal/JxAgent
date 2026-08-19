#!/usr/bin/env bash
# Upload the final LoRA adapter (plus configs + dataset manifest) to a
# PRIVATE Hugging Face model repository. The base model stays untouched and
# the adapter is NOT merged before evaluation.
set -euo pipefail

CKPT="${1:?usage: upload_adapter.sh <checkpoint_dir>}"
[ -d "$CKPT" ] || { echo "[upload] no such checkpoint dir: $CKPT" >&2; exit 1; }
REPO="${JXAGENT_ADAPTER_REPO:-<YOUR_PRIVATE_HF_MODEL_REPO>}"
DATA_DIR="${JXAGENT_DATA_DIR:-$HOME/data/JxAgentData}"

if [ "$REPO" = "<YOUR_PRIVATE_HF_MODEL_REPO>" ]; then
  echo "[upload] set JXAGENT_ADAPTER_REPO (private HF model repo)" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")/.." && pwd)"

# provenance bundle: training config + dataset manifest + dataset revision
cp "$HERE/configs/training.yaml" "$CKPT/jxagent_training.yaml"
cp "$HERE/configs/dataset.yaml" "$CKPT/jxagent_dataset.yaml"
cp "$DATA_DIR/final/manifest.json" "$CKPT/jxagent_dataset_manifest.json" 2>/dev/null || true
cp "$DATA_DIR/final/stats.json" "$CKPT/jxagent_dataset_stats.json" 2>/dev/null || true
git -C "$HERE" rev-parse HEAD > "$CKPT/jxagent_git_commit.txt" 2>/dev/null || \
  echo "unknown" > "$CKPT/jxagent_git_commit.txt"

# prefer the current `hf` CLI; fall back to deprecated huggingface-cli
if command -v hf >/dev/null 2>&1; then
  hf repo create "$REPO" --repo-type model --private 2>/dev/null || true  # exists already -> ignore
  hf upload "$REPO" "$CKPT" . --repo-type model
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli repo create "$REPO" --repo-type model --private 2>/dev/null || true
  huggingface-cli upload "$REPO" "$CKPT" . --repo-type model
else
  echo "[upload] neither hf nor huggingface-cli found (pip install -U huggingface_hub)" >&2
  exit 1
fi
echo "[upload] adapter uploaded to $REPO (private)"
