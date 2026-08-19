#!/usr/bin/env bash
# Download Qwen/Qwen3.8-27B into persistent storage with parallel downloads,
# then verify config/processor/tokenizer load and shard integrity.
set -euo pipefail

MODEL_ID="${JXAGENT_MODEL_ID:-Qwen/Qwen3.8-27B}"
DEST="${JXAGENT_MODEL_DIR:-$HOME/models/Qwen3.8-27B}"
# Pin a specific revision with JXAGENT_MODEL_REVISION; otherwise the resolved
# commit is recorded after download so the base model is reproducible for
# the base-vs-adapter evaluation scaffold.
REV="${JXAGENT_MODEL_REVISION:-}"
mkdir -p "$DEST"

echo "[model] downloading $MODEL_ID -> $DEST"
DL_ARGS=(--local-dir "$DEST" --max-workers 8 --exclude "*.pth" "original/*")
[ -n "$REV" ] && DL_ARGS=(--revision "$REV" "${DL_ARGS[@]}")
if command -v hf >/dev/null 2>&1; then
  hf download "$MODEL_ID" "${DL_ARGS[@]}"
else
  huggingface-cli download "$MODEL_ID" "${DL_ARGS[@]}"
fi

# Record the exact revision for provenance (base/adapter eval parity).
if [ -n "$REV" ]; then
  RESOLVED="$REV"
else
  RESOLVED=$(python - "$MODEL_ID" <<'PY' 2>/dev/null || echo unknown
import sys
from huggingface_hub import HfApi
print(HfApi().model_info(sys.argv[1]).sha)
PY
)
fi
printf '%s\n' "$RESOLVED" > "$DEST/REVISION.txt"
if [ "$RESOLVED" = "unknown" ]; then
  echo "[model][WARN] could not resolve the commit hash; pin it manually into $DEST/REVISION.txt" >&2
else
  echo "[model] base model revision pinned: $RESOLVED (recorded in REVISION.txt)"
fi

echo "[model] verifying"
python - "$DEST" <<'PY'
import json, os, sys
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
dest = sys.argv[1]
cfg = AutoConfig.from_pretrained(dest, trust_remote_code=True)
print("config ok:", cfg.model_type)
tok = AutoTokenizer.from_pretrained(dest, trust_remote_code=True)
print("tokenizer ok:", tok.__class__.__name__)
try:
    proc = AutoProcessor.from_pretrained(dest, trust_remote_code=True)
    print("processor ok:", proc.__class__.__name__)
except Exception as e:
    print("processor check:", e)
safetensors = [f for f in os.listdir(dest) if f.endswith(".safetensors")]
print("safetensors shards:", len(safetensors))
idx = os.path.join(dest, "model.safetensors.index.json")
if os.path.exists(idx):
    need = set(json.load(open(idx))["weight_map"].values())
    have = set(safetensors)
    assert need <= have, f"missing shards: {need - have}"
    print("all index shards present")
assert safetensors, "no safetensors found"
print("[model] VERIFY PASS")
PY

# Freeze the exact local interface. An unresolved native CUA contract is
# recorded but does not invalidate the weight download itself. Training
# preflight will fail closed until the manifest reaches status=verified.
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FREEZE_ARGS=("$DEST")
if [ -n "${JXAGENT_NATIVE_CONTRACT:-}" ]; then
  FREEZE_ARGS+=(--native-contract "$JXAGENT_NATIVE_CONTRACT")
fi
python "$ROOT_DIR/tools/freeze_qwen_interface.py" "${FREEZE_ARGS[@]}" || {
  rc=$?
  if [ "$rc" -eq 2 ]; then
    echo "[model][GATE] native Qwen3.8 CUA interface is unresolved; training remains blocked." >&2
  else
    exit "$rc"
  fi
}

echo "[model] done: $DEST"
