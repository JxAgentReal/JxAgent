#!/usr/bin/env bash
# JxAgent Run 1 - MI300X preflight checks (read-only; runs no training)
set -uo pipefail

log() { printf '[preflight] %s\n' "$*"; }
fail() { printf '[preflight][FAIL] %s\n' "$*"; FAILED=1; }
FAILED=0
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

log "1. ROCm"
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showproductname 2>/dev/null | head -5 || true
else
  fail "rocm-smi not found (ROCm not installed / not in PATH)"
fi

log "2. MI300X detection (expect gfx942)"
if command -v rocminfo >/dev/null 2>&1; then
  N=$(rocminfo 2>/dev/null | grep -c "gfx942" || true)
  log "gfx942 entries: ${N:-0}"
  [ "${N:-0}" -ge 1 ] || fail "no gfx942 (MI300X) found"
else
  fail "rocminfo not found"
fi

log "3. PyTorch (ROCm build)"
python - <<'PY' || FAILED=1
import torch
assert torch.version.hip is not None, "torch is not a ROCm/HIP build (CUDA wheel on AMD?)"
print("torch", torch.__version__, "| hip:", torch.version.hip)
print("cuda(avail):", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
assert torch.cuda.is_available(), "torch cannot see ROCm devices"
name = torch.cuda.get_device_name(0)
print("device 0:", name)
assert "MI300" in name or "gfx942" in name, f"unexpected device {name}"
PY

log "4. BF16 works (finite + cross-checked against fp32)"
python - <<'PY' || fail "BF16 test failed"
import math
import torch
x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
y = (x @ x).sum().item()
assert math.isfinite(y), f"bf16 matmul not finite: {y}"
yf = (x.float() @ x.float()).sum().item()
rel = abs(y - yf) / max(abs(yf), 1e-6)
assert rel < 0.05, f"bf16 vs fp32 mismatch: rel={rel}"
print(f"bf16 matmul ok: finite, rel err vs fp32 {rel:.2e}")
PY

log "5. ms-swift"
python -c "import swift; print('swift', swift.__version__)" || fail "ms-swift not importable"

log "6. Model files present"
MODEL_DIR="${JXAGENT_MODEL_DIR:-$HOME/models/Qwen3.8-27B}"
if [ -f "$MODEL_DIR/config.json" ]; then
  log "config found: $MODEL_DIR"
else
  log "model not downloaded yet at $MODEL_DIR (run download_model.sh)"
fi

log "7. Dataset present"
DATA_DIR="${JXAGENT_DATA_DIR:-$HOME/data/JxAgentData}"
if [ -f "$DATA_DIR/final/train.jsonl" ]; then
  log "dataset found: $DATA_DIR"
else
  log "dataset not downloaded yet at $DATA_DIR (run download_dataset.sh)"
fi

log "8. Native Qwen3.8 interface freeze"
if [ -f "$MODEL_DIR/config.json" ]; then
  python "$ROOT_DIR/tools/verify_interface_manifest.py" "$MODEL_DIR" || fail "native interface unresolved or drifted"
else
  log "skipped until model is downloaded"
fi

log "9. Assistant loss-token balance"
if [ -f "$MODEL_DIR/config.json" ] && [ -f "$DATA_DIR/final/train.jsonl" ]; then
  python "$ROOT_DIR/tools/measure_loss_tokens.py" --dataset "$DATA_DIR" --model "$MODEL_DIR" \
    --max-aux-share "${JXAGENT_MAX_AUX_LOSS_SHARE:-0.20}" || fail "loss-token balance gate failed"
else
  log "skipped until model and final dataset are both present"
fi

if [ "$FAILED" -ne 0 ]; then
  printf '[preflight] RESULT: FAILED\n'; exit 1
fi
printf '[preflight] RESULT: PASS\n'
