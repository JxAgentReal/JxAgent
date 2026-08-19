#!/usr/bin/env bash
# Install the training stack on the AMD cloud instance WITHOUT ever
# replacing a working ROCm PyTorch with a CUDA wheel.
#
# Strategy: if the preinstalled ROCm torch is healthy (HIP build + devices
# visible), its exact version is passed to pip as a CONSTRAINT for every
# later install, so the resolver can never swap it. If ms-swift ever needs a
# newer torch, pip fails LOUDLY instead of silently installing a CUDA build;
# upgrade the ROCm wheel explicitly from the rocm index in that case.
# DeepSpeed is intentionally NOT installed: it is not used by any JxAgent
# script and frequently fails to build on ROCm.
set -euo pipefail

PY=${PYTHON:-python}

echo "[setup] torch health check"
TORCH_OK=0
TORCH_VER=""
if $PY - <<'PY' 2>/dev/null
import torch
assert torch.version.hip is not None, "not a ROCm/HIP build"
assert torch.cuda.is_available(), "no ROCm devices visible"
assert torch.cuda.device_count() >= 1
PY
then
  TORCH_OK=1
  TORCH_VER=$($PY -c 'import torch; print(torch.__version__)')
  echo "[setup] preinstalled ROCm torch ${TORCH_VER} is healthy - preserving it"
else
  echo "[setup] no healthy ROCm torch; installing from the rocm6.3 wheel index"
  $PY -m pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
  TORCH_VER=$($PY -c 'import torch; print(torch.__version__)')
fi

TORCH_PIN="torch==${TORCH_VER}"
echo "[setup] torch constraint for all installs: ${TORCH_PIN}"

echo "[setup] ms-swift + deps (torch pinned, no deepspeed)"
$PY -m pip install -U ms-swift transformers accelerate datasets peft \
  qwen-vl-utils pillow "$TORCH_PIN" || {
  cat >&2 <<MSG
[setup][FAIL] dependency resolution failed. Most likely cause: ms-swift
requires a newer torch than the working ROCm build ${TORCH_VER}.
Do NOT let pip replace it with a CUDA wheel - upgrade the ROCm wheel
explicitly instead:
  $PY -m pip install -U torch --index-url https://download.pytorch.org/whl/rocm6.3
then re-run setup.sh.
MSG
  exit 1
}

echo "[setup] post-install verification"
$PY - <<'PY'
import math
import torch
assert torch.version.hip is not None, "torch is not a ROCm/HIP build after install"
assert torch.cuda.is_available(), "torch lost sight of the ROCm devices after install"
print("torch      :", torch.__version__, "| hip:", torch.version.hip)
print("devices    :", torch.cuda.device_count(), "| device 0:", torch.cuda.get_device_name(0))
x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
y = (x @ x).sum().item()
yf = (x.float() @ x.float()).sum().item()
assert math.isfinite(y), f"bf16 matmul not finite: {y}"
rel = abs(y - yf) / max(abs(yf), 1e-6)
assert rel < 0.05, f"bf16 vs fp32 mismatch: rel={rel}"
print(f"bf16 matmul: finite, rel err vs fp32 {rel:.2e}")
import swift, transformers, peft, accelerate
print("swift      :", swift.__version__)
print("transformers:", transformers.__version__)
print("peft       :", peft.__version__)
PY

if [ "$TORCH_OK" = "1" ]; then
  NOW=$($PY -c 'import torch; print(torch.__version__)')
  [ "$NOW" = "$TORCH_VER" ] || { echo "[setup][FAIL] torch was replaced during install (${TORCH_VER} -> ${NOW})" >&2; exit 1; }
  echo "[setup] torch unchanged: ${NOW}"
fi

echo "[setup] done. Run preflight.sh next."
