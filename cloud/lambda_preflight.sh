#!/usr/bin/env bash
# Bounded, read-only preflight for a Lambda/NVIDIA GPU instance.
# It does not download model weights, datasets, or start training.
set -euo pipefail

echo '== JxAgent Lambda/NVIDIA preflight =='
command -v nvidia-smi >/dev/null || { echo '[FAIL] nvidia-smi not found' >&2; exit 2; }
command -v python >/dev/null || { echo '[FAIL] python not found' >&2; exit 2; }
command -v git >/dev/null || { echo '[FAIL] git not found' >&2; exit 2; }

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python - <<'PY'
import sys
print('python:', sys.version.split()[0])
try:
    import torch
except Exception as e:
    raise SystemExit(f'[FAIL] torch import: {e}')
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda runtime:', torch.version.cuda)
print('gpu count:', torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit('[FAIL] CUDA GPU unavailable to PyTorch')
for i in range(torch.cuda.device_count()):
    p=torch.cuda.get_device_properties(i)
    print(f'gpu[{i}]: {p.name}, {p.total_memory/2**30:.1f} GiB')
    try:
        x=torch.ones((1024,), device=f'cuda:{i}', dtype=torch.bfloat16)
        y=x+x
        torch.cuda.synchronize(i)
        assert y.dtype == torch.bfloat16
        print(f'gpu[{i}] bf16 smoke: PASS')
    except Exception as e:
        raise SystemExit(f'[FAIL] gpu[{i}] bf16 smoke: {e}')
PY

df -h .
echo '[preflight] PASS'
