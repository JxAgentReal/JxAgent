#!/usr/bin/env bash
# Lightweight GPU watcher for MI300X: logs utilization + VRAM every N seconds.
# Usage: gpu_watch.sh [interval_seconds] [log_file]
set -euo pipefail
INTERVAL="${1:-30}"
LOG="${2:-gpu_watch.log}"
echo "timestamp utilization% vram_used_mb vram_total_mb" | tee "$LOG"
while true; do
  TS=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
  if command -v rocm-smi >/dev/null 2>&1; then
    UTIL=$(rocm-smi --showuse --csv 2>/dev/null | awk -F, 'NR==2{print $2}' | tr -d ' %' || echo "?")
    VRAM=$(rocm-smi --showmemuse --csv 2>/dev/null | awk -F, 'NR==2{print $2}' | tr -d ' %' || echo "?")
    echo "$TS $UTIL $VRAM" | tee -a "$LOG"
  else
    python -c "import torch;print('$TS', torch.cuda.utilization(), torch.cuda.memory_allocated()//2**20)" | tee -a "$LOG" || true
  fi
  sleep "$INTERVAL"
done
