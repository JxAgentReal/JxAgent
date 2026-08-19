#!/usr/bin/env bash
# Validate the downloaded prepared dataset BEFORE spending GPU hours.
# Fatal on: missing images, invalid JSON, empty targets, out-of-bounds
# coordinates, train/validation trajectory overlap.
set -euo pipefail

DATA_DIR="${JXAGENT_DATA_DIR:-$HOME/data/JxAgentData}"
PY=${PYTHON:-python}
HERE="$(cd "$(dirname "$0")/.." && pwd)"

"$PY" "$HERE/evaluation/validate_actions.py" --dataset-root "$DATA_DIR" --full

echo "[dataset-validate] stats:"
"$PY" - "$DATA_DIR" <<'PY'
import json, os, sys
root = sys.argv[1]
stats = json.load(open(os.path.join(root, "final", "stats.json")))
for k in ("total_samples", "train_samples", "validation_samples",
          "samples_per_source", "fatal_failure", "failures"):
    print(f"  {k}: {stats.get(k)}")
assert stats.get("fatal_failure") is False, "fatal dataset failure"
print("[dataset-validate] PASS")
PY
