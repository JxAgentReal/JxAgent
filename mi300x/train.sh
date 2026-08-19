#!/usr/bin/env bash
# Full Run 1 LoRA epoch.
#
# SAFETY: refuses to run unless JXAGENT_CONFIRM_TRAIN=1 is exported (never
# auto-started by another script). JXAGENT_DRY_RUN=1 prints the exact command
# without running anything (used by the offline parity test).
#
# Topology: the GLOBAL effective batch is held at ~32 on ANY GPU count
# (gradient_accumulation = 32 / (per_device x world_size), divisibility
# validated; nearest safe value announced if 32 is unreachable).
#
# Checkpoints: save interval = optimizer_steps/5 (hits ~20% exactly, ~55% at
# the nearest multiple, 100% at the end). Rotation is DISABLED during
# training (high save_total_limit) so a crash can never delete a gate; after
# the epoch the gate checkpoints are copied to <out>/gates/{early,middle,final}
# and the remaining periodic checkpoints are pruned.
#
# Resume fidelity: --save_only_model is NOT set, so every checkpoint carries
# optimizer/scheduler/RNG state for resume.sh.
set -euo pipefail
export JXAGENT_TAG=${JXAGENT_TAG:-train}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

MODEL_DIR="${JXAGENT_MODEL_DIR:-$HOME/models/Qwen3.8-27B}"
DATA_DIR="${JXAGENT_DATA_DIR:-$HOME/data/JxAgentData}"
OUT="${JXAGENT_TRAIN_OUT:-./output_run1}"
MODULES_JSON="${JXAGENT_MODULES_JSON:-$HERE/lora_modules.json}"

jx_compute_topology
SAMPLES=$(jx_train_sample_count)
EPOCH_STEPS=$(jx_optimizer_steps "$SAMPLES" "$EFFECTIVE_BATCH")
jx_compute_gates "$EPOCH_STEPS"
jx_print_plan "$SAMPLES" "$EPOCH_STEPS"
jx_log "checkpoints: interval=${SAVE_INTERVAL} gates: early~${EARLY_STEP} middle~${MID_STEP} final~${FINAL_STEP} (preserved to gates/ after the epoch)"

jx_build_swift_args

if [ "${JXAGENT_DRY_RUN:-0}" != "1" ] && [ "${JXAGENT_CONFIRM_TRAIN:-0}" != "1" ]; then
  cat >&2 <<MSG
[train] Full training is gated on purpose.
  1. smoke_train.sh must have passed
  2. throughput_test.sh must have printed an acceptable epoch estimate
  3. export JXAGENT_CONFIRM_TRAIN=1 to actually start
MSG
  exit 1
fi

jx_run_swift

jx_preserve_gates "$OUT"
jx_log "epoch complete: $OUT (gate artifacts: $OUT/gates/{early,middle,final})"
