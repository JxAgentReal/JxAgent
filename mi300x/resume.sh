#!/usr/bin/env bash
# Resume full training from the latest checkpoint after an instance failure.
#
# PARITY: sources the same common.sh and builds the IDENTICAL swift sft
# argument list as train.sh (jx_build_swift_args) - model, dataset,
# validation dataset, LoRA modules/rank/alpha/dropout, vision+aligner freeze,
# BF16, max context, per-device batch, gradient accumulation (world-size
# aware), learning rate, cosine scheduler, warmup, epochs, gradient
# checkpointing, attention backend, logging, evaluation and checkpoint
# policy, dataloader settings. The ONLY difference is
# --resume_from_checkpoint <latest>. Nothing relies on ms-swift defaults.
# The offline parity test (tests/test_training_infrastructure.py) fails if
# the two scripts' effective arguments ever diverge.
#
# Resume fidelity requires optimizer/scheduler/RNG state in the checkpoint -
# train.sh deliberately does NOT pass --save_only_model.
set -euo pipefail
export JXAGENT_TAG=${JXAGENT_TAG:-resume}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

MODEL_DIR="${JXAGENT_MODEL_DIR:-$HOME/models/Qwen3.8-27B}"
DATA_DIR="${JXAGENT_DATA_DIR:-$HOME/data/JxAgentData}"
OUT="${JXAGENT_TRAIN_OUT:-./output_run1}"
MODULES_JSON="${JXAGENT_MODULES_JSON:-$HERE/lora_modules.json}"

LATEST=$(jx_find_latest_checkpoint "$OUT")
if [ -z "$LATEST" ]; then
  jx_die "no checkpoint found under $OUT"
fi
jx_log "resuming from $LATEST"

# Identical topology / counts / gate math as train.sh. NOTE: the dataset must
# be the SAME train.jsonl as the original run - the step math and checkpoint
# gates are derived from it.
jx_compute_topology
SAMPLES=$(jx_train_sample_count)
EPOCH_STEPS=$(jx_optimizer_steps "$SAMPLES" "$EFFECTIVE_BATCH")
jx_compute_gates "$EPOCH_STEPS"
jx_print_plan "$SAMPLES" "$EPOCH_STEPS"

jx_build_swift_args

if [ "${JXAGENT_DRY_RUN:-0}" != "1" ] && [ "${JXAGENT_CONFIRM_TRAIN:-0}" != "1" ]; then
  jx_die "export JXAGENT_CONFIRM_TRAIN=1 to resume (same gate as train.sh)"
fi

JXAGENT_LOG_FILE="${JXAGENT_LOG_FILE:-$OUT/resume.log}" jx_run_swift --resume_from_checkpoint "$LATEST"

jx_preserve_gates "$OUT"
jx_log "resume complete: $OUT (gate artifacts: $OUT/gates/{early,middle,final})"
