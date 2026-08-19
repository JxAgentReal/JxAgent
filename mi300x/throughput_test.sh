#!/usr/bin/env bash
# Steady-state throughput benchmark. Uses the SAME topology (GPU count,
# world-size-aware effective batch), max context, dataset, LoRA config,
# gradient checkpointing and attention backend as the real run.
#
# Timing excludes model download/loading and dataset preprocessing by
# construction: log_tap.py timestamps each Trainer step log record ON ARRIVAL,
# and the stats only measure the window of MEASURE steps that starts AFTER
# WARMUP completed steps. Full training must be launched manually via
# train.sh; the epoch estimate uses the REAL train.jsonl sample count.
set -euo pipefail
export JXAGENT_TAG=${JXAGENT_TAG:-throughput}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

MODEL_DIR="${JXAGENT_MODEL_DIR:-$HOME/models/Qwen3.8-27B}"
DATA_DIR="${JXAGENT_DATA_DIR:-$HOME/data/JxAgentData}"
OUT="${JXAGENT_THROUGHPUT_OUT:-./output_throughput}"
MODULES_JSON="${JXAGENT_MODULES_JSON:-$HERE/lora_modules.json}"
BUDGET_HOURS="${JXAGENT_GPU_BUDGET_HOURS:-35}"
RESERVE_HOURS="${JXAGENT_RESERVE_HOURS:-6}"
WARMUP="${JXAGENT_WARMUP_STEPS:-10}"
MEASURE="${JXAGENT_MEASURE_STEPS:-100}"
TOTAL_STEPS=$(( WARMUP + MEASURE ))

jx_compute_topology
SAMPLES=$(jx_train_sample_count)
EPOCH_STEPS=$(jx_optimizer_steps "$SAMPLES" "$EFFECTIVE_BATCH")
jx_print_plan "$SAMPLES" "$EPOCH_STEPS"
jx_log "benchmark: ${WARMUP} warmup steps + ${MEASURE} measured steps (total ${TOTAL_STEPS})"

jx_load_target_regex
jx_resolve_attn_args

SWIFT_ARGS=(
  --model "$MODEL_DIR"
  --train_type lora
  --trust_remote_code true
  --lora_rank "${JXAGENT_LORA_RANK:-32}"
  --lora_alpha "${JXAGENT_LORA_ALPHA:-64}"
  --lora_dropout "${JXAGENT_LORA_DROPOUT:-0}"
  --target_modules "$TARGET_REGEX"
  --freeze_vit true
  --freeze_aligner true
  --torch_dtype bfloat16
  "${ATTN_ARGS[@]}"
  --dataset "$DATA_DIR/final/train.jsonl"
  --max_length "${JXAGENT_MAX_LENGTH:-8192}"
  --max_steps "$TOTAL_STEPS"
  --per_device_train_batch_size "$PER_DEVICE_BATCH"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --learning_rate "${JXAGENT_LEARNING_RATE:-1e-5}"
  --gradient_checkpointing true
  --save_steps $(( TOTAL_STEPS + 1 ))
  --save_total_limit 1
  --logging_steps 1
  --dataloader_num_workers "${JXAGENT_DATALOADER_WORKERS:-4}"
  --output_dir "$OUT"
)

if [ "${JXAGENT_DRY_RUN:-0}" = "1" ]; then
  jx_print_cmd
  exit 0
fi

mkdir -p "$OUT"
RAW_LOG="$OUT/throughput.log"
EVENTS="$OUT/step_times.jsonl"
VRAM_LOG="$OUT/rocm_smi_samples.log"

# Best-effort VRAM / GPU-utilization sampler (5 s cadence).
SAMPLER_PID=""
if command -v rocm-smi >/dev/null 2>&1; then
  (
    while :; do
      date +%s
      rocm-smi --showmemuse --showusegfx --csv 2>/dev/null
      sleep 5
    done
  ) > "$VRAM_LOG" 2>&1 &
  SAMPLER_PID=$!
else
  jx_warn "rocm-smi not found: VRAM/GPU-utilization peaks will not be sampled"
fi

cleanup() { [ -n "$SAMPLER_PID" ] && kill "$SAMPLER_PID" 2>/dev/null || true; }
trap cleanup EXIT

jx_log "log: $RAW_LOG"
RC=0
PYTHONUNBUFFERED=1 NPROC_PER_NODE="$WORLD_SIZE" swift sft "${SWIFT_ARGS[@]}" 2>&1 \
  | python "$HERE/log_tap.py" tap --raw "$RAW_LOG" --events "$EVENTS" || RC=$?
[ "$RC" -eq 0 ] || jx_die "swift sft exited ${RC} (see $RAW_LOG)"

VRAM_ARG=()
[ -s "$VRAM_LOG" ] && VRAM_ARG=(--vram "$VRAM_LOG")

python "$HERE/log_tap.py" stats \
  --events "$EVENTS" \
  --warmup "$WARMUP" --measure "$MEASURE" \
  --world "$WORLD_SIZE" --eff-batch "$EFFECTIVE_BATCH" \
  --samples "$SAMPLES" --epoch-steps "$EPOCH_STEPS" \
  --budget "$BUDGET_HOURS" --reserve "$RESERVE_HOURS" \
  ${JXAGENT_MEAN_TOKENS_PER_SAMPLE:+--mean-tokens "$JXAGENT_MEAN_TOKENS_PER_SAMPLE"} \
  "${VRAM_ARG[@]}"

echo "[throughput] DONE - full training NOT started automatically"
