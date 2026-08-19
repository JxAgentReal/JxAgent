#!/usr/bin/env bash
# Shared logic for every JxAgent MI300X training script (sourced, not executed).
#
# 1. World-size aware effective batch. The GLOBAL effective batch is held at
#    ~32 regardless of GPU count:
#        gradient_accumulation = target_effective_batch / (per_device * world_size)
#    Divisibility is validated; when exact 32 is impossible the nearest safe
#    value is chosen and PRINTED - never changed silently.
# 2. Train sample count is read from final/train.jsonl (no hardcoded sizes).
# 3. Checkpoint gates at ~20% / ~55% / 100% of the epoch, preserved in
#    <output>/gates/ after the epoch (rotation is disabled during training).
# 4. ONE canonical swift sft argument list built by jx_build_swift_args and
#    used by BOTH train.sh and resume.sh -> parity by construction.
#    JXAGENT_DRY_RUN=1 prints the full command instead of running anything
#    (used by the offline parity test).
# 5. One attention-backend choice (JXAGENT_ATTN_IMPL, default sdpa) shared by
#    smoke/throughput/train/resume so benchmark and training always match.
#
# Environment overrides:
#   JXAGENT_GPUS                 world size (default: torch.cuda.device_count)
#   JXAGENT_PER_DEVICE_BATCH     per-device batch (default 1)
#   JXAGENT_EFFECTIVE_BATCH      target GLOBAL effective batch (default 32)
#   JXAGENT_TRAIN_SAMPLES        override sample count (tests only; loud warning)
#   JXAGENT_MODULES_JSON         LoRA target registry (default: beside this file)
#   JXAGENT_ATTN_IMPL            attention backend (default sdpa)

JXAGENT_TAG="${JXAGENT_TAG:-jxagent}"

jx_log()  { printf '[%s] %s\n' "$JXAGENT_TAG" "$*"; }
jx_warn() { printf '[%s][WARN] %s\n' "$JXAGENT_TAG" "$*" >&2; }
jx_die()  { printf '[%s][FAIL] %s\n' "$JXAGENT_TAG" "$*" >&2; exit 1; }

jx_here() {
  # Directory of the script that sourced this file.
  local src="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
  cd "$(dirname "$src")" && pwd
}

# ---------------------------------------------------------------- topology --

jx_world_size() {
  if [ -n "${JXAGENT_GPUS:-}" ]; then
    echo "$JXAGENT_GPUS"
    return
  fi
  local ws
  ws=$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || true)
  if [ -z "$ws" ] || [ "$ws" -le 0 ] 2>/dev/null; then
    jx_warn "GPU auto-detection failed; assuming world size 1 (set JXAGENT_GPUS to override)"
    echo 1
    return
  fi
  echo "$ws"
}

# Sets globals: WORLD_SIZE PER_DEVICE_BATCH GRAD_ACCUM EFFECTIVE_BATCH.
jx_compute_topology() {
  WORLD_SIZE=$(jx_world_size)
  PER_DEVICE_BATCH="${JXAGENT_PER_DEVICE_BATCH:-1}"
  local target="${JXAGENT_EFFECTIVE_BATCH:-32}"
  [ "$WORLD_SIZE" -ge 1 ] 2>/dev/null || jx_die "invalid world size '$WORLD_SIZE'"
  [ "$PER_DEVICE_BATCH" -ge 1 ] 2>/dev/null || jx_die "invalid per-device batch '$PER_DEVICE_BATCH'"
  [ "$target" -ge 1 ] 2>/dev/null || jx_die "invalid target effective batch '$target'"

  local denom=$(( PER_DEVICE_BATCH * WORLD_SIZE ))
  if [ "$denom" -gt "$target" ]; then
    GRAD_ACCUM=1
    EFFECTIVE_BATCH=$denom
    jx_warn "world(${WORLD_SIZE}) x per_device(${PER_DEVICE_BATCH}) = ${denom} > target ${target}:"
    jx_warn "  target ${target} unreachable even at gradient_accumulation=1."
    jx_warn "  GLOBAL effective batch is ${EFFECTIVE_BATCH} (NOT ${target})."
    return
  fi
  local floor_g=$(( target / denom ))
  if [ $(( floor_g * denom )) -eq "$target" ]; then
    GRAD_ACCUM=$floor_g
    EFFECTIVE_BATCH=$target
    return
  fi
  # Not divisible: nearest safe value, explicitly announced.
  local ceil_g=$(( floor_g + 1 ))
  local eff_floor=$(( floor_g * denom ))
  local eff_ceil=$(( ceil_g * denom ))
  local d_floor=$(( target - eff_floor ))
  local d_ceil=$(( eff_ceil - target ))
  if [ "$d_ceil" -lt "$d_floor" ]; then
    GRAD_ACCUM=$ceil_g
    EFFECTIVE_BATCH=$eff_ceil
  else
    GRAD_ACCUM=$floor_g
    EFFECTIVE_BATCH=$eff_floor
  fi
  jx_warn "target effective batch ${target} is not divisible by world(${WORLD_SIZE}) x per_device(${PER_DEVICE_BATCH})."
  jx_warn "  chose gradient_accumulation=${GRAD_ACCUM} -> GLOBAL effective batch ${EFFECTIVE_BATCH} (nearest safe value; NOT ${target})."
}

jx_print_plan() {  # $1 = train sample count (or descriptor), $2 = optimizer steps
  jx_log "---- training plan ----"
  jx_log "GPU count (world size)    : ${WORLD_SIZE}"
  jx_log "per device batch          : ${PER_DEVICE_BATCH}"
  jx_log "gradient accumulation     : ${GRAD_ACCUM}"
  jx_log "GLOBAL effective batch    : ${EFFECTIVE_BATCH} (target ${JXAGENT_EFFECTIVE_BATCH:-32})"
  jx_log "train sample count        : ${1:-unknown}"
  jx_log "optimizer steps per epoch : ${2:-unknown}"
}

# ------------------------------------------------------------------ counts --

jx_train_sample_count() {  # echoes the REAL sample count of final/train.jsonl
  if [ -n "${JXAGENT_TRAIN_SAMPLES:-}" ]; then
    jx_warn "JXAGENT_TRAIN_SAMPLES=${JXAGENT_TRAIN_SAMPLES} overrides the real train.jsonl count (intended for tests/dry-runs only)"
    echo "$JXAGENT_TRAIN_SAMPLES"
    return
  fi
  local f="${DATA_DIR:?DATA_DIR not set}/final/train.jsonl"
  [ -f "$f" ] || jx_die "train split not found: $f (set JXAGENT_DATA_DIR)"
  local n
  n=$(grep -c '' "$f") || jx_die "cannot count lines in $f"
  [ "$n" -gt 0 ] || jx_die "$f is empty"
  echo "$n"
}

jx_optimizer_steps() {  # $1 samples, $2 effective batch -> ceil(samples/batch)
  echo $(( ( $1 + $2 - 1 ) / $2 ))
}

# ------------------------------------------------------- checkpoint gates --
# Gate checkpoints are wanted at ~20% / ~55% / 100% of the epoch. save_steps
# is a PERIOD, so we pick SAVE_INTERVAL = total/5 (hits the early gate
# exactly and the middle gate at the nearest multiple) and disable rotation
# with a high save_total_limit; jx_preserve_gates then copies the gate
# checkpoints to <out>/gates/{early,middle,final} and prunes the rest.
jx_compute_gates() {  # $1 total optimizer steps; sets SAVE_INTERVAL EARLY_STEP MID_STEP FINAL_STEP
  local total=$1
  [ "$total" -ge 1 ] || jx_die "total steps must be >= 1"
  SAVE_INTERVAL=$(( total / 5 ))
  [ "$SAVE_INTERVAL" -ge 1 ] || SAVE_INTERVAL=1
  EARLY_STEP=$SAVE_INTERVAL
  local n_saves=$(( total / SAVE_INTERVAL ))
  local mid_target=$(( ( 55 * total + 50 ) / 100 ))       # round(0.55 * total)
  local mid_mult=$(( ( mid_target + SAVE_INTERVAL / 2 ) / SAVE_INTERVAL ))  # nearest multiple
  [ "$mid_mult" -ge 1 ] || mid_mult=1
  [ "$mid_mult" -le "$n_saves" ] || mid_mult=$n_saves
  MID_STEP=$(( mid_mult * SAVE_INTERVAL ))
  FINAL_STEP=$total
}

jx_find_latest_checkpoint() {  # $1 output dir; echoes newest checkpoint dir ("" if none)
  python - "$1" <<'PY'
import glob, os, sys
pats = [os.path.join(sys.argv[1], "v*", "checkpoint-*"),
        os.path.join(sys.argv[1], "checkpoint-*")]
cks = set()
for p in pats:
    cks.update(x for x in glob.glob(p) if os.path.isdir(x))
def key(p):
    try:
        return int(p.rstrip("/\\").rsplit("-", 1)[-1])
    except ValueError:
        return -1
print(max(cks, key=key) if cks else "")
PY
}

# Preserve gate checkpoints and prune non-gate periodic ones. Rotation never
# ran during training (save_total_limit is high), so nothing is lost even if
# the run crashed before this point.
jx_preserve_gates() {  # $1 output dir; uses EARLY_STEP MID_STEP FINAL_STEP
  local out="$1"
  local cks=() steps=() d s i
  while IFS= read -r d; do
    [ -n "$d" ] || continue
    cks+=("$d")
    s="${d##*/checkpoint-}"
    steps+=("$s")
  done < <(find "$out" -maxdepth 2 -type d -name 'checkpoint-*' 2>/dev/null)
  [ "${#cks[@]}" -gt 0 ] || jx_die "gate preservation: no checkpoints found under $out"

  local latest_idx=0
  for i in "${!cks[@]}"; do
    [ "${steps[$i]}" -gt "${steps[$latest_idx]}" ] && latest_idx=$i
  done

  pick_for() {  # $1 target step -> nearest checkpoint dir
    local t=$1 best="" bs="" diff
    for i in "${!cks[@]}"; do
      if [ "${steps[$i]}" -gt "$t" ]; then diff=$(( steps[i] - t )); else diff=$(( t - steps[i] )); fi
      if [ -z "$bs" ] || [ "$diff" -lt "$bs" ]; then best="${cks[$i]}"; bs=$diff; fi
    done
    echo "$best"
  }

  local early mid final
  early=$(pick_for "$EARLY_STEP")
  mid=$(pick_for "$MID_STEP")
  final="${cks[$latest_idx]}"                      # 100% gate = last checkpoint
  mkdir -p "$out/gates"
  local name dst src
  for name in early middle final; do
    case "$name" in early) src=$early ;; middle) src=$mid ;; final) src=$final ;; esac
    [ -n "$src" ] || jx_die "gate preservation: no candidate checkpoint for '$name'"
    dst="$out/gates/$name"
    rm -rf "$dst"
    cp -a "$src" "$dst"
    [ -f "$dst/adapter_model.safetensors" ] || jx_die "gate '$name' has no adapter weights ($dst)"
  done

  if [ "${JXAGENT_KEEP_ALL_CHECKPOINTS:-0}" != "1" ]; then
    for i in "${!cks[@]}"; do
      d="${cks[$i]}"
      [ "$i" = "$latest_idx" ] && continue
      case "$d" in "$early"|"$mid"|"$final") continue ;; esac
      rm -rf "$d"
    done
    jx_log "pruned non-gate periodic checkpoints (kept latest step ${steps[$latest_idx]} + gates/)"
  fi
  jx_log "gate checkpoints preserved under $out/gates: early(~${EARLY_STEP}) middle(~${MID_STEP}) final(${steps[$latest_idx]})"
}

# ------------------------------------------------------- swift invocation --

jx_load_target_regex() {
  [ -f "$MODULES_JSON" ] || jx_die "lora_modules.json missing ($MODULES_JSON). Run: python mi300x/inspect_modules.py \$JXAGENT_MODEL_DIR  (or set JXAGENT_MODULES_JSON)"
  TARGET_REGEX=$(python - "$MODULES_JSON" <<'PY' || jx_die "cannot read $MODULES_JSON"
import json, sys
mods = json.load(open(sys.argv[1]))["target_modules"]
assert isinstance(mods, list) and len(mods) == 1 and isinstance(mods[0], str), \
    "expected target_modules to be a single regex string produced by inspect_modules.py"
print(mods[0])
PY
)
}

jx_resolve_attn_args() {
  # Same backend choice for smoke / throughput / train / resume.
  # Default sdpa: PyTorch SDPA works on ROCm without any CUDA-only wheel.
  local impl="${JXAGENT_ATTN_IMPL:-sdpa}"
  ATTN_IMPL="$impl"
  ATTN_ARGS=()
  if ! command -v swift >/dev/null 2>&1; then
    jx_warn "swift CLI not found (offline/dry-run): assuming --attn_implementation ${impl} is accepted"
    ATTN_ARGS=(--attn_implementation "$impl")
    return
  fi
  if swift sft --help 2>&1 | grep -q -- '--attn_implementation'; then
    ATTN_ARGS=(--attn_implementation "$impl")
  else
    jx_warn "installed ms-swift has no --attn_implementation flag; attention backend left to the model default (unpinned)"
  fi
}

jx_resolve_eval_flag() {
  # transformers renamed evaluation_strategy -> eval_strategy; use what the
  # installed version actually defines.
  EVAL_FLAG=$(python - <<'PY' 2>/dev/null || true
try:
    from transformers import TrainingArguments
    print("eval_strategy" if "eval_strategy" in TrainingArguments.__dataclass_fields__
          else "evaluation_strategy")
except Exception:
    print("")
PY
)
  if [ -z "$EVAL_FLAG" ]; then
    EVAL_FLAG=evaluation_strategy
    jx_warn "transformers not importable; defaulting to --${EVAL_FLAG} flag spelling"
  fi
}

# Canonical argument list shared by train.sh and resume.sh (parity by
# construction). NOTE: --save_only_model is deliberately NOT set - faithful
# resume requires optimizer/scheduler/RNG state inside every checkpoint.
jx_build_swift_args() {
  jx_load_target_regex
  jx_resolve_attn_args
  jx_resolve_eval_flag
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
    --loss_scale default
    "${ATTN_ARGS[@]}"
    --dataset "$DATA_DIR/final/train.jsonl"
    --val_dataset "$DATA_DIR/final/validation.jsonl"
    --max_length "${JXAGENT_MAX_LENGTH:-8192}"
    --num_train_epochs "${JXAGENT_NUM_TRAIN_EPOCHS:-1}"
    --per_device_train_batch_size "$PER_DEVICE_BATCH"
    --per_device_eval_batch_size "${JXAGENT_PER_DEVICE_EVAL_BATCH_SIZE:-1}"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --learning_rate "${JXAGENT_LEARNING_RATE:-1e-5}"
    --lr_scheduler_type "${JXAGENT_LR_SCHEDULER:-cosine}"
    --warmup_ratio "${JXAGENT_WARMUP_RATIO:-0.03}"
    --gradient_checkpointing true
    --save_steps "$SAVE_INTERVAL"
    --save_total_limit "${JXAGENT_SAVE_TOTAL_LIMIT:-999}"
    "--${EVAL_FLAG}" steps
    --eval_steps "$SAVE_INTERVAL"
    --logging_steps "${JXAGENT_LOGGING_STEPS:-20}"
    --dataloader_num_workers "${JXAGENT_DATALOADER_WORKERS:-4}"
    --output_dir "$OUT"
  )
}

jx_print_cmd() {  # prints the exact command that would run
  printf 'NPROC_PER_NODE=%q swift sft' "$WORLD_SIZE"
  printf ' %q' "${SWIFT_ARGS[@]}" "$@"
  printf '\n'
}

jx_run_swift() {  # extra args (e.g. --resume_from_checkpoint) appended
  if [ "${JXAGENT_DRY_RUN:-0}" = "1" ]; then
    jx_print_cmd "$@"
    exit 0
  fi
  local project_root
  project_root="$(cd "$(jx_here)/.." && pwd)"
  python "$project_root/tools/verify_interface_manifest.py" "$MODEL_DIR" \
    || jx_die "native Qwen3.8 interface freeze gate failed"
  python "$project_root/tools/measure_loss_tokens.py" --dataset "$DATA_DIR" --model "$MODEL_DIR" \
    --max-aux-share "${JXAGENT_MAX_AUX_LOSS_SHARE:-0.20}" \
    || jx_die "assistant loss-token balance gate failed"
  mkdir -p "$OUT"
  local log="${JXAGENT_LOG_FILE:-$OUT/train.log}"
  jx_log "log: $log"
  local rc=0
  PYTHONUNBUFFERED=1 NPROC_PER_NODE="$WORLD_SIZE" swift sft "${SWIFT_ARGS[@]}" "$@" 2>&1 | tee -a "$log" || rc=$?
  [ "$rc" -eq 0 ] || jx_die "swift sft exited ${rc}"
}
