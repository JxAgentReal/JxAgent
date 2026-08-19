#!/usr/bin/env bash
# 3-step LoRA smoke gate. Uses the SAME world-size-aware topology as full
# training. Verifies, via verify_freeze.py (full module paths, not
# substrings):
#   - trainable parameters: zero in the vision encoder, zero in the aligner,
#     and exactly the inspected LM modules (checked BEFORE training)
#   - after training: base vision/aligner tensors byte-identical to the
#     pre-training snapshot
#   - at least one LM LoRA tensor actually updated (lora_B non-zero)
#   - checkpoint contains optimizer.pt + scheduler.pt (faithful resume)
# Does NOT continue to full training automatically.
set -euo pipefail
export JXAGENT_TAG=${JXAGENT_TAG:-smoke}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HERE/common.sh"

MODEL_DIR="${JXAGENT_MODEL_DIR:-$HOME/models/Qwen3.8-27B}"
DATA_DIR="${JXAGENT_DATA_DIR:-$HOME/data/JxAgentData}"
OUT="${JXAGENT_SMOKE_OUT:-./output_smoke}"
MODULES_JSON="${JXAGENT_MODULES_JSON:-$HERE/lora_modules.json}"
SNAPSHOT="$OUT/freeze_snapshot.json"

jx_compute_topology
jx_print_plan "smoke (3 optimizer steps max)" 3

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
  --max_steps 3
  --per_device_train_batch_size "$PER_DEVICE_BATCH"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --learning_rate "${JXAGENT_LEARNING_RATE:-1e-5}"
  --gradient_checkpointing true
  --save_steps 3
  --save_total_limit 2
  --logging_steps 1
  --output_dir "$OUT"
)

mkdir -p "$OUT"

echo "[smoke] freeze snapshot BEFORE training (full-path trainable-parameter check)"
python "$HERE/verify_freeze.py" snapshot \
  --model "$MODEL_DIR" --modules "$MODULES_JSON" --out "$SNAPSHOT" \
  --rank "${JXAGENT_LORA_RANK:-32}" --alpha "${JXAGENT_LORA_ALPHA:-64}" \
  --device "${JXAGENT_VERIFY_DEVICE:-cpu}"

JXAGENT_LOG_FILE="$OUT/smoke.log" jx_run_swift

CKPT=$(jx_find_latest_checkpoint "$OUT")
[ -n "$CKPT" ] || jx_die "no checkpoint saved under $OUT"
echo "[smoke] verifying checkpoint: $CKPT"

python "$HERE/verify_freeze.py" verify \
  --model "$MODEL_DIR" --modules "$MODULES_JSON" \
  --snapshot "$SNAPSHOT" --checkpoint "$CKPT" \
  --device "${JXAGENT_VERIFY_DEVICE:-cpu}"

echo "[smoke] DONE - do NOT start full training automatically; run throughput_test.sh next"
