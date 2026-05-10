#!/usr/bin/env bash

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"

if [[ -n "${NCCL_IB_GID_INDEX:-}" ]]; then
  export NCCL_IB_GID_INDEX
fi

if [[ -n "${NCCL_SOCKET_IFNAME:-}" ]]; then
  export NCCL_SOCKET_IFNAME
fi

ROOT="${ROOT:-/gemini/space/gjx/FG-CLIP}"
MODEL_DIR="${MODEL_DIR:-$ROOT/siglip2-so400m-patch16-naflex}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
FGCLIP_DEBUG_COLLECTIVES="${FGCLIP_DEBUG_COLLECTIVES:-0}"
FGCLIP_DEBUG_VERBOSE_COLLECTIVES="${FGCLIP_DEBUG_VERBOSE_COLLECTIVES:-0}"
FGCLIP_DEBUG_SAMPLE_LIMIT="${FGCLIP_DEBUG_SAMPLE_LIMIT:-4}"
FGCLIP_DEBUG_RANKS="${FGCLIP_DEBUG_RANKS:-}"

if [[ "$FGCLIP_DEBUG_COLLECTIVES" == "1" ]]; then
  export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
  export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
  export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
  export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-COLL}"
  export FGCLIP_DEBUG_COLLECTIVES
  export FGCLIP_DEBUG_VERBOSE_COLLECTIVES
  export FGCLIP_DEBUG_SAMPLE_LIMIT
  export FGCLIP_DEBUG_RANKS
fi

COYO_SOURCE="${COYO_SOURCE:-/gemini/space/datasets/TeleMM-Data/cleaned/pretrain/Caption/coyo/coyo_balanced_recaptioned_397B_27B_0421.jsonl}"
LLAVA_SOURCE="${LLAVA_SOURCE:-/gemini/space/datasets/TeleMM-Data/cleaned/pretrain/Caption/pt_llava-ov-mid-v1.jsonl}"
DENSEFUSION_SOURCE="${DENSEFUSION_SOURCE:-/gemini/space/datasets/TeleMM-Data/cleaned/pretrain/Caption/DenseFusion-1M.jsonl}"
WUKONG_SOURCE="${WUKONG_SOURCE:-/gemini/space/datasets/TeleMM-Data/cleaned/pretrain/Caption/wukong/20260430}"
ZERO_SOURCE="${ZERO_SOURCE:-/gemini/space/cuixiuqi/work/MMData/data_processed/zero_27B_cleaned}"
FINEHARD_SOURCE="${FINEHARD_SOURCE:-/gemini/space/FG-CLIP/FineHARD/json_files}"

FINEHARD_IMAGE_ROOT="${FINEHARD_IMAGE_ROOT:-/gemini/space/FG-CLIP/data}"
ZERO_IMAGE_ROOT="${ZERO_IMAGE_ROOT:-/gemini/space/datasets/TeleMM-Data/raw/pretrain/Caption/zero}"

DATA_WORK_DIR="${DATA_WORK_DIR:-$ROOT/data/TeleMM}"
DATA_PATH="${DATA_PATH:-$DATA_WORK_DIR/stage1_longonly_multisource_manifest.txt}"
INDEX_CACHE_ROOT="${INDEX_CACHE_ROOT:-$ROOT/data/index_cache}"
HOSTFILE="${HOSTFILE:-$ROOT/hostfile}"

MASTER_ADDR="${MASTER_ADDR:-10.233.113.55}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-4}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
TRAIN_WORLD_SIZE="${TRAIN_WORLD_SIZE:-$((NUM_NODES * NUM_GPUS_PER_NODE))}"

MAX_NUM_PATCHES="${MAX_NUM_PATCHES:-1024}"
ADD_BOX_LOSS="${ADD_BOX_LOSS:-False}"
USE_HARD_NEG="${USE_HARD_NEG:-False}"
USE_SHORT_CAPTION="${USE_SHORT_CAPTION:-False}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-50000000}"
BASE_SEQ_LENGTH="${BASE_SEQ_LENGTH:-64}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-196}"
LOSS_TYPE="${LOSS_TYPE:-reduce}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-288}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-True}"
REPORT_TO="${REPORT_TO:-tensorboard}"
SAVE_STEPS="${SAVE_STEPS:-400}"
DDP_TIMEOUT_SECONDS="${DDP_TIMEOUT_SECONDS:-1800}"
SAMPLE_TIMEOUT_SECONDS="${SAMPLE_TIMEOUT_SECONDS:-20}"
PREPROCESS_TIMEOUT_SECONDS="${PREPROCESS_TIMEOUT_SECONDS:-20}"
MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-$MAX_SEQ_LENGTH}"
TRAIN_SEED="${TRAIN_SEED:-20260506}"
DATA_SEED="${DATA_SEED:-20260506}"
FULL_DETERMINISM="${FULL_DETERMINISM:-False}"

if (( MAX_CAPTION_TOKENS > 0 )); then
  DEFAULT_LOG_DIR="$ROOT/output/stage1_from_siglip2_longonly_302M_filtered_cap${MAX_CAPTION_TOKENS}"
else
  DEFAULT_LOG_DIR="$ROOT/output/stage1_from_siglip2_longonly_302M"
fi
LOG_DIR="${LOG_DIR:-$DEFAULT_LOG_DIR}"

COYO_RECORD_LIMIT="${COYO_RECORD_LIMIT:-}"
COYO_SAMPLE_SEED="${COYO_SAMPLE_SEED:-20260506}"
COYO_INDEX_PATH="${COYO_INDEX_PATH:-$DATA_WORK_DIR/coyo_random${COYO_RECORD_LIMIT:-all}_seed${COYO_SAMPLE_SEED}.idx}"

LLAVA_RECORD_LIMIT="${LLAVA_RECORD_LIMIT:-}"
LLAVA_SAMPLE_SEED="${LLAVA_SAMPLE_SEED:-20260506}"
LLAVA_INDEX_PATH="${LLAVA_INDEX_PATH:-$DATA_WORK_DIR/llava_random${LLAVA_RECORD_LIMIT:-all}_seed${LLAVA_SAMPLE_SEED}.idx}"

DENSEFUSION_RECORD_LIMIT="${DENSEFUSION_RECORD_LIMIT:-}"
DENSEFUSION_SAMPLE_SEED="${DENSEFUSION_SAMPLE_SEED:-20260506}"
DENSEFUSION_INDEX_PATH="${DENSEFUSION_INDEX_PATH:-$DATA_WORK_DIR/densefusion_random${DENSEFUSION_RECORD_LIMIT:-all}_seed${DENSEFUSION_SAMPLE_SEED}.idx}"

if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]]; then
  echo "NODE_RANK must be a non-negative integer, got: $NODE_RANK" >&2
  exit 1
fi

if (( NODE_RANK >= NUM_NODES )); then
  echo "NODE_RANK must be in [0, $((NUM_NODES - 1))], got: $NODE_RANK" >&2
  exit 1
fi

if [[ ! -f "$HOSTFILE" ]]; then
  echo "Hostfile not found: $HOSTFILE" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
mkdir -p "$DATA_WORK_DIR"
mkdir -p "$INDEX_CACHE_ROOT"
cd "$ROOT"

TRAIN_LOG="$LOG_DIR/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
CONFIG_SNAPSHOT="$LOG_DIR/run_config_node${NODE_RANK}.env"
MANIFEST_SNAPSHOT="$LOG_DIR/$(basename "$DATA_PATH").snapshot.txt"
SCRIPT_SNAPSHOT="$LOG_DIR/$(basename "$0").snapshot.sh"
HOSTFILE_SNAPSHOT="$LOG_DIR/$(basename "$HOSTFILE").snapshot"
exec > >(tee -a "$TRAIN_LOG") 2>&1

build_random_index_if_needed() {
  local source="$1"
  local limit="$2"
  local seed="$3"
  local index_path="$4"

  if [[ -z "$limit" ]]; then
    return
  fi

  if [[ "$source" != *.jsonl ]]; then
    echo "Random sampling via index is only supported for a single .jsonl source: $source" >&2
    exit 1
  fi

  local expected_bytes=$((limit * 8))
  if [[ "$NODE_RANK" == "0" ]]; then
    "$PYTHON_BIN" "$ROOT/scripts/build_jsonl_offset_index.py" \
      --jsonl "$source" \
      --output "$index_path" \
      --sample-size "$limit" \
      --seed "$seed"
  else
    echo "Waiting for random index: $index_path"
    while [[ ! -f "$index_path" || "$(stat -c%s "$index_path" 2>/dev/null || echo 0)" != "$expected_bytes" ]]; do
      sleep 10
    done
  fi
}

append_manifest_line() {
  local source="$1"
  local limit="$2"
  local seed="$3"
  local index_path="$4"

  if [[ -n "$limit" ]]; then
    printf "%s\t%s\t%s\t%s\n" "$source" "$limit" "$seed" "$index_path" >> "$DATA_PATH"
  else
    printf "%s\n" "$source" >> "$DATA_PATH"
  fi
}

: > "$DATA_PATH"

build_random_index_if_needed "$COYO_SOURCE" "$COYO_RECORD_LIMIT" "$COYO_SAMPLE_SEED" "$COYO_INDEX_PATH"
build_random_index_if_needed "$LLAVA_SOURCE" "$LLAVA_RECORD_LIMIT" "$LLAVA_SAMPLE_SEED" "$LLAVA_INDEX_PATH"
build_random_index_if_needed "$DENSEFUSION_SOURCE" "$DENSEFUSION_RECORD_LIMIT" "$DENSEFUSION_SAMPLE_SEED" "$DENSEFUSION_INDEX_PATH"

append_manifest_line "$COYO_SOURCE" "$COYO_RECORD_LIMIT" "$COYO_SAMPLE_SEED" "$COYO_INDEX_PATH"
append_manifest_line "$LLAVA_SOURCE" "$LLAVA_RECORD_LIMIT" "$LLAVA_SAMPLE_SEED" "$LLAVA_INDEX_PATH"
append_manifest_line "$DENSEFUSION_SOURCE" "$DENSEFUSION_RECORD_LIMIT" "$DENSEFUSION_SAMPLE_SEED" "$DENSEFUSION_INDEX_PATH"
append_manifest_line "$FINEHARD_SOURCE" "" "" ""
append_manifest_line "$WUKONG_SOURCE" "" "" ""
append_manifest_line "$ZERO_SOURCE" "" "" ""

{
  printf "RUN_TIMESTAMP=%s\n" "$(date --iso-8601=seconds 2>/dev/null || date '+%Y-%m-%dT%H:%M:%S%z')"
  printf "ROOT=%q\n" "$ROOT"
  printf "MODEL_DIR=%q\n" "$MODEL_DIR"
  printf "PYTHON_BIN=%q\n" "$PYTHON_BIN"
  printf "HOSTFILE=%q\n" "$HOSTFILE"
  printf "MASTER_ADDR=%q\n" "$MASTER_ADDR"
  printf "MASTER_PORT=%q\n" "$MASTER_PORT"
  printf "NODE_RANK=%q\n" "$NODE_RANK"
  printf "NUM_NODES=%q\n" "$NUM_NODES"
  printf "NUM_GPUS_PER_NODE=%q\n" "$NUM_GPUS_PER_NODE"
  printf "TRAIN_WORLD_SIZE=%q\n" "$TRAIN_WORLD_SIZE"
  printf "CUDA_VISIBLE_DEVICES=%q\n" "${CUDA_VISIBLE_DEVICES:-}"
  printf "NCCL_IB_DISABLE=%q\n" "${NCCL_IB_DISABLE:-}"
  printf "NCCL_DEBUG=%q\n" "${NCCL_DEBUG:-}"
  printf "NCCL_IB_GID_INDEX=%q\n" "${NCCL_IB_GID_INDEX:-}"
  printf "NCCL_SOCKET_IFNAME=%q\n" "${NCCL_SOCKET_IFNAME:-}"
  printf "MAX_NUM_PATCHES=%q\n" "$MAX_NUM_PATCHES"
  printf "ADD_BOX_LOSS=%q\n" "$ADD_BOX_LOSS"
  printf "USE_HARD_NEG=%q\n" "$USE_HARD_NEG"
  printf "USE_SHORT_CAPTION=%q\n" "$USE_SHORT_CAPTION"
  printf "MAX_IMAGE_PIXELS=%q\n" "$MAX_IMAGE_PIXELS"
  printf "BASE_SEQ_LENGTH=%q\n" "$BASE_SEQ_LENGTH"
  printf "MAX_SEQ_LENGTH=%q\n" "$MAX_SEQ_LENGTH"
  printf "LOSS_TYPE=%q\n" "$LOSS_TYPE"
  printf "PER_DEVICE_TRAIN_BATCH_SIZE=%q\n" "$PER_DEVICE_TRAIN_BATCH_SIZE"
  printf "PER_DEVICE_EVAL_BATCH_SIZE=%q\n" "$PER_DEVICE_EVAL_BATCH_SIZE"
  printf "GRADIENT_ACCUMULATION_STEPS=%q\n" "$GRADIENT_ACCUMULATION_STEPS"
  printf "NUM_TRAIN_EPOCHS=%q\n" "$NUM_TRAIN_EPOCHS"
  printf "GRADIENT_CHECKPOINTING=%q\n" "$GRADIENT_CHECKPOINTING"
  printf "DATALOADER_NUM_WORKERS=%q\n" "$DATALOADER_NUM_WORKERS"
  printf "DATALOADER_PIN_MEMORY=%q\n" "$DATALOADER_PIN_MEMORY"
  printf "REPORT_TO=%q\n" "$REPORT_TO"
  printf "SAVE_STEPS=%q\n" "$SAVE_STEPS"
  printf "DDP_TIMEOUT_SECONDS=%q\n" "$DDP_TIMEOUT_SECONDS"
  printf "FGCLIP_DEBUG_COLLECTIVES=%q\n" "$FGCLIP_DEBUG_COLLECTIVES"
  printf "FGCLIP_DEBUG_VERBOSE_COLLECTIVES=%q\n" "$FGCLIP_DEBUG_VERBOSE_COLLECTIVES"
  printf "FGCLIP_DEBUG_SAMPLE_LIMIT=%q\n" "$FGCLIP_DEBUG_SAMPLE_LIMIT"
  printf "FGCLIP_DEBUG_RANKS=%q\n" "$FGCLIP_DEBUG_RANKS"
  printf "SAMPLE_TIMEOUT_SECONDS=%q\n" "$SAMPLE_TIMEOUT_SECONDS"
  printf "PREPROCESS_TIMEOUT_SECONDS=%q\n" "$PREPROCESS_TIMEOUT_SECONDS"
  printf "MAX_CAPTION_TOKENS=%q\n" "$MAX_CAPTION_TOKENS"
  printf "TRAIN_SEED=%q\n" "$TRAIN_SEED"
  printf "DATA_SEED=%q\n" "$DATA_SEED"
  printf "FULL_DETERMINISM=%q\n" "$FULL_DETERMINISM"
  printf "COYO_SOURCE=%q\n" "$COYO_SOURCE"
  printf "COYO_RECORD_LIMIT=%q\n" "$COYO_RECORD_LIMIT"
  printf "COYO_SAMPLE_SEED=%q\n" "$COYO_SAMPLE_SEED"
  printf "COYO_INDEX_PATH=%q\n" "$COYO_INDEX_PATH"
  printf "LLAVA_SOURCE=%q\n" "$LLAVA_SOURCE"
  printf "LLAVA_RECORD_LIMIT=%q\n" "$LLAVA_RECORD_LIMIT"
  printf "LLAVA_SAMPLE_SEED=%q\n" "$LLAVA_SAMPLE_SEED"
  printf "LLAVA_INDEX_PATH=%q\n" "$LLAVA_INDEX_PATH"
  printf "DENSEFUSION_SOURCE=%q\n" "$DENSEFUSION_SOURCE"
  printf "DENSEFUSION_RECORD_LIMIT=%q\n" "$DENSEFUSION_RECORD_LIMIT"
  printf "DENSEFUSION_SAMPLE_SEED=%q\n" "$DENSEFUSION_SAMPLE_SEED"
  printf "DENSEFUSION_INDEX_PATH=%q\n" "$DENSEFUSION_INDEX_PATH"
  printf "WUKONG_SOURCE=%q\n" "$WUKONG_SOURCE"
  printf "ZERO_SOURCE=%q\n" "$ZERO_SOURCE"
  printf "ZERO_IMAGE_ROOT=%q\n" "$ZERO_IMAGE_ROOT"
  printf "FINEHARD_SOURCE=%q\n" "$FINEHARD_SOURCE"
  printf "FINEHARD_IMAGE_ROOT=%q\n" "$FINEHARD_IMAGE_ROOT"
  printf "DATA_WORK_DIR=%q\n" "$DATA_WORK_DIR"
  printf "DATA_PATH=%q\n" "$DATA_PATH"
  printf "INDEX_CACHE_ROOT=%q\n" "$INDEX_CACHE_ROOT"
  printf "LOG_DIR=%q\n" "$LOG_DIR"
  printf "TRAIN_LOG=%q\n" "$TRAIN_LOG"
} > "$CONFIG_SNAPSHOT"

if [[ "$NODE_RANK" == "0" ]]; then
  cp "$0" "$SCRIPT_SNAPSHOT"
  cp "$HOSTFILE" "$HOSTFILE_SNAPSHOT"
  cp "$DATA_PATH" "$MANIFEST_SNAPSHOT"
fi

echo "Starting distributed train with NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-unset} NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-unset}"
echo "TRAIN_WORLD_SIZE=$TRAIN_WORLD_SIZE MAX_NUM_PATCHES=$MAX_NUM_PATCHES PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE"
echo "DATA_PATH=$DATA_PATH"
echo "INDEX_CACHE_ROOT=$INDEX_CACHE_ROOT"
echo "FINEHARD_IMAGE_ROOT=$FINEHARD_IMAGE_ROOT"
echo "ZERO_IMAGE_ROOT=$ZERO_IMAGE_ROOT"
echo "USE_SHORT_CAPTION=$USE_SHORT_CAPTION"
echo "MAX_IMAGE_PIXELS=$MAX_IMAGE_PIXELS"
echo "LOSS_TYPE=$LOSS_TYPE DATALOADER_PIN_MEMORY=$DATALOADER_PIN_MEMORY DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS"
echo "DDP_TIMEOUT_SECONDS=$DDP_TIMEOUT_SECONDS FGCLIP_DEBUG_COLLECTIVES=$FGCLIP_DEBUG_COLLECTIVES FGCLIP_DEBUG_RANKS=${FGCLIP_DEBUG_RANKS:-all}"
echo "BASE_SEQ_LENGTH=$BASE_SEQ_LENGTH MAX_SEQ_LENGTH=$MAX_SEQ_LENGTH MAX_CAPTION_TOKENS=$MAX_CAPTION_TOKENS"
echo "TRAIN_SEED=$TRAIN_SEED DATA_SEED=$DATA_SEED FULL_DETERMINISM=$FULL_DETERMINISM"
echo "Config snapshot: $CONFIG_SNAPSHOT"
if [[ "$NODE_RANK" == "0" ]]; then
  echo "Script snapshot: $SCRIPT_SNAPSHOT"
  echo "Hostfile snapshot: $HOSTFILE_SNAPSHOT"
  echo "Manifest snapshot: $MANIFEST_SNAPSHOT"
fi
echo "Training log: $TRAIN_LOG"
echo "Manifest:"
cat "$DATA_PATH"

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits \
  -l 2 > "$LOG_DIR/gpu_usage_node${NODE_RANK}.csv" &
MON_PID=$!

cleanup() {
  status=$?
  kill "$MON_PID" 2>/dev/null || true
  echo "Finished at: $(date)"
  echo "Exit status: $status"
  echo "Training log: $TRAIN_LOG"
  exit "$status"
}

trap cleanup EXIT

deepspeed \
  --hostfile "$HOSTFILE" \
  --no_ssh \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  --num_nodes "$NUM_NODES" \
  --num_gpus "$NUM_GPUS_PER_NODE" \
  fgclip2/train/train.py \
    --deepspeed "$ROOT/scripts/zero0.json" \
    --base_model "$MODEL_DIR" \
    --model_name_or_path "$MODEL_DIR" \
    --data_path "$DATA_PATH" \
    --image_folder "$FINEHARD_IMAGE_ROOT" \
    --extra_image_folders "$ZERO_IMAGE_ROOT" \
    --index_cache_root "$INDEX_CACHE_ROOT" \
    --missing_image_log_path "$LOG_DIR/missing_images_node${NODE_RANK}.jsonl" \
    --large_image_log_path "$LOG_DIR/large_images_node${NODE_RANK}.jsonl" \
    --bad_sample_log_path "$LOG_DIR/bad_samples_node${NODE_RANK}.jsonl" \
    --sample_timeout_seconds "$SAMPLE_TIMEOUT_SECONDS" \
    --preprocess_timeout_seconds "$PREPROCESS_TIMEOUT_SECONDS" \
    --max_caption_tokens "$MAX_CAPTION_TOKENS" \
    --max_image_pixels "$MAX_IMAGE_PIXELS" \
    --cn_and_en_2_train False \
    --loss_type "$LOSS_TYPE" \
    --from_siglip2 True \
    --naflex_train True \
    --max_num_patches "$MAX_NUM_PATCHES" \
    --output_dir "$LOG_DIR" \
    --train_use_word_size "$TRAIN_WORLD_SIZE" \
    --add_box_loss "$ADD_BOX_LOSS" \
    --use_hard_neg "$USE_HARD_NEG" \
    --box_image_size 512 \
    --base_seq_length "$BASE_SEQ_LENGTH" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --use_short_caption "$USE_SHORT_CAPTION" \
    --save_safetensors True \
    --bf16 True \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --save_strategy "steps" \
    --save_steps "$SAVE_STEPS" \
    --learning_rate 1e-6 \
    --weight_decay 0.001 \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-6 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing "$GRADIENT_CHECKPOINTING" \
    --seed "$TRAIN_SEED" \
    --data_seed "$DATA_SEED" \
    --ddp_timeout "$DDP_TIMEOUT_SECONDS" \
    --full_determinism "$FULL_DETERMINISM" \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --dataloader_pin_memory "$DATALOADER_PIN_MEMORY" \
    --lazy_preprocess True \
    --bad_sample_log_path "$LOG_DIR/bad_samples_node${NODE_RANK}.jsonl" \
    --report_to "tensorboard"
