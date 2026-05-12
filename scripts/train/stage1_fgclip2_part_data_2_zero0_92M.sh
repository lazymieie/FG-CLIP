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
DENSE_SOURCE="${DENSE_SOURCE:-/gemini/space/datasets/TeleMM-Data/cleaned/pretrain/Caption/DenseFusion-1M.jsonl}"
PT_SOURCE="${PT_SOURCE:-/gemini/space/datasets/TeleMM-Data/cleaned/pretrain/Caption/pt_llava-ov-mid-v1.jsonl}"
DATA_WORK_DIR="${DATA_WORK_DIR:-$ROOT/data/TeleMM}"
DATA_PATH="${DATA_PATH:-$DATA_WORK_DIR/stage1_longonly_92M_manifest.txt}"
IMG_ROOT="${IMG_ROOT:-$ROOT/data}"
LOG_DIR="${LOG_DIR:-$ROOT/output/stage1_siglip2_bs288_so_zero0_92M}"
HOSTFILE="${HOSTFILE:-$ROOT/hostfile}"

MASTER_ADDR="${MASTER_ADDR:-10.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"
NUM_NODES="${NUM_NODES:-2}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
TRAIN_WORLD_SIZE="${TRAIN_WORLD_SIZE:-$((NUM_NODES * NUM_GPUS_PER_NODE))}"

MAX_NUM_PATCHES="${MAX_NUM_PATCHES:-1024}"
ADD_BOX_LOSS="${ADD_BOX_LOSS:-False}"
USE_HARD_NEG="${USE_HARD_NEG:-False}"
USE_LONG_CAPTION="${USE_LONG_CAPTION:-True}"
USE_SHORT_CAPTION="${USE_SHORT_CAPTION:-False}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-50000000}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-288}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

if [[ "$NODE_RANK" != "0" && "$NODE_RANK" != "1" ]]; then
  echo "NODE_RANK must be 0 or 1, got: $NODE_RANK" >&2
  exit 1
fi

if [[ ! -f "$HOSTFILE" ]]; then
  echo "Hostfile not found: $HOSTFILE" >&2
  exit 1
fi

mkdir -p "$LOG_DIR"
mkdir -p "$DATA_WORK_DIR"
cd "$ROOT"

TRAIN_LOG="$LOG_DIR/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$TRAIN_LOG") 2>&1

printf "%s\n%s\n" "$DENSE_SOURCE" "$PT_SOURCE" > "$DATA_PATH"

echo "Starting distributed train with NODE_RANK=$NODE_RANK MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "NCCL_IB_DISABLE=$NCCL_IB_DISABLE NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-unset} NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-unset}"
echo "TRAIN_WORLD_SIZE=$TRAIN_WORLD_SIZE MAX_NUM_PATCHES=$MAX_NUM_PATCHES PER_DEVICE_TRAIN_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE"
echo "DENSE_SOURCE=$DENSE_SOURCE"
echo "PT_SOURCE=$PT_SOURCE"
echo "DATA_PATH=$DATA_PATH"
echo "IMG_ROOT=$IMG_ROOT"
echo "LOG_DIR=$LOG_DIR"
echo "USE_LONG_CAPTION=$USE_LONG_CAPTION"
echo "USE_SHORT_CAPTION=$USE_SHORT_CAPTION"
echo "MAX_IMAGE_PIXELS=$MAX_IMAGE_PIXELS"
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
    --image_folder "$IMG_ROOT" \
    --missing_image_log_path "$LOG_DIR/missing_images_node${NODE_RANK}.jsonl" \
    --large_image_log_path "$LOG_DIR/large_images_node${NODE_RANK}.jsonl" \
    --max_image_pixels "$MAX_IMAGE_PIXELS" \
    --cn_and_en_2_train False \
    --loss_type reduce \
    --from_siglip2 True \
    --naflex_train True \
    --max_num_patches "$MAX_NUM_PATCHES" \
    --output_dir "$LOG_DIR" \
    --train_use_word_size "$TRAIN_WORLD_SIZE" \
    --add_box_loss "$ADD_BOX_LOSS" \
    --use_hard_neg "$USE_HARD_NEG" \
    --box_image_size 512 \
    --base_seq_length 64 \
    --max_seq_length 196 \
    --use_long_caption "$USE_LONG_CAPTION" \
    --use_short_caption "$USE_SHORT_CAPTION" \
    --save_safetensors True \
    --bf16 True \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --num_train_epochs "$NUM_TRAIN_EPOCHS" \
    --save_strategy "steps" \
    --save_steps 358 \
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
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to "none"
