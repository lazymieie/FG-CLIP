#!/usr/bin/env bash
set -euo pipefail
set -o pipefail

MODE="${MODE:?MODE must be one of: short, long, both}"

case "$MODE" in
  short)
    MODE_LABEL="short_only"
    DEFAULT_USE_LONG_CAPTION="False"
    DEFAULT_USE_SHORT_CAPTION="True"
    ;;
  long)
    MODE_LABEL="long_only"
    DEFAULT_USE_LONG_CAPTION="True"
    DEFAULT_USE_SHORT_CAPTION="False"
    ;;
  both)
    MODE_LABEL="long_short"
    DEFAULT_USE_LONG_CAPTION="True"
    DEFAULT_USE_SHORT_CAPTION="True"
    ;;
  *)
    echo "Unsupported MODE: $MODE" >&2
    exit 1
    ;;
esac

export NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-5}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

ROOT="${ROOT:-/gemini/space/gjx/FG-CLIP}"
MODEL_DIR="${MODEL_DIR:-$ROOT/siglip2-so400m-patch16-naflex}"
COYO_SOURCE="${COYO_SOURCE:-/gemini/space/datasets/TeleMM-Data/cleaned/pretrain/ShortCaption/cleaned/coyo_filtered.jsonl}"
INPUT_DATA_PATH="${INPUT_DATA_PATH:-$COYO_SOURCE}"
IMG_ROOT="${IMG_ROOT:-$ROOT/data}"
LOG_DIR="${LOG_DIR:-$ROOT/output/coyo_caption_validation_${MODE_LABEL}_full}"

USE_LONG_CAPTION="${USE_LONG_CAPTION:-$DEFAULT_USE_LONG_CAPTION}"
USE_SHORT_CAPTION="${USE_SHORT_CAPTION:-$DEFAULT_USE_SHORT_CAPTION}"
LONG_CAPTION_FIELD="${LONG_CAPTION_FIELD:-long_caption}"
SHORT_CAPTION_FIELD="${SHORT_CAPTION_FIELD:-messages}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-50000000}"
MAX_NUM_PATCHES="${MAX_NUM_PATCHES:-1024}"
BASE_SEQ_LENGTH="${BASE_SEQ_LENGTH:-64}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-196}"
if [[ "$MODE" == "short" ]]; then
  DEFAULT_MAX_CAPTION_TOKENS="0"
else
  DEFAULT_MAX_CAPTION_TOKENS="$MAX_SEQ_LENGTH"
fi
MAX_CAPTION_TOKENS="${MAX_CAPTION_TOKENS:-$DEFAULT_MAX_CAPTION_TOKENS}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-288}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
SAVE_STEPS="${SAVE_STEPS:-10}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"

mkdir -p "$LOG_DIR"
cd "$ROOT"

TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$TRAIN_LOG") 2>&1

echo "Training log: $TRAIN_LOG"
echo "Started at: $(date)"
echo "MODE=$MODE"
echo "MODE_LABEL=$MODE_LABEL"
echo "ROOT=$ROOT"
echo "MODEL_DIR=$MODEL_DIR"
echo "COYO_SOURCE=$COYO_SOURCE"
echo "INPUT_DATA_PATH=$INPUT_DATA_PATH"
echo "LOG_DIR=$LOG_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "USE_LONG_CAPTION=$USE_LONG_CAPTION"
echo "USE_SHORT_CAPTION=$USE_SHORT_CAPTION"
echo "LONG_CAPTION_FIELD=$LONG_CAPTION_FIELD"
echo "SHORT_CAPTION_FIELD=$SHORT_CAPTION_FIELD"
echo "MAX_CAPTION_TOKENS=$MAX_CAPTION_TOKENS"
echo "MAX_IMAGE_PIXELS=$MAX_IMAGE_PIXELS"

if [[ ! -f "$INPUT_DATA_PATH" ]]; then
    echo "Input data not found: $INPUT_DATA_PATH" >&2
    exit 1
fi

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv,noheader,nounits \
  -l 2 > "$LOG_DIR/gpu_usage.csv" &
MON_PID=$!

cleanup() {
    status=$?
    kill $MON_PID 2>/dev/null
    echo "Finished at: $(date)"
    echo "Exit status: $status"
    echo "Training log: $TRAIN_LOG"
    exit $status
}

trap cleanup EXIT

deepspeed --num_gpus 8 fgclip2/train/train.py \
    --deepspeed "$ROOT/scripts/zero0.json" \
    --base_model "$MODEL_DIR" \
    --model_name_or_path "$MODEL_DIR" \
    --data_path "$INPUT_DATA_PATH" \
    --image_folder "$IMG_ROOT" \
    --missing_image_log_path "$LOG_DIR/missing_images.jsonl" \
    --large_image_log_path "$LOG_DIR/large_images.jsonl" \
    --max_image_pixels "$MAX_IMAGE_PIXELS" \
    --max_caption_tokens "$MAX_CAPTION_TOKENS" \
    --cn_and_en_2_train False \
    --loss_type reduce \
    --from_siglip2 True \
    --naflex_train True \
    --max_num_patches "$MAX_NUM_PATCHES" \
    --output_dir "$LOG_DIR" \
    --train_use_word_size 8 \
    --add_box_loss False \
    --use_hard_neg False \
    --box_image_size 512 \
    --base_seq_length "$BASE_SEQ_LENGTH" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --use_long_caption "$USE_LONG_CAPTION" \
    --long_caption_field "$LONG_CAPTION_FIELD" \
    --use_short_caption "$USE_SHORT_CAPTION" \
    --short_caption_field "$SHORT_CAPTION_FIELD" \
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
    --gradient_checkpointing True \
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS" \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to "none"
