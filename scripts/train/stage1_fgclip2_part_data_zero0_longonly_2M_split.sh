#!/usr/bin/env bash
set -o pipefail

export NCCL_IB_GID_INDEX=5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONFAULTHANDLER="${PYTHONFAULTHANDLER:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

ROOT="/gemini/space/gjx/FG-CLIP"
MODEL_DIR="$ROOT/siglip2-so400m-patch16-naflex"
DENSE_SOURCE="/gemini/space/datasets/TeleMM-Data/cleaned/pretrain/Caption/DenseFusion-1M.jsonl"
PT_SOURCE="/gemini/space/datasets/TeleMM-Data/cleaned/pretrain/Caption/pt_llava-ov-mid-v1.jsonl"
DATA_WORK_DIR="$ROOT/data/TeleMM"
PT_SAMPLE_SIZE="${PT_SAMPLE_SIZE:-1000000}"
PT_SAMPLE_PATH="${PT_SAMPLE_PATH:-$DATA_WORK_DIR/pt_llava-ov-mid-v1_sample${PT_SAMPLE_SIZE}.jsonl}"
DATA_PATH="${DATA_PATH:-$DATA_WORK_DIR/stage1_longonly_2M_manifest.txt}"
IMG_ROOT="${IMG_ROOT:-$ROOT/data}"
LOG_DIR="${LOG_DIR:-$ROOT/output/stage1_split_siglip2_bs288_so_zero0_longonly_2M}"
USE_SHORT_CAPTION="${USE_SHORT_CAPTION:-False}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-50000000}"

mkdir -p "$LOG_DIR"
mkdir -p "$DATA_WORK_DIR"
cd "$ROOT"

TRAIN_LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$TRAIN_LOG") 2>&1

echo "Training log: $TRAIN_LOG"
echo "Started at: $(date)"
echo "ROOT=$ROOT"
echo "MODEL_DIR=$MODEL_DIR"
echo "DENSE_SOURCE=$DENSE_SOURCE"
echo "PT_SOURCE=$PT_SOURCE"
echo "PT_SAMPLE_PATH=$PT_SAMPLE_PATH"
echo "DATA_PATH=$DATA_PATH"
echo "LOG_DIR=$LOG_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "USE_SHORT_CAPTION=$USE_SHORT_CAPTION"
echo "MAX_IMAGE_PIXELS=$MAX_IMAGE_PIXELS"

if [[ ! -f "$PT_SAMPLE_PATH" ]]; then
    echo "Sampling $PT_SAMPLE_SIZE rows from $PT_SOURCE"
    shuf -n "$PT_SAMPLE_SIZE" "$PT_SOURCE" > "$PT_SAMPLE_PATH"
else
    echo "Reusing existing sampled file: $PT_SAMPLE_PATH"
fi

printf "%s\n%s\n" "$DENSE_SOURCE" "$PT_SAMPLE_PATH" > "$DATA_PATH"
echo "Manifest:"
cat "$DATA_PATH"

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

deepspeed --num_gpus 8 fgclip2/train/train_stage1.py \
    --deepspeed "$ROOT/scripts/zero0.json" \
    --base_model "$MODEL_DIR" \
    --model_name_or_path "$MODEL_DIR" \
    --data_path "$DATA_PATH" \
    --image_folder "$IMG_ROOT" \
    --missing_image_log_path "$LOG_DIR/missing_images.jsonl" \
    --large_image_log_path "$LOG_DIR/large_images.jsonl" \
    --max_image_pixels "$MAX_IMAGE_PIXELS" \
    --cn_and_en_2_train False \
    --loss_type reduce \
    --from_siglip2 True \
    --naflex_train True \
    --max_num_patches 1024 \
    --output_dir "$LOG_DIR" \
    --train_use_word_size 8 \
    --box_image_size 512 \
    --base_seq_length 64 \
    --max_seq_length 196 \
    --use_short_caption "$USE_SHORT_CAPTION" \
    --save_safetensors True \
    --bf16 True \
    --per_device_train_batch_size 288 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 1 \
    --save_strategy "steps" \
    --save_steps 10 \
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
    --dataloader_num_workers 4 \
    --dataloader_pin_memory True \
    --lazy_preprocess True \
    --report_to "none"
