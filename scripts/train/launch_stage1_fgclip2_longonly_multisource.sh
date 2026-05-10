#!/usr/bin/env bash

set -euo pipefail

ROOT="${ROOT:-/gemini/space/gjx/FG-CLIP}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$ROOT/scripts/train/stage1_fgclip2_longonly_multisource.sh}"
HOSTFILE="${HOSTFILE:-/gemini/space/gjx/utils/hostfile}"
INDEX_CACHE_ROOT="${INDEX_CACHE_ROOT:-$ROOT/data/index_cache}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-$ROOT/output/stage1_from_siglip2_longonly_302M}"
SSH_KEY="${SSH_KEY:-/gemini/space/gjx/utils/id_rsa}"
SSH_USER="${SSH_USER:-root}"
SSH_OPTS="${SSH_OPTS:--o IdentitiesOnly=yes -o StrictHostKeyChecking=no}"
REMOTE_ACTIVATE_CMD="${REMOTE_ACTIVATE_CMD:-source /opt/miniconda/bin/activate /gemini/space/gjx/miniconda3/envs/fgclip_clean}"

MASTER_PORT="${MASTER_PORT:-29500}"
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-8}"
TRAIN_SEED="${TRAIN_SEED:-20260506}"
DATA_SEED="${DATA_SEED:-20260506}"
FULL_DETERMINISM="${FULL_DETERMINISM:-False}"
LOSS_TYPE="${LOSS_TYPE:-reduce}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-True}"
REPORT_TO="${REPORT_TO:-tensorboard}"
SAVE_STEPS="${SAVE_STEPS:-800}"
DDP_TIMEOUT_SECONDS="${DDP_TIMEOUT_SECONDS:-1800}"
SAMPLE_TIMEOUT_SECONDS="${SAMPLE_TIMEOUT_SECONDS:-20}"
PREPROCESS_TIMEOUT_SECONDS="${PREPROCESS_TIMEOUT_SECONDS:-20}"
FGCLIP_DEBUG_COLLECTIVES="${FGCLIP_DEBUG_COLLECTIVES:-0}"
FGCLIP_DEBUG_VERBOSE_COLLECTIVES="${FGCLIP_DEBUG_VERBOSE_COLLECTIVES:-0}"
FGCLIP_DEBUG_SAMPLE_LIMIT="${FGCLIP_DEBUG_SAMPLE_LIMIT:-4}"
FGCLIP_DEBUG_RANKS="${FGCLIP_DEBUG_RANKS:-}"
LAUNCH_DELAY_SEC="${LAUNCH_DELAY_SEC:-3}"
AUTO_FOLLOW_RANK0_LOG="${AUTO_FOLLOW_RANK0_LOG:-True}"
FOLLOW_POLL_SEC="${FOLLOW_POLL_SEC:-2}"

LAUNCH_LOG_DIR="${LAUNCH_LOG_DIR:-$ROOT/output/stage1_fgclip2_longonly_multisource_launcher}"
LAUNCH_LOG="$LAUNCH_LOG_DIR/launcher_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LAUNCH_LOG_DIR"
exec > >(tee -a "$LAUNCH_LOG") 2>&1

if [[ ! -f "$HOSTFILE" ]]; then
  echo "Hostfile not found: $HOSTFILE" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "Train script not found: $TRAIN_SCRIPT" >&2
  exit 1
fi

if [[ ! -f "$SSH_KEY" ]]; then
  echo "SSH key not found: $SSH_KEY" >&2
  exit 1
fi

mapfile -t HOSTS < <(awk 'NF {print $1}' "$HOSTFILE")

if [[ "${#HOSTS[@]}" -eq 0 ]]; then
  echo "No hosts found in hostfile: $HOSTFILE" >&2
  exit 1
fi

NUM_NODES="${NUM_NODES:-${#HOSTS[@]}}"
if [[ "$NUM_NODES" -ne "${#HOSTS[@]}" ]]; then
  echo "NUM_NODES=$NUM_NODES does not match hostfile entries=${#HOSTS[@]}" >&2
  exit 1
fi

MASTER_ADDR="${MASTER_ADDR:-${HOSTS[0]}}"

echo "Launch log: $LAUNCH_LOG"
echo "ROOT=$ROOT"
echo "TRAIN_SCRIPT=$TRAIN_SCRIPT"
echo "HOSTFILE=$HOSTFILE"
echo "INDEX_CACHE_ROOT=$INDEX_CACHE_ROOT"
echo "TRAIN_LOG_DIR=$TRAIN_LOG_DIR"
echo "SSH_KEY=$SSH_KEY"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "NUM_NODES=$NUM_NODES"
echo "NUM_GPUS_PER_NODE=$NUM_GPUS_PER_NODE"
echo "TRAIN_SEED=$TRAIN_SEED DATA_SEED=$DATA_SEED FULL_DETERMINISM=$FULL_DETERMINISM"
echo "LOSS_TYPE=$LOSS_TYPE DATALOADER_NUM_WORKERS=$DATALOADER_NUM_WORKERS DATALOADER_PIN_MEMORY=$DATALOADER_PIN_MEMORY"
echo "REPORT_TO=$REPORT_TO SAVE_STEPS=$SAVE_STEPS DDP_TIMEOUT_SECONDS=$DDP_TIMEOUT_SECONDS"
echo "FGCLIP_DEBUG_COLLECTIVES=$FGCLIP_DEBUG_COLLECTIVES FGCLIP_DEBUG_VERBOSE_COLLECTIVES=$FGCLIP_DEBUG_VERBOSE_COLLECTIVES"
echo "FGCLIP_DEBUG_SAMPLE_LIMIT=$FGCLIP_DEBUG_SAMPLE_LIMIT FGCLIP_DEBUG_RANKS=${FGCLIP_DEBUG_RANKS:-all}"
echo "AUTO_FOLLOW_RANK0_LOG=$AUTO_FOLLOW_RANK0_LOG"
echo "Hosts:"
printf '  %s\n' "${HOSTS[@]}"

for rank in "${!HOSTS[@]}"; do
  ip="${HOSTS[$rank]}"
  remote_launch_log="$LAUNCH_LOG_DIR/remote_launch_rank${rank}.log"
  remote_pid_file="$LAUNCH_LOG_DIR/remote_launch_rank${rank}.pid"

  echo "===== launching rank=$rank host=$ip ====="
  ssh -i "$SSH_KEY" $SSH_OPTS "$SSH_USER@$ip" /bin/bash <<EOF
set -euo pipefail
$REMOTE_ACTIVATE_CMD
mkdir -p "$LAUNCH_LOG_DIR"
cd "$ROOT"
export ROOT="$ROOT"
export HOSTFILE="$HOSTFILE"
export INDEX_CACHE_ROOT="$INDEX_CACHE_ROOT"
export LOG_DIR="$TRAIN_LOG_DIR"
export MASTER_ADDR="$MASTER_ADDR"
export MASTER_PORT="$MASTER_PORT"
export NODE_RANK="$rank"
export NUM_NODES="$NUM_NODES"
export NUM_GPUS_PER_NODE="$NUM_GPUS_PER_NODE"
export TRAIN_SEED="$TRAIN_SEED"
export DATA_SEED="$DATA_SEED"
export FULL_DETERMINISM="$FULL_DETERMINISM"
export LOSS_TYPE="$LOSS_TYPE"
export DATALOADER_NUM_WORKERS="$DATALOADER_NUM_WORKERS"
export DATALOADER_PIN_MEMORY="$DATALOADER_PIN_MEMORY"
export REPORT_TO="$REPORT_TO"
export SAVE_STEPS="$SAVE_STEPS"
export DDP_TIMEOUT_SECONDS="$DDP_TIMEOUT_SECONDS"
export SAMPLE_TIMEOUT_SECONDS="$SAMPLE_TIMEOUT_SECONDS"
export PREPROCESS_TIMEOUT_SECONDS="$PREPROCESS_TIMEOUT_SECONDS"
export FGCLIP_DEBUG_COLLECTIVES="$FGCLIP_DEBUG_COLLECTIVES"
export FGCLIP_DEBUG_VERBOSE_COLLECTIVES="$FGCLIP_DEBUG_VERBOSE_COLLECTIVES"
export FGCLIP_DEBUG_SAMPLE_LIMIT="$FGCLIP_DEBUG_SAMPLE_LIMIT"
export FGCLIP_DEBUG_RANKS="$FGCLIP_DEBUG_RANKS"
nohup bash "$TRAIN_SCRIPT" > "$remote_launch_log" 2>&1 &
echo \$! > "$remote_pid_file"
echo "HOST=\$(hostname)"
echo "RANK=$rank"
echo "PID=\$(cat "$remote_pid_file")"
echo "LOG=$remote_launch_log"
EOF

  if (( rank + 1 < NUM_NODES )); then
    sleep "$LAUNCH_DELAY_SEC"
  fi
done

echo "All launch commands submitted."
echo "Local launcher log: $LAUNCH_LOG"
echo "Remote launcher logs dir: $LAUNCH_LOG_DIR"

if [[ "$AUTO_FOLLOW_RANK0_LOG" == "True" ]]; then
  rank0_host="${HOSTS[0]}"
  echo "Following rank0 log on $rank0_host ..."
  ssh -i "$SSH_KEY" $SSH_OPTS "$SSH_USER@$rank0_host" /bin/bash <<EOF
set -euo pipefail
$REMOTE_ACTIVATE_CMD
latest_log=""
while [[ -z "\$latest_log" ]]; do
  latest_log=\$(ls -1t "$TRAIN_LOG_DIR"/train_node0_*.log 2>/dev/null | head -n 1 || true)
  if [[ -z "\$latest_log" ]]; then
    sleep "$FOLLOW_POLL_SEC"
  fi
done
echo "FOLLOW_FILE=\$latest_log"
tail -F "\$latest_log"
EOF
fi
