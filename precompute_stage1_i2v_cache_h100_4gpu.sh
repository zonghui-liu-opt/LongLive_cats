#!/usr/bin/env bash
set -euo pipefail

STAGE1_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${STAGE1_PROJECT_DIR}"

STAGE1_CONFIG_PATH="${LONG_LIVE_STAGE1_CONFIG_PATH:-configs/train_i2v_ar.yaml}"
STAGE1_CACHE_GPUS="${LONG_LIVE_STAGE1_CACHE_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}"
STAGE1_MASTER_PORT="${LONG_LIVE_STAGE1_CACHE_MASTER_PORT:-29641}"

IFS=',' read -r -a STAGE1_GPU_IDS <<< "${STAGE1_CACHE_GPUS}"
if [[ "${#STAGE1_GPU_IDS[@]}" -ne 4 ]]; then
  echo "Cache extraction requires exactly 4 GPU ids; got: ${STAGE1_CACHE_GPUS}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${STAGE1_CACHE_GPUS}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "Stage-1 cache extraction: GPUs=${CUDA_VISIBLE_DEVICES} config=${STAGE1_CONFIG_PATH}"
torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  --master_port="${STAGE1_MASTER_PORT}" \
  scripts/precompute_stage1_i2v_cache.py \
  --config-path "${STAGE1_CONFIG_PATH}"
