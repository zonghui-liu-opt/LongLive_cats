#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/configs/infer_i2v_stage1_teacher_forcing_rollout.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"

required_env=(
  LONG_LIVE_STAGE1_ARCHITECTURE_ROOT
  LONG_LIVE_STAGE1_T5_CHECKPOINT
  LONG_LIVE_STAGE1_TOKENIZER_DIR
  LONG_LIVE_STAGE1_VAE_CHECKPOINT
  LONG_LIVE_STAGE1_BASE_CHECKPOINT
  LONG_LIVE_STAGE1_CHECKPOINT_DIR
  LONG_LIVE_STAGE1_ROLLOUT_INPUT
  LONG_LIVE_STAGE1_ROLLOUT_METADATA
  LONG_LIVE_STAGE1_ROLLOUT_OUTPUT
)
for name in "${required_env[@]}"; do
  if [[ -z "${!name:-}" ]]; then
    echo "Missing required environment variable: ${name}" >&2
    exit 2
  fi
done

if [[ ! -d "${LONG_LIVE_STAGE1_ROLLOUT_INPUT}" ]]; then
  echo "Stage-1 rollout input directory does not exist: ${LONG_LIVE_STAGE1_ROLLOUT_INPUT}" >&2
  exit 2
fi
if [[ ! -f "${LONG_LIVE_STAGE1_ROLLOUT_METADATA}" ]]; then
  echo "Stage-1 rollout metadata file does not exist: ${LONG_LIVE_STAGE1_ROLLOUT_METADATA}" >&2
  exit 2
fi

if [[ "${CUDA_VISIBLE_DEVICES:-0}" == *,* ]]; then
  echo "Stage-1 rollout currently supports exactly one visible GPU" >&2
  exit 2
fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

checkpoint_dir="${LONG_LIVE_STAGE1_CHECKPOINT_DIR}"
for artifact in \
  _SUCCESS \
  adapter_ema.safetensors \
  adapter_raw.safetensors \
  base_reference.json \
  checkpoint_manifest.json \
  resolved_config.yaml; do
  if [[ ! -f "${checkpoint_dir}/${artifact}" ]]; then
    echo "Incomplete Stage-1 checkpoint; missing ${checkpoint_dir}/${artifact}" >&2
    exit 2
  fi
done

chunk_size="${STAGE1_ROLLOUT_CHUNK_SIZE:-8}"
window_size="${STAGE1_ROLLOUT_WINDOW_SIZE:-17}"
sampling_steps="${STAGE1_ROLLOUT_SAMPLING_STEPS:-50}"
timestep_shift="${STAGE1_ROLLOUT_TIMESTEP_SHIFT:-5.0}"
guidance_scale="${STAGE1_ROLLOUT_GUIDANCE_SCALE:-5.0}"
expected_step="${STAGE1_ROLLOUT_EXPECTED_STEP:-4500}"

echo "Stage-1 rollout profile: C=${chunk_size} totalW=${window_size} S=1 K=${sampling_steps} shift=${timestep_shift} CFG=${guidance_scale} expected_step=${expected_step}"
echo "Checkpoint: ${checkpoint_dir}/adapter_ema.safetensors"
echo "Input metadata: ${LONG_LIVE_STAGE1_ROLLOUT_METADATA}"
echo "Output: ${LONG_LIVE_STAGE1_ROLLOUT_OUTPUT}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/inference.py" \
  --config_path "${CONFIG_PATH}" \
  --stage1_rollout_chunk_size "${chunk_size}" \
  --stage1_rollout_window_size "${window_size}" \
  --stage1_rollout_sampling_steps "${sampling_steps}" \
  --stage1_rollout_timestep_shift "${timestep_shift}" \
  --stage1_rollout_guidance_scale "${guidance_scale}" \
  --stage1_rollout_expected_step "${expected_step}" \
  "$@"
