#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT

: "${LONG_LIVE_STAGE2_INFERENCE_CHECKPOINT:?set the committed Stage-2 checkpoint directory}"
: "${LONG_LIVE_STAGE2_SOURCE_MANIFEST:?set the attested Stage-2 F25 source manifest}"
: "${LONG_LIVE_STAGE2_ARCHITECTURE_ROOT:?set the Wan2.2-TI2V-5B architecture root}"
: "${LONG_LIVE_STAGE2_T5_CHECKPOINT:?set the Wan UMT5 encoder checkpoint}"
: "${LONG_LIVE_STAGE2_TOKENIZER_DIR:?set the Wan UMT5 tokenizer directory}"
: "${LONG_LIVE_STAGE2_VAE_CHECKPOINT:?set the Wan VAE checkpoint}"
: "${LONG_LIVE_STAGE2_INFERENCE_OUTPUT:?set the Stage-2 inference output directory}"

STAGE2_PYTHON="${STAGE2_PYTHON:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python}"
STAGE2_TORCHRUN="${STAGE2_TORCHRUN:-$(dirname -- "${STAGE2_PYTHON}")/torchrun}"
STAGE2_INFERENCE_NPROC="${STAGE2_INFERENCE_NPROC:-8}"
STAGE2_INFERENCE_HEARTBEAT_SECONDS="${STAGE2_INFERENCE_HEARTBEAT_SECONDS:-30}"
[[ -x "${STAGE2_PYTHON}" ]] || { echo "not executable: ${STAGE2_PYTHON}" >&2; exit 1; }
[[ -x "${STAGE2_TORCHRUN}" ]] || { echo "not executable: ${STAGE2_TORCHRUN}" >&2; exit 1; }
[[ "${STAGE2_INFERENCE_NPROC}" =~ ^[1-9][0-9]*$ ]] || {
  echo "STAGE2_INFERENCE_NPROC must be a positive integer" >&2
  exit 1
}
[[ "${STAGE2_INFERENCE_HEARTBEAT_SECONDS}" =~ ^[1-9][0-9]*$ ]] || {
  echo "STAGE2_INFERENCE_HEARTBEAT_SECONDS must be a positive integer" >&2
  exit 1
}
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
  [[ "${#visible_gpus[@]}" -eq "${STAGE2_INFERENCE_NPROC}" ]] || {
    echo "CUDA_VISIBLE_DEVICES has ${#visible_gpus[@]} devices, but STAGE2_INFERENCE_NPROC=${STAGE2_INFERENCE_NPROC}" >&2
    exit 1
  }
fi

export PYTHONUNBUFFERED=1

cd -- "${PROJECT_ROOT}"

echo "STAGE2_INFERENCE_PREFLIGHT=START"
"${STAGE2_PYTHON}" -I -B - "${PROJECT_ROOT}" <<'PY'
import sys

sys.path.insert(0, sys.argv[1])
from utils.stage1_causal_validation import resolve_ffprobe

print(f"STAGE2_FFPROBE=PASS ({resolve_ffprobe()})", flush=True)
PY
echo "STAGE2_INFERENCE_PREFLIGHT=PASS"

INFERENCE_PID=""
HEARTBEAT_PID=""

on_interrupt() {
  trap - INT TERM
  if [[ -n "${INFERENCE_PID}" ]] && kill -0 "${INFERENCE_PID}" 2>/dev/null; then
    kill -TERM "${INFERENCE_PID}" 2>/dev/null || true
    wait "${INFERENCE_PID}" 2>/dev/null || true
  fi
  [[ -z "${HEARTBEAT_PID}" ]] || kill "${HEARTBEAT_PID}" 2>/dev/null || true
  exit 130
}
trap on_interrupt INT TERM

echo "STAGE2_INFERENCE_LAUNCH=START nproc=${STAGE2_INFERENCE_NPROC} checkpoint=${LONG_LIVE_STAGE2_INFERENCE_CHECKPOINT}"
"${STAGE2_TORCHRUN}" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node="${STAGE2_INFERENCE_NPROC}" \
  --max-restarts=0 \
  --no-python "${STAGE2_PYTHON}" -I -B \
  scripts/run_stage2_inference.py \
  --config configs/infer_i2v_stage2_baseline.yaml &
INFERENCE_PID="$!"

(
  while sleep "${STAGE2_INFERENCE_HEARTBEAT_SECONDS}"; do
    kill -0 "${INFERENCE_PID}" 2>/dev/null || exit 0
    echo "STAGE2_INFERENCE_HEARTBEAT=RUNNING pid=${INFERENCE_PID} time=$(date '+%F %T %z')"
  done
) &
HEARTBEAT_PID="$!"

if wait "${INFERENCE_PID}"; then
  inference_status=0
else
  inference_status="$?"
fi
kill "${HEARTBEAT_PID}" 2>/dev/null || true
wait "${HEARTBEAT_PID}" 2>/dev/null || true
if [[ "${inference_status}" -ne 0 ]]; then
  echo "STAGE2_INFERENCE_LAUNCH=FAIL exit_code=${inference_status}" >&2
  exit "${inference_status}"
fi
echo "STAGE2_INFERENCE_LAUNCH=PASS"
