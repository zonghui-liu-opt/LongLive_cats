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
: "${LONG_LIVE_STAGE2_INFERENCE_OUTPUT:?set an output directory outside the clean checkout}"

STAGE2_PYTHON="${STAGE2_PYTHON:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python}"
STAGE2_TORCHRUN="${STAGE2_TORCHRUN:-$(dirname -- "${STAGE2_PYTHON}")/torchrun}"
[[ -x "${STAGE2_PYTHON}" ]] || { echo "not executable: ${STAGE2_PYTHON}" >&2; exit 1; }
[[ -x "${STAGE2_TORCHRUN}" ]] || { echo "not executable: ${STAGE2_TORCHRUN}" >&2; exit 1; }

cd -- "${PROJECT_ROOT}"
exec "${STAGE2_TORCHRUN}" \
  --standalone \
  --nnodes=1 \
  --nproc-per-node=8 \
  --max-restarts=0 \
  --no-python "${STAGE2_PYTHON}" -I -B \
  scripts/run_stage2_inference.py \
  --config configs/infer_i2v_stage2_baseline.yaml
