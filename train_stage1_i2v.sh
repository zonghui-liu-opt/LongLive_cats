#!/usr/bin/env bash
set -euo pipefail

STAGE1_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${STAGE1_PROJECT_DIR}"

STAGE1_CONFIG_PATH="${LONG_LIVE_STAGE1_CONFIG_PATH:-configs/train_i2v_ar.yaml}"
STAGE1_TRAIN_DIR="${LONG_LIVE_STAGE1_TRAIN_DIR:-logs/train_i2v_ar}"
STAGE1_DRY_RUN_DIR="${LONG_LIVE_STAGE1_DRY_RUN_DIR:-logs/train_i2v_ar_dry_run}"

stage1_canonical_path() {
  python - "$1" <<'PY'
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

stage1_has_resumable_checkpoint() {
  local stage1_root="$1"
  local stage1_marker
  for stage1_marker in "${stage1_root}"/checkpoint_model_[0-9][0-9][0-9][0-9][0-9][0-9]/_RESUMABLE_SUCCESS; do
    if [[ -f "${stage1_marker}" ]]; then
      return 0
    fi
  done
  return 1
}

STAGE1_TRAIN_DIR_CANONICAL="$(stage1_canonical_path "${STAGE1_TRAIN_DIR}")"
STAGE1_DRY_RUN_DIR_CANONICAL="$(stage1_canonical_path "${STAGE1_DRY_RUN_DIR}")"

if [[ "${STAGE1_DRY_RUN_DIR_CANONICAL}" == "${STAGE1_TRAIN_DIR_CANONICAL}" ]]; then
  echo "Dry-run and formal training directories must be different." >&2
  exit 2
fi

# A non-empty formal directory is accepted only as an auto-resume target. The
# trainer will validate every marker, manifest, hash, and topology before use.
STAGE1_RESUME_MODE=0
if [[ -d "${STAGE1_TRAIN_DIR}" ]] && find "${STAGE1_TRAIN_DIR}" -mindepth 1 -print -quit | grep -q .; then
  if ! stage1_has_resumable_checkpoint "${STAGE1_TRAIN_DIR}"; then
    echo "Formal directory is non-empty but has no resumable checkpoint; choose a clean LONG_LIVE_STAGE1_TRAIN_DIR." >&2
    exit 2
  fi
  STAGE1_RESUME_MODE=1
fi

if [[ "${STAGE1_RESUME_MODE}" -eq 0 ]]; then
  if [[ -d "${STAGE1_DRY_RUN_DIR}" ]] && find "${STAGE1_DRY_RUN_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "Dry-run directory is not empty; choose a clean LONG_LIVE_STAGE1_DRY_RUN_DIR." >&2
    exit 2
  fi

  # Process 1: real base/cache, exactly two micro-steps and one optimizer update.
  torchrun --standalone --nnodes=1 --nproc_per_node=6 train.py \
    --config_path "${STAGE1_CONFIG_PATH}" \
    --logdir "${STAGE1_DRY_RUN_DIR}" \
    --stage1-dry-run-one-update \
    --no-save \
    --no-visualize \
    --disable-wandb \
    --no-auto-resume

  # The formal directory was clean before dry-run and must still be clean.
  if [[ -d "${STAGE1_TRAIN_DIR}" ]] && find "${STAGE1_TRAIN_DIR}" -mindepth 1 -print -quit | grep -q .; then
    echo "Formal directory changed during dry-run; refusing to start training." >&2
    exit 2
  fi
else
  # A restart must not replay the isolated dry-run. The trainer performs the
  # authoritative manifest/hash/topology validation before restoring state.
  if ! stage1_has_resumable_checkpoint "${STAGE1_TRAIN_DIR}"; then
    echo "Resumable marker disappeared before restart; refusing to continue." >&2
    exit 2
  fi
fi

# Process 2: a fresh distributed process group. No dry-run state is reused.
torchrun --standalone --nnodes=1 --nproc_per_node=6 train.py \
  --config_path "${STAGE1_CONFIG_PATH}" \
  --logdir "${STAGE1_TRAIN_DIR}" \
  --no-visualize \
  --disable-wandb
