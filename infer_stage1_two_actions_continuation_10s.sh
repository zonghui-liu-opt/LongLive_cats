#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  infer_stage1_two_actions_continuation_10s.sh [EMPTY_WORK_DIR]

Set EMPTY_WORK_DIR as the only positional argument or through
LONG_LIVE_STAGE1_CONTINUATION_WORK_DIR. The directory may not exist yet, or
it must already be empty. Existing runs are never removed or overwritten.
EOF
}

if [[ $# -gt 1 ]]; then
  usage >&2
  exit 2
fi

: "${LONG_LIVE_STAGE1_PROJECT_ROOT:?Set LONG_LIVE_STAGE1_PROJECT_ROOT to the LongLive repository root}"
: "${LONG_LIVE_STAGE1_TRAIN_DIR:?Set LONG_LIVE_STAGE1_TRAIN_DIR to the Stage-1 training output root}"
: "${LONG_LIVE_STAGE1_ARCHITECTURE_ROOT:?Set LONG_LIVE_STAGE1_ARCHITECTURE_ROOT}"
: "${LONG_LIVE_STAGE1_T5_CHECKPOINT:?Set LONG_LIVE_STAGE1_T5_CHECKPOINT}"
: "${LONG_LIVE_STAGE1_TOKENIZER_DIR:?Set LONG_LIVE_STAGE1_TOKENIZER_DIR}"
: "${LONG_LIVE_STAGE1_VAE_CHECKPOINT:?Set LONG_LIVE_STAGE1_VAE_CHECKPOINT}"
: "${LONG_LIVE_STAGE1_BASE_CHECKPOINT:?Set LONG_LIVE_STAGE1_BASE_CHECKPOINT}"
: "${LONG_LIVE_STAGE1_BASE_MANIFEST:?Set LONG_LIVE_STAGE1_BASE_MANIFEST}"

stage1_python="${LONG_LIVE_STAGE1_PYTHON:-python}"
project_root="${LONG_LIVE_STAGE1_PROJECT_ROOT%/}"
work_dir="${1:-${LONG_LIVE_STAGE1_CONTINUATION_WORK_DIR:-}}"
training_checkpoint="${LONG_LIVE_STAGE1_TRAIN_DIR%/}/checkpoint_model_003750"
metadata_path="$project_root/testsets/metadata_8cases_two_actions_continuation_480x832_253frames.csv"

if [[ -z "$work_dir" ]]; then
  usage >&2
  echo "error: provide a new work directory or set LONG_LIVE_STAGE1_CONTINUATION_WORK_DIR" >&2
  exit 2
fi
if [[ -e "$work_dir" && ! -d "$work_dir" ]]; then
  echo "error: work path exists but is not a directory: $work_dir" >&2
  exit 2
fi
if [[ -d "$work_dir" && -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "error: work directory must be empty; keep the existing run as a failure snapshot: $work_dir" >&2
  exit 2
fi
if [[ ! -d "$project_root" ]]; then
  echo "error: project root is not a directory: $project_root" >&2
  exit 2
fi
if [[ ! -d "$training_checkpoint" ]]; then
  echo "error: Stage-1 checkpoint is missing: $training_checkpoint" >&2
  exit 2
fi
if [[ ! -f "$metadata_path" ]]; then
  echo "error: continuation metadata is missing: $metadata_path" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "error: continuation inference requires exactly one visible GPU" >&2
  exit 2
fi

cd "$project_root"
exec "$stage1_python" scripts/run_stage1_continuation_validation.py \
  --training-checkpoint "$training_checkpoint" \
  --metadata "$metadata_path" \
  --work-dir "$work_dir"
