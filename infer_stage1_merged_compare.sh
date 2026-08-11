#!/usr/bin/env bash
set -euo pipefail

project_root="${LONG_LIVE_STAGE1_PROJECT_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0}"
stage1_python="${LONG_LIVE_STAGE1_PYTHON:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python}"
merged_checkpoint="${LONG_LIVE_STAGE1_MERGED_CHECKPOINT:-$project_root/checkpoints/stage2/stage1_step3750_ema_merged.pt}"
reference_dir="${LONG_LIVE_STAGE1_REFERENCE_DIR:-$project_root/work_dir/stage1_600cats_phaseA10epochs_phaseB20epochs_all/checkpoint_model_003750}"
metadata_path="${LONG_LIVE_STAGE1_COMPARE_METADATA:-$project_root/testsets/metadata_6cases_480x832.csv}"
work_dir="${1:-${LONG_LIVE_STAGE1_MERGED_COMPARE_WORK_DIR:-$project_root/work_dir/stage1_step3750_merged_comparison}}"

if [[ ! -d "$project_root" ]]; then
  echo "error: project root is not a directory: $project_root" >&2
  exit 2
fi
if [[ ! -x "$stage1_python" ]]; then
  echo "error: Python executable is missing: $stage1_python" >&2
  exit 2
fi
if [[ ! -f "$merged_checkpoint" ]]; then
  echo "error: merged checkpoint is missing: $merged_checkpoint" >&2
  exit 2
fi
if [[ ! -f "$reference_dir/prepared/prepared_manifest.json" ]]; then
  echo "error: reference prepared manifest is missing under: $reference_dir" >&2
  exit 2
fi
if [[ ! -f "$metadata_path" ]]; then
  echo "error: metadata CSV is missing: $metadata_path" >&2
  exit 2
fi
if [[ -e "$work_dir" && ! -d "$work_dir" ]]; then
  echo "error: work path exists but is not a directory: $work_dir" >&2
  exit 2
fi
if [[ -d "$work_dir" && -n "$(find "$work_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "error: work directory must be empty: $work_dir" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
  echo "error: this comparison runner expects exactly one visible GPU" >&2
  exit 2
fi

cd "$project_root"
exec "$stage1_python" scripts/run_stage1_merged_checkpoint_comparison.py \
  --merged-checkpoint "$merged_checkpoint" \
  --reference-checkpoint-dir "$reference_dir" \
  --metadata "$metadata_path" \
  --work-dir "$work_dir"
