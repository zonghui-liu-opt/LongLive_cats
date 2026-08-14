#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "STAGE2_F25_CACHE_FAILED (line ${LINENO})" >&2' ERR

SOURCE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_F25_WORK_ROOT="${STAGE2_F25_WORK_ROOT:-${STAGE2_WORK_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_pretrain_check}}"

REPO_ROOT="$SOURCE_REPO"
cd "$REPO_ROOT"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$STAGE2_F25_WORK_ROOT/pycache"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

STAGE2_PYTHON="${STAGE2_PYTHON:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python}"
STAGE2_TORCHRUN="${STAGE2_TORCHRUN:-$(dirname "$STAGE2_PYTHON")/torchrun}"
STAGE2_CONFIG="${STAGE2_CONFIG:-configs/train_i2v_stage2_600cats.yaml}"
STAGE2_CACHE_GPUS="${STAGE2_CACHE_GPUS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"

ARCH_ROOT="${ARCH_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B}"
TEACHER_CKPT="${TEACHER_CKPT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/ckpts}"
METADATA_600="${METADATA_600:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/metadata_600clips_480x832_buckets.csv}"
ACTION_SIDECAR_600="${ACTION_SIDECAR_600:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/action_labels_600cats.csv}"
STAGE1_CACHE_MANIFEST="${STAGE1_CACHE_MANIFEST:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/cache_480x832_buckets/ar_stage1_i2v_600cats/cache_manifest.json}"
OPERATOR_ID="${OPERATOR_ID:-l00832862}"

STAGE2_CACHE_DIR="${STAGE2_CACHE_DIR:-$STAGE2_F25_WORK_ROOT/f25_600_v1}"
NEGATIVE_DIR="${NEGATIVE_DIR:-$STAGE2_F25_WORK_ROOT/negative_v1}"
LOG_DIR="${STAGE2_F25_LOG_DIR:-$STAGE2_F25_WORK_ROOT/logs/f25_cache}"
ASSET_DIR="${STAGE2_ASSET_DIR:-$STAGE2_F25_WORK_ROOT/assets}"
TEACHER_MANIFEST="${TEACHER_MANIFEST:-$ASSET_DIR/real_score_teacher.manifest.json}"
G_MERGED="${G_MERGED:-$ASSET_DIR/stage1_step3750_ema_merged.pt}"
G_MANIFEST="${G_MANIFEST:-${G_MERGED%.pt}.manifest.json}"
F25_BASE="$STAGE2_CACHE_DIR/cache_manifest.json"
F25_SUCCESS="$STAGE2_CACHE_DIR/_F25_SUCCESS.json"
F25_ATTESTED="$STAGE2_CACHE_DIR/cache_manifest.attested.json"
FINAL_CACHE_MANIFEST="$STAGE2_CACHE_DIR/stage2_i2v_manifest.json"
NEGATIVE_MANIFEST="$NEGATIVE_DIR/negative_conditioning_manifest.json"
NEGATIVE_TENSORS="$NEGATIVE_DIR/negative_conditioning.safetensors"
VAE_CKPT="$ARCH_ROOT/Wan2.2_VAE.pth"
T5_CKPT="$ARCH_ROOT/models_t5_umt5-xxl-enc-bf16.pth"
TOKENIZER_DIR="$ARCH_ROOT/google/umt5-xxl"

[[ -x "$STAGE2_PYTHON" ]] || { echo "不可执行：$STAGE2_PYTHON" >&2; exit 1; }
[[ -x "$STAGE2_TORCHRUN" ]] || { echo "不可执行：$STAGE2_TORCHRUN" >&2; exit 1; }
for path in \
  "$STAGE2_CONFIG" "$METADATA_600" "$ACTION_SIDECAR_600" \
  "$STAGE1_CACHE_MANIFEST" "$VAE_CKPT" "$T5_CKPT" "$TOKENIZER_DIR"; do
  [[ -e "$path" ]] || { echo "缺少：$path" >&2; exit 1; }
done
for path in \
  "$METADATA_600" "$ACTION_SIDECAR_600" "$STAGE1_CACHE_MANIFEST" \
  "$ARCH_ROOT" "$STAGE2_CACHE_DIR" "$NEGATIVE_DIR" "$LOG_DIR"; do
  [[ "$path" == /* ]] || { echo "正式数据与输出路径必须是绝对路径：$path" >&2; exit 1; }
done

mapfile -t ACTION_IDS < <(
  "$STAGE2_PYTHON" -I -B scripts/validate_stage2_i2v_cache_inputs.py \
    --metadata-path "$METADATA_600" \
    --action-labels-path "$ACTION_SIDECAR_600" \
    --source-cache-manifest "$STAGE1_CACHE_MANIFEST" \
    --expected-num-samples 600 \
    --expected-num-actions 3 \
    --action-ids-only
)
[[ "${#ACTION_IDS[@]}" == "3" ]] || {
  echo "动作标签预检未返回正好3个动作。" >&2
  exit 1
}
echo "CHECK_INPUTS_PASS actions=${ACTION_IDS[*]}"

IFS=',' read -r -a GPU_IDS <<< "$STAGE2_CACHE_GPUS"
[[ "${#GPU_IDS[@]}" == "8" ]] || {
  echo "Stage-2 F25提取需要正好8张GPU；当前：$STAGE2_CACHE_GPUS" >&2
  exit 2
}
declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_IDS[@]}"; do
  [[ "$gpu_id" =~ ^[0-9]+$ ]] || {
    echo "GPU ID必须是非负整数：$gpu_id" >&2
    exit 2
  }
  [[ -z "${SEEN_GPU_IDS[$gpu_id]+x}" ]] || {
    echo "GPU ID不能重复：$gpu_id" >&2
    exit 2
  }
  SEEN_GPU_IDS[$gpu_id]=1
done
export CUDA_VISIBLE_DEVICES="$STAGE2_CACHE_GPUS"

"$STAGE2_PYTHON" -B - <<'PY'
import torch

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
names = [torch.cuda.get_device_name(index) for index in range(8)]
assert all("H100" in name for name in names), names
assert all(
    torch.cuda.get_device_properties(index).total_memory >= 79 * 1024**3
    for index in range(8)
)
assert torch.cuda.is_bf16_supported()
print("CHECK_H100_PASS devices=" + " | ".join(names))
PY

mkdir -p "$LOG_DIR"
export LONG_LIVE_STAGE2_METADATA_PATH="$METADATA_600"
export LONG_LIVE_STAGE2_SOURCE_MANIFEST="$F25_ATTESTED"
export LONG_LIVE_STAGE2_ACTION_LABELS_PATH="$ACTION_SIDECAR_600"
export LONG_LIVE_STAGE2_CACHE_DIR="$STAGE2_CACHE_DIR"
export LONG_LIVE_STAGE2_NEGATIVE_MANIFEST="$NEGATIVE_MANIFEST"
# The formal cache manifest binds the complete launch hash, including model
# locations. Use the same defaults as prepare_stage2.sh so later training can
# validate this cache without a launch-hash mismatch.
export LONG_LIVE_STAGE2_ARCHITECTURE_ROOT="$ARCH_ROOT"
export LONG_LIVE_STAGE2_GENERATOR_BASE="$G_MERGED"
export LONG_LIVE_STAGE2_GENERATOR_MANIFEST="$G_MANIFEST"
export LONG_LIVE_STAGE2_REAL_SCORE_BASE="$TEACHER_CKPT"
export LONG_LIVE_STAGE2_REAL_SCORE_MANIFEST="$TEACHER_MANIFEST"

"$STAGE2_PYTHON" -B - "$STAGE2_CONFIG" <<'PY'
import sys
from omegaconf import OmegaConf
from utils.stage2_config import resolve_stage2_config

r = resolve_stage2_config(OmegaConf.load(sys.argv[1]))
assert r.video_latent_frames == 25
assert r.initial_latent_frames == 1
assert r.future_latent_frames == 24
PY

"$STAGE2_TORCHRUN" \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B \
  scripts/prepare_stage2_i2v_f25_cache.py \
  --config-path "$STAGE2_CONFIG" \
  --source-cache-manifest "$STAGE1_CACHE_MANIFEST" \
  --vae-checkpoint "$VAE_CKPT" \
  2>&1 | tee "$LOG_DIR/f25.log"

[[ -f "$F25_BASE" && -f "$F25_SUCCESS" ]] || {
  echo "F25结果不完整：$STAGE2_CACHE_DIR" >&2
  exit 1
}
F25_BASE_SELF_SHA="$("$STAGE2_PYTHON" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["manifest_sha256"])' \
  "$F25_BASE")"

TEXT_ATTESTATION='I attest that this legacy positive cache was encoded with the locked Wan seq512/whitespace/add-special-tokens/right-padding/exact-zero-padding contract.'
if [[ ! -e "$F25_ATTESTED" ]]; then
  "$STAGE2_PYTHON" -I -B scripts/audit_stage2_i2v_cache.py \
    upgrade-source-manifest \
    --base-source-cache-manifest "$F25_BASE" \
    --output-manifest "$F25_ATTESTED" \
    --expected-source-manifest-sha256 "$F25_BASE_SELF_SHA" \
    --t5-checkpoint "$T5_CKPT" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --operator-id "$OPERATOR_ID" \
    --operator-attestation "$TEXT_ATTESTATION" \
    --expected-num-samples 600 \
    2>&1 | tee "$LOG_DIR/source_attestation.log"
fi

"$STAGE2_PYTHON" -B - "$F25_ATTESTED" <<'PY'
import sys
from pathlib import Path
from utils.stage2_i2v_data import load_source_cache_manifest

load_source_cache_manifest(
    Path(sys.argv[1]),
    expected_num_samples=600,
    require_text_encoding_upgrade=True,
)
PY
echo "CHECK_F25_PASS manifest=$F25_ATTESTED"

if [[ ! -e "$NEGATIVE_DIR" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "$STAGE2_PYTHON" -I -B \
    scripts/audit_stage2_i2v_cache.py prepare-negative \
    --source-cache-manifest "$F25_ATTESTED" \
    --t5-checkpoint "$T5_CKPT" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --output-dir "$NEGATIVE_DIR" \
    --expected-num-samples 600 \
    --device cuda:0 \
    2>&1 | tee "$LOG_DIR/negative.log"
elif [[ ! -f "$NEGATIVE_MANIFEST" || ! -f "$NEGATIVE_TENSORS" ]]; then
  echo "negative conditioning目录是半成品，请更换NEGATIVE_DIR：$NEGATIVE_DIR" >&2
  exit 1
fi

"$STAGE2_PYTHON" -B - "$F25_ATTESTED" "$NEGATIVE_MANIFEST" <<'PY'
import sys
from pathlib import Path
from utils.stage2_i2v_data import load_negative_conditioning, load_source_cache_manifest

source = load_source_cache_manifest(
    Path(sys.argv[1]), expected_num_samples=600, require_text_encoding_upgrade=True
)
load_negative_conditioning(
    Path(sys.argv[2]), source_cache_manifest=source, load_tensors=False
)
PY
echo "CHECK_NEGATIVE_PASS manifest=$NEGATIVE_MANIFEST"

if [[ ! -e "$FINAL_CACHE_MANIFEST" ]]; then
  "$STAGE2_PYTHON" -I -B scripts/audit_stage2_i2v_cache.py audit \
    --config-path "$STAGE2_CONFIG" \
    --source-cache-manifest "$F25_ATTESTED" \
    --action-labels-path "$ACTION_SIDECAR_600" \
    --action-id "${ACTION_IDS[0]}" \
    --action-id "${ACTION_IDS[1]}" \
    --action-id "${ACTION_IDS[2]}" \
    2>&1 | tee "$LOG_DIR/cache_audit.log"
fi

"$STAGE2_PYTHON" -B - \
  "$FINAL_CACHE_MANIFEST" "$METADATA_600" "$F25_ATTESTED" \
  "$NEGATIVE_MANIFEST" "$STAGE2_CONFIG" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
from utils.stage2_action_contract import STAGE2_EXPECTED_ACTION_COUNTS
from utils.stage2_config import resolve_stage2_config
from utils.stage2_i2v_data import (
    load_stage2_i2v_manifest,
    validate_stage2_i2v_runtime_bindings,
)

resolved = resolve_stage2_config(OmegaConf.load(sys.argv[5]))
manifest = load_stage2_i2v_manifest(Path(sys.argv[1]), expected_num_samples=600)
validate_stage2_i2v_runtime_bindings(
    manifest,
    metadata_path=Path(sys.argv[2]),
    source_cache_manifest_path=Path(sys.argv[3]),
    negative_conditioning_manifest_path=Path(sys.argv[4]),
    config_contract_sha256=resolved.contract_hash(),
    config_launch_sha256=resolved.launch_hash(),
    expected_num_samples=600,
)
assert manifest["actions"]["counts"] == STAGE2_EXPECTED_ACTION_COUNTS
assert all(record["real_future_shape"][:2] == [24, 48] for record in manifest["records"])
assert {
    tuple(record["latent_spatial_shape"]) for record in manifest["records"]
} <= {(30, 52), (52, 30)}
PY

echo "CHECK_AUDIT_PASS manifest=$FINAL_CACHE_MANIFEST"
echo "STAGE2_F25_CACHE_PASS"
echo "LONG_LIVE_STAGE2_SOURCE_MANIFEST=$F25_ATTESTED"
echo "LONG_LIVE_STAGE2_CACHE_DIR=$STAGE2_CACHE_DIR"
echo "LONG_LIVE_STAGE2_NEGATIVE_MANIFEST=$NEGATIVE_MANIFEST"
echo "STAGE2_WORK_ROOT=$STAGE2_F25_WORK_ROOT"
