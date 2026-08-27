#!/usr/bin/env bash
set -Eeuo pipefail

# Production wrapper for C4/W16/S1 Stage-2 checkpoint inference sweeps.
# It only selects a committed profile matrix and canonical checkpoint list.
# Model loading, output locking, artifact validation, and resume are delegated
# to infer_stage2_tmp.sh so there is one implementation of those contracts.

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_ROOT
readonly BASE_RUNNER="$SCRIPT_ROOT/infer_stage2_tmp.sh"
readonly MATCHED_CONFIG="configs/infer_i2v_stage2_c4w16_k432_sweep.yaml"
readonly COMPRESSED_CONFIG="configs/infer_i2v_stage2_cw_compression_k4_sweep.yaml"
readonly MIN_STEP=40

say() {
  printf '[Stage-2 profile sweep] %s\n' "$*"
}

fail() {
  printf '[Stage-2 profile sweep][失败] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Stage-2 C4/W16/S1 多权重 × 多profile H100推理入口

用法：
  bash run_stage2_profile_sweep_h100.sh MODE PROFILE CHECKPOINT...

MODE：
  plan-quick    CPU-only预检quick计划，不初始化CUDA/torchrun
  quick         运行quick集合；默认4卡
  plan-formal   CPU-only预检formal计划，不初始化CUDA/torchrun
  formal        运行formal集合；默认8卡

PROFILE：
  matched       C4/W16/S1，K=4/3/2；quick/formal每权重12/168组结果
  compressed    K4固定，比较C4W16、C4W12、C4W8、C2W16、C2W8；
                quick/formal每权重20/280组结果
  both          固定先matched、后compressed；quick/formal每权重共32/448组结果

CHECKPOINT：
  显式step       例如 40 80 G120；会按数字排序、去重
  all            自动发现SNAPSHOT_ROOT下所有完整且G>=40的checkpoint

推荐顺序：
  bash run_stage2_profile_sweep_h100.sh plan-quick matched all
  bash run_stage2_profile_sweep_h100.sh quick matched all
  bash run_stage2_profile_sweep_h100.sh plan-formal matched 80 120
  bash run_stage2_profile_sweep_h100.sh formal matched 80 120
  bash run_stage2_profile_sweep_h100.sh quick compressed 80 120
  bash run_stage2_profile_sweep_h100.sh quick both 80 120

C4训练默认目录：
  STAGE2_WORK_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_stage2_h100_c4w16s1_micro1_acc8_longrun
  STAGE2_TRAIN_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_training_h100_c4w16s1_micro1_acc8_longrun
  SNAPSHOT_ROOT=$STAGE2_TRAIN_ROOT/formal_b1_c4w16s1_all_epochs

如实际权重不在默认位置，只需在命令前覆盖SNAPSHOT_ROOT。GPU环境覆盖时请同时设置：
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 STAGE2_INFERENCE_NPROC=7 ...

每个配置、每个checkpoint只加载一次Generator/T5/VAE并跑完该配置全部profile；
both会按两个配置各加载一次。输出由配置名、quick/formal和resolved contract hash
隔离；中断后原命令重跑会严格续跑video+trace。已完整的checkpoint重跑仍会先完成
模型bootstrap再逐产物校验skip，因此失败后优先只重跑对应的matched或compressed阶段。
EOF
}

normalize_step() {
  local raw="${1#g}"
  raw="${raw#G}"
  [[ "$raw" =~ ^[0-9]{1,6}$ ]] || \
    fail "checkpoint step必须是0-999999，例如40、G80或000120：$1"
  printf '%06d' "$((10#$raw))"
}

validate_complete_checkpoint() {
  local step="$1"
  local checkpoint="$SNAPSHOT_ROOT/checkpoint_stage2_g${step}"
  local filename
  [[ -d "$checkpoint" && ! -L "$checkpoint" ]] || \
    fail "checkpoint不是普通目录：$checkpoint"
  for filename in _SUCCESS checkpoint_manifest.json provenance.json; do
    [[ -f "$checkpoint/$filename" && ! -L "$checkpoint/$filename" ]] || \
      fail "checkpoint缺少普通文件${filename}：$checkpoint/$filename"
  done
  [[ ! -s "$checkpoint/_SUCCESS" ]] || \
    fail "checkpoint的_SUCCESS必须是零字节完成标记：$checkpoint/_SUCCESS"
}

discover_all_steps() {
  local checkpoint
  local name
  local step
  local candidates=()
  shopt -s nullglob
  candidates=(
    "$SNAPSHOT_ROOT"/checkpoint_stage2_g[0-9][0-9][0-9][0-9][0-9][0-9]
  )
  shopt -u nullglob

  for checkpoint in "${candidates[@]}"; do
    [[ ! -L "$checkpoint" ]] || fail "拒绝symlink checkpoint：$checkpoint"
    [[ -d "$checkpoint" ]] || continue
    if [[ ! -e "$checkpoint/_SUCCESS" && ! -L "$checkpoint/_SUCCESS" ]]; then
      continue
    fi
    name="${checkpoint##*/}"
    step="${name#checkpoint_stage2_g}"
    if ((10#$step < MIN_STEP)); then
      continue
    fi
    validate_complete_checkpoint "$step"
    SELECTED_STEPS+=("$step")
  done
}

canonicalize_steps() {
  local step
  local sorted_steps=()
  ((${#SELECTED_STEPS[@]} > 0)) || \
    fail "没有找到可推理checkpoint；检查SNAPSHOT_ROOT和G${MIN_STEP}下限"
  while IFS= read -r step; do
    [[ -n "$step" ]] && sorted_steps+=("$step")
  done < <(printf '%s\n' "${SELECTED_STEPS[@]}" | LC_ALL=C sort -u)
  SELECTED_STEPS=("${sorted_steps[@]}")
}

configure_gpu_defaults() {
  local default_nproc="$1"
  local default_cuda="$2"
  local gpu_values=()

  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" && -z "${STAGE2_INFERENCE_NPROC:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$default_cuda"
    export STAGE2_INFERENCE_NPROC="$default_nproc"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -z "${STAGE2_INFERENCE_NPROC:-}" ]]; then
    IFS=',' read -r -a gpu_values <<< "$CUDA_VISIBLE_DEVICES"
    export STAGE2_INFERENCE_NPROC="${#gpu_values[@]}"
    return
  fi
  if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    fail "设置STAGE2_INFERENCE_NPROC时必须同时设置CUDA_VISIBLE_DEVICES"
  fi
  export CUDA_VISIBLE_DEVICES STAGE2_INFERENCE_NPROC
}

validate_gpu_layout() {
  local gpu
  local gpu_values=()
  local seen=","
  [[ "$STAGE2_INFERENCE_NPROC" =~ ^[1-9][0-9]*$ ]] || \
    fail "STAGE2_INFERENCE_NPROC必须是正整数"
  [[ "$CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]] || \
    fail "CUDA_VISIBLE_DEVICES必须是无空格、无空项的逗号分隔GPU编号"
  IFS=',' read -r -a gpu_values <<< "$CUDA_VISIBLE_DEVICES"
  [[ "${#gpu_values[@]}" -eq "$STAGE2_INFERENCE_NPROC" ]] || \
    fail "CUDA_VISIBLE_DEVICES有${#gpu_values[@]}张卡，但NPROC=$STAGE2_INFERENCE_NPROC"
  for gpu in "${gpu_values[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || fail "非法GPU编号：$gpu"
    [[ "$seen" != *",$gpu,"* ]] || fail "GPU编号重复：$gpu"
    seen+="$gpu,"
  done
}

MODE="${1:-help}"
if [[ "$MODE" == "help" || "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi
shift || true

PROFILE_SET="${1:-}"
[[ -n "$PROFILE_SET" ]] || fail "缺少PROFILE；使用--help查看示例"
shift || true
(($# > 0)) || fail "至少指定一个checkpoint step或all"

case "$MODE" in
  plan-quick)
    EVALUATION="quick"
    PLAN_ONLY=1
    DEFAULT_NPROC=4
    DEFAULT_CUDA="0,1,2,3"
    ;;
  quick)
    EVALUATION="quick"
    PLAN_ONLY=0
    DEFAULT_NPROC=4
    DEFAULT_CUDA="0,1,2,3"
    ;;
  plan-formal)
    EVALUATION="formal"
    PLAN_ONLY=1
    DEFAULT_NPROC=8
    DEFAULT_CUDA="0,1,2,3,4,5,6,7"
    ;;
  formal)
    EVALUATION="formal"
    PLAN_ONLY=0
    DEFAULT_NPROC=8
    DEFAULT_CUDA="0,1,2,3,4,5,6,7"
    ;;
  *)
    usage >&2
    fail "未知MODE：$MODE"
    ;;
esac

case "$PROFILE_SET" in
  matched) INFERENCE_CONFIGS=("$MATCHED_CONFIG") ;;
  compressed) INFERENCE_CONFIGS=("$COMPRESSED_CONFIG") ;;
  both) INFERENCE_CONFIGS=("$MATCHED_CONFIG" "$COMPRESSED_CONFIG") ;;
  *) fail "PROFILE只能是matched、compressed或both：$PROFILE_SET" ;;
esac

export STAGE2_PROJECT_ROOT="$SCRIPT_ROOT"
export STAGE2_PYTHON="${STAGE2_PYTHON:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python}"
export STAGE2_WORK_ROOT="${STAGE2_WORK_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_stage2_h100_c4w16s1_micro1_acc8_longrun}"
export STAGE2_TRAIN_ROOT="${STAGE2_TRAIN_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_training_h100_c4w16s1_micro1_acc8_longrun}"
export SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-$STAGE2_TRAIN_ROOT/formal_b1_c4w16s1_all_epochs}"

[[ -f "$BASE_RUNNER" && ! -L "$BASE_RUNNER" ]] || \
  fail "底层推理入口不存在或是symlink：$BASE_RUNNER"
for inference_config in "${INFERENCE_CONFIGS[@]}"; do
  [[ -f "$STAGE2_PROJECT_ROOT/$inference_config" && \
     ! -L "$STAGE2_PROJECT_ROOT/$inference_config" ]] || \
    fail "profile配置不存在或是symlink：$STAGE2_PROJECT_ROOT/$inference_config"
done
[[ -d "$SNAPSHOT_ROOT" && ! -L "$SNAPSHOT_ROOT" ]] || \
  fail "SNAPSHOT_ROOT不存在或是symlink：$SNAPSHOT_ROOT"

for name in STAGE2_EARLY_CHECKPOINT STAGE2_EARLY_OUTPUT STAGE2_EARLY_LOG; do
  [[ -z "${!name:-}" ]] || fail "本批量入口禁止设置${name}"
done

SELECTED_STEPS=()
if [[ "$1" == "all" ]]; then
  [[ "$#" -eq 1 ]] || fail "all不能和显式checkpoint混用"
  discover_all_steps
else
  for requested_step in "$@"; do
    step="$(normalize_step "$requested_step")"
    ((10#$step >= MIN_STEP)) || \
      fail "G${step}早于最小可推理step G$(printf '%06d' "$MIN_STEP")"
    validate_complete_checkpoint "$step"
    SELECTED_STEPS+=("$step")
  done
fi
canonicalize_steps
configure_gpu_defaults "$DEFAULT_NPROC" "$DEFAULT_CUDA"
validate_gpu_layout

export STAGE2_SWEEP_EVALUATION="$EVALUATION"
export STAGE2_INFERENCE_PLAN_ONLY="$PLAN_ONLY"

say "evaluation=${EVALUATION} plan_only=${PLAN_ONLY}"
say "checkpoint_count=${#SELECTED_STEPS[@]} steps=${SELECTED_STEPS[*]}"
say "snapshot_root=${SNAPSHOT_ROOT}"
say "GPU=${CUDA_VISIBLE_DEVICES} nproc=${STAGE2_INFERENCE_NPROC}"

for inference_config in "${INFERENCE_CONFIGS[@]}"; do
  export STAGE2_INFERENCE_CONFIG="$inference_config"
  say "PHASE=START profile=${PROFILE_SET} config=${inference_config}"
  if bash "$BASE_RUNNER" "${SELECTED_STEPS[@]}"; then
    say "PHASE=PASS config=${inference_config}"
  else
    status="$?"
    fail "PHASE=FAIL config=${inference_config} status=${status}"
  fi
done
say "STAGE2_PROFILE_SWEEP=PASS phases=${#INFERENCE_CONFIGS[@]} checkpoints=${#SELECTED_STEPS[@]}"
