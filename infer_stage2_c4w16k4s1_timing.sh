#!/usr/bin/env bash
set -Eeuo pipefail

# Keep model bootstrap, artifact validation, locking, and resume in one runner.
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_ROOT
readonly TIMING_CONFIG="configs/infer_i2v_stage2_c4w16k4s1_timing.yaml"

fail() {
  printf '[Stage-2 C4W16K4S1 timing][失败] %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Stage-2 C4/W16/S1/K4 推理与分阶段计时

用法：
  bash infer_stage2_c4w16k4s1_timing.sh MODE CHECKPOINT_STEP

MODE：
  plan-quick    CPU 预检 quick 计划，不启动 CUDA/torchrun
  quick         4 个视频（2 单动作、2 双动作，seed=1）
  plan-formal   CPU 预检 formal 计划，不启动 CUDA/torchrun
  formal        56 个视频（6 单动作、8 双动作，各 4 个 seed）

示例：
  bash infer_stage2_c4w16k4s1_timing.sh plan-quick G120
  bash infer_stage2_c4w16k4s1_timing.sh quick G120
  bash infer_stage2_c4w16k4s1_timing.sh formal G120

CHECKPOINT_STEP 是训练 checkpoint 步数，例如 120/G120/000120；
与本入口固定的 4 步 denoising（K4）无关。必须明确指定。
若设置 STAGE2_EARLY_CHECKPOINT，目录名为 checkpoint_stage2_gNNNNNN 时
可省略 CHECKPOINT_STEP；其他目录名仍需显式指定步数。

默认使用 GPU 0、单卡数据并行进程，按样本报告 DiT、VAE decode、视频后处理。
可设置 CUDA_VISIBLE_DEVICES（自动推导进程数），或同时设置
CUDA_VISIBLE_DEVICES / STAGE2_INFERENCE_NPROC；其余路径覆盖沿用
infer_stage2_tmp.sh。默认权重目录匹配 C4/W16/S1 的 8 卡训练入口。

结果：每样本 trace/log，以及输出根目录的 timing_summary.json、
timing_samples.csv、index.html。详细计时口径见
docs/STAGE2_C4W16K4S1_TIMING_ZH.md。
EOF
}

MODE="${1:-}"
if [[ "$MODE" == "--help" || "$MODE" == "-h" || "$MODE" == "help" ]]; then
  usage
  exit 0
fi
[[ -n "$MODE" ]] || fail "缺少 MODE 和 checkpoint；使用 --help 查看用法"
shift
case "$MODE" in
  quick|formal)
    export STAGE2_SWEEP_EVALUATION="$MODE"
    export STAGE2_INFERENCE_PLAN_ONLY=0
    ;;
  plan-quick|plan-formal)
    export STAGE2_SWEEP_EVALUATION="${MODE#plan-}"
    export STAGE2_INFERENCE_PLAN_ONLY=1
    ;;
  *) fail "未知 MODE：$MODE；使用 --help 查看用法" ;;
esac

(($# <= 1)) || fail "每次只指定一个 checkpoint step"
REQUESTED_STEP="${1:-}"
if [[ -z "$REQUESTED_STEP" && -n "${STAGE2_EARLY_CHECKPOINT:-}" ]]; then
  CHECKPOINT_DIRECTORY="${STAGE2_EARLY_CHECKPOINT%/}"
  CHECKPOINT_BASENAME="${CHECKPOINT_DIRECTORY##*/}"
  if [[ "$CHECKPOINT_BASENAME" =~ ^checkpoint_stage2_g([0-9]{6})$ ]]; then
    REQUESTED_STEP="${BASH_REMATCH[1]}"
  fi
fi
[[ -n "$REQUESTED_STEP" ]] || \
  fail "必须指定 checkpoint step，例如 G120；不会默认选择 G70"
NORMALIZED_STEP="${REQUESTED_STEP#g}"
NORMALIZED_STEP="${NORMALIZED_STEP#G}"
[[ "$NORMALIZED_STEP" =~ ^[0-9]{1,6}$ ]] || \
  fail "checkpoint step 必须是 0-999999，例如 G120"
printf -v NORMALIZED_STEP '%06d' "$((10#$NORMALIZED_STEP))"
if [[ -n "${STAGE2_EARLY_CHECKPOINT:-}" ]]; then
  CHECKPOINT_DIRECTORY="${STAGE2_EARLY_CHECKPOINT%/}"
  CHECKPOINT_BASENAME="${CHECKPOINT_DIRECTORY##*/}"
  if [[ "$CHECKPOINT_BASENAME" =~ ^checkpoint_stage2_g([0-9]{6})$ && "${BASH_REMATCH[1]}" != "$NORMALIZED_STEP" ]]; then
    fail "显式 step 与 STAGE2_EARLY_CHECKPOINT 目录中的训练步数不同"
  fi
fi

if [[ -n "${STAGE2_INFERENCE_CONFIG:-}" && "$STAGE2_INFERENCE_CONFIG" != "$TIMING_CONFIG" ]]; then
  fail "本入口固定 C4/W16/S1/K4 配置，不能覆盖 STAGE2_INFERENCE_CONFIG"
fi
export STAGE2_INFERENCE_CONFIG="$TIMING_CONFIG"
export STAGE2_PROJECT_ROOT="${STAGE2_PROJECT_ROOT:-$SCRIPT_ROOT}"
export STAGE2_WORK_ROOT="${STAGE2_WORK_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_stage2_h100_c4w16s1_micro1_acc8_longrun}"
export STAGE2_TRAIN_ROOT="${STAGE2_TRAIN_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_training_h100_c4w16s1_micro1_acc8_longrun}"
export SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-${STAGE2_FORMAL_DIR:-$STAGE2_TRAIN_ROOT/formal_b1_c4w16s1_all_epochs}}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  [[ -z "${STAGE2_INFERENCE_NPROC:-${INFER_NPROC:-}}" || "${STAGE2_INFERENCE_NPROC:-${INFER_NPROC:-}}" == "1" ]] || \
    fail "指定多个进程时必须同时设置 CUDA_VISIBLE_DEVICES"
  export CUDA_VISIBLE_DEVICES=0
fi
if [[ -z "${STAGE2_INFERENCE_NPROC:-}" ]]; then
  if [[ -n "${INFER_NPROC:-}" ]]; then
    export STAGE2_INFERENCE_NPROC="$INFER_NPROC"
  else
    IFS=',' read -r -a TIMING_GPUS <<< "$CUDA_VISIBLE_DEVICES"
    export STAGE2_INFERENCE_NPROC="${#TIMING_GPUS[@]}"
  fi
fi

exec bash "$SCRIPT_ROOT/infer_stage2_tmp.sh" "$NORMALIZED_STEP"
