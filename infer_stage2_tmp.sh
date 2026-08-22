#!/usr/bin/env bash
set -Eeuo pipefail

say() {
    printf '[Stage-2 推理] %s\n' "$*"
}

fail() {
    printf '[Stage-2 推理][失败] %s\n' "$*" >&2
    stop_active_process
    release_output_lock
    exit 1
}

ACTIVE_PID=""
CURRENT_LOG=""
CURRENT_LOCK=""

stop_active_process() {
    if [[ -n "$ACTIVE_PID" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
        kill -TERM "$ACTIVE_PID" 2>/dev/null || true
        wait "$ACTIVE_PID" 2>/dev/null || true
    fi
    ACTIVE_PID=""
}

release_output_lock() {
    if [[ -z "$CURRENT_LOCK" ]]; then
        return
    fi
    if [[ "$CURRENT_LOCK" == *.stage2-run.lock ]] && \
        [[ -d "$CURRENT_LOCK" ]] && [[ ! -L "$CURRENT_LOCK" ]]; then
        rm -f -- "$CURRENT_LOCK/owner"
        rmdir -- "$CURRENT_LOCK" 2>/dev/null || true
    fi
    CURRENT_LOCK=""
}

acquire_output_lock() {
    local output="$1"
    local lock="${output}.stage2-run.lock"
    local owner="unknown"
    if ! mkdir -- "$lock" 2>/dev/null; then
        if [[ -f "$lock/owner" && ! -L "$lock/owner" ]]; then
            IFS= read -r owner < "$lock/owner" || true
        fi
        fail "同一输出目录已有推理任务：${output}；owner=${owner}；lock=$lock"
    fi
    CURRENT_LOCK="$lock"
    printf 'pid=%s host=%s started=%s\n' \
        "$$" "$(hostname)" "$(date '+%F_%T_%z')" > "$CURRENT_LOCK/owner"
}

on_error() {
    local status="$?"
    local line="$1"
    trap - ERR
    stop_active_process
    release_output_lock
    printf '[Stage-2 推理][失败] 脚本第 %s 行退出，状态码=%s' \
        "$line" "$status" >&2
    if [[ -n "$CURRENT_LOG" ]]; then
        printf '，日志=%s' "$CURRENT_LOG" >&2
    fi
    printf '\n' >&2
    exit "$status"
}

on_interrupt() {
    trap - INT TERM
    if [[ -n "$ACTIVE_PID" ]] && kill -0 "$ACTIVE_PID" 2>/dev/null; then
        say "收到中断信号，正在停止推理子进程 PID=$ACTIVE_PID"
    fi
    stop_active_process
    release_output_lock
    exit 130
}

trap 'on_error "$LINENO"' ERR
trap on_interrupt INT TERM

export STAGE2_PROJECT_ROOT="${STAGE2_PROJECT_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0}"
export STAGE2_PYTHON="${STAGE2_PYTHON:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python}"
export STAGE2_TORCHRUN="${STAGE2_TORCHRUN:-$(dirname -- "$STAGE2_PYTHON")/torchrun}"
export STAGE2_WORK_ROOT="${STAGE2_WORK_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_stage2_new}"
export STAGE2_TRAIN_ROOT="${STAGE2_TRAIN_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_training}"
export SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-$STAGE2_TRAIN_ROOT/early_checkpoint_snapshots}"

export ARCH_ROOT="${ARCH_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B}"
export LONG_LIVE_STAGE2_SOURCE_MANIFEST="${LONG_LIVE_STAGE2_SOURCE_MANIFEST:-$STAGE2_WORK_ROOT/stage2_600cats_f25_v1/cache_manifest.attested.json}"
export LONG_LIVE_STAGE2_ARCHITECTURE_ROOT="${LONG_LIVE_STAGE2_ARCHITECTURE_ROOT:-$ARCH_ROOT}"
export LONG_LIVE_STAGE2_T5_CHECKPOINT="${LONG_LIVE_STAGE2_T5_CHECKPOINT:-$ARCH_ROOT/models_t5_umt5-xxl-enc-bf16.pth}"
export LONG_LIVE_STAGE2_TOKENIZER_DIR="${LONG_LIVE_STAGE2_TOKENIZER_DIR:-$ARCH_ROOT/google/umt5-xxl}"
export LONG_LIVE_STAGE2_VAE_CHECKPOINT="${LONG_LIVE_STAGE2_VAE_CHECKPOINT:-$ARCH_ROOT/Wan2.2_VAE.pth}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export STAGE2_INFERENCE_NPROC="${STAGE2_INFERENCE_NPROC:-${INFER_NPROC:-8}}"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export LONG_LIVE_FFPROBE="${LONG_LIVE_FFPROBE:-}"

readonly INFERENCE_CONFIG="${STAGE2_INFERENCE_CONFIG:-configs/infer_i2v_stage2_baseline.yaml}"
readonly SWEEP_EVALUATION="${STAGE2_SWEEP_EVALUATION:-}"
readonly PLAN_ONLY="${STAGE2_INFERENCE_PLAN_ONLY:-0}"
readonly EXPECTED_ASSET_API="longlive_stage2_inference_assets/v2"
readonly HEARTBEAT_SECONDS="${STAGE2_INFERENCE_HEARTBEAT_SECONDS:-30}"

run_stage2_cli() {
    local command=(
        "$STAGE2_PYTHON" -I -B scripts/run_stage2_inference.py
        --config "$INFERENCE_CONFIG"
    )
    if [[ -n "$SWEEP_EVALUATION" ]]; then
        command+=(--evaluation "$SWEEP_EVALUATION")
    fi
    command+=("$@")
    "${command[@]}"
}

launch_stage2_torchrun() {
    local log="$1"
    local command=(
        "$STAGE2_TORCHRUN"
        --standalone
        --nnodes=1
        --nproc-per-node="$STAGE2_INFERENCE_NPROC"
        --max-restarts=0
        --no-python "$STAGE2_PYTHON" -I -B
        scripts/run_stage2_inference.py
        --config "$INFERENCE_CONFIG"
    )
    if [[ -n "$SWEEP_EVALUATION" ]]; then
        command+=(--evaluation "$SWEEP_EVALUATION")
    fi
    "${command[@]}" > >(tee -a "$log") 2>&1 &
    ACTIVE_PID="$!"
}

require_file() {
    local path="$1"
    local label="$2"
    [[ -f "$path" ]] || fail "$label 不存在：$path"
}

require_dir() {
    local path="$1"
    local label="$2"
    [[ -d "$path" ]] || fail "$label 不存在：$path"
}

normalize_step() {
    local raw="${1#g}"
    raw="${raw#G}"
    [[ "$raw" =~ ^[0-9]{1,6}$ ]] || fail "权重步数必须是 0-999999，例如 60、70、000070"
    printf '%06d' "$((10#$raw))"
}

validate_gpu_layout() {
    [[ "$STAGE2_INFERENCE_NPROC" =~ ^[1-9][0-9]*$ ]] || \
        fail "STAGE2_INFERENCE_NPROC 必须是正整数"
    local gpu_values=()
    local gpu
    local seen=","
    IFS=',' read -r -a gpu_values <<< "$CUDA_VISIBLE_DEVICES"
    [[ "${#gpu_values[@]}" -eq "$STAGE2_INFERENCE_NPROC" ]] || \
        fail "CUDA_VISIBLE_DEVICES 有 ${#gpu_values[@]} 张卡，但 NPROC=$STAGE2_INFERENCE_NPROC"
    for gpu in "${gpu_values[@]}"; do
        [[ "$gpu" =~ ^[0-9]+$ ]] || fail "非法 GPU 编号：$gpu"
        [[ "$seen" != *",$gpu,"* ]] || fail "GPU 编号重复：$gpu"
        seen+="$gpu,"
    done
}

preflight_code() {
    "$STAGE2_PYTHON" -I -B - "$STAGE2_PROJECT_ROOT" "$EXPECTED_ASSET_API" <<'PY'
import sys

project_root, expected = sys.argv[1:]
sys.path.insert(0, project_root)
from utils.stage2_inference_assets import STAGE2_INFERENCE_ASSET_API_VERSION
from utils.stage1_causal_validation import resolve_ffprobe

if STAGE2_INFERENCE_ASSET_API_VERSION != expected:
    raise SystemExit(
        "Stage-2 跨节点权重校验代码不是最新版："
        f"expected={expected}, actual={STAGE2_INFERENCE_ASSET_API_VERSION}"
    )
print(f"STAGE2_INFERENCE_ASSET_API=PASS ({expected})", flush=True)
print(f"STAGE2_FFPROBE=PASS ({resolve_ffprobe()})", flush=True)
PY
}

validate_common_inputs() {
    [[ -x "$STAGE2_PYTHON" ]] || fail "Python 不可执行：$STAGE2_PYTHON"
    [[ -x "$STAGE2_TORCHRUN" ]] || fail "torchrun 不可执行：$STAGE2_TORCHRUN"
    require_dir "$STAGE2_PROJECT_ROOT" "项目目录"
    [[ "$INFERENCE_CONFIG" != /* && "$INFERENCE_CONFIG" != *".."* ]] || \
        fail "STAGE2_INFERENCE_CONFIG 必须是项目内不含 .. 的相对路径"
    [[ -z "$SWEEP_EVALUATION" || "$SWEEP_EVALUATION" =~ ^(quick|formal)$ ]] || \
        fail "STAGE2_SWEEP_EVALUATION 只能是 quick 或 formal"
    [[ "$PLAN_ONLY" =~ ^(0|1)$ ]] || \
        fail "STAGE2_INFERENCE_PLAN_ONLY 只能是 0 或 1"
    require_file "$STAGE2_PROJECT_ROOT/scripts/run_stage2_inference.py" "推理入口"
    require_file "$STAGE2_PROJECT_ROOT/$INFERENCE_CONFIG" "推理配置"
    require_file "$LONG_LIVE_STAGE2_SOURCE_MANIFEST" "F25 数据清单"
    require_dir "$LONG_LIVE_STAGE2_ARCHITECTURE_ROOT" "模型结构目录"
    require_file "$LONG_LIVE_STAGE2_ARCHITECTURE_ROOT/config.json" "模型结构配置"
    require_file "$LONG_LIVE_STAGE2_T5_CHECKPOINT" "T5 权重"
    require_dir "$LONG_LIVE_STAGE2_TOKENIZER_DIR" "Tokenizer 目录"
    require_file "$LONG_LIVE_STAGE2_VAE_CHECKPOINT" "VAE 权重"
    validate_gpu_layout
    [[ "$HEARTBEAT_SECONDS" =~ ^[1-9][0-9]*$ ]] || \
        fail "STAGE2_INFERENCE_HEARTBEAT_SECONDS 必须是正整数"
    cd -- "$STAGE2_PROJECT_ROOT"
    preflight_code
}

verify_outputs() {
    local output="$1"
    local expected="$2"
    local video_count
    local trace_count
    require_file "$output/manifest.json" "最终 manifest"
    require_file "$output/index.html" "可视化页面"
    require_dir "$output/videos" "视频目录"
    require_dir "$output/traces" "trace 目录"
    video_count="$(find "$output/videos" -type f -name '*.mp4' | wc -l | tr -d '[:space:]')"
    trace_count="$(find "$output/traces" -type f -name '*.json' | wc -l | tr -d '[:space:]')"
    [[ "$video_count" -eq "$expected" ]] || \
        fail "视频数量错误：期望 ${expected}，实际 $video_count"
    [[ "$trace_count" -eq "$expected" ]] || \
        fail "trace 数量错误：期望 ${expected}，实际 $trace_count"
    say "验收通过：$video_count 个视频 + $trace_count 个 trace"
    say "查看结果：$output/index.html"
}

infer_one() {
    local step="$1"
    local checkpoint="$SNAPSHOT_ROOT/checkpoint_stage2_g${step}"
    local output="$STAGE2_TRAIN_ROOT/inference_early_g${step}"
    local log="$STAGE2_TRAIN_ROOT/inference_early_g${step}.log"
    local config_tag="${INFERENCE_CONFIG##*/}"
    local plan_counts
    local expected_artifacts
    local base_samples
    local resolved_contract_hash
    local status
    local output_overridden=0
    local log_overridden=0

    config_tag="${config_tag%.yaml}"
    [[ "$config_tag" =~ ^[A-Za-z0-9._-]+$ ]] || \
        fail "推理配置文件名不能安全用于输出目录：$config_tag"
    if [[ "$INFERENCE_CONFIG" != "configs/infer_i2v_stage2_baseline.yaml" ]]; then
        output="${output}_${config_tag}"
        log="${log%.log}_${config_tag}.log"
    fi
    if [[ -n "$SWEEP_EVALUATION" ]]; then
        output="${output}_${SWEEP_EVALUATION}"
        log="${log%.log}_${SWEEP_EVALUATION}.log"
    fi

    if [[ -n "${STAGE2_EARLY_CHECKPOINT:-}" ]]; then
        checkpoint="$STAGE2_EARLY_CHECKPOINT"
    fi
    if [[ -n "${STAGE2_EARLY_OUTPUT:-}" ]]; then
        output="$STAGE2_EARLY_OUTPUT"
        output_overridden=1
    fi
    if [[ -n "${STAGE2_EARLY_LOG:-}" ]]; then
        log="$STAGE2_EARLY_LOG"
        log_overridden=1
    fi

    require_dir "$checkpoint" "快照权重目录"
    require_file "$checkpoint/_SUCCESS" "快照完成标记"
    require_file "$checkpoint/checkpoint_manifest.json" "权重 manifest"
    require_file "$checkpoint/provenance.json" "权重 provenance"
    export LONG_LIVE_STAGE2_INFERENCE_CHECKPOINT="$checkpoint"
    export LONG_LIVE_STAGE2_INFERENCE_OUTPUT="$output"

    plan_counts="$(
        run_stage2_cli \
            --print-plan-field expected_sample_count \
            --print-plan-field base_samples_per_profile \
            --print-plan-field resolved_contract_hash
    )"
    IFS=$'\t' read -r \
        expected_artifacts base_samples resolved_contract_hash <<< "$plan_counts"
    [[ "$expected_artifacts" =~ ^[1-9][0-9]*$ ]] || \
        fail "推理计划返回了非法产物数：$expected_artifacts"
    [[ "$base_samples" =~ ^[1-9][0-9]*$ ]] || \
        fail "推理计划返回了非法基础样本数：$base_samples"
    [[ "$resolved_contract_hash" =~ ^[0-9a-f]{64}$ ]] || \
        fail "推理计划返回了非法合同哈希：$resolved_contract_hash"

    if [[ "$INFERENCE_CONFIG" != "configs/infer_i2v_stage2_baseline.yaml" ]] && \
        [[ "$output_overridden" -eq 0 ]]; then
        output="${output}_${resolved_contract_hash:0:16}"
        if [[ "$log_overridden" -eq 0 ]]; then
            log="${log%.log}_${resolved_contract_hash:0:16}.log"
        fi
    fi
    if [[ -e "$output" && ! -d "$output" ]]; then
        fail "输出路径已存在但不是目录：$output"
    fi
    mkdir -p -- "$(dirname -- "$output")" "$(dirname -- "$log")"
    export LONG_LIVE_STAGE2_INFERENCE_OUTPUT="$output"
    CURRENT_LOG="$log"

    say "开始 G${step}：$(date '+%F %T %z')"
    say "权重：$checkpoint"
    say "输出：$output"
    say "配置：${INFERENCE_CONFIG}；计划产物：$expected_artifacts"
    say "GPU：${CUDA_VISIBLE_DEVICES}（$STAGE2_INFERENCE_NPROC 卡数据并行）"
    if ((base_samples % STAGE2_INFERENCE_NPROC != 0)); then
        say "提示：每 profile 有 $base_samples 个基础样本，不能被 $STAGE2_INFERENCE_NPROC 整除；跨 profile 的 latent cache 复用率会降低"
    fi
    if [[ -d "$output" ]]; then
        say "检测到已有输出，将校验并续跑完整 video+trace；不会覆盖或删除"
    fi
    say "模型与数据认证阶段可能较久；每 ${HEARTBEAT_SECONDS}s 会打印一次心跳"
    say "完整日志：$log"
    run_stage2_cli --plan-only | tee -a "$log"
    if [[ "$PLAN_ONLY" == "1" ]]; then
        say "仅规划模式完成；未初始化模型、CUDA 或 torchrun"
        CURRENT_LOG=""
        return
    fi

    acquire_output_lock "$output"
    launch_stage2_torchrun "$log"

    while kill -0 "$ACTIVE_PID" 2>/dev/null; do
        sleep "$HEARTBEAT_SECONDS" &
        wait "$!"
        if kill -0 "$ACTIVE_PID" 2>/dev/null; then
            say "仍在运行：G${step}，PID=${ACTIVE_PID}，$(date '+%F %T %z')" | tee -a "$log"
        fi
    done

    if wait "$ACTIVE_PID"; then
        status=0
    else
        status="$?"
    fi
    ACTIVE_PID=""
    if [[ "$status" -ne 0 ]]; then
        fail "G${step} 推理失败，状态码=${status}；查看日志：$log"
    fi

    verify_outputs "$output" "$expected_artifacts"
    release_output_lock
    say "G${step} 完成：$(date '+%F %T %z')"
    CURRENT_LOG=""
}

if [[ "$#" -eq 0 ]]; then
    set -- 70
fi
if [[ "$#" -gt 1 ]] && {
    [[ -n "${STAGE2_EARLY_CHECKPOINT:-}" ]] ||
    [[ -n "${STAGE2_EARLY_OUTPUT:-}" ]] ||
    [[ -n "${STAGE2_EARLY_LOG:-}" ]]
}; then
    fail "一次推理多个步数时不能使用 STAGE2_EARLY_CHECKPOINT/OUTPUT/LOG 覆盖"
fi

say "启动前检查中；训练进程不会被停止或修改"
validate_common_inputs
for requested_step in "$@"; do
    infer_one "$(normalize_step "$requested_step")"
done
say "全部任务完成"
