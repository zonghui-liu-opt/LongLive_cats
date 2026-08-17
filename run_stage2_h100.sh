#!/usr/bin/env bash
set -Eeuo pipefail

# One operator-facing entrypoint for the complete Stage-2 H100 lifecycle.
# It only orchestrates existing production commands; training and inference
# algorithms remain in train.py / infer_stage2_baseline.sh.

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_ROOT
export STAGE2_PROJECT_ROOT="$SCRIPT_ROOT"

export STAGE2_PYTHON="${STAGE2_PYTHON:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python}"
export STAGE2_TORCHRUN="${STAGE2_TORCHRUN:-$(dirname -- "$STAGE2_PYTHON")/torchrun}"
export STAGE2_WORK_ROOT="${STAGE2_WORK_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_stage2_new}"
export STAGE2_TRAIN_ROOT="${STAGE2_TRAIN_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_training}"

export ARCH_ROOT="${ARCH_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B}"
export TEACHER_CKPT="${TEACHER_CKPT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/ckpts}"
export TEACHER_PROVENANCE_RECORD="${TEACHER_PROVENANCE_RECORD:-$TEACHER_CKPT/merge_manifest.json}"
export STAGE1_BASE="${STAGE1_BASE:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/checkpoints/stage1/converted_causal_base.pt}"
export STAGE1_CKPT="${STAGE1_CKPT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/results/stage1_600cats_phaseA10epochs_phaseB20epochs/checkpoint_model_003075}"
export METADATA_600="${METADATA_600:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/metadata_600clips_480x832_buckets.csv}"
export STAGE1_CACHE_MANIFEST="${STAGE1_CACHE_MANIFEST:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/cache_480x832_buckets/ar_stage1_i2v_600cats/cache_manifest.json}"
export ACTION_SIDECAR_600="${ACTION_SIDECAR_600:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/action_labels_600cats.csv}"
export ATTEST_STAGE2_TEACHER="${ATTEST_STAGE2_TEACHER:-1}"
export STAGE2_CONFIG="${STAGE2_CONFIG:-$SCRIPT_ROOT/configs/train_i2v_stage2_600cats.yaml}"
export ACTIVE_CONFIG="$STAGE2_CONFIG"

export STAGE2_SMOKE_DIR="${STAGE2_SMOKE_DIR:-$STAGE2_TRAIN_ROOT/smoke_micro2_acc4}"
export STAGE2_FORMAL_DIR="${STAGE2_FORMAL_DIR:-$STAGE2_TRAIN_ROOT/formal_baseline}"
export STAGE2_B0_DIR="${STAGE2_B0_DIR:-$STAGE2_TRAIN_ROOT/formal_matched_b0}"
export STAGE2_B0_CONFIG="${STAGE2_B0_CONFIG:-$STAGE2_WORK_ROOT/configs/train_i2v_stage2_600cats_b0.yaml}"
export STAGE2_INFERENCE_OUTPUT="${STAGE2_INFERENCE_OUTPUT:-$STAGE2_TRAIN_ROOT/inference_g280_baseline}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$STAGE2_WORK_ROOT/pycache}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# prepare_stage2.sh runs in a child shell, so every invocation of this wrapper
# explicitly reconstructs the resolved runtime asset environment.
export LONG_LIVE_STAGE2_ARCHITECTURE_ROOT="$ARCH_ROOT"
export LONG_LIVE_STAGE2_GENERATOR_BASE="$STAGE2_WORK_ROOT/assets/stage1_step3075_ema_merged.pt"
export LONG_LIVE_STAGE2_GENERATOR_MANIFEST="$STAGE2_WORK_ROOT/assets/stage1_step3075_ema_merged.manifest.json"
export LONG_LIVE_STAGE2_REAL_SCORE_BASE="$TEACHER_CKPT"
export LONG_LIVE_STAGE2_REAL_SCORE_MANIFEST="$STAGE2_WORK_ROOT/assets/real_score_teacher.manifest.json"
export LONG_LIVE_STAGE2_METADATA_PATH="$METADATA_600"
export LONG_LIVE_STAGE2_SOURCE_MANIFEST="$STAGE2_WORK_ROOT/stage2_600cats_f25_v1/cache_manifest.attested.json"
export LONG_LIVE_STAGE2_ACTION_LABELS_PATH="$ACTION_SIDECAR_600"
export LONG_LIVE_STAGE2_CACHE_DIR="$STAGE2_WORK_ROOT/stage2_600cats_f25_v1"
export LONG_LIVE_STAGE2_NEGATIVE_MANIFEST="$STAGE2_WORK_ROOT/negative_v1/negative_conditioning_manifest.json"

export LONG_LIVE_STAGE2_INFERENCE_CHECKPOINT="$STAGE2_FORMAL_DIR/checkpoint_stage2_g000280"
export LONG_LIVE_STAGE2_T5_CHECKPOINT="$ARCH_ROOT/models_t5_umt5-xxl-enc-bf16.pth"
export LONG_LIVE_STAGE2_TOKENIZER_DIR="$ARCH_ROOT/google/umt5-xxl"
export LONG_LIVE_STAGE2_VAE_CHECKPOINT="$ARCH_ROOT/Wan2.2_VAE.pth"
export LONG_LIVE_STAGE2_INFERENCE_OUTPUT="$STAGE2_INFERENCE_OUTPUT"

DRY_RUN=0
CURRENT_STAGE="bootstrap"

usage() {
  cat <<'EOF'
Stage-2 H100 简明指导脚本

按顺序运行下面 6 条命令；每条只有看到对应 PASS 才能继续：

  1. bash run_stage2_h100.sh prepare
     作用：检查/准备 step3075、600条数据、198/202/200 动作、8×H100 FSDP2。
     正确：STAGE2_GUIDE_PREPARE=PASS

  2. bash run_stage2_h100.sh smoke
     作用：依次执行 C0冷启动、C1精确恢复DMD、C2精确恢复DFD。
     正确：STAGE2_GUIDE_SMOKE=PASS

  3. bash run_stage2_h100.sh train
     作用：正式训练 B1；中断后重复同一命令会自动恢复，最终到 G280。
     正确：STAGE2_GUIDE_TRAIN_B1=PASS

  4. bash run_stage2_h100.sh control
     作用：从 B1 的同一个 G240 分叉并训练纯DMD matched-control B0。
     正确：STAGE2_GUIDE_TRAIN_B0=PASS

  5. bash run_stage2_h100.sh plot
     作用：严格校验两条训练lineage并各生成9张PNG、9张SVG和HTML。
     正确：STAGE2_GUIDE_PLOT=PASS

  6. bash run_stage2_h100.sh infer
     作用：使用 B1 G280 Generator-EMA 生成56个视频和56份trace。
     正确：STAGE2_GUIDE_INFER=PASS samples=56

随时查看进度：
  bash run_stage2_h100.sh status

只查看将执行的命令（不会创建目录或启动GPU）：
  bash run_stage2_h100.sh --dry-run smoke

注意：没有 all 子命令。smoke、长训练、推理之间必须由操作者确认上一条 PASS。
任一步没有打印规定的 PASS 就立刻停止；保留该阶段日志，不要删除或手工拼接产物。
EOF
}

fail() {
  local message="$1"
  echo "ERROR: $message" >&2
  echo "STAGE2_GUIDE_RESULT=FAIL stage=$CURRENT_STAGE" >&2
  exit 1
}

unexpected_error() {
  local status=$?
  echo "STAGE2_GUIDE_RESULT=FAIL stage=$CURRENT_STAGE exit_code=$status" >&2
  exit "$status"
}
trap unexpected_error ERR

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

announce() {
  local purpose="$1"
  local marker="$2"
  echo "[作用] $purpose"
  echo "[通过条件] 命令退出为0、产物严格复验通过，最后打印：$marker"
}

require_runtime() {
  [[ -x "$STAGE2_PYTHON" ]] || fail "Python不可执行：$STAGE2_PYTHON"
  [[ -x "$STAGE2_TORCHRUN" ]] || fail "torchrun不可执行：$STAGE2_TORCHRUN"
  git -C "$SCRIPT_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
    fail "脚本必须位于Git checkout：$SCRIPT_ROOT"
  [[ -f "$STAGE2_CONFIG" ]] || fail "训练配置不存在：$STAGE2_CONFIG"
  [[ -f "$SCRIPT_ROOT/train.py" ]] || fail "缺少train.py"
}

require_clean_checkout() {
  local dirty
  dirty="$(git -C "$SCRIPT_ROOT" status --porcelain=v1 --untracked-files=all)"
  [[ -z "$dirty" ]] || fail "Git checkout不干净；请先提交本轮代码，并把运行产物放到仓库外。"
}

require_external_path() {
  local label="$1"
  local path="$2"
  "$STAGE2_PYTHON" -I -B - "$SCRIPT_ROOT" "$label" "$path" <<'PY'
from pathlib import Path
import sys

project = Path(sys.argv[1]).resolve(strict=True)
label = sys.argv[2]
target = Path(sys.argv[3]).expanduser().resolve(strict=False)
if target == project or project in target.parents:
    raise SystemExit(f"{label}必须位于Git checkout之外：{target}")
PY
}

require_distinct_run_paths() {
  "$STAGE2_PYTHON" -I -B - \
    "$STAGE2_SMOKE_DIR" "$STAGE2_FORMAL_DIR" "$STAGE2_B0_DIR" \
    "$STAGE2_INFERENCE_OUTPUT" <<'PY'
from itertools import combinations
from pathlib import Path
import sys

paths = [Path(value).expanduser().resolve(strict=False) for value in sys.argv[1:]]
for left, right in combinations(paths, 2):
    if left == right or left in right.parents or right in left.parents:
        raise SystemExit(f"Stage-2运行目录必须彼此独立且不能嵌套：{left} <> {right}")
PY
}

require_training_arm() {
  local config="$1"
  local expected="$2"
  "$STAGE2_PYTHON" -I -B - "$SCRIPT_ROOT" "$config" "$expected" <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from omegaconf import OmegaConf
from utils.stage2_config import resolve_stage2_config

resolved = resolve_stage2_config(OmegaConf.load(Path(sys.argv[2]).resolve(strict=True)))
expected = sys.argv[3]
if resolved.phase_b_mode != expected:
    raise SystemExit(
        f"Stage-2 config phase_b_mode must be {expected}, got {resolved.phase_b_mode}"
    )
PY
}

run_logged() {
  local log_path="$1"
  shift
  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_command "$@"
    echo "    stdout/stderr -> $log_path"
    return 0
  fi
  mkdir -p "$(dirname -- "$log_path")"
  "$@" 2>&1 | tee -a "$log_path"
}

torchrun_train_command() {
  local config="$1"
  local logdir="$2"
  shift 2
  print_command \
    "$STAGE2_TORCHRUN" --standalone --nnodes=1 --nproc-per-node=8 \
    --max-restarts=0 --no-python "$STAGE2_PYTHON" -I -B \
    "$SCRIPT_ROOT/train.py" --config_path "$config" --logdir "$logdir" \
    "$@" --no-visualize
}

run_torchrun_train() {
  local log_path="$1"
  local config="$2"
  local logdir="$3"
  shift 3
  run_logged "$log_path" \
    "$STAGE2_TORCHRUN" --standalone --nnodes=1 --nproc-per-node=8 \
    --max-restarts=0 --no-python "$STAGE2_PYTHON" -I -B \
    "$SCRIPT_ROOT/train.py" --config_path "$config" --logdir "$logdir" \
    "$@" --no-visualize
}

prepare_complete() {
  local log_path="$STAGE2_WORK_ROOT/logs/prepare_release.log"
  [[ -f "$log_path" ]] || return 1
  local marker
  for marker in \
    CHECK_1_TEACHER_PASS \
    CHECK_2_GENERATOR_PASS \
    CHECK_3_CONFIG_PASS \
    CHECK_4_DATA_PASS \
    CHECK_5_ROLE_INIT_PASS \
    CHECK_6_FSDP2_ACCUMULATION_PASS \
    STAGE2_PRETRAIN_PASS; do
    grep -Fqx "$marker" "$log_path" || return 1
  done
  grep -Fqx 'STAGE2_FSDP2_ACCUMULATION_GATE=PASS mode=H100-world8' "$log_path" || return 1
  [[ -f "$LONG_LIVE_STAGE2_GENERATOR_BASE" ]] || return 1
  [[ -f "$LONG_LIVE_STAGE2_GENERATOR_MANIFEST" ]] || return 1
  [[ -f "$LONG_LIVE_STAGE2_REAL_SCORE_MANIFEST" ]] || return 1
  [[ -f "$LONG_LIVE_STAGE2_SOURCE_MANIFEST" ]] || return 1
  [[ -f "$LONG_LIVE_STAGE2_NEGATIVE_MANIFEST" ]] || return 1
  [[ -f "$STAGE2_WORK_ROOT/role_init/ROLE_INIT_COMPLETE" ]] || return 1
  "$STAGE2_PYTHON" -I -B - \
    "$SCRIPT_ROOT" "$STAGE2_CONFIG" "$STAGE2_WORK_ROOT/run/resolved_config.json" <<'PY'
import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from omegaconf import OmegaConf
from utils.stage2_config import resolve_stage2_config

resolved = resolve_stage2_config(OmegaConf.load(Path(sys.argv[2]).resolve(strict=True)))
record = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
expected = {
    **resolved.to_dict(),
    "contract_sha256": resolved.contract_hash(),
    "launch_sha256": resolved.launch_hash(),
}
expected = json.loads(json.dumps(expected, allow_nan=False, sort_keys=True))
assert record == expected
PY
}

require_prepared() {
  prepare_complete || fail "prepare尚未完整通过；先运行：bash run_stage2_h100.sh prepare"
}

validate_smoke_mode() {
  local mode="$1"
  local metrics="$STAGE2_SMOKE_DIR/metrics/stage2_train_metrics.jsonl"
  [[ -f "$metrics" ]] || return 1
  "$STAGE2_PYTHON" -I -B - \
    "$SCRIPT_ROOT" "$metrics" "$mode" "$ACTIVE_CONFIG" <<'PY'
from pathlib import Path
import sys
import json

sys.path.insert(0, sys.argv[1])
from omegaconf import OmegaConf
from utils.stage2_checkpoint import validate_stage2_checkpoint
from utils.stage2_config import resolve_stage2_config
from utils.stage2_metrics import load_stage2_metrics, stage2_records_for_latest_lineage

metrics_path = Path(sys.argv[2]).resolve(strict=True)
records = load_stage2_metrics(metrics_path)
mode = sys.argv[3]
resolved = resolve_stage2_config(OmegaConf.load(Path(sys.argv[4]).resolve(strict=True)))
cycles = stage2_records_for_latest_lineage(records, record_type="cycle_summary")
matches = [record for record in cycles if record.get("smoke_mode") == mode]
assert len(matches) == 1
cycle = matches[0]
run_starts = {
    item["run_id"]: item for item in records if item.get("record_type") == "run_start"
}
run_start = run_starts[cycle["run_id"]]
assert run_start["config_contract_sha256"] == resolved.contract_hash()
assert run_start["config_launch_sha256"] == resolved.launch_hash()
expected_config = json.loads(
    json.dumps(resolved.to_dict(), allow_nan=False, sort_keys=True)
)
assert run_start["resolved_config"] == expected_config
acceptance = cycle.get("smoke_acceptance")
assert isinstance(acceptance, dict)
assert acceptance.get("smoke_mode") == mode
assert acceptance.get("status") == "PASS"
assert acceptance.get("nonfinite_attempts") == 0
assert cycle.get("nonfinite_attempts") == 0
assert acceptance.get("parent_next_f1_probe_consumed") is (mode in {"C1", "C2"})
if mode in {"C0", "C1"}:
    assert isinstance(acceptance.get("next_f1_probe"), dict)
ends = stage2_records_for_latest_lineage(records, record_type="run_end")
matching_ends = [record for record in ends if record.get("smoke_mode") == mode]
# C0/C1 publish their authenticated checkpoint before appending run_end.  A
# hard interruption in that tiny window must still allow the next smoke mode
# to consume the committed checkpoint.  C2 has no checkpoint, so it requires
# its terminal run_end.
if mode == "C2":
    assert len(matching_ends) == 1
    assert matching_ends[0].get("status") == "smoke_complete"
else:
    assert len(matching_ends) <= 1
    if matching_ends:
        assert matching_ends[0].get("status") == "smoke_complete"
train_steps = stage2_records_for_latest_lineage(records, record_type="train_step")
generators = [
    item
    for item in train_steps
    if item.get("smoke_mode") == mode and item.get("role") == "generator"
]
assert len(generators) == 1
generator = generators[0]
expected_g = {"C0": 1, "C1": 2, "C2": 3}[mode]
expected_branch = "dfd" if mode == "C2" else "dmd"
expected_probability = 1.0 if mode == "C2" else 0.0
assert generator["completed_generator_updates"] == expected_g
assert generator["branch"] == expected_branch
assert generator["branch_is_dfd"] == int(mode == "C2")
assert generator["dfd_probability"] == expected_probability
if mode in {"C0", "C1"}:
    completed_g = 1 if mode == "C0" else 2
    checkpoint = metrics_path.parents[1] / f"checkpoint_stage2_g{completed_g:06d}"
    checkpoint_manifest = validate_stage2_checkpoint(
        checkpoint,
        expected_contract_hash=resolved.contract_hash(),
        expected_phase_b_mode=resolved.phase_b_mode,
    )
    assert checkpoint_manifest["config"]["launch_hash"] == resolved.launch_hash()
    assert metrics_path.read_bytes().startswith(
        (checkpoint / "metrics_lineage.jsonl").read_bytes()
    )
PY
}

smoke_mode_complete() {
  local mode="$1"
  validate_smoke_mode "$mode" >/dev/null 2>&1 || return 1
  case "$mode" in
    C0) [[ -f "$STAGE2_SMOKE_DIR/checkpoint_stage2_g000001/_SUCCESS" ]] ;;
    C1) [[ -f "$STAGE2_SMOKE_DIR/checkpoint_stage2_g000002/_SUCCESS" ]] ;;
    C2) [[ ! -e "$STAGE2_SMOKE_DIR/checkpoint_stage2_g000003/_SUCCESS" ]] ;;
    *) return 1 ;;
  esac
}

smoke_complete() {
  smoke_mode_complete C0 && smoke_mode_complete C1 && smoke_mode_complete C2
}

validate_formal_metrics() {
  local root="$1"
  local config="$2"
  local expected_phase_b_mode="$3"
  local metrics="$root/metrics/stage2_train_metrics.jsonl"
  [[ -f "$metrics" ]] || return 1
  "$STAGE2_PYTHON" -I -B - \
    "$SCRIPT_ROOT" "$root" "$config" "$expected_phase_b_mode" <<'PY'
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from omegaconf import OmegaConf
from utils.stage2_checkpoint import validate_stage2_checkpoint
from utils.stage2_config import resolve_stage2_config
from utils.stage2_metrics import load_stage2_metrics, stage2_records_for_latest_lineage

root = Path(sys.argv[2]).resolve(strict=True)
config = Path(sys.argv[3]).resolve(strict=True)
expected_phase_b_mode = sys.argv[4]
resolved = resolve_stage2_config(OmegaConf.load(config))
assert resolved.phase_b_mode == expected_phase_b_mode
checkpoint = root / "checkpoint_stage2_g000280"
checkpoint_manifest = validate_stage2_checkpoint(
    checkpoint,
    expected_contract_hash=resolved.contract_hash(),
    expected_phase_b_mode=expected_phase_b_mode,
)
assert checkpoint_manifest["completed_generator_updates"] == 280
records = load_stage2_metrics(root / "metrics" / "stage2_train_metrics.jsonl")
ends = stage2_records_for_latest_lineage(records, record_type="run_end")
assert ends
record = ends[-1]
assert record.get("status") == "complete"
assert record.get("dry_run") is False
assert record.get("smoke_mode") is None
assert record.get("completed_generator_updates") == 280
assert record.get("completed_fake_updates") == 1400
assert record.get("completed_cycles") == 280
run_starts = {
    item["run_id"]: item for item in records if item.get("record_type") == "run_start"
}
run_start = run_starts[record["run_id"]]
assert run_start["resolved_config"]["derived"]["phase_b_mode"] == expected_phase_b_mode
assert run_start["config_contract_sha256"] == resolved.contract_hash()
assert run_start["config_launch_sha256"] == resolved.launch_hash()
assert checkpoint_manifest["config"]["contract_hash"] == run_start["config_contract_sha256"]
assert checkpoint_manifest["config"]["launch_hash"] == run_start["config_launch_sha256"]
train_steps = stage2_records_for_latest_lineage(records, record_type="train_step")
generator = [item for item in train_steps if item.get("role") == "generator"]
fake_score = [item for item in train_steps if item.get("role") == "fake_score"]
cycles = stage2_records_for_latest_lineage(records, record_type="cycle_summary")
assert [item["completed_generator_updates"] for item in generator] == list(
    range(1, 281)
)
assert [item["completed_fake_updates"] for item in fake_score] == list(
    range(1, 1401)
)
assert [item["completed_cycles"] for item in cycles] == list(range(1, 281))
checkpoint_metrics = (checkpoint / "metrics_lineage.jsonl").read_bytes()
live_metrics = (root / "metrics" / "stage2_train_metrics.jsonl").read_bytes()
assert live_metrics.startswith(checkpoint_metrics)
PY
}

formal_complete() {
  local root="$1"
  local config="$2"
  local expected_phase_b_mode="$3"
  [[ -f "$root/checkpoint_stage2_g000280/_SUCCESS" ]] || return 1
  validate_formal_metrics "$root" "$config" "$expected_phase_b_mode" >/dev/null 2>&1
}

plot_set_complete() {
  local root="$1"
  [[ -f "$root/index.html" ]] || return 1
  local png_count svg_count
  png_count="$(find "$root" -maxdepth 1 -type f -name '*.png' | wc -l | tr -d '[:space:]')"
  svg_count="$(find "$root" -maxdepth 1 -type f -name '*.svg' | wc -l | tr -d '[:space:]')"
  [[ "$png_count" == 9 && "$svg_count" == 9 ]]
}

plots_complete() {
  plot_set_complete "$STAGE2_FORMAL_DIR/plots" && plot_set_complete "$STAGE2_B0_DIR/plots"
}

inference_complete() {
  local root="$STAGE2_INFERENCE_OUTPUT"
  [[ -f "$root/manifest.json" && -f "$root/index.html" ]] || return 1
  "$STAGE2_PYTHON" -I -B - \
    "$SCRIPT_ROOT" "$root" "$ACTIVE_CONFIG" \
    "$STAGE2_FORMAL_DIR/checkpoint_stage2_g000280" \
    "$SCRIPT_ROOT/configs/infer_i2v_stage2_baseline.yaml" <<'PY'
import json
from pathlib import Path
import sys

sys.path.insert(0, sys.argv[1])
from omegaconf import OmegaConf
from utils.stage2_checkpoint import validate_stage2_checkpoint
from utils.stage2_config import resolve_stage2_config
from utils.stage2_inference_batch import build_stage2_inference_samples
from utils.stage2_inference_config import load_stage2_inference_config
from utils.stage2_inference_artifacts import validate_stage2_inference_manifest_artifacts

root = Path(sys.argv[2]).resolve(strict=True)
resolved = resolve_stage2_config(OmegaConf.load(Path(sys.argv[3]).resolve(strict=True)))
checkpoint = Path(sys.argv[4]).resolve(strict=True)
inference_config = load_stage2_inference_config(Path(sys.argv[5]).resolve(strict=True))
samples = build_stage2_inference_samples(
    single_metadata=inference_config.single_metadata,
    two_action_metadata=inference_config.two_action_metadata,
    seeds=inference_config.seeds,
    profiles=inference_config.profiles,
)
checkpoint_manifest = validate_stage2_checkpoint(
    checkpoint,
    expected_contract_hash=resolved.contract_hash(),
    expected_phase_b_mode="dmd_dfd",
)
ema_entries = [
    item
    for item in checkpoint_manifest["files"]
    if item.get("name") == "generator_ema.safetensors"
]
assert len(ema_entries) == 1
expected_checkpoint = {
    "directory": str(checkpoint),
    "manifest_sha256": checkpoint_manifest["manifest_sha256"],
    "completed_generator_updates": 280,
    "contract_hash": resolved.contract_hash(),
    "generator_ema_sha256": ema_entries[0]["sha256"],
}
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
validated = validate_stage2_inference_manifest_artifacts(
    root,
    manifest,
    expected_checkpoint=expected_checkpoint,
    expected_resolved_config=inference_config.to_dict(),
    expected_samples=samples,
)
assert validated["status"] == "complete"
assert validated["expected_sample_count"] == 56
PY
}

run_prepare() {
  CURRENT_STAGE="prepare"
  announce \
    "准备并验证step3075、198/202/200动作数据、模型角色和8×H100累积梯度" \
    "STAGE2_GUIDE_PREPARE=PASS"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_command bash "$SCRIPT_ROOT/prepare_stage2.sh"
    return 0
  fi
  require_runtime
  require_training_arm "$ACTIVE_CONFIG" dmd_dfd
  require_clean_checkout
  require_distinct_run_paths
  require_external_path STAGE2_WORK_ROOT "$STAGE2_WORK_ROOT"
  require_external_path STAGE2_TRAIN_ROOT "$STAGE2_TRAIN_ROOT"
  mkdir -p "$STAGE2_WORK_ROOT/logs" "$STAGE2_TRAIN_ROOT"
  : >"$STAGE2_WORK_ROOT/logs/prepare_release.log"
  run_logged "$STAGE2_WORK_ROOT/logs/prepare_release.log" bash "$SCRIPT_ROOT/prepare_stage2.sh"
  prepare_complete || fail "prepare命令结束，但六项门禁或产物复验不完整。"
  echo "STAGE2_GUIDE_PREPARE=PASS"
  echo "NEXT: bash run_stage2_h100.sh smoke"
}

run_smoke() {
  CURRENT_STAGE="smoke"
  announce \
    "依次执行C0冷启动、C1精确恢复纯DMD、C2精确恢复强制DFD" \
    "STAGE2_GUIDE_SMOKE=PASS"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    local mode
    for mode in C0 C1 C2; do
      torchrun_train_command "$ACTIVE_CONFIG" "$STAGE2_SMOKE_DIR" \
        --stage2-smoke "$mode"
    done
    return 0
  fi
  require_runtime
  require_training_arm "$ACTIVE_CONFIG" dmd_dfd
  require_clean_checkout
  require_distinct_run_paths
  require_external_path STAGE2_SMOKE_DIR "$STAGE2_SMOKE_DIR"
  require_prepared
  mkdir -p "$STAGE2_SMOKE_DIR"
  local mode
  for mode in C0 C1 C2; do
    if smoke_mode_complete "$mode"; then
      echo "STAGE2_SMOKE_${mode}=ALREADY_PASS"
      continue
    fi
    run_torchrun_train "$STAGE2_SMOKE_DIR/${mode}.log" \
      "$ACTIVE_CONFIG" "$STAGE2_SMOKE_DIR" --stage2-smoke "$mode"
    smoke_mode_complete "$mode" || fail "$mode 退出后未通过metrics/checkpoint精确复验。"
    echo "STAGE2_SMOKE_${mode}=PASS"
  done
  smoke_complete || fail "C0/C1/C2未形成完整smoke lineage。"
  echo "STAGE2_GUIDE_SMOKE=PASS"
  echo "NEXT: bash run_stage2_h100.sh train"
}

run_train_b1() {
  CURRENT_STAGE="train-b1"
  announce \
    "正式训练B1到G280；重复本命令会从同目录最新完整checkpoint自动恢复" \
    "STAGE2_GUIDE_TRAIN_B1=PASS"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    torchrun_train_command "$ACTIVE_CONFIG" "$STAGE2_FORMAL_DIR"
    return 0
  fi
  require_runtime
  require_training_arm "$ACTIVE_CONFIG" dmd_dfd
  require_clean_checkout
  require_distinct_run_paths
  require_external_path STAGE2_FORMAL_DIR "$STAGE2_FORMAL_DIR"
  require_prepared
  smoke_complete || fail "smoke尚未完整通过；先运行：bash run_stage2_h100.sh smoke"
  if formal_complete "$STAGE2_FORMAL_DIR" "$ACTIVE_CONFIG" dmd_dfd; then
    echo "STAGE2_TRAIN_B1=ALREADY_PASS"
  else
    mkdir -p "$STAGE2_FORMAL_DIR"
    run_torchrun_train "$STAGE2_FORMAL_DIR/formal.log" \
      "$ACTIVE_CONFIG" "$STAGE2_FORMAL_DIR"
  fi
  formal_complete "$STAGE2_FORMAL_DIR" "$ACTIVE_CONFIG" dmd_dfd || \
    fail "B1未到G280、phase arm错误或run_end不是complete。"
  echo "STAGE2_GUIDE_TRAIN_B1=PASS"
  echo "NEXT: bash run_stage2_h100.sh control"
}

ensure_b0_config() {
  local anchor="$STAGE2_FORMAL_DIR/checkpoint_stage2_g000240"
  "$STAGE2_PYTHON" -I -B - \
    "$ACTIVE_CONFIG" "$STAGE2_B0_CONFIG" "$anchor" <<'PY'
from pathlib import Path
import os
import sys

from omegaconf import OmegaConf

source, destination, anchor = map(Path, sys.argv[1:])
config = OmegaConf.load(source)
config.checkpoints.init_from_stage1 = None
config.checkpoints.resume_stage2 = str(anchor.resolve(strict=True))
config.training.phase_b_mode = "dmd_only"
config.training.phase_b_dfd_probability_max = 0.0
expected = OmegaConf.to_container(config, resolve=False)
if destination.exists():
    actual = OmegaConf.to_container(OmegaConf.load(destination), resolve=False)
    if actual != expected:
        raise SystemExit(f"existing B0 config differs from the locked transform: {destination}")
else:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    OmegaConf.save(config, temporary)
    os.replace(temporary, destination)
PY
}

run_train_b0() {
  CURRENT_STAGE="train-b0"
  announce \
    "从B1的同一G240分叉，正式训练纯DMD matched-control B0到G280" \
    "STAGE2_GUIDE_TRAIN_B0=PASS"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "  生成仓库外B0配置：$STAGE2_B0_CONFIG"
    echo "  固定变换：init_from_stage1=null, resume_stage2=G240, phase_b_mode=dmd_only, dfd_probability=0"
    torchrun_train_command "$STAGE2_B0_CONFIG" "$STAGE2_B0_DIR"
    return 0
  fi
  require_runtime
  require_clean_checkout
  require_distinct_run_paths
  require_external_path STAGE2_B0_DIR "$STAGE2_B0_DIR"
  require_external_path STAGE2_B0_CONFIG "$STAGE2_B0_CONFIG"
  require_prepared
  formal_complete "$STAGE2_FORMAL_DIR" "$ACTIVE_CONFIG" dmd_dfd || \
    fail "B1尚未完整到G280。"
  [[ -f "$STAGE2_FORMAL_DIR/checkpoint_stage2_g000240/_SUCCESS" ]] || \
    fail "缺少B1共同父点G240。"
  ensure_b0_config
  if formal_complete "$STAGE2_B0_DIR" "$STAGE2_B0_CONFIG" dmd_only; then
    echo "STAGE2_TRAIN_B0=ALREADY_PASS"
  else
    mkdir -p "$STAGE2_B0_DIR"
    run_torchrun_train "$STAGE2_B0_DIR/formal.log" \
      "$STAGE2_B0_CONFIG" "$STAGE2_B0_DIR"
  fi
  formal_complete "$STAGE2_B0_DIR" "$STAGE2_B0_CONFIG" dmd_only || \
    fail "B0未到G280、phase arm错误或run_end不是complete。"
  echo "STAGE2_GUIDE_TRAIN_B0=PASS"
  echo "NEXT: bash run_stage2_h100.sh plot"
}

run_plot() {
  CURRENT_STAGE="plot"
  announce \
    "严格校验B1/B0完整lineage与时间闭合，并各生成9张PNG、9张SVG和HTML" \
    "STAGE2_GUIDE_PLOT=PASS"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_command "$STAGE2_PYTHON" -B "$SCRIPT_ROOT/scripts/plot_stage2_training.py" \
      --jsonl "$STAGE2_FORMAL_DIR/metrics/stage2_train_metrics.jsonl" \
      --output-dir "$STAGE2_FORMAL_DIR/plots" --formats png svg --require-complete
    print_command "$STAGE2_PYTHON" -B "$SCRIPT_ROOT/scripts/plot_stage2_training.py" \
      --jsonl "$STAGE2_B0_DIR/metrics/stage2_train_metrics.jsonl" \
      --output-dir "$STAGE2_B0_DIR/plots" --formats png svg --require-complete
    return 0
  fi
  require_runtime
  require_clean_checkout
  require_distinct_run_paths
  require_external_path STAGE2_FORMAL_PLOTS "$STAGE2_FORMAL_DIR/plots"
  require_external_path STAGE2_B0_PLOTS "$STAGE2_B0_DIR/plots"
  formal_complete "$STAGE2_FORMAL_DIR" "$ACTIVE_CONFIG" dmd_dfd || \
    fail "B1尚未完整训练。"
  formal_complete "$STAGE2_B0_DIR" "$STAGE2_B0_CONFIG" dmd_only || \
    fail "B0尚未完整训练。"
  "$STAGE2_PYTHON" -B "$SCRIPT_ROOT/scripts/plot_stage2_training.py" \
    --jsonl "$STAGE2_FORMAL_DIR/metrics/stage2_train_metrics.jsonl" \
    --output-dir "$STAGE2_FORMAL_DIR/plots" \
    --formats png svg --require-complete
  "$STAGE2_PYTHON" -B "$SCRIPT_ROOT/scripts/plot_stage2_training.py" \
    --jsonl "$STAGE2_B0_DIR/metrics/stage2_train_metrics.jsonl" \
    --output-dir "$STAGE2_B0_DIR/plots" \
    --formats png svg --require-complete
  plots_complete || fail "绘图命令结束，但两组9×PNG/9×SVG/index.html不完整。"
  echo "STAGE2_GUIDE_PLOT=PASS"
  echo "B1_HTML=$STAGE2_FORMAL_DIR/plots/index.html"
  echo "B0_HTML=$STAGE2_B0_DIR/plots/index.html"
  echo "NEXT: bash run_stage2_h100.sh infer"
}

run_infer() {
  CURRENT_STAGE="infer"
  announce \
    "加载B1 G280 Generator-EMA，生成24个单动作和32个双动作视频" \
    "STAGE2_GUIDE_INFER=PASS samples=56"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_command bash "$SCRIPT_ROOT/infer_stage2_baseline.sh"
    return 0
  fi
  require_runtime
  require_clean_checkout
  require_distinct_run_paths
  require_external_path STAGE2_INFERENCE_OUTPUT "$STAGE2_INFERENCE_OUTPUT"
  require_external_path STAGE2_TRAIN_ROOT "$STAGE2_TRAIN_ROOT"
  require_prepared
  formal_complete "$STAGE2_FORMAL_DIR" "$ACTIVE_CONFIG" dmd_dfd || \
    fail "B1尚未完整到G280。"
  if inference_complete >/dev/null 2>&1; then
    echo "STAGE2_INFERENCE=ALREADY_PASS"
  else
    if [[ -d "$STAGE2_INFERENCE_OUTPUT" ]] && \
       find "$STAGE2_INFERENCE_OUTPUT" -mindepth 1 -print -quit | grep -q .; then
      fail "推理目录非空但不是完整批次；请保留/归档整个目录，并用新的STAGE2_INFERENCE_OUTPUT重跑。"
    fi
    run_logged "$STAGE2_TRAIN_ROOT/inference_g280_baseline.log" \
      bash "$SCRIPT_ROOT/infer_stage2_baseline.sh"
  fi
  inference_complete || fail "推理结束，但56个MP4、56个trace或manifest/index复验失败。"
  echo "STAGE2_GUIDE_INFER=PASS samples=56"
  echo "OPEN_FOR_VISUAL_REVIEW=$STAGE2_INFERENCE_OUTPUT/index.html"
}

status_item() {
  local name="$1"
  shift
  if "$@" >/dev/null 2>&1; then
    echo "STAGE2_STATUS_${name}=PASS"
    return 0
  fi
  echo "STAGE2_STATUS_${name}=INCOMPLETE"
  return 1
}

run_status() {
  CURRENT_STAGE="status"
  require_runtime
  require_distinct_run_paths
  local all_complete=0
  status_item PREPARE prepare_complete || all_complete=1
  status_item SMOKE smoke_complete || all_complete=1
  status_item TRAIN_B1 formal_complete \
    "$STAGE2_FORMAL_DIR" "$ACTIVE_CONFIG" dmd_dfd || all_complete=1
  status_item TRAIN_B0 formal_complete \
    "$STAGE2_B0_DIR" "$STAGE2_B0_CONFIG" dmd_only || all_complete=1
  status_item PLOT plots_complete || all_complete=1
  status_item INFER inference_complete || all_complete=1
  if [[ "$all_complete" -eq 0 ]]; then
    echo "STAGE2_STATUS=COMPLETE"
  else
    echo "STAGE2_STATUS=INCOMPLETE"
  fi
}

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi

COMMAND="${1:-help}"
shift || true
[[ "$#" -eq 0 ]] || fail "不接受额外参数：$*"

cd -- "$SCRIPT_ROOT"
case "$COMMAND" in
  help|-h|--help) usage ;;
  prepare|1-prepare) run_prepare ;;
  smoke|2-smoke) run_smoke ;;
  train|3-train) run_train_b1 ;;
  control|b0|4-control) run_train_b0 ;;
  plot|5-plot) run_plot ;;
  infer|6-infer) run_infer ;;
  status) run_status ;;
  *)
    usage >&2
    fail "未知子命令：$COMMAND"
    ;;
esac
