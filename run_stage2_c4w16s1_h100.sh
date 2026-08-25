#!/usr/bin/env bash
set -Eeuo pipefail

# Thin, topology-locked launcher for the C4/W16/S1 Stage-2 experiment.
# The production lifecycle remains owned by run_stage2_h100.sh; this wrapper
# supplies an isolated config/lineage and adds the all-epoch completion gate.

SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_ROOT
readonly BASE_LAUNCHER="$SCRIPT_ROOT/run_stage2_h100.sh"
readonly C4_CONFIG="$SCRIPT_ROOT/configs/train_i2v_stage2_600cats_c4w16s1_micro1_acc8.yaml"
readonly EXPECTED_PROFILE="h100_c4w16s1_micro1_acc8_longrun"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

STAGE2_GPUS="${STAGE2_GPUS:-8}"
case "$STAGE2_GPUS" in
  4) STAGE2_TOPOLOGY_SUFFIX="_4gpus" ;;
  8) STAGE2_TOPOLOGY_SUFFIX="" ;;
  *) fail "STAGE2_GPUS must be 4 or 8, got: $STAGE2_GPUS" ;;
esac
export STAGE2_GPUS

if [[ -n "${STAGE2_CONFIG:-}" && "$STAGE2_CONFIG" != "$C4_CONFIG" ]]; then
  fail "本入口禁止覆盖STAGE2_CONFIG；C4/W16/S1固定使用：$C4_CONFIG"
fi
export STAGE2_CONFIG="$C4_CONFIG"

export STAGE2_PYTHON="${STAGE2_PYTHON:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python}"
export STAGE2_WORK_ROOT="${STAGE2_WORK_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_stage2_h100_c4w16s1_micro1_acc8_longrun${STAGE2_TOPOLOGY_SUFFIX}}"
export STAGE2_TRAIN_ROOT="${STAGE2_TRAIN_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_training_h100_c4w16s1_micro1_acc8_longrun${STAGE2_TOPOLOGY_SUFFIX}}"
export STAGE2_SMOKE_DIR="${STAGE2_SMOKE_DIR:-$STAGE2_TRAIN_ROOT/smoke_c4w16s1}"
export STAGE2_FORMAL_DIR="${STAGE2_FORMAL_DIR:-$STAGE2_TRAIN_ROOT/formal_b1_c4w16s1_all_epochs}"
export STAGE2_B0_DIR="${STAGE2_B0_DIR:-$STAGE2_TRAIN_ROOT/formal_b0_c4w16s1_all_epochs}"
export STAGE2_B0_CONFIG="${STAGE2_B0_CONFIG:-$STAGE2_WORK_ROOT/configs/train_i2v_stage2_c4w16s1_b0.yaml}"
export STAGE2_INFERENCE_OUTPUT="${STAGE2_INFERENCE_OUTPUT:-$STAGE2_TRAIN_ROOT/inference_g004000_c4w16s1}"
export STAGE2_INFERENCE_ROOT="${STAGE2_INFERENCE_ROOT:-$STAGE2_TRAIN_ROOT/inference_batch_c4w16s1}"
export STAGE2_INFERENCE_CHECKPOINT_ROOT="${STAGE2_INFERENCE_CHECKPOINT_ROOT:-$STAGE2_FORMAL_DIR}"
export STAGE2_LIVE_PLOT_DIR="${STAGE2_LIVE_PLOT_DIR:-$STAGE2_FORMAL_DIR/plots_live}"

usage() {
  cat <<'EOF'
Stage-2 C4/W16/S1 H100 训练入口

锁定合同：
  - 4 current latent + 12 history latent + 1 global initial-image sink
  - K4 / 24 future latents / global batch 64
  - 8卡 micro1×acc8，或4卡 micro1×acc16
  - A360+B40；Generator/Fake-score LR=5e-6/1e-6
  - 每个generator epoch（10G）保存完整checkpoint，共400个；不按milestone清理

一条命令完成 prepare → C0/C1/C2 smoke → 可恢复正式训练：
  bash run_stage2_c4w16s1_h100.sh all

也可分步执行：
  bash run_stage2_c4w16s1_h100.sh prepare
  bash run_stage2_c4w16s1_h100.sh smoke
  bash run_stage2_c4w16s1_h100.sh train
  bash run_stage2_c4w16s1_h100.sh verify
  bash run_stage2_c4w16s1_h100.sh plot-live

预览命令而不写文件或启动GPU：
  bash run_stage2_c4w16s1_h100.sh --dry-run all

注意：这是400个完整可恢复checkpoint，按当前LoRA/Adam状态估算需预留约1TiB。
G10/G20/G30保存raw G/F权重；Generator EMA沿用原合同，从G40开始存在。
EOF
}

run_base() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    bash "$BASE_LAUNCHER" --dry-run "$@"
  else
    bash "$BASE_LAUNCHER" "$@"
  fi
}

verify_all_epoch_checkpoints() {
  [[ "$DRY_RUN" -eq 0 ]] || {
    echo "[dry-run] verify 400 epoch checkpoints under $STAGE2_FORMAL_DIR"
    return 0
  }
  [[ -x "$STAGE2_PYTHON" ]] || fail "Python不可执行：$STAGE2_PYTHON"
  "$STAGE2_PYTHON" -I -B - \
    "$SCRIPT_ROOT" "$C4_CONFIG" "$STAGE2_FORMAL_DIR" "$EXPECTED_PROFILE" <<'PY'
from pathlib import Path
import re
import sys

project_root = Path(sys.argv[1]).resolve(strict=True)
config_path = Path(sys.argv[2]).resolve(strict=True)
run_root = Path(sys.argv[3]).resolve(strict=True)
expected_profile = sys.argv[4]
sys.path.insert(0, str(project_root))

from omegaconf import OmegaConf
from utils.stage2_checkpoint import validate_stage2_checkpoint
from utils.stage2_config import resolve_stage2_config
from utils.stage2_metrics import load_stage2_metrics, stage2_records_for_latest_lineage

resolved = resolve_stage2_config(OmegaConf.load(config_path))
if resolved.profile != expected_profile:
    raise SystemExit(
        f"C4/W16/S1 profile mismatch: {resolved.profile!r} != {expected_profile!r}"
    )
expected = set(
    range(
        resolved.checkpoint_interval_generator_updates,
        resolved.total_generator_updates + 1,
        resolved.checkpoint_interval_generator_updates,
    )
)
expected.update(resolved.milestone_generator_updates)
if len(expected) != resolved.phase_a_epochs + resolved.phase_b_epochs:
    raise SystemExit(
        f"expected one checkpoint per epoch, got {len(expected)} checkpoints for "
        f"{resolved.phase_a_epochs + resolved.phase_b_epochs} epochs"
    )

pattern = re.compile(r"checkpoint_stage2_g([0-9]{6})")
actual = {}
for child in run_root.iterdir():
    match = pattern.fullmatch(child.name)
    if match is None:
        continue
    step = int(match.group(1))
    if child.is_symlink() or not child.is_dir():
        raise SystemExit(f"checkpoint is not a real directory: {child}")
    actual[step] = child
if set(actual) != expected:
    missing = sorted(expected - set(actual))
    extra = sorted(set(actual) - expected)
    raise SystemExit(
        f"all-epoch checkpoint set mismatch: missing={missing[:20]}, extra={extra[:20]}"
    )
for step, directory in sorted(actual.items()):
    for filename in ("_SUCCESS", "checkpoint_manifest.json"):
        path = directory / filename
        if path.is_symlink() or not path.is_file():
            raise SystemExit(f"G{step} missing authenticated {filename}: {path}")

records = load_stage2_metrics(run_root / resolved.jsonl_path)
events = stage2_records_for_latest_lineage(records, record_type="checkpoint_event")
event_steps = [int(item["completed_generator_updates"]) for item in events]
# Checkpoint publication intentionally precedes its metrics event.  A crash in
# that narrow window leaves an authenticated directory but no event, so the
# exact directory set above is authoritative and events are advisory audits.
if event_steps != sorted(event_steps) or len(event_steps) != len(set(event_steps)):
    raise SystemExit(
        "checkpoint_event clock must be strictly increasing without duplicates"
    )
unexpected_event_steps = sorted(set(event_steps) - expected)
if unexpected_event_steps:
    raise SystemExit(
        f"checkpoint_event contains non-epoch steps: {unexpected_event_steps[:20]}"
    )
if any(item.get("removed") not in (None, []) for item in events):
    raise SystemExit("keep-all profile unexpectedly removed an epoch checkpoint")

terminal = actual[resolved.total_generator_updates]
validate_stage2_checkpoint(
    terminal,
    expected_contract_hash=resolved.contract_hash(),
    expected_phase_b_mode=resolved.phase_b_mode,
)
print(
    "STAGE2_C4W16S1_ALL_EPOCHS=PASS "
    f"checkpoints={len(expected)} first=G{min(expected)} last=G{max(expected)}"
)
PY
}

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
  shift
fi
COMMAND="${1:-help}"
shift || true
[[ "$#" -eq 0 ]] || fail "不接受额外参数：$*"

case "$COMMAND" in
  help|-h|--help)
    usage
    ;;
  prepare|smoke|plot-live)
    run_base "$COMMAND"
    ;;
  train)
    run_base train
    verify_all_epoch_checkpoints
    ;;
  verify|status)
    verify_all_epoch_checkpoints
    ;;
  all)
    run_base prepare
    run_base smoke
    run_base train
    verify_all_epoch_checkpoints
    ;;
  *)
    usage >&2
    fail "未知子命令：$COMMAND"
    ;;
esac
