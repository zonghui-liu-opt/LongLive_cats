from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).parents[1]
GUIDE = PROJECT_ROOT / "run_stage2_h100.sh"
DEFAULT_TRAIN_CONFIG = (
    PROJECT_ROOT / "configs" / "train_i2v_stage2_600cats_micro1_acc8.yaml"
)
BASELINE_TRAIN_CONFIG = PROJECT_ROOT / "configs" / "train_i2v_stage2_600cats.yaml"


def _run(
    *arguments: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(GUIDE), *arguments],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _isolated_env(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    work_root = tmp_path / "work"
    train_root = tmp_path / "train"
    true_executable = shutil.which("true")
    assert true_executable is not None
    env = dict(os.environ)
    env.update(
        {
            "STAGE2_PYTHON": sys.executable,
            "STAGE2_TORCHRUN": true_executable,
            "STAGE2_WORK_ROOT": str(work_root),
            "STAGE2_TRAIN_ROOT": str(train_root),
        }
    )
    for name in (
        "STAGE2_CONFIG",
        "ACTIVE_CONFIG",
        "STAGE2_SMOKE_DIR",
        "STAGE2_FORMAL_DIR",
        "STAGE2_B0_DIR",
        "STAGE2_B0_CONFIG",
        "STAGE2_INFERENCE_OUTPUT",
        "STAGE2_INFERENCE_ROOT",
        "STAGE2_INFERENCE_CHECKPOINT_ROOT",
        "STAGE2_INFERENCE_NPROC",
        "STAGE2_INFERENCE_HEARTBEAT_SECONDS",
    ):
        env.pop(name, None)
    return env, work_root, train_root


def test_stage2_h100_guide_has_valid_shell_and_plain_operator_help():
    syntax = subprocess.run(
        ["bash", "-n", str(GUIDE)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    completed = _run("help")
    assert completed.returncode == 0, completed.stderr
    required = (
        "train_i2v_stage2_600cats_micro1_acc8.yaml",
        "8卡×micro1×acc8=global64",
        "A360/B40",
        "G/F LR=1e-5/2e-6",
        "bash run_stage2_h100.sh prepare",
        "bash run_stage2_h100.sh smoke",
        "bash run_stage2_h100.sh train",
        "bash run_stage2_h100.sh control",
        "bash run_stage2_h100.sh plot-live",
        "bash run_stage2_h100.sh plot",
        "bash run_stage2_h100.sh infer",
        "bash run_stage2_h100.sh status",
        "step3075",
        "198/202/200",
        "STAGE2_GUIDE_PREPARE=PASS",
        "STAGE2_GUIDE_SMOKE=PASS",
        "STAGE2_GUIDE_TRAIN_B1=PASS",
        "STAGE2_GUIDE_TRAIN_B0=PASS",
        "STAGE2_GUIDE_PLOT=PASS",
        "STAGE2_GUIDE_INFER=PASS samples=56",
        "bash run_stage2_h100.sh infer all",
        "bash run_stage2_h100.sh infer 40 80 120 160 200 240 400 800 1200 1600 2400 3200 3600 4000",
        "checkpoints=N samples_per_checkpoint=56 total_samples=56*N",
        "没有顶层 all 子命令",
    )
    assert all(item in completed.stdout for item in required)
    assert "3750" not in completed.stdout
    assert "200/200/200" not in completed.stdout


def test_stage2_h100_guide_dry_runs_exact_commands_without_writes(tmp_path):
    env, work_root, train_root = _isolated_env(tmp_path)
    outputs: dict[str, str] = {}
    for command in (
        "prepare",
        "smoke",
        "train",
        "control",
        "plot-live",
        "plot",
        "infer",
    ):
        completed = _run("--dry-run", command, env=env)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        outputs[command] = completed.stdout
        assert not any(
            line.startswith("STAGE2_GUIDE_") and "=PASS" in line
            for line in completed.stdout.splitlines()
        )
    assert not work_root.exists()
    assert not train_root.exists()

    assert "prepare_stage2.sh" in outputs["prepare"]
    smoke = outputs["smoke"]
    assert str(DEFAULT_TRAIN_CONFIG) in smoke
    assert "smoke_longrun" in smoke
    assert smoke.count("--stage2-smoke") == 3
    assert smoke.index("--stage2-smoke C0") < smoke.index("--stage2-smoke C1")
    assert smoke.index("--stage2-smoke C1") < smoke.index("--stage2-smoke C2")
    assert smoke.count("--nproc-per-node=8") == 3

    train = outputs["train"]
    assert str(DEFAULT_TRAIN_CONFIG) in train
    assert "--stage2-smoke" not in train
    assert "--no-auto-resume" not in train
    assert "--nproc-per-node=8" in train
    assert "formal_b1_longrun" in train
    assert "G4000" in train

    control = outputs["control"]
    for field in (
        "init_from_stage1=null",
        "resume_stage2=G3600",
        "phase_b_mode=dmd_only",
        "dfd_probability=0",
        "formal_matched_b0",
    ):
        assert field in control

    assert outputs["plot"].count("--require-complete") == 2
    assert outputs["plot"].count("--formats png svg") == 2
    assert "--require-complete" not in outputs["plot-live"]
    assert "formal_b1_longrun/plots_live" in outputs["plot-live"]
    assert outputs["plot-live"].count("--formats png svg") == 1
    assert "infer_stage2_baseline.sh" in outputs["infer"]


def test_stage2_h100_guide_preserves_explicit_config_override(tmp_path):
    env, _, _ = _isolated_env(tmp_path)
    custom_config = tmp_path / "custom_stage2.yaml"
    shutil.copyfile(DEFAULT_TRAIN_CONFIG, custom_config)
    env["STAGE2_CONFIG"] = str(custom_config)

    for command in ("smoke", "train"):
        completed = _run("--dry-run", command, env=env)
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert str(custom_config) in completed.stdout
        assert str(DEFAULT_TRAIN_CONFIG) not in completed.stdout


def test_stage2_h100_guide_derives_legacy_baseline_endpoints_from_override(tmp_path):
    env, _, _ = _isolated_env(tmp_path)
    env["STAGE2_CONFIG"] = str(BASELINE_TRAIN_CONFIG)

    train = _run("--dry-run", "train", env=env)
    control = _run("--dry-run", "control", env=env)
    infer = _run("--dry-run", "infer", env=env)
    assert train.returncode == control.returncode == infer.returncode == 0
    assert "G280" in train.stdout
    assert "resume_stage2=G240" in control.stdout
    assert "checkpoint_stage2_g000280" in infer.stdout
    assert "inference_g000280_baseline" in infer.stdout


def test_stage2_h100_guide_batch_infer_sorts_and_deduplicates_steps(tmp_path):
    env, work_root, train_root = _isolated_env(tmp_path)
    completed = _run("--dry-run", "infer", "280", "80", "G80", "000260", env=env)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.count("infer_stage2_baseline.sh") == 3
    assert completed.stdout.index("G000080:") < completed.stdout.index("G000260:")
    assert completed.stdout.index("G000260:") < completed.stdout.index("G000280:")
    for step in ("000080", "000260", "000280"):
        assert f"checkpoint_stage2_g{step}" in completed.stdout
        assert f"inference_g{step}_baseline" in completed.stdout
    assert (
        "checkpoints=3 samples_per_checkpoint=56 total_samples=168" in completed.stdout
    )
    assert not work_root.exists()
    assert not train_root.exists()


def test_stage2_h100_guide_batch_infer_all_discovers_only_complete_ema_steps(
    tmp_path,
):
    env, _, train_root = _isolated_env(tmp_path)
    formal_root = train_root / "formal_b1_longrun"
    for step in (30, 80, 120):
        checkpoint = formal_root / f"checkpoint_stage2_g{step:06d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "_SUCCESS").write_text("PASS\n", encoding="utf-8")
    (formal_root / "checkpoint_stage2_g000100").mkdir()

    completed = _run("--dry-run", "infer", "all", env=env)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert completed.stdout.count("infer_stage2_baseline.sh") == 2
    assert "checkpoint_stage2_g000080" in completed.stdout
    assert "checkpoint_stage2_g000120" in completed.stdout
    assert "checkpoint_stage2_g000030" not in completed.stdout
    assert "checkpoint_stage2_g000100" not in completed.stdout


def test_stage2_h100_guide_batch_infer_can_scan_preserved_snapshot_root(tmp_path):
    env, _, train_root = _isolated_env(tmp_path)
    snapshot_root = train_root / "early_checkpoint_snapshots"
    checkpoint = snapshot_root / "checkpoint_stage2_g000060"
    checkpoint.mkdir(parents=True)
    (checkpoint / "_SUCCESS").write_text("PASS\n", encoding="utf-8")
    env["STAGE2_INFERENCE_CHECKPOINT_ROOT"] = str(snapshot_root)

    completed = _run("--dry-run", "infer", "all", env=env)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert str(checkpoint) in completed.stdout
    assert "inference_g000060_baseline" in completed.stdout


def test_stage2_h100_guide_batch_infer_rejects_invalid_selection(tmp_path):
    env, _, _ = _isolated_env(tmp_path)
    too_early = _run("--dry-run", "infer", "30", env=env)
    assert too_early.returncode != 0
    assert "Generator EMA从G40开始" in too_early.stderr

    mixed = _run("--dry-run", "infer", "all", "280", env=env)
    assert mixed.returncode != 0
    assert "all不能与显式步数混用" in mixed.stderr


def test_stage2_h100_guide_empty_status_is_read_only_and_incomplete(tmp_path):
    env, work_root, train_root = _isolated_env(tmp_path)
    completed = _run("status", env=env)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for stage in ("PREPARE", "SMOKE", "TRAIN_B1", "TRAIN_B0", "PLOT", "INFER"):
        assert f"STAGE2_STATUS_{stage}=INCOMPLETE" in completed.stdout
    assert completed.stdout.rstrip().endswith("STAGE2_STATUS=INCOMPLETE")
    assert not work_root.exists()
    assert not train_root.exists()


def test_stage2_h100_guide_rejects_unknown_or_extra_arguments():
    unknown = _run("all")
    assert unknown.returncode != 0
    assert "未知子命令：all" in unknown.stderr
    assert "STAGE2_GUIDE_RESULT=FAIL" in unknown.stderr

    extra = _run("help", "unexpected")
    assert extra.returncode != 0
    assert "不接受额外参数" in extra.stderr
    assert "STAGE2_GUIDE_RESULT=FAIL" in extra.stderr


def test_stage2_h100_guide_rejects_overlapping_run_directories(tmp_path):
    env, _, train_root = _isolated_env(tmp_path)
    shared = train_root / "shared"
    env["STAGE2_SMOKE_DIR"] = str(shared)
    env["STAGE2_FORMAL_DIR"] = str(shared)
    completed = _run("status", env=env)
    assert completed.returncode != 0
    assert "运行目录必须彼此独立" in completed.stderr
    assert not train_root.exists()


def test_stage2_h100_guide_keeps_release_safety_and_validation_order():
    text = GUIDE.read_text(encoding="utf-8")
    assert "set -Eeuo pipefail" in text
    assert "require_clean_checkout" not in text
    assert "git -C" not in text
    assert "require_external_path" not in text
    assert "checkpoint_model_003075" in text
    assert "stage1_step3075_ema_merged.pt" in text
    assert "STAGE2_FSDP2_ACCUMULATION_GATE=PASS mode=H100-world8" in text
    assert "--max-restarts=0" in text
    assert "--nproc-per-node=8" in text
    assert "--require-complete" in text
    assert "expected_sample_count" in text
    assert "== 56" in text
    assert 'export ACTIVE_CONFIG="$STAGE2_CONFIG"' in text
    assert (
        'export STAGE2_CONFIG="${STAGE2_CONFIG:-$SCRIPT_ROOT/configs/'
        'train_i2v_stage2_600cats_micro1_acc8.yaml}"' in text
    )
    assert "validate_stage2_checkpoint" in text
    assert "live_metrics.startswith(checkpoint_metrics)" in text
    assert 'expected_branch = "dfd" if mode == "C2" else "dmd"' in text
    assert 'generator["dfd_probability"] == expected_probability' in text
    assert 'require_training_arm "$ACTIVE_CONFIG" dmd_dfd' in text
    assert "expected_resolved_config=inference_config.to_dict()" in text
    assert "expected_samples=samples" in text
    assert "require_distinct_run_paths" in text
    assert "_audit_stage2_dmd_runtime_api" in text
    assert "STAGE2_DMD_RUNTIME_API=PASS" in text
    assert "STAGE2_PARAMETER_NAMES_API=PASS" in text
    assert "STAGE2_LORA_LOAD_API=PASS" in text
    assert "expected_parameter_names" in text
    assert "_canonicalize_stage2_optimizer_state" in text
    assert 'project_root / "model" / "stage2_dmd.py"' in text
    assert "tee -a" in text
    assert "rm -rf" not in text
    assert "git clean" not in text
    assert "git reset" not in text
    assert "--no-auto-resume" not in text
    assert "推理目录非空但不是完整批次" not in text
    assert "checkpoint_model_003750" not in text

    ordered_guards = {
        "run_prepare()": (
            "prepare_complete || fail",
            'echo "STAGE2_GUIDE_PREPARE=PASS"',
        ),
        "run_smoke()": (
            "smoke_complete || fail",
            'echo "STAGE2_GUIDE_SMOKE=PASS"',
        ),
        "run_train_b1()": (
            'formal_complete "$STAGE2_FORMAL_DIR" "$ACTIVE_CONFIG" dmd_dfd ||',
            'echo "STAGE2_GUIDE_TRAIN_B1=PASS"',
        ),
        "run_train_b0()": (
            'formal_complete "$STAGE2_B0_DIR" "$STAGE2_B0_CONFIG" dmd_only ||',
            'echo "STAGE2_GUIDE_TRAIN_B0=PASS"',
        ),
        "run_plot()": (
            "plots_complete || fail",
            'echo "STAGE2_GUIDE_PLOT=PASS"',
        ),
        "run_infer()": (
            'inference_complete "$output" "$checkpoint" "$((10#$step))" ||',
            'echo "STAGE2_GUIDE_INFER=PASS samples=56"',
        ),
    }
    function_starts = sorted((text.index(name), name) for name in ordered_guards)
    for index, (start, name) in enumerate(function_starts):
        end = (
            function_starts[index + 1][0]
            if index + 1 < len(function_starts)
            else len(text)
        )
        section = text[start:end]
        guard, marker = ordered_guards[name]
        assert section.index(guard) < section.index(marker)
