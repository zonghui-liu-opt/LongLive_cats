from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).parents[1]
GUIDE = PROJECT_ROOT / "run_stage2_h100.sh"


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
        "ACTIVE_CONFIG",
        "STAGE2_SMOKE_DIR",
        "STAGE2_FORMAL_DIR",
        "STAGE2_B0_DIR",
        "STAGE2_B0_CONFIG",
        "STAGE2_INFERENCE_OUTPUT",
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
        "bash run_stage2_h100.sh prepare",
        "bash run_stage2_h100.sh smoke",
        "bash run_stage2_h100.sh train",
        "bash run_stage2_h100.sh control",
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
        "没有 all 子命令",
    )
    assert all(item in completed.stdout for item in required)
    assert "3750" not in completed.stdout
    assert "200/200/200" not in completed.stdout


def test_stage2_h100_guide_dry_runs_exact_commands_without_writes(tmp_path):
    env, work_root, train_root = _isolated_env(tmp_path)
    outputs: dict[str, str] = {}
    for command in ("prepare", "smoke", "train", "control", "plot", "infer"):
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
    assert smoke.count("--stage2-smoke") == 3
    assert smoke.index("--stage2-smoke C0") < smoke.index("--stage2-smoke C1")
    assert smoke.index("--stage2-smoke C1") < smoke.index("--stage2-smoke C2")
    assert smoke.count("--nproc-per-node=8") == 3

    train = outputs["train"]
    assert "--stage2-smoke" not in train
    assert "--no-auto-resume" not in train
    assert "--nproc-per-node=8" in train
    assert "formal_baseline" in train

    control = outputs["control"]
    for field in (
        "init_from_stage1=null",
        "resume_stage2=G240",
        "phase_b_mode=dmd_only",
        "dfd_probability=0",
        "formal_matched_b0",
    ):
        assert field in control

    assert outputs["plot"].count("--require-complete") == 2
    assert outputs["plot"].count("--formats png svg") == 2
    assert "infer_stage2_baseline.sh" in outputs["infer"]


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
    assert 'git -C "$SCRIPT_ROOT" status --porcelain=v1 --untracked-files=all' in text
    assert "checkpoint_model_003075" in text
    assert "stage1_step3075_ema_merged.pt" in text
    assert "STAGE2_FSDP2_ACCUMULATION_GATE=PASS mode=H100-world8" in text
    assert "--max-restarts=0" in text
    assert "--nproc-per-node=8" in text
    assert "--require-complete" in text
    assert "expected_sample_count" in text
    assert "== 56" in text
    assert 'export ACTIVE_CONFIG="$STAGE2_CONFIG"' in text
    assert "validate_stage2_checkpoint" in text
    assert "live_metrics.startswith(checkpoint_metrics)" in text
    assert 'expected_branch = "dfd" if mode == "C2" else "dmd"' in text
    assert 'generator["dfd_probability"] == expected_probability' in text
    assert 'require_training_arm "$ACTIVE_CONFIG" dmd_dfd' in text
    assert "expected_resolved_config=inference_config.to_dict()" in text
    assert "expected_samples=samples" in text
    assert "require_distinct_run_paths" in text
    assert "tee -a" in text
    assert "rm -rf" not in text
    assert "git clean" not in text
    assert "git reset" not in text
    assert "--no-auto-resume" not in text
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
            "inference_complete || fail",
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
