from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT_ROOT = Path(__file__).parents[1]
ENTRYPOINT = PROJECT_ROOT / "run_stage2_c4w16s1_h100.sh"
CONFIG = PROJECT_ROOT / "configs" / "train_i2v_stage2_600cats_c4w16s1_micro1_acc8.yaml"


def _environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    work_root = tmp_path / "c4-work"
    train_root = tmp_path / "c4-train"
    true_executable = shutil.which("true")
    assert true_executable is not None
    environment = dict(os.environ)
    environment.update(
        {
            "STAGE2_PYTHON": sys.executable,
            "STAGE2_TORCHRUN": true_executable,
            "STAGE2_WORK_ROOT": str(work_root),
            "STAGE2_TRAIN_ROOT": str(train_root),
        }
    )
    for name in (
        "STAGE2_GPUS",
        "STAGE2_CONFIG",
        "ACTIVE_CONFIG",
        "STAGE2_SMOKE_DIR",
        "STAGE2_FORMAL_DIR",
        "STAGE2_B0_DIR",
        "STAGE2_B0_CONFIG",
        "STAGE2_INFERENCE_OUTPUT",
        "STAGE2_INFERENCE_ROOT",
        "STAGE2_INFERENCE_CHECKPOINT_ROOT",
        "STAGE2_LIVE_PLOT_DIR",
        "LONG_LIVE_STAGE2_GRADIENT_ACCUMULATION_STEPS",
        "LONG_LIVE_STAGE2_PREFLIGHT_MICRO2_ACCUMULATION_STEPS",
    ):
        environment.pop(name, None)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    return environment, work_root, train_root


def _run(*arguments: str, env: dict[str, str] | None = None):
    return subprocess.run(
        ["bash", str(ENTRYPOINT), *arguments],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_c4w16_h100_entrypoint_has_valid_shell_and_complete_operator_contract():
    syntax = subprocess.run(
        ["bash", "-n", str(ENTRYPOINT)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr

    help_result = _run("help")
    assert help_result.returncode == 0, help_result.stderr
    for text in (
        "4 current latent + 12 history latent + 1 global initial-image sink",
        "Generator/Fake-score LR=5e-6/1e-6",
        "每个generator epoch（10G）保存完整checkpoint，共400个",
        "bash run_stage2_c4w16s1_h100.sh all",
        "约1TiB",
        "G10/G20/G30",
        "从G40开始存在",
    ):
        assert text in help_result.stdout


def test_c4w16_h100_all_dry_run_is_isolated_and_does_not_write(tmp_path):
    environment, work_root, train_root = _environment(tmp_path)
    completed = _run("--dry-run", "all", env=environment)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert str(CONFIG) in completed.stdout
    assert "profile=h100_c4w16s1_micro1_acc8_longrun" in completed.stdout
    assert "topology=8xH100 micro1 acc8 global64" in completed.stdout
    assert "prepare_stage2.sh" in completed.stdout
    assert completed.stdout.count("--stage2-smoke") == 3
    assert "formal_b1_c4w16s1_all_epochs" in completed.stdout
    assert "[dry-run] verify 400 epoch checkpoints" in completed.stdout
    assert "formal_b1_longrun" not in completed.stdout
    assert not work_root.exists()
    assert not train_root.exists()


def test_c4w16_h100_world4_preserves_global64_and_unique_default_roots(tmp_path):
    environment, _, _ = _environment(tmp_path)
    environment.pop("STAGE2_WORK_ROOT")
    environment.pop("STAGE2_TRAIN_ROOT")
    environment["STAGE2_GPUS"] = "4"
    environment["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    completed = _run("--dry-run", "train", env=environment)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "topology=4xH100 micro1 acc16 global64" in completed.stdout
    assert "--nproc-per-node=4" in completed.stdout
    assert "c4w16s1_micro1_acc8_longrun_4gpus" in completed.stdout


def test_c4w16_h100_entrypoint_rejects_config_override(tmp_path):
    environment, _, _ = _environment(tmp_path)
    environment["STAGE2_CONFIG"] = str(
        PROJECT_ROOT / "configs" / "train_i2v_stage2_600cats_micro1_acc8.yaml"
    )

    completed = _run("--dry-run", "train", env=environment)

    assert completed.returncode != 0
    assert "禁止覆盖STAGE2_CONFIG" in completed.stderr


def test_c4w16_completion_gate_checks_exact_epoch_set_and_no_removals():
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "set(actual) != expected" in text
    assert "stage2_records_for_latest_lineage" in text
    assert 'record_type="checkpoint_event"' in text
    assert "event_steps != sorted(event_steps)" in text
    assert "len(event_steps) != len(set(event_steps))" in text
    assert "set(event_steps) - expected" in text
    assert "event_steps != sorted(expected)" not in text
    assert 'item.get("removed") not in (None, [])' in text
    assert "validate_stage2_checkpoint" in text
    assert "STAGE2_C4W16S1_ALL_EPOCHS=PASS" in text
