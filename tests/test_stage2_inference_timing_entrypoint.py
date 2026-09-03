from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import run_stage2_inference as inference_entrypoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = PROJECT_ROOT / "infer_stage2_c4w16k4s1_timing.sh"
CONFIG = PROJECT_ROOT / "configs" / "infer_i2v_stage2_c4w16k4s1_timing.yaml"


@pytest.mark.parametrize(
    ("mode", "samples", "forwards"),
    [("quick", 2, 62), ("formal", 24, 744)],
)
def test_timing_plan_selects_only_requested_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    samples: int,
    forwards: int,
) -> None:
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("LONG_LIVE_STAGE2_SOURCE_MANIFEST", str(source_manifest))
    monkeypatch.setenv("LONG_LIVE_STAGE2_INFERENCE_CHECKPOINT", "/checkpoint")
    monkeypatch.setenv("LONG_LIVE_STAGE2_ARCHITECTURE_ROOT", "/architecture")
    monkeypatch.setenv("LONG_LIVE_STAGE2_T5_CHECKPOINT", "/t5.pth")
    monkeypatch.setenv("LONG_LIVE_STAGE2_TOKENIZER_DIR", "/tokenizer")
    monkeypatch.setenv("LONG_LIVE_STAGE2_VAE_CHECKPOINT", "/vae.pth")
    monkeypatch.setenv("LONG_LIVE_STAGE2_INFERENCE_OUTPUT", str(tmp_path / "out"))
    config = inference_entrypoint.load_stage2_inference_run_config(CONFIG)
    config = inference_entrypoint._override_sweep_evaluation(config, mode)
    plan = inference_entrypoint.build_stage2_inference_plan(config)

    assert config.profiles == ("c4w16k4s1",)
    assert config.two_action_row_ids == ()
    assert plan["evaluation_mode"] == mode
    assert plan["expected_sample_count"] == samples
    assert plan["single_action_sample_count"] == samples
    assert plan["two_action_sample_count"] == 0
    assert plan["generator_forward_calls"] == forwards
    profile = plan["profiles"][0]
    assert (
        profile["chunk_frames"],
        profile["local_window_frames"],
        profile["global_sink_frames"],
        profile["num_denoising_steps"],
        profile["fresh_episode_dit_calls"],
    ) == (4, 16, 1, 4, 31)
    # A mode round-trip must not silently re-enable the excluded dataset.
    switched = inference_entrypoint._override_sweep_evaluation(config, "formal")
    switched = inference_entrypoint._override_sweep_evaluation(switched, "quick")
    assert switched.two_action_row_ids == ()


@pytest.fixture
def launch_stub(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    launcher = tmp_path / ENTRYPOINT.name
    shutil.copy2(ENTRYPOINT, launcher)
    (tmp_path / "infer_stage2_tmp.sh").write_text(
        """#!/usr/bin/env bash
set -eu
printf 'step=%s\\n' "$*"
printf 'config=%s\\n' "$STAGE2_INFERENCE_CONFIG"
printf 'evaluation=%s\\n' "$STAGE2_SWEEP_EVALUATION"
printf 'plan=%s\\n' "$STAGE2_INFERENCE_PLAN_ONLY"
printf 'cuda=%s\\n' "$CUDA_VISIBLE_DEVICES"
printf 'nproc=%s\\n' "$STAGE2_INFERENCE_NPROC"
printf 'project=%s\\n' "$STAGE2_PROJECT_ROOT"
printf 'snapshot=%s\\n' "$SNAPSHOT_ROOT"
""",
        encoding="utf-8",
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("STAGE2_", "LONG_LIVE_STAGE2_"))
        and key not in {"CUDA_VISIBLE_DEVICES", "INFER_NPROC", "SNAPSHOT_ROOT"}
    }
    return launcher, environment


def _launch(
    stub: tuple[Path, dict[str, str]], *arguments: str
) -> subprocess.CompletedProcess[str]:
    launcher, environment = stub
    return subprocess.run(
        ["bash", str(launcher), *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("mode", "evaluation", "plan"),
    [
        ("quick", "quick", "0"),
        ("formal", "formal", "0"),
        ("plan-quick", "quick", "1"),
        ("plan-formal", "formal", "1"),
    ],
)
def test_timing_launch_forwards_exact_profile_and_explicit_step(
    launch_stub: tuple[Path, dict[str, str]],
    mode: str,
    evaluation: str,
    plan: str,
) -> None:
    result = _launch(launch_stub, mode, "G120")
    assert result.returncode == 0, result.stderr
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert fields["step"] == "000120"
    assert fields["config"] == f"configs/{CONFIG.name}"
    assert fields["evaluation"] == evaluation
    assert fields["plan"] == plan
    assert fields["cuda"] == "0"
    assert fields["nproc"] == "1"
    assert fields["project"] == str(launch_stub[0].parent)
    assert fields["snapshot"].endswith(
        "LongLive-2.0_training_h100_c4w16s1_micro1_acc8_longrun/"
        "formal_b1_c4w16s1_all_epochs"
    )


def test_timing_launch_respects_checkpoint_gpu_and_lineage_overrides(
    launch_stub: tuple[Path, dict[str, str]], tmp_path: Path
) -> None:
    _, environment = launch_stub
    environment.update(
        {
            "STAGE2_EARLY_CHECKPOINT": "/custom/checkpoint_stage2_g000080/",
            "CUDA_VISIBLE_DEVICES": "3,5",
            "STAGE2_FORMAL_DIR": str(tmp_path / "formal"),
        }
    )
    result = _launch(launch_stub, "quick")
    assert result.returncode == 0, result.stderr
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert fields["step"] == "000080"
    assert fields["cuda"] == "3,5"
    assert fields["nproc"] == "2"
    assert fields["snapshot"] == str(tmp_path / "formal")


@pytest.mark.parametrize(
    ("arguments", "overrides", "expected_error"),
    [
        (("quick",), {}, "必须指定 checkpoint step"),
        (("quick", "G120", "G160"), {}, "每次只指定一个"),
        (
            ("quick", "G120"),
            {"STAGE2_EARLY_CHECKPOINT": "/custom/checkpoint_stage2_g000080"},
            "训练步数不同",
        ),
        (
            ("quick", "G120"),
            {"STAGE2_INFERENCE_CONFIG": "configs/infer_i2v_stage2_baseline.yaml"},
            "固定 C4/W16/S1/K4",
        ),
        (
            ("quick", "G120"),
            {"STAGE2_INFERENCE_NPROC": "4"},
            "必须同时设置 CUDA_VISIBLE_DEVICES",
        ),
    ],
)
def test_timing_launch_rejects_ambiguous_or_conflicting_requests(
    launch_stub: tuple[Path, dict[str, str]],
    arguments: tuple[str, ...],
    overrides: dict[str, str],
    expected_error: str,
) -> None:
    launch_stub[1].update(overrides)
    result = _launch(launch_stub, *arguments)
    assert result.returncode != 0
    assert expected_error in result.stderr
    assert "step=" not in result.stdout
