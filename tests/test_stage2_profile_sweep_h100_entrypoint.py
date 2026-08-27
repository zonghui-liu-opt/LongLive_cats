from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts import run_stage2_inference as inference_entrypoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = PROJECT_ROOT / "run_stage2_profile_sweep_h100.sh"
MATCHED_CONFIG = PROJECT_ROOT / "configs" / "infer_i2v_stage2_c4w16_k432_sweep.yaml"
COMPRESSED_CONFIG = (
    PROJECT_ROOT / "configs" / "infer_i2v_stage2_cw_compression_k4_sweep.yaml"
)


def _plan(
    config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    formal: bool = False,
) -> dict:
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.setenv("LONG_LIVE_STAGE2_SOURCE_MANIFEST", str(source_manifest))
    monkeypatch.setenv("LONG_LIVE_STAGE2_INFERENCE_CHECKPOINT", "/checkpoint")
    monkeypatch.setenv("LONG_LIVE_STAGE2_ARCHITECTURE_ROOT", "/architecture")
    monkeypatch.setenv("LONG_LIVE_STAGE2_T5_CHECKPOINT", "/t5.pth")
    monkeypatch.setenv("LONG_LIVE_STAGE2_TOKENIZER_DIR", "/tokenizer")
    monkeypatch.setenv("LONG_LIVE_STAGE2_VAE_CHECKPOINT", "/vae.pth")
    monkeypatch.setenv("LONG_LIVE_STAGE2_INFERENCE_OUTPUT", str(tmp_path / "out"))
    config = inference_entrypoint.load_stage2_inference_run_config(config_path)
    if formal:
        config = inference_entrypoint._override_sweep_evaluation(config, "formal")
    return inference_entrypoint.build_stage2_inference_plan(config)


@pytest.mark.parametrize(
    (
        "config_path",
        "topologies",
        "training_allowed",
        "profile_set_sha256",
        "quick_samples",
        "quick_forwards",
        "formal_samples",
        "formal_forwards",
    ),
    [
        (
            MATCHED_CONFIG,
            [(4, 16, 1, 2), (4, 16, 1, 3), (4, 16, 1, 4)],
            [False, False, True],
            "4a4fd70b2690b6a74b7dfa2c99f06f6d96ffbbedd8e9cf38083e9573e92c3310",
            12,
            444,
            168,
            6504,
        ),
        (
            COMPRESSED_CONFIG,
            [
                (2, 8, 1, 4),
                (2, 16, 1, 4),
                (4, 8, 1, 4),
                (4, 12, 1, 4),
                (4, 16, 1, 4),
            ],
            [False, False, False, False, True],
            "bc5a27b1e4aec9b1cd53c61b6ddbf57cf5a7068713e65e94921d5e7e5950f1c5",
            20,
            1280,
            280,
            18760,
        ),
    ],
)
def test_repo_profile_sweep_configs_have_exact_topologies_and_counts(
    config_path: Path,
    topologies: list[tuple[int, int, int, int]],
    training_allowed: list[bool],
    profile_set_sha256: str,
    quick_samples: int,
    quick_forwards: int,
    formal_samples: int,
    formal_forwards: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    quick = _plan(config_path, tmp_path, monkeypatch)
    formal = _plan(config_path, tmp_path, monkeypatch, formal=True)
    resolved = inference_entrypoint.load_stage2_inference_run_config(config_path)

    assert [
        (
            item["chunk_frames"],
            item["local_window_frames"],
            item["global_sink_frames"],
            item["num_denoising_steps"],
        )
        for item in quick["profiles"]
    ] == topologies
    assert quick["evaluation_mode"] == "quick"
    assert quick["base_samples_per_profile"] == 4
    assert quick["expected_sample_count"] == quick_samples
    assert quick["generator_forward_calls"] == quick_forwards
    assert quick["rank_reuse_divisors_up_to_8"] == [1, 2, 4]
    assert formal["evaluation_mode"] == "formal"
    assert formal["base_samples_per_profile"] == 56
    assert formal["expected_sample_count"] == formal_samples
    assert formal["generator_forward_calls"] == formal_forwards
    assert formal["rank_reuse_divisors_up_to_8"] == [1, 2, 4, 7, 8]
    assert [item["name"] for item in quick["profiles"]] == [
        item["name"] for item in formal["profiles"]
    ]
    assert resolved.profile_set_sha256 == profile_set_sha256
    assert [
        spec.training_allowed for spec in resolved.rollout_specs
    ] == training_allowed


def _complete_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint_stage2_g{step:06d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "_SUCCESS").touch()
    for filename in ("checkpoint_manifest.json", "provenance.json"):
        (checkpoint / filename).write_text("{}\n", encoding="utf-8")
    return checkpoint


def _fake_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path, Path]:
    project = tmp_path / "project"
    config_dir = project / "configs"
    config_dir.mkdir(parents=True)
    launcher = project / ENTRYPOINT.name
    shutil.copy2(ENTRYPOINT, launcher)
    for config in (MATCHED_CONFIG, COMPRESSED_CONFIG):
        shutil.copy2(config, config_dir / config.name)
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir()
    capture = tmp_path / "capture.txt"
    runner = project / "infer_stage2_tmp.sh"
    runner.write_text(
        """#!/usr/bin/env bash
set -eu
{
  printf 'args=%s\\n' \"$*\"
  printf 'config=%s\\n' \"$STAGE2_INFERENCE_CONFIG\"
  printf 'evaluation=%s\\n' \"$STAGE2_SWEEP_EVALUATION\"
  printf 'plan=%s\\n' \"$STAGE2_INFERENCE_PLAN_ONLY\"
  printf 'cuda=%s\\n' \"$CUDA_VISIBLE_DEVICES\"
  printf 'nproc=%s\\n' \"$STAGE2_INFERENCE_NPROC\"
} >> \"$STAGE2_TEST_CAPTURE\"
if [[ \"${STAGE2_TEST_FAIL_CONFIG:-}\" == \"$STAGE2_INFERENCE_CONFIG\" ]]; then
  exit 9
fi
""",
        encoding="utf-8",
    )
    runner.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        {
            "STAGE2_WORK_ROOT": str(tmp_path / "work"),
            "STAGE2_TRAIN_ROOT": str(tmp_path / "train"),
            "SNAPSHOT_ROOT": str(snapshot_root),
            "STAGE2_TEST_CAPTURE": str(capture),
        }
    )
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "STAGE2_INFERENCE_NPROC",
        "STAGE2_EARLY_CHECKPOINT",
        "STAGE2_EARLY_OUTPUT",
        "STAGE2_EARLY_LOG",
    ):
        environment.pop(name, None)
    return environment, snapshot_root, capture, launcher


def _run(
    *arguments: str,
    env: dict[str, str] | None = None,
    entrypoint: Path = ENTRYPOINT,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(entrypoint), *arguments],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_profile_sweep_shell_help_and_syntax() -> None:
    syntax = subprocess.run(
        ["bash", "-n", str(ENTRYPOINT)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr
    completed = _run("--help")
    assert completed.returncode == 0, completed.stderr
    for expected in (
        "plan-quick",
        "plan-formal",
        "matched",
        "compressed",
        "both",
        "12/168",
        "20/280",
        "32/448",
        "all",
        "严格续跑video+trace",
    ):
        assert expected in completed.stdout
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "infer_stage2_tmp.sh" in text
    assert "STAGE2_PROFILE_SWEEP_RUNNER" not in text
    assert "scripts/run_stage2_inference.py" not in text
    assert "rm -rf" not in text


def test_explicit_steps_are_normalized_sorted_deduplicated_and_quick_defaults(
    tmp_path: Path,
) -> None:
    environment, snapshot_root, capture, launcher = _fake_environment(tmp_path)
    _complete_checkpoint(snapshot_root, 40)
    _complete_checkpoint(snapshot_root, 80)

    completed = _run(
        "quick",
        "matched",
        "80",
        "G40",
        "000080",
        env=environment,
        entrypoint=launcher,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "checkpoint_count=2 steps=000040 000080" in completed.stdout
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "args=000040 000080",
        "config=configs/infer_i2v_stage2_c4w16_k432_sweep.yaml",
        "evaluation=quick",
        "plan=0",
        "cuda=0,1,2,3",
        "nproc=4",
    ]


def test_all_discovers_only_complete_g40_plus_and_formal_plan_defaults(
    tmp_path: Path,
) -> None:
    environment, snapshot_root, capture, launcher = _fake_environment(tmp_path)
    _complete_checkpoint(snapshot_root, 20)
    _complete_checkpoint(snapshot_root, 120)
    _complete_checkpoint(snapshot_root, 80)
    incomplete = snapshot_root / "checkpoint_stage2_g000100"
    incomplete.mkdir()

    completed = _run(
        "plan-formal", "compressed", "all", env=environment, entrypoint=launcher
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "checkpoint_count=2 steps=000080 000120" in completed.stdout
    assert capture.read_text(encoding="utf-8").splitlines() == [
        "args=000080 000120",
        "config=configs/infer_i2v_stage2_cw_compression_k4_sweep.yaml",
        "evaluation=formal",
        "plan=1",
        "cuda=0,1,2,3,4,5,6,7",
        "nproc=8",
    ]


def test_visible_devices_derive_nproc_without_changing_the_selection(
    tmp_path: Path,
) -> None:
    environment, snapshot_root, capture, launcher = _fake_environment(tmp_path)
    _complete_checkpoint(snapshot_root, 40)
    environment["CUDA_VISIBLE_DEVICES"] = "1,3,5"

    completed = _run(
        "plan-quick", "matched", "40", env=environment, entrypoint=launcher
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    captured = capture.read_text(encoding="utf-8")
    assert "cuda=1,3,5" in captured
    assert "nproc=3" in captured


def test_both_runs_matched_then_compressed_and_stops_after_child_failure(
    tmp_path: Path,
) -> None:
    environment, snapshot_root, capture, launcher = _fake_environment(tmp_path)
    _complete_checkpoint(snapshot_root, 40)

    completed = _run("quick", "both", "40", env=environment, entrypoint=launcher)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    captured = capture.read_text(encoding="utf-8")
    matched = "config=configs/infer_i2v_stage2_c4w16_k432_sweep.yaml"
    compressed = "config=configs/infer_i2v_stage2_cw_compression_k4_sweep.yaml"
    assert captured.count("args=000040") == 2
    assert captured.index(matched) < captured.index(compressed)
    assert "STAGE2_PROFILE_SWEEP=PASS phases=2 checkpoints=1" in completed.stdout

    capture.unlink()
    environment["STAGE2_TEST_FAIL_CONFIG"] = (
        "configs/infer_i2v_stage2_c4w16_k432_sweep.yaml"
    )
    failed = _run("quick", "both", "40", env=environment, entrypoint=launcher)
    assert failed.returncode != 0
    failed_capture = capture.read_text(encoding="utf-8")
    assert matched in failed_capture
    assert compressed not in failed_capture
    assert "PHASE=FAIL" in failed.stderr


def test_launcher_delegates_plan_only_to_the_real_tmp_contract(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshots"
    _complete_checkpoint(snapshot_root, 40)
    train_root = tmp_path / "train"
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    architecture = tmp_path / "architecture"
    architecture.mkdir()
    (architecture / "config.json").write_text("{}\n", encoding="utf-8")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    t5 = tmp_path / "t5.pth"
    t5.write_bytes(b"t5")
    vae = tmp_path / "vae.pth"
    vae.write_bytes(b"vae")
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        """#!/usr/bin/env bash
set -eu
for value in "$@"; do
  if [[ "$value" == "-" ]]; then
    printf 'STAGE2_INFERENCE_ASSET_API=PASS (fake)\\n'
    printf 'STAGE2_FFPROBE=PASS (fake)\\n'
    exit 0
  fi
done
for value in "$@"; do
  if [[ "$value" == "--print-plan-field" ]]; then
    printf '12\\t4\\t%s\\n' 'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd'
    exit 0
  fi
done
printf '{"expected_sample_count": 12}\\n'
""",
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        {
            "STAGE2_PYTHON": str(fake_python),
            "STAGE2_TORCHRUN": shutil.which("true") or "/usr/bin/true",
            "STAGE2_WORK_ROOT": str(tmp_path / "work"),
            "STAGE2_TRAIN_ROOT": str(train_root),
            "SNAPSHOT_ROOT": str(snapshot_root),
            "LONG_LIVE_STAGE2_SOURCE_MANIFEST": str(source_manifest),
            "LONG_LIVE_STAGE2_ARCHITECTURE_ROOT": str(architecture),
            "LONG_LIVE_STAGE2_T5_CHECKPOINT": str(t5),
            "LONG_LIVE_STAGE2_TOKENIZER_DIR": str(tokenizer),
            "LONG_LIVE_STAGE2_VAE_CHECKPOINT": str(vae),
        }
    )
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "STAGE2_INFERENCE_NPROC",
        "STAGE2_EARLY_CHECKPOINT",
        "STAGE2_EARLY_OUTPUT",
        "STAGE2_EARLY_LOG",
    ):
        environment.pop(name, None)

    completed = _run("plan-quick", "matched", "40", env=environment)

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "STAGE2_INFERENCE_ASSET_API=PASS" in completed.stdout
    assert "仅规划模式完成；未初始化模型、CUDA 或 torchrun" in completed.stdout
    assert "STAGE2_PROFILE_SWEEP=PASS phases=1 checkpoints=1" in completed.stdout
    log = (
        train_root / "inference_early_g000040_infer_i2v_stage2_c4w16_k432_sweep_quick_"
        "dddddddddddddddd.log"
    )
    assert log.is_file()
    assert not list(train_root.glob("*.stage2-run.lock"))


def test_profile_sweep_shell_fails_closed_on_unsafe_batch_inputs(
    tmp_path: Path,
) -> None:
    environment, snapshot_root, _, launcher = _fake_environment(tmp_path)
    _complete_checkpoint(snapshot_root, 40)

    mixed = _run("quick", "matched", "all", "40", env=environment, entrypoint=launcher)
    assert mixed.returncode != 0
    assert "all不能和显式checkpoint混用" in mixed.stderr

    too_early = _run("quick", "matched", "20", env=environment, entrypoint=launcher)
    assert too_early.returncode != 0
    assert "早于最小可推理step" in too_early.stderr

    (snapshot_root / "checkpoint_stage2_g000040" / "_SUCCESS").write_text(
        "PASS\n", encoding="utf-8"
    )
    nonempty_success = _run(
        "quick", "matched", "40", env=environment, entrypoint=launcher
    )
    assert nonempty_success.returncode != 0
    assert "_SUCCESS必须是零字节完成标记" in nonempty_success.stderr
    (snapshot_root / "checkpoint_stage2_g000040" / "_SUCCESS").write_bytes(b"")

    environment["STAGE2_EARLY_OUTPUT"] = str(tmp_path / "override")
    overridden = _run("quick", "matched", "40", env=environment, entrypoint=launcher)
    assert overridden.returncode != 0
    assert "禁止设置STAGE2_EARLY_OUTPUT" in overridden.stderr

    environment.pop("STAGE2_EARLY_OUTPUT")
    environment["CUDA_VISIBLE_DEVICES"] = "0,0"
    duplicate_gpu = _run("quick", "matched", "40", env=environment, entrypoint=launcher)
    assert duplicate_gpu.returncode != 0
    assert "GPU编号重复" in duplicate_gpu.stderr

    environment["CUDA_VISIBLE_DEVICES"] = "0,"
    trailing_comma = _run(
        "quick", "matched", "40", env=environment, entrypoint=launcher
    )
    assert trailing_comma.returncode != 0
    assert "无空格、无空项" in trailing_comma.stderr
