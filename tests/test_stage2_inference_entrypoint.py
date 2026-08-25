from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from scripts import run_stage2_inference as entrypoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cli_loads_strict_config_and_delegates(monkeypatch, capsys) -> None:
    sentinel = object()
    seen = {}
    monkeypatch.setattr(
        entrypoint,
        "load_stage2_inference_run_config",
        lambda path: seen.setdefault("config", (path, sentinel))[1],
    )
    monkeypatch.setattr(
        entrypoint,
        "run_stage2_inference",
        lambda config: {
            "status": "complete",
            "rank": 0,
            "world_size": 8,
            "config_is_sentinel": config is sentinel,
        },
    )

    assert entrypoint.main(["--config", "profiles.yaml"]) == 0
    assert seen["config"] == ("profiles.yaml", sentinel)
    assert '"status": "complete"' in capsys.readouterr().out


def test_cli_plan_only_never_enters_distributed_runtime(monkeypatch, capsys) -> None:
    sentinel = object()
    monkeypatch.setattr(
        entrypoint,
        "load_stage2_inference_run_config",
        lambda path: sentinel,
    )
    monkeypatch.setattr(
        entrypoint,
        "_override_sweep_evaluation",
        lambda config, mode: config,
    )
    monkeypatch.setattr(
        entrypoint,
        "build_stage2_inference_plan",
        lambda config: {
            "expected_sample_count": 32,
            "profile_count": 8,
            "config_is_sentinel": config is sentinel,
        },
    )
    monkeypatch.setattr(
        entrypoint,
        "run_stage2_inference",
        lambda config: pytest.fail("plan-only initialized the distributed runtime"),
    )

    assert entrypoint.main(["--config", "sweep.yaml", "--plan-only"]) == 0
    assert '"expected_sample_count": 32' in capsys.readouterr().out

    assert (
        entrypoint.main(
            [
                "--config",
                "sweep.yaml",
                "--print-plan-field",
                "expected_sample_count",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "32"

    assert (
        entrypoint.main(
            [
                "--config",
                "sweep.yaml",
                "--print-plan-field",
                "expected_sample_count",
                "--print-plan-field",
                "profile_count",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "32\t8"


def test_cli_help_is_cpu_safe() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "run_stage2_inference.py"),
            "--help",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Any canonical rollout profile" in completed.stdout


def test_repo_sweep_plan_promotes_same_profiles_from_quick_to_formal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    config = entrypoint.load_stage2_inference_run_config(
        PROJECT_ROOT / "configs" / "infer_i2v_stage2_sweep.yaml"
    )
    quick = entrypoint.build_stage2_inference_plan(config)
    formal_config = entrypoint._override_sweep_evaluation(config, "formal")
    formal = entrypoint.build_stage2_inference_plan(formal_config)

    assert quick["evaluation_mode"] == "quick"
    assert quick["base_samples_per_profile"] == 4
    assert quick["profile_count"] == 9
    assert quick["expected_sample_count"] == 36
    assert len(quick["sample_plan_sha256"]) == 64
    assert formal["evaluation_mode"] == "formal"
    assert formal["base_samples_per_profile"] == 56
    assert formal["expected_sample_count"] == 504
    assert [item["name"] for item in quick["profiles"]] == [
        item["name"] for item in formal["profiles"]
    ]


def test_plan_only_uses_the_real_metadata_sample_planner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    malformed_metadata = tmp_path / "malformed.csv"
    malformed_metadata.write_text("not,the,required,columns\n", encoding="utf-8")
    monkeypatch.setenv("LONG_LIVE_STAGE2_SOURCE_MANIFEST", str(source_manifest))
    monkeypatch.setenv("LONG_LIVE_STAGE2_INFERENCE_CHECKPOINT", "/checkpoint")
    monkeypatch.setenv("LONG_LIVE_STAGE2_ARCHITECTURE_ROOT", "/architecture")
    monkeypatch.setenv("LONG_LIVE_STAGE2_T5_CHECKPOINT", "/t5.pth")
    monkeypatch.setenv("LONG_LIVE_STAGE2_TOKENIZER_DIR", "/tokenizer")
    monkeypatch.setenv("LONG_LIVE_STAGE2_VAE_CHECKPOINT", "/vae.pth")
    monkeypatch.setenv("LONG_LIVE_STAGE2_INFERENCE_OUTPUT", str(tmp_path / "out"))
    config = entrypoint.load_stage2_inference_run_config(
        PROJECT_ROOT / "configs" / "infer_i2v_stage2_sweep.yaml"
    )

    with pytest.raises((KeyError, ValueError, RuntimeError)):
        entrypoint.build_stage2_inference_plan(
            replace(config, single_metadata=str(malformed_metadata))
        )


def test_release_shell_is_strict_observable_baseline_torchrun() -> None:
    shell = PROJECT_ROOT / "infer_stage2_baseline.sh"
    text = shell.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(shell)], check=True)
    assert "set -euo pipefail" in text
    assert '"${STAGE2_TORCHRUN}"' in text
    assert '--no-python "${STAGE2_PYTHON}" -I -B' in text
    assert '--nproc-per-node="${STAGE2_INFERENCE_NPROC}"' in text
    assert 'STAGE2_INFERENCE_NPROC="${STAGE2_INFERENCE_NPROC:-8}"' in text
    assert "STAGE2_INFERENCE_HEARTBEAT=RUNNING" in text
    assert "STAGE2_FFPROBE=PASS" in text
    assert "resolve_ffprobe" in text
    assert "PYTHONUNBUFFERED=1" in text
    assert "configs/infer_i2v_stage2_baseline.yaml" in text
    assert "LONG_LIVE_STAGE2_INFERENCE_CHECKPOINT" in text
    assert "LONG_LIVE_STAGE2_INFERENCE_OUTPUT" in text


def test_early_checkpoint_shell_is_observable_and_cross_node_safe() -> None:
    shell = PROJECT_ROOT / "infer_stage2_tmp.sh"
    text = shell.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(shell)], check=True)

    assert text.startswith("#!/usr/bin/env bash\nset -Eeuo pipefail\n")
    assert "STAGE2_INFERENCE_ASSET_API=PASS" in text
    assert "STAGE2_FFPROBE=PASS" in text
    assert "LONG_LIVE_FFPROBE" in text
    assert "longlive_stage2_inference_assets/v2" in text
    assert '--nproc-per-node="$STAGE2_INFERENCE_NPROC"' in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "仍在运行" in text
    assert "_SUCCESS" in text
    assert "checkpoint_manifest.json" in text
    assert "provenance.json" in text
    assert "manifest.json" in text
    assert "index.html" in text
    assert "STAGE2_INFERENCE_CONFIG" in text
    assert "STAGE2_SWEEP_EVALUATION" in text
    assert "STAGE2_INFERENCE_PLAN_ONLY" in text
    assert "--print-plan-field expected_sample_count" in text
    assert "--print-plan-field resolved_contract_hash" in text
    assert "stage2-run.lock" in text
    assert "stop_active_process" in text
    assert "evaluation_args" not in text
    assert 'STAGE2_WORK_ROOT="${STAGE2_WORK_ROOT:-' in text
    assert 'SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-' in text
    assert 'verify_outputs "$output" "$expected_artifacts"' in text
    assert "EXPECTED_ARTIFACTS=56" not in text
    assert "rm -rf" not in text
    assert "checkpoint_stage2_g000280" not in text


def test_early_checkpoint_plan_only_runs_with_bash3_and_caller_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    (project / "scripts").mkdir(parents=True)
    (project / "configs").mkdir()
    (project / "scripts" / "run_stage2_inference.py").write_text("# fake\n")
    (project / "configs" / "sweep.yaml").write_text("schema: fake\n")
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text("{}\n")
    architecture = tmp_path / "architecture"
    architecture.mkdir()
    (architecture / "config.json").write_text("{}\n")
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    t5 = tmp_path / "t5.pth"
    t5.write_bytes(b"t5")
    vae = tmp_path / "vae.pth"
    vae.write_bytes(b"vae")
    train_root = tmp_path / "caller-train"
    snapshot_root = tmp_path / "caller-snapshots"
    checkpoint = snapshot_root / "checkpoint_stage2_g000070"
    checkpoint.mkdir(parents=True)
    for name in ("_SUCCESS", "checkpoint_manifest.json", "provenance.json"):
        (checkpoint / name).write_text("{}\n")

    fake_python = tmp_path / "fake-python"
    fake_python.write_text("""#!/usr/bin/env bash
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
        printf '32\\t4\\t%s\\n' 'eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee'
        exit 0
    fi
done
printf '{"expected_sample_count": 32}\\n'
""")
    fake_python.chmod(0o755)
    fake_torchrun = tmp_path / "fake-torchrun"
    fake_torchrun.write_text("#!/bin/sh\nexit 0\n")
    fake_torchrun.chmod(0o755)

    environment = {
        "STAGE2_PROJECT_ROOT": str(project),
        "STAGE2_PYTHON": str(fake_python),
        "STAGE2_TORCHRUN": str(fake_torchrun),
        "STAGE2_WORK_ROOT": str(tmp_path / "caller-work"),
        "STAGE2_TRAIN_ROOT": str(train_root),
        "SNAPSHOT_ROOT": str(snapshot_root),
        "LONG_LIVE_STAGE2_SOURCE_MANIFEST": str(source_manifest),
        "LONG_LIVE_STAGE2_ARCHITECTURE_ROOT": str(architecture),
        "LONG_LIVE_STAGE2_T5_CHECKPOINT": str(t5),
        "LONG_LIVE_STAGE2_TOKENIZER_DIR": str(tokenizer),
        "LONG_LIVE_STAGE2_VAE_CHECKPOINT": str(vae),
        "LONG_LIVE_FFPROBE": "/bin/true",
        "STAGE2_INFERENCE_CONFIG": "configs/sweep.yaml",
        "STAGE2_INFERENCE_PLAN_ONLY": "1",
        "STAGE2_INFERENCE_NPROC": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }
    for name in ("STAGE2_EARLY_CHECKPOINT", "STAGE2_EARLY_OUTPUT", "STAGE2_EARLY_LOG"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    completed = subprocess.run(
        ["bash", str(PROJECT_ROOT / "infer_stage2_tmp.sh"), "70"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "仅规划模式完成" in completed.stdout
    assert "caller-snapshots/checkpoint_stage2_g000070" in completed.stdout
    assert (train_root / "inference_early_g000070_sweep_eeeeeeeeeeeeeeee.log").is_file()


def test_stage2_entrypoint_never_imports_legacy_inference_pipeline() -> None:
    sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            PROJECT_ROOT / "utils" / "stage2_inference_runtime.py",
            PROJECT_ROOT / "scripts" / "run_stage2_inference.py",
        )
    )
    assert "CausalDiffusionInferencePipeline" not in sources
    assert "causal_diffusion_inference" not in sources
    assert "load_stage2_ema_generator_for_inference" in sources
    assert "Stage2RolloutPipeline" in sources
