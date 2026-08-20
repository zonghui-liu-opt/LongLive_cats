from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import run_stage2_inference as entrypoint

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cli_loads_strict_config_and_delegates(monkeypatch, capsys) -> None:
    sentinel = object()
    seen = {}
    monkeypatch.setattr(
        entrypoint,
        "load_stage2_inference_config",
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
    assert "EXPECTED_ARTIFACTS=56" in text
    assert "rm -rf" not in text
    assert "checkpoint_stage2_g000280" not in text


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
