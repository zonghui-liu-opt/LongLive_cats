from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest

import train as train_entrypoint
import trainer as trainer_package
import utils.stage2_config as stage2_config

PROJECT_ROOT = Path(__file__).parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_i2v_stage2_600cats.yaml"


def _run_train_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(PROJECT_ROOT)
    return subprocess.run(
        [sys.executable, "-B", str(PROJECT_ROOT / "train.py"), *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def _capture_stage2_options(monkeypatch, tmp_path, *arguments: str):
    raw = OmegaConf.create(
        {
            "config_schema": "longlive_stage2_train/v1",
            "algorithm": {"trainer": "stage2_distillation"},
        }
    )
    resolved = SimpleNamespace(trainer="stage2_distillation")
    captured = []

    class TinyStage2Trainer:
        def __init__(self, config, **options):
            assert config is resolved
            captured.append(options)

        def train(self):
            return None

    monkeypatch.setattr(train_entrypoint.OmegaConf, "load", lambda _path: raw)
    monkeypatch.setattr(stage2_config, "resolve_stage2_config", lambda _raw: resolved)
    monkeypatch.setattr(
        trainer_package,
        "Stage2DistillationTrainer",
        TinyStage2Trainer,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config_path",
            str(CONFIG_PATH),
            "--logdir",
            str(tmp_path / "run"),
            *arguments,
        ],
    )

    train_entrypoint.main()

    assert len(captured) == 1
    return captured[0]


def test_stage2_raw_yaml_dispatches_before_legacy_normalization(monkeypatch, tmp_path):
    raw = OmegaConf.create(
        {
            "config_schema": "longlive_stage2_train/v1",
            "algorithm": {"trainer": "stage2_distillation"},
        }
    )
    resolved = SimpleNamespace(trainer="stage2_distillation")
    calls = []

    class TinyStage2Trainer:
        def __init__(self, config, **options):
            calls.append(("construct", config, options))

        def train(self):
            calls.append(("train",))

    monkeypatch.setattr(train_entrypoint.OmegaConf, "load", lambda _path: raw)
    monkeypatch.setattr(
        train_entrypoint,
        "normalize_config",
        lambda _raw: pytest.fail("legacy normalize_config saw Stage-2 raw YAML"),
    )
    monkeypatch.setattr(
        stage2_config,
        "resolve_stage2_config",
        lambda value: calls.append(("resolve", value)) or resolved,
    )
    monkeypatch.setattr(
        trainer_package,
        "Stage2DistillationTrainer",
        TinyStage2Trainer,
        raising=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config_path",
            str(CONFIG_PATH),
            "--logdir",
            str(tmp_path / "run"),
            "--no-save",
            "--no-visualize",
            "--no-auto-resume",
        ],
    )

    train_entrypoint.main()

    assert calls[0] == ("resolve", raw)
    assert calls[1][0:2] == ("construct", resolved)
    assert calls[1][2] == {
        "output_dir": str(tmp_path / "run"),
        "no_save": True,
        "no_visualize": True,
        "auto_resume": False,
        "smoke_mode": None,
    }
    assert calls[2] == ("train",)


def test_stage2_strict_schema_error_returns_nonzero_exit(tmp_path):
    raw = OmegaConf.load(CONFIG_PATH)
    raw["unexpected_stage2_section"] = {"must_fail": True}
    invalid = tmp_path / "invalid_stage2.yaml"
    OmegaConf.save(raw, invalid)

    completed = _run_train_cli(
        "--config_path",
        str(invalid),
        "--logdir",
        str(tmp_path / "run"),
        "--no-save",
    )

    assert completed.returncode != 0
    assert "unexpected_stage2_section" in completed.stderr
    assert "Traceback" in completed.stderr


@pytest.mark.parametrize(
    ("mode", "expected_no_save", "expected_auto_resume"),
    [
        pytest.param("C0", False, False, id="C0-cold-boundary-save"),
        pytest.param("C1", False, True, id="C1-resume-boundary-save"),
        pytest.param("C2", True, True, id="C2-resume-forced-dfd-discard"),
    ],
)
def test_stage2_smoke_profiles_translate_to_one_unambiguous_trainer_contract(
    monkeypatch,
    tmp_path,
    mode,
    expected_no_save,
    expected_auto_resume,
):
    options = _capture_stage2_options(
        monkeypatch,
        tmp_path,
        "--stage2-smoke",
        mode,
    )

    assert options == {
        "output_dir": str(tmp_path / "run"),
        "no_save": expected_no_save,
        "no_visualize": False,
        "auto_resume": expected_auto_resume,
        "smoke_mode": mode,
    }


def test_stage2_smoke_cli_accepts_only_c0_c1_c2(tmp_path):
    completed = _run_train_cli(
        "--config_path",
        str(CONFIG_PATH),
        "--logdir",
        str(tmp_path / "invalid-smoke"),
        "--stage2-smoke",
        "C3",
    )

    assert completed.returncode != 0
    assert "invalid choice: 'C3'" in completed.stderr
    assert "choose from 'C0', 'C1', 'C2'" in completed.stderr


def test_stage2_training_exception_is_not_swallowed(monkeypatch, tmp_path):
    raw = OmegaConf.create(
        {
            "config_schema": "longlive_stage2_train/v1",
            "algorithm": {"trainer": "stage2_distillation"},
        }
    )
    resolved = SimpleNamespace(trainer="stage2_distillation")

    class ExpectedTrainingFailure(RuntimeError):
        pass

    class FailingStage2Trainer:
        def __init__(self, config, **_options):
            assert config is resolved

        def train(self):
            raise ExpectedTrainingFailure("stage2 failure must reach torchrun")

    monkeypatch.setattr(train_entrypoint.OmegaConf, "load", lambda _path: raw)
    monkeypatch.setattr(stage2_config, "resolve_stage2_config", lambda _raw: resolved)
    monkeypatch.setattr(
        trainer_package,
        "Stage2DistillationTrainer",
        FailingStage2Trainer,
        raising=False,
    )
    monkeypatch.setattr(
        train_entrypoint,
        "normalize_config",
        lambda _raw: pytest.fail("legacy normalize_config saw Stage-2 raw YAML"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config_path",
            str(CONFIG_PATH),
            "--logdir",
            str(tmp_path / "run"),
            "--no-save",
        ],
    )

    with pytest.raises(ExpectedTrainingFailure, match="must reach torchrun"):
        train_entrypoint.main()
