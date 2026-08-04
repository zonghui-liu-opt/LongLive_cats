import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

import scripts.run_stage1_continuation_validation as continuation_runner
from utils.stage1_checkpoint import write_checkpoint_manifest, write_success_marker
from utils.stage1_io import atomic_write_json

BASE_SHA256 = "b" * 64


def _write_artifact_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint_model_{step:06d}"
    checkpoint.mkdir()
    (checkpoint / "adapter_raw.safetensors").write_bytes(b"raw")
    (checkpoint / "adapter_ema.safetensors").write_bytes(b"ema")
    (checkpoint / "resolved_config.yaml").write_text(
        "model_kwargs: {}\n", encoding="utf-8"
    )
    atomic_write_json(
        checkpoint / "base_reference.json",
        {"base_sha256": BASE_SHA256},
    )
    write_checkpoint_manifest(
        checkpoint,
        completed_step=step,
        world_size=6,
        sequence_parallel_size=3,
        data_parallel_size=2,
        resumable=False,
    )
    write_success_marker(checkpoint, resumable=False)
    return checkpoint


def _runner_args(
    root: Path,
    *,
    checkpoint: Path,
    work_dir: Path,
    **overrides,
) -> SimpleNamespace:
    values = {
        "training_checkpoint": str(checkpoint),
        "metadata": str(root / "metadata.csv"),
        "work_dir": str(work_dir),
        "base_checkpoint": str(root / "base.pt"),
        "base_manifest": str(root / "base.manifest.json"),
        "source_checkpoint": str(root / "source.pt"),
        "architecture_root": str(root / "architecture"),
        "t5_checkpoint": str(root / "t5.pt"),
        "tokenizer_dir": str(root / "tokenizer"),
        "vae_checkpoint": str(root / "vae.pt"),
        "merge_device": "cpu",
        "minimum_first_frame_psnr_db": 12.0,
        "minimum_frame_std": 5.0,
        "minimum_temporal_abs_diff": 0.05,
        "keep_merged": False,
        "skip_base_finite_check": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _records() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            row_id=row_id, case_group=f"cat_{row_id // 2}_order_{row_id % 2}"
        )
        for row_id in range(8)
    ]


def _samples(root: Path) -> list[dict[str, object]]:
    return [
        {
            "row_id": row_id,
            "case_group": f"cat_{row_id // 2}_order_{row_id % 2}",
            "sink_size": sink_size,
            "output_video": str(
                root / f"cat_{row_id // 2}_order_{row_id % 2}" / f"sink{sink_size}.mp4"
            ),
        }
        for row_id in range(8)
        for sink_size in (0, 1)
    ]


def _run_with_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    work_name: str,
    keep_merged: bool = False,
    failing_stage: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    checkpoint = _write_artifact_checkpoint(tmp_path, 3750)
    work_dir = tmp_path / work_name
    records = _records()
    calls: dict[str, object] = {
        "base": [],
        "merge": [],
        "prepare": [],
        "command": [],
        "output": [],
        "html": [],
        "merge_config": [],
    }
    merge_config = object()
    samples = _samples(work_dir / "checkpoint_model_003750" / "continuation")

    monkeypatch.setattr(
        continuation_runner,
        "load_continuation_metadata",
        lambda path: records,
    )

    def fake_load_merge_config(resolved_checkpoint, architecture_root):
        calls["merge_config"].append((resolved_checkpoint, architecture_root))
        return merge_config

    monkeypatch.setattr(
        continuation_runner,
        "_load_merge_config",
        fake_load_merge_config,
    )

    def fake_base_validator(*args, **kwargs):
        calls["base"].append((args, kwargs))
        return {"status": "pass", "output_sha256": BASE_SHA256}

    def fake_merge(**kwargs):
        assert kwargs["config"] is merge_config
        calls["merge"].append(kwargs)
        Path(kwargs["output_path"]).write_bytes(b"merged EMA")
        atomic_write_json(kwargs["output_manifest_path"], {"kind": "ema"})
        return {"output": {"sha256": "c" * 64}}

    def fake_prepare(**kwargs):
        calls["prepare"].append(kwargs)
        prepared_root = Path(kwargs["output_root"])
        config_root = prepared_root / "configs"
        config_root.mkdir(parents=True)
        portrait = config_root / "portrait.yaml"
        landscape = config_root / "landscape.yaml"
        portrait.write_text("fixture: portrait\n", encoding="utf-8")
        landscape.write_text("fixture: landscape\n", encoding="utf-8")
        atomic_write_json(prepared_root / "prepared_manifest.json", {"fixture": True})
        return {
            "buckets": [
                {
                    "bucket_id": "portrait_832x480",
                    "config_path": str(portrait),
                },
                {
                    "bucket_id": "landscape_480x832",
                    "config_path": str(landscape),
                },
            ]
        }

    def fake_command(command, **kwargs):
        calls["command"].append((command, kwargs))
        if failing_stage == "command":
            raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(returncode=0)

    def fake_output_validator(*args, **kwargs):
        calls["output"].append((args, kwargs))
        if failing_stage == "output":
            raise RuntimeError("fixture output validation failure")
        return {
            "status": "pass",
            "sample_count": 16,
            "samples": samples,
        }

    def fake_html_writer(*args, **kwargs):
        calls["html"].append((args, kwargs))
        if failing_stage == "html":
            raise RuntimeError("fixture HTML failure")
        output_path = Path(args[0])
        output_path.write_text("<html>fixture</html>\n", encoding="utf-8")
        return output_path

    args = _runner_args(
        tmp_path,
        checkpoint=checkpoint,
        work_dir=work_dir,
        keep_merged=keep_merged,
    )
    report = continuation_runner.run_validation(
        args,
        base_validator=fake_base_validator,
        merge_fn=fake_merge,
        prepare_fn=fake_prepare,
        output_validator=fake_output_validator,
        html_writer=fake_html_writer,
        command_runner=fake_command,
    )
    return report, calls


@pytest.mark.parametrize("keep_merged", [False, True])
def test_runner_locks_step_sampling_merge_and_two_geometry_commands(
    tmp_path, monkeypatch, keep_merged
):
    report, calls = _run_with_fakes(
        tmp_path,
        monkeypatch,
        work_name=f"successful-{keep_merged}",
        keep_merged=keep_merged,
    )

    work_dir = tmp_path / f"successful-{keep_merged}"
    checkpoint = (tmp_path / "checkpoint_model_003750").resolve()
    merged_path = work_dir / "checkpoint_model_003750" / "stage1_causal_ema_merged.pt"
    checkpoint_report = report["checkpoint"]
    assert report["status"] == "pass"
    assert report["sample_count"] == 16
    assert checkpoint_report["optimizer_step"] == 3750
    assert checkpoint_report["sample_count"] == 16
    assert checkpoint_report["merged_checkpoint_retained"] is keep_merged
    assert merged_path.exists() is keep_merged
    assert report["sampling"] == {
        "solver": "unipc",
        "sampling_steps": 50,
        "guidance_scale": 5.0,
        "seed": 1,
        "sink_sizes": [0, 1],
    }

    assert calls["base"] == [
        (
            (str(tmp_path / "base.pt"), str(tmp_path / "base.manifest.json")),
            {
                "source_checkpoint": str(tmp_path / "source.pt"),
                "expected_num_frame_per_block": 8,
                "check_finite": True,
            },
        )
    ]

    assert calls["merge_config"] == [(checkpoint, str(tmp_path / "architecture"))]
    assert len(calls["merge"]) == 1
    merge_call = calls["merge"][0]
    assert merge_call == {
        "base_checkpoint": str(tmp_path / "base.pt"),
        "training_checkpoint": checkpoint,
        "output_path": merged_path,
        "output_manifest_path": (
            work_dir / "checkpoint_model_003750" / "merge_manifest.json"
        ),
        "config": merge_call["config"],
        "device": "cpu",
    }
    assert merge_call["config"] is not None

    assert len(calls["prepare"]) == 1
    prepare_call = calls["prepare"][0]
    assert prepare_call["base_checkpoint"] == merged_path
    assert prepare_call["sampling_steps"] == 50
    assert prepare_call["guidance_scale"] == 5.0
    assert prepare_call["seed"] == 1
    assert Path(prepare_call["video_root"]) == (
        work_dir / "checkpoint_model_003750" / "continuation"
    )

    command_calls = calls["command"]
    assert len(command_calls) == 2
    expected_configs = [
        work_dir
        / "checkpoint_model_003750"
        / "prepared_continuation"
        / "configs"
        / "portrait.yaml",
        work_dir
        / "checkpoint_model_003750"
        / "prepared_continuation"
        / "configs"
        / "landscape.yaml",
    ]
    for (command, kwargs), expected_config in zip(
        command_calls, expected_configs, strict=True
    ):
        assert command == [
            sys.executable,
            str(continuation_runner.PROJECT_ROOT / "inference.py"),
            "--config_path",
            str(expected_config),
        ]
        assert kwargs == {
            "cwd": continuation_runner.PROJECT_ROOT,
            "check": True,
        }

    assert len(calls["output"]) == 1
    output_args, output_kwargs = calls["output"][0]
    assert output_args == (
        work_dir
        / "checkpoint_model_003750"
        / "prepared_continuation"
        / "prepared_manifest.json",
    )
    assert output_kwargs == {
        "minimum_first_frame_psnr_db": 12.0,
        "minimum_frame_std": 5.0,
        "minimum_temporal_abs_diff": 0.05,
    }

    assert len(calls["html"]) == 1
    html_args, html_kwargs = calls["html"][0]
    assert html_args[0] == work_dir / "comparison.html"
    assert len(html_args[1]) == 8
    assert len(html_args[2]) == 16
    assert html_kwargs == {"work_dir": work_dir}
    persisted = json.loads(
        (work_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    assert persisted["status"] == "pass"
    assert persisted["sample_count"] == 16


def test_runner_rejects_any_checkpoint_other_than_step_3750_before_merge(
    tmp_path, monkeypatch
):
    checkpoint = _write_artifact_checkpoint(tmp_path, 3749)
    monkeypatch.setattr(
        continuation_runner,
        "load_continuation_metadata",
        lambda _path: _records(),
    )
    merge_calls = []
    work_dir = tmp_path / "wrong-step"
    args = _runner_args(tmp_path, checkpoint=checkpoint, work_dir=work_dir)

    with pytest.raises(ValueError, match="requires optimizer step 3750, got 3749"):
        continuation_runner.run_validation(
            args,
            base_validator=lambda *_args, **_kwargs: {
                "status": "pass",
                "output_sha256": BASE_SHA256,
            },
            merge_fn=lambda **kwargs: merge_calls.append(kwargs),
        )

    assert merge_calls == []
    report = json.loads(
        (work_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "failed"
    assert report["checkpoint"] is None
    assert "step 3750" in report["error"]


def test_runner_requires_an_empty_work_directory_before_any_dependency_runs(
    tmp_path, monkeypatch
):
    checkpoint = _write_artifact_checkpoint(tmp_path, 3750)
    work_dir = tmp_path / "not-empty"
    work_dir.mkdir()
    (work_dir / "do-not-overwrite.txt").write_text("user data", encoding="utf-8")
    metadata_calls = []
    monkeypatch.setattr(
        continuation_runner,
        "load_continuation_metadata",
        lambda path: metadata_calls.append(path),
    )
    args = _runner_args(tmp_path, checkpoint=checkpoint, work_dir=work_dir)

    with pytest.raises(FileExistsError, match="must be empty"):
        continuation_runner.run_validation(args)

    assert metadata_calls == []
    assert (work_dir / "do-not-overwrite.txt").read_text(encoding="utf-8") == (
        "user data"
    )
    assert not (work_dir / "validation_report.json").exists()


@pytest.mark.parametrize(
    ("failing_stage", "expected_exception", "message"),
    [
        ("command", subprocess.CalledProcessError, "returned non-zero exit status 1"),
        ("output", RuntimeError, "fixture output validation failure"),
        ("html", RuntimeError, "fixture HTML failure"),
    ],
)
def test_runner_failure_report_is_atomic_and_retains_merged_checkpoint(
    tmp_path, monkeypatch, failing_stage, expected_exception, message
):
    with pytest.raises(expected_exception, match=message):
        _run_with_fakes(
            tmp_path,
            monkeypatch,
            work_name=f"failure-{failing_stage}",
            failing_stage=failing_stage,
        )

    work_dir = tmp_path / f"failure-{failing_stage}"
    merged_path = work_dir / "checkpoint_model_003750" / "stage1_causal_ema_merged.pt"
    report = json.loads(
        (work_dir / "validation_report.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "failed"
    assert report["checkpoint"]["status"] == "failed"
    assert report["checkpoint"]["merged_checkpoint_retained"] is True
    assert "error" in report
    assert "error" in report["checkpoint"]
    assert "active_bucket" not in report["checkpoint"]
    assert merged_path.read_bytes() == b"merged EMA"


def test_cli_parser_exposes_only_operational_inputs_and_locked_experiment_help(
    monkeypatch, tmp_path, capsys
):
    for name in (
        "LONG_LIVE_STAGE1_BASE_CHECKPOINT",
        "LONG_LIVE_STAGE1_BASE_MANIFEST",
        "LONG_LIVE_STAGE1_ARCHITECTURE_ROOT",
        "LONG_LIVE_STAGE1_T5_CHECKPOINT",
        "LONG_LIVE_STAGE1_TOKENIZER_DIR",
        "LONG_LIVE_STAGE1_VAE_CHECKPOINT",
    ):
        monkeypatch.delenv(name, raising=False)

    parser = continuation_runner.build_parser()
    required = [
        "--work-dir",
        str(tmp_path / "work"),
        "--base-checkpoint",
        str(tmp_path / "base.pt"),
        "--base-manifest",
        str(tmp_path / "base.manifest.json"),
        "--architecture-root",
        str(tmp_path / "architecture"),
        "--t5-checkpoint",
        str(tmp_path / "t5.pt"),
        "--tokenizer-dir",
        str(tmp_path / "tokenizer"),
        "--vae-checkpoint",
        str(tmp_path / "vae.pt"),
    ]
    args = parser.parse_args(required)
    assert args.training_checkpoint is None
    assert Path(args.metadata) == continuation_runner.FORMAL_METADATA
    assert args.keep_merged is False
    assert args.merge_device == "cpu"
    assert not hasattr(args, "sampling_steps")
    assert not hasattr(args, "guidance_scale")
    assert not hasattr(args, "seed")

    selected = parser.parse_args(
        required
        + [
            "--training-checkpoint",
            str(tmp_path / "checkpoint_model_003750"),
            "--keep-merged",
            "--skip-base-finite-check",
        ]
    )
    assert selected.keep_merged is True
    assert selected.skip_base_finite_check is True

    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    normalized_help = " ".join(help_text.split())
    assert "step-3750" in normalized_help
    assert "sink=0 and sink=1" in normalized_help
    assert "--keep-merged" in normalized_help
    assert "--work-dir" in normalized_help


def test_main_prints_passing_report_and_returns_zero(monkeypatch, capsys):
    expected = {"status": "pass", "sample_count": 16}
    sentinel_args = SimpleNamespace()

    class FakeParser:
        def parse_args(self, argv):
            assert argv == ["--fixture"]
            return sentinel_args

    monkeypatch.setattr(continuation_runner, "build_parser", lambda: FakeParser())
    monkeypatch.setattr(
        continuation_runner,
        "run_validation",
        lambda args: expected if args is sentinel_args else None,
    )

    assert continuation_runner.main(["--fixture"]) == 0
    assert json.loads(capsys.readouterr().out) == expected
