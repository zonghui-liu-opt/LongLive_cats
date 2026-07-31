import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

from scripts.run_stage1_training_checkpoints_validation import (
    _parse_steps,
    run_validation,
)
from utils.config import DEFAULT_NEGATIVE_PROMPT
from utils.stage1_causal_validation import discover_stage1_training_checkpoints
from utils.stage1_checkpoint import write_checkpoint_manifest, write_success_marker
from utils.stage1_io import atomic_write_json


def _write_artifact_checkpoint(root: Path, step: int, base_sha256: str) -> Path:
    directory = root / f"checkpoint_model_{step:06d}"
    directory.mkdir()
    (directory / "adapter_raw.safetensors").write_bytes(b"raw")
    (directory / "adapter_ema.safetensors").write_bytes(b"ema")
    (directory / "resolved_config.yaml").write_text("model_kwargs: {}\n", encoding="utf-8")
    atomic_write_json(directory / "base_reference.json", {"base_sha256": base_sha256})
    write_checkpoint_manifest(
        directory,
        completed_step=step,
        world_size=6,
        sequence_parallel_size=3,
        data_parallel_size=2,
        resumable=False,
    )
    write_success_marker(directory, resumable=False)
    return directory


def _write_testset(root: Path) -> Path:
    image_path = root / "cat.png"
    Image.new("RGB", (832, 480), color=(220, 220, 220)).save(image_path)
    metadata = root / "metadata.csv"
    with metadata.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_image", "prompt", "height", "width", "bucket"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "input_image": image_path.name,
                "prompt": "a cat moves",
                "height": 480,
                "width": 832,
                "bucket": "landscape",
            }
        )
    return metadata


def test_checkpoint_discovery_is_sorted_selectable_and_rejects_incomplete(tmp_path):
    base_sha256 = "a" * 64
    _write_artifact_checkpoint(tmp_path, 300, base_sha256)
    selected = _write_artifact_checkpoint(tmp_path, 75, base_sha256)

    assert discover_stage1_training_checkpoints(
        tmp_path, expected_base_sha256=base_sha256
    ) == [selected.resolve(), (tmp_path / "checkpoint_model_000300").resolve()]
    assert discover_stage1_training_checkpoints(
        tmp_path, steps=[300], expected_base_sha256=base_sha256
    ) == [(tmp_path / "checkpoint_model_000300").resolve()]
    with pytest.raises(RuntimeError, match="missing"):
        discover_stage1_training_checkpoints(
            tmp_path, steps=[150], expected_base_sha256=base_sha256
        )

    (tmp_path / "checkpoint_model_000450").mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        discover_stage1_training_checkpoints(tmp_path, steps=[75])


def test_parse_steps_is_strict():
    assert _parse_steps("75,300,750") == [75, 300, 750]
    with pytest.raises(argparse.ArgumentTypeError, match="duplicates"):
        _parse_steps("75,75")


def test_batch_runner_uses_ema_merged_checkpoint_and_writes_comparison(tmp_path):
    base_sha256 = "b" * 64
    checkpoint = _write_artifact_checkpoint(tmp_path, 75, base_sha256)
    metadata = _write_testset(tmp_path)
    work_dir = tmp_path / "validation"
    command_calls = []

    def fake_base_validator(*_args, **_kwargs):
        return {"status": "pass", "output_sha256": base_sha256}

    def fake_merge(**kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.write_bytes(b"merged-ema")
        atomic_write_json(kwargs["output_manifest_path"], {"kind": "ema"})
        assert Path(kwargs["training_checkpoint"]) == checkpoint
        return {"output": {"sha256": "c" * 64}}

    output_video_holder = {}

    def fake_prepare(**kwargs):
        merged_path = Path(kwargs["base_checkpoint"])
        assert merged_path.name == "stage1_causal_ema_merged.pt"
        assert merged_path.read_bytes() == b"merged-ema"
        assert kwargs["negative_prompt"] == DEFAULT_NEGATIVE_PROMPT
        prepared_root = Path(kwargs["output_root"])
        config_path = prepared_root / "configs" / "landscape.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("fixture: true\n", encoding="utf-8")
        output_dir = prepared_root / "videos" / "landscape_480x832"
        output_dir.mkdir(parents=True)
        output_video = output_dir / "rank0-0-0_regular.mp4"
        output_video.write_bytes(b"video")
        output_video_holder["path"] = output_video
        atomic_write_json(prepared_root / "prepared_manifest.json", {"fixture": True})
        return {
            "buckets": [
                {
                    "bucket_id": "landscape_480x832",
                    "config_path": str(config_path),
                }
            ]
        }

    def fake_output_validator(*_args, **_kwargs):
        return {
            "status": "pass",
            "sample_count": 1,
            "samples": [
                {
                    "row_id": 0,
                    "output_video": str(output_video_holder["path"]),
                    "metrics": {
                        "first_frame_psnr_db": 31.0,
                        "mean_frame_std": 22.0,
                        "mean_temporal_abs_diff": 1.5,
                    },
                }
            ],
        }

    args = SimpleNamespace(
        metadata=str(metadata),
        training_root=None,
        training_checkpoint=[str(checkpoint)],
        steps=None,
        work_dir=str(work_dir),
        base_checkpoint=str(tmp_path / "base.pt"),
        base_manifest=str(tmp_path / "base.manifest.json"),
        source_checkpoint=None,
        architecture_root=str(tmp_path / "architecture"),
        t5_checkpoint=str(tmp_path / "t5.pt"),
        tokenizer_dir=str(tmp_path / "tokenizer"),
        vae_checkpoint=str(tmp_path / "vae.pt"),
        sampling_steps=50,
        guidance_scale=5.0,
        seed=1,
        negative_prompt=DEFAULT_NEGATIVE_PROMPT,
        merge_device="cpu",
        minimum_first_frame_psnr_db=12.0,
        minimum_frame_std=5.0,
        minimum_temporal_abs_diff=0.05,
        keep_merged=False,
        continue_on_error=False,
        skip_base_finite_check=True,
    )
    report = run_validation(
        args,
        base_validator=fake_base_validator,
        merge_fn=fake_merge,
        prepare_fn=fake_prepare,
        output_validator=fake_output_validator,
        command_runner=lambda *call_args, **call_kwargs: command_calls.append(
            (call_args, call_kwargs)
        ),
    )

    assert report["status"] == "pass"
    assert report["passed_checkpoint_count"] == 1
    assert report["checkpoints"][0]["merged_checkpoint_retained"] is False
    assert not (work_dir / "checkpoint_model_000075" / "stage1_causal_ema_merged.pt").exists()
    assert len(command_calls) == 1
    comparison = (work_dir / "comparison.html").read_text(encoding="utf-8")
    assert "step 000075" in comparison
    assert "rank0-0-0_regular.mp4" in comparison
    persisted = json.loads((work_dir / "validation_report.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "pass"
