import json
from pathlib import Path

from omegaconf import OmegaConf

import scripts.run_stage1_merged_checkpoint_comparison as comparison
from scripts.run_stage1_merged_checkpoint_comparison import (
    clone_reference_preparation,
    write_side_by_side_video,
)
from utils.stage1_io import atomic_write_json, canonical_json_sha256


def _write_reference_preparation(root: Path) -> Path:
    data_root = root / "prepared" / "datasets" / "landscape_480x832"
    carrier = data_root / "video" / "0000_row0000" / "000.mp4"
    caption = data_root / "caption" / "0000_row0000" / "000.json"
    carrier.parent.mkdir(parents=True)
    caption.parent.mkdir(parents=True)
    carrier.write_bytes(b"carrier")
    atomic_write_json(caption, {"caption": "a cat moves"})

    output_dir = root / "prepared" / "videos" / "landscape_480x832"
    output_dir.mkdir(parents=True)
    config_path = root / "prepared" / "configs" / "landscape_480x832.yaml"
    config_path.parent.mkdir(parents=True)
    source_checkpoint = root / "old-merged.pt"
    source_checkpoint.write_bytes(b"old")
    config = {
        "model_kwargs": {
            "model_name": "Wan2.2-TI2V-5B",
            "num_frame_per_block": 8,
        },
        "i2v": True,
        "use_ema": False,
        "output_folder": str(output_dir),
        "num_samples": 1,
        "num_output_frames": 24,
        "save_with_index": True,
        "data": {
            "data_path": str(data_root),
            "image_or_video_shape": [1, 24, 48, 30, 52],
        },
        "inference": {
            "sampling_steps": 50,
            "guidance_scale": 5.0,
            "negative_prompt": "bad quality",
        },
        "checkpoints": {"generator_ckpt": str(source_checkpoint)},
        "logging": {"seed": 1},
    }
    config_path.write_text(
        OmegaConf.to_yaml(OmegaConf.create(config), sort_keys=False),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "metadata": {
            "path": str(root / "metadata.csv"),
            "sha256": "a" * 64,
            "record_count": 1,
            "aggregate_row_sha256": "b" * 64,
        },
        "model_paths": {"base_checkpoint": str(source_checkpoint)},
        "frame_policy": {
            "num_latent_frames": 24,
            "num_frame_per_block": 8,
            "temporal_compression_ratio": 4,
            "expected_pixel_frames": 93,
            "carrier_frames": 97,
            "fps": 24,
        },
        "sampling": {
            "solver": "unipc",
            "sampling_steps": 50,
            "guidance_scale": 5.0,
            "seed": 1,
            "negative_prompt": "bad quality",
        },
        "buckets": [
            {
                "bucket_id": "landscape_480x832",
                "height": 480,
                "width": 832,
                "data_root": str(data_root),
                "config_path": str(config_path),
                "output_dir": str(output_dir),
                "records": [
                    {
                        "row_id": 0,
                        "carrier_video": str(carrier),
                        "caption_json": str(caption),
                    }
                ],
            }
        ],
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = root / "prepared" / "prepared_manifest.json"
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def test_clone_reference_preparation_changes_only_runtime_destinations(tmp_path):
    manifest_path = _write_reference_preparation(tmp_path / "reference")
    source_config = OmegaConf.load(
        tmp_path / "reference" / "prepared" / "configs" / "landscape_480x832.yaml"
    )
    merged = tmp_path / "stage1_step3750_ema_merged.pt"
    merged.write_bytes(b"merged")
    output_root = tmp_path / "comparison" / "merged_inference"

    cloned = clone_reference_preparation(
        reference_manifest_path=manifest_path,
        merged_checkpoint=merged,
        output_root=output_root,
    )

    cloned_config = OmegaConf.load(cloned["buckets"][0]["config_path"])
    assert cloned_config.generator_ckpt == str(merged.resolve())
    assert cloned_config.checkpoints.generator_ckpt == str(merged.resolve())
    assert cloned_config.output_folder == str(
        (output_root / "videos" / "landscape_480x832").resolve()
    )
    assert cloned_config.data.data_path == source_config.data.data_path
    assert cloned_config.inference.sampling_steps == 50
    assert cloned_config.inference.guidance_scale == 5.0
    assert cloned_config.logging.seed == 1
    assert cloned["model_paths"]["base_checkpoint"] == str(merged.resolve())

    persisted = json.loads(
        (output_root / "prepared_manifest.json").read_text(encoding="utf-8")
    )
    recorded_hash = persisted.pop("manifest_sha256")
    assert recorded_hash == canonical_json_sha256(persisted)


def test_side_by_side_video_keeps_reference_on_left(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison.shutil, "which", lambda name: f"/usr/bin/{name}")
    left = tmp_path / "reference.mp4"
    right = tmp_path / "merged.mp4"
    output = tmp_path / "paired.mp4"
    left.write_bytes(b"left")
    right.write_bytes(b"right")
    calls = []

    def fake_probe(path):
        path = Path(path).resolve()
        if path == output.resolve():
            return {"width": 1664, "height": 480, "frame_count": 93, "fps": 24.0}
        return {"width": 832, "height": 480, "frame_count": 93, "fps": 24.0}

    def fake_runner(command, *, check):
        assert check is True
        calls.append(command)
        Path(command[-1]).write_bytes(b"paired")

    stream = write_side_by_side_video(
        left,
        right,
        output,
        command_runner=fake_runner,
        video_probe=fake_probe,
    )

    command = calls[0]
    first_input = command.index("-i")
    second_input = command.index("-i", first_input + 1)
    assert command[first_input + 1] == str(left.resolve())
    assert command[second_input + 1] == str(right.resolve())
    assert "hstack" in command[command.index("-filter_complex") + 1]
    assert stream["width"] == 1664
    assert output.read_bytes() == b"paired"
