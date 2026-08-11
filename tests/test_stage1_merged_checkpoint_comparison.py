import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest

import scripts.run_stage1_merged_checkpoint_comparison as comparison
from scripts.run_stage1_merged_checkpoint_comparison import (
    build_parallel_inference_jobs,
    clone_reference_lora_preparation,
    clone_reference_preparation,
    run_parallel_inference_jobs,
    validate_merged_checkpoint_provenance,
    write_side_by_side_video,
)
from utils.stage1_io import atomic_write_json, canonical_json_sha256


def _write_reference_preparation(
    root: Path,
    *,
    bucket_specs=(("landscape_480x832", 480, 832, 1),),
) -> Path:
    source_checkpoint = root / "old-merged.pt"
    source_checkpoint.parent.mkdir(parents=True)
    source_checkpoint.write_bytes(b"old")
    buckets = []
    row_id = 0
    for bucket_id, height, width, sample_count in bucket_specs:
        data_root = root / "prepared" / "datasets" / bucket_id
        output_dir = root / "prepared" / "videos" / bucket_id
        output_dir.mkdir(parents=True)
        records = []
        for bucket_index in range(sample_count):
            sample_name = f"{bucket_index:04d}_row{row_id:04d}"
            carrier = data_root / "video" / sample_name / "000.mp4"
            caption = data_root / "caption" / sample_name / "000.json"
            carrier.parent.mkdir(parents=True)
            caption.parent.mkdir(parents=True)
            carrier.write_bytes(b"carrier")
            atomic_write_json(caption, {"caption": f"cat row {row_id} moves"})
            records.append(
                {
                    "row_id": row_id,
                    "bucket_index": bucket_index,
                    "carrier_video": str(carrier),
                    "caption_json": str(caption),
                }
            )
            row_id += 1

        config_path = root / "prepared" / "configs" / f"{bucket_id}.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
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
                "image_or_video_shape": [
                    1,
                    24,
                    48,
                    height // 16,
                    width // 16,
                ],
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
        buckets.append(
            {
                "bucket_id": bucket_id,
                "height": height,
                "width": width,
                "data_root": str(data_root),
                "config_path": str(config_path),
                "output_dir": str(output_dir),
                "records": records,
            }
        )

    manifest = {
        "schema_version": 1,
        "metadata": {
            "path": str(root / "metadata.csv"),
            "sha256": "a" * 64,
            "record_count": row_id,
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
        "buckets": buckets,
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
    assert list(cloned_config.inference.sample_seeds) == [1]
    assert cloned_config.inference.dataloader_num_workers == 1
    assert cloned_config.inference.prefetch_factor == 2
    assert cloned_config.inference.pin_memory is True
    assert cloned_config.logging.seed == 1
    assert cloned["inference_variant"]["name"] == "premerged"
    assert cloned["buckets"][0]["output_model_type"] == "regular"
    assert cloned["model_paths"]["base_checkpoint"] == str(merged.resolve())

    persisted = json.loads(
        (output_root / "prepared_manifest.json").read_text(encoding="utf-8")
    )
    recorded_hash = persisted.pop("manifest_sha256")
    assert recorded_hash == canonical_json_sha256(persisted)


def test_merged_provenance_requires_the_same_base_step_and_ema_adapter(tmp_path):
    merged = tmp_path / "stage1_step3750_ema_merged.pt"
    merged.write_bytes(b"merged")
    manifest_path = merged.with_suffix(".manifest.json")
    manifest = {
        "schema": "longlive_stage1_merge_manifest",
        "schema_version": 2,
        "base": {"sha256": "a" * 64},
        "training_checkpoint": {
            "completed_step": 3750,
            "manifest_sha256": "b" * 64,
            "adapter": "adapter_ema.safetensors",
            "ema_adapter": {"sha256": "c" * 64},
        },
        "output": {
            "sha256": comparison.sha256_file(merged),
            "size": merged.stat().st_size,
            "dtype": "bfloat16",
            "strict_reload": True,
        },
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(manifest_path, manifest)

    report = validate_merged_checkpoint_provenance(
        merged_checkpoint=merged,
        merged_manifest_path=manifest_path,
        base_sha256="a" * 64,
        training_manifest_sha256="b" * 64,
        training_step=3750,
        adapter_sha256="c" * 64,
    )

    assert report["source_training_step"] == 3750
    assert report["source_adapter"] == "adapter_ema.safetensors"

    with pytest.raises(RuntimeError, match="provenance mismatch"):
        validate_merged_checkpoint_provenance(
            merged_checkpoint=merged,
            merged_manifest_path=manifest_path,
            base_sha256="a" * 64,
            training_manifest_sha256="b" * 64,
            training_step=3750,
            adapter_sha256="d" * 64,
        )


def test_clone_reference_lora_preparation_uses_base_and_ema_adapter(tmp_path):
    manifest_path = _write_reference_preparation(tmp_path / "reference")
    base = tmp_path / "converted_causal_base.pt"
    adapter = tmp_path / "adapter_ema.safetensors"
    base.write_bytes(b"base")
    adapter.write_bytes(b"adapter")
    output_root = tmp_path / "comparison" / "lora_inference"

    cloned = clone_reference_lora_preparation(
        reference_manifest_path=manifest_path,
        base_checkpoint=base,
        lora_checkpoint=adapter,
        adapter_config={
            "type": "lora",
            "rank": 32,
            "alpha": 32,
            "dropout": 0.0,
            "target_patterns": [r"^blocks\.[0-9]+\.self_attn\.q$"],
        },
        output_root=output_root,
    )

    config = OmegaConf.load(cloned["buckets"][0]["config_path"])
    assert config.generator_ckpt == str(base.resolve())
    assert config.checkpoints.generator_ckpt == str(base.resolve())
    assert config.lora_ckpt == str(adapter.resolve())
    assert config.checkpoints.lora_ckpt == str(adapter.resolve())
    assert config.adapter.rank == 32
    assert config.merge_lora is False
    assert config.inference.merge_lora is False
    assert list(config.inference.sample_seeds) == [1]
    assert cloned["inference_variant"]["name"] == "dynamic_lora"
    assert cloned["buckets"][0]["output_model_type"] == "lora"


def test_four_gpu_jobs_cover_two_three_sample_buckets_without_duplicates(tmp_path):
    manifest_path = _write_reference_preparation(
        tmp_path / "reference",
        bucket_specs=(
            ("landscape_480x832", 480, 832, 3),
            ("portrait_832x480", 832, 480, 3),
        ),
    )
    merged = tmp_path / "merged.pt"
    merged.write_bytes(b"merged")
    output_root = tmp_path / "comparison" / "merged_inference"
    clone_reference_preparation(
        reference_manifest_path=manifest_path,
        merged_checkpoint=merged,
        output_root=output_root,
    )

    jobs = build_parallel_inference_jobs(
        output_root / "prepared_manifest.json",
        gpu_ids=("0", "1", "2", "3"),
    )

    assert len(jobs) == 4
    assert [job.gpu_id for job in jobs] == ["0", "1", "2", "3"]
    assert [job.sample_indices for job in jobs] == [(0, 2), (1,), (0, 2), (1,)]
    assert {
        (job.bucket_id, index) for job in jobs for index in job.sample_indices
    } == {
        ("landscape_480x832", 0),
        ("landscape_480x832", 1),
        ("landscape_480x832", 2),
        ("portrait_832x480", 0),
        ("portrait_832x480", 1),
        ("portrait_832x480", 2),
    }
    for job in jobs:
        config = OmegaConf.load(job.config_path)
        assert tuple(config.inference.sample_indices) == job.sample_indices


def test_parallel_jobs_isolate_each_process_to_one_gpu(tmp_path):
    manifest_path = _write_reference_preparation(
        tmp_path / "reference",
        bucket_specs=(
            ("landscape_480x832", 480, 832, 3),
            ("portrait_832x480", 832, 480, 3),
        ),
    )
    merged = tmp_path / "merged.pt"
    merged.write_bytes(b"merged")
    output_root = tmp_path / "comparison" / "merged_inference"
    clone_reference_preparation(
        reference_manifest_path=manifest_path,
        merged_checkpoint=merged,
        output_root=output_root,
    )
    jobs = build_parallel_inference_jobs(
        output_root / "prepared_manifest.json",
        gpu_ids=("0", "1", "2", "3"),
    )
    calls = []

    def fake_runner(command, *, cwd, env, stdout, stderr, check):
        assert cwd == comparison.PROJECT_ROOT
        assert stderr == comparison.subprocess.STDOUT
        assert check is True
        assert env["CUDA_VISIBLE_DEVICES"] in {"0", "1", "2", "3"}
        assert all(
            name not in env
            for name in ("LOCAL_RANK", "RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
        )
        assert command[-2] == "--config_path"
        stdout.write(f"gpu={env['CUDA_VISIBLE_DEVICES']}\n".encode())
        calls.append((tuple(command), env["CUDA_VISIBLE_DEVICES"]))

    reports = run_parallel_inference_jobs(jobs, command_runner=fake_runner)

    assert len(calls) == 4
    assert len(reports) == 4
    assert {report["gpu_id"] for report in reports} == {"0", "1", "2", "3"}
    assert all(Path(report["log_path"]).read_text() for report in reports)


def test_four_gpu_comparison_assigns_one_format_bucket_per_gpu(tmp_path):
    manifest_path = _write_reference_preparation(
        tmp_path / "reference",
        bucket_specs=(
            ("landscape_480x832", 480, 832, 3),
            ("portrait_832x480", 832, 480, 3),
        ),
    )
    merged = tmp_path / "merged.pt"
    base = tmp_path / "base.pt"
    adapter = tmp_path / "adapter_ema.safetensors"
    for path in (merged, base, adapter):
        path.write_bytes(path.name.encode())
    merged_root = tmp_path / "comparison" / "merged_inference"
    lora_root = tmp_path / "comparison" / "lora_inference"
    clone_reference_preparation(
        reference_manifest_path=manifest_path,
        merged_checkpoint=merged,
        output_root=merged_root,
    )
    clone_reference_lora_preparation(
        reference_manifest_path=manifest_path,
        base_checkpoint=base,
        lora_checkpoint=adapter,
        adapter_config={"type": "lora", "rank": 2},
        output_root=lora_root,
    )

    jobs = build_parallel_inference_jobs(
        merged_root / "prepared_manifest.json",
        gpu_ids=("0", "1"),
    ) + build_parallel_inference_jobs(
        lora_root / "prepared_manifest.json",
        gpu_ids=("2", "3"),
    )

    assert [
        (job.gpu_id, job.variant, job.bucket_id, job.sample_indices) for job in jobs
    ] == [
        ("0", "premerged", "landscape_480x832", (0, 1, 2)),
        ("1", "premerged", "portrait_832x480", (0, 1, 2)),
        ("2", "dynamic_lora", "landscape_480x832", (0, 1, 2)),
        ("3", "dynamic_lora", "portrait_832x480", (0, 1, 2)),
    ]


def test_side_by_side_video_keeps_premerged_on_left(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison.shutil, "which", lambda name: f"/usr/bin/{name}")
    left = tmp_path / "premerged.mp4"
    right = tmp_path / "dynamic-lora.mp4"
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
