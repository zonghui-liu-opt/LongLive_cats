import csv
import json
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image
import pytest

import scripts.run_stage1_merged_checkpoint_comparison as comparison
from scripts.run_stage1_merged_checkpoint_comparison import (
    build_parser,
    build_parallel_inference_jobs,
    clone_reference_lora_preparation,
    clone_reference_preparation,
    prepare_fresh_comparison_inputs,
    run_parallel_inference_jobs,
    validate_merged_checkpoint_provenance,
    write_side_by_side_video,
)
from utils.stage1_io import atomic_write_json, canonical_json_sha256
from utils.stage1_causal_validation import prepare_causal_testsets


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


def test_reference_directory_is_optional_at_cli_parse_time(tmp_path):
    args = build_parser().parse_args(
        [
            "--merged-checkpoint",
            str(tmp_path / "merged.pt"),
            "--merged-manifest",
            str(tmp_path / "merged.manifest.json"),
            "--base-checkpoint",
            str(tmp_path / "base.pt"),
            "--training-checkpoint",
            str(tmp_path / "checkpoint_model_003750"),
            "--metadata",
            str(tmp_path / "metadata.csv"),
            "--work-dir",
            str(tmp_path / "work"),
        ]
    )

    assert args.reference_checkpoint_dir is None
    assert args.sampling_steps == 50
    assert args.guidance_scale == 5.0
    assert args.seed == 1


def test_fresh_comparison_preparation_uses_locked_stage1_contract(tmp_path):
    source_manifest_path = _write_reference_preparation(tmp_path / "source")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    calls = []

    def fake_builder(**kwargs):
        calls.append(kwargs)
        output_root = Path(kwargs["output_root"])
        output_root.mkdir(parents=True)
        atomic_write_json(output_root / "prepared_manifest.json", source_manifest)
        return source_manifest

    manifest_path, manifest = prepare_fresh_comparison_inputs(
        metadata_path=tmp_path / "metadata.csv",
        output_root=tmp_path / "work" / "shared_prepared",
        base_checkpoint=tmp_path / "base.pt",
        architecture_root=tmp_path / "architecture",
        t5_checkpoint=tmp_path / "t5.pt",
        tokenizer_dir=tmp_path / "tokenizer",
        vae_checkpoint=tmp_path / "vae.pt",
        preparation_builder=fake_builder,
    )

    assert manifest_path.name == "prepared_manifest.json"
    assert manifest == source_manifest
    assert len(calls) == 1
    assert calls[0]["num_latent_frames"] == 24
    assert calls[0]["num_frame_per_block"] == 8
    assert calls[0]["minimum_source_frames"] == 97
    assert calls[0]["sampling_steps"] == 50
    assert calls[0]["guidance_scale"] == 5.0
    assert calls[0]["seed"] == 1


def test_comparison_html_omits_missing_historical_reference(tmp_path):
    sample = {
        "row_id": 0,
        "bucket_id": "landscape_480x832",
        "sample_seed": 1,
        "comparison_video": str(tmp_path / "side_by_side" / "row0.mp4"),
    }

    without_reference = comparison._comparison_html(
        samples=[sample],
        work_dir=tmp_path,
    )
    with_reference = comparison._comparison_html(
        samples=[
            {
                **sample,
                "reference_video": str(tmp_path / "old" / "rank0.mp4"),
            }
        ],
        work_dir=tmp_path,
    )

    assert "原 infer_stage1 历史参考" not in without_reference
    assert "reference_video" not in without_reference
    assert "原 infer_stage1 历史参考" in with_reference


def test_run_comparison_without_reference_builds_shared_inputs(
    tmp_path,
    monkeypatch,
):
    Image.new("RGB", (832, 480), color=(220, 220, 220)).save(
        tmp_path / "landscape.png"
    )
    Image.new("RGB", (480, 832), color=(200, 200, 200)).save(
        tmp_path / "portrait.png"
    )
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_image", "prompt", "height", "width", "bucket"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "input_image": "landscape.png",
                    "prompt": "landscape cat moves",
                    "height": 480,
                    "width": 832,
                    "bucket": "landscape",
                },
                {
                    "input_image": "portrait.png",
                    "prompt": "portrait cat moves",
                    "height": 832,
                    "width": 480,
                    "bucket": "portrait",
                },
            ]
        )

    merged = tmp_path / "merged.pt"
    merged_manifest = tmp_path / "merged.manifest.json"
    base = tmp_path / "base.pt"
    for path in (merged, merged_manifest, base):
        path.write_bytes(path.name.encode())
    training = tmp_path / "checkpoint_model_003750"
    training.mkdir()
    (training / "adapter_ema.safetensors").write_bytes(b"adapter")
    (training / "resolved_config.yaml").write_text(
        "adapter:\n  type: lora\n  rank: 2\n",
        encoding="utf-8",
    )
    architecture = tmp_path / "architecture"
    tokenizer = tmp_path / "tokenizer"
    architecture.mkdir()
    tokenizer.mkdir()
    t5 = tmp_path / "t5.pt"
    vae = tmp_path / "vae.pt"
    t5.write_bytes(b"t5")
    vae.write_bytes(b"vae")
    work_dir = tmp_path / "comparison"

    args = build_parser().parse_args(
        [
            "--merged-checkpoint",
            str(merged),
            "--merged-manifest",
            str(merged_manifest),
            "--base-checkpoint",
            str(base),
            "--training-checkpoint",
            str(training),
            "--metadata",
            str(metadata),
            "--architecture-root",
            str(architecture),
            "--t5-checkpoint",
            str(t5),
            "--tokenizer-dir",
            str(tokenizer),
            "--vae-checkpoint",
            str(vae),
            "--work-dir",
            str(work_dir),
        ]
    )

    monkeypatch.setattr(
        comparison,
        "validate_checkpoint",
        lambda *_args, **_kwargs: {"manifest_sha256": "b" * 64},
    )
    monkeypatch.setattr(
        comparison,
        "validate_merged_checkpoint_provenance",
        lambda **_kwargs: {"path": str(merged), "sha256": "c" * 64},
    )

    def fake_carrier_writer(_image, output, *, frame_count, fps):
        Path(output).write_bytes(f"frames={frame_count},fps={fps}".encode())

    def fake_carrier_probe(path):
        portrait = "portrait_832x480" in str(path)
        return {
            "width": 480 if portrait else 832,
            "height": 832 if portrait else 480,
            "frame_count": 97,
            "fps": 24.0,
        }

    def preparation_builder(**kwargs):
        return prepare_causal_testsets(
            **kwargs,
            carrier_writer=fake_carrier_writer,
            carrier_probe=fake_carrier_probe,
        )

    def fake_parallel_runner(jobs, *, command_runner):
        del command_runner
        reports = []
        for job in jobs:
            config = OmegaConf.load(job.config_path)
            model_type = "lora" if job.variant == "dynamic_lora" else "regular"
            for index in job.sample_indices:
                output = Path(config.output_folder) / f"rank0-{index}-0_{model_type}.mp4"
                output.write_bytes(f"{job.variant}-{job.bucket_id}-{index}".encode())
            reports.append(
                {
                    **comparison.asdict(job),
                    "elapsed_seconds": 1.0,
                    "status": "pass",
                }
            )
        return reports

    monkeypatch.setattr(
        comparison,
        "run_parallel_inference_jobs",
        fake_parallel_runner,
    )

    def fake_output_validator(manifest_path, **_kwargs):
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        samples = []
        for bucket in manifest["buckets"]:
            model_type = bucket["output_model_type"]
            for record in bucket["records"]:
                output = Path(bucket["output_dir"]) / (
                    f"rank0-{record['bucket_index']}-0_{model_type}.mp4"
                )
                assert output.is_file()
                samples.append(
                    {"row_id": record["row_id"], "output_video": str(output)}
                )
        return {"status": "pass", "samples": samples}

    def fake_pair_writer(left, right, output, *, command_runner):
        del left, right, command_runner
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"paired")
        return {"width": 1664, "height": 480, "frame_count": 93, "fps": 24.0}

    report = comparison.run_comparison(
        args,
        output_validator=fake_output_validator,
        pair_writer=fake_pair_writer,
        preparation_builder=preparation_builder,
    )

    assert report["status"] == "pass"
    assert report["reference_checkpoint_dir"] is None
    assert report["input_preparation"]["mode"] == "fresh"
    assert report["input_preparation"]["reference_checkpoint_dir"] is None
    assert Path(report["input_preparation"]["prepared_manifest"]).is_file()
    assert report["sample_count"] == 2
    assert all("reference_video" not in sample for sample in report["samples"])
    html_text = Path(report["comparison_html"]).read_text(encoding="utf-8")
    assert "原 infer_stage1 历史参考" not in html_text


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
    filter_graph = command[command.index("-filter_complex") + 1]
    assert filter_graph.count("setpts=N/(24*TB)") == 2
    assert "hstack=inputs=2:shortest=0" in filter_graph
    assert "fps=fps=24:start_time=0:eof_action=pass" in filter_graph
    assert command[command.index("-r") + 1] == "24"
    assert command[command.index("-fps_mode") + 1] == "cfr"
    assert stream["width"] == 1664
    assert output.read_bytes() == b"paired"
