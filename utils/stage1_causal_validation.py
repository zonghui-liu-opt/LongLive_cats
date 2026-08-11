"""Deterministic validation helpers for Stage-1 causal bases and checkpoints."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

import numpy as np
from PIL import Image

from utils.config import DEFAULT_NEGATIVE_PROMPT
from utils.stage1_io import (
    atomic_output_path,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)


CAUSAL_BASE_FORMAT = "longlive_causal_base_init"
CAUSAL_BASE_VERSION = 1
CAUSAL_TESTSET_SCHEMA_VERSION = 1
CAUSAL_OUTPUT_REPORT_VERSION = 1
REQUIRED_TESTSET_COLUMNS = (
    "input_image",
    "prompt",
    "height",
    "width",
    "bucket",
)


def discover_stage1_training_checkpoints(
    training_root: str | os.PathLike[str],
    *,
    steps: list[int] | tuple[int, ...] | None = None,
    expected_base_sha256: str | None = None,
) -> list[Path]:
    """Return committed Stage-1 adapter checkpoints in optimizer-step order.

    A directory matching ``checkpoint_model_XXXXXX`` is never silently ignored:
    an uncommitted or corrupt directory is an error.  This prevents a batch
    comparison from looking complete while accidentally omitting a training
    checkpoint that failed part-way through publication.
    """

    from utils.stage1_checkpoint import (
        CHECKPOINT_DIRECTORY_PATTERN,
        checkpoint_step,
        validate_checkpoint,
    )

    training_root = Path(training_root).expanduser().resolve()
    if not training_root.is_dir():
        raise FileNotFoundError(training_root)
    requested_steps = None if steps is None else {int(step) for step in steps}
    if requested_steps is not None:
        if not requested_steps:
            raise ValueError("steps must contain at least one optimizer step")
        if any(step <= 0 for step in requested_steps):
            raise ValueError(f"optimizer steps must be positive: {sorted(requested_steps)}")

    discovered: dict[int, Path] = {}
    for path in sorted(training_root.iterdir()):
        if not path.is_dir() or CHECKPOINT_DIRECTORY_PATTERN.fullmatch(path.name) is None:
            continue
        step = checkpoint_step(path)
        if step in discovered:
            raise RuntimeError(
                f"duplicate Stage-1 optimizer step {step}: {discovered[step]} and {path}"
            )
        # Validate every checkpoint-shaped directory, even when --steps would
        # not select it, so incomplete artifacts cannot be silently hidden.
        validate_checkpoint(
            path,
            require_resumable=False,
            expected_base_sha256=expected_base_sha256,
        )
        discovered[step] = path.resolve()

    if not discovered:
        raise RuntimeError(f"No committed checkpoint_model_XXXXXX directories found in {training_root}")
    if requested_steps is not None:
        missing = sorted(requested_steps - set(discovered))
        if missing:
            raise RuntimeError(
                f"Requested Stage-1 optimizer steps are missing from {training_root}: {missing}"
            )
        discovered = {
            step: path for step, path in discovered.items() if step in requested_steps
        }
    return [discovered[step] for step in sorted(discovered)]


@dataclass(frozen=True)
class CausalTestsetRecord:
    row_id: int
    input_image: str
    prompt: str
    height: int
    width: int
    bucket: str
    bucket_id: str
    image_sha256: str
    row_sha256: str


def _canonical_dtype(dtype: Any) -> str:
    return str(dtype).removeprefix("torch.")


def _expected_bucket(height: int, width: int) -> str:
    if width > height:
        return "landscape"
    if height > width:
        return "portrait"
    raise ValueError("Square inputs are not supported by the Stage-1 portrait/landscape contract.")


def _bucket_id(bucket: str, height: int, width: int) -> str:
    return f"{bucket}_{height}x{width}"


def load_causal_testset_records(
    metadata_path: str | os.PathLike[str],
    *,
    allow_repeated_input_images: bool = False,
) -> list[CausalTestsetRecord]:
    """Load the image/prompt test CSV without silently changing its geometry."""

    metadata_path = Path(metadata_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [name for name in REQUIRED_TESTSET_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"Testset metadata is missing required columns {missing}: {metadata_path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Testset metadata is empty: {metadata_path}")

    records: list[CausalTestsetRecord] = []
    seen_images: set[Path] = set()
    for row_id, raw_row in enumerate(rows):
        row = {str(key): "" if value is None else str(value) for key, value in raw_row.items()}
        prompt = row["prompt"].strip()
        if not prompt:
            raise ValueError(f"row {row_id}: prompt must be non-empty")
        try:
            height = int(row["height"])
            width = int(row["width"])
        except ValueError as exc:
            raise ValueError(f"row {row_id}: height/width must be integers") from exc
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError(
                f"row {row_id}: dimensions must be positive and divisible by 16, got {height}x{width}"
            )
        bucket = row["bucket"].strip().lower()
        expected_bucket = _expected_bucket(height, width)
        if bucket != expected_bucket:
            raise ValueError(
                f"row {row_id}: bucket {bucket!r} does not match {height}x{width} ({expected_bucket})"
            )

        image_path = Path(row["input_image"].strip()).expanduser()
        if not image_path.is_absolute():
            image_path = metadata_path.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"row {row_id}: missing input image: {image_path}")
        if image_path in seen_images and not allow_repeated_input_images:
            raise ValueError(f"row {row_id}: duplicate input image: {image_path}")
        seen_images.add(image_path)

        with Image.open(image_path) as image:
            if image.mode != "RGB":
                raise ValueError(f"row {row_id}: input image must be RGB, got {image.mode}: {image_path}")
            if image.size != (width, height):
                raise ValueError(
                    f"row {row_id}: image size {image.size} does not match CSV {(width, height)}"
                )
            orientation = image.getexif().get(274)
            if orientation not in (None, 1):
                raise ValueError(
                    f"row {row_id}: EXIF orientation must be absent/1, got {orientation}: {image_path}"
                )

        image_sha256 = sha256_file(image_path)
        canonical_row = {key: row[key] for key in sorted(row)}
        row_sha256 = canonical_json_sha256(
            {
                "row_id": row_id,
                "canonical_row": canonical_row,
                "resolved_input_image": os.fspath(image_path),
                "image_sha256": image_sha256,
            }
        )
        records.append(
            CausalTestsetRecord(
                row_id=row_id,
                input_image=os.fspath(image_path),
                prompt=prompt,
                height=height,
                width=width,
                bucket=bucket,
                bucket_id=_bucket_id(bucket, height, width),
                image_sha256=image_sha256,
                row_sha256=row_sha256,
            )
        )
    return records


def _run_checked(command: list[str]) -> None:
    subprocess.run(command, check=True)


def probe_video(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Return stable video-stream metadata through ffprobe."""

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise RuntimeError("ffprobe is required for Stage-1 causal video validation.")
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,nb_read_frames,nb_frames,avg_frame_rate",
        "-of",
        "json",
        os.fspath(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"Expected exactly one video stream in {path}, got {len(streams)}")
    stream = streams[0]
    frame_count_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frame_count_value in (None, "N/A"):
        raise RuntimeError(f"ffprobe did not report a frame count for {path}")
    rate = str(stream.get("avg_frame_rate", "0/0"))
    numerator, denominator = rate.split("/", 1)
    fps = float(numerator) / float(denominator) if float(denominator) else 0.0
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frame_count": int(frame_count_value),
        "fps": fps,
    }


def write_static_carrier_video(
    image_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    frame_count: int,
    fps: int,
    command_runner: Callable[[list[str]], None] = _run_checked,
) -> None:
    """Encode a first-frame carrier without resizing or cropping the input image."""

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to prepare causal I2V testsets.")
    output_path = Path(output_path)
    with atomic_output_path(output_path, suffix=".mp4") as temporary:
        command_runner(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-loop",
                "1",
                "-framerate",
                str(fps),
                "-i",
                os.fspath(image_path),
                "-frames:v",
                str(frame_count),
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "1",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-f",
                "mp4",
                os.fspath(temporary),
            ]
        )


def _write_inference_config(path: Path, values: dict[str, Any]) -> None:
    from omegaconf import OmegaConf

    text = OmegaConf.to_yaml(OmegaConf.create(values), resolve=False, sort_keys=False)
    atomic_write_bytes(path, text.encode("utf-8"))


def resolve_stage1_model_paths(
    *,
    base_checkpoint: str | os.PathLike[str],
    architecture_root: str | os.PathLike[str],
    t5_checkpoint: str | os.PathLike[str],
    tokenizer_dir: str | os.PathLike[str],
    vae_checkpoint: str | os.PathLike[str],
) -> dict[str, Path]:
    """Resolve and validate the model assets shared by Stage-1 runners."""

    paths = {
        "base_checkpoint": Path(base_checkpoint).expanduser().resolve(),
        "architecture_root": Path(architecture_root).expanduser().resolve(),
        "t5_checkpoint": Path(t5_checkpoint).expanduser().resolve(),
        "tokenizer_dir": Path(tokenizer_dir).expanduser().resolve(),
        "vae_checkpoint": Path(vae_checkpoint).expanduser().resolve(),
    }
    for name, path in paths.items():
        expected = (
            path.is_dir()
            if name in {"architecture_root", "tokenizer_dir"}
            else path.is_file()
        )
        if not expected:
            raise FileNotFoundError(f"Missing {name}: {path}")
    return paths


def prepare_causal_testsets(
    *,
    metadata_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    base_checkpoint: str | os.PathLike[str],
    architecture_root: str | os.PathLike[str],
    t5_checkpoint: str | os.PathLike[str],
    tokenizer_dir: str | os.PathLike[str],
    vae_checkpoint: str | os.PathLike[str],
    allow_repeated_input_images: bool = False,
    num_latent_frames: int = 24,
    num_frame_per_block: int = 8,
    temporal_compression_ratio: int = 4,
    minimum_source_frames: int = 97,
    fps: int = 24,
    sampling_steps: int = 50,
    guidance_scale: float = 5.0,
    seed: int = 1,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    carrier_writer: Callable[..., None] = write_static_carrier_video,
    carrier_probe: Callable[[str | os.PathLike[str]], dict[str, Any]] = probe_video,
) -> dict[str, Any]:
    """Create deterministic portrait/landscape datasets and inference YAMLs."""

    if num_latent_frames <= 0 or num_latent_frames % num_frame_per_block:
        raise ValueError("num_latent_frames must be positive and divisible by num_frame_per_block")
    if temporal_compression_ratio <= 0 or fps <= 0 or sampling_steps <= 0:
        raise ValueError("temporal compression, fps, and sampling_steps must be positive")

    metadata_path = Path(metadata_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    manifest_path = output_root / "prepared_manifest.json"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Preparation output must be empty to prevent stale inference samples: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    paths = resolve_stage1_model_paths(
        base_checkpoint=base_checkpoint,
        architecture_root=architecture_root,
        t5_checkpoint=t5_checkpoint,
        tokenizer_dir=tokenizer_dir,
        vae_checkpoint=vae_checkpoint,
    )

    records = load_causal_testset_records(
        metadata_path,
        allow_repeated_input_images=allow_repeated_input_images,
    )
    grouped: dict[str, list[CausalTestsetRecord]] = defaultdict(list)
    for record in records:
        grouped[record.bucket_id].append(record)

    pixel_frames = 1 + (num_latent_frames - 1) * temporal_compression_ratio
    carrier_frames = max(minimum_source_frames, pixel_frames)
    bucket_entries = []
    for bucket_id in sorted(grouped):
        bucket_records = grouped[bucket_id]
        height = bucket_records[0].height
        width = bucket_records[0].width
        data_root = output_root / "datasets" / bucket_id
        output_dir = output_root / "videos" / bucket_id
        config_path = output_root / "configs" / f"{bucket_id}.yaml"
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        prepared_records = []
        for bucket_index, record in enumerate(bucket_records):
            sample_name = f"{bucket_index:04d}_row{record.row_id:04d}"
            video_dir = data_root / "video" / sample_name
            caption_dir = data_root / "caption" / sample_name
            video_dir.mkdir(parents=True, exist_ok=True)
            caption_dir.mkdir(parents=True, exist_ok=True)
            carrier_path = video_dir / "000.mp4"
            caption_path = caption_dir / "000.json"
            carrier_writer(
                record.input_image,
                carrier_path,
                frame_count=carrier_frames,
                fps=fps,
            )
            carrier_stream = carrier_probe(carrier_path)
            if (
                int(carrier_stream["width"]) != width
                or int(carrier_stream["height"]) != height
                or int(carrier_stream["frame_count"]) != carrier_frames
                or abs(float(carrier_stream["fps"]) - fps) > 0.01
            ):
                raise RuntimeError(
                    f"carrier video contract mismatch for row {record.row_id}: {carrier_stream}"
                )
            atomic_write_json(caption_path, {"caption": record.prompt})
            prepared_records.append(
                {
                    **asdict(record),
                    "bucket_index": bucket_index,
                    "sample_name": sample_name,
                    "carrier_video": os.fspath(carrier_path),
                    "carrier_sha256": sha256_file(carrier_path),
                    "carrier_stream": carrier_stream,
                    "caption_json": os.fspath(caption_path),
                }
            )

        config = {
            "model_kwargs": {
                "model_name": "Wan2.2-TI2V-5B",
                "timestep_shift": 5.0,
                "num_frame_per_block": num_frame_per_block,
                "architecture_root": os.fspath(paths["architecture_root"]),
                "init_weights": False,
                "local_attn_size": -1,
                "sink_size": 0,
            },
            "model_paths": {
                "architecture_root": os.fspath(paths["architecture_root"]),
                "t5_checkpoint": os.fspath(paths["t5_checkpoint"]),
                "tokenizer_dir": os.fspath(paths["tokenizer_dir"]),
                "vae_checkpoint": os.fspath(paths["vae_checkpoint"]),
            },
            "i2v": True,
            "use_ema": False,
            "model_quant": False,
            "fp8_quant": False,
            "torch_compile": False,
            "output_folder": os.fspath(output_dir),
            "num_samples": 1,
            "num_output_frames": num_latent_frames,
            "save_latents_only": False,
            "save_with_index": True,
            "allow_padding": False,
            "min_latent_frames": 0,
            "max_chunks_per_shot": 0,
            "uniform_prompt": True,
            "data": {
                "data_path": os.fspath(data_root),
                "image_or_video_shape": [
                    1,
                    num_latent_frames,
                    48,
                    height // 16,
                    width // 16,
                ],
            },
            "inference": {
                "sampling_steps": sampling_steps,
                "independent_first_frame": True,
                "sink_size": 0,
                "local_attn_size": -1,
                "guidance_scale": guidance_scale,
                "negative_prompt": negative_prompt,
                "multi_shot_sink": False,
                "streaming_vae": False,
                "async_vae": False,
                "vae_type": "wan",
            },
            "checkpoints": {"generator_ckpt": os.fspath(paths["base_checkpoint"])},
            "logging": {"seed": seed},
        }
        _write_inference_config(config_path, config)
        bucket_entries.append(
            {
                "bucket_id": bucket_id,
                "bucket": bucket_records[0].bucket,
                "height": height,
                "width": width,
                "latent_height": height // 16,
                "latent_width": width // 16,
                "data_root": os.fspath(data_root),
                "config_path": os.fspath(config_path),
                "output_dir": os.fspath(output_dir),
                "records": prepared_records,
            }
        )

    manifest = {
        "schema_version": CAUSAL_TESTSET_SCHEMA_VERSION,
        "metadata": {
            "path": os.fspath(metadata_path),
            "sha256": sha256_file(metadata_path),
            "record_count": len(records),
            "aggregate_row_sha256": canonical_json_sha256(
                [record.row_sha256 for record in records]
            ),
        },
        "model_paths": {name: os.fspath(path) for name, path in paths.items()},
        "frame_policy": {
            "num_latent_frames": num_latent_frames,
            "num_frame_per_block": num_frame_per_block,
            "temporal_compression_ratio": temporal_compression_ratio,
            "expected_pixel_frames": pixel_frames,
            "carrier_frames": carrier_frames,
            "fps": fps,
        },
        "sampling": {
            "solver": "unipc",
            "sampling_steps": sampling_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "negative_prompt": negative_prompt,
        },
        "buckets": bucket_entries,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(manifest_path, manifest)
    return manifest


def validate_converted_causal_base(
    base_checkpoint: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    source_checkpoint: str | os.PathLike[str] | None = None,
    expected_num_frame_per_block: int = 8,
    check_finite: bool = True,
) -> dict[str, Any]:
    """Independently audit the immutable converted checkpoint and its provenance."""

    import torch

    base_checkpoint = Path(base_checkpoint).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("checkpoint_format") != CAUSAL_BASE_FORMAT:
        raise RuntimeError("base manifest checkpoint_format is not causal init")
    if int(manifest.get("checkpoint_version", -1)) != CAUSAL_BASE_VERSION:
        raise RuntimeError("unsupported converted causal base version")
    if manifest.get("strict_reload") is not True:
        raise RuntimeError("converter manifest does not record a successful fresh strict reload")

    output_meta = manifest.get("output", {})
    actual_output_sha256 = sha256_file(base_checkpoint)
    if output_meta.get("sha256") != actual_output_sha256:
        raise RuntimeError("converted base SHA256 differs from its manifest")
    if int(output_meta.get("size", -1)) != base_checkpoint.stat().st_size:
        raise RuntimeError("converted base size differs from its manifest")

    payload = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("checkpoint_format") != CAUSAL_BASE_FORMAT:
        raise RuntimeError("checkpoint payload format is not causal init")
    if int(payload.get("checkpoint_version", -1)) != CAUSAL_BASE_VERSION:
        raise RuntimeError("checkpoint payload version is unsupported")
    if payload.get("source_aggregate_sha256") != manifest.get("source", {}).get("aggregate_sha256"):
        raise RuntimeError("payload and manifest source aggregate hashes differ")
    generator = payload.get("generator")
    if not isinstance(generator, dict) or not generator:
        raise RuntimeError("converted checkpoint has no non-empty generator state")

    expected_tensors = manifest.get("generator_state", {}).get("tensors", [])
    expected_by_key = {item["key"]: item for item in expected_tensors}
    if len(expected_by_key) != len(expected_tensors):
        raise RuntimeError("manifest generator tensor keys are not unique")
    if set(generator) != set(expected_by_key):
        missing = sorted(set(expected_by_key) - set(generator))
        unexpected = sorted(set(generator) - set(expected_by_key))
        raise RuntimeError(
            f"generator keys differ from manifest: missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    dtype_counts: Counter[str] = Counter()
    total_numel = 0
    for key, tensor in generator.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"generator state {key!r} is not a tensor")
        expected = expected_by_key[key]
        if list(tensor.shape) != list(expected["shape"]):
            raise RuntimeError(f"shape mismatch for {key}: {list(tensor.shape)} vs {expected['shape']}")
        dtype_name = _canonical_dtype(tensor.dtype)
        if dtype_name != expected["dtype"]:
            raise RuntimeError(f"dtype mismatch for {key}: {dtype_name} vs {expected['dtype']}")
        if tensor.numel() != int(expected["numel"]):
            raise RuntimeError(f"numel mismatch for {key}")
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16:
            raise RuntimeError(f"converted floating tensor is not BF16: {key} ({tensor.dtype})")
        if check_finite and tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise RuntimeError(f"converted tensor contains non-finite values: {key}")
        dtype_counts[dtype_name] += 1
        total_numel += tensor.numel()

    summary = manifest.get("generator_state", {})
    if int(summary.get("tensor_count", -1)) != len(generator):
        raise RuntimeError("generator tensor_count differs from manifest")
    if int(summary.get("total_numel", -1)) != total_numel:
        raise RuntimeError("generator total_numel differs from manifest")
    if dict(sorted(dtype_counts.items())) != summary.get("dtype_counts"):
        raise RuntimeError("generator dtype_counts differ from manifest")

    manifest_causal_config = manifest.get("causal_config") or None
    payload_causal_config = payload.get("causal_config") or None
    if (
        manifest_causal_config is not None
        and payload_causal_config is not None
        and manifest_causal_config != payload_causal_config
    ):
        raise RuntimeError("payload and manifest causal configs differ")
    causal_config = manifest_causal_config or payload_causal_config
    if causal_config is not None:
        if int(causal_config.get("num_frame_per_block", -1)) != expected_num_frame_per_block:
            raise RuntimeError(
                "converted base num_frame_per_block mismatch: "
                f"expected {expected_num_frame_per_block}, got {causal_config.get('num_frame_per_block')}"
            )

    source_report: dict[str, Any] = {"checked": False}
    if source_checkpoint is not None:
        from scripts.convert_diffsynth_wan22_to_longlive import _hash_source_files

        entries, aggregate = _hash_source_files(source_checkpoint)
        if entries != manifest.get("source", {}).get("files"):
            raise RuntimeError("current source checkpoint files differ from the conversion manifest")
        if aggregate != manifest.get("source", {}).get("aggregate_sha256"):
            raise RuntimeError("current source checkpoint aggregate hash differs from the manifest")
        source_report = {
            "checked": True,
            "file_count": len(entries),
            "aggregate_sha256": aggregate,
        }

    coverage = manifest.get("coverage", {})
    if coverage:
        if int(coverage.get("expected_keys", -1)) != int(
            manifest.get("source_state", {}).get("tensor_count", -2)
        ):
            raise RuntimeError("coverage expected_keys differs from source tensor_count")
        if int(coverage.get("loaded_keys", -1)) != len(generator):
            raise RuntimeError("coverage loaded_keys differs from generator tensor_count")
        if float(coverage.get("key_percent", -1.0)) != 100.0:
            raise RuntimeError("converter did not record 100% key coverage")
        if float(coverage.get("shape_percent", -1.0)) != 100.0:
            raise RuntimeError("converter did not record 100% shape coverage")

    del payload, generator
    return {
        "status": "pass",
        "base_checkpoint": os.fspath(base_checkpoint),
        "manifest_path": os.fspath(manifest_path),
        "output_sha256": actual_output_sha256,
        "tensor_count": int(summary["tensor_count"]),
        "total_numel": int(summary["total_numel"]),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "strict_reload": True,
        "causal_config": causal_config,
        "source": source_report,
    }


def _video_pixel_metrics(video_path: Path, source_image_path: Path) -> dict[str, float]:
    import cv2

    with Image.open(source_image_path) as image:
        source = np.asarray(image.convert("RGB"), dtype=np.float32)
    capture = cv2.VideoCapture(os.fspath(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not decode {video_path}")
    first = None
    previous = None
    frame_count = 0
    frame_stds: list[float] = []
    temporal_diffs: list[float] = []
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        if first is None:
            first = rgb
        if previous is not None:
            temporal_diffs.append(float(np.mean(np.abs(rgb - previous))))
        frame_stds.append(float(rgb.std()))
        previous = rgb
        frame_count += 1
    capture.release()
    if first is None or frame_count == 0:
        raise RuntimeError(f"Decoded no frames from {video_path}")
    if first.shape != source.shape:
        raise RuntimeError(
            f"decoded first frame shape {first.shape} differs from source {source.shape}: {video_path}"
        )
    mse = float(np.mean((first - source) ** 2))
    psnr = math.inf if mse == 0.0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return {
        "decoded_frame_count": float(frame_count),
        "first_frame_psnr_db": psnr,
        "mean_frame_std": float(np.mean(frame_stds)),
        "mean_temporal_abs_diff": float(np.mean(temporal_diffs)) if temporal_diffs else 0.0,
    }


def validate_causal_video_output(
    video_path: str | os.PathLike[str],
    source_image_path: str | os.PathLike[str],
    *,
    row_id: int,
    expected_width: int,
    expected_height: int,
    expected_frames: int,
    expected_fps: float,
    minimum_first_frame_psnr_db: float = 12.0,
    minimum_frame_std: float = 5.0,
    minimum_temporal_abs_diff: float = 0.05,
    probe: Callable[[str | os.PathLike[str]], dict[str, Any]] = probe_video,
    pixel_metrics: Callable[[Path, Path], dict[str, float]] = _video_pixel_metrics,
) -> dict[str, Any]:
    """Apply the shared Stage-1 technical gate to one explicitly mapped MP4."""

    video_path = Path(video_path)
    source_image_path = Path(source_image_path)
    stream = probe(video_path)
    expected_geometry = (int(expected_width), int(expected_height))
    actual_geometry = (int(stream["width"]), int(stream["height"]))
    if actual_geometry != expected_geometry:
        raise RuntimeError(
            f"output geometry mismatch for row {row_id}: "
            f"expected {expected_geometry}, got {actual_geometry}"
        )
    if int(stream["frame_count"]) != int(expected_frames):
        raise RuntimeError(
            f"output frame count mismatch for row {row_id}: "
            f"expected {expected_frames}, got {stream['frame_count']}"
        )
    if abs(float(stream["fps"]) - float(expected_fps)) > 0.01:
        raise RuntimeError(
            f"output FPS mismatch for row {row_id}: "
            f"expected {expected_fps}, got {stream['fps']}"
        )
    metrics = pixel_metrics(video_path, source_image_path)
    if int(metrics["decoded_frame_count"]) != int(expected_frames):
        raise RuntimeError(f"decoder did not read every frame from {video_path}")
    if metrics["first_frame_psnr_db"] < minimum_first_frame_psnr_db:
        raise RuntimeError(
            f"first-frame reconstruction PSNR is too low for row {row_id}: "
            f"{metrics['first_frame_psnr_db']:.3f} dB"
        )
    if metrics["mean_frame_std"] < minimum_frame_std:
        raise RuntimeError(
            f"output is nearly flat for row {row_id}: "
            f"std={metrics['mean_frame_std']:.3f}"
        )
    if metrics["mean_temporal_abs_diff"] < minimum_temporal_abs_diff:
        raise RuntimeError(
            f"output is temporally frozen for row {row_id}: "
            f"mean_abs_diff={metrics['mean_temporal_abs_diff']:.6f}"
        )
    return {"stream": stream, "metrics": metrics}


def validate_causal_testset_outputs(
    prepared_manifest_path: str | os.PathLike[str],
    *,
    minimum_first_frame_psnr_db: float = 12.0,
    minimum_frame_std: float = 5.0,
    minimum_temporal_abs_diff: float = 0.05,
    probe: Callable[[str | os.PathLike[str]], dict[str, Any]] = probe_video,
    pixel_metrics: Callable[[Path, Path], dict[str, float]] = _video_pixel_metrics,
) -> dict[str, Any]:
    """Validate every expected MP4 and reject stale or technically broken output."""

    prepared_manifest_path = Path(prepared_manifest_path).expanduser().resolve()
    manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != CAUSAL_TESTSET_SCHEMA_VERSION:
        raise RuntimeError("unsupported prepared testset manifest version")
    frame_policy = manifest["frame_policy"]
    expected_frames = int(frame_policy["expected_pixel_frames"])
    expected_fps = float(frame_policy["fps"])

    samples = []
    for bucket in manifest["buckets"]:
        output_dir = Path(bucket["output_dir"])
        output_model_type = str(bucket.get("output_model_type", "regular"))
        if output_model_type not in {"regular", "lora", "ema"}:
            raise RuntimeError(
                f"unsupported output_model_type for {bucket['bucket_id']}: "
                f"{output_model_type!r}"
            )
        expected_names = {
            f"rank0-{record['bucket_index']}-0_{output_model_type}.mp4"
            for record in bucket["records"]
        }
        actual_names = {path.name for path in output_dir.glob("*.mp4")}
        if actual_names != expected_names:
            raise RuntimeError(
                f"output set mismatch for {bucket['bucket_id']}: "
                f"missing={sorted(expected_names - actual_names)}, "
                f"unexpected={sorted(actual_names - expected_names)}"
            )
        for record in bucket["records"]:
            video_path = output_dir / (
                f"rank0-{record['bucket_index']}-0_{output_model_type}.mp4"
            )
            gate = validate_causal_video_output(
                video_path,
                record["input_image"],
                row_id=int(record["row_id"]),
                expected_width=int(bucket["width"]),
                expected_height=int(bucket["height"]),
                expected_frames=expected_frames,
                expected_fps=expected_fps,
                minimum_first_frame_psnr_db=minimum_first_frame_psnr_db,
                minimum_frame_std=minimum_frame_std,
                minimum_temporal_abs_diff=minimum_temporal_abs_diff,
                probe=probe,
                pixel_metrics=pixel_metrics,
            )
            stream = gate["stream"]
            metrics = gate["metrics"]
            samples.append(
                {
                    "row_id": int(record["row_id"]),
                    "bucket_id": bucket["bucket_id"],
                    "output_video": os.fspath(video_path),
                    "output_sha256": sha256_file(video_path),
                    "stream": stream,
                    "metrics": metrics,
                }
            )

    return {
        "schema_version": CAUSAL_OUTPUT_REPORT_VERSION,
        "status": "pass",
        "prepared_manifest": os.fspath(prepared_manifest_path),
        "sample_count": len(samples),
        "thresholds": {
            "minimum_first_frame_psnr_db": minimum_first_frame_psnr_db,
            "minimum_frame_std": minimum_frame_std,
            "minimum_temporal_abs_diff": minimum_temporal_abs_diff,
        },
        "samples": sorted(samples, key=lambda item: item["row_id"]),
    }
