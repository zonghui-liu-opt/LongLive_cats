"""Strict CSV/media parsing and deterministic cache dataset for Stage-1 I2V."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from utils.stage1_io import (
    aggregate_file_hash,
    atomic_output_path,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
    tree_file_hashes,
)
from utils.stage1_i2v_schema import (
    STAGE1_CACHE_SCHEMA_VERSION,
    STAGE1_REQUIRED_COLUMNS,
)


STAGE1_CACHE_MANIFEST_NAME = "cache_manifest.json"
EXPECTED_CACHE_DTYPES = {
    "video_latent": torch.bfloat16,
    "initial_latent": torch.bfloat16,
    "prompt_embeds": torch.bfloat16,
    "prompt_mask": torch.bool,
}


def _canonical_row(fieldnames: Sequence[str], row: Mapping[str, str]) -> dict[str, str]:
    # Keep unknown columns in original header order while producing a stable
    # JSON representation.  Duplicate CSV headers are rejected by the loader.
    return {name: str(row.get(name, "")) for name in fieldnames}


def _resolved_path(csv_root: Path, value: str, *, label: str, row_number: int) -> Path:
    value = value.strip()
    if not value:
        raise ValueError(f"CSV row {row_number}: {label} must be non-empty.")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = csv_root / path
    return path.resolve()


def _strict_positive_dimension(value: str, *, label: str, row_number: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"CSV row {row_number}: {label} must be an integer.") from exc
    if result <= 0 or str(result) != str(value).strip():
        raise ValueError(
            f"CSV row {row_number}: {label} must be a canonical positive integer, "
            f"got {value!r}."
        )
    return result


def orientation_for_shape(height: int, width: int) -> str:
    if height == width:
        raise ValueError("Square media is not part of the Stage-1 orientation buckets.")
    return "landscape" if width > height else "portrait"


@dataclass(frozen=True)
class Stage1I2VRecord:
    row_id: int
    video_path: Path
    prompt: str
    input_image_path: Path
    height: int
    width: int
    bucket: str
    canonical_row: Mapping[str, str] = field(repr=False)
    row_sha256: str

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.height, self.width

    def source_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "canonical_row": dict(self.canonical_row),
            "row_sha256": self.row_sha256,
            "video_path": str(self.video_path),
            "input_image_path": str(self.input_image_path),
        }


def validate_input_image(record: Stage1I2VRecord) -> None:
    from PIL import Image

    with Image.open(record.input_image_path) as image:
        orientation = image.getexif().get(274)
        if orientation not in (None, 1):
            raise ValueError(
                f"row {record.row_id}: input image EXIF orientation must be 1 or "
                f"absent, got {orientation}."
            )
        if image.mode != "RGB":
            raise ValueError(
                f"row {record.row_id}: input image must be RGB without alpha/palette, "
                f"got mode={image.mode!r}."
            )
        if (image.height, image.width) != record.spatial_shape:
            raise ValueError(
                f"row {record.row_id}: input image shape {(image.height, image.width)} "
                f"does not match CSV {record.spatial_shape}."
            )


def load_stage1_i2v_manifest(
    metadata_path: str | os.PathLike[str],
    *,
    expected_num_samples: int | None = None,
    require_files: bool = True,
    validate_images: bool = True,
) -> list[Stage1I2VRecord]:
    """Load a deterministic CSV manifest, failing on every semantic mismatch."""
    metadata_path = Path(metadata_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    records: list[Stage1I2VRecord] = []
    seen_videos: dict[Path, int] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {metadata_path}")
        fieldnames = [str(name) for name in reader.fieldnames]
        if len(fieldnames) != len(set(fieldnames)):
            duplicates = sorted(name for name, count in Counter(fieldnames).items() if count > 1)
            raise ValueError(f"CSV has duplicate columns: {duplicates}")
        missing = [name for name in STAGE1_REQUIRED_COLUMNS if name not in fieldnames]
        if missing:
            raise ValueError(f"CSV is missing required columns: {missing}")

        for row_id, row in enumerate(reader):
            row_number = row_id + 2
            if None in row:
                raise ValueError(f"CSV row {row_number} contains extra unheaded values.")
            canonical = _canonical_row(fieldnames, row)
            prompt = canonical["prompt"].strip()
            if not prompt:
                raise ValueError(f"CSV row {row_number}: prompt must be non-empty.")
            height = _strict_positive_dimension(
                canonical["height"], label="height", row_number=row_number
            )
            width = _strict_positive_dimension(
                canonical["width"], label="width", row_number=row_number
            )
            bucket = canonical["bucket"].strip().lower()
            expected_bucket = orientation_for_shape(height, width)
            if bucket != expected_bucket:
                raise ValueError(
                    f"CSV row {row_number}: bucket={bucket!r} does not match "
                    f"shape {(height, width)} ({expected_bucket!r})."
                )
            video_path = _resolved_path(
                metadata_path.parent, canonical["video"], label="video", row_number=row_number
            )
            image_path = _resolved_path(
                metadata_path.parent,
                canonical["input_image"],
                label="input_image",
                row_number=row_number,
            )
            if video_path in seen_videos:
                raise ValueError(
                    f"CSV row {row_number}: duplicate video also used by row "
                    f"{seen_videos[video_path] + 2}: {video_path}"
                )
            seen_videos[video_path] = row_id
            if require_files:
                if not video_path.is_file():
                    raise FileNotFoundError(f"CSV row {row_number}: video not found: {video_path}")
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"CSV row {row_number}: input image not found: {image_path}"
                    )
            record = Stage1I2VRecord(
                row_id=row_id,
                video_path=video_path,
                prompt=prompt,
                input_image_path=image_path,
                height=height,
                width=width,
                bucket=bucket,
                canonical_row=canonical,
                row_sha256=canonical_json_sha256(canonical),
            )
            if require_files and validate_images:
                validate_input_image(record)
            records.append(record)

    if expected_num_samples is not None and len(records) != int(expected_num_samples):
        raise ValueError(
            f"Expected exactly {int(expected_num_samples)} samples, found {len(records)} "
            f"in {metadata_path}."
        )
    if not records:
        raise ValueError(f"CSV contains no samples: {metadata_path}")
    return records


def _rotation_metadata(stream: Any) -> float:
    values: list[float] = []
    raw_rotate = getattr(stream, "metadata", {}).get("rotate")
    if raw_rotate not in (None, "", 0, "0"):
        try:
            values.append(float(raw_rotate))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unparseable video rotate metadata: {raw_rotate!r}") from exc
    rotation = getattr(stream, "rotation", None)
    if rotation not in (None, 0, 0.0):
        values.append(float(rotation))
    return max((abs(value) for value in values), default=0.0)


def decode_stage1_video(
    record: Stage1I2VRecord,
    *,
    min_source_frames: int = 97,
    selected_frame_start: int = 0,
    selected_frame_count: int = 93,
    expected_fps: float = 24.0,
    fps_abs_tolerance: float = 1e-3,
) -> torch.Tensor:
    """Sequentially decode RGB presentation frames without seek/resample/resize."""
    if min_source_frames < selected_frame_start + selected_frame_count:
        raise ValueError("min_source_frames must cover the selected frame window.")
    try:
        import av
    except ImportError as exc:  # pragma: no cover - depends on deployment image
        raise RuntimeError("PyAV is required for Stage-1 video precomputation.") from exc

    decoded: list[torch.Tensor] = []
    with av.open(str(record.video_path), mode="r") as container:
        streams = list(container.streams.video)
        if len(streams) != 1:
            raise ValueError(
                f"row {record.row_id}: expected exactly one video stream, found {len(streams)}."
            )
        stream = streams[0]
        rotation = _rotation_metadata(stream)
        if not math.isclose(rotation, 0.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"row {record.row_id}: video rotation metadata must be zero/absent, "
                f"got {rotation}."
            )
        rate = getattr(stream, "average_rate", None)
        if rate is None:
            raise ValueError(f"row {record.row_id}: video has no average frame rate metadata.")
        fps = float(rate)
        if not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=fps_abs_tolerance):
            raise ValueError(
                f"row {record.row_id}: expected {expected_fps} fps, got {fps}."
            )

        decoded_count = 0
        for frame in container.decode(stream):
            if (int(frame.height), int(frame.width)) != record.spatial_shape:
                raise ValueError(
                    f"row {record.row_id}: decoded frame {decoded_count} shape "
                    f"{(frame.height, frame.width)} does not match CSV "
                    f"{record.spatial_shape}."
                )
            if selected_frame_start <= decoded_count < selected_frame_start + selected_frame_count:
                rgb = frame.to_ndarray(format="rgb24")
                if rgb.dtype.name != "uint8" or rgb.shape != (record.height, record.width, 3):
                    raise ValueError(
                        f"row {record.row_id}: unexpected RGB frame {decoded_count} "
                        f"shape/dtype {rgb.shape}/{rgb.dtype}."
                    )
                decoded.append(torch.from_numpy(rgb.copy()))
            decoded_count += 1
            # Decode far enough to prove the minimum source length, but never
            # seek or decode the tail that training explicitly ignores.
            if decoded_count >= min_source_frames:
                break

    if decoded_count < min_source_frames:
        raise ValueError(
            f"row {record.row_id}: source has only {decoded_count} decoded frames; "
            f"at least {min_source_frames} are required."
        )
    if len(decoded) != selected_frame_count:
        raise RuntimeError(
            f"row {record.row_id}: selected {len(decoded)} frames, expected "
            f"{selected_frame_count}."
        )
    # [T,H,W,C] uint8 -> [1,C,T,H,W] float32 in [-1,1]
    stacked = torch.stack(decoded, dim=0)
    return rgb_uint8_to_vae_input(stacked).permute(3, 0, 1, 2).unsqueeze(0)


def load_stage1_input_image(record: Stage1I2VRecord) -> torch.Tensor:
    from PIL import Image
    import numpy as np

    validate_input_image(record)
    with Image.open(record.input_image_path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.uint8).copy()
    tensor = rgb_uint8_to_vae_input(torch.from_numpy(array))
    return tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)


def rgb_uint8_to_vae_input(value: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.uint8 or value.shape[-1] != 3:
        raise ValueError(
            f"Expected RGB uint8 tensor with channel-last shape, got "
            f"{tuple(value.shape)} {value.dtype}."
        )
    return value.to(torch.float32).div_(255.0).sub_(0.5).div_(0.5)


def validate_cache_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    expected_spatial_shapes: Iterable[tuple[int, int]] = ((30, 52), (52, 30)),
) -> None:
    expected_keys = set(EXPECTED_CACHE_DTYPES)
    if set(tensors) != expected_keys:
        raise ValueError(
            f"Cache tensor keys mismatch: expected {sorted(expected_keys)}, "
            f"got {sorted(tensors)}."
        )
    for name, dtype in EXPECTED_CACHE_DTYPES.items():
        if tensors[name].dtype != dtype:
            raise ValueError(
                f"Cache tensor {name} must have dtype {dtype}, got {tensors[name].dtype}."
            )
        if not tensors[name].isfinite().all():
            raise ValueError(f"Cache tensor {name} contains non-finite values.")
    video = tensors["video_latent"]
    initial = tensors["initial_latent"]
    prompt = tensors["prompt_embeds"]
    mask = tensors["prompt_mask"]
    allowed = {tuple(map(int, shape)) for shape in expected_spatial_shapes}
    if video.ndim != 4 or tuple(video.shape[:2]) != (24, 48):
        raise ValueError(f"video_latent must be [24,48,H,W], got {tuple(video.shape)}.")
    if tuple(video.shape[-2:]) not in allowed:
        raise ValueError(
            f"video_latent spatial shape {tuple(video.shape[-2:])} not in {sorted(allowed)}."
        )
    if tuple(initial.shape) != (1, 48, *video.shape[-2:]):
        raise ValueError(
            f"initial_latent must be [1,48,H,W] matching video, got {tuple(initial.shape)}."
        )
    if tuple(prompt.shape) != (512, 4096):
        raise ValueError(f"prompt_embeds must be [512,4096], got {tuple(prompt.shape)}.")
    if tuple(mask.shape) != (512,):
        raise ValueError(f"prompt_mask must be [512], got {tuple(mask.shape)}.")


def save_cache_artifact(
    path: str | os.PathLike[str],
    tensors: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    from safetensors.torch import save_file

    canonical = {name: tensor.detach().contiguous().cpu() for name, tensor in tensors.items()}
    validate_cache_tensors(canonical)
    text_metadata = {str(key): str(value) for key, value in metadata.items()}
    with atomic_output_path(path, suffix=".safetensors.tmp") as temporary:
        save_file(canonical, str(temporary), metadata=text_metadata)
    path = Path(path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "tensors": {
            name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype).replace("torch.", "")}
            for name, tensor in sorted(canonical.items())
        },
    }


def load_cache_artifact(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
    expected_spatial_shapes: Iterable[tuple[int, int]] = ((30, 52), (52, 30)),
) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    path = Path(path)
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise RuntimeError(
                f"Cache artifact hash mismatch for {path}: expected {expected_sha256}, got {actual}."
            )
    tensors = load_file(str(path), device="cpu")
    validate_cache_tensors(tensors, expected_spatial_shapes=expected_spatial_shapes)
    return tensors


def cache_artifact_name(row_id: int) -> str:
    if int(row_id) != row_id or row_id < 0:
        raise ValueError(f"row_id must be a non-negative integer, got {row_id!r}.")
    return f"sample_{int(row_id):06d}.safetensors"


def build_source_fingerprint(
    records: Sequence[Stage1I2VRecord],
    *,
    model_paths: Mapping[str, str | os.PathLike[str]],
    preprocessing: Mapping[str, Any],
) -> dict[str, Any]:
    source_records = []
    for record in records:
        if not record.video_path.is_file() or not record.input_image_path.is_file():
            raise FileNotFoundError(
                f"Source files missing for row {record.row_id}; offline cache trust is forbidden."
            )
        source_records.append(
            {
                "row_id": record.row_id,
                "row_sha256": record.row_sha256,
                "video_sha256": sha256_file(record.video_path),
                "input_image_sha256": sha256_file(record.input_image_path),
            }
        )
    models = {}
    for label, path in sorted(model_paths.items()):
        files = tree_file_hashes(path)
        models[str(label)] = {"files": files, "aggregate_sha256": aggregate_file_hash(files)}
    value = {
        "records": source_records,
        "models": models,
        "preprocessing": dict(preprocessing),
        "cache_schema_version": STAGE1_CACHE_SCHEMA_VERSION,
    }
    value["aggregate_sha256"] = canonical_json_sha256(value)
    return value


def write_cache_manifest(
    cache_dir: str | os.PathLike[str],
    *,
    records: Sequence[Stage1I2VRecord],
    artifacts: Sequence[Mapping[str, Any]],
    source_fingerprint: Mapping[str, Any],
) -> Path:
    if len(records) != len(artifacts):
        raise ValueError(
            f"Cache manifest requires one artifact per record: {len(records)} != {len(artifacts)}."
        )
    entries = []
    for record, artifact in zip(records, artifacts):
        entry = dict(artifact)
        entry.update(
            row_id=record.row_id,
            row_sha256=record.row_sha256,
            height=record.height,
            width=record.width,
            bucket=record.bucket,
        )
        entries.append(entry)
    manifest = {
        "schema": "longlive_stage1_i2v_cache",
        "schema_version": STAGE1_CACHE_SCHEMA_VERSION,
        "num_samples": len(entries),
        "source_fingerprint": dict(source_fingerprint),
        "records": entries,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    path = Path(cache_dir) / STAGE1_CACHE_MANIFEST_NAME
    atomic_write_json(path, manifest)
    return path


def load_cache_manifest(
    cache_dir: str | os.PathLike[str], *, expected_num_samples: int | None = None
) -> dict[str, Any]:
    path = Path(cache_dir) / STAGE1_CACHE_MANIFEST_NAME
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_hash = manifest.pop("manifest_sha256", None)
    actual_hash = canonical_json_sha256(manifest)
    manifest["manifest_sha256"] = expected_hash
    if expected_hash != actual_hash:
        raise RuntimeError(
            f"Cache manifest hash mismatch for {path}: expected {expected_hash}, got {actual_hash}."
        )
    if manifest.get("schema") != "longlive_stage1_i2v_cache" or int(
        manifest.get("schema_version", -1)
    ) != STAGE1_CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported cache manifest schema in {path}.")
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != int(manifest.get("num_samples", -1)):
        raise RuntimeError(f"Invalid record count in cache manifest {path}.")
    if expected_num_samples is not None and len(records) != int(expected_num_samples):
        raise RuntimeError(
            f"Expected {expected_num_samples} cache records, found {len(records)}."
        )
    for expected_id, entry in enumerate(records):
        if int(entry.get("row_id", -1)) != expected_id:
            raise RuntimeError(
                f"Cache row ids must be contiguous/stable; expected {expected_id}, "
                f"got {entry.get('row_id')}."
            )
    return manifest


def audit_stage1_cache(
    cache_dir: str | os.PathLike[str],
    *,
    source_fingerprint: Mapping[str, Any] | None = None,
    expected_num_samples: int | None = None,
    allowed_latent_spatial_shapes: Iterable[tuple[int, int]] = ((30, 52), (52, 30)),
) -> dict[str, Any]:
    manifest = load_cache_manifest(cache_dir, expected_num_samples=expected_num_samples)
    if source_fingerprint is not None:
        expected = source_fingerprint.get("aggregate_sha256")
        actual = manifest.get("source_fingerprint", {}).get("aggregate_sha256")
        if expected != actual:
            raise RuntimeError(
                f"Cache source fingerprint mismatch: expected {expected}, got {actual}."
            )
    root = Path(cache_dir)
    bucket_counts: Counter[tuple[int, int]] = Counter()
    total_bytes = 0
    for entry in manifest["records"]:
        artifact_path = root / entry["path"]
        load_cache_artifact(
            artifact_path,
            expected_sha256=entry["sha256"],
            expected_spatial_shapes=allowed_latent_spatial_shapes,
        )
        total_bytes += artifact_path.stat().st_size
        bucket_counts[(int(entry["height"]), int(entry["width"]))] += 1
    return {
        "num_samples": len(manifest["records"]),
        "total_bytes": total_bytes,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "manifest_sha256": manifest["manifest_sha256"],
    }


class Stage1I2VCacheDataset(Dataset):
    """Cache-only dataset; artifacts are expected to be audited at startup."""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        *,
        expected_num_samples: int | None = None,
        verify_on_read: bool = False,
        allowed_latent_spatial_shapes: Iterable[tuple[int, int]] = ((30, 52), (52, 30)),
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        self.manifest = load_cache_manifest(
            self.cache_dir, expected_num_samples=expected_num_samples
        )
        self.entries = list(self.manifest["records"])
        self.verify_on_read = bool(verify_on_read)
        self.allowed_shapes = tuple(tuple(map(int, shape)) for shape in allowed_latent_spatial_shapes)
        self.spatial_shapes = [
            (int(entry["height"]), int(entry["width"])) for entry in self.entries
        ]

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[int(index)]
        tensors = load_cache_artifact(
            self.cache_dir / entry["path"],
            expected_sha256=entry["sha256"] if self.verify_on_read else None,
            expected_spatial_shapes=self.allowed_shapes,
        )
        clean_latent = tensors["video_latent"].clone()
        clean_latent[0:1].copy_(tensors["initial_latent"])
        return {
            "sample_id": int(entry["row_id"]),
            "clean_latent": clean_latent,
            "initial_latent": tensors["initial_latent"],
            "prompt_embeds": tensors["prompt_embeds"],
            "prompt_mask": tensors["prompt_mask"],
            "num_valid_latent_frames": 24,
            "height": int(entry["height"]),
            "width": int(entry["width"]),
            "bucket": entry["bucket"],
        }


def stage1_i2v_cache_collate(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty Stage-1 batch.")
    shapes = {tuple(item["clean_latent"].shape) for item in batch}
    if len(shapes) != 1:
        raise ValueError(f"A Stage-1 micro-batch cannot mix latent shapes: {sorted(shapes)}")
    result = {
        "sample_id": torch.tensor([int(item["sample_id"]) for item in batch], dtype=torch.long),
        "clean_latent": torch.stack([item["clean_latent"] for item in batch]),
        "initial_latent": torch.stack([item["initial_latent"] for item in batch]),
        "prompt_embeds": torch.stack([item["prompt_embeds"] for item in batch]),
        "prompt_mask": torch.stack([item["prompt_mask"] for item in batch]),
        "num_valid_latent_frames": torch.tensor(
            [int(item["num_valid_latent_frames"]) for item in batch], dtype=torch.long
        ),
        "height": torch.tensor([int(item["height"]) for item in batch], dtype=torch.long),
        "width": torch.tensor([int(item["width"]) for item in batch], dtype=torch.long),
        "bucket": [str(item["bucket"]) for item in batch],
    }
    return result
