"""Strict metadata and output contracts for Stage-1 continuation inference."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from utils.config import DEFAULT_NEGATIVE_PROMPT
from utils.stage1_causal_validation import (
    _write_inference_config,
    resolve_stage1_model_paths,
    validate_causal_video_output,
)
from utils.stage1_i2v_data import (
    _canonical_row,
    _resolved_path,
    _strict_positive_dimension,
    orientation_for_shape,
    validate_input_image,
)
from utils.stage1_io import atomic_write_json, canonical_json_sha256, sha256_file

CONTINUATION_METADATA_COLUMNS = (
    "input_image",
    "action_a_prompt",
    "hold_prompt",
    "action_b_prompt",
    "height",
    "width",
    "bucket",
    "case_group",
    "cat_id",
    "action_order",
    "action_a_blocks",
    "hold_blocks",
    "action_b_blocks",
    "soft_reanchor",
)
CONTINUATION_CATS = (
    "ragdoll",
    "russian_forest",
    "siamese",
    "tabby",
)
CONTINUATION_ACTION_ORDERS = (
    "jump_then_toy",
    "toy_then_jump",
)
CONTINUATION_CAT_IDENTITIES = {
    "ragdoll": "布偶猫",
    "russian_forest": "俄罗斯森林长毛猫",
    "siamese": "暹罗猫",
    "tabby": "狸花猫",
}
CONTINUATION_BLOCK_SIZE = 8
CONTINUATION_CACHE_FRAMES = 24
CONTINUATION_EXPECTED_BLOCKS = (3, 2, 3)
CONTINUATION_EXPECTED_ROWS = 8
CONTINUATION_PREPARED_SCHEMA = "longlive_stage1_continuation_prepared"
CONTINUATION_PREPARED_SCHEMA_VERSION = 1
CONTINUATION_OUTPUT_SCHEMA = "longlive_stage1_continuation_output_validation"
CONTINUATION_OUTPUT_SCHEMA_VERSION = 1
CONTINUATION_TOTAL_LATENT_FRAMES = 64
CONTINUATION_EXPECTED_PIXEL_FRAMES = 253
CONTINUATION_FPS = 24
CONTINUATION_SINK_SIZES = (0, 1)

_COMMON_PROMPT_CONSTRAINTS = (
    "全程摄像机静止",
    "纯白背景",
    "全景构图",
    "猫咪始终居中",
)
_FULL_BODY_CONSTRAINTS = (
    "整个身体",
    "四肢",
    "尾巴",
    "100%留在画面内",
)
_LOCAL_TIME_ANCHORS = (
    "0-1秒：",
    "1-3秒：",
    "3-4秒：",
)
_HOLD_REQUIRED_PHRASES = (
    "保持标准直立蹲坐",
    "呼吸",
    "耳朵轻微动作",
    "运动幅度很小",
)
_HOLD_FORBIDDEN_PHRASES = (
    "眨眼",
    "跳跃",
    "逗猫棒",
    "扑抓",
    "后腿发力",
    "抬一只前爪",
    "前爪交替",
    "落地",
)


@dataclass(frozen=True)
class ContinuationMetadataRecord:
    row_id: int
    input_image_path: Path
    action_a_prompt: str
    hold_prompt: str
    action_b_prompt: str
    height: int
    width: int
    bucket: str
    case_group: str
    cat_id: str
    action_order: str
    action_a_blocks: int
    hold_blocks: int
    action_b_blocks: int
    soft_reanchor: bool
    image_sha256: str
    canonical_row: Mapping[str, str] = field(repr=False)
    row_sha256: str

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def block_schedule(self) -> tuple[int, int, int]:
        return self.action_a_blocks, self.hold_blocks, self.action_b_blocks

    @property
    def latent_frame_schedule(self) -> tuple[int, int, int]:
        return tuple(blocks * CONTINUATION_BLOCK_SIZE for blocks in self.block_schedule)

    @property
    def total_latent_frames(self) -> int:
        return sum(self.latent_frame_schedule)

    def source_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "canonical_row": dict(self.canonical_row),
            "row_sha256": self.row_sha256,
            "input_image_path": os.fspath(self.input_image_path),
            "image_sha256": self.image_sha256,
        }


def _require_canonical_text(
    row: Mapping[str, str], name: str, *, row_number: int
) -> str:
    raw = row[name]
    value = raw.strip()
    if not value:
        raise ValueError(f"CSV row {row_number}: {name} must be non-empty.")
    if raw != value:
        raise ValueError(
            f"CSV row {row_number}: {name} must not contain surrounding whitespace."
        )
    return value


def _validate_common_prompt(
    prompt: str,
    *,
    prompt_name: str,
    cat_id: str,
    row_number: int,
) -> None:
    identity = CONTINUATION_CAT_IDENTITIES[cat_id]
    required = (identity, *_COMMON_PROMPT_CONSTRAINTS, *_FULL_BODY_CONSTRAINTS)
    missing = [phrase for phrase in required if phrase not in prompt]
    if missing:
        raise ValueError(
            f"CSV row {row_number}: {prompt_name} is not self-contained; "
            f"missing constraints {missing}."
        )


def _validate_action_prompt(
    prompt: str,
    *,
    action: str,
    prompt_name: str,
    row_number: int,
) -> None:
    clock_phrase = "本阶段从当前画面重新以0秒计时"
    if prompt.count(clock_phrase) != 1:
        raise ValueError(
            f"CSV row {row_number}: {prompt_name} must reset its textual clock once."
        )
    positions = []
    for anchor in _LOCAL_TIME_ANCHORS:
        if prompt.count(anchor) != 1:
            raise ValueError(
                f"CSV row {row_number}: {prompt_name} must contain {anchor!r} once."
            )
        positions.append(prompt.index(anchor))
    if positions != sorted(positions):
        raise ValueError(
            f"CSV row {row_number}: {prompt_name} local time anchors are out of order."
        )

    if action == "jump":
        if prompt.count("跳跃") != 1:
            raise ValueError(
                f"CSV row {row_number}: {prompt_name} must describe exactly one jump."
            )
        forbidden = ("逗猫棒", "扑抓")
    elif action == "toy":
        if prompt.count("逗猫棒") != 1 or prompt.count("扑抓") != 1:
            raise ValueError(
                f"CSV row {row_number}: {prompt_name} must describe exactly one toy interaction."
            )
        forbidden = ("跳跃",)
    else:  # pragma: no cover - guarded by the action-order table
        raise AssertionError(action)
    present = [phrase for phrase in forbidden if phrase in prompt]
    if present:
        raise ValueError(
            f"CSV row {row_number}: {prompt_name} mixes actions {present}."
        )


def _validate_hold_prompt(prompt: str, *, row_number: int) -> None:
    missing = [phrase for phrase in _HOLD_REQUIRED_PHRASES if phrase not in prompt]
    if missing:
        raise ValueError(
            f"CSV row {row_number}: hold_prompt is missing HOLD semantics {missing}."
        )
    forbidden = [phrase for phrase in _HOLD_FORBIDDEN_PHRASES if phrase in prompt]
    if forbidden:
        raise ValueError(
            f"CSV row {row_number}: hold_prompt contains forbidden action terms {forbidden}."
        )


def _validate_prompt_contract(record: ContinuationMetadataRecord) -> None:
    row_number = record.row_id + 2
    prompts = {
        "action_a_prompt": record.action_a_prompt,
        "hold_prompt": record.hold_prompt,
        "action_b_prompt": record.action_b_prompt,
    }
    for prompt_name, prompt in prompts.items():
        _validate_common_prompt(
            prompt,
            prompt_name=prompt_name,
            cat_id=record.cat_id,
            row_number=row_number,
        )

    action_a, action_b = {
        "jump_then_toy": ("jump", "toy"),
        "toy_then_jump": ("toy", "jump"),
    }[record.action_order]
    _validate_action_prompt(
        record.action_a_prompt,
        action=action_a,
        prompt_name="action_a_prompt",
        row_number=row_number,
    )
    _validate_hold_prompt(record.hold_prompt, row_number=row_number)
    _validate_action_prompt(
        record.action_b_prompt,
        action=action_b,
        prompt_name="action_b_prompt",
        row_number=row_number,
    )


def _validate_continuation_matrix(
    records: Sequence[ContinuationMetadataRecord],
) -> None:
    if len(records) != CONTINUATION_EXPECTED_ROWS:
        raise ValueError(
            f"Continuation metadata must contain exactly {CONTINUATION_EXPECTED_ROWS} rows, "
            f"found {len(records)}."
        )

    cat_counts = Counter(record.cat_id for record in records)
    expected_cat_counts = {cat_id: 2 for cat_id in CONTINUATION_CATS}
    if dict(cat_counts) != expected_cat_counts:
        raise ValueError(
            f"Continuation metadata cat matrix mismatch: expected "
            f"{expected_cat_counts}, got {dict(cat_counts)}."
        )

    by_cat: dict[str, list[ContinuationMetadataRecord]] = defaultdict(list)
    for record in records:
        by_cat[record.cat_id].append(record)
    expected_orders = set(CONTINUATION_ACTION_ORDERS)
    for cat_id in CONTINUATION_CATS:
        observed = [record.action_order for record in by_cat[cat_id]]
        if len(observed) != 2 or set(observed) != expected_orders:
            raise ValueError(
                f"cat_id {cat_id!r} must contain exactly one row for each action order; "
                f"got {observed}."
            )

    image_counts = Counter(record.input_image_path for record in records)
    if len(image_counts) != 4 or any(count != 2 for count in image_counts.values()):
        raise ValueError(
            "Continuation metadata must use exactly four input images, each exactly twice; "
            f"got {dict(image_counts)}."
        )
    image_cats: dict[Path, set[str]] = defaultdict(set)
    cat_images: dict[str, set[Path]] = defaultdict(set)
    for record in records:
        image_cats[record.input_image_path].add(record.cat_id)
        cat_images[record.cat_id].add(record.input_image_path)
    if any(len(cat_ids) != 1 for cat_ids in image_cats.values()) or any(
        len(paths) != 1 for paths in cat_images.values()
    ):
        raise ValueError("Each cat_id must map to one exclusive input image.")


def load_continuation_metadata(
    metadata_path: str | os.PathLike[str],
    *,
    validate_images: bool = True,
) -> list[ContinuationMetadataRecord]:
    """Load the fixed 8-case continuation matrix without relaxing old loaders."""

    metadata_path = Path(metadata_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    records: list[ContinuationMetadataRecord] = []
    seen_groups: dict[str, int] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {metadata_path}")
        fieldnames = [str(name) for name in reader.fieldnames]
        duplicates = sorted(
            name for name, count in Counter(fieldnames).items() if count > 1
        )
        if duplicates:
            raise ValueError(
                f"Continuation metadata has duplicate columns: {duplicates}"
            )
        if tuple(fieldnames) != CONTINUATION_METADATA_COLUMNS:
            missing = [
                name for name in CONTINUATION_METADATA_COLUMNS if name not in fieldnames
            ]
            unknown = [
                name for name in fieldnames if name not in CONTINUATION_METADATA_COLUMNS
            ]
            raise ValueError(
                "Continuation metadata header must exactly match the locked 14 columns "
                f"in order; missing={missing}, unknown={unknown}, got={fieldnames}."
            )

        for row_id, raw_row in enumerate(reader):
            row_number = row_id + 2
            if None in raw_row:
                raise ValueError(
                    f"CSV row {row_number} contains extra unheaded values."
                )
            if any(value is None for value in raw_row.values()):
                raise ValueError(f"CSV row {row_number} is missing one or more values.")
            row = _canonical_row(fieldnames, raw_row)

            height = _strict_positive_dimension(
                row["height"], label="height", row_number=row_number
            )
            width = _strict_positive_dimension(
                row["width"], label="width", row_number=row_number
            )
            if height % 16 or width % 16:
                raise ValueError(
                    f"CSV row {row_number}: height/width must be divisible by 16."
                )
            bucket = _require_canonical_text(row, "bucket", row_number=row_number)
            expected_bucket = orientation_for_shape(height, width)
            if bucket != expected_bucket:
                raise ValueError(
                    f"CSV row {row_number}: bucket={bucket!r} does not match "
                    f"shape {(height, width)} ({expected_bucket!r})."
                )

            cat_id = _require_canonical_text(row, "cat_id", row_number=row_number)
            if cat_id not in CONTINUATION_CATS:
                raise ValueError(
                    f"CSV row {row_number}: unsupported cat_id {cat_id!r}."
                )
            action_order = _require_canonical_text(
                row, "action_order", row_number=row_number
            )
            if action_order not in CONTINUATION_ACTION_ORDERS:
                raise ValueError(
                    f"CSV row {row_number}: unsupported action_order {action_order!r}."
                )
            case_group = _require_canonical_text(
                row, "case_group", row_number=row_number
            )
            expected_group = f"{cat_id}_{action_order}"
            if case_group != expected_group:
                raise ValueError(
                    f"CSV row {row_number}: case_group {case_group!r} must equal "
                    f"{expected_group!r}."
                )
            if case_group in seen_groups:
                raise ValueError(
                    f"CSV row {row_number}: duplicate case_group also used by row "
                    f"{seen_groups[case_group] + 2}: {case_group}."
                )
            seen_groups[case_group] = row_id

            block_values = tuple(
                _strict_positive_dimension(row[name], label=name, row_number=row_number)
                for name in (
                    "action_a_blocks",
                    "hold_blocks",
                    "action_b_blocks",
                )
            )
            if block_values != CONTINUATION_EXPECTED_BLOCKS:
                raise ValueError(
                    f"CSV row {row_number}: block schedule must be "
                    f"{CONTINUATION_EXPECTED_BLOCKS}, got {block_values}."
                )
            if sum(block_values) != 8:  # defensive: keep the total explicit
                raise ValueError(f"CSV row {row_number}: block schedule must total 8.")
            if row["soft_reanchor"] != "true":
                raise ValueError(
                    f"CSV row {row_number}: soft_reanchor must be canonical lowercase "
                    f"'true', got {row['soft_reanchor']!r}."
                )

            input_image_path = _resolved_path(
                metadata_path.parent,
                row["input_image"],
                label="input_image",
                row_number=row_number,
            )
            if not input_image_path.is_file():
                raise FileNotFoundError(
                    f"CSV row {row_number}: input image not found: {input_image_path}"
                )
            image_sha256 = sha256_file(input_image_path)
            record = ContinuationMetadataRecord(
                row_id=row_id,
                input_image_path=input_image_path,
                action_a_prompt=_require_canonical_text(
                    row, "action_a_prompt", row_number=row_number
                ),
                hold_prompt=_require_canonical_text(
                    row, "hold_prompt", row_number=row_number
                ),
                action_b_prompt=_require_canonical_text(
                    row, "action_b_prompt", row_number=row_number
                ),
                height=height,
                width=width,
                bucket=bucket,
                case_group=case_group,
                cat_id=cat_id,
                action_order=action_order,
                action_a_blocks=block_values[0],
                hold_blocks=block_values[1],
                action_b_blocks=block_values[2],
                soft_reanchor=True,
                image_sha256=image_sha256,
                canonical_row=row,
                row_sha256=canonical_json_sha256(
                    {
                        "row_id": row_id,
                        "canonical_row": row,
                        "resolved_input_image": os.fspath(input_image_path),
                        "image_sha256": image_sha256,
                    }
                ),
            )
            if validate_images:
                validate_input_image(record)
            _validate_prompt_contract(record)
            records.append(record)

    if not records:
        raise ValueError(f"Continuation metadata is empty: {metadata_path}")
    _validate_continuation_matrix(records)
    return records


def _require_empty_output_directory(path: Path, *, label: str) -> None:
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"{label} exists but is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"{label} must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _prepared_record_entry(
    record: ContinuationMetadataRecord,
    *,
    video_root: Path,
) -> dict[str, Any]:
    case_root = video_root / record.case_group
    case_root.mkdir(parents=True, exist_ok=False)
    outputs = []
    for sink_size in CONTINUATION_SINK_SIZES:
        stem = f"sink{sink_size}"
        outputs.append(
            {
                "sink_size": sink_size,
                "output_video": os.fspath(case_root / f"{stem}.mp4"),
                "session_trace": os.fspath(case_root / f"{stem}.session.json"),
            }
        )
    return {
        **record.source_dict(),
        "case_group": record.case_group,
        "cat_id": record.cat_id,
        "action_order": record.action_order,
        "height": record.height,
        "width": record.width,
        "bucket": record.bucket,
        "action_a_prompt": record.action_a_prompt,
        "hold_prompt": record.hold_prompt,
        "action_b_prompt": record.action_b_prompt,
        "block_schedule": list(record.block_schedule),
        "latent_frame_schedule": list(record.latent_frame_schedule),
        "soft_reanchor": record.soft_reanchor,
        "outputs": outputs,
    }


def prepare_continuation_inference(
    *,
    metadata_path: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    video_root: str | os.PathLike[str],
    base_checkpoint: str | os.PathLike[str],
    architecture_root: str | os.PathLike[str],
    t5_checkpoint: str | os.PathLike[str],
    tokenizer_dir: str | os.PathLike[str],
    vae_checkpoint: str | os.PathLike[str],
    sampling_steps: int = 50,
    guidance_scale: float = 5.0,
    seed: int = 1,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
) -> dict[str, Any]:
    """Prepare two geometry configs and one strict 8-case continuation manifest."""

    if int(sampling_steps) != 50:
        raise ValueError("continuation sampling_steps is locked to 50")
    if float(guidance_scale) != 5.0:
        raise ValueError("continuation guidance_scale is locked to 5.0")
    if int(seed) != 1:
        raise ValueError("continuation seed is locked to 1")
    if negative_prompt != DEFAULT_NEGATIVE_PROMPT:
        raise ValueError("continuation negative prompt must remain the Stage-1 default")

    metadata_path = Path(metadata_path).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    video_root = Path(video_root).expanduser().resolve()
    if output_root == video_root:
        raise ValueError("preparation root and video root must be distinct")
    _require_empty_output_directory(output_root, label="Continuation preparation root")
    _require_empty_output_directory(video_root, label="Continuation video root")

    paths = resolve_stage1_model_paths(
        base_checkpoint=base_checkpoint,
        architecture_root=architecture_root,
        t5_checkpoint=t5_checkpoint,
        tokenizer_dir=tokenizer_dir,
        vae_checkpoint=vae_checkpoint,
    )
    records = load_continuation_metadata(metadata_path)
    prepared_records = [
        _prepared_record_entry(record, video_root=video_root) for record in records
    ]
    entries_by_row = {int(entry["row_id"]): entry for entry in prepared_records}

    grouped: dict[tuple[str, int, int], list[ContinuationMetadataRecord]] = defaultdict(
        list
    )
    for record in records:
        grouped[(record.bucket, record.height, record.width)].append(record)

    manifest_path = output_root / "prepared_manifest.json"
    config_root = output_root / "configs"
    config_root.mkdir()
    bucket_entries = []
    for (bucket, height, width), bucket_records in sorted(grouped.items()):
        bucket_id = f"{bucket}_{height}x{width}"
        config_path = config_root / f"{bucket_id}.yaml"
        config = {
            "model_kwargs": {
                "model_name": "Wan2.2-TI2V-5B",
                "timestep_shift": 5.0,
                "num_frame_per_block": CONTINUATION_BLOCK_SIZE,
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
            "kv_quant": False,
            "torch_compile": False,
            "use_relative_rope": False,
            "rope_method": "linear",
            "output_folder": os.fspath(video_root),
            "num_samples": 1,
            "num_output_frames": CONTINUATION_TOTAL_LATENT_FRAMES,
            "save_latents_only": False,
            "save_with_index": False,
            "uniform_prompt": False,
            "data": {
                "image_or_video_shape": [
                    1,
                    CONTINUATION_TOTAL_LATENT_FRAMES,
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
                "shot_clean_recache": False,
                "multi_shot_rope_offset": 0.0,
                "streaming_vae": False,
                "async_vae": False,
                "vae_type": "wan",
            },
            "checkpoints": {
                "generator_ckpt": os.fspath(paths["base_checkpoint"]),
            },
            "logging": {"seed": seed},
            "continuation": {
                "enabled": True,
                "manifest_path": os.fspath(manifest_path),
                "bucket_id": bucket_id,
            },
        }
        _write_inference_config(config_path, config)
        bucket_entries.append(
            {
                "bucket_id": bucket_id,
                "bucket": bucket,
                "height": height,
                "width": width,
                "latent_height": height // 16,
                "latent_width": width // 16,
                "config_path": os.fspath(config_path),
                "config_sha256": sha256_file(config_path),
                "row_ids": [record.row_id for record in bucket_records],
                "case_groups": [record.case_group for record in bucket_records],
            }
        )

    manifest: dict[str, Any] = {
        "schema": CONTINUATION_PREPARED_SCHEMA,
        "schema_version": CONTINUATION_PREPARED_SCHEMA_VERSION,
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
            "num_latent_frames": CONTINUATION_TOTAL_LATENT_FRAMES,
            "num_frame_per_block": CONTINUATION_BLOCK_SIZE,
            "temporal_compression_ratio": 4,
            "expected_pixel_frames": CONTINUATION_EXPECTED_PIXEL_FRAMES,
            "fps": CONTINUATION_FPS,
            "latent_boundaries": {
                "action_a": [0, 23],
                "hold": [24, 39],
                "action_b": [40, 63],
            },
            "pixel_boundaries": {
                "action_a": [0, 92],
                "hold": [93, 156],
                "action_b": [157, 252],
            },
        },
        "sampling": {
            "solver": "unipc",
            "sampling_steps": sampling_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "negative_prompt": negative_prompt,
            "negative_prompt_sha256": hashlib.sha256(
                negative_prompt.encode("utf-8")
            ).hexdigest(),
        },
        "sink_sizes": list(CONTINUATION_SINK_SIZES),
        "video_root": os.fspath(video_root),
        "buckets": bucket_entries,
        "records": [entries_by_row[row_id] for row_id in sorted(entries_by_row)],
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(manifest_path, manifest)
    return manifest


def load_prepared_continuation_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    validate_metadata: bool = True,
) -> tuple[dict[str, Any], list[ContinuationMetadataRecord]]:
    """Load a prepared manifest and bind every entry back to strict metadata."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != CONTINUATION_PREPARED_SCHEMA:
        raise RuntimeError("unsupported continuation prepared manifest schema")
    if int(manifest.get("schema_version", -1)) != CONTINUATION_PREPARED_SCHEMA_VERSION:
        raise RuntimeError("unsupported continuation prepared manifest version")
    recorded_hash = manifest.get("manifest_sha256")
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    if recorded_hash != canonical_json_sha256(unhashed):
        raise RuntimeError("continuation prepared manifest SHA256 mismatch")

    metadata = manifest.get("metadata", {})
    metadata_path = Path(str(metadata.get("path", ""))).expanduser().resolve()
    if sha256_file(metadata_path) != metadata.get("sha256"):
        raise RuntimeError(
            "continuation metadata SHA256 differs from prepared manifest"
        )
    records = load_continuation_metadata(
        metadata_path,
        validate_images=validate_metadata,
    )
    if int(metadata.get("record_count", -1)) != len(records):
        raise RuntimeError("continuation metadata record count changed")
    if metadata.get("aggregate_row_sha256") != canonical_json_sha256(
        [record.row_sha256 for record in records]
    ):
        raise RuntimeError("continuation metadata row identities changed")

    frame_policy = manifest.get("frame_policy", {})
    locked_frame_policy = {
        "num_latent_frames": CONTINUATION_TOTAL_LATENT_FRAMES,
        "num_frame_per_block": CONTINUATION_BLOCK_SIZE,
        "temporal_compression_ratio": 4,
        "expected_pixel_frames": CONTINUATION_EXPECTED_PIXEL_FRAMES,
        "fps": CONTINUATION_FPS,
    }
    for name, expected in locked_frame_policy.items():
        if frame_policy.get(name) != expected:
            raise RuntimeError(
                f"continuation frame policy {name} must be {expected}, "
                f"got {frame_policy.get(name)!r}"
            )
    sampling = manifest.get("sampling", {})
    if (
        sampling.get("solver") != "unipc"
        or sampling.get("sampling_steps") != 50
        or float(sampling.get("guidance_scale", -1)) != 5.0
        or sampling.get("seed") != 1
        or sampling.get("negative_prompt") != DEFAULT_NEGATIVE_PROMPT
    ):
        raise RuntimeError("continuation sampling contract differs from locked values")
    if manifest.get("sink_sizes") != list(CONTINUATION_SINK_SIZES):
        raise RuntimeError("continuation sink matrix must be exactly [0, 1]")

    video_root_value = manifest.get("video_root")
    if not isinstance(video_root_value, str) or not video_root_value:
        raise RuntimeError("continuation video_root must be a non-empty path")
    video_root = Path(video_root_value).expanduser().resolve()
    if not video_root.is_dir():
        raise RuntimeError(f"continuation video_root is missing: {video_root}")

    manifest_records = manifest.get("records")
    if not isinstance(manifest_records, list) or len(manifest_records) != len(records):
        raise RuntimeError(
            "continuation prepared record set must contain exactly 8 rows"
        )
    by_row = {record.row_id: record for record in records}
    observed_rows: set[int] = set()
    observed_groups: set[str] = set()
    observed_videos: set[Path] = set()
    observed_traces: set[Path] = set()
    for entry in manifest_records:
        row_id = int(entry.get("row_id", -1))
        if row_id in observed_rows or row_id not in by_row:
            raise RuntimeError(
                f"invalid or duplicate continuation prepared row_id {row_id}"
            )
        record = by_row[row_id]
        if entry.get("case_group") in observed_groups:
            raise RuntimeError("duplicate continuation prepared case_group")
        observed_rows.add(row_id)
        observed_groups.add(str(entry.get("case_group")))
        expected_fields = {
            "case_group": record.case_group,
            "cat_id": record.cat_id,
            "action_order": record.action_order,
            "height": record.height,
            "width": record.width,
            "bucket": record.bucket,
            "row_sha256": record.row_sha256,
            "image_sha256": record.image_sha256,
            "block_schedule": list(record.block_schedule),
            "latent_frame_schedule": list(record.latent_frame_schedule),
            "soft_reanchor": True,
        }
        for name, expected in expected_fields.items():
            if entry.get(name) != expected:
                raise RuntimeError(
                    f"prepared row {row_id} field {name} differs from metadata"
                )
        outputs = entry.get("outputs")
        if (
            not isinstance(outputs, list)
            or any(not isinstance(item, Mapping) for item in outputs)
            or [item.get("sink_size") for item in outputs] != [0, 1]
        ):
            raise RuntimeError(f"prepared row {row_id} output sink mapping is invalid")
        case_root = (video_root / record.case_group).resolve()
        try:
            case_root.relative_to(video_root)
        except ValueError as exc:
            raise RuntimeError("continuation case path escaped its video root") from exc
        if not case_root.is_dir():
            raise RuntimeError(
                f"continuation case output directory is missing: {case_root}"
            )
        for output in outputs:
            sink_size = int(output["sink_size"])
            try:
                video_path = (
                    Path(os.fspath(output.get("output_video"))).expanduser().resolve()
                )
                trace_path = (
                    Path(os.fspath(output.get("session_trace"))).expanduser().resolve()
                )
            except TypeError as exc:
                raise RuntimeError(
                    "continuation output paths must be path strings"
                ) from exc
            try:
                video_path.relative_to(video_root)
                trace_path.relative_to(video_root)
            except ValueError as exc:
                raise RuntimeError(
                    "continuation output escaped its video root"
                ) from exc
            if (
                video_path != case_root / f"sink{sink_size}.mp4"
                or trace_path != case_root / f"sink{sink_size}.session.json"
            ):
                raise RuntimeError("continuation output path mapping is not canonical")
            if video_path in observed_videos or trace_path in observed_traces:
                raise RuntimeError("continuation output path mapping is duplicated")
            observed_videos.add(video_path)
            observed_traces.add(trace_path)

    expected_output_count = CONTINUATION_EXPECTED_ROWS * len(CONTINUATION_SINK_SIZES)
    if (
        len(observed_videos) != expected_output_count
        or len(observed_traces) != expected_output_count
    ):
        raise RuntimeError("continuation prepared output mapping is incomplete")

    bucket_rows: list[int] = []
    buckets = manifest.get("buckets")
    if not isinstance(buckets, list) or len(buckets) != 2:
        raise RuntimeError("continuation preparation must contain two geometry buckets")
    for bucket in buckets:
        config_path = Path(str(bucket.get("config_path", "")))
        if not config_path.is_file() or sha256_file(config_path) != bucket.get(
            "config_sha256"
        ):
            raise RuntimeError(
                f"continuation config is missing or changed: {config_path}"
            )
        row_ids = [int(value) for value in bucket.get("row_ids", [])]
        case_groups = [str(value) for value in bucket.get("case_groups", [])]
        if case_groups != [by_row[row_id].case_group for row_id in row_ids]:
            raise RuntimeError(
                "continuation bucket row/case mapping differs from metadata"
            )
        for row_id in row_ids:
            record = by_row.get(row_id)
            if record is None or (
                record.bucket != bucket.get("bucket")
                or record.height != int(bucket.get("height", -1))
                or record.width != int(bucket.get("width", -1))
            ):
                raise RuntimeError("continuation bucket geometry mapping is invalid")
        bucket_rows.extend(row_ids)
    if sorted(bucket_rows) != sorted(by_row) or len(bucket_rows) != len(
        set(bucket_rows)
    ):
        raise RuntimeError("continuation bucket rows are missing or duplicated")
    return manifest, records


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _validate_continuation_session_trace(
    trace: Mapping[str, Any],
    *,
    entry: Mapping[str, Any],
    record: ContinuationMetadataRecord,
    sink_size: int,
    manifest: Mapping[str, Any],
    output_video: Path,
    session_trace: Path,
) -> None:
    if trace.get("schema") != "longlive_stage1_continuation_session":
        raise RuntimeError("unsupported continuation session trace schema")
    if int(trace.get("schema_version", -1)) != 1:
        raise RuntimeError("unsupported continuation session trace version")
    if trace.get("status") != "finished" or trace.get("failure") is not None:
        raise RuntimeError(
            f"continuation session did not finish cleanly for {record.case_group}/sink{sink_size}"
        )
    metadata = trace.get("metadata", {})
    expected_metadata = {
        "row_id": record.row_id,
        "row_sha256": record.row_sha256,
        "image_sha256": record.image_sha256,
        "case_group": record.case_group,
        "cat_id": record.cat_id,
        "action_order": record.action_order,
        "block_schedule": list(record.block_schedule),
        "soft_reanchor": True,
    }
    for name, expected in expected_metadata.items():
        if metadata.get(name) != expected:
            raise RuntimeError(
                f"session trace metadata {name} mismatch for {record.case_group}/sink{sink_size}"
            )

    sampling = trace.get("sampling", {})
    expected_sampling = manifest["sampling"]
    for name in (
        "solver",
        "sampling_steps",
        "guidance_scale",
        "seed",
        "negative_prompt_sha256",
    ):
        if sampling.get(name) != expected_sampling.get(name):
            raise RuntimeError(
                f"session trace sampling {name} mismatch for {record.case_group}/sink{sink_size}"
            )
    if sampling.get("sink_size") != sink_size:
        raise RuntimeError("session trace sink_size mismatch")

    locked = trace.get("locked", {})
    latent_height = record.height // 16
    latent_width = record.width // 16
    frame_seq_length = latent_height * latent_width // 4
    locked_expected = {
        "batch_size": 1,
        "latent_shape": [48, latent_height, latent_width],
        "dtype": "bfloat16",
        "guidance_scale": 5.0,
        "sample_solver": "unipc",
        "sampling_steps": 50,
        "timestep_shift": 5.0,
        "num_train_timesteps": 1000,
        "negative_prompt_sha256": expected_sampling["negative_prompt_sha256"],
        "block_size": CONTINUATION_BLOCK_SIZE,
        "sink_size": sink_size,
        "pipeline_sink_size": 0,
        "global_sink_size": 0,
        "multi_shot_sink": False,
        "shot_clean_recache": False,
        "multi_shot_rope_offset": 0.0,
        "quantize_kv": False,
        "independent_first_frame": True,
        "streaming_vae": False,
        "async_vae": False,
        "local_attn_size_config": -1,
        "use_relative_rope": False,
        "rope_method": "linear",
        "effective_t_scale": 1.0,
        "effective_local_attn_size": -1,
        "effective_attention_local_size": CONTINUATION_CACHE_FRAMES,
        "effective_use_relative_rope": False,
        "effective_original_seq_len": None,
        "effective_rope_temporal_offset": 0.0,
        "frame_seq_length": frame_seq_length,
    }
    for name, expected in locked_expected.items():
        if locked.get(name) != expected:
            raise RuntimeError(
                f"session trace locked {name} mismatch for {record.case_group}/sink{sink_size}"
            )
    if not str(locked.get("device", "")).startswith("cuda"):
        raise RuntimeError("continuation session must run on CUDA")
    if int(trace.get("cursor_frames", -1)) != CONTINUATION_TOTAL_LATENT_FRAMES:
        raise RuntimeError("continuation session cursor must finish at 64")
    for name in ("initial_latent_sha256", "noise_identity_sha256"):
        if not _is_sha256(trace.get(name)):
            raise RuntimeError(f"session trace has invalid {name}")

    segments = trace.get("segments")
    expected_segment_values = (
        (
            "action_a",
            0,
            23,
            24,
            3,
            record.action_a_prompt,
            False,
        ),
        ("hold", 24, 39, 16, 2, record.hold_prompt, False),
        (
            "action_b",
            40,
            63,
            24,
            3,
            record.action_b_prompt,
            True,
        ),
    )
    if not isinstance(segments, list) or len(segments) != 3:
        raise RuntimeError("continuation session must contain exactly three segments")
    expected_prompt_hashes = []
    for segment, expected in zip(segments, expected_segment_values, strict=True):
        name, start, end, frames, blocks, prompt, carry = expected
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        expected_prompt_hashes.append(prompt_hash)
        segment_expected = {
            "name": name,
            "start_latent": start,
            "end_latent": end,
            "latent_frames": frames,
            "blocks": blocks,
            "carry_last_latent_as_anchor": carry,
        }
        for field_name, field_value in segment_expected.items():
            if segment.get(field_name) != field_value:
                raise RuntimeError(
                    f"session segment {name} field {field_name} mismatch"
                )
        prompt_audit = segment.get("prompt", {})
        if (
            prompt_audit.get("sha256") != prompt_hash
            or not 0 < int(prompt_audit.get("token_count", 0)) < 512
            or prompt_audit.get("token_limit") != 512
            or prompt_audit.get("cleaning") != "whitespace"
            or prompt_audit.get("add_special_tokens") is not True
            or prompt_audit.get("truncation") is not False
        ):
            raise RuntimeError(f"session segment {name} prompt audit is invalid")

    negative_prompt_audit = trace.get("negative_prompt", {})
    if (
        negative_prompt_audit.get("sha256")
        != expected_sampling["negative_prompt_sha256"]
        or not 0 < int(negative_prompt_audit.get("token_count", 0)) < 512
        or negative_prompt_audit.get("token_limit") != 512
        or negative_prompt_audit.get("cleaning") != "whitespace"
        or negative_prompt_audit.get("add_special_tokens") is not True
        or negative_prompt_audit.get("truncation") is not False
    ):
        raise RuntimeError("continuation negative prompt audit is invalid")

    noise_slices = trace.get("noise_slices")
    expected_noise_ranges = ((0, 24), (24, 40), (40, 64))
    if not isinstance(noise_slices, list) or len(noise_slices) != 3:
        raise RuntimeError("continuation session noise slice trace is invalid")
    for index, (noise_slice, expected_range) in enumerate(
        zip(noise_slices, expected_noise_ranges, strict=True)
    ):
        if (
            noise_slice.get("segment_index") != index
            or noise_slice.get("start_latent") != expected_range[0]
            or noise_slice.get("end_latent_exclusive") != expected_range[1]
            or not _is_sha256(noise_slice.get("sha256"))
        ):
            raise RuntimeError("continuation session noise slice mapping is invalid")

    blocks = trace.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 8:
        raise RuntimeError("continuation session block trace must contain eight blocks")
    expected_globals = [frame_seq_length * value for value in range(8, 65, 8)]
    expected_locals = [
        frame_seq_length * value for value in (8, 16, 24, 24, 24, 24, 24, 24)
    ]
    expected_segment_indices = [0, 0, 0, 1, 1, 2, 2, 2]
    expected_block_prompt_hashes = [
        expected_prompt_hashes[segment_index]
        for segment_index in expected_segment_indices
    ]
    for block_index, block in enumerate(blocks):
        expected_end = (block_index + 1) * CONTINUATION_BLOCK_SIZE
        block_expected = {
            "block_index": block_index,
            "segment_index": expected_segment_indices[block_index],
            "global_start_latent": block_index * CONTINUATION_BLOCK_SIZE,
            "global_end_latent": expected_end - 1,
            "positive_prompt_sha256": expected_block_prompt_hashes[block_index],
            "negative_prompt_sha256": expected_sampling["negative_prompt_sha256"],
            "global_end_tokens": expected_globals[block_index],
            "local_end_tokens": expected_locals[block_index],
            "global_end_frames": expected_end,
            "local_end_frames": min(expected_end, 24),
            "cache_capacity_tokens": frame_seq_length * 24,
            "effective_attention_local_size": CONTINUATION_CACHE_FRAMES,
            "pinned_start": -1,
            "pinned_len": 0,
        }
        for name, expected in block_expected.items():
            if block.get(name) != expected:
                raise RuntimeError(
                    f"session block {block_index} field {name} mismatch: "
                    f"expected {expected!r}, got {block.get(name)!r}"
                )
    expected_anchors = [
        {"kind": "initial", "source": "initial_latent", "destination": 0},
        {"kind": "soft_reanchor", "source": 39, "destination": 40},
    ]
    if trace.get("anchors") != expected_anchors:
        raise RuntimeError("continuation session anchor mapping is invalid")

    result = trace.get("result", {})
    if (
        result.get("latent_shape") != [1, 64, 48, latent_height, latent_width]
        or result.get("pixel_shape")
        != [1, CONTINUATION_EXPECTED_PIXEL_FRAMES, 3, record.height, record.width]
        or result.get("decode_calls") != 1
        or result.get("decode_mode") != "single_full_sequence"
    ):
        raise RuntimeError("continuation session result shape/decode policy is invalid")
    if result.get("latent_boundaries") != manifest["frame_policy"]["latent_boundaries"]:
        raise RuntimeError("continuation latent boundaries differ from manifest")
    if result.get("pixel_boundaries") != manifest["frame_policy"]["pixel_boundaries"]:
        raise RuntimeError("continuation pixel boundaries differ from manifest")
    trace_output = trace.get("output", {})
    if Path(str(trace_output.get("output_video", ""))).resolve() != output_video:
        raise RuntimeError("continuation trace output path mismatch")
    if Path(str(trace_output.get("session_trace", ""))).resolve() != session_trace:
        raise RuntimeError("continuation trace session path mismatch")


def validate_continuation_outputs(
    prepared_manifest_path: str | os.PathLike[str],
    *,
    minimum_first_frame_psnr_db: float = 12.0,
    minimum_frame_std: float = 5.0,
    minimum_temporal_abs_diff: float = 0.05,
    video_validator: Callable[..., dict[str, Any]] = validate_causal_video_output,
) -> dict[str, Any]:
    """Validate the exact 16-video/16-trace matrix and enrich traces atomically."""

    manifest, records = load_prepared_continuation_manifest(
        prepared_manifest_path,
        validate_metadata=True,
    )
    record_by_row = {record.row_id: record for record in records}
    video_root = Path(manifest["video_root"]).resolve()
    expected_videos: set[Path] = set()
    expected_traces: set[Path] = set()
    mapped_outputs: list[
        tuple[
            Mapping[str, Any], ContinuationMetadataRecord, Mapping[str, Any], Path, Path
        ]
    ] = []
    for entry in manifest["records"]:
        record = record_by_row[int(entry["row_id"])]
        for output in entry["outputs"]:
            sink_size = int(output["sink_size"])
            video_path = Path(output["output_video"]).resolve()
            trace_path = Path(output["session_trace"]).resolve()
            try:
                video_path.relative_to(video_root)
                trace_path.relative_to(video_root)
            except ValueError as exc:
                raise RuntimeError(
                    "continuation output escaped its video root"
                ) from exc
            expected_video_name = f"sink{sink_size}.mp4"
            expected_trace_name = f"sink{sink_size}.session.json"
            if (
                video_path.parent.name != record.case_group
                or trace_path.parent != video_path.parent
                or video_path.name != expected_video_name
                or trace_path.name != expected_trace_name
            ):
                raise RuntimeError("continuation output path mapping is not canonical")
            if video_path in expected_videos or trace_path in expected_traces:
                raise RuntimeError(
                    "continuation manifest maps an output more than once"
                )
            expected_videos.add(video_path)
            expected_traces.add(trace_path)
            mapped_outputs.append((entry, record, output, video_path, trace_path))

    actual_videos = {path.resolve() for path in video_root.rglob("*.mp4")}
    actual_traces = {path.resolve() for path in video_root.rglob("*.session.json")}
    if actual_videos != expected_videos:
        raise RuntimeError(
            "continuation MP4 set mismatch: "
            f"missing={sorted(os.fspath(path) for path in expected_videos - actual_videos)}, "
            f"unexpected={sorted(os.fspath(path) for path in actual_videos - expected_videos)}"
        )
    if actual_traces != expected_traces:
        raise RuntimeError(
            "continuation session trace set mismatch: "
            f"missing={sorted(os.fspath(path) for path in expected_traces - actual_traces)}, "
            f"unexpected={sorted(os.fspath(path) for path in actual_traces - expected_traces)}"
        )

    trace_payloads: dict[Path, dict[str, Any]] = {}
    pair_identity: dict[str, dict[int, tuple[str, str]]] = defaultdict(dict)
    for entry, record, output, video_path, trace_path in mapped_outputs:
        sink_size = int(output["sink_size"])
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        _validate_continuation_session_trace(
            trace,
            entry=entry,
            record=record,
            sink_size=sink_size,
            manifest=manifest,
            output_video=video_path,
            session_trace=trace_path,
        )
        pair_identity[record.case_group][sink_size] = (
            str(trace["initial_latent_sha256"]),
            str(trace["noise_identity_sha256"]),
        )
        trace_payloads[trace_path] = trace
    for case_group, identities in pair_identity.items():
        if set(identities) != set(CONTINUATION_SINK_SIZES):
            raise RuntimeError(f"incomplete sink identity pair for {case_group}")
        if identities[0] != identities[1]:
            raise RuntimeError(
                f"sink0/1 initial latent or noise identity differs for {case_group}"
            )

    samples = []
    pending_trace_updates: list[tuple[Path, dict[str, Any]]] = []
    for entry, record, output, video_path, trace_path in mapped_outputs:
        sink_size = int(output["sink_size"])
        gate = video_validator(
            video_path,
            record.input_image_path,
            row_id=record.row_id,
            expected_width=record.width,
            expected_height=record.height,
            expected_frames=CONTINUATION_EXPECTED_PIXEL_FRAMES,
            expected_fps=CONTINUATION_FPS,
            minimum_first_frame_psnr_db=minimum_first_frame_psnr_db,
            minimum_frame_std=minimum_frame_std,
            minimum_temporal_abs_diff=minimum_temporal_abs_diff,
        )
        trace = trace_payloads[trace_path]
        trace["technical_validation"] = {
            "status": "pass",
            "video_sha256": sha256_file(video_path),
            **gate,
        }
        pending_trace_updates.append((trace_path, trace))
        samples.append(
            {
                "row_id": record.row_id,
                "case_group": record.case_group,
                "cat_id": record.cat_id,
                "action_order": record.action_order,
                "sink_size": sink_size,
                "seed": 1,
                "soft_reanchor": True,
                "output_video": os.fspath(video_path),
                "session_trace": os.fspath(trace_path),
                "noise_identity_sha256": trace["noise_identity_sha256"],
                "initial_latent_sha256": trace["initial_latent_sha256"],
                "stream": gate["stream"],
                "metrics": gate["metrics"],
            }
        )

    # No trace advertises a passing technical gate until all 16 outputs pass.
    for trace_path, trace in pending_trace_updates:
        atomic_write_json(trace_path, trace)
    return {
        "schema": CONTINUATION_OUTPUT_SCHEMA,
        "schema_version": CONTINUATION_OUTPUT_SCHEMA_VERSION,
        "status": "pass",
        "sample_count": len(samples),
        "expected_sample_count": CONTINUATION_EXPECTED_ROWS
        * len(CONTINUATION_SINK_SIZES),
        "prepared_manifest": os.fspath(
            Path(prepared_manifest_path).expanduser().resolve()
        ),
        "samples": samples,
    }
