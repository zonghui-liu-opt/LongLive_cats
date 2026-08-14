"""Strict Stage-2 I2V cache audit, manifest, and cache-only dataset.

Stage-2 has a stricter temporal contract than the official Stage-1 F24 cache.
Its F25 ``video_latent`` contains one cached sink slot followed by 24 real
future latents, while the separately cached ``initial_latent`` remains the only
authoritative conditioning sink.  Nothing in this module substitutes
``video_latent[0]`` for that explicit input-image latent.

The expensive, full-cache audit is intended to run once before training.  The
dataset then consumes the immutable audit manifest and can optionally recheck
artifact hashes on every read.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

from utils.config import DEFAULT_NEGATIVE_PROMPT
from utils.stage1_io import (
    aggregate_file_hash,
    atomic_output_path,
    atomic_write_json,
    canonical_json_bytes,
    canonical_json_sha256,
    sha256_file,
    tree_file_hashes,
)
from utils.stage1_i2v_schema import STAGE1_CACHE_SCHEMA_VERSION

if TYPE_CHECKING:
    from utils.stage1_i2v_data import Stage1I2VRecord

STAGE2_CACHE_MANIFEST_NAME = "stage2_i2v_manifest.json"
STAGE2_CACHE_SCHEMA = "longlive_stage2_i2v_cache"
STAGE2_CACHE_SCHEMA_VERSION = 1
STAGE2_F25_SOURCE_CACHE_SCHEMA = "longlive_stage2_i2v_f25_source_cache"
STAGE2_F25_SOURCE_CACHE_SCHEMA_VERSION = 1
STAGE2_F25_PREPARATION_SCHEMA = "longlive_stage2_i2v_f25_preparation"
STAGE2_F25_PREPARATION_SCHEMA_VERSION = 1
STAGE2_F25_SUCCESS_NAME = "_F25_SUCCESS.json"
STAGE2_F25_SUCCESS_SCHEMA = "longlive_stage2_i2v_f25_success"
STAGE2_F25_SUCCESS_SCHEMA_VERSION = 1
STAGE2_NEGATIVE_SCHEMA = "longlive_stage2_negative_conditioning"
STAGE2_NEGATIVE_SCHEMA_VERSION = 1
STAGE2_TEXT_ENCODING_UPGRADE_SCHEMA = "longlive_stage2_text_encoding_upgrade"
STAGE2_TEXT_ENCODING_UPGRADE_SCHEMA_VERSION = 1
STAGE2_TEXT_ENCODING_UPGRADE_KEY = "stage2_text_encoding_upgrade"
_STAGE1_SOURCE_CACHE_SCHEMA = "longlive_stage1_i2v_cache"
STAGE2_F25_FRAME_POLICY: dict[str, Any] = {
    "minimum_source_pixel_frames": 97,
    "selected_frame_start": 0,
    "selected_frame_count": 97,
    "selected_frame_end_inclusive": 96,
    "selection": "contiguous_presentation_order_no_seek_no_resample",
    "expected_fps": 24.0,
    "fps_absolute_tolerance": 1.0e-3,
    "allow_resize": False,
    "allow_padding": False,
    "pixel_dtype": "uint8_rgb24",
    "vae_input_range": [-1.0, 1.0],
    "vae_input_layout": "BCTHW",
    "latent_dtype": "bfloat16",
    "latent_frames": 25,
    "latent_channels": 48,
    "temporal_compression_ratio": 4,
    "training_slice": "video_latent[1:25]",
    "fabricate_or_duplicate_frames": False,
}
STAGE2_F25_FRAME_POLICY_SHA256 = canonical_json_sha256(STAGE2_F25_FRAME_POLICY)
STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION = (
    "I attest that this legacy positive cache was encoded with the locked Wan "
    "seq512/whitespace/add-special-tokens/right-padding/exact-zero-padding contract."
)
STAGE2_NEGATIVE_PROMPT_SHA256 = (
    "ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e"
)
if hashlib.sha256(DEFAULT_NEGATIVE_PROMPT.encode("utf-8")).hexdigest() != (
    STAGE2_NEGATIVE_PROMPT_SHA256
):
    raise RuntimeError(
        "utils.config.DEFAULT_NEGATIVE_PROMPT changed; Stage-2 negative cache "
        "must remain byte-for-byte compatible with the locked Stage-1 prompt."
    )

STAGE2_CACHE_TENSOR_SCHEMA: dict[str, Any] = {
    "video_latent": {
        "dtype": "bfloat16",
        "shape": [25, 48, "H", "W"],
        "meaning": "cached_sink_slot_0_then_real_future_slots_1_to_24",
    },
    "initial_latent": {
        "dtype": "bfloat16",
        "shape": [1, 48, "H", "W"],
        "meaning": "explicit_input_image_latent_and_only_conditioning_sink",
    },
    "prompt_embeds": {
        "dtype": "bfloat16",
        "shape": [512, 4096],
        "padding": "exact_zero_where_prompt_mask_is_false",
    },
    "prompt_mask": {
        "dtype": "bool",
        "shape": [512],
        "meaning": "right_padded_valid_token_prefix",
    },
    "training_slice": "real_future_equals_video_latent[1:25]",
    "allowed_latent_spatial_shapes": [[30, 52], [52, 30]],
}
STAGE2_CACHE_TENSOR_SCHEMA_SHA256 = canonical_json_sha256(STAGE2_CACHE_TENSOR_SCHEMA)

_CACHE_TENSOR_DTYPES = {
    "video_latent": torch.bfloat16,
    "initial_latent": torch.bfloat16,
    "prompt_embeds": torch.bfloat16,
    "prompt_mask": torch.bool,
}
_NEGATIVE_TENSOR_DTYPES = {
    "prompt_embeds": torch.bfloat16,
    "prompt_mask": torch.bool,
}
_NEGATIVE_ENCODER_KEYS = {
    "t5_checkpoint_aggregate_sha256",
    "tokenizer_aggregate_sha256",
    "tokenizer_revision",
    "cleaning",
    "add_special_tokens",
    "sequence_length",
    "padding_side",
    "embedding_padding_value",
}
_LOCKED_TEXT_ENCODING_SETTINGS: dict[str, Any] = {
    "cleaning": "whitespace",
    "add_special_tokens": True,
    "sequence_length": 512,
    "padding_side": "right",
    "embedding_padding_value": 0.0,
}
_LOCKED_TOKENIZER_RUNTIME_AUDIT: dict[str, Any] = {
    **_LOCKED_TEXT_ENCODING_SETTINGS,
    "validated_special_token_growth": True,
    "validated_right_padding_mask": True,
}
_TEXT_ENCODING_UPGRADE_KEYS = {
    "schema",
    "schema_version",
    "original_source_manifest",
    "verification",
    "operator_attestation",
}
_TEXT_ENCODING_UPGRADE_ORIGINAL_KEYS = {
    "path",
    "manifest_sha256",
    "source_fingerprint_sha256",
}
_TEXT_ENCODING_UPGRADE_VERIFICATION_KEYS = {
    "validator_file",
    "validator_file_sha256",
    "text_encoding_contract_sha256",
    "t5_checkpoint_aggregate_sha256",
    "tokenizer_aggregate_sha256",
    "tokenizer_runtime_audit",
}
_TEXT_ENCODING_UPGRADE_ATTESTATION_KEYS = {"operator_id", "statement"}
_NEGATIVE_ARTIFACT_KEYS = {
    "path",
    "size",
    "sha256",
    "tensors",
    "prompt_valid_tokens",
    "padding_sha256",
}
_NEGATIVE_MANIFEST_KEYS = {
    "schema",
    "schema_version",
    "text",
    "text_utf8_sha256",
    "positive_cache_manifest_sha256",
    "encoder",
    "artifact",
    "manifest_sha256",
}
_STAGE2_MANIFEST_KEYS = {
    "schema",
    "schema_version",
    "common_tensor_schema",
    "common_tensor_schema_sha256",
    "num_samples",
    "total_bytes",
    "orientation_counts",
    "actions",
    "negative_conditioning",
    "provenance",
    "records",
    "manifest_sha256",
}
_STAGE2_PROVENANCE_KEYS = {
    "metadata_path",
    "metadata_sha256",
    "source_cache_manifest_path",
    "source_cache_manifest_sha256",
    "config_contract_sha256",
    "config_launch_sha256",
}
_STAGE2_NEGATIVE_BINDING_KEYS = {
    "manifest_path",
    "manifest_sha256",
    "artifact_sha256",
    "text_utf8_sha256",
}
_STAGE2_RECORD_KEYS = {
    "row_id",
    "row_sha256",
    "video",
    "prompt_utf8_sha256",
    "action_id",
    "height",
    "width",
    "bucket",
    "path",
    "size",
    "sha256",
    "latent_spatial_shape",
    "real_future_shape",
    "prompt_valid_tokens",
    "prompt_padding_sha256",
    "initial_vs_video0",
    "tensors",
}
_F25_BASE_MANIFEST_KEYS = {
    "schema",
    "schema_version",
    "num_samples",
    "source_fingerprint",
    "preparation",
    "records",
    "manifest_sha256",
}
_F25_PREPARATION_KEYS = {
    "schema",
    "schema_version",
    "input_manifest",
    "producer",
    "config",
    "metadata",
    "frame_policy",
    "frame_policy_sha256",
    "vae",
    "output_policy",
    "summary",
    "contract_sha256",
}
_F25_INPUT_MANIFEST_KEYS = {
    "path",
    "file_sha256",
    "manifest_sha256",
    "source_fingerprint_sha256",
    "schema",
    "schema_version",
}
_F25_PRODUCER_KEYS = {"file", "file_sha256"}
_F25_CONFIG_KEYS = {
    "path",
    "file_sha256",
    "contract_sha256",
    "launch_sha256",
    "data_contract_sha256",
}
_F25_METADATA_KEYS = {"path", "file_sha256"}
_F25_VAE_KEYS = {"aggregate_sha256"}
_F25_OUTPUT_POLICY_KEYS = {
    "independent_output_directory",
    "in_place_overwrite",
    "atomic_artifacts",
    "atomic_completion_sidecars",
    "stable_row_modulo_sharding",
}
_F25_SUMMARY_KEYS = {
    "reused_f25",
    "reencoded_f24",
    "reverified_f25",
}
_F25_RECORD_REQUIRED_KEYS = {
    "row_id",
    "row_sha256",
    "height",
    "width",
    "bucket",
    "path",
    "size",
    "sha256",
    "tensors",
    "decision",
    "input_artifact",
    "source_video_sha256",
    "source_input_image_sha256",
    "vae_aggregate_sha256",
    "frame_policy_sha256",
    "preparation_contract_sha256",
    "verification",
}
_F25_INPUT_ARTIFACT_KEYS = {"path", "size", "sha256", "tensors"}
_F25_RECORD_VERIFICATION_KEYS = {
    "output_reuses_input_bytes",
    "f24_prefix_exact",
    "f25_reverification_exact",
}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def stage2_f25_data_contract_sha256(
    *,
    expected_num_samples: int,
    metadata_sha256: str,
    allowed_latent_spatial_shapes: Iterable[tuple[int, int]] = ((30, 52), (52, 30)),
) -> str:
    """Hash only cache semantics, not optimizer/training-launch choices."""

    if type(expected_num_samples) is not int or expected_num_samples <= 0:
        raise ValueError("expected_num_samples must be a positive integer.")
    metadata_sha256 = _require_sha256(metadata_sha256, "F25 metadata SHA256")
    shapes = sorted([list(map(int, shape)) for shape in allowed_latent_spatial_shapes])
    if shapes != [[30, 52], [52, 30]]:
        raise ValueError("F25 latent spatial shapes must be exactly 30x52 and 52x30.")
    return canonical_json_sha256(
        {
            "expected_num_samples": expected_num_samples,
            "video_latent_frames": 25,
            "initial_latent_frames": 1,
            "future_latent_frames": 24,
            "latent_channels": 48,
            "cache_dtype": "bfloat16",
            "allowed_latent_spatial_shapes": shapes,
            "metadata_sha256": metadata_sha256,
            "frame_policy": STAGE2_F25_FRAME_POLICY,
            "frame_policy_sha256": STAGE2_F25_FRAME_POLICY_SHA256,
            "tensor_schema_sha256": STAGE2_CACHE_TENSOR_SCHEMA_SHA256,
        }
    )


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a JSON object.")
    return dict(value)


def _require_exact_keys(
    value: Any, *, label: str, expected: set[str]
) -> dict[str, Any]:
    mapping = _require_mapping(value, label)
    actual = set(mapping)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(f"{label} keys mismatch: missing={missing}, extra={extra}.")
    return mapping


def _require_required_keys(
    value: Any, *, label: str, required: set[str]
) -> dict[str, Any]:
    """Require functional fields while allowing inert legacy metadata."""

    mapping = _require_mapping(value, label)
    missing = sorted(required - set(mapping))
    if missing:
        raise RuntimeError(f"{label} keys mismatch: missing={missing}.")
    return mapping


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise RuntimeError(f"{label} must be a lowercase SHA256 hex string.")
    try:
        parsed = bytes.fromhex(value)
    except ValueError as exc:
        raise RuntimeError(f"{label} must be a lowercase SHA256 hex string.") from exc
    if len(parsed) != 32 or value != value.lower():
        raise RuntimeError(f"{label} must be a lowercase SHA256 hex string.")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise RuntimeError(
            f"{label} must be a non-empty string without surrounding whitespace."
        )
    return value


def _load_self_hashed_json(
    path: str | os.PathLike[str], *, label: str
) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    manifest = _require_mapping(value, label)
    claimed = _require_sha256(
        manifest.get("manifest_sha256"), f"{label}.manifest_sha256"
    )
    unhashed = dict(manifest)
    unhashed.pop("manifest_sha256", None)
    actual = canonical_json_sha256(unhashed)
    if claimed != actual:
        raise RuntimeError(
            f"{label} hash mismatch for {path}: expected {claimed}, got {actual}."
        )
    return manifest


def _safe_relative_file(root: Path, value: Any, *, label: str) -> Path:
    relative = Path(_require_nonempty_string(value, label))
    if relative.is_absolute():
        raise RuntimeError(f"{label} must be relative to {root}, got {relative}.")
    root = root.resolve()
    candidate = root / relative
    if candidate.is_symlink():
        raise RuntimeError(f"{label} must not be a symlink: {candidate}.")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"{label} escapes its manifest directory: {relative}."
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash tensor dtype, shape, and exact contiguous CPU bytes."""

    value = tensor.detach().contiguous().cpu()
    header = canonical_json_bytes(
        {"dtype": _dtype_name(value.dtype), "shape": list(value.shape)}
    )
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, byteorder="big", signed=False))
    digest.update(header)
    digest.update(value.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _tensor_description(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": _dtype_name(tensor.dtype),
        "sha256": tensor_sha256(tensor),
    }


def _validate_prompt_tensors(
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    *,
    label: str,
    require_padding: bool,
    check_values: bool = True,
) -> dict[str, Any]:
    if tuple(prompt_embeds.shape) != (512, 4096):
        raise ValueError(
            f"{label}.prompt_embeds must be [512,4096], got "
            f"{tuple(prompt_embeds.shape)}."
        )
    if tuple(prompt_mask.shape) != (512,):
        raise ValueError(
            f"{label}.prompt_mask must be [512], got {tuple(prompt_mask.shape)}."
        )
    if prompt_embeds.dtype != torch.bfloat16:
        raise ValueError(
            f"{label}.prompt_embeds must be bfloat16, got {prompt_embeds.dtype}."
        )
    if prompt_mask.dtype != torch.bool:
        raise ValueError(f"{label}.prompt_mask must be bool, got {prompt_mask.dtype}.")
    if check_values and not torch.isfinite(prompt_embeds).all():
        raise ValueError(f"{label}.prompt_embeds contains non-finite values.")

    valid_tokens = int(prompt_mask.sum().item())
    if valid_tokens <= 0:
        raise ValueError(f"{label}.prompt_mask contains no valid tokens.")
    expected_mask = torch.arange(512, dtype=torch.long) < valid_tokens
    if not torch.equal(prompt_mask.cpu(), expected_mask):
        raise ValueError(
            f"{label}.prompt_mask must be one contiguous valid prefix followed by padding."
        )
    if require_padding and valid_tokens == 512:
        raise ValueError(f"{label}.prompt_mask must contain right-padding tokens.")

    padding = prompt_embeds[~prompt_mask] if check_values else None
    if (
        padding is not None
        and padding.numel()
        and torch.count_nonzero(padding).item() != 0
    ):
        raise ValueError(f"{label}.prompt_embeds padding positions must be exact zero.")
    return {
        "prompt_valid_tokens": valid_tokens,
        "padding_sha256": tensor_sha256(padding) if padding is not None else None,
    }


def validate_stage2_cache_tensors(
    tensors: Mapping[str, torch.Tensor],
    *,
    expected_spatial_shapes: Iterable[tuple[int, int]] = ((30, 52), (52, 30)),
    label: str = "cache",
    full_audit: bool = True,
) -> dict[str, Any]:
    """Validate one Stage-2 F25 cache artifact and return audit facts."""

    actual_keys = set(tensors)
    expected_keys = set(_CACHE_TENSOR_DTYPES)
    if actual_keys != expected_keys:
        raise ValueError(
            f"{label} tensor keys mismatch: expected {sorted(expected_keys)}, "
            f"got {sorted(actual_keys)}."
        )
    for name, expected_dtype in _CACHE_TENSOR_DTYPES.items():
        tensor = tensors[name]
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"{label}.{name} must have dtype {expected_dtype}, got {tensor.dtype}."
            )
        if (
            full_audit
            and tensor.is_floating_point()
            and not torch.isfinite(tensor).all()
        ):
            raise ValueError(f"{label}.{name} contains non-finite values.")

    video = tensors["video_latent"]
    initial = tensors["initial_latent"]
    if video.ndim == 4 and tuple(video.shape[:2]) == (24, 48):
        raise ValueError(
            f"{label}.video_latent is an official Stage-1 F24 cache: slot 0 is "
            "the cached sink, so it contains only 23 new future targets and is "
            "missing Stage-2 future frame 24. Generate a separate Stage-2 F25 "
            "cache before audit; this gate never fabricates or drops frames."
        )
    if video.ndim != 4 or tuple(video.shape[:2]) != (25, 48):
        raise ValueError(
            f"{label}.video_latent must be exactly [25,48,H,W] (F25), got "
            f"{tuple(video.shape)}."
        )
    allowed = {tuple(map(int, shape)) for shape in expected_spatial_shapes}
    spatial_shape = tuple(int(value) for value in video.shape[-2:])
    if spatial_shape not in allowed:
        raise ValueError(
            f"{label}.video_latent spatial shape {spatial_shape} is not one of "
            f"{sorted(allowed)}."
        )
    if tuple(initial.shape) != (1, 48, *spatial_shape):
        raise ValueError(
            f"{label}.initial_latent must be exactly [1,48,H,W] and match video; "
            f"got {tuple(initial.shape)}."
        )
    prompt_facts = _validate_prompt_tensors(
        tensors["prompt_embeds"],
        tensors["prompt_mask"],
        label=label,
        require_padding=False,
        check_values=full_audit,
    )
    facts: dict[str, Any] = {
        "latent_spatial_shape": list(spatial_shape),
        "real_future_shape": list(video[1:25].shape),
        "prompt_valid_tokens": prompt_facts["prompt_valid_tokens"],
        "tensors": {},
    }
    if not full_audit:
        facts["tensors"] = {
            name: {
                "shape": list(tensors[name].shape),
                "dtype": _dtype_name(tensors[name].dtype),
            }
            for name in sorted(tensors)
        }
        return facts

    difference = initial.float() - video[0:1].float()
    num_different = int(torch.count_nonzero(difference).item())
    facts.update(
        prompt_padding_sha256=prompt_facts["padding_sha256"],
        initial_vs_video0={
            "exact_equal": num_different == 0,
            "num_different": num_different,
            "max_abs_diff": float(difference.abs().max().item()),
            "mean_abs_diff": float(difference.abs().mean().item()),
        },
        tensors={name: _tensor_description(tensors[name]) for name in sorted(tensors)},
    )
    return facts


def _load_verified_safetensors_snapshot(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> tuple[bytes, dict[str, torch.Tensor]]:
    """Verify and parse one immutable byte snapshot of a cache artifact."""

    if type(expected_size) is not int or expected_size <= 0:
        raise RuntimeError(f"{label} expected_size must be a positive integer.")
    expected_sha256 = _require_sha256(expected_sha256, f"{label} expected_sha256")
    payload = path.read_bytes()
    if len(payload) != expected_size:
        raise RuntimeError(
            f"{label} size mismatch: expected {expected_size}, got {len(payload)}."
        )
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{label} hash mismatch: expected {expected_sha256}, got {actual_sha256}."
        )
    from safetensors.torch import load

    return payload, load(payload)


def _load_verified_safetensors_bytes(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    label: str,
) -> dict[str, torch.Tensor]:
    _, tensors = _load_verified_safetensors_snapshot(
        path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        label=label,
    )
    return tensors


def _validate_stage1_record_metadata_snapshot(
    payload: bytes,
    *,
    row_id: int,
    row_sha256: str,
    source_aggregate_sha256: str,
) -> None:
    if len(payload) < 8:
        raise RuntimeError(f"Stage-1 source row {row_id} has a truncated header.")
    header_size = int.from_bytes(payload[:8], byteorder="little", signed=False)
    if header_size <= 0 or 8 + header_size > len(payload):
        raise RuntimeError(f"Stage-1 source row {row_id} has an invalid header.")
    try:
        header = json.loads(payload[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Stage-1 source row {row_id} has invalid metadata JSON."
        ) from exc
    expected = {
        "schema": "longlive_stage1_i2v_cache_record",
        "schema_version": "1",
        "row_id": str(row_id),
        "row_sha256": row_sha256,
        "source_aggregate_sha256": source_aggregate_sha256,
        "prompt_mode": "repeat_global",
    }
    metadata = header.get("__metadata__")
    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"Stage-1 source row {row_id} has no provenance metadata.")
    for key, expected_value in expected.items():
        if metadata.get(key) != expected_value:
            raise RuntimeError(f"Stage-1 source row {row_id} metadata {key} mismatch.")


def _source_model_hashes(source_manifest: Mapping[str, Any]) -> tuple[str, str]:
    fingerprint = _require_mapping(
        source_manifest.get("source_fingerprint"),
        "source cache manifest.source_fingerprint",
    )
    models = _require_mapping(
        fingerprint.get("models"), "source cache manifest.source_fingerprint.models"
    )
    try:
        t5 = _require_mapping(models["t5_checkpoint"], "positive t5_checkpoint")
        tokenizer = _require_mapping(models["tokenizer_dir"], "positive tokenizer_dir")
    except KeyError as exc:
        raise RuntimeError(
            "Source cache manifest must fingerprint positive-cache models under "
            "source_fingerprint.models.t5_checkpoint and tokenizer_dir."
        ) from exc
    t5_hash = _require_sha256(
        t5.get("aggregate_sha256"), "positive t5 checkpoint aggregate hash"
    )
    tokenizer_hash = _require_sha256(
        tokenizer.get("aggregate_sha256"),
        "positive tokenizer aggregate hash",
    )
    return t5_hash, tokenizer_hash


def _source_text_encoding_contract(
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the positive cache's exact, self-hashed text-encoding contract."""

    fingerprint = _require_mapping(
        source_manifest.get("source_fingerprint"),
        "source cache manifest.source_fingerprint",
    )
    t5_hash, tokenizer_hash = _source_model_hashes(source_manifest)
    contract = _require_exact_keys(
        fingerprint.get("text_encoding"),
        label="source cache manifest.source_fingerprint.text_encoding",
        expected=_NEGATIVE_ENCODER_KEYS,
    )
    expected = {
        "t5_checkpoint_aggregate_sha256": t5_hash,
        "tokenizer_aggregate_sha256": tokenizer_hash,
        "tokenizer_revision": f"local-tree-sha256:{tokenizer_hash}",
        **_LOCKED_TEXT_ENCODING_SETTINGS,
    }
    for key, expected_value in expected.items():
        if contract[key] != expected_value or type(contract[key]) is not type(
            expected_value
        ):
            raise RuntimeError(
                "Positive cache text encoding contract mismatch for "
                f"{key}: expected {expected_value!r}, got {contract[key]!r}."
            )
    return dict(expected)


def _source_encoder_hashes(source_manifest: Mapping[str, Any]) -> tuple[str, str]:
    contract = _source_text_encoding_contract(source_manifest)
    return (
        contract["t5_checkpoint_aggregate_sha256"],
        contract["tokenizer_aggregate_sha256"],
    )


def _validate_locked_tokenizer_runtime_audit(
    value: Any, *, label: str
) -> dict[str, Any]:
    audit = _require_exact_keys(
        value,
        label=label,
        expected=set(_LOCKED_TOKENIZER_RUNTIME_AUDIT),
    )
    for key, expected in _LOCKED_TOKENIZER_RUNTIME_AUDIT.items():
        if audit[key] != expected or type(audit[key]) is not type(expected):
            raise RuntimeError(
                f"{label}.{key} differs from the locked Wan tokenizer contract."
            )
    return dict(audit)


def _validate_source_fingerprint_aggregate(
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    fingerprint = _require_mapping(
        manifest.get("source_fingerprint"),
        "source cache manifest.source_fingerprint",
    )
    fingerprint_hash = _require_sha256(
        fingerprint.get("aggregate_sha256"),
        "source cache fingerprint aggregate_sha256",
    )
    fingerprint_payload = dict(fingerprint)
    fingerprint_payload.pop("aggregate_sha256")
    actual_fingerprint_hash = canonical_json_sha256(fingerprint_payload)
    if actual_fingerprint_hash != fingerprint_hash:
        raise RuntimeError(
            "Source cache fingerprint aggregate hash mismatch: "
            f"expected {fingerprint_hash}, got {actual_fingerprint_hash}."
        )
    return fingerprint, fingerprint_hash


def _stage1_manifest_declares_reusable_f25_policy(
    manifest: Mapping[str, Any],
) -> bool:
    """Return whether a legacy producer explicitly proves the locked 97-frame path."""

    if manifest.get("schema") != _STAGE1_SOURCE_CACHE_SCHEMA:
        return False
    fingerprint = _require_mapping(
        manifest.get("source_fingerprint"), "source cache manifest.source_fingerprint"
    )
    preprocessing = fingerprint.get("preprocessing")
    # Missing or legacy/malformed policy declarations are not proof that an
    # actual F25 artifact came from frames 0..96.  They therefore select the
    # full-bitwise reverification path; the separate source/model provenance
    # gates still reject a fingerprint that cannot be trusted at all.
    if not isinstance(preprocessing, Mapping):
        return False
    required = {
        "min_source_frames": 97,
        "selected_frame_start": 0,
        "selected_frame_count": 97,
        "expected_fps": 24,
        "allow_resize": False,
        "allow_padding": False,
        "dtype": "bfloat16",
    }
    return all(
        key in preprocessing
        and preprocessing[key] == expected
        and type(preprocessing[key]) is type(expected)
        for key, expected in required.items()
    )


def _validate_f25_source_preparation(
    manifest: Mapping[str, Any], *, expected_num_samples: int
) -> None:
    """Validate the immutable Stage-1 -> native Stage-2 F25 provenance chain."""

    allowed_manifest_keys = set(_F25_BASE_MANIFEST_KEYS)
    if STAGE2_TEXT_ENCODING_UPGRADE_KEY in manifest:
        allowed_manifest_keys.add(STAGE2_TEXT_ENCODING_UPGRADE_KEY)
    _require_exact_keys(
        manifest,
        label="Stage-2 F25 source manifest",
        expected=allowed_manifest_keys,
    )
    preparation = _require_exact_keys(
        manifest.get("preparation"),
        label="Stage-2 F25 preparation",
        expected=_F25_PREPARATION_KEYS,
    )
    if (
        preparation["schema"] != STAGE2_F25_PREPARATION_SCHEMA
        or type(preparation["schema_version"]) is not int
        or preparation["schema_version"] != STAGE2_F25_PREPARATION_SCHEMA_VERSION
    ):
        raise RuntimeError("Unsupported Stage-2 F25 preparation schema.")

    frame_policy = _require_mapping(
        preparation["frame_policy"], "Stage-2 F25 preparation.frame_policy"
    )
    if frame_policy != STAGE2_F25_FRAME_POLICY:
        raise RuntimeError("Stage-2 F25 preparation frame policy is not locked 0..96.")
    if preparation["frame_policy_sha256"] != STAGE2_F25_FRAME_POLICY_SHA256:
        raise RuntimeError("Stage-2 F25 preparation frame-policy hash mismatch.")

    input_binding = _require_exact_keys(
        preparation["input_manifest"],
        label="Stage-2 F25 preparation.input_manifest",
        expected=_F25_INPUT_MANIFEST_KEYS,
    )
    input_path = Path(
        _require_nonempty_string(input_binding["path"], "F25 input manifest path")
    ).expanduser()
    if not input_path.is_absolute():
        raise RuntimeError("Stage-2 F25 input manifest path must be absolute.")
    input_path = input_path.resolve()
    input_file_sha256 = _require_sha256(
        input_binding["file_sha256"], "F25 input manifest file SHA256"
    )
    if sha256_file(input_path) != input_file_sha256:
        raise RuntimeError("Stage-2 F25 input manifest file changed after preparation.")
    input_manifest = _load_self_hashed_json(
        input_path, label="Stage-2 F25 input source manifest"
    )
    if input_manifest.get("schema") != _STAGE1_SOURCE_CACHE_SCHEMA:
        raise RuntimeError(
            "Stage-2 F25 preparation must point to the original Stage-1 source schema."
        )
    if STAGE2_TEXT_ENCODING_UPGRADE_KEY in input_manifest or "text_encoding" in (
        input_manifest.get("source_fingerprint") or {}
    ):
        raise RuntimeError(
            "Prepare F25 from the immutable pre-upgrade Stage-1 manifest; apply the "
            "text-encoding attestation to the native F25 manifest afterwards."
        )
    input_fingerprint, input_fingerprint_sha256 = _validate_source_manifest_base(
        input_manifest, expected_num_samples=expected_num_samples
    )
    expected_input_values = {
        "schema": _STAGE1_SOURCE_CACHE_SCHEMA,
        "schema_version": STAGE1_CACHE_SCHEMA_VERSION,
        "manifest_sha256": input_manifest["manifest_sha256"],
        "source_fingerprint_sha256": input_fingerprint_sha256,
    }
    for key, expected in expected_input_values.items():
        if input_binding[key] != expected or type(input_binding[key]) is not type(
            expected
        ):
            raise RuntimeError(
                f"Stage-2 F25 input manifest provenance mismatch for {key}."
            )

    producer = _require_required_keys(
        preparation["producer"],
        label="Stage-2 F25 preparation.producer",
        required=_F25_PRODUCER_KEYS,
    )
    if producer["file"] != "utils/stage2_f25_cache.py":
        raise RuntimeError("Unexpected Stage-2 F25 producer file.")
    producer_sha256 = _require_sha256(
        producer["file_sha256"], "F25 producer file SHA256"
    )
    producer_path = Path(__file__).resolve().parents[1] / producer["file"]
    if sha256_file(producer_path) != producer_sha256:
        raise RuntimeError("Stage-2 F25 producer file changed after preparation.")

    config = _require_exact_keys(
        preparation["config"],
        label="Stage-2 F25 preparation.config",
        expected=_F25_CONFIG_KEYS,
    )
    config_path = Path(
        _require_nonempty_string(config["path"], "F25 preparation config path")
    ).expanduser()
    if not config_path.is_absolute():
        raise RuntimeError("Stage-2 F25 config path must be absolute.")
    _require_sha256(config["file_sha256"], "F25 preparation config file SHA256")
    _require_sha256(config["contract_sha256"], "F25 config contract SHA256")
    _require_sha256(config["launch_sha256"], "F25 config launch SHA256")
    _require_sha256(config["data_contract_sha256"], "F25 data contract SHA256")

    metadata = _require_exact_keys(
        preparation["metadata"],
        label="Stage-2 F25 preparation.metadata",
        expected=_F25_METADATA_KEYS,
    )
    metadata_path = Path(
        _require_nonempty_string(metadata["path"], "F25 preparation metadata path")
    ).expanduser()
    if not metadata_path.is_absolute():
        raise RuntimeError("Stage-2 F25 metadata path must be absolute.")
    if sha256_file(metadata_path.resolve()) != _require_sha256(
        metadata["file_sha256"], "F25 preparation metadata file SHA256"
    ):
        raise RuntimeError("Stage-2 F25 metadata changed after preparation.")

    vae = _require_exact_keys(
        preparation["vae"],
        label="Stage-2 F25 preparation.vae",
        expected=_F25_VAE_KEYS,
    )
    vae_sha256 = _require_sha256(vae["aggregate_sha256"], "F25 VAE aggregate SHA256")
    models = _require_mapping(
        input_fingerprint.get("models"), "F25 input source models"
    )
    for model_key in ("vae_checkpoint", "t5_checkpoint", "tokenizer_dir"):
        model_entry = _require_mapping(
            models.get(model_key), f"F25 input source model {model_key}"
        )
        files = model_entry.get("files")
        if not isinstance(files, list) or not files:
            raise RuntimeError(
                f"F25 input source model {model_key} must include its file list."
            )
        aggregate = _require_sha256(
            model_entry.get("aggregate_sha256"),
            f"F25 input source model {model_key} aggregate SHA256",
        )
        if aggregate_file_hash(files) != aggregate:
            raise RuntimeError(
                f"F25 input source model {model_key} file-list aggregate mismatch."
            )
    input_vae = _require_mapping(models.get("vae_checkpoint"), "F25 input VAE model")
    if input_vae.get("aggregate_sha256") != vae_sha256:
        raise RuntimeError("Stage-2 F25 VAE hash differs from the Stage-1 source VAE.")

    output_policy = _require_exact_keys(
        preparation["output_policy"],
        label="Stage-2 F25 preparation.output_policy",
        expected=_F25_OUTPUT_POLICY_KEYS,
    )
    expected_output_policy = {
        "independent_output_directory": True,
        "in_place_overwrite": False,
        "atomic_artifacts": True,
        "atomic_completion_sidecars": True,
        "stable_row_modulo_sharding": True,
    }
    if output_policy != expected_output_policy:
        raise RuntimeError("Stage-2 F25 output safety policy mismatch.")

    contract_payload = {
        key: copy.deepcopy(preparation[key])
        for key in (
            "schema",
            "schema_version",
            "input_manifest",
            "producer",
            "config",
            "metadata",
            "frame_policy",
            "frame_policy_sha256",
            "vae",
            "output_policy",
        )
    }
    contract_sha256 = _require_sha256(
        preparation["contract_sha256"], "F25 preparation contract SHA256"
    )
    if canonical_json_sha256(contract_payload) != contract_sha256:
        raise RuntimeError("Stage-2 F25 preparation contract hash mismatch.")

    summary = _require_exact_keys(
        preparation["summary"],
        label="Stage-2 F25 preparation.summary",
        expected=_F25_SUMMARY_KEYS,
    )
    for key, value in summary.items():
        if type(value) is not int or value < 0:
            raise RuntimeError(f"Stage-2 F25 preparation.summary.{key} is invalid.")
    if sum(summary.values()) != expected_num_samples:
        raise RuntimeError("Stage-2 F25 preparation summary does not cover every row.")

    source_fingerprint = _require_mapping(
        manifest["source_fingerprint"], "Stage-2 F25 source fingerprint"
    )
    expected_fingerprint = copy.deepcopy(input_fingerprint)
    expected_fingerprint.pop("aggregate_sha256")
    expected_fingerprint["preprocessing"] = copy.deepcopy(STAGE2_F25_FRAME_POLICY)
    expected_fingerprint["cache_schema"] = STAGE2_F25_SOURCE_CACHE_SCHEMA
    expected_fingerprint["cache_schema_version"] = (
        STAGE2_F25_SOURCE_CACHE_SCHEMA_VERSION
    )
    expected_fingerprint["preparation_contract_sha256"] = contract_sha256
    if STAGE2_TEXT_ENCODING_UPGRADE_KEY in manifest:
        expected_fingerprint["text_encoding"] = copy.deepcopy(
            _require_mapping(
                source_fingerprint.get("text_encoding"),
                "upgraded Stage-2 F25 text encoding contract",
            )
        )
    elif "text_encoding" in source_fingerprint:
        raise RuntimeError(
            "An F25 base manifest cannot add text encoding without the append-only "
            "attestation block."
        )
    expected_fingerprint["aggregate_sha256"] = canonical_json_sha256(
        expected_fingerprint
    )
    if source_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "Stage-2 F25 source fingerprint is not the unique derivation of its input."
        )

    input_records = list(input_manifest["records"])
    source_records = _require_mapping(
        {item["row_id"]: item for item in input_fingerprint.get("records", [])},
        "F25 input fingerprint records",
    )
    observed_summary = {key: 0 for key in _F25_SUMMARY_KEYS}
    for row_id, raw_entry in enumerate(manifest["records"]):
        entry = _require_mapping(raw_entry, f"Stage-2 F25 source record {row_id}")
        missing = sorted(_F25_RECORD_REQUIRED_KEYS - set(entry))
        allowed = set(_F25_RECORD_REQUIRED_KEYS) | {"action_id"}
        extra = sorted(set(entry) - allowed)
        if missing or extra:
            raise RuntimeError(
                f"Stage-2 F25 source record {row_id} keys mismatch: "
                f"missing={missing}, extra={extra}."
            )
        if type(entry["row_id"]) is not int or entry["row_id"] != row_id:
            raise RuntimeError("Stage-2 F25 source row ids must be contiguous.")
        input_entry = input_records[row_id]
        if entry["row_sha256"] != input_entry.get("row_sha256"):
            raise RuntimeError(f"Stage-2 F25 row {row_id} row hash changed.")
        input_artifact = _require_exact_keys(
            entry["input_artifact"],
            label=f"Stage-2 F25 row {row_id}.input_artifact",
            expected=_F25_INPUT_ARTIFACT_KEYS,
        )
        for key in ("path", "size", "sha256"):
            if input_artifact[key] != input_entry.get(key):
                raise RuntimeError(
                    f"Stage-2 F25 row {row_id} input artifact {key} changed."
                )
        input_artifact_path = _safe_relative_file(
            input_path.parent,
            input_entry.get("path"),
            label=f"Stage-2 F25 input row {row_id}.path",
        )
        input_payload, actual_input_tensors = _load_verified_safetensors_snapshot(
            input_artifact_path,
            expected_size=input_entry["size"],
            expected_sha256=input_entry["sha256"],
            label=f"Stage-2 F25 input artifact row {row_id}",
        )
        _validate_stage1_record_metadata_snapshot(
            input_payload,
            row_id=row_id,
            row_sha256=input_entry["row_sha256"],
            source_aggregate_sha256=input_fingerprint_sha256,
        )
        actual_input_descriptions = {
            name: _tensor_description(actual_input_tensors[name])
            for name in sorted(actual_input_tensors)
        }
        if input_artifact["tensors"] != actual_input_descriptions:
            raise RuntimeError(
                f"Stage-2 F25 row {row_id} input tensor snapshot mismatch."
            )
        _check_source_tensor_description(
            input_entry, actual_input_descriptions, row_id=row_id
        )
        decision = entry["decision"]
        if decision not in observed_summary:
            raise RuntimeError(f"Stage-2 F25 row {row_id} decision is invalid.")
        observed_summary[decision] += 1
        input_video = _require_mapping(
            input_artifact["tensors"], f"Stage-2 F25 row {row_id} input tensors"
        ).get("video_latent")
        input_video = _require_mapping(
            input_video, f"Stage-2 F25 row {row_id} input video_latent"
        )
        input_shape = input_video.get("shape")
        if decision == "reencoded_f24" and (
            not isinstance(input_shape, list) or input_shape[:2] != [24, 48]
        ):
            raise RuntimeError(f"Stage-2 F25 row {row_id} was not an F24 input.")
        if decision in {"reused_f25", "reverified_f25"} and (
            not isinstance(input_shape, list) or input_shape[:2] != [25, 48]
        ):
            raise RuntimeError(f"Stage-2 F25 row {row_id} was not an F25 input.")
        if (
            decision == "reused_f25"
            and not _stage1_manifest_declares_reusable_f25_policy(input_manifest)
        ):
            raise RuntimeError(
                f"Stage-2 F25 row {row_id} was reused without 97-frame provenance."
            )
        if (
            decision == "reverified_f25"
            and _stage1_manifest_declares_reusable_f25_policy(input_manifest)
        ):
            raise RuntimeError(
                f"Stage-2 F25 row {row_id} was redundantly marked reverified."
            )
        output_descriptions = _require_mapping(
            entry["tensors"], f"Stage-2 F25 row {row_id} output tensors"
        )
        verification = _require_exact_keys(
            entry["verification"],
            label=f"Stage-2 F25 row {row_id}.verification",
            expected=_F25_RECORD_VERIFICATION_KEYS,
        )
        expected_verification = {
            "reused_f25": {
                "output_reuses_input_bytes": True,
                "f24_prefix_exact": None,
                "f25_reverification_exact": None,
            },
            "reverified_f25": {
                "output_reuses_input_bytes": True,
                "f24_prefix_exact": None,
                "f25_reverification_exact": True,
            },
            "reencoded_f24": {
                "output_reuses_input_bytes": False,
                "f24_prefix_exact": True,
                "f25_reverification_exact": None,
            },
        }[decision]
        if verification != expected_verification:
            raise RuntimeError(
                f"Stage-2 F25 row {row_id} verification evidence mismatch."
            )
        if decision in {"reused_f25", "reverified_f25"}:
            if (
                entry["size"] != input_entry["size"]
                or entry["sha256"] != input_entry["sha256"]
                or output_descriptions != actual_input_descriptions
            ):
                raise RuntimeError(
                    f"Stage-2 F25 row {row_id} did not preserve verified input bytes."
                )
        else:
            for name in ("initial_latent", "prompt_embeds", "prompt_mask"):
                if output_descriptions.get(name) != actual_input_descriptions.get(name):
                    raise RuntimeError(
                        f"Stage-2 F25 row {row_id} changed copied tensor {name}."
                    )
        fingerprint_record = source_records.get(row_id)
        if not isinstance(fingerprint_record, Mapping):
            raise RuntimeError(f"F25 input fingerprint is missing row {row_id}.")
        expected_hashes = {
            "source_video_sha256": fingerprint_record.get("video_sha256"),
            "source_input_image_sha256": fingerprint_record.get("input_image_sha256"),
            "vae_aggregate_sha256": vae_sha256,
            "frame_policy_sha256": STAGE2_F25_FRAME_POLICY_SHA256,
            "preparation_contract_sha256": contract_sha256,
        }
        for key, expected in expected_hashes.items():
            if entry[key] != expected:
                raise RuntimeError(f"Stage-2 F25 row {row_id}.{key} mismatch.")
        if entry["tensors"].get("video_latent", {}).get("shape", [])[:2] != [25, 48]:
            raise RuntimeError(f"Stage-2 F25 row {row_id} output is not F25.")
        if "action_id" in input_entry:
            if entry.get("action_id") != input_entry["action_id"]:
                raise RuntimeError(f"Stage-2 F25 row {row_id} action_id changed.")
        elif "action_id" in entry:
            raise RuntimeError(f"Stage-2 F25 row {row_id} invented an action_id.")
    if observed_summary != summary:
        raise RuntimeError(
            "Stage-2 F25 decision counts differ from preparation summary."
        )


def _validate_source_manifest_base(
    manifest: Mapping[str, Any], *, expected_num_samples: int
) -> tuple[dict[str, Any], str]:
    if type(expected_num_samples) is not int or expected_num_samples <= 0:
        raise ValueError("expected_num_samples must be a positive integer.")
    schema = manifest.get("schema")
    schema_version = manifest.get("schema_version")
    supported_schema = (
        schema == _STAGE1_SOURCE_CACHE_SCHEMA
        and type(schema_version) is int
        and schema_version == STAGE1_CACHE_SCHEMA_VERSION
    ) or (
        schema == STAGE2_F25_SOURCE_CACHE_SCHEMA
        and type(schema_version) is int
        and schema_version == STAGE2_F25_SOURCE_CACHE_SCHEMA_VERSION
    )
    if not supported_schema:
        raise RuntimeError(
            "Unsupported source cache manifest schema; Stage-2 accepts the official "
            "Stage-1 migration input or the native Stage-2 F25 source schema."
        )
    records = manifest.get("records")
    if not isinstance(records, list):
        raise RuntimeError("Source cache manifest.records must be a list.")
    if len(records) != int(expected_num_samples):
        raise RuntimeError(
            f"Expected {expected_num_samples} source-cache records, found {len(records)}."
        )
    if type(manifest.get("num_samples")) is not int or manifest.get(
        "num_samples"
    ) != len(records):
        raise RuntimeError("Source cache manifest num_samples does not match records.")
    fingerprint, fingerprint_hash = _validate_source_fingerprint_aggregate(manifest)
    expected_fingerprint_schema_version = (
        STAGE1_CACHE_SCHEMA_VERSION
        if schema == _STAGE1_SOURCE_CACHE_SCHEMA
        else STAGE2_F25_SOURCE_CACHE_SCHEMA_VERSION
    )
    if (
        type(fingerprint.get("cache_schema_version")) is not int
        or fingerprint.get("cache_schema_version")
        != expected_fingerprint_schema_version
    ):
        raise RuntimeError("Source cache fingerprint schema version mismatch.")
    seen_paths: set[str] = set()
    for row_id, raw_entry in enumerate(records):
        entry = _require_mapping(raw_entry, f"source cache record {row_id}")
        if type(entry.get("row_id")) is not int or entry.get("row_id") != row_id:
            raise RuntimeError(
                "Source cache row ids must be contiguous and stable; "
                f"expected {row_id}, got {entry.get('row_id')!r}."
            )
        relative = _require_nonempty_string(
            entry.get("path"), f"source cache record {row_id}.path"
        )
        if relative in seen_paths:
            raise RuntimeError(
                f"Duplicate cache artifact path in source manifest: {relative}."
            )
        seen_paths.add(relative)
        _require_sha256(entry.get("sha256"), f"source cache record {row_id}.sha256")
        size = entry.get("size")
        if type(size) is not int or size <= 0:
            raise RuntimeError(
                f"source cache record {row_id}.size must be a positive integer."
            )
        _require_sha256(
            entry.get("row_sha256"), f"source cache record {row_id}.row_sha256"
        )
    if schema == STAGE2_F25_SOURCE_CACHE_SCHEMA:
        _validate_f25_source_preparation(
            manifest, expected_num_samples=expected_num_samples
        )
    return fingerprint, fingerprint_hash


def _validate_text_encoding_upgrade(
    source_manifest: Mapping[str, Any],
    *,
    expected_num_samples: int,
) -> dict[str, Any] | None:
    raw_upgrade = source_manifest.get(STAGE2_TEXT_ENCODING_UPGRADE_KEY)
    if raw_upgrade is None:
        return None
    upgrade = _require_exact_keys(
        raw_upgrade,
        label=STAGE2_TEXT_ENCODING_UPGRADE_KEY,
        expected=_TEXT_ENCODING_UPGRADE_KEYS,
    )
    if (
        upgrade["schema"] != STAGE2_TEXT_ENCODING_UPGRADE_SCHEMA
        or type(upgrade["schema_version"]) is not int
        or upgrade["schema_version"] != STAGE2_TEXT_ENCODING_UPGRADE_SCHEMA_VERSION
    ):
        raise RuntimeError("Unsupported Stage-2 text-encoding upgrade schema.")
    original = _require_exact_keys(
        upgrade["original_source_manifest"],
        label="text-encoding upgrade original_source_manifest",
        expected=_TEXT_ENCODING_UPGRADE_ORIGINAL_KEYS,
    )
    original_path = Path(
        _require_nonempty_string(original["path"], "legacy source manifest path")
    ).expanduser()
    if not original_path.is_absolute():
        raise RuntimeError("Legacy source manifest provenance path must be absolute.")
    original_path = original_path.resolve()
    original_manifest = _load_self_hashed_json(
        original_path, label="legacy source cache manifest"
    )
    original_manifest_hash = _require_sha256(
        original["manifest_sha256"], "legacy source manifest SHA256"
    )
    if original_manifest["manifest_sha256"] != original_manifest_hash:
        raise RuntimeError(
            "Legacy source manifest changed after text-encoding upgrade."
        )
    original_fingerprint, original_fingerprint_hash = _validate_source_manifest_base(
        original_manifest,
        expected_num_samples=int(expected_num_samples),
    )
    if "text_encoding" in original_fingerprint:
        raise RuntimeError(
            "Text-encoding upgrade provenance does not point to a legacy manifest."
        )
    if original["source_fingerprint_sha256"] != original_fingerprint_hash:
        raise RuntimeError(
            "Legacy source fingerprint hash differs from upgrade provenance."
        )

    verification = _require_required_keys(
        upgrade["verification"],
        label="text-encoding upgrade verification",
        required=_TEXT_ENCODING_UPGRADE_VERIFICATION_KEYS,
    )
    if verification["validator_file"] != "utils/wan_5b_wrapper.py":
        raise RuntimeError("Unexpected text-encoding validator file.")
    validator_path = (
        Path(__file__).resolve().parents[1] / verification["validator_file"]
    )
    validator_hash = _require_sha256(
        verification["validator_file_sha256"],
        "text-encoding validator file SHA256",
    )
    if sha256_file(validator_path) != validator_hash:
        raise RuntimeError("Text-encoding validator file changed after upgrade.")

    contract = _source_text_encoding_contract(source_manifest)
    contract_hash = _require_sha256(
        verification["text_encoding_contract_sha256"],
        "text-encoding contract SHA256",
    )
    if canonical_json_sha256(contract) != contract_hash:
        raise RuntimeError(
            "Text-encoding contract hash differs from upgrade provenance."
        )
    if (
        verification["t5_checkpoint_aggregate_sha256"]
        != contract["t5_checkpoint_aggregate_sha256"]
        or verification["tokenizer_aggregate_sha256"]
        != contract["tokenizer_aggregate_sha256"]
    ):
        raise RuntimeError("Text-encoding model hashes differ from upgrade provenance.")
    _validate_locked_tokenizer_runtime_audit(
        verification["tokenizer_runtime_audit"],
        label="text-encoding tokenizer runtime audit",
    )

    attestation = _require_exact_keys(
        upgrade["operator_attestation"],
        label="text-encoding operator attestation",
        expected=_TEXT_ENCODING_UPGRADE_ATTESTATION_KEYS,
    )
    _require_nonempty_string(attestation["operator_id"], "operator_id")
    if attestation["statement"] != STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION:
        raise RuntimeError("Text-encoding operator attestation statement mismatch.")

    # The upgrade is deliberately a unique, append-only transformation.  Rebuild
    # it from the still-present legacy manifest and compare canonical bytes so a
    # re-signed edit to a record, path, action label, or any other legacy field
    # cannot masquerade as an attested upgrade.
    expected_upgraded = copy.deepcopy(original_manifest)
    expected_upgraded.pop("manifest_sha256")
    expected_fingerprint = copy.deepcopy(original_fingerprint)
    expected_fingerprint.pop("aggregate_sha256")
    expected_fingerprint["text_encoding"] = contract
    expected_fingerprint["aggregate_sha256"] = canonical_json_sha256(
        expected_fingerprint
    )
    expected_upgraded["source_fingerprint"] = expected_fingerprint
    expected_upgraded[STAGE2_TEXT_ENCODING_UPGRADE_KEY] = copy.deepcopy(dict(upgrade))
    expected_upgraded["manifest_sha256"] = canonical_json_sha256(expected_upgraded)
    if canonical_json_bytes(source_manifest) != canonical_json_bytes(expected_upgraded):
        raise RuntimeError(
            "Source cache manifest differs from the unique append-only upgrade of "
            "its recorded legacy manifest."
        )
    return upgrade


def _validate_f25_success_and_materialized_artifacts(
    manifest: Mapping[str, Any], *, manifest_path: Path
) -> None:
    """Require the atomic producer marker and close input/output tensor identity."""

    upgrade = manifest.get(STAGE2_TEXT_ENCODING_UPGRADE_KEY)
    if upgrade is None:
        base_manifest_path = manifest_path
    else:
        try:
            base_manifest_path = (
                Path(upgrade["original_source_manifest"]["path"]).expanduser().resolve()
            )
        except (KeyError, TypeError) as exc:
            raise RuntimeError("F25 text upgrade has no base-manifest path.") from exc
    marker_path = base_manifest_path.parent / STAGE2_F25_SUCCESS_NAME
    marker = _load_self_hashed_json(marker_path, label="Stage-2 F25 success marker")
    expected_marker_keys = {
        "schema",
        "schema_version",
        "manifest_path",
        "manifest_file_sha256",
        "source_manifest_sha256",
        "preparation_contract_sha256",
        "num_samples",
        "manifest_sha256",
    }
    _require_exact_keys(
        marker,
        label="Stage-2 F25 success marker",
        expected=expected_marker_keys,
    )
    if (
        marker["schema"] != STAGE2_F25_SUCCESS_SCHEMA
        or marker["schema_version"] != STAGE2_F25_SUCCESS_SCHEMA_VERSION
    ):
        raise RuntimeError("Unsupported Stage-2 F25 success marker schema.")
    base_manifest = _load_self_hashed_json(
        base_manifest_path, label="Stage-2 F25 successful base manifest"
    )
    expected_marker = {
        "manifest_path": base_manifest_path.name,
        "manifest_file_sha256": sha256_file(base_manifest_path),
        "source_manifest_sha256": base_manifest["manifest_sha256"],
        "preparation_contract_sha256": manifest["preparation"]["contract_sha256"],
        "num_samples": manifest["num_samples"],
    }
    for key, expected in expected_marker.items():
        if marker[key] != expected or type(marker[key]) is not type(expected):
            raise RuntimeError(f"Stage-2 F25 success marker {key} mismatch.")
    if base_manifest.get("schema") != STAGE2_F25_SOURCE_CACHE_SCHEMA:
        raise RuntimeError(
            "Stage-2 F25 success marker does not bind a native F25 base."
        )
    if base_manifest["preparation"] != manifest["preparation"]:
        raise RuntimeError("F25 upgraded manifest changed the preparation provenance.")
    if base_manifest["records"] != manifest["records"]:
        raise RuntimeError("F25 upgraded manifest changed materialized records.")

    input_manifest_path = (
        Path(manifest["preparation"]["input_manifest"]["path"]).expanduser().resolve()
    )
    input_manifest = _load_self_hashed_json(
        input_manifest_path, label="F25 materialization input manifest"
    )
    input_records = input_manifest["records"]
    cache_root = base_manifest_path.parent
    input_root = input_manifest_path.parent
    for row_id, (entry, input_entry) in enumerate(
        zip(manifest["records"], input_records)
    ):
        output_path = _safe_relative_file(
            cache_root, entry["path"], label=f"F25 output row {row_id}.path"
        )
        output_tensors = _load_verified_safetensors_bytes(
            output_path,
            expected_size=entry["size"],
            expected_sha256=entry["sha256"],
            label=f"F25 materialized artifact row {row_id}",
        )
        output_facts = validate_stage2_cache_tensors(
            output_tensors, label=f"F25 materialized row {row_id}"
        )
        if entry["tensors"] != output_facts["tensors"]:
            raise RuntimeError(f"F25 row {row_id} output tensor snapshot mismatch.")
        input_path = _safe_relative_file(
            input_root,
            input_entry["path"],
            label=f"F25 original input row {row_id}.path",
        )
        input_tensors = _load_verified_safetensors_bytes(
            input_path,
            expected_size=input_entry["size"],
            expected_sha256=input_entry["sha256"],
            label=f"F25 original input artifact row {row_id}",
        )
        decision = entry["decision"]
        if decision in {"reused_f25", "reverified_f25"}:
            if (
                entry["size"] != input_entry["size"]
                or entry["sha256"] != input_entry["sha256"]
                or any(
                    not torch.equal(output_tensors[name], input_tensors[name])
                    for name in output_tensors
                )
            ):
                raise RuntimeError(f"F25 row {row_id} was not reused byte-for-byte.")
        elif decision == "reencoded_f24":
            if not torch.equal(
                output_tensors["video_latent"][:24], input_tensors["video_latent"]
            ):
                raise RuntimeError(f"F25 row {row_id} failed F24 prefix parity.")
            for name in ("initial_latent", "prompt_embeds", "prompt_mask"):
                if not torch.equal(output_tensors[name], input_tensors[name]):
                    raise RuntimeError(f"F25 row {row_id} changed copied {name}.")
        else:  # validated earlier; retain a local tripwire
            raise RuntimeError(f"F25 row {row_id} has an unsupported decision.")


def load_source_cache_manifest(
    path: str | os.PathLike[str],
    *,
    expected_num_samples: int,
    require_text_encoding_upgrade: bool = False,
) -> dict[str, Any]:
    """Load the producer manifest used to bind cache and text provenance."""

    if type(expected_num_samples) is not int or expected_num_samples <= 0:
        raise ValueError("expected_num_samples must be a positive integer.")
    manifest_path = Path(path).expanduser().resolve()
    manifest = _load_self_hashed_json(manifest_path, label="source cache manifest")
    _validate_source_manifest_base(manifest, expected_num_samples=expected_num_samples)
    if (
        require_text_encoding_upgrade
        and STAGE2_TEXT_ENCODING_UPGRADE_KEY not in manifest
    ):
        raise RuntimeError(
            "This command requires the attested output of upgrade-source-manifest; "
            "the legacy Stage-1 manifest must never be edited in place."
        )
    upgrade = _validate_text_encoding_upgrade(
        manifest,
        expected_num_samples=expected_num_samples,
    )
    if upgrade is None:
        # A native F25 base manifest intentionally has no text-encoding contract
        # until upgrade-source-manifest adds the append-only attestation block.
        # Its positive encoder assets are still bound by their aggregate hashes.
        _source_model_hashes(manifest)
    else:
        _source_encoder_hashes(manifest)
    if require_text_encoding_upgrade and upgrade is None:  # defensive tripwire
        raise AssertionError("required text-encoding upgrade validation was skipped")
    if manifest.get("schema") == STAGE2_F25_SOURCE_CACHE_SCHEMA:
        _validate_f25_success_and_materialized_artifacts(
            manifest, manifest_path=manifest_path
        )
    return manifest


def load_f25_preparation_input_manifest(
    path: str | os.PathLike[str], *, expected_num_samples: int
) -> dict[str, Any]:
    """Load the immutable pre-upgrade Stage-1 manifest used for F25 migration."""

    manifest = _load_self_hashed_json(path, label="Stage-2 F25 migration input")
    _validate_source_manifest_base(
        manifest, expected_num_samples=int(expected_num_samples)
    )
    if manifest.get("schema") != _STAGE1_SOURCE_CACHE_SCHEMA:
        raise RuntimeError(
            "F25 migration input must be the original Stage-1 cache manifest."
        )
    fingerprint = _require_mapping(
        manifest.get("source_fingerprint"), "F25 migration input source fingerprint"
    )
    if "text_encoding" in fingerprint or STAGE2_TEXT_ENCODING_UPGRADE_KEY in manifest:
        raise RuntimeError(
            "Use the immutable pre-upgrade Stage-1 manifest for F25 migration; "
            "run upgrade-source-manifest on the native F25 output afterwards."
        )
    return manifest


def upgrade_legacy_source_cache_manifest_text_encoding(
    legacy_manifest_path: str | os.PathLike[str],
    output_manifest_path: str | os.PathLike[str],
    *,
    expected_source_manifest_sha256: str,
    t5_checkpoint_path: str | os.PathLike[str],
    tokenizer_dir: str | os.PathLike[str],
    operator_id: str,
    operator_attestation: str,
    expected_num_samples: int = 600,
) -> Path:
    """Create an attested Stage-2 provenance wrapper for a legacy manifest.

    The original source manifest and all 600 cache artifacts remain byte-for-
    byte untouched.  Only a new, independently self-hashed manifest is written.
    """

    legacy_input = Path(legacy_manifest_path).expanduser()
    output_input = Path(output_manifest_path).expanduser()
    if legacy_input.is_symlink():
        raise RuntimeError("Legacy source manifest must not be a symlink.")
    legacy_path = legacy_input.resolve()
    output_path = output_input.resolve()
    if legacy_path == output_path:
        raise ValueError(
            "The upgraded manifest must not overwrite the legacy manifest."
        )
    if output_input.is_symlink() or output_path.exists():
        raise FileExistsError(
            "The upgraded manifest output must be a new, non-symlink path: "
            f"{output_path}"
        )
    expected_source_manifest_sha256 = _require_sha256(
        expected_source_manifest_sha256,
        "expected_source_manifest_sha256",
    )
    operator_id = _require_nonempty_string(operator_id, "operator_id")
    if operator_attestation != STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION:
        raise RuntimeError(
            "operator_attestation must exactly match the locked Stage-2 statement."
        )

    original_bytes_hash = sha256_file(legacy_path)
    legacy = _load_self_hashed_json(legacy_path, label="legacy source cache manifest")
    if legacy["manifest_sha256"] != expected_source_manifest_sha256:
        raise RuntimeError(
            "Legacy source manifest SHA does not match the operator-supplied "
            "expected SHA."
        )
    if type(expected_num_samples) is not int or expected_num_samples <= 0:
        raise ValueError("expected_num_samples must be a positive integer.")
    fingerprint, fingerprint_hash = _validate_source_manifest_base(
        legacy, expected_num_samples=expected_num_samples
    )
    if legacy.get("schema") == STAGE2_F25_SOURCE_CACHE_SCHEMA:
        _validate_f25_success_and_materialized_artifacts(
            legacy, manifest_path=legacy_path
        )
    if "text_encoding" in fingerprint or STAGE2_TEXT_ENCODING_UPGRADE_KEY in legacy:
        raise RuntimeError(
            "upgrade-source-manifest accepts only an unmodified pre-attestation "
            "Stage-1 input or native Stage-2 F25 base manifest."
        )
    expected_t5_hash, expected_tokenizer_hash = _source_model_hashes(legacy)
    actual_t5_hash = aggregate_file_hash(tree_file_hashes(t5_checkpoint_path))
    actual_tokenizer_hash = aggregate_file_hash(tree_file_hashes(tokenizer_dir))
    if actual_t5_hash != expected_t5_hash:
        raise RuntimeError(
            "T5 checkpoint tree differs from the legacy positive-cache fingerprint."
        )
    if actual_tokenizer_hash != expected_tokenizer_hash:
        raise RuntimeError(
            "Tokenizer tree differs from the legacy positive-cache fingerprint."
        )

    from utils import wan_5b_wrapper

    runtime_audit = wan_5b_wrapper.audit_wan_text_encoding_tokenizer_contract(
        tokenizer_dir
    )
    runtime_audit = _validate_locked_tokenizer_runtime_audit(
        runtime_audit,
        label="Wan tokenizer runtime audit",
    )
    validator_path = Path(wan_5b_wrapper.__file__).resolve()
    project_root = Path(__file__).resolve().parents[1]
    try:
        validator_relative = validator_path.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            "Wan text validator must live inside the project directory."
        ) from exc
    if validator_relative != "utils/wan_5b_wrapper.py":
        raise RuntimeError(
            f"Unexpected Wan text validator path: {validator_relative!r}."
        )

    contract = {
        "t5_checkpoint_aggregate_sha256": actual_t5_hash,
        "tokenizer_aggregate_sha256": actual_tokenizer_hash,
        "tokenizer_revision": f"local-tree-sha256:{actual_tokenizer_hash}",
        **_LOCKED_TEXT_ENCODING_SETTINGS,
    }
    upgraded = copy.deepcopy(legacy)
    upgraded.pop("manifest_sha256")
    upgraded_fingerprint = copy.deepcopy(fingerprint)
    upgraded_fingerprint.pop("aggregate_sha256")
    upgraded_fingerprint["text_encoding"] = contract
    upgraded_fingerprint["aggregate_sha256"] = canonical_json_sha256(
        upgraded_fingerprint
    )
    upgraded["source_fingerprint"] = upgraded_fingerprint
    upgraded[STAGE2_TEXT_ENCODING_UPGRADE_KEY] = {
        "schema": STAGE2_TEXT_ENCODING_UPGRADE_SCHEMA,
        "schema_version": STAGE2_TEXT_ENCODING_UPGRADE_SCHEMA_VERSION,
        "original_source_manifest": {
            "path": str(legacy_path),
            "manifest_sha256": legacy["manifest_sha256"],
            "source_fingerprint_sha256": fingerprint_hash,
        },
        "verification": {
            "validator_file": validator_relative,
            "validator_file_sha256": sha256_file(validator_path),
            "text_encoding_contract_sha256": canonical_json_sha256(contract),
            "t5_checkpoint_aggregate_sha256": actual_t5_hash,
            "tokenizer_aggregate_sha256": actual_tokenizer_hash,
            "tokenizer_runtime_audit": runtime_audit,
        },
        "operator_attestation": {
            "operator_id": operator_id,
            "statement": operator_attestation,
        },
    }
    upgraded["manifest_sha256"] = canonical_json_sha256(upgraded)

    def verify_upgrade_inputs_unchanged() -> None:
        """Recheck every attested input at the exact publication boundary."""

        final_t5_hash = aggregate_file_hash(tree_file_hashes(t5_checkpoint_path))
        final_tokenizer_hash = aggregate_file_hash(tree_file_hashes(tokenizer_dir))
        if final_t5_hash != actual_t5_hash or final_t5_hash != expected_t5_hash:
            raise RuntimeError("T5 checkpoint tree changed during the upgrade audit.")
        if (
            final_tokenizer_hash != actual_tokenizer_hash
            or final_tokenizer_hash != expected_tokenizer_hash
        ):
            raise RuntimeError("Tokenizer tree changed during the upgrade audit.")
        if sha256_file(legacy_path) != original_bytes_hash:
            raise RuntimeError("Source base manifest changed during the upgrade audit.")

    # The tokenizer probe executes code against the tokenizer tree.  Rehash
    # immediately after it, then again after the potentially long native-F25
    # closure scan while the candidate is still only a temporary file.
    verify_upgrade_inputs_unchanged()
    # Validate the exact candidate bytes before the atomic replace.  A failed
    # closure check must never leave an attested-looking, non-overwritable file.
    with atomic_output_path(output_path, suffix=".json.tmp") as temporary:
        temporary.write_bytes(canonical_json_bytes(upgraded) + b"\n")
        load_source_cache_manifest(
            temporary,
            expected_num_samples=expected_num_samples,
            require_text_encoding_upgrade=True,
        )
        verify_upgrade_inputs_unchanged()
    return output_path


def save_negative_conditioning_artifact(
    path: str | os.PathLike[str],
    *,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
) -> dict[str, Any]:
    """Save one offline negative embedding artifact after strict validation."""

    from safetensors.torch import save_file

    tensors = {
        "prompt_embeds": prompt_embeds.detach().contiguous().cpu().to(torch.bfloat16),
        "prompt_mask": prompt_mask.detach().contiguous().cpu().to(torch.bool),
    }
    prompt_facts = _validate_prompt_tensors(
        tensors["prompt_embeds"],
        tensors["prompt_mask"],
        label="negative conditioning",
        require_padding=True,
    )
    path = Path(path).expanduser().resolve()
    with atomic_output_path(path, suffix=".safetensors.tmp") as temporary:
        save_file(tensors, str(temporary))
    payload = path.read_bytes()
    return {
        "path": path.name,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "tensors": {
            name: _tensor_description(tensors[name]) for name in sorted(tensors)
        },
        "prompt_valid_tokens": prompt_facts["prompt_valid_tokens"],
        "padding_sha256": prompt_facts["padding_sha256"],
    }


def write_negative_conditioning_manifest(
    output_path: str | os.PathLike[str],
    *,
    artifact_path: str | os.PathLike[str],
    source_cache_manifest_path: str | os.PathLike[str],
    expected_num_samples: int,
    require_text_encoding_upgrade: bool = False,
) -> Path:
    """Bind an offline negative embedding to the positive cache provenance."""

    output_path = Path(output_path).expanduser().resolve()
    artifact_path = Path(artifact_path).expanduser().resolve()
    source = load_source_cache_manifest(
        source_cache_manifest_path,
        expected_num_samples=expected_num_samples,
        require_text_encoding_upgrade=require_text_encoding_upgrade,
    )
    if source.get("schema") != STAGE2_F25_SOURCE_CACHE_SCHEMA:
        raise RuntimeError(
            "Stage-2 training runtime accepts only a native attested F25 source "
            "manifest; legacy Stage-1 manifests are migration inputs only."
        )
    positive_text_encoding = _source_text_encoding_contract(source)
    try:
        relative_artifact = artifact_path.relative_to(output_path.parent).as_posix()
    except ValueError as exc:
        raise ValueError(
            "Negative artifact must be inside the negative manifest directory."
        ) from exc
    payload = artifact_path.read_bytes()
    artifact_size = len(payload)
    artifact_sha256 = hashlib.sha256(payload).hexdigest()
    from safetensors.torch import load

    tensors = load(payload)
    if set(tensors) != set(_NEGATIVE_TENSOR_DTYPES):
        raise ValueError(
            "Negative conditioning artifact must contain only prompt_embeds and "
            "prompt_mask."
        )
    prompt_facts = _validate_prompt_tensors(
        tensors["prompt_embeds"],
        tensors["prompt_mask"],
        label="negative conditioning",
        require_padding=True,
    )
    artifact = {
        "path": relative_artifact,
        "size": artifact_size,
        "sha256": artifact_sha256,
        "tensors": {
            name: _tensor_description(tensors[name]) for name in sorted(tensors)
        },
        "prompt_valid_tokens": prompt_facts["prompt_valid_tokens"],
        "padding_sha256": prompt_facts["padding_sha256"],
    }
    manifest: dict[str, Any] = {
        "schema": STAGE2_NEGATIVE_SCHEMA,
        "schema_version": STAGE2_NEGATIVE_SCHEMA_VERSION,
        "text": DEFAULT_NEGATIVE_PROMPT,
        "text_utf8_sha256": STAGE2_NEGATIVE_PROMPT_SHA256,
        "positive_cache_manifest_sha256": source["manifest_sha256"],
        "encoder": positive_text_encoding,
        "artifact": artifact,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(output_path, manifest)
    return output_path


def load_negative_conditioning(
    manifest_path: str | os.PathLike[str],
    *,
    source_cache_manifest: Mapping[str, Any],
    load_tensors: bool = True,
) -> dict[str, Any]:
    """Validate exact Stage-1 negative text, encoder provenance, and tensors."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _load_self_hashed_json(
        manifest_path, label="negative conditioning manifest"
    )
    _require_exact_keys(
        manifest,
        label="negative conditioning manifest",
        expected=_NEGATIVE_MANIFEST_KEYS,
    )
    if manifest["schema"] != STAGE2_NEGATIVE_SCHEMA or manifest["schema_version"] != 1:
        raise RuntimeError("Unsupported negative conditioning manifest schema.")
    if manifest["text"] != DEFAULT_NEGATIVE_PROMPT:
        raise RuntimeError(
            "Negative conditioning text is not byte-for-byte DEFAULT_NEGATIVE_PROMPT."
        )
    if manifest["text_utf8_sha256"] != STAGE2_NEGATIVE_PROMPT_SHA256:
        raise RuntimeError("Negative conditioning UTF-8 text hash mismatch.")
    if manifest["positive_cache_manifest_sha256"] != source_cache_manifest.get(
        "manifest_sha256"
    ):
        raise RuntimeError(
            "Negative conditioning was not built from this positive cache manifest."
        )

    encoder = _require_exact_keys(
        manifest["encoder"],
        label="negative conditioning encoder",
        expected=_NEGATIVE_ENCODER_KEYS,
    )
    positive_encoder = _source_text_encoding_contract(source_cache_manifest)
    for key, expected in positive_encoder.items():
        if encoder[key] != expected or type(encoder[key]) is not type(expected):
            raise RuntimeError(
                "Negative and positive cache text encoding differ for "
                f"{key}: expected {expected!r}, got {encoder[key]!r}."
            )

    artifact = _require_exact_keys(
        manifest["artifact"],
        label="negative conditioning artifact",
        expected=_NEGATIVE_ARTIFACT_KEYS,
    )
    artifact_path = _safe_relative_file(
        manifest_path.parent,
        artifact["path"],
        label="negative conditioning artifact.path",
    )
    expected_size = artifact["size"]
    if isinstance(expected_size, bool) or not isinstance(expected_size, int):
        raise RuntimeError("Negative conditioning artifact.size must be an integer.")
    expected_file_hash = _require_sha256(
        artifact["sha256"], "negative conditioning artifact.sha256"
    )
    tensors = _load_verified_safetensors_bytes(
        artifact_path,
        expected_size=expected_size,
        expected_sha256=expected_file_hash,
        label="Negative conditioning artifact",
    )
    if set(tensors) != set(_NEGATIVE_TENSOR_DTYPES):
        raise RuntimeError(
            "Negative conditioning artifact tensor keys must be prompt_embeds and "
            "prompt_mask only."
        )
    prompt_facts = _validate_prompt_tensors(
        tensors["prompt_embeds"],
        tensors["prompt_mask"],
        label="negative conditioning",
        require_padding=True,
    )
    actual_descriptions = {
        name: _tensor_description(tensors[name]) for name in sorted(tensors)
    }
    if artifact["tensors"] != actual_descriptions:
        raise RuntimeError("Negative conditioning tensor shape/dtype/hash mismatch.")
    if artifact["prompt_valid_tokens"] != prompt_facts["prompt_valid_tokens"]:
        raise RuntimeError("Negative conditioning prompt_valid_tokens mismatch.")
    if artifact["padding_sha256"] != prompt_facts["padding_sha256"]:
        raise RuntimeError("Negative conditioning padding hash mismatch.")

    result = {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "artifact_path": artifact_path,
    }
    if load_tensors:
        result["prompt_embeds"] = tensors["prompt_embeds"]
        result["prompt_mask"] = tensors["prompt_mask"]
    return result


def _load_action_sidecar(
    path: str | os.PathLike[str],
    *,
    records: Sequence[Stage1I2VRecord],
    expected_action_ids: Sequence[str],
) -> tuple[list[str], str]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    expected_videos = [str(record.canonical_row["video"]) for record in records]
    expected_set = set(expected_videos)
    if len(expected_set) != len(expected_videos):
        raise RuntimeError("Metadata video values are not unique.")
    values: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["video", "action_id"]:
            raise ValueError("Action sidecar header must be exactly: video,action_id.")
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"Action sidecar row {row_number} contains extra unheaded values."
                )
            video = row["video"]
            action_id = row["action_id"]
            if video != video.strip() or not video:
                raise ValueError(
                    f"Action sidecar row {row_number} has an invalid video value."
                )
            if video not in expected_set:
                raise ValueError(
                    f"Action sidecar row {row_number} references unknown video {video!r}."
                )
            if video in values:
                raise ValueError(f"Action sidecar contains duplicate video {video!r}.")
            if action_id not in expected_action_ids:
                raise ValueError(
                    f"Action sidecar row {row_number} has unknown action_id "
                    f"{action_id!r}; expected one of {list(expected_action_ids)}."
                )
            values[video] = action_id
    missing = [video for video in expected_videos if video not in values]
    if missing:
        raise ValueError(
            f"Action sidecar is missing {len(missing)} metadata videos; first={missing[0]!r}."
        )
    return [values[video] for video in expected_videos], sha256_file(path)


def _resolve_action_labels(
    *,
    source_manifest: Mapping[str, Any],
    records: Sequence[Stage1I2VRecord],
    expected_action_ids: Sequence[str],
    action_labels_path: str | os.PathLike[str] | None,
) -> tuple[list[str], dict[str, Any]]:
    entries = source_manifest["records"]
    has_action = ["action_id" in entry for entry in entries]
    if any(has_action) and not all(has_action):
        raise RuntimeError(
            "Source cache manifest has only partial action_id coverage; refusing fallback."
        )

    sidecar_labels: list[str] | None = None
    sidecar_hash: str | None = None
    if action_labels_path is not None:
        sidecar_labels, sidecar_hash = _load_action_sidecar(
            action_labels_path,
            records=records,
            expected_action_ids=expected_action_ids,
        )

    if all(has_action):
        manifest_labels = []
        for row_id, entry in enumerate(entries):
            value = entry["action_id"]
            if value not in expected_action_ids:
                raise RuntimeError(
                    f"Source cache record {row_id} has unknown action_id {value!r}; "
                    f"expected one of {list(expected_action_ids)}."
                )
            manifest_labels.append(value)
        if sidecar_labels is not None and sidecar_labels != manifest_labels:
            mismatch = next(
                index
                for index, (left, right) in enumerate(
                    zip(manifest_labels, sidecar_labels)
                )
                if left != right
            )
            raise RuntimeError(
                "Action sidecar disagrees with cache manifest at row "
                f"{mismatch}: {manifest_labels[mismatch]!r} != "
                f"{sidecar_labels[mismatch]!r}."
            )
        provenance: dict[str, Any] = {
            "kind": "cache_manifest",
            "field": "action_id",
            "sha256": source_manifest["manifest_sha256"],
        }
        if sidecar_hash is not None:
            provenance["confirmed_sidecar_sha256"] = sidecar_hash
            provenance["confirmed_sidecar_path"] = str(
                Path(action_labels_path).expanduser().resolve()
            )
        return manifest_labels, provenance

    if sidecar_labels is None or sidecar_hash is None:
        raise RuntimeError(
            "Source cache manifest has no action_id. Provide an operator-confirmed "
            "video,action_id sidecar; prompt text and row order are never classifiers."
        )
    return sidecar_labels, {
        "kind": "operator_confirmed_sidecar",
        "path": str(Path(action_labels_path).expanduser().resolve()),
        "sha256": sidecar_hash,
    }


def _check_source_tensor_description(
    source_entry: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    row_id: int,
) -> None:
    declared = source_entry.get("tensors")
    if declared is None:
        return
    declared = _require_mapping(declared, f"source cache record {row_id}.tensors")
    if set(declared) != set(actual):
        raise RuntimeError(
            f"Source cache record {row_id} tensor names do not match artifact."
        )
    for name, actual_description in actual.items():
        description = _require_mapping(
            declared[name], f"source cache record {row_id}.tensors.{name}"
        )
        if description.get("shape") != actual_description["shape"]:
            raise RuntimeError(
                f"Source cache record {row_id} declares the wrong {name} shape."
            )
        if description.get("dtype") != actual_description["dtype"]:
            raise RuntimeError(
                f"Source cache record {row_id} declares the wrong {name} dtype."
            )
        if (
            "sha256" in description
            and description["sha256"] != actual_description["sha256"]
        ):
            raise RuntimeError(
                f"Source cache record {row_id} declares the wrong {name} hash."
            )


def audit_stage2_i2v_cache(
    *,
    metadata_path: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    source_cache_manifest_path: str | os.PathLike[str],
    negative_conditioning_manifest_path: str | os.PathLike[str],
    expected_action_ids: Sequence[str],
    config_contract_sha256: str,
    config_launch_sha256: str,
    action_labels_path: str | os.PathLike[str] | None = None,
    output_manifest_path: str | os.PathLike[str] | None = None,
    expected_num_samples: int = 600,
    expected_action_counts: Mapping[str, int] | None = None,
    allowed_latent_spatial_shapes: Iterable[tuple[int, int]] = ((30, 52), (52, 30)),
    require_text_encoding_upgrade: bool = False,
    require_native_f25_source: bool = False,
) -> dict[str, Any]:
    """Scan every Stage-2 cache artifact and atomically write the training gate."""

    from utils.stage1_i2v_data import load_stage1_i2v_manifest

    if isinstance(expected_num_samples, bool) or int(expected_num_samples) <= 0:
        raise ValueError("expected_num_samples must be a positive integer.")
    expected_num_samples = int(expected_num_samples)
    action_order = tuple(expected_action_ids)
    if len(action_order) != 3 or len(set(action_order)) != 3:
        raise ValueError(
            "expected_action_ids must contain exactly three unique values."
        )
    for index, action_id in enumerate(action_order):
        _require_nonempty_string(action_id, f"expected_action_ids[{index}]")
    if expected_action_counts is None:
        if expected_num_samples % len(action_order):
            raise ValueError(
                "expected_num_samples must be divisible by the number of actions "
                "when expected_action_counts is omitted."
            )
        uniform_count = expected_num_samples // len(action_order)
        normalized_expected_counts = {
            action_id: uniform_count for action_id in action_order
        }
    else:
        if not isinstance(expected_action_counts, Mapping):
            raise ValueError("expected_action_counts must be a mapping.")
        if set(expected_action_counts) != set(action_order):
            raise ValueError(
                "expected_action_counts keys must exactly match expected_action_ids."
            )
        normalized_expected_counts = {}
        for action_id in action_order:
            count = expected_action_counts[action_id]
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(
                    f"expected_action_counts[{action_id!r}] must be a positive integer."
                )
            normalized_expected_counts[action_id] = count
        if sum(normalized_expected_counts.values()) != expected_num_samples:
            raise ValueError("expected_action_counts must sum to expected_num_samples.")
    config_contract_sha256 = _require_sha256(
        config_contract_sha256, "config_contract_sha256"
    )
    config_launch_sha256 = _require_sha256(config_launch_sha256, "config_launch_sha256")
    cache_root = Path(cache_dir).expanduser().resolve()
    if not cache_root.is_dir():
        raise FileNotFoundError(cache_root)
    metadata_path = Path(metadata_path).expanduser().resolve()
    source_manifest_path = Path(source_cache_manifest_path).expanduser().resolve()
    records = load_stage1_i2v_manifest(
        metadata_path,
        expected_num_samples=expected_num_samples,
        require_files=False,
        validate_images=False,
    )
    source_manifest = load_source_cache_manifest(
        source_manifest_path,
        expected_num_samples=expected_num_samples,
        require_text_encoding_upgrade=require_text_encoding_upgrade,
    )
    if (
        require_native_f25_source
        and source_manifest.get("schema") != STAGE2_F25_SOURCE_CACHE_SCHEMA
    ):
        raise RuntimeError(
            "Formal Stage-2 audit accepts only the native attested F25 source "
            "manifest produced by prepare_stage2_i2v_f25_cache.py."
        )
    if source_manifest.get("schema") == STAGE2_F25_SOURCE_CACHE_SCHEMA:
        expected_data_contract = stage2_f25_data_contract_sha256(
            expected_num_samples=expected_num_samples,
            metadata_sha256=sha256_file(metadata_path),
            allowed_latent_spatial_shapes=allowed_latent_spatial_shapes,
        )
        recorded_data_contract = source_manifest["preparation"]["config"][
            "data_contract_sha256"
        ]
        if recorded_data_contract != expected_data_contract:
            raise RuntimeError(
                "Stage-2 F25 data contract differs from the current metadata/tensor "
                "contract; optimizer-only config changes are intentionally excluded."
            )
    source_entries = list(source_manifest["records"])
    actions, action_source = _resolve_action_labels(
        source_manifest=source_manifest,
        records=records,
        expected_action_ids=action_order,
        action_labels_path=action_labels_path,
    )
    action_counts = Counter(actions)
    if dict(action_counts) != normalized_expected_counts:
        raise RuntimeError(
            "Action counts must be exactly "
            f"{normalized_expected_counts}, got {dict(action_counts)}."
        )

    negative = load_negative_conditioning(
        negative_conditioning_manifest_path,
        source_cache_manifest=source_manifest,
        load_tensors=False,
    )
    allowed_shapes = tuple(
        tuple(map(int, shape)) for shape in allowed_latent_spatial_shapes
    )
    if set(allowed_shapes) != {(30, 52), (52, 30)}:
        raise ValueError(
            "Stage-2 baseline latent spatial shapes are locked to (30,52) and (52,30)."
        )

    audited_records: list[dict[str, Any]] = []
    total_bytes = 0
    orientation_counts: Counter[str] = Counter()
    for row_id, (record, source_entry, action_id) in enumerate(
        zip(records, source_entries, actions)
    ):
        if source_entry["row_sha256"] != record.row_sha256:
            raise RuntimeError(
                f"Cache/metadata row hash mismatch at row {row_id}: expected "
                f"{record.row_sha256}, got {source_entry['row_sha256']}."
            )
        for field_name, expected in (
            ("height", record.height),
            ("width", record.width),
            ("bucket", record.bucket),
        ):
            if field_name in source_entry and source_entry[field_name] != expected:
                raise RuntimeError(
                    f"Source cache record {row_id}.{field_name} does not match metadata."
                )
        artifact_path = _safe_relative_file(
            cache_root,
            source_entry["path"],
            label=f"source cache record {row_id}.path",
        )
        expected_file_hash = source_entry["sha256"]
        size = source_entry["size"]
        tensors = _load_verified_safetensors_bytes(
            artifact_path,
            expected_size=size,
            expected_sha256=expected_file_hash,
            label=f"cache artifact row {row_id}",
        )
        facts = validate_stage2_cache_tensors(
            tensors,
            expected_spatial_shapes=allowed_shapes,
            label=f"cache row {row_id}",
        )
        expected_spatial = (record.height // 16, record.width // 16)
        if record.height % 16 or record.width % 16:
            raise RuntimeError(
                f"Metadata row {row_id} dimensions are not divisible by latent ratio 16."
            )
        if tuple(facts["latent_spatial_shape"]) != expected_spatial:
            raise RuntimeError(
                f"Cache row {row_id} latent orientation/shape "
                f"{tuple(facts['latent_spatial_shape'])} does not match metadata "
                f"{(record.height, record.width)} -> {expected_spatial}."
            )
        _check_source_tensor_description(source_entry, facts["tensors"], row_id=row_id)
        relative_path = artifact_path.relative_to(cache_root).as_posix()
        total_bytes += size
        orientation_counts[record.bucket] += 1
        audited_records.append(
            {
                "row_id": row_id,
                "row_sha256": record.row_sha256,
                "video": str(record.canonical_row["video"]),
                "prompt_utf8_sha256": hashlib.sha256(
                    record.prompt.encode("utf-8")
                ).hexdigest(),
                "action_id": action_id,
                "height": record.height,
                "width": record.width,
                "bucket": record.bucket,
                "path": relative_path,
                "size": size,
                "sha256": expected_file_hash,
                **facts,
            }
        )

    output_path = (
        Path(output_manifest_path or cache_root / STAGE2_CACHE_MANIFEST_NAME)
        .expanduser()
        .resolve()
    )
    manifest: dict[str, Any] = {
        "schema": STAGE2_CACHE_SCHEMA,
        "schema_version": STAGE2_CACHE_SCHEMA_VERSION,
        "common_tensor_schema": STAGE2_CACHE_TENSOR_SCHEMA,
        "common_tensor_schema_sha256": STAGE2_CACHE_TENSOR_SCHEMA_SHA256,
        "num_samples": len(audited_records),
        "total_bytes": total_bytes,
        "orientation_counts": dict(sorted(orientation_counts.items())),
        "actions": {
            "ids": list(action_order),
            "counts": {
                action_id: action_counts[action_id] for action_id in action_order
            },
            "source": action_source,
        },
        "negative_conditioning": {
            "manifest_path": str(negative["manifest_path"]),
            "manifest_sha256": negative["manifest"]["manifest_sha256"],
            "artifact_sha256": negative["manifest"]["artifact"]["sha256"],
            "text_utf8_sha256": STAGE2_NEGATIVE_PROMPT_SHA256,
        },
        "provenance": {
            "metadata_path": str(metadata_path),
            "metadata_sha256": sha256_file(metadata_path),
            "source_cache_manifest_path": str(source_manifest_path),
            "source_cache_manifest_sha256": source_manifest["manifest_sha256"],
            "config_contract_sha256": config_contract_sha256,
            "config_launch_sha256": config_launch_sha256,
        },
        "records": audited_records,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(output_path, manifest)
    manifest["manifest_path"] = str(output_path)
    return manifest


def load_stage2_i2v_manifest(
    path_or_cache_dir: str | os.PathLike[str],
    *,
    expected_num_samples: int | None = None,
) -> dict[str, Any]:
    path = Path(path_or_cache_dir).expanduser().resolve()
    if path.is_dir():
        path = path / STAGE2_CACHE_MANIFEST_NAME
    manifest = _load_self_hashed_json(path, label="Stage-2 cache manifest")
    _require_exact_keys(
        manifest,
        label="Stage-2 cache manifest",
        expected=_STAGE2_MANIFEST_KEYS,
    )
    if (
        manifest["schema"] != STAGE2_CACHE_SCHEMA
        or manifest["schema_version"] != STAGE2_CACHE_SCHEMA_VERSION
    ):
        raise RuntimeError(f"Unsupported Stage-2 cache manifest schema in {path}.")
    if manifest["common_tensor_schema"] != STAGE2_CACHE_TENSOR_SCHEMA:
        raise RuntimeError("Stage-2 common tensor schema content mismatch.")
    if manifest["common_tensor_schema_sha256"] != STAGE2_CACHE_TENSOR_SCHEMA_SHA256:
        raise RuntimeError("Stage-2 common tensor schema hash mismatch.")
    records = manifest["records"]
    if not isinstance(records, list) or len(records) != manifest["num_samples"]:
        raise RuntimeError("Stage-2 manifest record count is invalid.")
    if expected_num_samples is not None and len(records) != int(expected_num_samples):
        raise RuntimeError(
            f"Expected {expected_num_samples} Stage-2 records, found {len(records)}."
        )
    actions = _require_exact_keys(
        manifest["actions"],
        label="Stage-2 manifest.actions",
        expected={"ids", "counts", "source"},
    )
    action_ids = actions.get("ids")
    counts = actions.get("counts")
    if (
        not isinstance(action_ids, list)
        or len(action_ids) != 3
        or len(set(action_ids)) != 3
    ):
        raise RuntimeError("Stage-2 manifest must define exactly three action ids.")
    if not isinstance(counts, dict) or set(counts) != set(action_ids):
        raise RuntimeError("Stage-2 manifest action counts must be an object.")
    observed_actions: Counter[str] = Counter()
    observed_orientations: Counter[str] = Counter()
    seen_paths: set[str] = set()
    observed_total_bytes = 0
    for row_id, raw_entry in enumerate(records):
        entry = _require_exact_keys(
            raw_entry,
            label=f"Stage-2 record {row_id}",
            expected=_STAGE2_RECORD_KEYS,
        )
        if (
            isinstance(entry["row_id"], bool)
            or not isinstance(entry["row_id"], int)
            or entry["row_id"] != row_id
        ):
            raise RuntimeError(
                f"Stage-2 row ids must be contiguous; expected {row_id}, "
                f"got {entry['row_id']!r}."
            )
        _require_sha256(entry["row_sha256"], f"record {row_id}.row_sha256")
        _require_sha256(
            entry["prompt_utf8_sha256"], f"record {row_id}.prompt_utf8_sha256"
        )
        _require_sha256(
            entry["prompt_padding_sha256"],
            f"record {row_id}.prompt_padding_sha256",
        )
        _require_nonempty_string(entry["video"], f"record {row_id}.video")
        bucket = _require_nonempty_string(entry["bucket"], f"record {row_id}.bucket")
        action_id = entry.get("action_id")
        if action_id not in action_ids:
            raise RuntimeError(f"Stage-2 record {row_id} has unknown action_id.")
        observed_actions[action_id] += 1
        observed_orientations[bucket] += 1
        relative = _require_nonempty_string(entry.get("path"), f"record {row_id}.path")
        if relative in seen_paths:
            raise RuntimeError(f"Duplicate Stage-2 cache artifact path {relative!r}.")
        seen_paths.add(relative)
        _require_sha256(entry.get("sha256"), f"record {row_id}.sha256")
        for key in ("height", "width", "size", "prompt_valid_tokens"):
            value = entry[key]
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RuntimeError(f"Stage-2 record {row_id}.{key} must be positive.")
        observed_total_bytes += entry["size"]
        spatial = entry["latent_spatial_shape"]
        expected_spatial = [entry["height"] // 16, entry["width"] // 16]
        if spatial != expected_spatial or tuple(spatial) not in {
            (30, 52),
            (52, 30),
        }:
            raise RuntimeError(
                f"Stage-2 record {row_id} latent spatial geometry is invalid."
            )
        if entry["real_future_shape"] != [24, 48, *spatial]:
            raise RuntimeError(f"Stage-2 record {row_id} real_future_shape is invalid.")
        expected_tensor_shapes = {
            "video_latent": ([25, 48, *spatial], "bfloat16"),
            "initial_latent": ([1, 48, *spatial], "bfloat16"),
            "prompt_embeds": ([512, 4096], "bfloat16"),
            "prompt_mask": ([512], "bool"),
        }
        tensor_descriptions = _require_exact_keys(
            entry["tensors"],
            label=f"Stage-2 record {row_id}.tensors",
            expected=set(expected_tensor_shapes),
        )
        for name, (shape, dtype) in expected_tensor_shapes.items():
            description = _require_exact_keys(
                tensor_descriptions[name],
                label=f"Stage-2 record {row_id}.tensors.{name}",
                expected={"shape", "dtype", "sha256"},
            )
            if description["shape"] != shape or description["dtype"] != dtype:
                raise RuntimeError(
                    f"Stage-2 record {row_id}.{name} tensor schema is invalid."
                )
            _require_sha256(description["sha256"], f"record {row_id}.{name}.sha256")
        difference = _require_exact_keys(
            entry["initial_vs_video0"],
            label=f"Stage-2 record {row_id}.initial_vs_video0",
            expected={"exact_equal", "num_different", "max_abs_diff", "mean_abs_diff"},
        )
        if not isinstance(difference["exact_equal"], bool):
            raise RuntimeError("initial_vs_video0.exact_equal must be bool.")
        num_different = difference["num_different"]
        if (
            isinstance(num_different, bool)
            or not isinstance(num_different, int)
            or num_different < 0
        ):
            raise RuntimeError("initial_vs_video0.num_different must be non-negative.")
        if difference["exact_equal"] is not (num_different == 0):
            raise RuntimeError("initial_vs_video0 exact_equal/count disagree.")
        for key in ("max_abs_diff", "mean_abs_diff"):
            value = difference[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise RuntimeError(f"initial_vs_video0.{key} must be numeric.")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise RuntimeError(
                    f"initial_vs_video0.{key} must be finite/non-negative."
                )
    expected_counts = {action_id: counts.get(action_id) for action_id in action_ids}
    if dict(observed_actions) != expected_counts:
        raise RuntimeError(
            f"Stage-2 manifest action counts mismatch: {dict(observed_actions)} != "
            f"{expected_counts}."
        )
    if manifest["orientation_counts"] != dict(sorted(observed_orientations.items())):
        raise RuntimeError("Stage-2 manifest orientation counts mismatch.")
    if (
        isinstance(manifest["total_bytes"], bool)
        or not isinstance(manifest["total_bytes"], int)
        or manifest["total_bytes"] != observed_total_bytes
    ):
        raise RuntimeError("Stage-2 manifest total_bytes mismatch.")
    return manifest


def validate_stage2_i2v_runtime_bindings(
    manifest: Mapping[str, Any],
    *,
    metadata_path: str | os.PathLike[str],
    source_cache_manifest_path: str | os.PathLike[str],
    negative_conditioning_manifest_path: str | os.PathLike[str],
    config_contract_sha256: str,
    config_launch_sha256: str,
    expected_num_samples: int,
) -> dict[str, Any]:
    """Revalidate every mutable external input before a training process starts."""

    from utils.stage1_i2v_data import load_stage1_i2v_manifest

    provenance = _require_required_keys(
        manifest.get("provenance"),
        label="Stage-2 manifest.provenance",
        required=_STAGE2_PROVENANCE_KEYS,
    )
    negative_binding = _require_exact_keys(
        manifest.get("negative_conditioning"),
        label="Stage-2 manifest.negative_conditioning",
        expected=_STAGE2_NEGATIVE_BINDING_KEYS,
    )
    config_contract_sha256 = _require_sha256(
        config_contract_sha256, "current config_contract_sha256"
    )
    config_launch_sha256 = _require_sha256(
        config_launch_sha256, "current config_launch_sha256"
    )
    metadata_path = Path(metadata_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    current_metadata_hash = sha256_file(metadata_path)
    metadata_records = load_stage1_i2v_manifest(
        metadata_path,
        expected_num_samples=expected_num_samples,
        require_files=False,
        validate_images=False,
    )
    expected_values = {
        "metadata_sha256": current_metadata_hash,
        "config_contract_sha256": config_contract_sha256,
        "config_launch_sha256": config_launch_sha256,
    }
    for key, actual in expected_values.items():
        if provenance[key] != actual or type(provenance[key]) is not type(actual):
            raise RuntimeError(
                f"Stage-2 runtime binding {key} mismatch: "
                f"manifest={provenance[key]!r}, current={actual!r}."
            )

    source = load_source_cache_manifest(
        source_cache_manifest_path,
        expected_num_samples=expected_num_samples,
        require_text_encoding_upgrade=True,
    )
    if source.get("schema") != STAGE2_F25_SOURCE_CACHE_SCHEMA:
        raise RuntimeError(
            "Stage-2 training runtime accepts only a native attested F25 source "
            "manifest; legacy Stage-1 manifests are migration inputs only."
        )
    if provenance["source_cache_manifest_sha256"] != source["manifest_sha256"]:
        raise RuntimeError(
            "Stage-2 runtime source cache manifest differs from the audited source."
        )

    actions = _require_exact_keys(
        manifest.get("actions"),
        label="Stage-2 manifest.actions",
        expected={"ids", "counts", "source"},
    )
    action_order = tuple(actions["ids"])
    action_source = _require_mapping(
        actions["source"], "Stage-2 manifest.actions.source"
    )
    source_kind = action_source.get("kind")
    if source_kind == "operator_confirmed_sidecar":
        if set(action_source) != {"kind", "path", "sha256"}:
            raise RuntimeError("Operator action source keys are invalid.")
        runtime_actions, action_hash = _load_action_sidecar(
            action_source["path"],
            records=metadata_records,
            expected_action_ids=action_order,
        )
        if action_hash != action_source["sha256"]:
            raise RuntimeError("Operator action sidecar hash changed after audit.")
    elif source_kind == "cache_manifest":
        allowed_source_keys = {
            "kind",
            "field",
            "sha256",
            "confirmed_sidecar_path",
            "confirmed_sidecar_sha256",
        }
        if set(action_source) - allowed_source_keys or not {
            "kind",
            "field",
            "sha256",
        }.issubset(action_source):
            raise RuntimeError("Cache-manifest action source keys are invalid.")
        if (
            action_source["field"] != "action_id"
            or action_source["sha256"] != source["manifest_sha256"]
        ):
            raise RuntimeError("Cache-manifest action provenance differs from audit.")
        source_entries = source["records"]
        if not all("action_id" in entry for entry in source_entries):
            raise RuntimeError(
                "Runtime source manifest lost complete action_id labels."
            )
        runtime_actions = [entry["action_id"] for entry in source_entries]
        has_confirmed_path = "confirmed_sidecar_path" in action_source
        has_confirmed_hash = "confirmed_sidecar_sha256" in action_source
        if has_confirmed_path is not has_confirmed_hash:
            raise RuntimeError(
                "Confirmed action sidecar path/hash must appear together."
            )
        if has_confirmed_path:
            confirmed_actions, confirmed_hash = _load_action_sidecar(
                action_source["confirmed_sidecar_path"],
                records=metadata_records,
                expected_action_ids=action_order,
            )
            if (
                confirmed_hash != action_source["confirmed_sidecar_sha256"]
                or confirmed_actions != runtime_actions
            ):
                raise RuntimeError(
                    "Confirmed action sidecar differs from cache labels."
                )
    else:
        raise RuntimeError(f"Unsupported Stage-2 action source kind: {source_kind!r}.")

    for row_id, (entry, source_entry, metadata_record, runtime_action) in enumerate(
        zip(
            manifest["records"],
            source["records"],
            metadata_records,
            runtime_actions,
        )
    ):
        expected_values = {
            "row_sha256": metadata_record.row_sha256,
            "video": str(metadata_record.canonical_row["video"]),
            "prompt_utf8_sha256": hashlib.sha256(
                metadata_record.prompt.encode("utf-8")
            ).hexdigest(),
            "action_id": runtime_action,
            "height": metadata_record.height,
            "width": metadata_record.width,
            "bucket": metadata_record.bucket,
            "path": source_entry["path"],
            "sha256": source_entry["sha256"],
            "latent_spatial_shape": [
                metadata_record.height // 16,
                metadata_record.width // 16,
            ],
        }
        for key, expected in expected_values.items():
            if entry[key] != expected or type(entry[key]) is not type(expected):
                raise RuntimeError(
                    f"Stage-2 runtime record {row_id}.{key} differs from its "
                    f"source/metadata binding."
                )
        if source_entry["row_sha256"] != metadata_record.row_sha256:
            raise RuntimeError(
                f"Runtime source record {row_id} differs from current metadata."
            )
        if "size" in source_entry and entry["size"] != source_entry["size"]:
            raise RuntimeError(
                f"Stage-2 runtime record {row_id}.size differs from source manifest."
            )

    negative = load_negative_conditioning(
        negative_conditioning_manifest_path,
        source_cache_manifest=source,
        load_tensors=True,
    )
    actual_negative = {
        "manifest_sha256": negative["manifest"]["manifest_sha256"],
        "artifact_sha256": negative["manifest"]["artifact"]["sha256"],
        "text_utf8_sha256": STAGE2_NEGATIVE_PROMPT_SHA256,
    }
    for key, actual in actual_negative.items():
        if negative_binding[key] != actual:
            raise RuntimeError(
                f"Stage-2 runtime negative conditioning {key} differs from audit."
            )
    return {"source_manifest": source, "negative_conditioning": negative}


class Stage2I2VCacheDataset(Dataset):
    """Training dataset exposing explicit sink and matched real future tensors."""

    def __init__(
        self,
        cache_dir: str | os.PathLike[str],
        *,
        metadata_path: str | os.PathLike[str],
        source_cache_manifest_path: str | os.PathLike[str],
        negative_conditioning_manifest_path: str | os.PathLike[str],
        config_contract_sha256: str,
        config_launch_sha256: str,
        manifest_path: str | os.PathLike[str] | None = None,
        expected_num_samples: int | None = 600,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(self.cache_dir)
        self.manifest_path = (
            Path(manifest_path or self.cache_dir / STAGE2_CACHE_MANIFEST_NAME)
            .expanduser()
            .resolve()
        )
        self.manifest = load_stage2_i2v_manifest(
            self.manifest_path, expected_num_samples=expected_num_samples
        )
        if expected_num_samples is None:
            expected_num_samples = int(self.manifest["num_samples"])
        runtime = validate_stage2_i2v_runtime_bindings(
            self.manifest,
            metadata_path=metadata_path,
            source_cache_manifest_path=source_cache_manifest_path,
            negative_conditioning_manifest_path=negative_conditioning_manifest_path,
            config_contract_sha256=config_contract_sha256,
            config_launch_sha256=config_launch_sha256,
            expected_num_samples=int(expected_num_samples),
        )
        self.source_manifest = runtime["source_manifest"]
        self.negative_conditioning = runtime["negative_conditioning"]
        self.entries = list(self.manifest["records"])
        self.action_order = tuple(self.manifest["actions"]["ids"])
        self._action_to_index = {
            action_id: index for index, action_id in enumerate(self.action_order)
        }
        self.action_ids = [str(entry["action_id"]) for entry in self.entries]
        self.spatial_shapes = [
            tuple(int(value) for value in entry["latent_spatial_shape"])
            for entry in self.entries
        ]
        # A training process may start long after the audit CLI ran.  Verify and
        # parse one exact byte snapshot of every artifact before startup.
        for entry in self.entries:
            artifact_path = _safe_relative_file(
                self.cache_dir,
                entry["path"],
                label=f"record {entry['row_id']}.path",
            )
            _load_verified_safetensors_bytes(
                artifact_path,
                expected_size=entry["size"],
                expected_sha256=entry["sha256"],
                label=f"Stage-2 startup cache row {entry['row_id']}",
            )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, Any]:
        entry = self.entries[int(index)]
        artifact_path = _safe_relative_file(
            self.cache_dir, entry["path"], label=f"record {entry['row_id']}.path"
        )
        tensors = _load_verified_safetensors_bytes(
            artifact_path,
            expected_size=entry["size"],
            expected_sha256=entry["sha256"],
            label=f"Stage-2 cache row {entry['row_id']}",
        )
        facts = validate_stage2_cache_tensors(
            tensors,
            label=f"cache row {entry['row_id']}",
            full_audit=False,
        )
        expected_runtime_descriptions = {
            name: {
                "shape": description["shape"],
                "dtype": description["dtype"],
            }
            for name, description in entry["tensors"].items()
        }
        if facts["tensors"] != expected_runtime_descriptions:
            raise RuntimeError(
                f"Stage-2 cache tensor shape/dtype mismatch for row {entry['row_id']}."
            )
        # The source-image latent remains separate.  The F25 cache's slot zero
        # is audited as a sink slot and is never returned as a future target.
        return {
            "sample_id": int(entry["row_id"]),
            "action_id": str(entry["action_id"]),
            "action_index": self._action_to_index[str(entry["action_id"])],
            "initial_latent": tensors["initial_latent"],
            "real_future": tensors["video_latent"][1:25],
            "prompt_embeds": tensors["prompt_embeds"],
            "prompt_mask": tensors["prompt_mask"],
            "height": int(entry["height"]),
            "width": int(entry["width"]),
            "bucket": str(entry["bucket"]),
        }


def stage2_i2v_cache_collate(
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty Stage-2 batch.")
    initial_shapes = {tuple(item["initial_latent"].shape) for item in batch}
    future_shapes = {tuple(item["real_future"].shape) for item in batch}
    if len(initial_shapes) != 1 or len(future_shapes) != 1:
        raise ValueError(
            "A Stage-2 microbatch cannot mix latent shapes/orientations: "
            f"initial={sorted(initial_shapes)}, future={sorted(future_shapes)}."
        )
    result = {
        "sample_id": torch.tensor(
            [int(item["sample_id"]) for item in batch], dtype=torch.long
        ),
        "action_id": [str(item["action_id"]) for item in batch],
        "action_index": torch.tensor(
            [int(item["action_index"]) for item in batch], dtype=torch.long
        ),
        "initial_latent": torch.stack([item["initial_latent"] for item in batch]),
        "real_future": torch.stack([item["real_future"] for item in batch]),
        "prompt_embeds": torch.stack([item["prompt_embeds"] for item in batch]),
        "prompt_mask": torch.stack([item["prompt_mask"] for item in batch]),
        "height": torch.tensor(
            [int(item["height"]) for item in batch], dtype=torch.long
        ),
        "width": torch.tensor([int(item["width"]) for item in batch], dtype=torch.long),
        "bucket": [str(item["bucket"]) for item in batch],
    }
    if tuple(result["real_future"].shape[1:3]) != (24, 48):
        raise RuntimeError("Stage-2 collate produced a non-[B,24,48,H,W] real future.")
    if tuple(result["initial_latent"].shape[1:3]) != (1, 48):
        raise RuntimeError("Stage-2 collate produced a non-[B,1,48,H,W] sink.")
    return result


__all__ = [
    "STAGE2_CACHE_MANIFEST_NAME",
    "STAGE2_CACHE_SCHEMA",
    "STAGE2_CACHE_TENSOR_SCHEMA",
    "STAGE2_CACHE_TENSOR_SCHEMA_SHA256",
    "STAGE2_NEGATIVE_PROMPT_SHA256",
    "STAGE2_TEXT_ENCODING_OPERATOR_ATTESTATION",
    "STAGE2_TEXT_ENCODING_UPGRADE_KEY",
    "Stage2I2VCacheDataset",
    "audit_stage2_i2v_cache",
    "load_negative_conditioning",
    "load_source_cache_manifest",
    "load_stage2_i2v_manifest",
    "save_negative_conditioning_artifact",
    "stage2_i2v_cache_collate",
    "tensor_sha256",
    "upgrade_legacy_source_cache_manifest_text_encoding",
    "validate_stage2_cache_tensors",
    "validate_stage2_i2v_runtime_bindings",
    "write_negative_conditioning_manifest",
]
