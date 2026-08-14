"""Deterministic Stage-1 cache -> native Stage-2 F25 materialization.

This module never changes a Stage-1 artifact in place.  A provenance-complete
F25 input is copied byte-for-byte.  If its old manifest lacks the 97-frame
policy, a fresh full-F25 encode is used only as a bitwise verification oracle;
the original bytes are still the published bytes.  Only F24 publishes a newly
encoded video latent from presentation frames 0..96.  The resulting base
manifest must subsequently pass the text-encoding attestation before audit.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from utils.stage1_io import (
    aggregate_file_hash,
    atomic_output_path,
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
    tree_file_hashes,
)
from utils.stage2_i2v_data import (
    STAGE2_F25_FRAME_POLICY,
    STAGE2_F25_FRAME_POLICY_SHA256,
    STAGE2_F25_PREPARATION_SCHEMA,
    STAGE2_F25_PREPARATION_SCHEMA_VERSION,
    STAGE2_F25_SOURCE_CACHE_SCHEMA,
    STAGE2_F25_SOURCE_CACHE_SCHEMA_VERSION,
    STAGE2_F25_SUCCESS_NAME,
    STAGE2_F25_SUCCESS_SCHEMA,
    STAGE2_F25_SUCCESS_SCHEMA_VERSION,
    STAGE2_CACHE_MANIFEST_NAME,
    STAGE2_CACHE_SCHEMA,
    _stage1_manifest_declares_reusable_f25_policy,
    load_f25_preparation_input_manifest,
    stage2_f25_data_contract_sha256,
    tensor_sha256,
    validate_stage2_cache_tensors,
)

if TYPE_CHECKING:
    from utils.stage1_i2v_data import Stage1I2VRecord

F25_BASE_MANIFEST_NAME = "cache_manifest.json"
F25_OUTPUT_OWNERSHIP_NAME = ".stage2_f25_output.json"
F25_COMPLETION_SCHEMA = "longlive_stage2_i2v_f25_record_completion"
F25_COMPLETION_SCHEMA_VERSION = 1
F25_OUTPUT_OWNERSHIP_SCHEMA = "longlive_stage2_i2v_f25_output_ownership"
F25_OUTPUT_OWNERSHIP_SCHEMA_VERSION = 1
_OUTPUT_POLICY = {
    "independent_output_directory": True,
    "in_place_overwrite": False,
    "atomic_artifacts": True,
    "atomic_completion_sidecars": True,
    "stable_row_modulo_sharding": True,
}
F25_DECISIONS = ("reused_f25", "reencoded_f24", "reverified_f25")
_COMPLETION_KEYS = {
    "schema",
    "schema_version",
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
    "manifest_sha256",
}


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value != value.lower():
        raise RuntimeError(f"{label} must be a lowercase SHA256 hex string.")
    try:
        if len(bytes.fromhex(value)) != 32:
            raise ValueError
    except ValueError as exc:
        raise RuntimeError(f"{label} must be a lowercase SHA256 hex string.") from exc
    return value


def _load_self_hashed_json(path: Path, *, label: str) -> dict[str, Any]:
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    expected = _require_sha256(value.get("manifest_sha256"), f"{label} hash")
    unsigned = dict(value)
    unsigned.pop("manifest_sha256")
    if canonical_json_sha256(unsigned) != expected:
        raise RuntimeError(f"{label} self-hash mismatch: {path}")
    return value


def _tree_hash(path: str | os.PathLike[str]) -> str:
    return aggregate_file_hash(tree_file_hashes(path))


def _tensor_description(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sha256": tensor_sha256(tensor),
    }


def _tensor_descriptions(tensors: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    return {name: _tensor_description(tensors[name]) for name in sorted(tensors)}


def _source_model_hash(
    source_manifest: Mapping[str, Any], *, key: str, label: str
) -> str:
    try:
        entry = source_manifest["source_fingerprint"]["models"][key]
    except (KeyError, TypeError) as exc:
        raise RuntimeError(
            f"Stage-1 source manifest is missing {label} provenance."
        ) from exc
    if not isinstance(entry, Mapping) or not isinstance(entry.get("files"), list):
        raise RuntimeError(f"Stage-1 {label} provenance must include its file list.")
    value = _require_sha256(
        entry.get("aggregate_sha256"), f"Stage-1 {label} aggregate SHA256"
    )
    if aggregate_file_hash(entry["files"]) != value:
        raise RuntimeError(f"Stage-1 {label} file-list aggregate hash mismatch.")
    return value


def _source_fingerprint_records(
    source_manifest: Mapping[str, Any], *, expected_num_samples: int
) -> list[dict[str, Any]]:
    try:
        records = source_manifest["source_fingerprint"]["records"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("Stage-1 source fingerprint has no records.") from exc
    if not isinstance(records, list) or len(records) != expected_num_samples:
        raise RuntimeError("Stage-1 source fingerprint record count mismatch.")
    result: list[dict[str, Any]] = []
    for row_id, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise RuntimeError(f"Stage-1 fingerprint row {row_id} is not an object.")
        item = dict(raw)
        if item.get("row_id") != row_id or type(item.get("row_id")) is not int:
            raise RuntimeError("Stage-1 fingerprint row ids are not stable/contiguous.")
        for key in ("row_sha256", "video_sha256", "input_image_sha256"):
            _require_sha256(item.get(key), f"Stage-1 fingerprint row {row_id}.{key}")
        result.append(item)
    return result


def _safe_source_artifact(source_root: Path, relative: Any, *, row_id: int) -> Path:
    if not isinstance(relative, str) or not relative or relative != relative.strip():
        raise RuntimeError(f"Stage-1 source row {row_id}.path is invalid.")
    raw = Path(relative)
    if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
        raise RuntimeError(f"Stage-1 source row {row_id}.path must be relative.")
    candidate = source_root / raw
    current = candidate
    while current != source_root:
        if current.is_symlink():
            raise RuntimeError(
                f"Stage-1 source row {row_id}.path traverses a symlink: {current}"
            )
        if current.parent == current:
            raise RuntimeError(f"Stage-1 source row {row_id}.path escapes its cache.")
        current = current.parent
    path = candidate.resolve()
    try:
        path.relative_to(source_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Stage-1 source row {row_id}.path escapes its cache."
        ) from exc
    if not path.is_file():
        raise RuntimeError(f"Stage-1 source artifact is missing or a symlink: {path}")
    return path


def _read_verified_source_artifact(
    path: Path,
    *,
    source_entry: Mapping[str, Any],
    row_id: int,
    expected_source_aggregate_sha256: str,
) -> tuple[bytes, dict[str, torch.Tensor]]:
    size = source_entry.get("size")
    if type(size) is not int or size <= 0:
        raise RuntimeError(f"Stage-1 source row {row_id}.size must be positive.")
    expected_sha256 = _require_sha256(
        source_entry.get("sha256"), f"Stage-1 source row {row_id}.sha256"
    )
    payload = path.read_bytes()
    if len(payload) != size:
        raise RuntimeError(f"Stage-1 source row {row_id} artifact size mismatch.")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RuntimeError(f"Stage-1 source row {row_id} artifact hash mismatch.")
    header_size = int.from_bytes(payload[:8], byteorder="little", signed=False)
    if header_size <= 0 or 8 + header_size > len(payload):
        raise RuntimeError(
            f"Stage-1 source row {row_id} has an invalid safetensors header."
        )
    try:
        header = json.loads(payload[8 : 8 + header_size].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Stage-1 source row {row_id} has invalid safetensors metadata."
        ) from exc
    metadata = header.get("__metadata__")
    expected_metadata = {
        "schema": "longlive_stage1_i2v_cache_record",
        "schema_version": "1",
        "row_id": str(row_id),
        "row_sha256": source_entry.get("row_sha256"),
        "source_aggregate_sha256": expected_source_aggregate_sha256,
        "prompt_mode": "repeat_global",
    }
    if not isinstance(metadata, Mapping):
        raise RuntimeError(
            f"Stage-1 source row {row_id} has no cache provenance metadata."
        )
    for key, expected in expected_metadata.items():
        if expected is None or metadata.get(key) != expected:
            raise RuntimeError(
                f"Stage-1 source row {row_id} safetensors metadata {key} mismatch."
            )

    from safetensors.torch import load

    tensors = load(payload)
    return payload, tensors


def _input_kind(
    tensors: Mapping[str, torch.Tensor],
    *,
    source_manifest: Mapping[str, Any],
    expected_spatial_shape: tuple[int, int],
    row_id: int,
) -> str:
    from utils.stage1_i2v_data import (
        validate_cache_tensors as validate_stage1_cache_tensors,
    )

    video = tensors.get("video_latent")
    if not isinstance(video, torch.Tensor):
        raise RuntimeError(f"Stage-1 source row {row_id} has no video_latent tensor.")
    if video.ndim == 4 and tuple(video.shape[:2]) == (24, 48):
        validate_stage1_cache_tensors(
            tensors, expected_spatial_shapes=(expected_spatial_shape,)
        )
        return "reencoded_f24"
    if video.ndim == 4 and tuple(video.shape[:2]) == (25, 48):
        validate_stage2_cache_tensors(
            tensors,
            expected_spatial_shapes=(expected_spatial_shape,),
            label=f"F25 migration input row {row_id}",
        )
        if _stage1_manifest_declares_reusable_f25_policy(source_manifest):
            return "reused_f25"
        return "reverified_f25"
    raise RuntimeError(
        f"Stage-1 source row {row_id} video_latent must be F24 or F25; "
        f"got {tuple(video.shape)}. This tool never pads, truncates, or duplicates."
    )


def _preparation_contract(
    *,
    source_manifest_path: Path,
    source_manifest: Mapping[str, Any],
    metadata_path: Path,
    expected_num_samples: int,
    config_path: Path,
    config_contract_sha256: str,
    config_launch_sha256: str,
    vae_aggregate_sha256: str,
) -> dict[str, Any]:
    preparation: dict[str, Any] = {
        "schema": STAGE2_F25_PREPARATION_SCHEMA,
        "schema_version": STAGE2_F25_PREPARATION_SCHEMA_VERSION,
        "input_manifest": {
            "path": str(source_manifest_path),
            "file_sha256": sha256_file(source_manifest_path),
            "manifest_sha256": source_manifest["manifest_sha256"],
            "source_fingerprint_sha256": source_manifest["source_fingerprint"][
                "aggregate_sha256"
            ],
            "schema": source_manifest["schema"],
            "schema_version": source_manifest["schema_version"],
        },
        "config": {
            "path": str(config_path),
            "file_sha256": sha256_file(config_path),
            "contract_sha256": _require_sha256(
                config_contract_sha256, "Stage-2 config contract SHA256"
            ),
            "launch_sha256": _require_sha256(
                config_launch_sha256, "Stage-2 config launch SHA256"
            ),
            "data_contract_sha256": stage2_f25_data_contract_sha256(
                expected_num_samples=expected_num_samples,
                metadata_sha256=sha256_file(metadata_path),
            ),
        },
        "metadata": {
            "path": str(metadata_path),
            "file_sha256": sha256_file(metadata_path),
        },
        "frame_policy": copy.deepcopy(STAGE2_F25_FRAME_POLICY),
        "frame_policy_sha256": STAGE2_F25_FRAME_POLICY_SHA256,
        "vae": {
            "aggregate_sha256": _require_sha256(
                vae_aggregate_sha256, "Stage-2 F25 VAE aggregate SHA256"
            )
        },
        "output_policy": copy.deepcopy(_OUTPUT_POLICY),
    }
    preparation["contract_sha256"] = canonical_json_sha256(preparation)
    return preparation


def _ownership_payload(contract_sha256: str) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema": F25_OUTPUT_OWNERSHIP_SCHEMA,
        "schema_version": F25_OUTPUT_OWNERSHIP_SCHEMA_VERSION,
        "preparation_contract_sha256": contract_sha256,
    }
    value["manifest_sha256"] = canonical_json_sha256(value)
    return value


def _ensure_output_ownership(output_dir: Path, contract_sha256: str) -> None:
    marker = output_dir / F25_OUTPUT_OWNERSHIP_NAME
    expected = _ownership_payload(contract_sha256)
    if marker.exists():
        if (
            marker.is_symlink()
            or _load_self_hashed_json(marker, label="Stage-2 F25 output ownership")
            != expected
        ):
            raise RuntimeError(
                "Output directory belongs to a different F25 preparation contract."
            )
        return
    unexpected = [path for path in output_dir.iterdir() if path.name != marker.name]
    if unexpected:
        raise RuntimeError(
            "Refusing a non-empty unowned F25 output directory; choose a new directory."
        )
    atomic_write_json(marker, expected)


def _completion_path(output_dir: Path, row_id: int) -> Path:
    return output_dir / f"sample_{row_id:06d}.complete.json"


def _artifact_path(output_dir: Path, row_id: int) -> Path:
    return output_dir / f"sample_{row_id:06d}.safetensors"


def _completion_payload(
    *,
    record: Stage1I2VRecord,
    decision: str,
    input_entry: Mapping[str, Any],
    input_tensors: Mapping[str, torch.Tensor],
    output_path: Path,
    output_tensors: Mapping[str, torch.Tensor],
    fingerprint_record: Mapping[str, Any],
    vae_aggregate_sha256: str,
    preparation_contract_sha256: str,
    verification: Mapping[str, Any],
) -> dict[str, Any]:
    output_payload = output_path.read_bytes()
    value: dict[str, Any] = {
        "schema": F25_COMPLETION_SCHEMA,
        "schema_version": F25_COMPLETION_SCHEMA_VERSION,
        "row_id": record.row_id,
        "row_sha256": record.row_sha256,
        "height": record.height,
        "width": record.width,
        "bucket": record.bucket,
        "path": output_path.name,
        "size": len(output_payload),
        "sha256": hashlib.sha256(output_payload).hexdigest(),
        "tensors": _tensor_descriptions(output_tensors),
        "decision": decision,
        "input_artifact": {
            "path": input_entry["path"],
            "size": input_entry["size"],
            "sha256": input_entry["sha256"],
            "tensors": _tensor_descriptions(input_tensors),
        },
        "source_video_sha256": fingerprint_record["video_sha256"],
        "source_input_image_sha256": fingerprint_record["input_image_sha256"],
        "vae_aggregate_sha256": vae_aggregate_sha256,
        "frame_policy_sha256": STAGE2_F25_FRAME_POLICY_SHA256,
        "preparation_contract_sha256": preparation_contract_sha256,
        "verification": dict(verification),
    }
    if "action_id" in input_entry:
        value["action_id"] = input_entry["action_id"]
    value["manifest_sha256"] = canonical_json_sha256(value)
    return value


def _validate_completion(
    path: Path,
    *,
    record: Stage1I2VRecord,
    decision: str,
    input_entry: Mapping[str, Any],
    input_tensors: Mapping[str, torch.Tensor],
    fingerprint_record: Mapping[str, Any],
    vae_aggregate_sha256: str,
    preparation_contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]] | None:
    if not path.is_file() or path.is_symlink():
        return None
    try:
        completion = _load_self_hashed_json(
            path, label=f"F25 row {record.row_id} completion"
        )
        expected_keys = set(_COMPLETION_KEYS)
        if "action_id" in input_entry:
            expected_keys.add("action_id")
        if set(completion) != expected_keys:
            return None
        expected = {
            "schema": F25_COMPLETION_SCHEMA,
            "schema_version": F25_COMPLETION_SCHEMA_VERSION,
            "row_id": record.row_id,
            "row_sha256": record.row_sha256,
            "decision": decision,
            "preparation_contract_sha256": preparation_contract_sha256,
            "source_video_sha256": fingerprint_record["video_sha256"],
            "source_input_image_sha256": fingerprint_record["input_image_sha256"],
            "vae_aggregate_sha256": vae_aggregate_sha256,
            "frame_policy_sha256": STAGE2_F25_FRAME_POLICY_SHA256,
        }
        if any(completion.get(key) != value for key, value in expected.items()):
            return None
        input_binding = completion.get("input_artifact")
        if not isinstance(input_binding, Mapping) or any(
            input_binding.get(key) != input_entry.get(key)
            for key in ("path", "size", "sha256")
        ):
            return None
        if input_binding.get("tensors") != _tensor_descriptions(input_tensors):
            return None
        if (
            "action_id" in input_entry
            and completion.get("action_id") != input_entry["action_id"]
        ):
            return None
        expected_artifact = _artifact_path(path.parent, record.row_id)
        if completion.get("path") != expected_artifact.name:
            return None
        artifact = path.parent / completion["path"]
        if artifact.is_symlink() or not artifact.is_file():
            return None
        payload = artifact.read_bytes()
        if (
            len(payload) != completion["size"]
            or hashlib.sha256(payload).hexdigest() != completion["sha256"]
        ):
            return None
        from safetensors.torch import load

        tensors = load(payload)
        validate_stage2_cache_tensors(
            tensors,
            expected_spatial_shapes=((record.height // 16, record.width // 16),),
            label=f"resumed F25 row {record.row_id}",
        )
        if completion["tensors"] != _tensor_descriptions(tensors):
            return None
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
        if completion.get("verification") != expected_verification:
            return None
        if decision in {"reused_f25", "reverified_f25"}:
            if (
                completion["size"] != input_entry["size"]
                or completion["sha256"] != input_entry["sha256"]
                or completion["tensors"] != _tensor_descriptions(input_tensors)
            ):
                return None
        else:
            if not torch.equal(
                tensors["video_latent"][:24], input_tensors["video_latent"]
            ):
                return None
            for name in ("initial_latent", "prompt_embeds", "prompt_mask"):
                if _tensor_description(tensors[name]) != _tensor_description(
                    input_tensors[name]
                ):
                    return None
        return completion, tensors
    except (KeyError, OSError, RuntimeError, ValueError):
        return None


def _save_reencoded_artifact(
    path: Path,
    tensors: Mapping[str, torch.Tensor],
    *,
    record: Stage1I2VRecord,
    decision: str,
    source_sha256: str,
    preparation_contract_sha256: str,
) -> None:
    from safetensors.torch import save_file

    canonical = {
        name: tensor.detach().contiguous().cpu() for name, tensor in tensors.items()
    }
    validate_stage2_cache_tensors(
        canonical,
        expected_spatial_shapes=((record.height // 16, record.width // 16),),
        label=f"reencoded F25 row {record.row_id}",
    )
    metadata = {
        "schema": "longlive_stage2_i2v_f25_record",
        "schema_version": "1",
        "row_id": str(record.row_id),
        "row_sha256": record.row_sha256,
        "decision": decision,
        "source_artifact_sha256": source_sha256,
        "preparation_contract_sha256": preparation_contract_sha256,
    }
    with atomic_output_path(path, suffix=".safetensors.tmp") as temporary:
        save_file(canonical, str(temporary), metadata=metadata)


def _encode_f25_video(
    record: Stage1I2VRecord,
    *,
    vae: Any,
    device: torch.device,
    expected_video_sha256: str,
    decode_video: Callable[..., torch.Tensor],
) -> torch.Tensor:
    if sha256_file(record.video_path) != expected_video_sha256:
        raise RuntimeError(
            f"row {record.row_id}: source video hash changed before decode."
        )
    pixels = decode_video(
        record,
        min_source_frames=97,
        selected_frame_start=0,
        selected_frame_count=97,
        expected_fps=24.0,
        fps_abs_tolerance=1.0e-3,
    )
    if tuple(pixels.shape[:3]) != (1, 3, 97):
        raise RuntimeError(
            f"row {record.row_id}: decoded pixels must be [1,3,97,H,W], "
            f"got {tuple(pixels.shape)}."
        )
    with torch.inference_mode():
        encoded = vae.encode_to_latent(pixels.to(device=device, dtype=torch.bfloat16))
    if sha256_file(record.video_path) != expected_video_sha256:
        raise RuntimeError(f"row {record.row_id}: source video changed during decode.")
    if tuple(encoded.shape[:3]) != (1, 25, 48):
        raise RuntimeError(
            f"row {record.row_id}: VAE(97 frames) must return [1,25,48,H,W], "
            f"got {tuple(encoded.shape)}."
        )
    expected_spatial = (record.height // 16, record.width // 16)
    if tuple(encoded.shape[-2:]) != expected_spatial:
        raise RuntimeError(
            f"row {record.row_id}: VAE latent spatial shape mismatch: "
            f"{tuple(encoded.shape[-2:])} != {expected_spatial}."
        )
    return encoded[0].to(device="cpu", dtype=torch.bfloat16).contiguous()


def _barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def _global_any(value: bool, *, device: torch.device) -> bool:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return value
    tensor = torch.tensor([int(value)], dtype=torch.int32, device=device)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return bool(tensor.item())


def stable_f25_row_ids(
    expected_num_samples: int, *, rank: int, world_size: int
) -> tuple[int, ...]:
    """Return the producer's one-and-only stable row modulo assignment."""

    if type(expected_num_samples) is not int or expected_num_samples <= 0:
        raise ValueError("expected_num_samples must be a positive integer.")
    if type(world_size) is not int or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank/world_size are invalid.")
    return tuple(
        row_id for row_id in range(expected_num_samples) if row_id % world_size == rank
    )


def _validated_derived_output_names(output_root: Path) -> set[str]:
    """Allow only the two deterministic downstream manifests on idempotent reruns."""

    allowed: set[str] = set()
    candidates = {
        STAGE2_CACHE_MANIFEST_NAME: STAGE2_CACHE_SCHEMA,
        "cache_manifest.attested.json": STAGE2_F25_SOURCE_CACHE_SCHEMA,
    }
    for name, expected_schema in candidates.items():
        path = output_root / name
        if not path.exists():
            continue
        value = _load_self_hashed_json(path, label=f"derived F25 output {name}")
        if value.get("schema") != expected_schema:
            raise RuntimeError(f"Derived F25 output {name} has the wrong schema.")
        if name == "cache_manifest.attested.json" and (
            "stage2_text_encoding_upgrade" not in value
        ):
            raise RuntimeError(
                "F25 attested manifest is missing its text upgrade block."
            )
        allowed.add(name)
    return allowed


def prepare_stage2_f25_cache(
    *,
    metadata_path: str | os.PathLike[str],
    source_cache_manifest_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    config_contract_sha256: str,
    config_launch_sha256: str,
    expected_num_samples: int,
    rank: int,
    world_size: int,
    device: torch.device,
    vae_checkpoint_path: str | os.PathLike[str] | None = None,
    vae_factory: Callable[[str | os.PathLike[str], torch.device], Any] | None = None,
    decode_video: Callable[..., torch.Tensor] | None = None,
) -> Path | None:
    """Materialize an independently owned native F25 cache and base manifest."""

    from utils.stage1_i2v_data import decode_stage1_video, load_stage1_i2v_manifest

    if decode_video is None:
        decode_video = decode_stage1_video

    if type(expected_num_samples) is not int or expected_num_samples <= 0:
        raise ValueError("expected_num_samples must be a positive integer.")
    if type(world_size) is not int or world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("rank/world_size are invalid.")
    metadata_path = Path(metadata_path).expanduser().resolve()
    source_manifest_path = Path(source_cache_manifest_path).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    output_root_input = Path(output_dir).expanduser()
    if output_root_input.is_symlink():
        raise RuntimeError("Stage-2 F25 output directory must not be a symlink.")
    output_root = output_root_input.resolve()
    source_root = source_manifest_path.parent.resolve()
    if (
        output_root == source_root
        or output_root.is_relative_to(source_root)
        or source_root.is_relative_to(output_root)
    ):
        raise RuntimeError(
            "Stage-2 F25 output must be an independent non-nested directory."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    records = load_stage1_i2v_manifest(
        metadata_path,
        expected_num_samples=expected_num_samples,
        require_files=True,
        validate_images=True,
    )
    source_manifest = load_f25_preparation_input_manifest(
        source_manifest_path, expected_num_samples=expected_num_samples
    )
    source_entries = list(source_manifest["records"])
    source_aggregate_sha256 = source_manifest["source_fingerprint"]["aggregate_sha256"]
    fingerprint_records = _source_fingerprint_records(
        source_manifest, expected_num_samples=expected_num_samples
    )
    vae_hash = _source_model_hash(
        source_manifest, key="vae_checkpoint", label="VAE checkpoint"
    )
    # Text assets are not loaded here, but their immutable hashes must exist so
    # the later F25 text attestation can bind the copied prompt tensors.
    _source_model_hash(source_manifest, key="t5_checkpoint", label="T5 checkpoint")
    _source_model_hash(source_manifest, key="tokenizer_dir", label="tokenizer")

    preparation = _preparation_contract(
        source_manifest_path=source_manifest_path,
        source_manifest=source_manifest,
        metadata_path=metadata_path,
        expected_num_samples=expected_num_samples,
        config_path=config_path,
        config_contract_sha256=config_contract_sha256,
        config_launch_sha256=config_launch_sha256,
        vae_aggregate_sha256=vae_hash,
    )
    contract_sha256 = preparation["contract_sha256"]
    if rank == 0:
        _ensure_output_ownership(output_root, contract_sha256)
    _barrier()
    # Every rank independently verifies the ownership marker after rank 0's write.
    _ensure_output_ownership(output_root, contract_sha256)

    assigned_row_ids = set(
        stable_f25_row_ids(expected_num_samples, rank=rank, world_size=world_size)
    )
    assigned: list[
        tuple[
            Stage1I2VRecord,
            dict[str, Any],
            dict[str, Any],
            bytes,
            dict[str, torch.Tensor],
            str,
        ]
    ] = []
    source_scan_completed = 0
    source_scan_total = len(assigned_row_ids)
    for record, raw_entry, fingerprint_record in zip(
        records, source_entries, fingerprint_records
    ):
        if record.row_id not in assigned_row_ids:
            continue
        entry = dict(raw_entry)
        if (
            entry.get("row_id") != record.row_id
            or entry.get("row_sha256") != record.row_sha256
        ):
            raise RuntimeError(
                f"row {record.row_id}: metadata/source manifest mismatch."
            )
        if fingerprint_record["row_sha256"] != record.row_sha256:
            raise RuntimeError(f"row {record.row_id}: source fingerprint row mismatch.")
        if sha256_file(record.video_path) != fingerprint_record["video_sha256"]:
            raise RuntimeError(f"row {record.row_id}: raw video hash mismatch.")
        if (
            sha256_file(record.input_image_path)
            != fingerprint_record["input_image_sha256"]
        ):
            raise RuntimeError(f"row {record.row_id}: input image hash mismatch.")
        source_artifact = _safe_source_artifact(
            source_root, entry.get("path"), row_id=record.row_id
        )
        payload, input_tensors = _read_verified_source_artifact(
            source_artifact,
            source_entry=entry,
            row_id=record.row_id,
            expected_source_aggregate_sha256=source_aggregate_sha256,
        )
        decision = _input_kind(
            input_tensors,
            source_manifest=source_manifest,
            expected_spatial_shape=(record.height // 16, record.width // 16),
            row_id=record.row_id,
        )
        assigned.append(
            (record, entry, fingerprint_record, payload, input_tensors, decision)
        )
        source_scan_completed += 1
        if (
            source_scan_completed % 10 == 0
            or source_scan_completed == source_scan_total
        ):
            print(
                f"[rank {rank}/{world_size}] source_scan="
                f"{source_scan_completed}/{source_scan_total} row_id={record.row_id}",
                flush=True,
            )

    resumed_rows: set[int] = set()
    for (
        record,
        entry,
        fingerprint_record,
        _payload,
        input_tensors,
        decision,
    ) in assigned:
        if (
            _validate_completion(
                _completion_path(output_root, record.row_id),
                record=record,
                decision=decision,
                input_entry=entry,
                input_tensors=input_tensors,
                fingerprint_record=fingerprint_record,
                vae_aggregate_sha256=vae_hash,
                preparation_contract_sha256=contract_sha256,
            )
            is not None
        ):
            resumed_rows.add(record.row_id)
    decision_counts = Counter(item[-1] for item in assigned)
    print(
        f"[rank {rank}/{world_size}] device={device} assigned={len(assigned)} "
        f"reused_f25={decision_counts['reused_f25']} "
        f"reverified_f25={decision_counts['reverified_f25']} "
        f"reencoded_f24={decision_counts['reencoded_f24']} "
        f"pending={len(assigned) - len(resumed_rows)} resumed={len(resumed_rows)}",
        flush=True,
    )
    needs_vae = any(
        item[-1] != "reused_f25" and item[0].row_id not in resumed_rows
        for item in assigned
    )
    global_needs_vae = _global_any(needs_vae, device=device)
    if global_needs_vae:
        if vae_checkpoint_path is None:
            raise RuntimeError(
                "At least one row requires 97-frame re-encoding; --vae-checkpoint is required."
            )
        actual_vae_hash = _tree_hash(vae_checkpoint_path)
        if actual_vae_hash != vae_hash:
            raise RuntimeError(
                "VAE checkpoint differs from the Stage-1 source cache provenance."
            )
    else:
        actual_vae_hash = vae_hash

    vae = None
    if needs_vae:
        if vae_factory is None:
            from utils.wan_5b_wrapper import WanVAEWrapper

            def vae_factory(path: str | os.PathLike[str], target: torch.device) -> Any:
                return (
                    WanVAEWrapper(vae_checkpoint=path)
                    .eval()
                    .requires_grad_(False)
                    .to(device=target)
                )

        vae = vae_factory(vae_checkpoint_path, device)

    counts: Counter[str] = Counter()
    completed_pending = 0
    total_pending = len(assigned) - len(resumed_rows)
    for record, entry, fingerprint_record, payload, input_tensors, decision in assigned:
        completion_path = _completion_path(output_root, record.row_id)
        if record.row_id in resumed_rows:
            counts[decision] += 1
            continue

        output_path = _artifact_path(output_root, record.row_id)
        if decision == "reused_f25":
            atomic_write_bytes(output_path, payload)
            if sha256_file(output_path) != entry["sha256"]:
                raise RuntimeError(
                    f"row {record.row_id}: F25 byte-for-byte reuse failed."
                )
            output_tensors = input_tensors
            verification = {
                "output_reuses_input_bytes": True,
                "f24_prefix_exact": None,
                "f25_reverification_exact": None,
            }
        else:
            if vae is None:
                raise AssertionError(
                    "VAE was not loaded for a row requiring re-encoding"
                )
            new_video = _encode_f25_video(
                record,
                vae=vae,
                device=device,
                expected_video_sha256=fingerprint_record["video_sha256"],
                decode_video=decode_video,
            )
            if decision == "reverified_f25":
                if not torch.equal(new_video, input_tensors["video_latent"]):
                    raise RuntimeError(
                        f"row {record.row_id}: unproven input F25 differs bitwise from "
                        "a fresh 0..96 encode; refusing to reuse or replace it."
                    )
                atomic_write_bytes(output_path, payload)
                if sha256_file(output_path) != entry["sha256"]:
                    raise RuntimeError(
                        f"row {record.row_id}: reverified F25 byte reuse failed."
                    )
                output_tensors = input_tensors
                verification = {
                    "output_reuses_input_bytes": True,
                    "f24_prefix_exact": None,
                    "f25_reverification_exact": True,
                }
            else:
                if not torch.equal(new_video[:24], input_tensors["video_latent"]):
                    raise RuntimeError(
                        f"row {record.row_id}: new F25[:24] differs bitwise from the old "
                        "F24 cache. Refusing publication because VAE/preprocessing "
                        "provenance is inconsistent."
                    )
                output_tensors = {
                    "video_latent": new_video,
                    "initial_latent": input_tensors["initial_latent"],
                    "prompt_embeds": input_tensors["prompt_embeds"],
                    "prompt_mask": input_tensors["prompt_mask"],
                }
                for name in ("initial_latent", "prompt_embeds", "prompt_mask"):
                    if tensor_sha256(output_tensors[name]) != tensor_sha256(
                        input_tensors[name]
                    ):
                        raise AssertionError(
                            f"row {record.row_id}: {name} was not copied exactly"
                        )
                _save_reencoded_artifact(
                    output_path,
                    output_tensors,
                    record=record,
                    decision=decision,
                    source_sha256=entry["sha256"],
                    preparation_contract_sha256=contract_sha256,
                )
                verification = {
                    "output_reuses_input_bytes": False,
                    "f24_prefix_exact": True,
                    "f25_reverification_exact": None,
                }
        completion = _completion_payload(
            record=record,
            decision=decision,
            input_entry=entry,
            input_tensors=input_tensors,
            output_path=output_path,
            output_tensors=output_tensors,
            fingerprint_record=fingerprint_record,
            vae_aggregate_sha256=vae_hash,
            preparation_contract_sha256=contract_sha256,
            verification=verification,
        )
        atomic_write_json(completion_path, completion)
        counts[decision] += 1
        completed_pending += 1
        if completed_pending % 10 == 0 or completed_pending == total_pending:
            print(
                f"[rank {rank}/{world_size}] completed={completed_pending}/"
                f"{total_pending} row_id={record.row_id} decision={decision}",
                flush=True,
            )

    if global_needs_vae and _tree_hash(vae_checkpoint_path) != actual_vae_hash:
        raise RuntimeError("VAE checkpoint tree changed during F25 materialization.")
    _barrier()
    if rank != 0:
        return None

    entries: list[dict[str, Any]] = []
    summary: Counter[str] = Counter()
    final_verified = 0
    for record, input_entry, fingerprint_record in zip(
        records, source_entries, fingerprint_records
    ):
        if sha256_file(record.video_path) != fingerprint_record["video_sha256"]:
            raise RuntimeError(
                f"row {record.row_id}: raw video changed before F25 manifest publication."
            )
        if (
            sha256_file(record.input_image_path)
            != fingerprint_record["input_image_sha256"]
        ):
            raise RuntimeError(
                f"row {record.row_id}: input image changed before F25 manifest publication."
            )
        input_path = _safe_source_artifact(
            source_root, input_entry.get("path"), row_id=record.row_id
        )
        _, input_tensors = _read_verified_source_artifact(
            input_path,
            source_entry=input_entry,
            row_id=record.row_id,
            expected_source_aggregate_sha256=source_aggregate_sha256,
        )
        decision = _input_kind(
            input_tensors,
            source_manifest=source_manifest,
            expected_spatial_shape=(record.height // 16, record.width // 16),
            row_id=record.row_id,
        )
        validated = _validate_completion(
            _completion_path(output_root, record.row_id),
            record=record,
            decision=decision,
            input_entry=input_entry,
            input_tensors=input_tensors,
            fingerprint_record=fingerprint_record,
            vae_aggregate_sha256=vae_hash,
            preparation_contract_sha256=contract_sha256,
        )
        if validated is None:
            raise RuntimeError(
                f"row {record.row_id}: missing/invalid completion artifact."
            )
        completion, _ = validated
        entry = dict(completion)
        entry.pop("schema")
        entry.pop("schema_version")
        entry.pop("manifest_sha256")
        entries.append(entry)
        summary[decision] += 1
        final_verified += 1
        if final_verified % 25 == 0 or final_verified == expected_num_samples:
            print(
                f"[rank 0/{world_size}] final_verify="
                f"{final_verified}/{expected_num_samples} row_id={record.row_id}",
                flush=True,
            )

    finalized_preparation = copy.deepcopy(preparation)
    finalized_preparation["summary"] = {key: summary[key] for key in F25_DECISIONS}
    fingerprint = copy.deepcopy(source_manifest["source_fingerprint"])
    fingerprint.pop("aggregate_sha256")
    fingerprint["preprocessing"] = copy.deepcopy(STAGE2_F25_FRAME_POLICY)
    fingerprint["cache_schema"] = STAGE2_F25_SOURCE_CACHE_SCHEMA
    fingerprint["cache_schema_version"] = STAGE2_F25_SOURCE_CACHE_SCHEMA_VERSION
    fingerprint["preparation_contract_sha256"] = contract_sha256
    fingerprint["aggregate_sha256"] = canonical_json_sha256(fingerprint)
    manifest: dict[str, Any] = {
        "schema": STAGE2_F25_SOURCE_CACHE_SCHEMA,
        "schema_version": STAGE2_F25_SOURCE_CACHE_SCHEMA_VERSION,
        "num_samples": expected_num_samples,
        "source_fingerprint": fingerprint,
        "preparation": finalized_preparation,
        "records": entries,
    }
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)

    if (
        sha256_file(source_manifest_path)
        != preparation["input_manifest"]["file_sha256"]
    ):
        raise RuntimeError(
            "Stage-1 source manifest changed during F25 materialization."
        )
    if sha256_file(metadata_path) != preparation["metadata"]["file_sha256"]:
        raise RuntimeError("Metadata changed during F25 materialization.")
    if sha256_file(config_path) != preparation["config"]["file_sha256"]:
        raise RuntimeError("Stage-2 config changed during F25 materialization.")
    manifest_path = output_root / F25_BASE_MANIFEST_NAME
    success_path = output_root / STAGE2_F25_SUCCESS_NAME
    expected_names = {F25_OUTPUT_OWNERSHIP_NAME, manifest_path.name, success_path.name}
    expected_names.update(
        _artifact_path(output_root, row_id).name
        for row_id in range(expected_num_samples)
    )
    expected_names.update(
        _completion_path(output_root, row_id).name
        for row_id in range(expected_num_samples)
    )
    expected_names.update(_validated_derived_output_names(output_root))
    extras = sorted(
        path.name for path in output_root.iterdir() if path.name not in expected_names
    )
    if extras:
        raise RuntimeError(f"Unexpected files in owned F25 output directory: {extras}")
    atomic_write_json(manifest_path, manifest)
    success: dict[str, Any] = {
        "schema": STAGE2_F25_SUCCESS_SCHEMA,
        "schema_version": STAGE2_F25_SUCCESS_SCHEMA_VERSION,
        "manifest_path": manifest_path.name,
        "manifest_file_sha256": sha256_file(manifest_path),
        "source_manifest_sha256": manifest["manifest_sha256"],
        "preparation_contract_sha256": contract_sha256,
        "num_samples": expected_num_samples,
    }
    success["manifest_sha256"] = canonical_json_sha256(success)
    atomic_write_json(success_path, success)
    actual_names = {path.name for path in output_root.iterdir()}
    if actual_names != expected_names:
        raise RuntimeError(
            "F25 output file-set changed during final publication: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}."
        )
    return manifest_path
