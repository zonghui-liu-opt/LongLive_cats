"""Canonical provenance manifests for Stage-2 role initialization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

from utils.stage1_io import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)

STAGE1_MERGE_MANIFEST_SCHEMA = "longlive_stage1_merge_manifest"
STAGE1_MERGE_MANIFEST_VERSION = 2
STAGE2_TEACHER_MANIFEST_SCHEMA = "longlive_stage2_teacher_manifest"
STAGE2_TEACHER_MANIFEST_VERSION = 1
STAGE2_ROLE_INIT_MANIFEST_SCHEMA = "longlive_stage2_role_init_manifest"
STAGE2_ROLE_INIT_MANIFEST_VERSION = 1
ROLE_INIT_COMPLETE_MARKER = "ROLE_INIT_COMPLETE"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TEACHER_CHECKPOINT_FORMATS = {
    "longlive_wrapper_pt",
    "wan_native_transformer",
}
_TEACHER_STATE_DICT_SELECTORS = {
    "generator",
    "real_score",
    "model",
    "root",
}
_REQUIRED_TEACHER_SEMANTIC_CONTRACT = {
    "backbone": "bidirectional_ti2v",
    "model_type": "ti2v",
    "prediction_type": "flow",
    "timestep_scope": "video_global",
    "latent_channels": 48,
    "patch_size": [1, 2, 2],
    "num_blocks": 30,
    "hidden_size": 3072,
    "ffn_size": 14336,
}
_LOCKED_WAN_ARCHITECTURE = {
    "model_type": "ti2v",
    "patch_size": [1, 2, 2],
    "text_len": 512,
    "in_dim": 48,
    "dim": 3072,
    "ffn_dim": 14336,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 48,
    "num_heads": 24,
    "num_layers": 30,
    "window_size": [-1, -1],
    "qk_norm": True,
    "cross_attn_norm": True,
    "eps": 1e-6,
}


def _file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _source_file_entry(path: Path, *, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _teacher_checkpoint_sources(
    checkpoint_path: str | os.PathLike[str], checkpoint_format: str
) -> tuple[Path, list[dict[str, Any]], list[dict[str, Any]]]:
    requested = Path(checkpoint_path).expanduser().resolve()
    if checkpoint_format == "wan_native_transformer":
        from wan_5b.textimage2video import (
            resolve_wan_checkpoint_path,
            wan_checkpoint_source_files,
        )

        resolved = Path(resolve_wan_checkpoint_path(requested)).resolve()
        paths = [
            Path(value).resolve() for value in wan_checkpoint_source_files(requested)
        ]
        root = resolved.parent
        names = [path.relative_to(root).as_posix() for path in paths]
    else:
        if not requested.is_file():
            raise FileNotFoundError(requested)
        resolved = requested
        paths = [resolved]
        names = [resolved.name]
    entries = [_source_file_entry(path, name=name) for path, name in zip(paths, names)]
    runtime_files = [
        {
            **entry,
            "path": str(path),
            "identity": _file_identity(path),
        }
        for path, entry in zip(paths, entries)
    ]
    return resolved, entries, runtime_files


def _native_safetensors_contract(
    resolved: Path, runtime_files: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not str(resolved).endswith((".safetensors", ".safetensors.index.json")):
        raise ValueError(
            "Stage-2 native teacher currently requires safetensors; .bin sources "
            "cannot prove immutable BF16 dtype without materializing every shard"
        )
    from safetensors import safe_open

    schema = []
    for entry in runtime_files:
        path = Path(entry["path"])
        if not path.name.endswith(".safetensors"):
            continue
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in sorted(handle.keys()):
                tensor_slice = handle.get_slice(key)
                dtype = tensor_slice.get_dtype()
                if dtype != "BF16":
                    raise TypeError(
                        "Stage-2 native teacher source tensor must be BF16: "
                        f"{entry['name']}:{key}={dtype}"
                    )
                schema.append(
                    {
                        "file": entry["name"],
                        "key": key,
                        "shape": list(tensor_slice.get_shape()),
                        "dtype": "bfloat16",
                    }
                )
    if not schema:
        raise ValueError("Stage-2 native teacher contains no safetensors tensors")
    return {
        "source_dtype": "bfloat16",
        "tensor_count": len(schema),
        "tensor_schema_sha256": canonical_json_sha256(schema),
    }


def _stage2_architecture_contract(
    architecture_root: str | os.PathLike[str],
) -> dict[str, Any]:
    root = Path(architecture_root).expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            "Stage-2 requires an explicit architecture_root/config.json: "
            f"{config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise TypeError("Stage-2 architecture config.json must be an object")
    explicit_fields = sorted(set(_LOCKED_WAN_ARCHITECTURE) & set(raw))
    inferred_fields = sorted(set(_LOCKED_WAN_ARCHITECTURE) - set(raw))
    resolved = {
        key: (
            list(raw[key])
            if isinstance(raw.get(key), tuple)
            else raw.get(key, expected)
        )
        for key, expected in _LOCKED_WAN_ARCHITECTURE.items()
    }
    if resolved != _LOCKED_WAN_ARCHITECTURE:
        wrong = {
            key: {"expected": expected, "actual": resolved.get(key)}
            for key, expected in _LOCKED_WAN_ARCHITECTURE.items()
            if resolved.get(key) != expected
        }
        raise ValueError(f"Stage-2 Wan architecture contract mismatch: {wrong}")
    return {
        "config_name": "config.json",
        "config_sha256": sha256_file(config_path),
        "resolved_fields": resolved,
        "resolved_fields_sha256": canonical_json_sha256(resolved),
        "explicit_fields": explicit_fields,
        "inferred_canonical_fields": inferred_fields,
    }


def _read_safetensors_contract(path: Path) -> tuple[dict[str, str], list[dict]]:
    from safetensors import safe_open

    dtype_names = {"F32": "float32", "BF16": "bfloat16", "F16": "float16"}
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        schema = []
        for key in sorted(handle.keys()):
            tensor_slice = handle.get_slice(key)
            dtype = tensor_slice.get_dtype()
            schema.append(
                {
                    "key": key,
                    "shape": list(tensor_slice.get_shape()),
                    "dtype": dtype_names.get(dtype, dtype),
                }
            )
    return metadata, schema


_FORBIDDEN_PAYLOAD_KEY_TOKENS = {
    "optimizer",
    "trainer",
    "rng",
    "sampler",
    "scheduler",
    "scaler",
    "ema",
}


def is_stage2_training_state_payload_key(key: Any) -> bool:
    if not isinstance(key, str):
        return True
    normalized = re.sub(r"[.\-]+", "_", key.strip().lower())
    tokens = {token for token in normalized.split("_") if token}
    return bool(tokens & _FORBIDDEN_PAYLOAD_KEY_TOKENS) or "global_step" in normalized


def _longlive_payload_contract(path: Path, selector: str) -> dict[str, Any]:
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "mmap" not in message or not any(
            token in message for token in ("zip", "serialization", "torch.save")
        ):
            raise
        payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("longlive teacher payload must be an object")
    forbidden = [key for key in payload if is_stage2_training_state_payload_key(key)]
    if forbidden:
        raise ValueError(
            f"longlive teacher payload contains forbidden keys: {forbidden}"
        )
    state = payload.get(selector)
    if not isinstance(state, Mapping) or not state:
        raise KeyError(f"longlive teacher payload lacks selector {selector!r}")
    if any(not isinstance(key, str) for key in state):
        raise TypeError("longlive teacher state_dict keys must be strings")
    if any(not isinstance(tensor, torch.Tensor) for tensor in state.values()):
        raise TypeError("longlive teacher state_dict values must be tensors")
    wrong_dtype = [
        key
        for key, tensor in state.items()
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16
    ]
    if wrong_dtype:
        raise TypeError(
            "longlive teacher floating tensors must be BF16 before manifest "
            f"publication: {wrong_dtype[:8]}"
        )
    lora_keys = [key for key in state if "lora_" in key]
    if lora_keys:
        raise ValueError(
            "longlive teacher must be an immutable full checkpoint without LoRA: "
            f"{lora_keys[:8]}"
        )
    state_keys = sorted(state)
    if all(key.startswith("model.") for key in state_keys):
        load_target = "role_wrapper"
    elif all(not key.startswith("model.") for key in state_keys):
        load_target = "bare_transformer"
    else:
        raise ValueError(
            "longlive teacher state_dict mixes wrapper-prefixed and bare keys"
        )
    metadata = {}
    for key, item in payload.items():
        if key == selector:
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            metadata[key] = item
        else:
            raise TypeError(
                "longlive teacher payload permits only one selected state_dict "
                f"and scalar metadata; unexpected {key!r}={type(item).__name__}"
            )
    return {
        "top_level_keys": sorted(payload),
        "state_dict_selector": selector,
        "state_tensor_count": len(state),
        "state_dict_keys_sha256": canonical_json_sha256(state_keys),
        "load_target": load_target,
        "metadata": metadata,
    }


def _sha256(value: Any, path: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{path} must be a lowercase SHA256 hex string")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{path} must be a non-empty canonical string")
    return value


def _manifest_with_self_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    if "manifest_sha256" in payload:
        raise ValueError("manifest payload must not pre-populate manifest_sha256")
    payload["manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def _read_self_hashed_manifest(path: str | os.PathLike[str]) -> tuple[Path, dict]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    with resolved.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"manifest must be a JSON object: {resolved}")
    expected = value.get("manifest_sha256")
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    actual = canonical_json_sha256(body)
    if expected != actual:
        raise RuntimeError(
            f"manifest self hash mismatch: expected={expected}, actual={actual}, "
            f"path={resolved}"
        )
    return resolved, value


def build_stage2_teacher_manifest(
    *,
    checkpoint_path: str | os.PathLike[str],
    architecture_root: str | os.PathLike[str],
    checkpoint_format: str,
    state_dict_selector: str,
    source_kind: str,
    source_identifier: str,
    source_sha256: str,
    conversion_command: Sequence[str],
    attest_cat_domain_bidirectional_ti2v: bool,
    attest_video_global_flow: bool,
) -> dict[str, Any]:
    """Create an operator-attested required contract for one teacher asset.

    This builder hashes provenance and records requirements.  It does not claim
    that a strict model load occurred; that evidence is emitted only by the
    world8 role-init preflight.
    """

    checkpoint_format = _nonempty_string(checkpoint_format, "checkpoint_format")
    if checkpoint_format not in _TEACHER_CHECKPOINT_FORMATS:
        raise ValueError(
            f"unsupported Stage-2 teacher checkpoint_format={checkpoint_format!r}"
        )
    state_dict_selector = _nonempty_string(state_dict_selector, "state_dict_selector")
    if state_dict_selector not in _TEACHER_STATE_DICT_SELECTORS:
        raise ValueError(
            f"unsupported Stage-2 teacher state_dict_selector={state_dict_selector!r}"
        )
    if checkpoint_format == "wan_native_transformer" and state_dict_selector != "root":
        raise ValueError("wan_native_transformer requires state_dict_selector='root'")
    if checkpoint_format == "longlive_wrapper_pt" and state_dict_selector == "root":
        raise ValueError(
            "longlive_wrapper_pt requires generator, real_score, or model selector"
        )
    if attest_cat_domain_bidirectional_ti2v is not True:
        raise ValueError(
            "operator must attest the cat-domain bidirectional TI2V teacher"
        )
    if attest_video_global_flow is not True:
        raise ValueError("operator must attest the video-global flow contract")
    source_kind = _nonempty_string(source_kind, "source_kind")
    source_identifier = _nonempty_string(source_identifier, "source_identifier")
    source_sha256 = _sha256(source_sha256, "source_sha256")
    if isinstance(conversion_command, (str, bytes)) or not conversion_command:
        raise ValueError("conversion_command must be a non-empty string sequence")
    command = tuple(
        _nonempty_string(value, f"conversion_command[{index}]")
        for index, value in enumerate(conversion_command)
    )
    checkpoint, source_files, runtime_files = _teacher_checkpoint_sources(
        checkpoint_path, checkpoint_format
    )
    source_files_sha256 = canonical_json_sha256(source_files)
    architecture = _stage2_architecture_contract(architecture_root)
    payload_contract = (
        _longlive_payload_contract(checkpoint, state_dict_selector)
        if checkpoint_format == "longlive_wrapper_pt"
        else None
    )
    native_tensor_contract = (
        _native_safetensors_contract(checkpoint, runtime_files)
        if checkpoint_format == "wan_native_transformer"
        else None
    )
    payload = {
        "schema": STAGE2_TEACHER_MANIFEST_SCHEMA,
        "schema_version": STAGE2_TEACHER_MANIFEST_VERSION,
        "role": "real_score",
        "model_name": "Wan2.2-TI2V-5B",
        "checkpoint": {
            "path": str(checkpoint),
            "format": checkpoint_format,
            "state_dict_selector": state_dict_selector,
            "source_files": source_files,
            "source_files_sha256": source_files_sha256,
            "payload_contract": payload_contract,
            "native_tensor_contract": native_tensor_contract,
            "dtype": "bfloat16",
            "strict_reload_required": True,
        },
        "architecture": architecture,
        "required_model_contract": dict(_REQUIRED_TEACHER_SEMANTIC_CONTRACT),
        "operator_attestation": {
            "cat_domain_bidirectional_ti2v": True,
            "video_global_flow": True,
        },
        "provenance": {
            "source_kind": source_kind,
            "source_identifier": source_identifier,
            "source_sha256": source_sha256,
            "conversion_command": list(command),
            "conversion_command_sha256": canonical_json_sha256(list(command)),
        },
    }
    return _manifest_with_self_hash(payload)


def validate_stage2_teacher_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    expected_checkpoint_path: str | os.PathLike[str],
    expected_architecture_root: str | os.PathLike[str],
) -> dict[str, Any]:
    manifest_file, value = _read_self_hashed_manifest(manifest_path)
    expected_top_level = {
        "schema",
        "schema_version",
        "role",
        "model_name",
        "checkpoint",
        "architecture",
        "required_model_contract",
        "operator_attestation",
        "provenance",
        "manifest_sha256",
    }
    if set(value) != expected_top_level:
        raise ValueError("teacher manifest top-level schema mismatch")
    if value.get("schema") != STAGE2_TEACHER_MANIFEST_SCHEMA:
        raise ValueError("teacher manifest schema mismatch")
    if value.get("schema_version") != STAGE2_TEACHER_MANIFEST_VERSION:
        raise ValueError("teacher manifest schema_version mismatch")
    if value.get("role") != "real_score":
        raise ValueError("teacher manifest role must be real_score")
    if value.get("model_name") != "Wan2.2-TI2V-5B":
        raise ValueError("teacher manifest model_name mismatch")
    contract = value.get("required_model_contract")
    if contract != _REQUIRED_TEACHER_SEMANTIC_CONTRACT:
        differing = {
            key: {
                "expected": expected,
                "actual": contract.get(key) if isinstance(contract, dict) else None,
            }
            for key, expected in _REQUIRED_TEACHER_SEMANTIC_CONTRACT.items()
            if not isinstance(contract, dict) or contract.get(key) != expected
        }
        raise ValueError(f"teacher required_model_contract mismatch: {differing}")
    if value.get("operator_attestation") != {
        "cat_domain_bidirectional_ti2v": True,
        "video_global_flow": True,
    }:
        raise ValueError("teacher operator_attestation is incomplete")
    architecture = _stage2_architecture_contract(expected_architecture_root)
    if value.get("architecture") != architecture:
        raise ValueError("teacher architecture/config hash differs from runtime")

    checkpoint = value.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise TypeError("teacher checkpoint manifest entry must be an object")
    checkpoint_format = checkpoint.get("format")
    if checkpoint_format not in _TEACHER_CHECKPOINT_FORMATS:
        raise ValueError("teacher checkpoint format is unsupported")
    selector = checkpoint.get("state_dict_selector")
    if selector not in _TEACHER_STATE_DICT_SELECTORS:
        raise ValueError("teacher checkpoint state_dict_selector is unsupported")
    if checkpoint_format == "wan_native_transformer" and selector != "root":
        raise ValueError("wan_native_transformer requires root selector")
    if checkpoint_format == "longlive_wrapper_pt" and selector == "root":
        raise ValueError(
            "longlive_wrapper_pt requires generator, real_score, or model selector"
        )
    if checkpoint.get("dtype") != "bfloat16":
        raise ValueError("teacher checkpoint dtype must be bfloat16")
    if checkpoint.get("strict_reload_required") is not True:
        raise ValueError("teacher checkpoint strict_reload_required must be true")
    actual_path, actual_files, runtime_files = _teacher_checkpoint_sources(
        expected_checkpoint_path, checkpoint_format
    )
    recorded_path = Path(checkpoint.get("path", "")).expanduser().resolve()
    if recorded_path != actual_path:
        raise ValueError(
            "teacher manifest checkpoint.path differs from the configured runtime path"
        )
    expected_files = checkpoint.get("source_files")
    if expected_files != actual_files:
        raise RuntimeError("teacher checkpoint source file hashes differ from manifest")
    source_files_sha256 = canonical_json_sha256(actual_files)
    if checkpoint.get("source_files_sha256") != source_files_sha256:
        raise RuntimeError("teacher checkpoint aggregate source hash mismatch")
    actual_sha = (
        actual_files[0]["sha256"]
        if checkpoint_format == "longlive_wrapper_pt"
        else source_files_sha256
    )
    expected_payload_contract = checkpoint.get("payload_contract")
    if checkpoint_format == "longlive_wrapper_pt":
        actual_payload_contract = _longlive_payload_contract(actual_path, selector)
        if expected_payload_contract != actual_payload_contract:
            raise RuntimeError(
                "teacher longlive payload contract differs from manifest"
            )
    elif expected_payload_contract is not None:
        raise ValueError("wan_native_transformer payload_contract must be null")
    expected_native_contract = checkpoint.get("native_tensor_contract")
    if checkpoint_format == "wan_native_transformer":
        actual_native_contract = _native_safetensors_contract(
            actual_path, runtime_files
        )
        if expected_native_contract != actual_native_contract:
            raise RuntimeError("teacher native tensor dtype/schema contract drifted")
    elif expected_native_contract is not None:
        raise ValueError("longlive_wrapper_pt native_tensor_contract must be null")

    provenance = value.get("provenance")
    if not isinstance(provenance, dict):
        raise TypeError("teacher provenance must be an object")
    _nonempty_string(provenance.get("source_kind"), "provenance.source_kind")
    _nonempty_string(
        provenance.get("source_identifier"), "provenance.source_identifier"
    )
    _sha256(provenance.get("source_sha256"), "provenance.source_sha256")
    command = provenance.get("conversion_command")
    if not isinstance(command, list) or not command:
        raise ValueError("teacher conversion_command must be a non-empty list")
    for index, item in enumerate(command):
        _nonempty_string(item, f"provenance.conversion_command[{index}]")
    expected_command_hash = canonical_json_sha256(command)
    if provenance.get("conversion_command_sha256") != expected_command_hash:
        raise RuntimeError("teacher conversion_command hash mismatch")
    architecture_path = (
        Path(expected_architecture_root).expanduser().resolve() / "config.json"
    )
    return {
        "manifest_path": str(manifest_file),
        "manifest_file_sha256": sha256_file(manifest_file),
        "manifest_sha256": value["manifest_sha256"],
        "checkpoint_path": str(actual_path),
        "checkpoint_sha256": actual_sha,
        "checkpoint_files": runtime_files,
        "checkpoint_format": checkpoint_format,
        "state_dict_selector": selector,
        "payload_contract": expected_payload_contract,
        "native_tensor_contract": expected_native_contract,
        "architecture": architecture,
        "architecture_file": {
            "name": "config.json",
            "path": str(architecture_path),
            "size": architecture_path.stat().st_size,
            "sha256": architecture["config_sha256"],
            "identity": _file_identity(architecture_path),
        },
        "model_contract": dict(contract),
        "operator_attestation": dict(value["operator_attestation"]),
        "provenance": dict(provenance),
    }


def validate_stage2_generator_manifest(
    manifest_path: str | os.PathLike[str],
    *,
    expected_checkpoint_path: str | os.PathLike[str],
    expected_step: int = 3750,
) -> dict[str, Any]:
    """Accept only the enriched Stage-1 EMA-merge v2 provenance contract."""

    manifest_file, value = _read_self_hashed_manifest(manifest_path)
    expected_top_level = {
        "schema",
        "schema_version",
        "checkpoint_format",
        "checkpoint_version",
        "model_name",
        "base",
        "training_checkpoint",
        "adapter",
        "adapter_a_b_tensors",
        "target_modules",
        "trainable_parameters",
        "output",
        "manifest_sha256",
    }
    if set(value) != expected_top_level:
        raise ValueError("generator merge manifest top-level schema mismatch")
    if value.get("schema") != STAGE1_MERGE_MANIFEST_SCHEMA:
        raise ValueError("generator merge manifest schema mismatch")
    if value.get("schema_version") != STAGE1_MERGE_MANIFEST_VERSION:
        raise ValueError(
            "Stage-2 requires longlive_stage1_merge_manifest schema_version=2; "
            "regenerate the Stage-1 step3750 merge manifest"
        )
    if value.get("checkpoint_format") != "longlive_stage1_causal_ema_merged":
        raise ValueError("generator checkpoint_format mismatch")
    if value.get("checkpoint_version") != 1:
        raise ValueError("generator checkpoint_version mismatch")
    if value.get("model_name") != "Wan2.2-TI2V-5B":
        raise ValueError("generator model_name mismatch")
    training = value.get("training_checkpoint")
    if not isinstance(training, dict):
        raise TypeError("generator training_checkpoint entry must be an object")
    if training.get("completed_step") != int(expected_step):
        raise ValueError(
            f"generator source must be Stage-1 step {expected_step}, got "
            f"{training.get('completed_step')}"
        )
    training_path = Path(training.get("path", "")).expanduser().resolve()
    if not training_path.is_dir():
        raise FileNotFoundError(training_path)
    expected_adapter_names = {
        "raw_adapter": "adapter_raw.safetensors",
        "ema_adapter": "adapter_ema.safetensors",
    }
    for kind in ("raw_adapter", "ema_adapter"):
        item = training.get(kind)
        if not isinstance(item, dict):
            raise ValueError(f"generator manifest is missing {kind}")
        if item.get("name") != expected_adapter_names[kind]:
            raise ValueError(f"generator {kind} name mismatch")
        _sha256(item.get("sha256"), f"training_checkpoint.{kind}.sha256")
        if not isinstance(item.get("size"), int) or item["size"] <= 0:
            raise ValueError(f"training_checkpoint.{kind}.size must be positive")
        metadata = item.get("metadata")
        expected_kind = "raw" if kind == "raw_adapter" else "ema"
        if not isinstance(metadata, dict) or metadata.get("kind") != expected_kind:
            raise ValueError(f"generator {kind} metadata kind mismatch")
        if metadata.get("completed_step") != str(expected_step):
            raise ValueError(f"generator {kind} metadata completed_step mismatch")
        if metadata.get("schema") != "longlive_stage1_lora_checkpoint":
            raise ValueError(f"generator {kind} metadata schema mismatch")
        if metadata.get("schema_version") != "1":
            raise ValueError(f"generator {kind} metadata schema_version mismatch")
        if metadata.get("tensor_count") != "360":
            raise ValueError(f"generator {kind} metadata tensor_count mismatch")
        if metadata.get("global_numel") != "57016320":
            raise ValueError(f"generator {kind} metadata global_numel mismatch")
        if metadata.get("dtype") != "float32":
            raise ValueError(f"generator {kind} metadata dtype mismatch")
        adapter_path = training_path / item["name"]
        if not adapter_path.is_file():
            raise FileNotFoundError(adapter_path)
        if adapter_path.stat().st_size != item["size"]:
            raise RuntimeError(f"generator {kind} size differs from its manifest")
        if sha256_file(adapter_path) != item["sha256"]:
            raise RuntimeError(f"generator {kind} SHA256 differs from its manifest")
    if training.get("adapter") != "adapter_ema.safetensors":
        raise ValueError("generator selected adapter must be adapter_ema.safetensors")
    if training.get("adapter_sha256") != training["ema_adapter"]["sha256"]:
        raise ValueError("generator selected adapter SHA differs from EMA SHA")
    resolved_config_path = training_path / "resolved_config.yaml"
    if sha256_file(resolved_config_path) != training.get("resolved_config_sha256"):
        raise RuntimeError("generator resolved_config SHA256 differs from its manifest")

    adapter = value.get("adapter")
    if not isinstance(adapter, dict):
        raise ValueError("generator manifest is missing adapter schema")
    expected_counts = {
        "rank": 32,
        "alpha": 32,
        "target_module_count": 180,
        "adapter_tensor_count": 360,
        "trainable_parameter_count": 57_016_320,
    }
    for key, expected in expected_counts.items():
        if adapter.get(key) != expected:
            raise ValueError(
                f"generator adapter.{key} mismatch: "
                f"expected={expected}, actual={adapter.get(key)}"
            )
    expected_patterns = [
        r"^blocks\.[0-9]+\.self_attn\.(q|k|v|o)$",
        r"^blocks\.[0-9]+\.ffn\.(0|2)$",
    ]
    locked_adapter_values = {
        "type": "lora",
        "dropout": 0.0,
        "bias": "none",
        "modules_to_save": [],
        "target_patterns": expected_patterns,
    }
    for key, expected in locked_adapter_values.items():
        if adapter.get(key) != expected:
            raise ValueError(
                f"generator adapter.{key} mismatch: "
                f"expected={expected}, actual={adapter.get(key)}"
            )
    targets = adapter.get("target_module_names")
    tensor_schema = adapter.get("tensor_schema")
    if not isinstance(targets, list) or len(targets) != 180:
        raise ValueError("generator target_module_names must contain 180 names")
    if targets != sorted(targets) or len(set(targets)) != 180:
        raise ValueError("generator target_module_names must be unique and sorted")
    from utils.stage2_roles import expected_stage2_target_names

    if targets != list(expected_stage2_target_names()):
        raise ValueError("generator target_module_names differ from Stage-2 allowlist")
    if adapter.get("target_schema_sha256") != canonical_json_sha256(targets):
        raise RuntimeError("generator target schema hash mismatch")
    if not isinstance(tensor_schema, list) or len(tensor_schema) != 360:
        raise ValueError("generator tensor_schema must contain 360 entries")
    if adapter.get("tensor_schema_sha256") != canonical_json_sha256(tensor_schema):
        raise RuntimeError("generator tensor schema hash mismatch")
    tensor_targets: dict[str, set[str]] = {}
    for entry in tensor_schema:
        if not isinstance(entry, dict) or set(entry) != {"key", "shape", "dtype"}:
            raise ValueError("generator tensor_schema entries must be exact objects")
        key = entry["key"]
        if not isinstance(key, str) or entry["dtype"] != "float32":
            raise ValueError("generator tensor_schema key/dtype mismatch")
        block_index = key.find("blocks.")
        if not key.startswith("base_model.model.blocks.") or block_index != len(
            "base_model.model."
        ):
            raise ValueError("generator tensor_schema key prefix is non-canonical")
        suffix = None
        for marker in (".lora_A.weight", ".lora_B.weight"):
            if key.endswith(marker):
                suffix = marker
                break
        if suffix is None:
            raise ValueError("generator tensor_schema contains a non-A/B key")
        target = key[block_index : -len(suffix)]
        if target not in targets:
            raise ValueError("generator tensor_schema target is outside the allowlist")
        if target.endswith(
            ("self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o")
        ):
            input_features = output_features = 3072
        elif target.endswith("ffn.0"):
            input_features, output_features = 3072, 14336
        elif target.endswith("ffn.2"):
            input_features, output_features = 14336, 3072
        else:  # guarded by the exact target allowlist
            raise AssertionError(target)
        expected_shape = (
            [32, input_features]
            if suffix == ".lora_A.weight"
            else [output_features, 32]
        )
        if entry["shape"] != expected_shape:
            raise ValueError(
                f"generator tensor_schema shape mismatch for {key}: "
                f"expected={expected_shape}, actual={entry['shape']}"
            )
        tensor_targets.setdefault(target, set()).add(suffix)
    if set(tensor_targets) != set(targets) or any(
        suffixes != {".lora_A.weight", ".lora_B.weight"}
        for suffixes in tensor_targets.values()
    ):
        raise ValueError("generator tensor_schema must contain one A/B pair per target")

    base = value.get("base")
    if not isinstance(base, dict):
        raise ValueError("generator manifest is missing its Stage-1 base")
    base_path = Path(base.get("path", "")).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    if base_path.stat().st_size != base.get("size"):
        raise RuntimeError("generator Stage-1 base size differs from its manifest")
    if sha256_file(base_path) != _sha256(base.get("sha256"), "base.sha256"):
        raise RuntimeError("generator Stage-1 base SHA256 differs from its manifest")

    # Revalidate the original immutable Stage-1 checkpoint instead of trusting
    # metadata copied into a re-hashed merge sidecar.  This proves _SUCCESS,
    # the checkpoint manifest self-hash, and every listed source file hash.
    from utils.stage1_checkpoint import validate_checkpoint

    checkpoint_manifest = validate_checkpoint(
        training_path,
        require_resumable=False,
        expected_base_sha256=base["sha256"],
        expected_topology=(6, 3, 2),
    )
    if checkpoint_manifest.get("manifest_sha256") != training.get("manifest_sha256"):
        raise RuntimeError("generator Stage-1 checkpoint manifest hash mismatch")
    redundant_counts = {
        "adapter_a_b_tensors": 360,
        "target_modules": 180,
        "trainable_parameters": 57_016_320,
    }
    for key, expected in redundant_counts.items():
        if value.get(key) != expected:
            raise ValueError(f"generator redundant {key} mismatch")
    for kind in ("raw_adapter", "ema_adapter"):
        item = training[kind]
        actual_metadata, actual_schema = _read_safetensors_contract(
            training_path / item["name"]
        )
        if actual_metadata != item["metadata"]:
            raise RuntimeError(
                f"generator {kind} actual metadata differs from manifest"
            )
        if actual_schema != tensor_schema:
            raise RuntimeError(
                f"generator {kind} actual tensor schema differs from manifest"
            )

    output = value.get("output")
    if not isinstance(output, dict):
        raise TypeError("generator manifest output must be an object")
    if output.get("dtype") != "bfloat16" or output.get("strict_reload") is not True:
        raise ValueError("generator output must be BF16 and strict_reload=true")
    expected_sha = _sha256(output.get("sha256"), "output.sha256")
    checkpoint = Path(expected_checkpoint_path).expanduser().resolve()
    recorded_output = Path(output.get("path", "")).expanduser().resolve()
    if recorded_output != checkpoint:
        raise ValueError(
            "generator manifest output.path differs from the configured runtime path"
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.stat().st_size != output.get("size"):
        raise RuntimeError("generator checkpoint size differs from its manifest")
    actual_sha = sha256_file(checkpoint)
    if actual_sha != expected_sha:
        raise RuntimeError("generator checkpoint SHA256 differs from its manifest")
    return {
        "manifest_path": str(manifest_file),
        "manifest_file_sha256": sha256_file(manifest_file),
        "manifest_sha256": value["manifest_sha256"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": actual_sha,
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_files": [
            {
                "name": checkpoint.name,
                "size": checkpoint.stat().st_size,
                "sha256": actual_sha,
                "path": str(checkpoint),
                "identity": _file_identity(checkpoint),
            }
        ],
        "checkpoint_format": value["checkpoint_format"],
        "state_dict_selector": "generator",
        "source_step": training["completed_step"],
        "adapter": dict(adapter),
        "source": {
            "base": dict(value.get("base", {})),
            "training_checkpoint": dict(training),
        },
    }


def require_stage2_cold_start(resolved_config: Any) -> None:
    mode = getattr(resolved_config, "initialization_mode", None)
    if mode == "resume_stage2":
        raise NotImplementedError(
            "Stage-2 resume is implemented only in Step 10; Batch 2 init-only "
            "must not fall back to a cold start."
        )
    if mode != "init_from_stage1":
        raise ValueError(f"unsupported Stage-2 initialization_mode={mode!r}")


def audit_stage2_init_assets(resolved_config: Any) -> dict[str, Any]:
    require_stage2_cold_start(resolved_config)
    generator = validate_stage2_generator_manifest(
        resolved_config.init_generator_manifest,
        expected_checkpoint_path=resolved_config.init_generator_checkpoint,
        expected_step=resolved_config.generator_stage1_step,
    )
    real_score = validate_stage2_teacher_manifest(
        resolved_config.init_real_score_manifest,
        expected_checkpoint_path=resolved_config.init_real_score_checkpoint,
        expected_architecture_root=resolved_config.architecture_root,
    )
    architecture_file = real_score["architecture_file"]
    return {
        "generator": {**generator, "architecture_file": architecture_file},
        "real_score": real_score,
        "fake_score": {
            **real_score,
            "immutable_source_role": "real_score",
        },
    }


def build_stage2_role_init_manifest(
    *,
    contract_hash: str,
    launch_hash: str,
    git_commit: str,
    generator_asset: Mapping[str, Any],
    real_score_asset: Mapping[str, Any],
    role_audits: Mapping[str, Any],
    role_isolation_audit: Mapping[str, Any],
    fsdp_audits: Mapping[str, Any],
    rank_consensus_sha256: str,
    side_effect_audit: Mapping[str, Any],
) -> dict[str, Any]:
    for value, path in (
        (contract_hash, "contract_hash"),
        (launch_hash, "launch_hash"),
        (rank_consensus_sha256, "rank_consensus_sha256"),
    ):
        _sha256(value, path)
    if (
        not isinstance(git_commit, str)
        or re.fullmatch(r"[0-9a-f]{40}", git_commit) is None
    ):
        raise ValueError("git_commit must be a 40-character lowercase Git SHA")
    if set(role_audits) != {"generator", "real_score", "fake_score"}:
        raise ValueError(
            "role_audits must contain exactly generator/real_score/fake_score"
        )
    expected_role_isolation = {
        "parameter_objects_disjoint": True,
        "parameter_storage_disjoint": True,
        "roles": ("generator", "real_score", "fake_score"),
    }
    actual_role_isolation = dict(role_isolation_audit)
    actual_role_isolation["roles"] = tuple(actual_role_isolation.get("roles", ()))
    if actual_role_isolation != expected_role_isolation:
        raise ValueError(
            "role isolation audit does not prove three independent role objects/storage"
        )
    expected_role_values = {
        "generator": (32, 180, 360, 57_016_320, "causal", False),
        "real_score": (None, 0, 0, 0, "bidirectional_ti2v", False),
        "fake_score": (64, 180, 360, 114_032_640, "bidirectional_ti2v", True),
    }
    from utils.stage2_roles import expected_stage2_target_names

    exact_target_names = expected_stage2_target_names()
    for role, (
        rank,
        targets,
        tensors,
        parameters,
        backbone,
        activation_checkpointing,
    ) in expected_role_values.items():
        audit = role_audits[role]
        if not isinstance(audit, Mapping):
            raise TypeError(f"role audit must be an object: {role}")
        _sha256(audit.get("base_checkpoint_sha256"), f"roles.{role}.base_sha256")
        if audit.get("strict_reload_succeeded") is not True:
            raise ValueError(f"roles.{role} strict reload was not proven")
        if audit.get("backbone") != backbone or audit.get("base_dtype") != "bfloat16":
            raise ValueError(f"roles.{role} backbone/base dtype contract mismatch")
        if audit.get("activation_checkpointing") is not activation_checkpointing:
            raise ValueError(f"roles.{role} activation-checkpoint contract mismatch")
        target = audit.get("target_audit")
        if role == "real_score":
            if target is not None or audit.get("trainable_policy") != "frozen":
                raise ValueError("real_score role audit must be frozen without adapter")
        else:
            if not isinstance(target, Mapping):
                raise ValueError(f"roles.{role}.target_audit is missing")
            actual = (
                target.get("rank"),
                target.get("target_module_count"),
                target.get("adapter_tensor_count"),
                target.get("trainable_parameter_count"),
            )
            if actual != (rank, targets, tensors, parameters):
                raise ValueError(f"roles.{role} adapter contract mismatch: {actual}")
            if tuple(target.get("target_module_names", ())) != exact_target_names:
                raise ValueError(f"roles.{role} exact target names drifted")
            _sha256(audit.get("adapter_digest"), f"roles.{role}.adapter_digest")
            if len(tuple(audit.get("canonical_adapter_keys", ()))) != tensors:
                raise ValueError(f"roles.{role} canonical adapter key count mismatch")
        if audit.get("pre_fsdp_adapter_tensor_count") != tensors:
            raise ValueError(f"roles.{role} pre-FSDP tensor count mismatch")
        if audit.get("pre_fsdp_trainable_parameters") != parameters:
            raise ValueError(f"roles.{role} pre-FSDP parameter count mismatch")
        post = audit.get("post_fsdp")
        if not isinstance(post, Mapping):
            raise ValueError(f"roles.{role}.post_fsdp is missing")
        required_post = {
            "trainable_tensor_count": tensors,
            "global_trainable_parameters": parameters,
            "all_parameters_are_dtensor": True,
            "fsdp_module_count": 31,
            "root_and_30_blocks_independently_wrapped": True,
            "mesh_shape": (8,),
            "mesh_dim_names": ("shard",),
            "placements": ("shard:0",),
        }
        wrong_post = {}
        for key, expected in required_post.items():
            actual = post.get(key)
            comparable = tuple(actual or ()) if isinstance(expected, tuple) else actual
            if comparable != expected:
                wrong_post[key] = {"expected": expected, "actual": actual}
        if wrong_post or int(post.get("frozen_tensor_count", 0)) <= 0:
            raise ValueError(f"roles.{role} post-FSDP contract mismatch: {wrong_post}")
    if role_audits["real_score"].get("base_checkpoint_sha256") != role_audits[
        "fake_score"
    ].get("base_checkpoint_sha256"):
        raise ValueError("real_score/fake_score base SHA mismatch")
    if role_audits["generator"].get("base_checkpoint_sha256") != generator_asset.get(
        "checkpoint_sha256"
    ):
        raise ValueError("generator asset/role checkpoint SHA mismatch")
    for role in ("real_score", "fake_score"):
        if role_audits[role].get("base_checkpoint_sha256") != real_score_asset.get(
            "checkpoint_sha256"
        ):
            raise ValueError(f"{role} asset/role checkpoint SHA mismatch")
    expected_side_effects = {
        "tripwires_enforced": [
            "module_call_impl",
            "direct_module_forward",
            "optimizer",
            "ema",
            "text_encoder",
            "vae",
            "dataloader",
        ],
        "forward_calls": 0,
        "optimizer_created": False,
        "ema_created": False,
        "text_encoder_created": False,
        "vae_created": False,
        "dataloader_created": False,
    }
    if dict(side_effect_audit) != expected_side_effects:
        raise ValueError("init-only side-effect tripwire audit is not clean")
    expected_fsdp = {
        "world_size": 8,
        "sequence_parallel_size": 1,
        "data_parallel_size": 8,
        "mesh_shape": (8,),
        "mesh_dim_names": ("shard",),
        "sharding_strategy": "FULL_SHARD",
    }
    for key, expected in expected_fsdp.items():
        actual = fsdp_audits.get(key)
        if isinstance(expected, tuple):
            actual = tuple(actual or ())
        if actual != expected:
            raise ValueError(
                f"role-init FSDP contract mismatch for {key}: "
                f"expected={expected}, actual={actual}"
            )
    payload = {
        "schema": STAGE2_ROLE_INIT_MANIFEST_SCHEMA,
        "schema_version": STAGE2_ROLE_INIT_MANIFEST_VERSION,
        "artifact_kind": "init_only_audit_not_training_checkpoint",
        "initialization_mode": "init_from_stage1",
        "config": {
            "contract_hash": contract_hash,
            "launch_hash": launch_hash,
        },
        "code": {"git_commit": git_commit},
        "assets": {
            "generator": dict(generator_asset),
            "real_score": dict(real_score_asset),
            "fake_score_base_sha256": real_score_asset.get("checkpoint_sha256"),
        },
        "roles": dict(role_audits),
        "role_isolation": dict(role_isolation_audit),
        "fsdp": dict(fsdp_audits),
        "rank_consensus_sha256": rank_consensus_sha256,
        "side_effects": dict(side_effect_audit),
    }
    payload["artifact_contract_sha256"] = canonical_json_sha256(
        {
            "config_contract_hash": contract_hash,
            "generator_checkpoint_sha256": generator_asset.get("checkpoint_sha256"),
            "real_score_checkpoint_sha256": real_score_asset.get("checkpoint_sha256"),
            "roles": role_audits,
            "role_isolation": role_isolation_audit,
            "fsdp": fsdp_audits,
        }
    )
    return _manifest_with_self_hash(payload)


def write_stage2_role_init_artifacts(
    output_directory: str | os.PathLike[str],
    manifest: Mapping[str, Any],
) -> Path:
    output = Path(output_directory).expanduser().resolve()
    value = dict(manifest)
    if value.get("schema") != STAGE2_ROLE_INIT_MANIFEST_SCHEMA:
        raise ValueError("role-init manifest schema mismatch")
    if value.get("schema_version") != STAGE2_ROLE_INIT_MANIFEST_VERSION:
        raise ValueError("role-init manifest schema_version mismatch")
    expected_hash = value.get("manifest_sha256")
    body = {key: item for key, item in value.items() if key != "manifest_sha256"}
    if expected_hash != canonical_json_sha256(body):
        raise RuntimeError("role-init manifest self hash mismatch before publication")
    if output.exists():
        raise FileExistsError(output)
    parent = output.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=parent))
    try:
        atomic_write_json(temporary / "role_init_manifest.json", dict(manifest))
        atomic_write_bytes(temporary / ROLE_INIT_COMPLETE_MARKER, b"")
        os.replace(temporary, output)
        directory_fd = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output
