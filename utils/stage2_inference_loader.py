"""Strict Generator-EMA-only loader for Stage-2 deployment inference.

This module deliberately constructs one causal Generator and never imports or
constructs the real-score/fake-score roles, optimizers, samplers, rank-local
RNG state, or the legacy inference pipeline.
"""

from __future__ import annotations

import gc
import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from utils.stage1_io import canonical_json_sha256, sha256_file

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RESOLVED_CONFIG_KEYS = {"config", "derived"}
_ARCHITECTURE_FILE_KEYS = {"name", "path", "size", "sha256", "identity"}
_FILE_IDENTITY_KEYS = {"device", "inode", "size", "mtime_ns", "ctime_ns"}


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in Stage-2 checkpoint: {key!r}")
        result[key] = value
    return result


def _default_resolve_training_config(value: Any) -> Any:
    from utils.stage2_config import resolve_stage2_config

    return resolve_stage2_config(value)


def _default_checkpoint_reader(*args: Any, **kwargs: Any) -> Any:
    from utils.stage2_checkpoint import load_stage2_generator_ema_checkpoint

    return load_stage2_generator_ema_checkpoint(*args, **kwargs)


def _default_generator_manifest_validator(*args: Any, **kwargs: Any) -> Any:
    from utils.stage2_role_manifest import validate_stage2_generator_manifest

    return validate_stage2_generator_manifest(*args, **kwargs)


def _default_generator_builder(resolved: Any, architecture_root: Path) -> Any:
    from accelerate import init_empty_weights

    from model.stage2_dmd import Stage2DiTRole
    from utils.wan_5b_wrapper import build_wan_model

    with init_empty_weights():
        transformer = build_wan_model(
            model_name=resolved.model_name,
            is_causal=True,
            architecture_root=architecture_root,
            init_weights=False,
            local_attn_size=resolved.physical_kv_capacity_frames,
            sink_size=resolved.global_sink_frames,
            num_frame_per_block=resolved.chunk_frames,
        )
    return Stage2DiTRole(transformer, role="generator", is_causal=True)


def _default_base_loader(*args: Any, **kwargs: Any) -> Any:
    from utils.stage2_role_init import strict_load_stage2_role_base

    return strict_load_stage2_role_base(*args, **kwargs)


def _default_role_contract_validator(wrapper: Any, resolved: Any) -> None:
    # Role initialization owns this single canonical Wan architecture audit.
    # Reusing it here prevents training and deployment from drifting apart.
    from utils.stage2_role_init import _audit_loaded_wan_contract

    _audit_loaded_wan_contract(wrapper, resolved)


def _default_lora_configurer(*args: Any, **kwargs: Any) -> Any:
    from utils.stage2_roles import configure_stage2_role_lora

    return configure_stage2_role_lora(*args, **kwargs)


def _default_lora_loader(*args: Any, **kwargs: Any) -> Any:
    from utils.lora_utils import strict_load_lora_state_dict

    return strict_load_lora_state_dict(*args, **kwargs)


@dataclass(frozen=True)
class Stage2InferenceLoaderOps:
    """Dependency boundary used by tiny CPU tests; defaults are production ops."""

    resolve_training_config: Callable[..., Any] = _default_resolve_training_config
    read_generator_ema_checkpoint: Callable[..., Any] = _default_checkpoint_reader
    validate_generator_manifest: Callable[..., Any] = (
        _default_generator_manifest_validator
    )
    build_generator: Callable[..., Any] = _default_generator_builder
    load_generator_base: Callable[..., Any] = _default_base_loader
    validate_generator_contract: Callable[..., Any] = _default_role_contract_validator
    configure_generator_lora: Callable[..., Any] = _default_lora_configurer
    load_generator_lora: Callable[..., Any] = _default_lora_loader


@dataclass(frozen=True)
class LoadedStage2InferenceGenerator:
    """One frozen, BF16, EMA-merged causal Generator plus its exact lineage."""

    generator: torch.nn.Module
    resolved_training_config: Any
    checkpoint_directory: Path
    checkpoint_manifest: Mapping[str, Any]
    checkpoint_provenance: Mapping[str, Any]
    generator_asset: Mapping[str, Any]
    base_load_audit: Mapping[str, Any]
    training_contract_hash: str
    training_launch_hash: str
    generator_ema_sha256: str

    def checkpoint_identity(self) -> dict[str, Any]:
        """Return the exact schema consumed by Stage-2 sample traces."""

        return {
            "directory": str(self.checkpoint_directory),
            "manifest_sha256": self.checkpoint_manifest["manifest_sha256"],
            "completed_generator_updates": self.checkpoint_manifest[
                "completed_generator_updates"
            ],
            "contract_hash": self.training_contract_hash,
            "generator_ema_sha256": self.generator_ema_sha256,
        }


def _read_and_resolve_checkpoint_config(
    checkpoint: str | Path,
    *,
    resolver: Callable[..., Any],
) -> tuple[Path, dict[str, Any], Any]:
    candidate = Path(checkpoint).expanduser()
    if candidate.is_symlink() or not candidate.is_dir():
        raise RuntimeError(
            f"Stage-2 checkpoint is not a regular directory: {candidate}"
        )
    directory = candidate.resolve()
    from utils.stage2_checkpoint import checkpoint_generator_updates

    checkpoint_generator_updates(directory)
    marker = directory / "_SUCCESS"
    if marker.is_symlink() or not marker.is_file() or marker.stat().st_size != 0:
        raise RuntimeError(f"Stage-2 checkpoint success marker is invalid: {marker}")
    config_path = directory / "resolved_config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise RuntimeError(
            f"Stage-2 checkpoint resolved config is missing/not regular: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as handle:
        saved = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(saved, dict) or set(saved) != _RESOLVED_CONFIG_KEYS:
        actual = sorted(saved) if isinstance(saved, dict) else type(saved).__name__
        raise ValueError(
            "Stage-2 resolved_config.json schema mismatch: "
            f"expected={sorted(_RESOLVED_CONFIG_KEYS)}, actual={actual}"
        )
    if not isinstance(saved["config"], Mapping) or not isinstance(
        saved["derived"], Mapping
    ):
        raise TypeError("Stage-2 saved config/derived values must both be mappings")
    canonical_json_sha256(saved)
    resolved = resolver(saved["config"])
    if not callable(getattr(resolved, "to_dict", None)):
        raise TypeError("Stage-2 training config resolver returned an invalid object")
    if canonical_json_sha256(resolved.to_dict()) != canonical_json_sha256(saved):
        raise RuntimeError(
            "Stage-2 checkpoint resolved config cannot be reproduced from its "
            "canonical training config"
        )
    return directory, saved, resolved


def _require_locked_generator_contract(resolved: Any) -> None:
    adapter = resolved.generator_adapter
    expected = {
        "generator_stage1_step": 3075,
        "generator_trainable": "adapter_only",
        "generator_conditioning_mode": "conditional_only",
        "generator_forward_mode": "single_conditional",
        "generator_self_kv_cache_branches": 1,
        "generator_cross_kv_cache_branches": 1,
        "rank": 32,
        "alpha": 32,
        "dropout": 0.0,
        "expected_target_modules": 180,
        "expected_trainable_parameters": 57_016_320,
        "expected_adapter_tensors": 360,
    }
    actual = {
        "generator_stage1_step": resolved.generator_stage1_step,
        "generator_trainable": resolved.generator_trainable,
        "generator_conditioning_mode": resolved.generator_conditioning_mode,
        "generator_forward_mode": resolved.generator_forward_mode,
        "generator_self_kv_cache_branches": (resolved.generator_self_kv_cache_branches),
        "generator_cross_kv_cache_branches": (
            resolved.generator_cross_kv_cache_branches
        ),
        "rank": adapter.rank,
        "alpha": adapter.alpha,
        "dropout": adapter.dropout,
        "expected_target_modules": adapter.expected_target_modules,
        "expected_trainable_parameters": adapter.expected_trainable_parameters,
        "expected_adapter_tensors": adapter.expected_adapter_tensors,
    }
    wrong = {
        key: {"expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if actual[key] != value or type(actual[key]) is not type(value)
    }
    if wrong:
        raise RuntimeError(f"Stage-2 inference Generator contract drifted: {wrong}")


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _plain_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{label} must be a non-negative plain integer")
    return value


def _current_file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _verify_runtime_architecture_file(
    recorded: Any,
    *,
    config_path: Path,
) -> dict[str, Any]:
    """Bind the recorded architecture hash to the current relocation path."""

    if not isinstance(recorded, Mapping) or set(recorded) != _ARCHITECTURE_FILE_KEYS:
        raise ValueError("Stage-2 Generator architecture_file schema mismatch")
    if recorded.get("name") != "config.json":
        raise ValueError("Stage-2 Generator architecture_file name must be config.json")
    recorded_path_value = recorded.get("path")
    if not isinstance(recorded_path_value, str) or not recorded_path_value:
        raise ValueError("Stage-2 recorded architecture path is invalid")
    recorded_path = Path(recorded_path_value).expanduser()
    if not recorded_path.is_absolute() or recorded_path != recorded_path.resolve():
        raise ValueError("Stage-2 recorded architecture path must be canonical")
    expected_size = _plain_nonnegative_int(
        recorded.get("size"),
        "Stage-2 recorded architecture size",
    )
    expected_sha256 = _sha256(
        recorded.get("sha256"),
        "Stage-2 recorded architecture SHA256",
    )
    recorded_identity = recorded.get("identity")
    if not isinstance(recorded_identity, Mapping) or set(recorded_identity) != (
        _FILE_IDENTITY_KEYS
    ):
        raise ValueError("Stage-2 recorded architecture identity schema mismatch")
    normalized_recorded_identity = {
        key: _plain_nonnegative_int(
            recorded_identity[key],
            f"Stage-2 recorded architecture identity.{key}",
        )
        for key in sorted(_FILE_IDENTITY_KEYS)
    }
    if normalized_recorded_identity["size"] != expected_size:
        raise RuntimeError("Stage-2 recorded architecture size/identity mismatch")

    current_path = config_path.resolve()
    if (
        current_path != config_path
        or config_path.is_symlink()
        or not config_path.is_file()
    ):
        raise RuntimeError(
            "Stage-2 runtime architecture config must be a regular canonical file"
        )
    current_identity = _current_file_identity(config_path)
    if current_identity["size"] != expected_size:
        raise RuntimeError("Stage-2 runtime architecture size differs from provenance")
    actual_sha256 = sha256_file(config_path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Stage-2 runtime architecture SHA256 differs from provenance"
        )
    return {
        "name": "config.json",
        "path": str(config_path),
        "size": expected_size,
        "sha256": expected_sha256,
        "identity": current_identity,
    }


def _generator_ema_sha256(manifest: Mapping[str, Any]) -> str:
    files = manifest.get("files")
    if not isinstance(files, list):
        raise TypeError("Stage-2 checkpoint manifest files must be a list")
    matches = [
        entry
        for entry in files
        if isinstance(entry, Mapping)
        and entry.get("name") == "generator_ema.safetensors"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "Stage-2 initialized checkpoint must contain one canonical Generator EMA"
        )
    return _sha256(matches[0].get("sha256"), "generator_ema SHA256")


def _require_generator_wrapper(wrapper: Any) -> None:
    if (
        not isinstance(wrapper, torch.nn.Module)
        or getattr(wrapper, "role", None) != "generator"
        or getattr(wrapper, "is_causal", None) is not True
        or not isinstance(getattr(wrapper, "model", None), torch.nn.Module)
    ):
        raise TypeError(
            "Stage-2 inference builder returned an invalid causal Generator"
        )


def _audit_merged_generator(
    wrapper: torch.nn.Module,
    *,
    device: torch.device,
) -> None:
    named_parameters = list(wrapper.named_parameters())
    if not named_parameters:
        raise RuntimeError("Stage-2 merged Generator has no parameters")
    residual_lora = [
        name
        for name, _ in named_parameters
        if "lora_" in name.lower() or "adapter" in name.lower()
    ]
    residual_lora.extend(
        name
        for name in wrapper.state_dict()
        if "lora_" in name.lower() or "adapter" in name.lower()
    )
    if residual_lora:
        raise RuntimeError(
            "Stage-2 EMA merge left adapter tensors in the deployed Generator: "
            f"{sorted(set(residual_lora))[:8]}"
        )
    trainable = [name for name, value in named_parameters if value.requires_grad]
    if trainable:
        raise RuntimeError(
            f"Stage-2 inference Generator is not frozen: {trainable[:8]}"
        )
    if wrapper.training:
        raise RuntimeError("Stage-2 inference Generator must be in eval mode")
    for module in (wrapper, wrapper.model):
        if bool(getattr(module, "is_gradient_checkpointing", False)) or bool(
            getattr(module, "gradient_checkpointing", False)
        ):
            raise RuntimeError(
                "Stage-2 inference Generator forbids activation checkpointing"
            )
    wrong_device: list[str] = []
    wrong_dtype: list[tuple[str, str]] = []
    for kind, values in (
        ("parameter", wrapper.named_parameters()),
        ("buffer", wrapper.named_buffers()),
    ):
        for name, value in values:
            if value.device != device:
                wrong_device.append(f"{kind}:{name}={value.device}")
            if value.is_floating_point() and value.dtype != torch.bfloat16:
                wrong_dtype.append((f"{kind}:{name}", str(value.dtype)))
    if wrong_device:
        raise RuntimeError(
            f"Stage-2 inference Generator device drift: {wrong_device[:8]}"
        )
    if wrong_dtype:
        raise TypeError(
            f"Stage-2 inference Generator must be entirely BF16: {wrong_dtype[:8]}"
        )


def load_stage2_ema_generator_for_inference(
    checkpoint: str | Path,
    *,
    architecture_root: str | Path | None = None,
    device: str | torch.device,
    trusted_generator_asset: Mapping[str, Any] | None = None,
    expected_recorded_generator_asset_sha256: str | None = None,
    ops: Stage2InferenceLoaderOps | None = None,
) -> LoadedStage2InferenceGenerator:
    """Load, strictly validate, safe-merge, freeze, and place Generator EMA.

    The committed checkpoint's own canonical training config determines all
    model semantics and both training hashes. ``architecture_root`` is only a
    runtime relocation for the architecture ``config.json``; exact base
    state-dict loading and the locked Wan contract still guard its contents.
    """

    operations = ops or Stage2InferenceLoaderOps()
    directory, saved_config, resolved = _read_and_resolve_checkpoint_config(
        checkpoint,
        resolver=operations.resolve_training_config,
    )
    _require_locked_generator_contract(resolved)
    contract_hash = _sha256(resolved.contract_hash(), "training contract hash")
    launch_hash = _sha256(resolved.launch_hash(), "training launch hash")

    checkpoint_payload = operations.read_generator_ema_checkpoint(
        directory,
        expected_contract_hash=contract_hash,
        expected_launch_hash=launch_hash,
        expected_resolved_config=saved_config,
    )
    if Path(checkpoint_payload.directory).resolve() != directory:
        raise RuntimeError("Stage-2 EMA reader returned a different checkpoint")
    if canonical_json_sha256(dict(checkpoint_payload.resolved_config)) != (
        canonical_json_sha256(saved_config)
    ):
        raise RuntimeError("Stage-2 EMA reader returned a different resolved config")
    manifest = dict(checkpoint_payload.manifest)
    if manifest.get("config", {}).get("contract_hash") != contract_hash:
        raise RuntimeError("Stage-2 checkpoint manifest contract hash drifted")
    if manifest.get("config", {}).get("launch_hash") != launch_hash:
        raise RuntimeError("Stage-2 checkpoint manifest launch hash drifted")
    completed_g = manifest.get("completed_generator_updates")
    if (
        isinstance(completed_g, bool)
        or not isinstance(completed_g, int)
        or completed_g < 40
    ):
        raise RuntimeError(
            "Stage-2 inference requires initialized Generator EMA (G>=40)"
        )

    provenance = dict(checkpoint_payload.provenance)
    assets = provenance.get("assets")
    if not isinstance(assets, Mapping) or not isinstance(
        assets.get("generator"), Mapping
    ):
        raise TypeError("Stage-2 checkpoint provenance lacks its Generator asset")
    recorded_asset_with_architecture = dict(assets["generator"])
    if (trusted_generator_asset is None) != (
        expected_recorded_generator_asset_sha256 is None
    ):
        raise ValueError(
            "trusted Generator asset and its recorded provenance hash must be "
            "provided together"
        )
    if expected_recorded_generator_asset_sha256 is not None:
        expected_recorded_hash = _sha256(
            expected_recorded_generator_asset_sha256,
            "expected recorded Generator asset SHA256",
        )
        if canonical_json_sha256(recorded_asset_with_architecture) != (
            expected_recorded_hash
        ):
            raise RuntimeError(
                "Stage-2 checkpoint recorded Generator asset differs from the "
                "rank-0 runtime attestation"
            )
    recorded_asset = dict(recorded_asset_with_architecture)
    recorded_architecture_file = recorded_asset.pop("architecture_file", None)
    trusted_architecture_file: Any = None
    if trusted_generator_asset is None:
        verified_asset = operations.validate_generator_manifest(
            recorded_asset.get("manifest_path"),
            expected_checkpoint_path=recorded_asset.get("checkpoint_path"),
            expected_step=3075,
        )
    else:
        if not isinstance(trusted_generator_asset, Mapping):
            raise TypeError("trusted Stage-2 Generator asset must be an object")
        verified_asset = dict(trusted_generator_asset)
        trusted_architecture_file = verified_asset.pop("architecture_file", None)
    if not isinstance(verified_asset, Mapping):
        raise TypeError("Stage-2 Generator manifest validator returned invalid data")
    if verified_asset.get("source_step") != 3075:
        raise RuntimeError("Stage-2 Generator base lineage is not Stage-1 step3075")
    if canonical_json_sha256(dict(verified_asset)) != canonical_json_sha256(
        recorded_asset
    ):
        raise RuntimeError(
            "Stage-2 checkpoint Generator provenance differs from live Stage-1 "
            "step3075 manifest validation"
        )

    if architecture_root is None:
        architecture_source = Path(resolved.architecture_root).expanduser()
    else:
        architecture_source = Path(architecture_root).expanduser()
    if architecture_source.is_symlink() or not architecture_source.is_dir():
        raise FileNotFoundError(architecture_source)
    architecture = architecture_source.resolve()
    config_path = architecture / "config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise FileNotFoundError(config_path)
    architecture_file = _verify_runtime_architecture_file(
        recorded_architecture_file,
        config_path=config_path,
    )
    if trusted_generator_asset is not None and canonical_json_sha256(
        architecture_file
    ) != canonical_json_sha256(trusted_architecture_file):
        raise RuntimeError(
            "Stage-2 runtime architecture differs from the rank-0 attestation"
        )
    verified_asset = {
        **dict(verified_asset),
        "architecture_file": architecture_file,
    }

    wrapper = operations.build_generator(resolved, architecture)
    _require_generator_wrapper(wrapper)
    base_audit = operations.load_generator_base(
        wrapper,
        asset=verified_asset,
        verify_content_hash=trusted_generator_asset is None,
    )
    if (
        not isinstance(base_audit, Mapping)
        or base_audit.get("strict_reload_succeeded") is not True
        or isinstance(base_audit.get("tensor_count"), bool)
        or not isinstance(base_audit.get("tensor_count"), int)
        or base_audit["tensor_count"] <= 0
        or isinstance(base_audit.get("parameter_count"), bool)
        or not isinstance(base_audit.get("parameter_count"), int)
        or base_audit["parameter_count"] <= 0
    ):
        raise RuntimeError("Stage-2 strict Generator base load audit is invalid")
    operations.validate_generator_contract(wrapper, resolved)
    wrapper.requires_grad_(False)

    from utils.stage2_role_init import stage2_role_seed

    configured, _target_audit = operations.configure_generator_lora(
        wrapper.model,
        role="generator",
        spec=resolved.generator_adapter,
        seed=stage2_role_seed(resolved.training_seed, "generator"),
        is_main_process=False,
    )
    if not isinstance(configured, torch.nn.Module):
        raise TypeError("Stage-2 Generator LoRA configurer returned a non-module")
    wrapper.model = configured
    operations.load_generator_lora(
        wrapper.model,
        checkpoint_payload.generator_ema,
        expected_dtype=torch.float32,
        require_finite=True,
        verify_tensors=True,
    )
    wrapper.model.eval()
    merge = getattr(wrapper.model, "merge_and_unload", None)
    if not callable(merge):
        raise TypeError("Stage-2 configured Generator does not support LoRA merge")
    merged = merge(safe_merge=True)
    if not isinstance(merged, torch.nn.Module):
        raise TypeError("Stage-2 safe EMA merge returned a non-module")
    wrapper.model = merged
    operations.validate_generator_contract(wrapper, resolved)

    ema_sha256 = _generator_ema_sha256(manifest)
    manifest_sha256 = _sha256(
        manifest.get("manifest_sha256"), "checkpoint manifest SHA256"
    )
    # Drop the canonical FP32 EMA mapping before moving the 5B merged base.
    del checkpoint_payload, _target_audit
    gc.collect()

    requested_device = torch.device(device)
    if requested_device.type == "meta":
        raise ValueError("Stage-2 inference Generator cannot target the meta device")
    if requested_device.type == "cuda" and requested_device.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError("Stage-2 CUDA inference requested without CUDA")
        target_device = torch.device("cuda", torch.cuda.current_device())
    else:
        target_device = requested_device
    wrapper.to(device=target_device, dtype=torch.bfloat16)
    wrapper.requires_grad_(False)
    wrapper.eval()
    _audit_merged_generator(wrapper, device=target_device)

    # Keep the value alive and explicitly checked before exposing identity().
    manifest["manifest_sha256"] = manifest_sha256
    return LoadedStage2InferenceGenerator(
        generator=wrapper,
        resolved_training_config=resolved,
        checkpoint_directory=directory,
        checkpoint_manifest=manifest,
        checkpoint_provenance=provenance,
        generator_asset=dict(verified_asset),
        base_load_audit=dict(base_audit),
        training_contract_hash=contract_hash,
        training_launch_hash=launch_hash,
        generator_ema_sha256=ema_sha256,
    )


__all__ = [
    "LoadedStage2InferenceGenerator",
    "Stage2InferenceLoaderOps",
    "load_stage2_ema_generator_for_inference",
]
