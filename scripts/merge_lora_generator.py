#!/usr/bin/env python3
"""Strictly merge a selected Stage-1 EMA adapter into its immutable causal base."""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from utils.config import normalize_config  # noqa: E402
from utils.inference_utils import load_generator_checkpoint  # noqa: E402
from utils.lora_utils import (  # noqa: E402
    audit_lora_model,
    configure_lora_for_model,
    load_lora_safetensors_strict,
    resolve_lora_target_modules,
)
from utils.nvfp4_checkpoint import cpu_state_dict  # noqa: E402
from utils.stage1_checkpoint import checkpoint_step, validate_checkpoint  # noqa: E402
from utils.stage1_io import (  # noqa: E402
    atomic_output_path,
    atomic_write_json,
    canonical_json_sha256,
    sha256_file,
)

MERGED_CHECKPOINT_FORMAT = "longlive_stage1_causal_ema_merged"
MERGED_CHECKPOINT_VERSION = 1


def _source_snapshot(paths: set[Path]) -> list[dict]:
    entries = []
    for path in sorted(paths, key=lambda value: str(value)):
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        entries.append(
            {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": sha256_file(path),
            }
        )
    return entries


def _load_resolved_config(training_checkpoint: Path):
    path = training_checkpoint / "resolved_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return normalize_config(OmegaConf.load(path))


def _validate_adapter_metadata(
    path: Path,
    completed_step: int,
    *,
    expected_kind: str,
    expected_tensor_count: int,
    expected_global_numel: int,
) -> dict[str, str]:
    from utils.stage1_checkpoint import CHECKPOINT_SCHEMA, CHECKPOINT_SCHEMA_VERSION
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": str(CHECKPOINT_SCHEMA_VERSION),
        "kind": expected_kind,
        "completed_step": str(completed_step),
        "tensor_count": str(expected_tensor_count),
        "global_numel": str(expected_global_numel),
        "dtype": "float32",
    }
    missing = sorted(set(expected) - set(metadata))
    wrong = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if missing or wrong:
        raise RuntimeError(
            f"Stage-1 {expected_kind} adapter metadata mismatch: "
            f"missing={missing}, wrong={wrong}"
        )
    return metadata


def _adapter_tensor_schema(path: Path) -> list[dict]:
    from safetensors import safe_open

    dtype_names = {
        "F32": "float32",
        "BF16": "bfloat16",
        "F16": "float16",
    }
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return [
            {
                "key": key,
                "shape": list(handle.get_slice(key).get_shape()),
                "dtype": dtype_names.get(
                    handle.get_slice(key).get_dtype(),
                    handle.get_slice(key).get_dtype(),
                ),
            }
            for key in sorted(handle.keys())
        ]


def merge_stage1_ema_checkpoint(
    *,
    base_checkpoint: str | os.PathLike[str],
    training_checkpoint: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    config=None,
    wrapper_builder: Callable[[], torch.nn.Module] | None = None,
    output_manifest_path: str | os.PathLike[str] | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """Merge the only permitted input, ``adapter_ema.safetensors``.

    ``wrapper_builder`` is injectable for tiny unit tests. The CLI constructs
    the real causal Wan wrapper from the resolved training configuration.
    """
    base_checkpoint = Path(base_checkpoint).expanduser().resolve()
    training_checkpoint = Path(training_checkpoint).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    output_manifest_path = (
        Path(
            output_manifest_path
            if output_manifest_path is not None
            else output_path.with_suffix(".manifest.json")
        )
        .expanduser()
        .resolve()
    )
    if output_path == output_manifest_path:
        raise ValueError("Merge checkpoint and manifest outputs must be distinct.")
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    if not training_checkpoint.is_dir():
        raise FileNotFoundError(training_checkpoint)
    source_paths = {
        base_checkpoint,
        training_checkpoint / "adapter_raw.safetensors",
        training_checkpoint / "adapter_ema.safetensors",
        training_checkpoint / "resolved_config.yaml",
        training_checkpoint / "checkpoint_manifest.json",
        training_checkpoint / "base_reference.json",
        training_checkpoint / "_SUCCESS",
    }
    unsafe_outputs = [
        path
        for path in (output_path, output_manifest_path)
        if path in source_paths or path.is_relative_to(training_checkpoint)
    ]
    if unsafe_outputs:
        raise ValueError(
            "Merge outputs must be outside the immutable Stage-1 checkpoint tree "
            f"and all source artifacts: {unsafe_outputs}"
        )
    if output_path.exists() or output_manifest_path.exists():
        raise FileExistsError(
            "Stage-1 merge outputs are immutable and must not already exist: "
            f"checkpoint={output_path}, manifest={output_manifest_path}"
        )
    # Freeze the exact source bundle before parsing config or opening model
    # weights.  A second identical snapshot is required immediately before
    # publishing the output.
    source_snapshot_before = _source_snapshot(source_paths)
    source_by_path = {Path(entry["path"]): entry for entry in source_snapshot_before}
    if config is None:
        config = _load_resolved_config(training_checkpoint)
    else:
        checkpoint_config = _load_resolved_config(training_checkpoint)
        supplied_config = normalize_config(
            OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        )
        supplied = OmegaConf.to_container(supplied_config, resolve=True)
        canonical = OmegaConf.to_container(checkpoint_config, resolve=True)
        if supplied != canonical:
            raise ValueError(
                "Stage-1 merge config override differs from the checkpoint's "
                "resolved_config.yaml"
            )
    if getattr(config, "adapter", None) is None:
        raise ValueError("Resolved Stage-1 config is missing adapter settings.")

    base_sha256 = source_by_path[base_checkpoint]["sha256"]
    checkpoint_manifest = validate_checkpoint(
        training_checkpoint,
        require_resumable=False,
        expected_base_sha256=base_sha256,
        expected_topology=(6, 3, 2),
    )
    completed_step = checkpoint_step(training_checkpoint)
    raw_path = training_checkpoint / "adapter_raw.safetensors"
    ema_path = training_checkpoint / "adapter_ema.safetensors"
    expected_adapter_tensors = int(config.adapter.get("expected_adapter_tensors", 360))
    expected_trainable_parameters = int(
        config.adapter.get("expected_trainable_parameters", 57_016_320)
    )
    raw_metadata = _validate_adapter_metadata(
        raw_path,
        completed_step,
        expected_kind="raw",
        expected_tensor_count=expected_adapter_tensors,
        expected_global_numel=expected_trainable_parameters,
    )
    ema_metadata = _validate_adapter_metadata(
        ema_path,
        completed_step,
        expected_kind="ema",
        expected_tensor_count=expected_adapter_tensors,
        expected_global_numel=expected_trainable_parameters,
    )

    if wrapper_builder is None:
        from utils.wan_5b_wrapper import WanDiffusionWrapper

        model_kwargs = OmegaConf.to_container(config.model_kwargs, resolve=True)

        def wrapper_builder():
            return WanDiffusionWrapper(**model_kwargs, is_causal=True)

    generator = wrapper_builder()
    if not isinstance(generator, torch.nn.Module) or not hasattr(generator, "model"):
        raise TypeError("wrapper_builder must return a module with a .model DiT child.")
    incompatible = load_generator_checkpoint(
        generator, str(base_checkpoint), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Strict base load was incompatible: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )
    generator.eval().requires_grad_(False)

    target_modules = resolve_lora_target_modules(
        generator.model, "generator", config.adapter
    )
    generator.model = configure_lora_for_model(
        generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=False,
    )
    audit = audit_lora_model(
        generator.model,
        target_module_names=target_modules,
        expected_target_modules=config.adapter.get("expected_target_modules", None),
        expected_trainable_parameters=config.adapter.get(
            "expected_trainable_parameters", None
        ),
        expected_adapter_tensors=config.adapter.get("expected_adapter_tensors", None),
        expected_trainable_dtype=torch.float32,
        require_lora_only=True,
    )
    adapter_state = load_lora_safetensors_strict(
        generator.model,
        ema_path,
        expected_dtype=torch.float32,
        require_finite=True,
        verify_tensors=True,
    )
    adapter_parameter_count = sum(tensor.numel() for tensor in adapter_state.values())
    if adapter_parameter_count != audit["trainable_parameter_count"]:
        raise RuntimeError(
            f"Adapter shape-derived parameter count {adapter_parameter_count} differs "
            f"from model audit {audit['trainable_parameter_count']}."
        )
    tensor_schema = sorted(
        [
            {
                "key": key,
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).removeprefix("torch."),
            }
            for key, tensor in adapter_state.items()
        ],
        key=lambda item: item["key"],
    )
    raw_tensor_schema = _adapter_tensor_schema(raw_path)
    ema_tensor_schema = _adapter_tensor_schema(ema_path)
    if raw_tensor_schema != tensor_schema or ema_tensor_schema != tensor_schema:
        raise RuntimeError(
            "Stage-1 raw/EMA adapter tensor schemas differ from the strict "
            "model-derived canonical schema."
        )

    device = torch.device(device)
    generator.to(device=device)
    generator.model = generator.model.merge_and_unload(safe_merge=True)
    if generator.model.__class__.__module__.startswith("peft") or any(
        "lora_" in name for name, _ in generator.model.named_modules()
    ):
        raise RuntimeError("PEFT/LoRA modules remain after merge_and_unload.")
    generator.to(dtype=torch.bfloat16)
    generator.eval().requires_grad_(False)
    merged_state = cpu_state_dict(generator)
    bad_keys = [key for key in merged_state if "lora_" in key]
    wrong_dtype = [
        key
        for key, tensor in merged_state.items()
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16
    ]
    nonfinite = [
        key
        for key, tensor in merged_state.items()
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item()
    ]
    if bad_keys or wrong_dtype or nonfinite:
        raise RuntimeError(
            f"Invalid merged state: lora_keys={bad_keys}, wrong_dtype={wrong_dtype}, "
            f"nonfinite={nonfinite}."
        )
    payload = {
        "generator": merged_state,
        "checkpoint_format": MERGED_CHECKPOINT_FORMAT,
        "checkpoint_version": MERGED_CHECKPOINT_VERSION,
        "model_name": getattr(config.model_kwargs, "model_name", None),
        "source_base_sha256": base_sha256,
        "source_training_step": completed_step,
        "source_adapter": "adapter_ema.safetensors",
        "dtype": "bfloat16",
    }
    with atomic_output_path(output_path) as temporary_output:
        torch.save(payload, temporary_output)
        del payload, merged_state, adapter_state, generator
        gc.collect()
        fresh = wrapper_builder()
        incompatible = load_generator_checkpoint(
            fresh, str(temporary_output), strict=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("Fresh causal strict reload failed after EMA merge.")
        if any("lora_" in key for key in fresh.state_dict()):
            raise RuntimeError("Fresh merged wrapper still exposes LoRA state keys.")
        del fresh
        gc.collect()
        source_snapshot_after = _source_snapshot(source_paths)
        if source_snapshot_after != source_snapshot_before:
            raise RuntimeError(
                "Stage-1 source artifacts changed during EMA merge; refusing "
                "to publish an ambiguously attributed checkpoint."
            )

    try:
        output_sha256 = sha256_file(output_path)
        target_module_names = list(target_modules)
        merge_manifest = {
            "schema": "longlive_stage1_merge_manifest",
            "schema_version": 2,
            "checkpoint_format": MERGED_CHECKPOINT_FORMAT,
            "checkpoint_version": MERGED_CHECKPOINT_VERSION,
            "model_name": getattr(config.model_kwargs, "model_name", None),
            "base": {
                "path": str(base_checkpoint),
                "sha256": base_sha256,
                "size": source_by_path[base_checkpoint]["size"],
            },
            "training_checkpoint": {
                "path": str(training_checkpoint),
                "completed_step": completed_step,
                "manifest_sha256": checkpoint_manifest["manifest_sha256"],
                "resolved_config_sha256": source_by_path[
                    training_checkpoint / "resolved_config.yaml"
                ]["sha256"],
                "adapter": "adapter_ema.safetensors",
                "adapter_sha256": source_by_path[ema_path]["sha256"],
                "raw_adapter": {
                    "name": raw_path.name,
                    "sha256": source_by_path[raw_path]["sha256"],
                    "size": source_by_path[raw_path]["size"],
                    "metadata": raw_metadata,
                },
                "ema_adapter": {
                    "name": ema_path.name,
                    "sha256": source_by_path[ema_path]["sha256"],
                    "size": source_by_path[ema_path]["size"],
                    "metadata": ema_metadata,
                },
            },
            "adapter": {
                "type": str(config.adapter.get("type", "lora")),
                "rank": int(config.adapter.rank),
                "alpha": int(config.adapter.alpha),
                "dropout": float(config.adapter.dropout),
                "bias": str(config.adapter.bias),
                "modules_to_save": list(config.adapter.modules_to_save),
                "target_patterns": list(config.adapter.target_patterns),
                "target_module_names": target_module_names,
                "target_module_count": audit["target_module_count"],
                "target_schema_sha256": canonical_json_sha256(target_module_names),
                "tensor_schema": tensor_schema,
                "tensor_schema_sha256": canonical_json_sha256(tensor_schema),
                "adapter_tensor_count": audit["adapter_tensor_count"],
                "trainable_parameter_count": audit["trainable_parameter_count"],
            },
            "adapter_a_b_tensors": audit["adapter_tensor_count"],
            "target_modules": audit["target_module_count"],
            "trainable_parameters": audit["trainable_parameter_count"],
            "output": {
                "path": str(output_path),
                "sha256": output_sha256,
                "size": output_path.stat().st_size,
                "dtype": "bfloat16",
                "strict_reload": True,
            },
        }
        merge_manifest["manifest_sha256"] = canonical_json_sha256(merge_manifest)
        atomic_write_json(output_manifest_path, merge_manifest)
    except Exception:
        # The destination was required not to exist, so this removes only the
        # checkpoint published by this failed invocation.  Never leave an
        # unpaired artifact that could be mistaken for a completed merge.
        output_manifest_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise
    return merge_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", "--generator_ckpt", dest="base_checkpoint")
    parser.add_argument("--training-checkpoint", dest="training_checkpoint")
    parser.add_argument(
        "--output-path", "--output_path", dest="output_path", required=True
    )
    parser.add_argument(
        "--config-path",
        "--config_path",
        dest="config_path",
        help="Optional resolved config override; normally read from the training checkpoint.",
    )
    parser.add_argument(
        "--lora_ckpt",
        help="Legacy argument is intentionally rejected for Stage-1; select a training checkpoint directory.",
    )
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lora_ckpt:
        raise ValueError(
            "Stage-1 merge always selects adapter_ema.safetensors from "
            "--training-checkpoint; --lora_ckpt is not accepted."
        )
    config = None
    base = args.base_checkpoint
    if args.config_path:
        config = normalize_config(OmegaConf.load(args.config_path))
        base = base or getattr(config, "generator_ckpt", None)
    if not base:
        raise ValueError("--base-checkpoint is required.")
    if not args.training_checkpoint:
        raise ValueError("--training-checkpoint is required.")
    manifest = merge_stage1_ema_checkpoint(
        base_checkpoint=base,
        training_checkpoint=args.training_checkpoint,
        output_path=args.output_path,
        config=config,
        device=args.device,
    )
    print(f"Merged EMA adapter to {manifest['output']['path']}")
    print(f"SHA256: {manifest['output']['sha256']}")


if __name__ == "__main__":
    main()
