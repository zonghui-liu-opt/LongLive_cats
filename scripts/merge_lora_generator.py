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

import torch
from omegaconf import OmegaConf

from utils.config import normalize_config
from utils.inference_utils import load_generator_checkpoint
from utils.lora_utils import (
    audit_lora_model,
    configure_lora_for_model,
    load_lora_safetensors_strict,
    resolve_lora_target_modules,
)
from utils.nvfp4_checkpoint import cpu_state_dict
from utils.stage1_checkpoint import checkpoint_step, validate_checkpoint
from utils.stage1_io import (
    atomic_output_path,
    atomic_write_json,
    sha256_file,
)


MERGED_CHECKPOINT_FORMAT = "longlive_stage1_causal_ema_merged"
MERGED_CHECKPOINT_VERSION = 1


def _load_resolved_config(training_checkpoint: Path):
    path = training_checkpoint / "resolved_config.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return normalize_config(OmegaConf.load(path))


def _validate_adapter_metadata(path: Path, completed_step: int) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if "optimizer_step" in metadata and int(metadata["optimizer_step"]) != completed_step:
        raise RuntimeError(
            f"EMA adapter metadata step {metadata['optimizer_step']} does not match "
            f"checkpoint directory step {completed_step}."
        )
    kind = metadata.get("adapter_kind")
    if kind not in (None, "ema"):
        raise RuntimeError(f"Expected EMA adapter metadata, got adapter_kind={kind!r}.")
    return metadata


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
    output_manifest_path = Path(
        output_manifest_path
        if output_manifest_path is not None
        else output_path.with_suffix(".manifest.json")
    ).expanduser().resolve()
    if output_path in {base_checkpoint, output_manifest_path}:
        raise ValueError("Merge outputs must not overwrite the base or each other.")
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    if config is None:
        config = _load_resolved_config(training_checkpoint)
    if getattr(config, "adapter", None) is None:
        raise ValueError("Resolved Stage-1 config is missing adapter settings.")

    base_sha256 = sha256_file(base_checkpoint)
    checkpoint_manifest = validate_checkpoint(
        training_checkpoint,
        require_resumable=False,
        expected_base_sha256=base_sha256,
    )
    completed_step = checkpoint_step(training_checkpoint)
    ema_path = training_checkpoint / "adapter_ema.safetensors"
    _validate_adapter_metadata(ema_path, completed_step)

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
        incompatible = load_generator_checkpoint(fresh, str(temporary_output), strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError("Fresh causal strict reload failed after EMA merge.")
        if any("lora_" in key for key in fresh.state_dict()):
            raise RuntimeError("Fresh merged wrapper still exposes LoRA state keys.")
        del fresh
        gc.collect()

    output_sha256 = sha256_file(output_path)
    merge_manifest = {
        "schema": "longlive_stage1_merge_manifest",
        "schema_version": 1,
        "checkpoint_format": MERGED_CHECKPOINT_FORMAT,
        "checkpoint_version": MERGED_CHECKPOINT_VERSION,
        "base": {"path": str(base_checkpoint), "sha256": base_sha256},
        "training_checkpoint": {
            "path": str(training_checkpoint),
            "completed_step": completed_step,
            "manifest_sha256": checkpoint_manifest["manifest_sha256"],
            "adapter": "adapter_ema.safetensors",
            "adapter_sha256": sha256_file(ema_path),
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
    atomic_write_json(output_manifest_path, merge_manifest)
    return merge_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkpoint", "--generator_ckpt", dest="base_checkpoint")
    parser.add_argument("--training-checkpoint", dest="training_checkpoint")
    parser.add_argument("--output-path", "--output_path", dest="output_path", required=True)
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
