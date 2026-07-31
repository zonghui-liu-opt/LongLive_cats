#!/usr/bin/env python3
"""Strictly convert DiffSynth Wan2.2-TI2V-5B weights to LongLive causal format."""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import os
from pathlib import Path
import sys
from typing import Callable

# ``python scripts/convert_...py`` sets sys.path[0] to ``scripts``.  Add the
# repository root so the documented direct invocation resolves project modules.
if __package__ in (None, ""):
    sys.path.insert(0, os.fspath(Path(__file__).resolve().parents[1]))

import torch

from utils.inference_utils import load_generator_checkpoint
from utils.stage1_io import (
    aggregate_file_hash,
    atomic_output_path,
    atomic_write_json,
    sha256_file,
)
from wan_5b.textimage2video import (
    load_wan_checkpoint_in_model,
    resolve_wan_checkpoint_path,
    wan_checkpoint_source_files,
)


CHECKPOINT_FORMAT = "longlive_causal_base_init"
CHECKPOINT_VERSION = 1
CONVERTER_VERSION = 1


def _relative_source_name(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _hash_source_files(source_checkpoint) -> tuple[list[dict], str]:
    source_path = Path(source_checkpoint).expanduser().resolve()
    root = source_path if source_path.is_dir() else source_path.parent
    entries = []
    for filename in wan_checkpoint_source_files(source_path):
        path = Path(filename).resolve()
        entries.append(
            {
                "path": _relative_source_name(path, root),
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return entries, aggregate_file_hash(entries)


def _tensor_summary(state_dict: dict[str, torch.Tensor]) -> dict:
    dtype_counts = Counter(str(tensor.dtype).removeprefix("torch.") for tensor in state_dict.values())
    tensors = [
        {
            "key": key,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "numel": tensor.numel(),
        }
        for key, tensor in sorted(state_dict.items())
    ]
    return {
        "tensor_count": len(tensors),
        "total_numel": sum(item["numel"] for item in tensors),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "tensors": tensors,
    }


def _validate_loaded_state(state_dict: dict[str, torch.Tensor]) -> None:
    if not state_dict:
        raise RuntimeError("Causal model has an empty state_dict after checkpoint load.")
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Model state {key!r} is not a tensor: {type(tensor)!r}")
        if tensor.is_meta:
            raise RuntimeError(f"Model state {key!r} remains on the meta device.")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all().item():
            raise ValueError(f"Model state {key!r} contains non-finite values.")


class _NativeGeneratorWrapper(torch.nn.Module):
    """Minimal native wrapper used by injectable tiny converter tests."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model


def convert_diffsynth_checkpoint(
    *,
    source_checkpoint,
    output_path,
    model_builder: Callable[[], torch.nn.Module],
    reload_wrapper_builder: Callable[[], torch.nn.Module] | None = None,
    manifest_path=None,
    model_name="Wan2.2-TI2V-5B",
    conversion_command: list[str] | None = None,
    causal_config: dict | None = None,
    use_meta_init: bool = True,
) -> dict:
    """Convert one flat/sharded checkpoint using injectable architecture builders.

    ``model_builder`` returns the bare causal DiT whose keys must exactly match
    the DiffSynth checkpoint.  ``reload_wrapper_builder`` returns a fresh native
    wrapper with a ``model`` child.  The injectable builders keep unit tests tiny
    while the CLI uses the real TI2V-5B architecture.
    """
    output_path = Path(output_path).expanduser().resolve()
    manifest_path = Path(
        manifest_path
        if manifest_path is not None
        else output_path.with_suffix(".manifest.json")
    ).expanduser().resolve()
    resolved_source = Path(resolve_wan_checkpoint_path(source_checkpoint)).resolve()
    source_files = {Path(path).resolve() for path in wan_checkpoint_source_files(source_checkpoint)}
    if output_path in source_files or manifest_path in source_files:
        raise ValueError("Converter outputs must not overwrite a source checkpoint file.")
    if output_path == manifest_path:
        raise ValueError("output_path and manifest_path must be different files.")

    source_entries_before, source_aggregate_before = _hash_source_files(source_checkpoint)

    if use_meta_init:
        from accelerate import init_empty_weights

        with init_empty_weights():
            causal_model = model_builder()
    else:
        causal_model = model_builder()
    if not isinstance(causal_model, torch.nn.Module):
        raise TypeError("model_builder must return torch.nn.Module.")

    load_wan_checkpoint_in_model(causal_model, resolved_source)
    source_state = causal_model.state_dict()
    _validate_loaded_state(source_state)
    source_state_summary = _tensor_summary(source_state)
    del source_state

    # Casting the model in place avoids retaining both the original 5B state
    # and a second cloned BF16 state during serialization.
    causal_model.to(device="cpu", dtype=torch.bfloat16)
    converted_state = {
        f"model.{key}": tensor.detach().cpu().contiguous()
        for key, tensor in causal_model.state_dict().items()
    }
    _validate_loaded_state(converted_state)
    non_bf16 = [
        key for key, tensor in converted_state.items()
        if tensor.is_floating_point() and tensor.dtype != torch.bfloat16
    ]
    if non_bf16:
        raise RuntimeError(
            f"Converted state contains non-BF16 floating tensors: {non_bf16[:10]}"
        )
    output_state_summary = _tensor_summary(converted_state)
    coverage = {
        "expected_keys": source_state_summary["tensor_count"],
        "loaded_keys": source_state_summary["tensor_count"],
        "key_percent": 100.0,
        "shape_percent": 100.0,
    }

    payload = {
        "generator": converted_state,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_name": model_name,
        "source_aggregate_sha256": source_aggregate_before,
        "converter_version": CONVERTER_VERSION,
        "causal_config": dict(causal_config or {}),
    }
    # Keep the output temporary until fresh strict reload and the second source
    # hash audit both succeed.  A failed conversion therefore cannot replace a
    # previously valid destination.
    with atomic_output_path(output_path) as temporary_output:
        torch.save(payload, temporary_output)

        # The specification requires a fresh native causal wrapper strict
        # reload, so release every source-model/payload reference first.
        del payload, converted_state, causal_model
        gc.collect()

        if reload_wrapper_builder is None:
            reload_wrapper = _NativeGeneratorWrapper(model_builder())
        else:
            reload_wrapper = reload_wrapper_builder()
        if not isinstance(reload_wrapper, torch.nn.Module):
            raise TypeError("reload_wrapper_builder must return torch.nn.Module.")
        incompatible = load_generator_checkpoint(
            reload_wrapper, os.fspath(temporary_output), strict=True
        )
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Fresh strict reload returned incompatible keys: "
                f"missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
        del reload_wrapper
        gc.collect()

        source_entries_after, source_aggregate_after = _hash_source_files(
            source_checkpoint
        )
        if (
            source_entries_after != source_entries_before
            or source_aggregate_after != source_aggregate_before
        ):
            raise RuntimeError("Source checkpoint files changed during conversion.")

    output_sha256 = sha256_file(output_path)

    manifest = {
        "schema_version": 1,
        "checkpoint_format": CHECKPOINT_FORMAT,
        "checkpoint_version": CHECKPOINT_VERSION,
        "converter_version": CONVERTER_VERSION,
        "model_name": model_name,
        "source": {
            "input": os.fspath(Path(source_checkpoint).expanduser().resolve()),
            "resolved_checkpoint": os.fspath(resolved_source),
            "files": source_entries_before,
            "aggregate_sha256": source_aggregate_before,
        },
        "output": {
            "path": os.fspath(output_path),
            "size": output_path.stat().st_size,
            "sha256": output_sha256,
        },
        "source_state": source_state_summary,
        "generator_state": output_state_summary,
        "coverage": coverage,
        "causal_config": dict(causal_config or {}),
        "conversion_command": list(conversion_command or []),
        "strict_reload": True,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def _real_model_builders(args):
    from utils.wan_5b_wrapper import WanDiffusionWrapper, build_wan_model

    def model_builder():
        return build_wan_model(
            model_name="Wan2.2-TI2V-5B",
            is_causal=True,
            architecture_root=args.architecture_root,
            init_weights=False,
            local_attn_size=args.local_attn_size,
            sink_size=args.sink_size,
            num_frame_per_block=args.num_frame_per_block,
        )

    def reload_wrapper_builder():
        return WanDiffusionWrapper(
            model_name="Wan2.2-TI2V-5B",
            is_causal=True,
            architecture_root=args.architecture_root,
            init_weights=False,
            local_attn_size=args.local_attn_size,
            sink_size=args.sink_size,
            num_frame_per_block=args.num_frame_per_block,
            timestep_shift=5.0,
        )

    return model_builder, reload_wrapper_builder


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkpoint",
        required=True,
        help="DiffSynth safetensors/bin file, shard index, or flat directory.",
    )
    parser.add_argument(
        "--architecture-root",
        required=True,
        help="Wan2.2-TI2V-5B architecture directory containing config.json.",
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--manifest-path", default=None)
    parser.add_argument("--local-attn-size", type=int, default=-1)
    parser.add_argument("--sink-size", type=int, default=0)
    parser.add_argument("--num-frame-per-block", type=int, default=8)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    model_builder, reload_wrapper_builder = _real_model_builders(args)
    command = [sys.executable, os.fspath(Path(__file__).resolve()), *(argv or sys.argv[1:])]
    manifest = convert_diffsynth_checkpoint(
        source_checkpoint=args.source_checkpoint,
        output_path=args.output_path,
        manifest_path=args.manifest_path,
        model_builder=model_builder,
        reload_wrapper_builder=reload_wrapper_builder,
        conversion_command=command,
        causal_config={
            "local_attn_size": args.local_attn_size,
            "sink_size": args.sink_size,
            "num_frame_per_block": args.num_frame_per_block,
        },
    )
    print(
        "Converted Wan2.2-TI2V-5B causal base: "
        f"{manifest['output']['path']} ({manifest['output']['sha256']})"
    )
    print(
        "Coverage: "
        f"keys={manifest['coverage']['key_percent']:.1f}%, "
        f"shapes={manifest['coverage']['shape_percent']:.1f}%, "
        f"fresh_strict_reload={manifest['strict_reload']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
