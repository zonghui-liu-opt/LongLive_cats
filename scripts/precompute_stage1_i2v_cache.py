#!/usr/bin/env python3
"""Deterministically precompute audited Wan Stage-1 I2V latent/text caches."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.distributed as dist
from omegaconf import OmegaConf

from utils.config import normalize_config
from utils.stage1_i2v_data import (
    build_source_fingerprint,
    cache_artifact_name,
    load_cache_artifact,
    decode_stage1_video,
    load_stage1_input_image,
    load_stage1_i2v_manifest,
    save_cache_artifact,
    validate_cache_tensors,
    write_cache_manifest,
)
from utils.stage1_io import sha256_file


def _initialize_distributed() -> tuple[int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    return rank, world_size, device


def _broadcast_rank0_object(value, rank: int):
    if not dist.is_initialized():
        return value
    values = [value if rank == 0 else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def _validate_precompute_runtime(config, *, world_size: int, device: torch.device) -> int:
    settings = config.get("cache_precompute", {})
    expected_world_size = int(settings.get("expected_world_size", world_size))
    if world_size != expected_world_size:
        raise RuntimeError(
            "Cache precompute world-size mismatch: "
            f"expected {expected_world_size}, got {world_size}."
        )
    require_cuda = bool(settings.get("require_cuda", False))
    if require_cuda and device.type != "cuda":
        raise RuntimeError("Cache precompute requires CUDA, but no CUDA device is available.")
    minimum_capability = settings.get("minimum_cuda_capability", None)
    if minimum_capability is not None:
        if device.type != "cuda":
            raise RuntimeError(
                "minimum_cuda_capability is configured, but cache precompute is not on CUDA."
            )
        minimum_capability = tuple(int(value) for value in minimum_capability)
        if len(minimum_capability) != 2:
            raise ValueError("minimum_cuda_capability must contain [major, minor].")
        actual_capability = torch.cuda.get_device_capability(device)
        if actual_capability < minimum_capability:
            raise RuntimeError(
                "Cache precompute requires CUDA capability >= "
                f"{minimum_capability}, got {actual_capability} on "
                f"{torch.cuda.get_device_name(device)}."
            )
    log_every_records = int(settings.get("log_every_records", 10))
    if log_every_records < 1:
        raise ValueError("cache_precompute.log_every_records must be at least 1.")
    return log_every_records


def _build_stage1_cache_vae(vae_checkpoint, *, device: torch.device):
    from utils.wan_5b_wrapper import (
        WanVAEWrapper,
        configure_wan_vae_runtime,
    )

    return configure_wan_vae_runtime(
        WanVAEWrapper(vae_checkpoint=vae_checkpoint),
        device=device,
        dtype=torch.bfloat16,
    )


def _existing_artifact_valid(path: Path, *, row_sha256: str, source_sha256: str) -> bool:
    if not path.is_file():
        return False
    from safetensors import safe_open

    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
        if metadata.get("row_sha256") != row_sha256:
            return False
        if metadata.get("source_aggregate_sha256") != source_sha256:
            return False
        load_cache_artifact(path)
    except Exception:
        return False
    return True


def _artifact_description(path: Path) -> dict:
    from safetensors.torch import load_file

    tensors = load_file(str(path), device="cpu")
    validate_cache_tensors(tensors)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
        "tensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
            }
            for name, tensor in sorted(tensors.items())
        },
    }


def _encode_record(record, *, vae, text_encoder, device, preprocessing):
    video_pixels = decode_stage1_video(
        record,
        min_source_frames=int(preprocessing.min_source_frames),
        selected_frame_start=int(preprocessing.selected_frame_start),
        selected_frame_count=int(preprocessing.selected_frame_count),
        expected_fps=float(preprocessing.expected_fps),
    )
    image_pixels = load_stage1_input_image(record)
    with torch.inference_mode():
        video_latent = vae.encode_to_latent(
            video_pixels.to(device=device, dtype=torch.bfloat16)
        )
        initial_latent = vae.encode_to_latent(
            image_pixels.to(device=device, dtype=torch.bfloat16)
        )
        encoded_prompt = text_encoder(
            text_prompts=[record.prompt], return_mask=True
        )
    if tuple(video_latent.shape[:3]) != (1, 24, 48):
        raise RuntimeError(
            f"row {record.row_id}: VAE video output must be [1,24,48,H,W], "
            f"got {tuple(video_latent.shape)}."
        )
    if tuple(initial_latent.shape[:3]) != (1, 1, 48):
        raise RuntimeError(
            f"row {record.row_id}: VAE image output must be [1,1,48,H,W], "
            f"got {tuple(initial_latent.shape)}."
        )
    if video_latent.shape[-2:] != initial_latent.shape[-2:]:
        raise RuntimeError(f"row {record.row_id}: video/image latent shapes differ.")
    tensors = {
        "video_latent": video_latent[0].to(dtype=torch.bfloat16, device="cpu"),
        "initial_latent": initial_latent[0].to(dtype=torch.bfloat16, device="cpu"),
        "prompt_embeds": encoded_prompt["prompt_embeds"][0].to(
            dtype=torch.bfloat16, device="cpu"
        ),
        "prompt_mask": encoded_prompt["prompt_mask"][0].to(
            dtype=torch.bool, device="cpu"
        ),
    }
    validate_cache_tensors(tensors)
    return tensors


def precompute(config_path: str, *, cache_dir_override: str | None = None) -> Path | None:
    config = normalize_config(OmegaConf.load(config_path))
    rank, world_size, device = _initialize_distributed()
    log_every_records = _validate_precompute_runtime(
        config, world_size=world_size, device=device
    )
    data = config.data
    preprocessing = config.preprocessing
    model_paths = config.model_paths
    cache_dir = Path(cache_dir_override or data.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    records = load_stage1_i2v_manifest(
        data.metadata_path,
        expected_num_samples=int(data.expected_num_samples),
        require_files=True,
        validate_images=True,
    )
    fingerprint = None
    if rank == 0:
        fingerprint = build_source_fingerprint(
            records,
            model_paths={
                "vae_checkpoint": model_paths.vae_checkpoint,
                "t5_checkpoint": model_paths.t5_checkpoint,
                "tokenizer_dir": model_paths.tokenizer_dir,
            },
            preprocessing=OmegaConf.to_container(preprocessing, resolve=True),
        )
    fingerprint = _broadcast_rank0_object(fingerprint, rank)
    source_sha256 = fingerprint["aggregate_sha256"]

    assigned_records = [
        record for record in records if record.row_id % world_size == rank
    ]
    pending_records = []
    for record in assigned_records:
        output_path = cache_dir / cache_artifact_name(record.row_id)
        if not _existing_artifact_valid(
            output_path,
            row_sha256=record.row_sha256,
            source_sha256=source_sha256,
        ):
            pending_records.append(record)
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
    )
    print(
        f"[rank {rank}/{world_size}] device={device_name} "
        f"assigned={len(assigned_records)} pending={len(pending_records)} "
        f"resumed={len(assigned_records) - len(pending_records)}",
        flush=True,
    )

    text_encoder = None
    vae = None
    if pending_records:
        from utils.wan_5b_wrapper import WanTextEncoder

        text_encoder = WanTextEncoder(
            t5_checkpoint=model_paths.t5_checkpoint,
            tokenizer_dir=model_paths.tokenizer_dir,
            device=device,
        ).eval().requires_grad_(False)
        vae = _build_stage1_cache_vae(
            model_paths.vae_checkpoint,
            device=device,
        )

    verified_shapes: set[tuple[int, int]] = set()
    started_at = time.perf_counter()
    for completed, record in enumerate(pending_records, start=1):
        output_path = cache_dir / cache_artifact_name(record.row_id)
        tensors = _encode_record(
            record,
            vae=vae,
            text_encoder=text_encoder,
            device=device,
            preprocessing=preprocessing,
        )
        save_cache_artifact(
            output_path,
            tensors,
            metadata={
                "schema": "longlive_stage1_i2v_cache_record",
                "schema_version": 1,
                "row_id": record.row_id,
                "row_sha256": record.row_sha256,
                "source_aggregate_sha256": source_sha256,
                "prompt_mode": "repeat_global",
            },
        )
        # Immediate online-vs-cached BF16 equivalence for at least one local
        # record of each orientation. Every rank's source tensor has already
        # been rounded to the exact cache dtype before this check.
        shape = tuple(tensors["video_latent"].shape[-2:])
        if shape not in verified_shapes:
            cached = load_cache_artifact(output_path)
            for name in tensors:
                torch.testing.assert_close(cached[name], tensors[name], rtol=0, atol=0)
            verified_shapes.add(shape)
        if completed % log_every_records == 0 or completed == len(pending_records):
            elapsed = time.perf_counter() - started_at
            print(
                f"[rank {rank}/{world_size}] completed={completed}/"
                f"{len(pending_records)} row_id={record.row_id} "
                f"elapsed={elapsed:.1f}s avg={elapsed / completed:.2f}s/record",
                flush=True,
            )

    if dist.is_initialized():
        dist.barrier()

    manifest_path = None
    if rank == 0:
        artifacts = []
        for record in records:
            path = cache_dir / cache_artifact_name(record.row_id)
            if not _existing_artifact_valid(
                path,
                row_sha256=record.row_sha256,
                source_sha256=source_sha256,
            ):
                raise RuntimeError(f"Missing or invalid cache artifact for row {record.row_id}: {path}")
            artifacts.append(_artifact_description(path))
        manifest_path = write_cache_manifest(
            cache_dir,
            records=records,
            artifacts=artifacts,
            source_fingerprint=fingerprint,
        )
        counts = Counter(record.spatial_shape for record in records)
        print("Stage-1 cache complete:", manifest_path)
        for shape, count in sorted(counts.items()):
            print(f"  bucket {shape[0]}x{shape[1]}: {count} samples ({'even' if count % 2 == 0 else 'odd'})")

    if dist.is_initialized():
        dist.barrier()
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", "--config_path", required=True)
    parser.add_argument("--cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    precompute(args.config_path, cache_dir_override=args.cache_dir)


if __name__ == "__main__":
    main()
