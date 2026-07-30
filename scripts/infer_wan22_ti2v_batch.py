#!/usr/bin/env python3
"""Batch inference for the original bidirectional Wan2.2-TI2V-5B model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass(frozen=True)
class Sample:
    index: int
    image_path: Path
    prompt: str
    height: int
    width: int
    bucket: str

    @property
    def output_stem(self) -> str:
        return f"{self.index:04d}_{self.image_path.stem}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run batched, full-sequence bidirectional Wan2.2-TI2V-5B "
            "inference from a metadata CSV."
        ))
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument(
        "--auxiliary-dir", type=Path, default=None,
        help=(
            "Directory containing T5, VAE and tokenizer files. Defaults to "
            "--checkpoint-dir; set this to the original model root when the "
            "merged directory contains DiT weights only."
        ))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/wan22_ti2v_5b_bidirectional"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frame-num", type=int, default=81)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--solver", choices=("unipc", "dpm++"), default="unipc")
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--parallel-mode", choices=("data", "ulysses"), default="data",
        help=(
            "data: each GPU handles different CSV rows; ulysses: all GPUs "
            "cooperate on the same batch with sequence parallelism."
        ))
    parser.add_argument(
        "--offload-model", action=argparse.BooleanOptionalAction, default=False,
        help="Move T5/DiT back to CPU between stages to reduce VRAM usage.")
    parser.add_argument(
        "--t5-cpu", action=argparse.BooleanOptionalAction, default=False,
        help="Run the text encoder on CPU.")
    parser.add_argument(
        "--convert-model-dtype", action=argparse.BooleanOptionalAction,
        default=True, help="Convert DiT parameters to the configured BF16 dtype.")
    parser.add_argument(
        "--resume", action=argparse.BooleanOptionalAction, default=True,
        help="Skip samples whose MP4 already exists.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate and print the work split without loading CUDA or weights.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.frame_num < 1 or (args.frame_num - 1) % 4 != 0:
        parser.error("--frame-num must be 4n+1, for example 81")
    if args.sampling_steps < 1:
        parser.error("--sampling-steps must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def load_samples(metadata_path: Path, limit: int | None = None) -> list[Sample]:
    metadata_path = metadata_path.expanduser().resolve()
    required = {"input_image", "prompt", "height", "width"}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(
                f"Metadata is missing required columns: {sorted(missing)}")
        rows = list(reader)

    samples = []
    for index, row in enumerate(rows[:limit] if limit is not None else rows):
        image_path = Path(row["input_image"])
        if not image_path.is_absolute():
            image_path = metadata_path.parent / image_path
        image_path = image_path.resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {image_path}")
        height, width = int(row["height"]), int(row["width"])
        with Image.open(image_path) as image:
            actual_width, actual_height = image.size
        if (actual_height, actual_width) != (height, width):
            raise ValueError(
                f"{image_path}: CSV size {(height, width)} does not match "
                f"image size {(actual_height, actual_width)}")
        prompt = row["prompt"].strip()
        if not prompt:
            raise ValueError(f"Row {index} has an empty prompt")
        samples.append(Sample(
            index=index,
            image_path=image_path,
            prompt=prompt,
            height=height,
            width=width,
            bucket=row.get("bucket", "").strip(),
        ))
    if not samples:
        raise ValueError(f"No samples found in {metadata_path}")
    return samples


def make_batches(samples: list[Sample], batch_size: int) -> list[list[Sample]]:
    """Preserve order within each resolution and never mix tensor shapes."""
    buckets: dict[tuple[int, int], list[Sample]] = defaultdict(list)
    key_order = []
    for sample in samples:
        key = (sample.height, sample.width)
        if key not in buckets:
            key_order.append(key)
        buckets[key].append(sample)
    batches = []
    for key in key_order:
        bucket = buckets[key]
        batches.extend(
            bucket[start:start + batch_size]
            for start in range(0, len(bucket), batch_size)
        )
    return batches


def validate_checkpoint_dir(
        checkpoint_dir: Path, auxiliary_dir: Path | None = None) -> None:
    """Validate native/DiffSynth merged layouts; config.json is optional."""
    auxiliary_dir = auxiliary_dir or checkpoint_dir
    required = [
        auxiliary_dir / "Wan2.2_VAE.pth",
        auxiliary_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        auxiliary_dir / "google" / "umt5-xxl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    model_files = (
        list(checkpoint_dir.glob("diffusion_pytorch_model*.safetensors"))
        or list(checkpoint_dir.glob("diffusion_pytorch_model*.bin"))
    )
    if not model_files:
        missing.append(str(checkpoint_dir / "diffusion_pytorch_model*"))
    if missing:
        raise FileNotFoundError(
            "Incomplete Wan2.2-TI2V-5B checkpoint; missing: "
            + ", ".join(missing))


def distributed_info(parallel_mode: str) -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Wan2.2-TI2V-5B inference")
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if parallel_mode == "ulysses" and world_size > 1:
        if 24 % world_size != 0:
            raise ValueError(
                f"Ulysses world size must divide 24 attention heads; got {world_size}")
        from wan_5b.distributed.util import set_sequence_parallel_group
        set_sequence_parallel_group(dist.group.WORLD)
    return rank, local_rank, world_size


def save_sample(video: torch.Tensor, sample: Sample, output_path: Path,
                *, seed: int, args: argparse.Namespace) -> None:
    from wan_5b.utils.utils import save_video

    temp_path = output_path.with_name(
        f".{output_path.stem}.{os.getpid()}.tmp.mp4")
    try:
        temp_path.unlink(missing_ok=True)
        save_video(video.unsqueeze(0), str(temp_path), fps=args.fps)
        if not temp_path.is_file() or temp_path.stat().st_size == 0:
            raise RuntimeError(
                f"Video writer did not create a valid file: {temp_path}")
        os.replace(temp_path, output_path)
    finally:
        temp_path.unlink(missing_ok=True)
    sidecar = {
        "input_image": str(sample.image_path),
        "prompt": sample.prompt,
        "height": sample.height,
        "width": sample.width,
        "bucket": sample.bucket,
        "seed": seed,
        "frame_num": args.frame_num,
        "fps": args.fps,
        "sampling_steps": args.sampling_steps,
        "solver": args.solver,
        "shift": args.shift,
        "guide_scale": args.guide_scale,
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "auxiliary_dir": str(
            (args.auxiliary_dir or args.checkpoint_dir).expanduser().resolve()),
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    samples = load_samples(args.metadata, args.limit)
    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    auxiliary_dir = (
        args.auxiliary_dir or args.checkpoint_dir).expanduser().resolve()
    validate_checkpoint_dir(checkpoint_dir, auxiliary_dir)

    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    all_batches = make_batches(samples, args.batch_size)
    if args.parallel_mode == "data" and env_world_size > 1:
        batches = all_batches[env_rank::env_world_size]
    else:
        batches = all_batches
    local_sample_count = sum(len(batch) for batch in batches)

    print(
        f"[rank {env_rank}] samples={local_sample_count}, batches={len(batches)}, "
        f"batch_size={args.batch_size}, mode={args.parallel_mode}",
        flush=True,
    )
    if args.dry_run:
        for batch in batches:
            shape = f"{batch[0].height}x{batch[0].width}"
            print(f"  {shape}: {[sample.index for sample in batch]}")
        return

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rank, local_rank, world_size = distributed_info(args.parallel_mode)
    use_sp = args.parallel_mode == "ulysses" and world_size > 1
    model_rank = rank if use_sp else 0

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from wan_5b import WanTI2V
    from wan_5b.configs import WAN_CONFIGS

    pipe = WanTI2V(
        config=WAN_CONFIGS["ti2v-5B"],
        checkpoint_dir=str(checkpoint_dir),
        auxiliary_dir=str(auxiliary_dir),
        device_id=local_rank,
        rank=model_rank,
        use_sp=use_sp,
        t5_cpu=args.t5_cpu,
        init_on_cpu=True,
        convert_model_dtype=args.convert_model_dtype,
    )

    should_save = not use_sp or rank == 0
    progress = tqdm(batches, desc=f"rank {rank}", disable=not should_save)
    try:
        for batch in progress:
            output_paths = [
                output_dir / f"{sample.output_stem}.mp4" for sample in batch
            ]
            if args.resume and all(path.is_file() and path.stat().st_size > 0
                                   for path in output_paths):
                continue

            images = []
            for sample in batch:
                with Image.open(sample.image_path) as image:
                    images.append(image.convert("RGB"))
            seeds = [args.seed + sample.index for sample in batch]
            videos = pipe.i2v_batch(
                input_prompts=[sample.prompt for sample in batch],
                imgs=images,
                max_area=batch[0].height * batch[0].width,
                frame_num=args.frame_num,
                shift=args.shift,
                sample_solver=args.solver,
                sampling_steps=args.sampling_steps,
                guide_scale=args.guide_scale,
                n_prompts=args.negative_prompt,
                seeds=seeds,
                offload_model=args.offload_model,
            )
            if should_save:
                if videos is None or len(videos) != len(batch):
                    raise RuntimeError(
                        f"Expected {len(batch)} decoded videos, got "
                        f"{None if videos is None else len(videos)}")
                for video, sample, output_path, seed in zip(
                        videos, batch, output_paths, seeds):
                    if args.resume and output_path.is_file() and output_path.stat().st_size > 0:
                        continue
                    save_sample(video, sample, output_path, seed=seed, args=args)
                    print(f"[rank {rank}] saved {output_path}", flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
