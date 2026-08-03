#!/usr/bin/env python3
"""Prepare the image-only testsets CSV for deterministic causal I2V inference."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import DEFAULT_NEGATIVE_PROMPT  # noqa: E402
from utils.stage1_causal_validation import prepare_causal_testsets  # noqa: E402


def _env_default(name: str):
    return os.environ.get(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="testsets/metadata_6cases_480x832.csv")
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--base-checkpoint",
        default=_env_default("LONG_LIVE_STAGE1_BASE_CHECKPOINT"),
        required=_env_default("LONG_LIVE_STAGE1_BASE_CHECKPOINT") is None,
    )
    parser.add_argument(
        "--architecture-root",
        default=_env_default("LONG_LIVE_STAGE1_ARCHITECTURE_ROOT"),
        required=_env_default("LONG_LIVE_STAGE1_ARCHITECTURE_ROOT") is None,
    )
    parser.add_argument(
        "--t5-checkpoint",
        default=_env_default("LONG_LIVE_STAGE1_T5_CHECKPOINT"),
        required=_env_default("LONG_LIVE_STAGE1_T5_CHECKPOINT") is None,
    )
    parser.add_argument(
        "--tokenizer-dir",
        default=_env_default("LONG_LIVE_STAGE1_TOKENIZER_DIR"),
        required=_env_default("LONG_LIVE_STAGE1_TOKENIZER_DIR") is None,
    )
    parser.add_argument(
        "--vae-checkpoint",
        default=_env_default("LONG_LIVE_STAGE1_VAE_CHECKPOINT"),
        required=_env_default("LONG_LIVE_STAGE1_VAE_CHECKPOINT") is None,
    )
    parser.add_argument(
        "--allow-repeated-input-images",
        action="store_true",
        help="Allow one validated input image to be reused by multiple metadata rows.",
    )
    parser.add_argument("--num-latent-frames", type=int, default=24)
    parser.add_argument("--num-frame-per-block", type=int, default=8)
    parser.add_argument("--minimum-source-frames", type=int, default=97)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare_causal_testsets(
        metadata_path=args.metadata,
        output_root=args.output_root,
        base_checkpoint=args.base_checkpoint,
        architecture_root=args.architecture_root,
        t5_checkpoint=args.t5_checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        vae_checkpoint=args.vae_checkpoint,
        allow_repeated_input_images=args.allow_repeated_input_images,
        num_latent_frames=args.num_latent_frames,
        num_frame_per_block=args.num_frame_per_block,
        minimum_source_frames=args.minimum_source_frames,
        sampling_steps=args.sampling_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
