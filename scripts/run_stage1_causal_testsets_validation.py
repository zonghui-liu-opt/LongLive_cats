#!/usr/bin/env python3
"""Run base audit, testsets preparation, causal inference, and MP4 gates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage1_causal_validation import (  # noqa: E402
    prepare_causal_testsets,
    validate_causal_testset_outputs,
    validate_converted_causal_base,
)
from utils.stage1_io import atomic_write_json  # noqa: E402


def _required_env(parser: argparse.ArgumentParser, flag: str, env_name: str):
    value = os.environ.get(env_name)
    parser.add_argument(flag, default=value, required=value is None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default="testsets/metadata_6cases_480x832.csv")
    parser.add_argument("--work-dir", required=True)
    _required_env(parser, "--source-checkpoint", "LONG_LIVE_STAGE1_SOURCE_CHECKPOINT")
    _required_env(parser, "--base-checkpoint", "LONG_LIVE_STAGE1_BASE_CHECKPOINT")
    _required_env(parser, "--base-manifest", "LONG_LIVE_STAGE1_BASE_MANIFEST")
    _required_env(parser, "--architecture-root", "LONG_LIVE_STAGE1_ARCHITECTURE_ROOT")
    _required_env(parser, "--t5-checkpoint", "LONG_LIVE_STAGE1_T5_CHECKPOINT")
    _required_env(parser, "--tokenizer-dir", "LONG_LIVE_STAGE1_TOKENIZER_DIR")
    _required_env(parser, "--vae-checkpoint", "LONG_LIVE_STAGE1_VAE_CHECKPOINT")
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--minimum-first-frame-psnr-db", type=float, default=12.0)
    parser.add_argument("--minimum-frame-std", type=float, default=5.0)
    parser.add_argument("--minimum-temporal-abs-diff", type=float, default=0.05)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Audit and prepare configs/carriers without launching CUDA inference.",
    )
    parser.add_argument(
        "--skip-finite-check",
        action="store_true",
        help="Skip the expensive full BF16 finite scan while retaining schema/hash checks.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    work_dir = Path(args.work_dir).expanduser().resolve()
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError(
            f"Validation work directory must be empty to reject stale videos: {work_dir}"
        )
    work_dir.mkdir(parents=True, exist_ok=True)

    base_report = validate_converted_causal_base(
        args.base_checkpoint,
        args.base_manifest,
        source_checkpoint=args.source_checkpoint,
        expected_num_frame_per_block=8,
        check_finite=not args.skip_finite_check,
    )
    prepared_root = work_dir / "prepared"
    prepared = prepare_causal_testsets(
        metadata_path=args.metadata,
        output_root=prepared_root,
        base_checkpoint=args.base_checkpoint,
        architecture_root=args.architecture_root,
        t5_checkpoint=args.t5_checkpoint,
        tokenizer_dir=args.tokenizer_dir,
        vae_checkpoint=args.vae_checkpoint,
        num_latent_frames=24,
        num_frame_per_block=8,
        minimum_source_frames=97,
        sampling_steps=args.sampling_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
    )
    report: dict[str, object] = {
        "status": "prepared" if args.prepare_only else "running",
        "base": base_report,
        "prepared_manifest": os.fspath(prepared_root / "prepared_manifest.json"),
        "inference_configs": [bucket["config_path"] for bucket in prepared["buckets"]],
    }
    atomic_write_json(work_dir / "validation_report.json", report)
    if args.prepare_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    try:
        for bucket in prepared["buckets"]:
            command = [
                sys.executable,
                os.fspath(PROJECT_ROOT / "inference.py"),
                "--config_path",
                bucket["config_path"],
            ]
            print(
                f"[stage1-causal-validation] running {bucket['bucket_id']}: "
                f"{' '.join(command)}"
            )
            report["active_bucket"] = bucket["bucket_id"]
            atomic_write_json(work_dir / "validation_report.json", report)
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)

        report.pop("active_bucket", None)
        output_report = validate_causal_testset_outputs(
            prepared_root / "prepared_manifest.json",
            minimum_first_frame_psnr_db=args.minimum_first_frame_psnr_db,
            minimum_frame_std=args.minimum_frame_std,
            minimum_temporal_abs_diff=args.minimum_temporal_abs_diff,
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        atomic_write_json(work_dir / "validation_report.json", report)
        raise
    report["status"] = "pass"
    report["outputs"] = output_report
    atomic_write_json(work_dir / "validation_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
