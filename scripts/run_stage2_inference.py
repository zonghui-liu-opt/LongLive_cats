#!/usr/bin/env python3
"""Run strict Stage-2 Generator-EMA batch inference under torchrun."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.stage2_rollout_profile import (  # noqa: E402 - isolated CLI bootstrap
    resolve_stage2_rollout_profile,
)
from utils.stage1_io import canonical_json_sha256  # noqa: E402 - isolated CLI bootstrap
from utils.stage2_inference_batch import (  # noqa: E402 - isolated CLI bootstrap
    STAGE2_SINGLE_DATASET,
    build_stage2_inference_samples,
)
from utils.stage2_inference_config import (  # noqa: E402 - isolated CLI bootstrap
    STAGE2_INFERENCE_CONFIG_SCHEMA,
    ResolvedStage2InferenceConfig,
    load_stage2_inference_config,
)
from utils.stage2_inference_runtime import (  # noqa: E402 - isolated CLI bootstrap
    run_stage2_inference,
)
from utils.stage2_inference_sweep_config import (  # noqa: E402 - isolated CLI bootstrap
    STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA,
    ResolvedStage2InferenceSweepConfig,
    load_stage2_inference_sweep_config,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "infer_i2v_stage2_baseline.yaml"
STAGE2_INFERENCE_PLAN_SCHEMA = "longlive_stage2_inference_plan/v1"

_FORMAL_SEEDS = (1, 2, 3, 4)
_FORMAL_SINGLE_ROWS = tuple(range(6))
_FORMAL_TWO_ACTION_ROWS = tuple(range(8))
_QUICK_SEEDS = (1,)
_QUICK_SINGLE_ROWS = (0, 3)
_QUICK_TWO_ACTION_ROWS = (0, 4)


def load_stage2_inference_run_config(
    path: str | Path,
) -> ResolvedStage2InferenceConfig | ResolvedStage2InferenceSweepConfig:
    """Dispatch one strict inference YAML by its versioned root schema."""

    from omegaconf import OmegaConf

    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(candidate)
    schema = OmegaConf.select(OmegaConf.load(candidate), "schema")
    if schema == STAGE2_INFERENCE_CONFIG_SCHEMA:
        return load_stage2_inference_config(candidate)
    if schema == STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA:
        return load_stage2_inference_sweep_config(candidate)
    raise ValueError(f"unknown Stage-2 inference config schema: {schema!r}")


def _override_sweep_evaluation(
    config: ResolvedStage2InferenceConfig | ResolvedStage2InferenceSweepConfig,
    mode: str | None,
) -> ResolvedStage2InferenceConfig | ResolvedStage2InferenceSweepConfig:
    if mode is None:
        return config
    if not isinstance(config, ResolvedStage2InferenceSweepConfig):
        raise TypeError("--evaluation is valid only for a Stage-2 sweep config")
    if mode == "formal":
        seeds = _FORMAL_SEEDS
        single_rows = _FORMAL_SINGLE_ROWS
        two_action_rows = _FORMAL_TWO_ACTION_ROWS
    elif mode == "quick":
        seeds = _QUICK_SEEDS
        single_rows = _QUICK_SINGLE_ROWS
        two_action_rows = _QUICK_TWO_ACTION_ROWS
    else:  # argparse guards the public CLI; keep the helper fail-closed.
        raise ValueError(f"unknown Stage-2 sweep evaluation mode: {mode!r}")
    return replace(
        config,
        evaluation_mode=mode,
        seeds=seeds,
        single_row_ids=single_rows,
        two_action_row_ids=two_action_rows,
    )


def build_stage2_inference_plan(
    config: ResolvedStage2InferenceConfig | ResolvedStage2InferenceSweepConfig,
) -> dict[str, Any]:
    """Build a CPU-only, exact work and cache plan before torchrun starts."""

    is_sweep = isinstance(config, ResolvedStage2InferenceSweepConfig)
    sample_options: dict[str, Any] = {}
    if is_sweep:
        sample_options = {
            "single_row_ids": config.single_row_ids,
            "two_action_row_ids": config.two_action_row_ids,
        }
    samples = build_stage2_inference_samples(
        single_metadata=config.single_metadata,
        two_action_metadata=config.two_action_metadata,
        seeds=config.seeds,
        profiles=config.profiles,
        **sample_options,
    )
    if len(samples) % len(config.profiles) != 0:
        raise RuntimeError("Stage-2 inference sample plan is not profile-complete")
    base_samples = len(samples) // len(config.profiles)
    baseline = resolve_stage2_rollout_profile("baseline_c8w16k4s1")
    baseline_calls = baseline.fresh_deploy_dit_calls
    profiles = []
    total_calls = 0
    for name in config.profiles:
        spec = resolve_stage2_rollout_profile(name)
        if spec.global_sink_frames == 1:
            single_calls = spec.fresh_deploy_dit_calls
            two_action_calls = 2 * spec.fresh_deploy_dit_calls - 1
        else:
            # Continuation-only profiles bootstrap episode A with the baseline.
            single_calls = baseline_calls
            two_action_calls = baseline_calls + spec.fresh_deploy_dit_calls - 1
        profile_calls = sum(
            (
                single_calls
                if sample.dataset == STAGE2_SINGLE_DATASET
                else two_action_calls
            )
            for sample in samples
            if sample.profile == name
        )
        total_calls += profile_calls
        self_kv_bytes = (
            spec.physical_kv_capacity_frames
            * 390
            * 3072
            * 2  # K and V
            * 2  # BF16 bytes
            * 30  # TI2V-5B transformer blocks
        )
        profiles.append(
            {
                "name": spec.name,
                "chunk_frames": spec.chunk_frames,
                "local_window_frames": spec.local_window_frames,
                "global_sink_frames": spec.global_sink_frames,
                "num_denoising_steps": spec.num_denoising_steps,
                "num_chunks": spec.num_chunks,
                "fresh_episode_dit_calls": spec.fresh_deploy_dit_calls,
                "generator_forward_calls": profile_calls,
                "self_kv_gib_per_sample": round(self_kv_bytes / 2**30, 6),
            }
        )
    divisors = [
        value
        for value in range(1, min(8, base_samples) + 1)
        if base_samples % value == 0
    ]
    return {
        "schema": STAGE2_INFERENCE_PLAN_SCHEMA,
        "run_kind": "deployment_sweep" if is_sweep else "formal_named_profiles",
        "evaluation_mode": config.evaluation_mode if is_sweep else "formal",
        "output_root": config.output_root,
        "profile_count": len(config.profiles),
        "base_samples_per_profile": base_samples,
        "expected_sample_count": len(samples),
        "sample_plan_sha256": canonical_json_sha256(
            [sample.to_manifest_source() for sample in samples]
        ),
        "rank_reuse_divisors_up_to_8": divisors,
        "generator_forward_calls": total_calls,
        "resolved_contract_hash": config.contract_hash(),
        "resolved_launch_hash": config.launch_hash(),
        "profiles": profiles,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the strict Stage-2 single/two-action EMA inference matrix. "
            "Launch with torchrun; profiles are selected only by the strict config."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=(
            "Stage-2 inference YAML. Any canonical rollout profile is accepted "
            "by the strict resolver; the release shell uses the baseline config."
        ),
    )
    parser.add_argument(
        "--evaluation",
        choices=("quick", "formal"),
        help=(
            "Override only a sweep config's evaluation subset. 'formal' expands "
            "to all 56 canonical samples per profile."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Resolve and print the exact CPU-only sweep plan; do not initialize CUDA.",
    )
    parser.add_argument(
        "--print-plan-field",
        action="append",
        choices=(
            "evaluation_mode",
            "profile_count",
            "base_samples_per_profile",
            "expected_sample_count",
            "generator_forward_calls",
            "resolved_contract_hash",
            "sample_plan_sha256",
        ),
        help="Print selected scalar plan fields as one tab-separated line and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_stage2_inference_run_config(args.config)
    config = _override_sweep_evaluation(config, args.evaluation)
    if args.plan_only or args.print_plan_field is not None:
        plan = build_stage2_inference_plan(config)
        if args.print_plan_field is not None:
            print("\t".join(str(plan[field]) for field in args.print_plan_field))
        else:
            print(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    result = run_stage2_inference(config)
    if result["rank"] == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
