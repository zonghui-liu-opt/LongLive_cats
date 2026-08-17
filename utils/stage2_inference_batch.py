"""Deterministic sample planning for Stage-2 batch deployment inference."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Sequence

from utils.stage1_causal_validation import load_causal_testset_records
from utils.stage1_continuation_validation import load_continuation_metadata
from utils.stage2_inference import STAGE2_INFERENCE_SEEDS

STAGE2_SINGLE_DATASET = "single_action"
STAGE2_TWO_ACTION_DATASET = "two_action"
STAGE2_BASELINE_PROFILE = "baseline_c8w16k4s1"
STAGE2_BASELINE_SINGLE_SAMPLES = 24
STAGE2_BASELINE_TWO_ACTION_SAMPLES = 32
STAGE2_BASELINE_TOTAL_SAMPLES = 56
_PROFILE_NAME = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")


@dataclass(frozen=True)
class Stage2InferenceSample:
    dataset: str
    row_id: int
    seed: int
    profile: str
    input_image: Path
    image_sha256: str
    row_sha256: str
    height: int
    width: int
    bucket: str
    prompts: tuple[str, ...]
    case_group: str

    @property
    def sample_key(self) -> str:
        return (
            f"{self.dataset}/row{self.row_id:03d}/"
            f"seed{self.seed:04d}/{self.profile}"
        )

    @property
    def input_image_path(self) -> Path:
        """Compatibility view for the shared strict Stage-1 image loader."""

        return self.input_image

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def output_relative_path(self) -> Path:
        return (
            Path("videos")
            / self.profile
            / self.dataset
            / (f"row{self.row_id:03d}_seed{self.seed:04d}.mp4")
        )

    @property
    def trace_relative_path(self) -> Path:
        return (
            Path("traces")
            / self.profile
            / self.dataset
            / (f"row{self.row_id:03d}_seed{self.seed:04d}.json")
        )

    def to_manifest_source(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "row_id": self.row_id,
            "seed": self.seed,
            "profile": self.profile,
            "sample_key": self.sample_key,
            "input_image": os.fspath(self.input_image),
            "image_sha256": self.image_sha256,
            "row_sha256": self.row_sha256,
            "height": self.height,
            "width": self.width,
            "bucket": self.bucket,
            "case_group": self.case_group,
            "prompts": list(self.prompts),
        }


def _formal_seeds(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in seeds):
        raise TypeError("Stage-2 inference seeds must be plain integers")
    if seeds != STAGE2_INFERENCE_SEEDS:
        raise ValueError(
            "formal Stage-2 inference requires seeds exactly "
            f"{STAGE2_INFERENCE_SEEDS}, got {seeds}"
        )
    return seeds


def _profile_names(values: Sequence[str]) -> tuple[str, ...]:
    names = tuple(values)
    if not names:
        raise ValueError("Stage-2 inference requires at least one rollout profile")
    if any(
        not isinstance(name, str)
        or not _PROFILE_NAME.fullmatch(name)
        or name != name.strip()
        for name in names
    ):
        raise ValueError(f"invalid Stage-2 rollout profile names: {names}")
    if len(set(names)) != len(names):
        raise ValueError("Stage-2 rollout profile names must be unique")
    # Import lazily so sample planning remains cheap and profile validation has
    # exactly one source of truth in the rollout-core module.
    from pipeline.stage2_rollout_profile import resolve_stage2_rollout_profile

    for name in names:
        resolved = resolve_stage2_rollout_profile(name)
        if getattr(resolved, "name", None) != name:
            raise RuntimeError("Stage-2 rollout profile resolver changed its name")
    return names


def build_stage2_inference_samples(
    *,
    single_metadata: str | os.PathLike[str],
    two_action_metadata: str | os.PathLike[str],
    seeds: Sequence[int] = STAGE2_INFERENCE_SEEDS,
    profiles: Sequence[str] = (STAGE2_BASELINE_PROFILE,),
) -> tuple[Stage2InferenceSample, ...]:
    """Build the exact dataset x row x seed x profile Cartesian product."""

    seed_values = _formal_seeds(seeds)
    profile_values = _profile_names(profiles)
    singles = load_causal_testset_records(single_metadata)
    doubles = load_continuation_metadata(two_action_metadata)
    if len(singles) != 6:
        raise RuntimeError(
            f"Stage-2 single-action metadata must contain 6 rows, got {len(singles)}"
        )
    if len(doubles) != 8:
        raise RuntimeError(
            "Stage-2 two-action metadata must contain 8 rows, " f"got {len(doubles)}"
        )

    samples: list[Stage2InferenceSample] = []
    for profile in profile_values:
        for record in singles:
            for seed in seed_values:
                samples.append(
                    Stage2InferenceSample(
                        dataset=STAGE2_SINGLE_DATASET,
                        row_id=record.row_id,
                        seed=seed,
                        profile=profile,
                        input_image=Path(record.input_image),
                        image_sha256=record.image_sha256,
                        row_sha256=record.row_sha256,
                        height=record.height,
                        width=record.width,
                        bucket=record.bucket,
                        prompts=(record.prompt,),
                        case_group=f"single_row{record.row_id:03d}",
                    )
                )
        for record in doubles:
            for seed in seed_values:
                samples.append(
                    Stage2InferenceSample(
                        dataset=STAGE2_TWO_ACTION_DATASET,
                        row_id=record.row_id,
                        seed=seed,
                        profile=profile,
                        input_image=record.input_image_path,
                        image_sha256=record.image_sha256,
                        row_sha256=record.row_sha256,
                        height=record.height,
                        width=record.width,
                        bucket=record.bucket,
                        # HOLD and soft-reanchor are intentionally absent.  The
                        # Stage-2 contract reuses only image/action A/action B.
                        prompts=(record.action_a_prompt, record.action_b_prompt),
                        case_group=record.case_group,
                    )
                )

    keys = [sample.sample_key for sample in samples]
    output_paths = [sample.output_relative_path.as_posix() for sample in samples]
    trace_paths = [sample.trace_relative_path.as_posix() for sample in samples]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Stage-2 batch planner produced duplicate sample keys")
    if len(output_paths) != len(set(output_paths)) or len(trace_paths) != len(
        set(trace_paths)
    ):
        raise RuntimeError("Stage-2 batch planner produced colliding artifact paths")
    expected = len(profile_values) * STAGE2_BASELINE_TOTAL_SAMPLES
    if len(samples) != expected:
        raise AssertionError(
            f"Stage-2 batch planner expected {expected} samples, got {len(samples)}"
        )
    return tuple(samples)


def shard_stage2_inference_samples(
    samples: Sequence[Stage2InferenceSample],
    *,
    rank: int,
    world_size: int,
) -> tuple[Stage2InferenceSample, ...]:
    """Use conflict-free rank-stride data parallelism without padding/drop."""

    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size <= 0
    ):
        raise ValueError("world_size must be a positive integer")
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or not 0 <= rank < world_size
    ):
        raise ValueError("rank must be in [0, world_size)")
    keys = [sample.sample_key for sample in samples]
    if len(keys) != len(set(keys)):
        raise ValueError("cannot shard duplicate Stage-2 sample keys")
    return tuple(samples[rank::world_size])


__all__ = [
    "STAGE2_BASELINE_PROFILE",
    "STAGE2_BASELINE_SINGLE_SAMPLES",
    "STAGE2_BASELINE_TOTAL_SAMPLES",
    "STAGE2_BASELINE_TWO_ACTION_SAMPLES",
    "STAGE2_SINGLE_DATASET",
    "STAGE2_TWO_ACTION_DATASET",
    "Stage2InferenceSample",
    "build_stage2_inference_samples",
    "shard_stage2_inference_samples",
]
