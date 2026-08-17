from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from utils.stage2_inference_batch import (
    STAGE2_BASELINE_PROFILE,
    STAGE2_BASELINE_TOTAL_SAMPLES,
    STAGE2_SINGLE_DATASET,
    STAGE2_TWO_ACTION_DATASET,
    build_stage2_inference_samples,
    shard_stage2_inference_samples,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINGLE_METADATA = PROJECT_ROOT / "testsets" / "metadata_6cases_480x832.csv"
TWO_METADATA = (
    PROJECT_ROOT
    / "testsets"
    / "metadata_8cases_two_actions_continuation_480x832_253frames.csv"
)


def _samples(*, profiles=(STAGE2_BASELINE_PROFILE,)):
    return build_stage2_inference_samples(
        single_metadata=SINGLE_METADATA,
        two_action_metadata=TWO_METADATA,
        profiles=profiles,
    )


def test_formal_baseline_plan_is_exact_24_single_plus_32_two_action():
    samples = _samples()
    assert len(samples) == STAGE2_BASELINE_TOTAL_SAMPLES == 56
    counts = Counter(sample.dataset for sample in samples)
    assert counts == {STAGE2_SINGLE_DATASET: 24, STAGE2_TWO_ACTION_DATASET: 32}
    assert {sample.seed for sample in samples} == {1, 2, 3, 4}
    assert {sample.profile for sample in samples} == {STAGE2_BASELINE_PROFILE}
    assert len({sample.sample_key for sample in samples}) == 56
    assert len({sample.output_relative_path for sample in samples}) == 56
    assert len({sample.trace_relative_path for sample in samples}) == 56

    per_row = Counter((sample.dataset, sample.row_id) for sample in samples)
    assert set(per_row.values()) == {4}
    assert sorted(
        row for dataset, row in per_row if dataset == STAGE2_SINGLE_DATASET
    ) == list(range(6))
    assert sorted(
        row for dataset, row in per_row if dataset == STAGE2_TWO_ACTION_DATASET
    ) == list(range(8))


def test_two_action_plan_reuses_only_a_b_and_never_hold_or_soft_reanchor():
    samples = [
        sample for sample in _samples() if sample.dataset == STAGE2_TWO_ACTION_DATASET
    ]
    assert all(len(sample.prompts) == 2 for sample in samples)
    assert all(
        "保持标准直立蹲坐，目光稳定" not in prompt
        for sample in samples
        for prompt in sample.prompts
    )
    assert all("hold" not in sample.to_manifest_source() for sample in samples)
    assert all("soft_reanchor" not in sample.to_manifest_source() for sample in samples)


def test_profile_is_part_of_every_sample_and_artifact_key():
    profiles = (STAGE2_BASELINE_PROFILE, "c4w8k2s1")
    samples = _samples(profiles=profiles)
    assert len(samples) == 112
    by_profile = Counter(sample.profile for sample in samples)
    assert by_profile == {profile: 56 for profile in profiles}
    assert len({sample.sample_key for sample in samples}) == 112
    assert len({sample.output_relative_path for sample in samples}) == 112


def test_rank_stride_sharding_has_no_padding_drop_or_collision():
    samples = _samples()
    shards = [
        shard_stage2_inference_samples(samples, rank=rank, world_size=8)
        for rank in range(8)
    ]
    flat = [sample for shard in shards for sample in shard]
    assert Counter(map(len, shards)) == {7: 8}
    assert len(flat) == len(samples)
    assert {sample.sample_key for sample in flat} == {
        sample.sample_key for sample in samples
    }
    assert len({sample.output_relative_path for sample in flat}) == len(samples)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"seeds": (1,)}, "seeds exactly"),
        ({"seeds": (1, 2, 3, True)}, "plain integers"),
        ({"profiles": ()}, "at least one"),
        ({"profiles": (STAGE2_BASELINE_PROFILE,) * 2}, "unique"),
        ({"profiles": ("unknown_profile",)}, "unknown"),
    ],
)
def test_batch_planner_rejects_nonformal_or_ambiguous_matrix(kwargs, message):
    with pytest.raises((TypeError, ValueError), match=message):
        build_stage2_inference_samples(
            single_metadata=SINGLE_METADATA,
            two_action_metadata=TWO_METADATA,
            **kwargs,
        )
