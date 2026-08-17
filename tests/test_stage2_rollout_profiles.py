from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pipeline.stage2_rollout_profile import (
    STAGE2_ROLLOUT_PROFILE_NAMES,
    Stage2RolloutSpec,
    resolve_stage2_rollout_profile,
)


@pytest.mark.parametrize(
    (
        "name",
        "chunk",
        "window",
        "history",
        "sink",
        "steps",
        "capacity",
        "calls",
        "training_allowed",
    ),
    [
        ("baseline_c8w16k4s1", 8, 16, 8, 1, 4, 17, 16, True),
        ("stress_c8w24k4s1", 8, 24, 16, 1, 4, 25, 16, False),
        ("lower_c8w8k4s1", 8, 8, 0, 1, 4, 9, 16, False),
        ("c4w12k4s1", 4, 12, 8, 1, 4, 13, 31, True),
        ("c4w8k4s1", 4, 8, 4, 1, 4, 9, 31, True),
        ("c4w8k2s1", 4, 8, 4, 1, 2, 9, 19, True),
        ("c8w16k4s4", 8, 16, 8, 4, 4, 20, 16, False),
        ("c8w16k4s8", 8, 16, 8, 8, 4, 24, 16, False),
    ],
)
def test_named_rollout_profiles_derive_one_topology_without_duplicate_fields(
    name,
    chunk,
    window,
    history,
    sink,
    steps,
    capacity,
    calls,
    training_allowed,
):
    spec = resolve_stage2_rollout_profile(name)
    assert spec.name == name
    assert spec.generated_episode_frames == 24
    assert spec.chunk_frames == chunk
    assert spec.num_chunks == 24 // chunk
    assert spec.local_window_frames == window
    assert spec.history_frames == history == window - chunk
    assert spec.global_sink_frames == sink
    assert spec.num_denoising_steps == steps
    assert spec.physical_kv_capacity_frames == capacity == sink + window
    assert spec.fresh_deploy_dit_calls == calls == 1 + (24 // chunk) * (steps + 1)
    assert spec.training_allowed is training_allowed
    assert spec.fresh_episode_allowed is (sink == 1)
    assert spec.solver == "unipc"
    assert spec.timestep_shift == 5.0


def test_rollout_profile_catalog_is_exact_and_immutable():
    assert STAGE2_ROLLOUT_PROFILE_NAMES == (
        "baseline_c8w16k4s1",
        "stress_c8w24k4s1",
        "lower_c8w8k4s1",
        "c4w12k4s1",
        "c4w8k4s1",
        "c4w8k2s1",
        "c8w16k4s4",
        "c8w16k4s8",
    )
    spec = resolve_stage2_rollout_profile("baseline_c8w16k4s1")
    with pytest.raises(FrozenInstanceError):
        spec.chunk_frames = 4


def test_rollout_profiles_reject_unknown_names_and_unnamed_combinations():
    with pytest.raises(ValueError, match="unknown Stage-2 rollout profile"):
        resolve_stage2_rollout_profile("c4w12k2s8")
    with pytest.raises(ValueError, match="does not match its named contract"):
        Stage2RolloutSpec(
            name="baseline_c8w16k4s1",
            chunk_frames=4,
            local_window_frames=12,
            global_sink_frames=1,
            num_denoising_steps=4,
        )
