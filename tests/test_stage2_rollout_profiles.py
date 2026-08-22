from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from pipeline.stage2_rollout_profile import (
    STAGE2_ROLLOUT_PROFILE_NAMES,
    Stage2RolloutSpec,
    build_stage2_deployment_rollout_spec,
    resolve_stage2_rollout_profile,
    resolve_stage2_shift5_schedule,
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


def test_named_profile_serialization_contract_remains_frozen():
    assert resolve_stage2_rollout_profile("baseline_c8w16k4s1").to_dict() == {
        "name": "baseline_c8w16k4s1",
        "generated_episode_frames": 24,
        "chunk_frames": 8,
        "num_chunks": 3,
        "local_window_frames": 16,
        "history_frames": 8,
        "global_sink_frames": 1,
        "physical_kv_capacity_frames": 17,
        "num_denoising_steps": 4,
        "solver": "unipc",
        "timestep_shift": 5.0,
        "fresh_deploy_dit_calls": 16,
        "training_allowed": True,
        "fresh_episode_allowed": True,
    }


@pytest.mark.parametrize(
    ("steps", "timesteps"),
    [
        (1, (999,)),
        (2, (999, 833)),
        (3, (999, 908, 713)),
        (4, (999, 937, 833, 624)),
        (5, (999, 952, 882, 768, 555)),
        (6, (999, 961, 908, 833, 713, 499)),
        (7, (999, 967, 925, 869, 789, 666, 454)),
        (8, (999, 972, 937, 892, 833, 749, 624, 416)),
    ],
)
def test_shift5_reference_schedule_is_golden_and_fp32_identified(steps, timesteps):
    schedule = resolve_stage2_shift5_schedule(steps)
    assert schedule.timesteps == timesteps
    assert len(schedule.sigma_fp32_bits) == steps + 1
    assert schedule.sigma_fp32_bits[-1] == "00000000"
    assert len(schedule.sigmas) == steps + 1
    assert all(
        left > right for left, right in zip(schedule.sigmas, schedule.sigmas[1:])
    )


def test_dynamic_deployment_profile_is_canonical_immutable_and_never_trainable():
    spec = build_stage2_deployment_rollout_spec(
        chunk_frames=6,
        local_window_frames=12,
        num_denoising_steps=3,
    )
    assert spec.name == (
        "deploy_c6w12k3s1_"
        "3c3fb97b2ac89ccbff99525949a96f37c4fe79a40d569a9a168798bfe698de74"
    )
    assert spec.chunk_frames == 6
    assert spec.local_window_frames == 12
    assert spec.global_sink_frames == 1
    assert spec.num_denoising_steps == 3
    assert spec.num_chunks == 4
    assert spec.history_frames == 6
    assert spec.physical_kv_capacity_frames == 13
    assert spec.fresh_deploy_dit_calls == 17
    assert spec.training_allowed is False
    assert spec.fresh_episode_allowed is True
    assert (
        build_stage2_deployment_rollout_spec(
            chunk_frames=6,
            local_window_frames=12,
            num_denoising_steps=3,
        )
        == spec
    )
    assert resolve_stage2_rollout_profile(spec.name) == spec
    with pytest.raises(FrozenInstanceError):
        spec.num_denoising_steps = 4


@pytest.mark.parametrize(
    ("chunk", "window", "steps", "message"),
    [
        (True, 8, 4, "positive integer"),
        (1, 8, 4, "divide 24"),
        (5, 10, 4, "divide 24"),
        (8, 4, 4, "local_window_frames"),
        (6, 8, 4, "local_window_frames"),
        (8, 32, 4, "local_window_frames"),
        (8, 16, 0, "positive integer"),
        (8, 16, 9, r"\[1, 8\]"),
    ],
)
def test_dynamic_deployment_profile_rejects_out_of_contract_dimensions(
    chunk, window, steps, message
):
    with pytest.raises(ValueError, match=message):
        build_stage2_deployment_rollout_spec(
            chunk_frames=chunk,
            local_window_frames=window,
            num_denoising_steps=steps,
        )


def test_dynamic_deployment_profile_rejects_a_forged_canonical_digest():
    spec = build_stage2_deployment_rollout_spec(
        chunk_frames=8,
        local_window_frames=16,
        num_denoising_steps=4,
    )
    with pytest.raises(ValueError, match="canonical name mismatch"):
        Stage2RolloutSpec(
            name=f"{spec.name[:-1]}0",
            chunk_frames=spec.chunk_frames,
            local_window_frames=spec.local_window_frames,
            global_sink_frames=spec.global_sink_frames,
            num_denoising_steps=spec.num_denoising_steps,
        )


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
