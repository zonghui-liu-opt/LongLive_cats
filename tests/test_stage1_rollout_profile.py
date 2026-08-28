from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from utils.config import DEFAULT_NEGATIVE_PROMPT
from utils.stage1_rollout_profile import (
    STAGE1_ROLLOUT_PROFILE_SCHEMA,
    Stage1RolloutProfile,
    resolve_stage1_rollout_profile,
    resolve_stage1_rollout_profile_overrides,
)


def test_default_profile_freezes_v1_topology_and_user_window_semantics():
    profile = Stage1RolloutProfile()

    assert profile.chunk_size == 8
    assert profile.window_size == 17
    assert profile.global_sink_size == 1
    assert profile.history_size == profile.kv_cache_size == 8
    assert profile.local_window_size == 16
    assert profile.generated_frames == 24
    assert profile.num_chunks == 3
    assert profile.default_block_size is True
    assert profile.default_topology is True
    assert profile.sampling_steps == 50
    assert profile.timestep_shift == 5.0
    assert profile.guidance_scale == 5.0
    assert profile.solver == "unipc"
    assert profile.kv_quant is False
    assert profile.negative_prompt is DEFAULT_NEGATIVE_PROMPT


def test_profile_is_immutable_and_negative_prompt_is_not_a_constructor_field():
    profile = Stage1RolloutProfile()
    with pytest.raises(FrozenInstanceError):
        profile.window_size = 25
    with pytest.raises(TypeError, match="negative_prompt"):
        Stage1RolloutProfile(negative_prompt="replace me")


@pytest.mark.parametrize("sampling_steps", [4, 50])
def test_only_requested_sampling_budgets_resolve(sampling_steps):
    profile = resolve_stage1_rollout_profile(
        {
            "sampling_steps": sampling_steps,
            "timestep_shift": 3,
            "guidance_scale": 1,
        }
    )
    assert profile.sampling_steps == sampling_steps
    assert profile.timestep_shift == 3.0
    assert profile.guidance_scale == 1.0


@pytest.mark.parametrize(
    ("chunk_size", "window_size", "history_size", "local_window_size", "chunks"),
    [
        (8, 17, 8, 16, 3),
        (8, 25, 16, 24, 3),
        (4, 9, 4, 8, 6),
        (4, 17, 12, 16, 6),
        (2, 5, 2, 4, 12),
        (2, 17, 14, 16, 12),
    ],
)
def test_approved_chunk_and_total_window_topologies_resolve(
    chunk_size, window_size, history_size, local_window_size, chunks
):
    profile = Stage1RolloutProfile(
        chunk_size=chunk_size,
        window_size=window_size,
    )
    assert profile.history_size == profile.kv_cache_size == history_size
    assert profile.local_window_size == local_window_size
    assert profile.num_chunks == chunks
    assert profile.default_block_size is (chunk_size == 8)
    assert profile.default_topology is (chunk_size == 8 and window_size == 17)


def test_resolver_supports_default_instance_and_mapping_without_mutating_input():
    assert resolve_stage1_rollout_profile() == Stage1RolloutProfile()
    profile = Stage1RolloutProfile(sampling_steps=4)
    assert resolve_stage1_rollout_profile(profile) is profile

    raw = {
        "schema_version": STAGE1_ROLLOUT_PROFILE_SCHEMA,
        "window_size": 17,
        "sampling_steps": 4,
    }
    snapshot = dict(raw)
    assert resolve_stage1_rollout_profile(raw).sampling_steps == 4
    assert raw == snapshot


def test_cli_style_overrides_round_trip_through_nested_preflight_source():
    configured = {
        "chunk_size": 8,
        "window_size": 17,
        "sampling_steps": 50,
        "timestep_shift": 5.0,
        "guidance_scale": 5.0,
    }
    profile, merged = resolve_stage1_rollout_profile_overrides(
        configured,
        {
            "chunk_size": 4,
            "window_size": 17,
            "sampling_steps": 4,
            "timestep_shift": 3.0,
            "guidance_scale": 2.0,
        },
    )
    assert profile == Stage1RolloutProfile(
        chunk_size=4,
        window_size=17,
        sampling_steps=4,
        timestep_shift=3.0,
        guidance_scale=2.0,
    )
    assert Stage1RolloutProfile.from_mapping(merged) == profile
    assert configured["sampling_steps"] == 50

    with pytest.raises(ValueError, match="unknown.*negative_prompt"):
        resolve_stage1_rollout_profile_overrides({}, {"negative_prompt": "bad"})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("chunk_size", 3, "one of"),
        ("chunk_size", 6, "one of"),
        ("chunk_size", True, "positive integer"),
        ("window_size", True, "positive integer"),
        ("global_sink_size", 2, "global_sink_size=1"),
        ("generated_frames", 32, "generated_frames=24"),
        ("sampling_steps", 8, "exactly 4 or 50"),
        ("sampling_steps", True, "positive integer"),
        ("solver", "euler", "solver='unipc'"),
        ("kv_quant", True, "kv_quant=false"),
        ("kv_quant", 0, "must be a bool"),
    ],
)
def test_profile_rejects_fixed_contract_drift(field, value, message):
    with pytest.raises(ValueError, match=message):
        Stage1RolloutProfile(**{field: value})


@pytest.mark.parametrize(
    ("chunk_size", "window_size", "message"),
    [
        (8, 9, "at least one full"),
        (8, 18, "multiple of chunk_size"),
        (4, 8, "at least one full"),
        (4, 10, "multiple of chunk_size"),
        (2, 6, "multiple of chunk_size"),
        (8, 33, "local_window_size"),
        (4, 29, "local_window_size"),
        (2, 27, "local_window_size"),
    ],
)
def test_profile_rejects_invalid_history_alignment_and_oversized_local_window(
    chunk_size, window_size, message
):
    with pytest.raises(ValueError, match=message):
        Stage1RolloutProfile(chunk_size=chunk_size, window_size=window_size)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("timestep_shift", 0, ">0"),
        ("timestep_shift", -1, ">0"),
        ("timestep_shift", float("inf"), "finite"),
        ("timestep_shift", float("nan"), "finite"),
        ("timestep_shift", True, "bool is not accepted"),
        ("timestep_shift", "5", "finite real number"),
        ("guidance_scale", 0.999, ">=1"),
        ("guidance_scale", float("inf"), "finite"),
        ("guidance_scale", float("nan"), "finite"),
        ("guidance_scale", False, "bool is not accepted"),
    ],
)
def test_profile_rejects_invalid_shift_and_cfg(field, value, message):
    with pytest.raises(ValueError, match=message):
        Stage1RolloutProfile(**{field: value})


def test_mapping_cannot_override_negative_prompt_even_with_the_fixed_value():
    with pytest.raises(ValueError, match="cannot be configured"):
        Stage1RolloutProfile.from_mapping({"negative_prompt": DEFAULT_NEGATIVE_PROMPT})


def test_mapping_rejects_unknown_fields_schema_drift_and_non_mapping_inputs():
    with pytest.raises(ValueError, match="unknown.*history_size"):
        Stage1RolloutProfile.from_mapping({"history_size": 8})
    with pytest.raises(ValueError, match="schema_version"):
        Stage1RolloutProfile.from_mapping({"schema_version": "v2"})
    with pytest.raises(TypeError, match="must be a mapping"):
        Stage1RolloutProfile.from_mapping([])
    with pytest.raises(TypeError, match="None.*mapping"):
        resolve_stage1_rollout_profile("default")


def test_canonical_identity_is_complete_defensive_and_hash_stable():
    profile = Stage1RolloutProfile()
    assert (
        profile.to_dict()
        == profile.to_canonical_dict()
        == {
            "schema_version": "longlive.stage1_rollout_profile/v1",
            "chunk_size": 8,
            "window_size": 17,
            "global_sink_size": 1,
            "history_size": 8,
            "local_window_size": 16,
            "generated_frames": 24,
            "num_chunks": 3,
            "default_block_size": True,
            "default_topology": True,
            "sampling_steps": 50,
            "timestep_shift": 5.0,
            "guidance_scale": 5.0,
            "solver": "unipc",
            "kv_quant": False,
            "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        }
    )
    assert profile.canonical_sha256 == (
        "59a5bd9a4b29200a4a87fa264f667770addf09e25fb2e546328fbe9f7f00c3ac"
    )

    external = profile.canonical_dict
    external["window_size"] = 999
    assert profile.window_size == 17
    assert profile.canonical_dict["window_size"] == 17


def test_hash_changes_for_each_allowed_runtime_choice_and_numeric_normalizes():
    default = Stage1RolloutProfile()
    assert Stage1RolloutProfile(timestep_shift=5).canonical_sha256 == (
        Stage1RolloutProfile(timestep_shift=5.0).canonical_sha256
    )
    assert Stage1RolloutProfile(sampling_steps=4).canonical_sha256 != (
        default.canonical_sha256
    )
    assert Stage1RolloutProfile(timestep_shift=4.0).canonical_sha256 != (
        default.canonical_sha256
    )
    assert Stage1RolloutProfile(guidance_scale=1.0).canonical_sha256 != (
        default.canonical_sha256
    )
    assert Stage1RolloutProfile(chunk_size=4, window_size=9).canonical_sha256 != (
        default.canonical_sha256
    )
