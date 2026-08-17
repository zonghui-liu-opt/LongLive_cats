from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pipeline.stage2_rollout_profile import (
    STAGE2_ROLLOUT_PROFILE_NAMES,
    resolve_stage2_rollout_profile,
)
from tests.test_stage2_inference import _FakeVAE
from tests.test_stage2_rollout import _FakeGenerator, _inputs, _pipeline
from utils.stage2_inference import (
    generate_stage2_single_action,
    generate_stage2_two_action,
)
from utils.stage2_inference_artifacts import _validate_generation_trace
from utils.stage2_inference_batch import (
    STAGE2_SINGLE_DATASET,
    STAGE2_TWO_ACTION_DATASET,
    build_stage2_inference_samples,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SINGLE_METADATA = PROJECT_ROOT / "testsets" / "metadata_6cases_480x832.csv"
TWO_ACTION_METADATA = (
    PROJECT_ROOT
    / "testsets"
    / "metadata_8cases_two_actions_continuation_480x832_253frames.csv"
)


@pytest.mark.parametrize("profile_name", STAGE2_ROLLOUT_PROFILE_NAMES)
def test_core_single_action_trace_satisfies_the_artifact_gate(profile_name):
    sample = next(
        item
        for item in build_stage2_inference_samples(
            single_metadata=SINGLE_METADATA,
            two_action_metadata=TWO_ACTION_METADATA,
            profiles=(profile_name,),
        )
        if item.dataset == STAGE2_SINGLE_DATASET
    )
    target_spec = resolve_stage2_rollout_profile(profile_name)
    actual_profile = (
        "baseline_c8w16k4s1" if target_spec.global_sink_frames > 1 else profile_name
    )
    pipeline, _ = _pipeline(_FakeGenerator(), spec=actual_profile)
    initial, _, conditioning = _inputs(prompt=1.0, batch=1)

    result = generate_stage2_single_action(
        pipeline,
        _FakeVAE(),
        initial_latent=initial,
        conditional_dict=conditioning,
        seeds=[sample.seed],
    )

    assert result.video.shape[1] == 96
    assert len(result.trace["episodes"]) == 1
    _validate_generation_trace(sample, result.trace, target_spec)


@pytest.mark.parametrize(
    ("profile_name", "episode_a_calls", "episode_b_calls"),
    [
        ("baseline_c8w16k4s1", 16, 15),
        ("stress_c8w24k4s1", 16, 15),
        ("lower_c8w8k4s1", 16, 15),
        ("c4w12k4s1", 31, 30),
        ("c4w8k4s1", 31, 30),
        ("c4w8k2s1", 19, 18),
        ("c8w16k4s4", 16, 15),
        ("c8w16k4s8", 16, 15),
    ],
)
def test_core_two_action_trace_satisfies_the_artifact_gate(
    profile_name,
    episode_a_calls,
    episode_b_calls,
):
    sample = next(
        item
        for item in build_stage2_inference_samples(
            single_metadata=SINGLE_METADATA,
            two_action_metadata=TWO_ACTION_METADATA,
            profiles=(profile_name,),
        )
        if item.dataset == STAGE2_TWO_ACTION_DATASET
    )
    target_spec = resolve_stage2_rollout_profile(profile_name)
    generator = _FakeGenerator()
    if target_spec.global_sink_frames > 1:
        episode_a, _ = _pipeline(generator)
        episode_b, _ = _pipeline(generator, spec=profile_name)
    else:
        episode_a, _ = _pipeline(generator, spec=profile_name)
        episode_b = episode_a
    initial, _, action_a = _inputs(prompt=1.0, batch=1)
    action_b = {"prompt_embeds": torch.full((1, 2, 3), 2.0, dtype=torch.bfloat16)}

    result = generate_stage2_two_action(
        episode_a,
        episode_b,
        _FakeVAE(),
        initial_latent=initial,
        action_a_conditional_dict=action_a,
        action_b_conditional_dict=action_b,
        seeds=[sample.seed],
    )

    assert result.video.shape[1] == 192
    assert [
        episode["cache_audit"]["generator_forward_calls"]
        for episode in result.trace["episodes"]
    ] == [episode_a_calls, episode_b_calls]
    _validate_generation_trace(sample, result.trace, target_spec)
