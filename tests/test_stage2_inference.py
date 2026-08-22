from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from utils import stage2_inference
from utils.stage2_inference import (
    STAGE2_DECODE_INPUT_LATENTS,
    STAGE2_OUTPUT_PIXEL_FRAMES_PER_EPISODE,
    build_stage2_noise_plan,
    decode_stage2_episode,
    generate_stage2_single_action,
    generate_stage2_two_action,
)


def _initial(batch: int = 1) -> torch.Tensor:
    return torch.zeros(batch, 1, 48, 2, 3, dtype=torch.bfloat16)


class _FakeVAEModel:
    def __init__(self) -> None:
        self.clear_calls = 0

    def clear_cache(self) -> None:
        self.clear_calls += 1


class _FakeVAE:
    def __init__(self) -> None:
        self.model = _FakeVAEModel()
        self.inputs: list[torch.Tensor] = []

    def decode_to_pixel(self, value: torch.Tensor) -> torch.Tensor:
        self.inputs.append(value.clone())
        batch = value.shape[0]
        # Sentinel frame 0 proves that decode removes the source pixel frame,
        # rather than dropping a generated frame at the end.
        output = torch.zeros(batch, 97, 3, 4, 5)
        output[:, 0].fill_(-1.0)
        for frame in range(1, 97):
            output[:, frame].fill_(-1.0 + frame / 48.0)
        return output


class _FakeRolloutPipeline:
    def __init__(self, sink_frames: int = 1) -> None:
        self.global_sink_frames = sink_frames
        self.calls = []
        self.reset_calls = []

    def generate_full_episode(
        self,
        *,
        initial_latent,
        noise,
        conditional_dict,
        state=None,
        capture_prefix_sink_frames=None,
    ):
        self.calls.append(
            {
                "noise": noise.clone(),
                "conditional_dict": conditional_dict,
                "state": state,
                "capture_prefix_sink_frames": capture_prefix_sink_frames,
            }
        )
        offset = float(len(self.calls))
        latents = noise + offset
        snapshot = (
            {"sink_frames": capture_prefix_sink_frames}
            if capture_prefix_sink_frames is not None
            else None
        )
        result = SimpleNamespace(
            latents=latents,
            exit_step=3,
            requires_grad=False,
            scheduler_timesteps=(999, 937, 833, 624),
            scheduler_sigmas=(1.0, 0.75, 0.5, 0.25, 0.0),
            chunk_timesteps=((999, 937, 833, 624),) * 3,
            chunk_sigmas=((1.0, 0.75, 0.5, 0.25),) * 3,
            cache_audit={"capacity_frames": self.global_sink_frames + 16},
            rollout_mode="full_denoising",
            chunk_trace=({"chunk_index": 0},),
            initial_latent_sha256_per_sample=("initial",),
            generated_latent_sha256_per_sample=("generated",),
            prefix_snapshot=snapshot,
        )
        return result, {"complete": True, "sink_frames": self.global_sink_frames}

    def reset_for_new_episode(self, state, *, prefix_snapshot=None):
        self.reset_calls.append((state, prefix_snapshot))
        return {
            "complete": False,
            "sink_frames": self.global_sink_frames,
            "prefix_snapshot": prefix_snapshot,
        }


def test_two_action_noise_is_one_contiguous_draw_with_disjoint_a_b_slices():
    initial = _initial()
    plan = build_stage2_noise_plan(initial, seeds=[7], episodes=2)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(7)
    expected = torch.randn(
        (1, 48, 48, 2, 3),
        generator=generator,
        dtype=torch.bfloat16,
    )
    assert torch.equal(plan.noise, expected)
    assert torch.equal(plan.episode(0), expected[:, :24])
    assert torch.equal(plan.episode(1), expected[:, 24:])
    assert not torch.equal(plan.episode(0), plan.episode(1))
    assert plan.episode_sha256[0] != plan.episode_sha256[1]


def test_noise_mapping_is_invariant_to_batching_and_rank_assignment():
    together = build_stage2_noise_plan(_initial(3), seeds=[4, 1, 9], episodes=2)
    reordered = build_stage2_noise_plan(_initial(3), seeds=[9, 4, 1], episodes=2)
    alone = build_stage2_noise_plan(_initial(), seeds=[1], episodes=2)

    assert torch.equal(together.noise[0], reordered.noise[1])
    assert torch.equal(together.noise[1], reordered.noise[2])
    assert torch.equal(together.noise[2], reordered.noise[0])
    assert torch.equal(together.noise[1], alone.noise[0])
    assert together.full_sha256[1] == alone.full_sha256[0]


def test_b_is_not_generated_by_resetting_the_same_seed():
    plan = build_stage2_noise_plan(_initial(), seeds=[123], episodes=2)
    reset = build_stage2_noise_plan(_initial(), seeds=[123], episodes=1)

    assert torch.equal(plan.episode(0), reset.episode(0))
    assert not torch.equal(plan.episode(1), reset.episode(0))


@pytest.mark.parametrize(
    ("seeds", "episodes", "message"),
    [
        ([], 2, "one value per"),
        ([True], 2, "plain integer"),
        ([1], 3, "one or two"),
    ],
)
def test_noise_plan_rejects_ambiguous_inputs(seeds, episodes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        build_stage2_noise_plan(_initial(), seeds=seeds, episodes=episodes)


def test_decode_uses_sink_plus_24_then_drops_only_pixel_frame_zero():
    vae = _FakeVAE()
    initial = torch.full_like(_initial(), 11.0)
    future = torch.full((1, 24, 48, 2, 3), 22.0, dtype=torch.bfloat16)

    result = decode_stage2_episode(
        vae,
        initial_latent=initial,
        future_latents=future,
    )

    assert vae.model.clear_calls == 1
    assert len(vae.inputs) == 1
    assert vae.inputs[0].shape[1] == STAGE2_DECODE_INPUT_LATENTS
    assert torch.equal(vae.inputs[0][:, :1], initial)
    assert torch.equal(vae.inputs[0][:, 1:], future)
    assert result.video.shape == (
        1,
        STAGE2_OUTPUT_PIXEL_FRAMES_PER_EPISODE,
        3,
        4,
        5,
    )
    assert torch.allclose(result.video[:, 0], torch.full((1, 3, 4, 5), 1.0 / 96.0))
    assert bool((result.video >= 0).all())
    assert bool((result.video <= 1).all())


def test_decode_does_not_hash_full_decode_tensors(monkeypatch):
    hashed_shapes = []

    def _record_hash(value):
        hashed_shapes.append(tuple(value.shape))
        return "unused"

    monkeypatch.setattr(stage2_inference, "tensor_identity_sha256", _record_hash)
    result = decode_stage2_episode(
        _FakeVAE(),
        initial_latent=_initial(),
        future_latents=torch.zeros(1, 24, 48, 2, 3, dtype=torch.bfloat16),
    )

    assert hashed_shapes == []
    assert not hasattr(result, "decode_input_sha256")
    assert not hasattr(result, "video_sha256")


def test_decode_fails_closed_on_wrong_future_or_pixel_frame_count():
    vae = _FakeVAE()
    with pytest.raises(ValueError, match="future_latents shape mismatch"):
        decode_stage2_episode(
            vae,
            initial_latent=_initial(),
            future_latents=torch.zeros(1, 23, 48, 2, 3, dtype=torch.bfloat16),
        )

    class _ShortVAE(_FakeVAE):
        def decode_to_pixel(self, value: torch.Tensor) -> torch.Tensor:
            return torch.zeros(value.shape[0], 96, 3, 4, 5)

    with pytest.raises(RuntimeError, match="frame count mismatch"):
        decode_stage2_episode(
            _ShortVAE(),
            initial_latent=_initial(),
            future_latents=torch.zeros(1, 24, 48, 2, 3, dtype=torch.bfloat16),
        )


def _conditioning(value: float) -> dict[str, torch.Tensor]:
    return {"prompt_embeds": torch.full((1, 4, 8), value, dtype=torch.bfloat16)}


def test_single_action_orchestration_returns_exactly_96_frames_and_trace():
    pipeline = _FakeRolloutPipeline()
    vae = _FakeVAE()
    result = generate_stage2_single_action(
        pipeline,
        vae,
        initial_latent=_initial(),
        conditional_dict=_conditioning(1.0),
        seeds=[1],
    )

    assert result.mode == "single_action"
    assert result.video.shape[1] == 96
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["capture_prefix_sink_frames"] is None
    assert result.trace["output_pixel_frames"] == 96
    assert result.trace["noise_stream"]["episode_order"] == ["single"]
    assert result.trace["episodes"][0]["requires_grad"] is False
    assert result.trace["vae_events"][0]["dropped_pixel_frame_indices"] == [0]


def test_two_action_orchestration_uses_disjoint_noise_resets_kv_and_decodes_separately():
    pipeline = _FakeRolloutPipeline()
    vae = _FakeVAE()
    result = generate_stage2_two_action(
        pipeline,
        pipeline,
        vae,
        initial_latent=_initial(),
        action_a_conditional_dict=_conditioning(1.0),
        action_b_conditional_dict=_conditioning(2.0),
        seeds=[5],
    )

    expected = build_stage2_noise_plan(_initial(), seeds=[5], episodes=2)
    assert len(pipeline.calls) == 2
    assert torch.equal(pipeline.calls[0]["noise"], expected.episode(0))
    assert torch.equal(pipeline.calls[1]["noise"], expected.episode(1))
    assert not torch.equal(pipeline.calls[0]["noise"], pipeline.calls[1]["noise"])
    assert len(pipeline.reset_calls) == 1
    assert pipeline.reset_calls[0][1] is None
    assert vae.model.clear_calls == 2
    assert len(vae.inputs) == 2
    assert torch.equal(vae.inputs[0][:, :1], _initial())
    assert torch.equal(vae.inputs[1][:, :1], _initial())
    assert torch.equal(vae.inputs[0][:, 1:], result.episode_latents[0])
    assert torch.equal(vae.inputs[1][:, 1:], result.episode_latents[1])
    assert result.video.shape[1] == 192
    assert result.trace["noise_stream"] == {
        "policy": "per_sample_seed_single_contiguous_stream",
        "rng_initializations_per_sample": 1,
        "episode_slots": 24,
        "episode_order": ["A", "B"],
    }
    assert torch.equal(result.video[:, :96], result.episode_videos[0])
    assert torch.equal(result.video[:, 96:], result.episode_videos[1])
    assert result.trace["reset_events"] == [
        {
            "after_episode": "A",
            "retained_sink_frames": 1,
            "cleared_non_sink_self_kv": True,
            "cleared_cross_kv": True,
            "next_future_start_frame": 1,
            "prefix_snapshot_restored": False,
        }
    ]


@pytest.mark.parametrize("sink_frames", [4, 8])
def test_multi_sink_two_action_captures_and_restores_episode1_prefix(sink_frames):
    episode1 = _FakeRolloutPipeline(sink_frames=1)
    episode2 = _FakeRolloutPipeline(sink_frames=sink_frames)
    result = generate_stage2_two_action(
        episode1,
        episode2,
        _FakeVAE(),
        initial_latent=_initial(),
        action_a_conditional_dict=_conditioning(1.0),
        action_b_conditional_dict=_conditioning(2.0),
        seeds=[3],
    )

    assert episode1.calls[0]["capture_prefix_sink_frames"] == sink_frames
    assert episode2.reset_calls[0][1] == {"sink_frames": sink_frames}
    assert episode2.calls[0]["state"]["prefix_snapshot"] == {"sink_frames": sink_frames}
    assert result.trace["reset_events"][0]["next_future_start_frame"] == sink_frames
    assert result.trace["reset_events"][0]["prefix_snapshot_restored"] is True
