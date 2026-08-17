"""Strict, model-agnostic primitives for Stage-2 deployment inference.

The heavy Generator/T5/VAE bootstrap and the rollout kernel deliberately live
outside this module.  This file owns the reproducible per-sample noise plan and
the exact ``25 latent -> 97 pixel -> drop frame 0 -> 96 pixel`` decode contract
shared by single- and two-action inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import torch

STAGE2_INFERENCE_SEEDS = (1, 2, 3, 4)
STAGE2_EPISODE_FUTURE_LATENTS = 24
STAGE2_DECODE_INPUT_LATENTS = 25
STAGE2_DECODED_PIXEL_FRAMES_WITH_SINK = 97
STAGE2_OUTPUT_PIXEL_FRAMES_PER_EPISODE = 96
STAGE2_INFERENCE_FPS = 24
STAGE2_INFERENCE_TRACE_SCHEMA = "longlive_stage2_inference_trace/v1"
STAGE2_NOISE_STREAM_POLICY = "per_sample_seed_single_contiguous_stream"


def _plain_seed(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be a plain integer")
    if value < 0 or value >= 2**63:
        raise ValueError(f"{label} must be in [0, 2**63)")
    return value


def tensor_identity_sha256(value: torch.Tensor) -> str:
    """Hash dtype, shape, and raw bytes without changing the source tensor."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor_identity_sha256 expects a torch.Tensor")
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _per_sample_hashes(value: torch.Tensor) -> tuple[str, ...]:
    if value.ndim < 1:
        raise ValueError("per-sample hashing requires a batch dimension")
    return tuple(
        tensor_identity_sha256(value[index]) for index in range(value.shape[0])
    )


@dataclass(frozen=True)
class Stage2NoisePlan:
    """One deterministic contiguous random stream per batch sample.

    ``noise`` has shape ``[B, episodes*24, C, H, W]``.  For two actions the
    first 24 temporal slots are action A and the next 24 are action B.  A new
    RNG is created for each sample, so batching and rank assignment cannot
    change that sample's bytes.
    """

    noise: torch.Tensor
    seeds: tuple[int, ...]
    episodes: int
    full_sha256: tuple[str, ...]
    episode_sha256: tuple[tuple[str, ...], ...]

    def episode(self, index: int) -> torch.Tensor:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("episode index must be an integer")
        if index < 0 or index >= self.episodes:
            raise IndexError(index)
        start = index * STAGE2_EPISODE_FUTURE_LATENTS
        return self.noise[:, start : start + STAGE2_EPISODE_FUTURE_LATENTS]


def build_stage2_noise_plan(
    initial_latent: torch.Tensor,
    *,
    seeds: Sequence[int],
    episodes: int,
) -> Stage2NoisePlan:
    """Draw the user-approved per-sample contiguous A/B noise plan.

    The explicit seed is restarted once per *sample*, never once per episode.
    Consequently action A and B consume disjoint portions of one RNG stream,
    while rerunning the same sample/seed on another batch or rank reproduces
    exactly the same latent-noise bytes on the same PyTorch/device backend.
    """

    if not isinstance(initial_latent, torch.Tensor) or initial_latent.ndim != 5:
        raise ValueError("initial_latent must have shape [B,1,C,H,W]")
    if initial_latent.shape[1] != 1:
        raise ValueError("initial_latent must contain exactly one source sink")
    if not initial_latent.is_floating_point():
        raise TypeError("initial_latent must be floating point")
    if isinstance(episodes, bool) or not isinstance(episodes, int):
        raise TypeError("episodes must be an integer")
    if episodes not in {1, 2}:
        raise ValueError("Stage-2 inference supports one or two episodes")
    seed_tuple = tuple(
        _plain_seed(value, label=f"seeds[{index}]") for index, value in enumerate(seeds)
    )
    if len(seed_tuple) != int(initial_latent.shape[0]):
        raise ValueError(
            "seeds must contain exactly one value per initial_latent sample"
        )

    temporal_frames = episodes * STAGE2_EPISODE_FUTURE_LATENTS
    sample_shape = (temporal_frames, *initial_latent.shape[2:])
    samples = []
    for seed in seed_tuple:
        generator = torch.Generator(device=initial_latent.device)
        generator.manual_seed(seed)
        samples.append(
            torch.randn(
                sample_shape,
                generator=generator,
                device=initial_latent.device,
                dtype=initial_latent.dtype,
            )
        )
    noise = torch.stack(samples, dim=0)
    if not bool(torch.isfinite(noise).all().item()):
        raise RuntimeError("Stage-2 noise generation produced NaN or Inf")
    episode_hashes = tuple(
        _per_sample_hashes(
            noise[
                :,
                index
                * STAGE2_EPISODE_FUTURE_LATENTS : (index + 1)
                * STAGE2_EPISODE_FUTURE_LATENTS,
            ]
        )
        for index in range(episodes)
    )
    return Stage2NoisePlan(
        noise=noise,
        seeds=seed_tuple,
        episodes=episodes,
        full_sha256=_per_sample_hashes(noise),
        episode_sha256=episode_hashes,
    )


def clear_stage2_vae_cache(vae: Any) -> bool:
    """Clear a Wan VAE cache when exposed and report whether it was called."""

    model = getattr(vae, "model", None)
    clear_cache = getattr(model, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
        return True
    return False


@dataclass(frozen=True)
class Stage2DecodedEpisode:
    decode_input: torch.Tensor
    video: torch.Tensor
    decode_input_sha256: tuple[str, ...]
    video_sha256: tuple[str, ...]
    vae_cache_cleared: bool


def decode_stage2_episode(
    vae: Any,
    *,
    initial_latent: torch.Tensor,
    future_latents: torch.Tensor,
) -> Stage2DecodedEpisode:
    """Decode one action and remove the source-sink pixel frame exactly once."""

    if not isinstance(initial_latent, torch.Tensor) or initial_latent.ndim != 5:
        raise ValueError("initial_latent must have shape [B,1,C,H,W]")
    if initial_latent.shape[1] != 1:
        raise ValueError("initial_latent must contain exactly one source sink")
    if not isinstance(future_latents, torch.Tensor) or future_latents.ndim != 5:
        raise ValueError("future_latents must have shape [B,24,C,H,W]")
    expected_future = (
        initial_latent.shape[0],
        STAGE2_EPISODE_FUTURE_LATENTS,
        *initial_latent.shape[2:],
    )
    if tuple(future_latents.shape) != tuple(expected_future):
        raise ValueError(
            "future_latents shape mismatch: "
            f"expected={tuple(expected_future)}, actual={tuple(future_latents.shape)}"
        )
    if (
        initial_latent.dtype != future_latents.dtype
        or initial_latent.device != future_latents.device
    ):
        raise TypeError("initial and future latents must share dtype/device")
    if not bool(torch.isfinite(initial_latent).all().item()) or not bool(
        torch.isfinite(future_latents).all().item()
    ):
        raise ValueError("Stage-2 decode latents contain NaN or Inf")

    decode_input = torch.cat((initial_latent, future_latents), dim=1)
    if decode_input.shape[1] != STAGE2_DECODE_INPUT_LATENTS:
        raise AssertionError("Stage-2 VAE input must contain exactly 25 latents")
    cache_cleared = clear_stage2_vae_cache(vae)
    decoded = vae.decode_to_pixel(decode_input)
    if not isinstance(decoded, torch.Tensor) or decoded.ndim != 5:
        raise RuntimeError("Stage-2 VAE decode must return [B,T,C,H,W]")
    if decoded.shape[0] != initial_latent.shape[0]:
        raise RuntimeError("Stage-2 VAE decode changed the batch size")
    if decoded.shape[1] != STAGE2_DECODED_PIXEL_FRAMES_WITH_SINK:
        raise RuntimeError(
            "Stage-2 VAE decode frame count mismatch: "
            f"expected={STAGE2_DECODED_PIXEL_FRAMES_WITH_SINK}, "
            f"actual={decoded.shape[1]}"
        )
    if not bool(torch.isfinite(decoded).all().item()):
        raise RuntimeError("Stage-2 VAE decode returned NaN or Inf")
    video = (decoded[:, 1:].float() * 0.5 + 0.5).clamp(0.0, 1.0)
    if video.shape[1] != STAGE2_OUTPUT_PIXEL_FRAMES_PER_EPISODE:
        raise AssertionError("Stage-2 episode output must contain 96 pixel frames")
    return Stage2DecodedEpisode(
        decode_input=decode_input,
        video=video,
        decode_input_sha256=_per_sample_hashes(decode_input),
        video_sha256=_per_sample_hashes(video),
        vae_cache_cleared=cache_cleared,
    )


def _conditioning_hashes(
    conditional_dict: Mapping[str, Any],
    *,
    batch_size: int,
) -> tuple[str, ...]:
    embeds = conditional_dict.get("prompt_embeds")
    if not isinstance(embeds, torch.Tensor) or embeds.ndim != 3:
        raise ValueError("conditional_dict.prompt_embeds must have shape [B,L,D]")
    if embeds.shape[0] != batch_size:
        raise ValueError("prompt_embeds batch differs from the latent batch")
    if not bool(torch.isfinite(embeds).all().item()):
        raise ValueError("prompt_embeds contains NaN or Inf")
    return _per_sample_hashes(embeds)


def _rollout_trace_payload(value: Any) -> dict[str, Any]:
    """Select the stable, JSON-safe deployment fields from a rollout result."""

    required = {
        "exit_step": int(getattr(value, "exit_step")),
        "requires_grad": bool(getattr(value, "requires_grad")),
        "scheduler_timesteps": list(getattr(value, "scheduler_timesteps")),
        "scheduler_sigmas": list(getattr(value, "scheduler_sigmas")),
        "chunk_timesteps": [list(item) for item in getattr(value, "chunk_timesteps")],
        "chunk_sigmas": [list(item) for item in getattr(value, "chunk_sigmas")],
        "cache_audit": dict(getattr(value, "cache_audit")),
        "rollout_mode": str(getattr(value, "rollout_mode")),
    }
    if required["requires_grad"]:
        raise RuntimeError("Stage-2 deployment rollout unexpectedly retained gradients")
    chunk_trace = getattr(value, "chunk_trace", ())
    required["chunk_trace"] = [dict(item) for item in chunk_trace]
    for field_name in (
        "initial_latent_sha256_per_sample",
        "generated_latent_sha256_per_sample",
    ):
        field_value = getattr(value, field_name, ())
        required[field_name] = list(field_value)
    return required


@dataclass(frozen=True)
class Stage2InferenceResult:
    mode: str
    video: torch.Tensor
    episode_latents: tuple[torch.Tensor, ...]
    episode_videos: tuple[torch.Tensor, ...]
    trace: Mapping[str, Any]


def generate_stage2_single_action(
    rollout_pipeline: Any,
    vae: Any,
    *,
    initial_latent: torch.Tensor,
    conditional_dict: Mapping[str, Any],
    seeds: Sequence[int],
) -> Stage2InferenceResult:
    """Generate and decode one 24-latent/96-pixel Stage-2 episode."""

    noise_plan = build_stage2_noise_plan(
        initial_latent,
        seeds=seeds,
        episodes=1,
    )
    prompt_hashes = _conditioning_hashes(
        conditional_dict,
        batch_size=int(initial_latent.shape[0]),
    )
    rollout, _ = rollout_pipeline.generate_full_episode(
        initial_latent=initial_latent,
        noise=noise_plan.episode(0),
        conditional_dict=conditional_dict,
        state=None,
    )
    decoded = decode_stage2_episode(
        vae,
        initial_latent=initial_latent,
        future_latents=rollout.latents,
    )
    trace = {
        "schema": STAGE2_INFERENCE_TRACE_SCHEMA,
        "mode": "single_action",
        "seeds": list(noise_plan.seeds),
        "noise_stream": {
            "policy": STAGE2_NOISE_STREAM_POLICY,
            "rng_initializations_per_sample": 1,
            "episode_slots": STAGE2_EPISODE_FUTURE_LATENTS,
            "episode_order": ["single"],
        },
        "initial_latent_sha256": list(_per_sample_hashes(initial_latent)),
        "noise_plan_sha256": list(noise_plan.full_sha256),
        "noise_episode_sha256": [list(noise_plan.episode_sha256[0])],
        "prompt_embedding_sha256": [list(prompt_hashes)],
        "episodes": [_rollout_trace_payload(rollout)],
        "vae_events": [
            {
                "episode": "single",
                "cache_cleared": decoded.vae_cache_cleared,
                "decode_input_latents": STAGE2_DECODE_INPUT_LATENTS,
                "decoded_pixel_frames_with_sink": (
                    STAGE2_DECODED_PIXEL_FRAMES_WITH_SINK
                ),
                "dropped_pixel_frame_indices": [0],
                "output_pixel_frames": STAGE2_OUTPUT_PIXEL_FRAMES_PER_EPISODE,
            }
        ],
        "output_pixel_frames": int(decoded.video.shape[1]),
    }
    return Stage2InferenceResult(
        mode="single_action",
        video=decoded.video,
        episode_latents=(rollout.latents,),
        episode_videos=(decoded.video,),
        trace=trace,
    )


def generate_stage2_two_action(
    episode1_pipeline: Any,
    episode2_pipeline: Any,
    vae: Any,
    *,
    initial_latent: torch.Tensor,
    action_a_conditional_dict: Mapping[str, Any],
    action_b_conditional_dict: Mapping[str, Any],
    seeds: Sequence[int],
) -> Stage2InferenceResult:
    """Generate two cleanly-reset actions and concatenate A96 + B96.

    ``episode1_pipeline`` is always the baseline S1 path.  For an S4/S8
    inference-only profile ``episode2_pipeline`` owns the larger cache and
    restores the detached chunk-1 prefix captured by episode 1.
    """

    if int(getattr(episode1_pipeline, "global_sink_frames")) != 1:
        raise ValueError("Stage-2 episode 1 must remain the baseline S1 path")
    episode2_sink_frames = int(getattr(episode2_pipeline, "global_sink_frames"))
    if episode2_sink_frames not in {1, 4, 8}:
        raise ValueError("Stage-2 episode 2 supports only S1/S4/S8 profiles")
    noise_plan = build_stage2_noise_plan(
        initial_latent,
        seeds=seeds,
        episodes=2,
    )
    batch_size = int(initial_latent.shape[0])
    prompt_a_hashes = _conditioning_hashes(
        action_a_conditional_dict,
        batch_size=batch_size,
    )
    prompt_b_hashes = _conditioning_hashes(
        action_b_conditional_dict,
        batch_size=batch_size,
    )
    capture_frames = episode2_sink_frames if episode2_sink_frames > 1 else None
    rollout_a, state = episode1_pipeline.generate_full_episode(
        initial_latent=initial_latent,
        noise=noise_plan.episode(0),
        conditional_dict=action_a_conditional_dict,
        state=None,
        capture_prefix_sink_frames=capture_frames,
    )
    prefix_snapshot = getattr(rollout_a, "prefix_snapshot", None)
    if episode2_sink_frames > 1 and prefix_snapshot is None:
        raise RuntimeError("multi-sink episode 2 requires an episode-1 prefix snapshot")
    if episode2_sink_frames == 1 and prefix_snapshot is not None:
        raise RuntimeError(
            "baseline S1 episode unexpectedly captured a prefix snapshot"
        )
    state = episode2_pipeline.reset_for_new_episode(
        state,
        prefix_snapshot=prefix_snapshot,
    )
    rollout_b, _ = episode2_pipeline.generate_full_episode(
        initial_latent=initial_latent,
        noise=noise_plan.episode(1),
        conditional_dict=action_b_conditional_dict,
        state=state,
    )

    decoded_a = decode_stage2_episode(
        vae,
        initial_latent=initial_latent,
        future_latents=rollout_a.latents,
    )
    decoded_b = decode_stage2_episode(
        vae,
        initial_latent=initial_latent,
        future_latents=rollout_b.latents,
    )
    video = torch.cat((decoded_a.video, decoded_b.video), dim=1)
    expected_frames = 2 * STAGE2_OUTPUT_PIXEL_FRAMES_PER_EPISODE
    if video.shape[1] != expected_frames:
        raise AssertionError("Stage-2 two-action output must contain 192 frames")
    trace = {
        "schema": STAGE2_INFERENCE_TRACE_SCHEMA,
        "mode": "two_action",
        "seeds": list(noise_plan.seeds),
        "noise_stream": {
            "policy": STAGE2_NOISE_STREAM_POLICY,
            "rng_initializations_per_sample": 1,
            "episode_slots": STAGE2_EPISODE_FUTURE_LATENTS,
            "episode_order": ["A", "B"],
        },
        "initial_latent_sha256": list(_per_sample_hashes(initial_latent)),
        "noise_plan_sha256": list(noise_plan.full_sha256),
        "noise_episode_sha256": [list(item) for item in noise_plan.episode_sha256],
        "prompt_embedding_sha256": [
            list(prompt_a_hashes),
            list(prompt_b_hashes),
        ],
        "episodes": [
            _rollout_trace_payload(rollout_a),
            _rollout_trace_payload(rollout_b),
        ],
        "reset_events": [
            {
                "after_episode": "A",
                "retained_sink_frames": episode2_sink_frames,
                "cleared_non_sink_self_kv": True,
                "cleared_cross_kv": True,
                "next_future_start_frame": episode2_sink_frames,
                "prefix_snapshot_restored": episode2_sink_frames > 1,
            }
        ],
        "vae_events": [
            {
                "episode": label,
                "cache_cleared": decoded.vae_cache_cleared,
                "decode_input_latents": STAGE2_DECODE_INPUT_LATENTS,
                "decoded_pixel_frames_with_sink": (
                    STAGE2_DECODED_PIXEL_FRAMES_WITH_SINK
                ),
                "dropped_pixel_frame_indices": [0],
                "output_pixel_frames": STAGE2_OUTPUT_PIXEL_FRAMES_PER_EPISODE,
            }
            for label, decoded in (("A", decoded_a), ("B", decoded_b))
        ],
        "output_pixel_frames": int(video.shape[1]),
    }
    return Stage2InferenceResult(
        mode="two_action",
        video=video,
        episode_latents=(rollout_a.latents, rollout_b.latents),
        episode_videos=(decoded_a.video, decoded_b.video),
        trace=trace,
    )


__all__ = [
    "STAGE2_DECODED_PIXEL_FRAMES_WITH_SINK",
    "STAGE2_DECODE_INPUT_LATENTS",
    "STAGE2_EPISODE_FUTURE_LATENTS",
    "STAGE2_INFERENCE_FPS",
    "STAGE2_INFERENCE_SEEDS",
    "STAGE2_INFERENCE_TRACE_SCHEMA",
    "STAGE2_NOISE_STREAM_POLICY",
    "STAGE2_OUTPUT_PIXEL_FRAMES_PER_EPISODE",
    "Stage2DecodedEpisode",
    "Stage2InferenceResult",
    "Stage2NoisePlan",
    "build_stage2_noise_plan",
    "clear_stage2_vae_cache",
    "decode_stage2_episode",
    "generate_stage2_single_action",
    "generate_stage2_two_action",
    "tensor_identity_sha256",
]
