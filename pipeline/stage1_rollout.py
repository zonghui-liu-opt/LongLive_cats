"""Profiled self-rollout for Stage-1 teacher-forcing LoRA inference.

The first image latent is preloaded as a permanent global sink.  Every future
chunk starts from its own slice of the caller-supplied noise plan; noisy DiT
forwards can read persistent history but cannot mutate it.  Only the final
clean x0 recache becomes context for the next chunk.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from pipeline.causal_diffusion_continuation import tensor_identity_sha256
from utils.stage1_rollout_profile import (
    Stage1RolloutProfile,
    resolve_stage1_rollout_profile,
)

STAGE1_ROLLOUT_TRACE_SCHEMA = "longlive.stage1_teacher_forcing_rollout_trace/v1"


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _per_sample_hashes(value: torch.Tensor) -> list[str]:
    return [
        tensor_identity_sha256(value[index : index + 1])
        for index in range(value.shape[0])
    ]


def _fp32_bits(values: Sequence[float]) -> list[str]:
    return [struct.pack("!f", float(value)).hex() for value in values]


def _cache_index(cache: Mapping[str, Any], name: str) -> int:
    value = cache.get(name)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise RuntimeError(f"self-KV cache {name!r} must be a scalar tensor")
    return int(value.item())


@dataclass(frozen=True)
class Stage1RolloutResult:
    """One decoded rollout plus reproducibility and cache evidence."""

    future_latents: torch.Tensor
    latents: torch.Tensor
    video: torch.Tensor
    trace: dict[str, Any]


class Stage1RolloutPipeline:
    """Isolated one-shot rollout adapter around the causal Stage-1 pipeline."""

    def __init__(
        self,
        pipeline,
        profile: Stage1RolloutProfile | Mapping[str, Any] | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.profile = resolve_stage1_rollout_profile(profile)

    def _validate_runtime(self) -> None:
        pipeline = self.pipeline
        if bool(getattr(pipeline, "quantize_kv", False)):
            raise ValueError("Stage-1 rollout requires unquantized self-KV")
        for name in (
            "multi_shot_sink",
            "shot_clean_recache",
            "streaming_vae",
            "async_vae",
        ):
            if bool(getattr(pipeline, name, False)):
                raise ValueError(
                    f"Stage-1 rollout is incompatible with pipeline.{name}"
                )
        model = pipeline._dit_model
        if bool(getattr(model, "is_gradient_checkpointing", False)) or bool(
            getattr(model, "gradient_checkpointing", False)
        ):
            raise ValueError("Stage-1 rollout requires activation checkpointing off")
        if getattr(pipeline, "_active_continuation_session", None) is not None:
            raise RuntimeError("Stage-1 rollout is unavailable during continuation")

    def _validate_inputs(
        self,
        initial_latent: torch.Tensor,
        future_noise: torch.Tensor,
        prompts: Sequence[str],
    ) -> list[str]:
        if not torch.is_tensor(initial_latent) or initial_latent.ndim != 5:
            raise ValueError("initial_latent must have shape [B,1,C,H,W]")
        if initial_latent.shape[1] != self.profile.global_sink_size:
            raise ValueError("initial_latent must contain exactly the one-frame sink")
        if not torch.is_tensor(future_noise) or future_noise.ndim != 5:
            raise ValueError("future_noise must have shape [B,24,C,H,W]")
        expected_noise_shape = (
            initial_latent.shape[0],
            self.profile.generated_frames,
            *initial_latent.shape[2:],
        )
        if tuple(future_noise.shape) != tuple(expected_noise_shape):
            raise ValueError(
                "future_noise shape mismatch: "
                f"expected={expected_noise_shape}, actual={tuple(future_noise.shape)}"
            )
        if (
            future_noise.dtype != initial_latent.dtype
            or future_noise.device != initial_latent.device
        ):
            raise TypeError("initial_latent and future_noise dtype/device must match")
        if not bool(torch.isfinite(initial_latent).all().item()):
            raise ValueError("initial_latent contains non-finite values")
        if not bool(torch.isfinite(future_noise).all().item()):
            raise ValueError("future_noise contains non-finite values")
        if isinstance(prompts, (str, bytes)) or not isinstance(prompts, Sequence):
            raise TypeError("prompts must contain one string per batch item")
        normalized = list(prompts)
        if len(normalized) != initial_latent.shape[0] or any(
            not isinstance(prompt, str) or not prompt.strip() for prompt in normalized
        ):
            raise ValueError("prompts must contain one non-empty string per batch item")
        return normalized

    def _call_clean_commit(
        self,
        latent: torch.Tensor,
        conditioning: Mapping[str, Any],
        kv_cache,
        crossattn_cache,
        *,
        current_start_frame: int,
    ) -> None:
        timestep = torch.zeros(
            latent.shape[0],
            latent.shape[1],
            dtype=torch.float32,
            device=latent.device,
        )
        value = self.pipeline.generator(
            noisy_image_or_video=latent,
            conditional_dict=dict(conditioning),
            timestep=timestep,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start_frame * self.pipeline.frame_seq_length,
            cache_start=current_start_frame * self.pipeline.frame_seq_length,
            commit_self_kv=True,
        )
        if (
            not isinstance(value, tuple)
            or len(value) < 1
            or not torch.is_tensor(value[0])
        ):
            raise TypeError(
                "generator must return a raw-flow tensor during clean recache"
            )

    def _audit_cache_branch(
        self,
        cache_branch,
        *,
        expected_global_end: int,
        expected_local_end: int,
        branch: str,
    ) -> dict[str, Any]:
        pipeline = self.pipeline
        expected_layers = int(pipeline.num_transformer_blocks)
        if len(cache_branch) != expected_layers or expected_layers <= 0:
            raise RuntimeError(f"{branch} self-KV layer count mismatch")
        expected_capacity = self.profile.window_size * pipeline.frame_seq_length
        for layer_index, cache in enumerate(cache_branch):
            key = cache.get("k")
            value = cache.get("v")
            if (
                not torch.is_tensor(key)
                or not torch.is_tensor(value)
                or key.shape != value.shape
                or key.ndim != 4
                or int(key.shape[1]) != expected_capacity
            ):
                raise RuntimeError(
                    f"{branch} self-KV layer {layer_index} has invalid capacity"
                )
            if (
                key.requires_grad
                or value.requires_grad
                or key.grad_fn is not None
                or value.grad_fn is not None
            ):
                raise RuntimeError(
                    f"{branch} self-KV layer {layer_index} retained autograd state"
                )
            if _cache_index(cache, "global_end_index") != expected_global_end:
                raise RuntimeError(
                    f"{branch} self-KV layer {layer_index} global cursor drifted"
                )
            if _cache_index(cache, "local_end_index") != expected_local_end:
                raise RuntimeError(
                    f"{branch} self-KV layer {layer_index} local cursor drifted"
                )
            if (
                _cache_index(cache, "pinned_start") != -1
                or _cache_index(cache, "pinned_len") != 0
            ):
                raise RuntimeError(
                    f"{branch} self-KV layer {layer_index} enabled a pinned sink"
                )
        return {
            "branch": branch,
            "layers": expected_layers,
            "capacity_frames": self.profile.window_size,
            "capacity_tokens": expected_capacity,
            "global_end_index": expected_global_end,
            "local_end_index": expected_local_end,
            "persistent_kv_detached": True,
        }

    def _audit_cache_pair(
        self,
        kv_pos,
        kv_neg,
        *,
        use_cfg: bool,
        expected_global_end: int,
        expected_local_end: int,
    ) -> dict[str, Any]:
        positive = self._audit_cache_branch(
            kv_pos,
            expected_global_end=expected_global_end,
            expected_local_end=expected_local_end,
            branch="positive",
        )
        negative = None
        if use_cfg:
            if kv_neg is kv_pos:
                raise RuntimeError("CFG positive/negative self-KV cannot be shared")
            negative = self._audit_cache_branch(
                kv_neg,
                expected_global_end=expected_global_end,
                expected_local_end=expected_local_end,
                branch="negative",
            )
            if (
                negative["global_end_index"] != positive["global_end_index"]
                or negative["local_end_index"] != positive["local_end_index"]
            ):
                raise RuntimeError("CFG self-KV cursors are not synchronized")
        return {
            "positive": positive,
            "negative": negative,
            "conditional_cache_branches": 2 if use_cfg else 1,
        }

    def run(
        self,
        *,
        initial_latent: torch.Tensor,
        future_noise: torch.Tensor,
        prompts: Sequence[str],
        checkpoint_provenance: Mapping[str, Any] | None = None,
    ) -> Stage1RolloutResult:
        """Generate 24 future latents and decode sink+future exactly once."""

        prompts = self._validate_inputs(initial_latent, future_noise, prompts)
        if checkpoint_provenance is not None and not isinstance(
            checkpoint_provenance, Mapping
        ):
            raise TypeError("checkpoint_provenance must be a mapping or None")

        pipeline = self.pipeline
        profile = self.profile
        use_cfg = profile.guidance_scale != 1.0
        field_names = (
            "num_frame_per_block",
            "local_attn_size",
            "sampling_steps",
            "sample_solver",
            "shift",
            "guidance_scale",
            "negative_prompt",
            "global_sink_size",
            "sink_size",
            "timesteps",
        )

        with pipeline._get_continuation_lock():
            self._validate_runtime()
            # Runtime profiles mutate shared pipeline/model fields.  Snapshot
            # them only after acquiring the same lock that guards the whole
            # rollout, otherwise a concurrent caller can capture this call's
            # temporary profile and restore it after we finish.
            missing = object()
            original_fields = {
                name: getattr(pipeline, name, missing) for name in field_names
            }
            dit = pipeline._dit_model
            original_dit_block = getattr(dit, "num_frame_per_block", missing)
            try:
                pipeline.num_frame_per_block = profile.chunk_size
                pipeline.local_attn_size = profile.window_size
                pipeline.sampling_steps = profile.sampling_steps
                pipeline.sample_solver = profile.solver
                pipeline.shift = profile.timestep_shift
                pipeline.guidance_scale = profile.guidance_scale
                pipeline.negative_prompt = profile.negative_prompt
                pipeline.global_sink_size = profile.global_sink_size
                pipeline.sink_size = profile.global_sink_size
                dit.num_frame_per_block = profile.chunk_size

                conditional_dict = pipeline.text_encoder(text_prompts=prompts)
                if not isinstance(conditional_dict, Mapping):
                    raise TypeError("text encoder must return a conditioning mapping")
                conditional_dict = dict(conditional_dict)
                unconditional_dict = None
                if use_cfg:
                    unconditional_dict = dict(
                        pipeline.text_encoder(
                            text_prompts=[profile.negative_prompt] * len(prompts)
                        )
                    )

                kv_pos, kv_neg = pipeline._build_kv_cache(
                    batch_size=initial_latent.shape[0],
                    dtype=initial_latent.dtype,
                    device=initial_latent.device,
                )
                cross_pos, cross_neg = pipeline._build_crossattn_cache(
                    batch_size=initial_latent.shape[0],
                    dtype=initial_latent.dtype,
                    device=initial_latent.device,
                )

                with pipeline._inference_runtime_overrides(
                    sink_size=profile.global_sink_size,
                    global_sink_size=profile.global_sink_size,
                ):
                    self._call_clean_commit(
                        initial_latent,
                        conditional_dict,
                        kv_pos,
                        cross_pos,
                        current_start_frame=0,
                    )
                    if use_cfg:
                        self._call_clean_commit(
                            initial_latent,
                            unconditional_dict,
                            kv_neg,
                            cross_neg,
                            current_start_frame=0,
                        )
                    sink_tokens = profile.global_sink_size * pipeline.frame_seq_length
                    sink_audit = self._audit_cache_pair(
                        kv_pos,
                        kv_neg,
                        use_cfg=use_cfg,
                        expected_global_end=sink_tokens,
                        expected_local_end=sink_tokens,
                    )

                    future_chunks: list[torch.Tensor] = []
                    chunk_traces: list[dict[str, Any]] = []
                    reference_timetable = None
                    reference_sigmas = None
                    for chunk_index in range(profile.num_chunks):
                        start = chunk_index * profile.chunk_size
                        stop = start + profile.chunk_size
                        current_start = profile.global_sink_size + start
                        before_global = _cache_index(kv_pos[0], "global_end_index")
                        before_local = _cache_index(kv_pos[0], "local_end_index")
                        execution_trace: dict[str, Any] = {}
                        clean = pipeline._denoise_and_recache_block(
                            noise_block=future_noise[:, start:stop],
                            conditional_dict=conditional_dict,
                            unconditional_dict=unconditional_dict,
                            use_cfg=use_cfg,
                            global_start_frame=current_start,
                            cache_start_frame=current_start,
                            kv_cache_pos=kv_pos,
                            kv_cache_neg=kv_neg,
                            crossattn_cache_pos=cross_pos,
                            crossattn_cache_neg=cross_neg,
                            noisy_commit_self_kv=False,
                            clean_commit_self_kv=True,
                            terminal_direct_x0=True,
                            execution_trace=execution_trace,
                        ).detach()
                        timetable = tuple(execution_trace["timesteps"])
                        sigmas = tuple(execution_trace["sigmas"])
                        if reference_timetable is None:
                            reference_timetable = timetable
                            reference_sigmas = sigmas
                        elif (
                            timetable != reference_timetable
                            or sigmas != reference_sigmas
                        ):
                            raise RuntimeError(
                                "UniPC schedule changed between rollout chunks"
                            )

                        expected_global = (
                            profile.global_sink_size + stop
                        ) * pipeline.frame_seq_length
                        expected_local = (
                            min(
                                profile.window_size,
                                profile.global_sink_size + stop,
                            )
                            * pipeline.frame_seq_length
                        )
                        cache_audit = self._audit_cache_pair(
                            kv_pos,
                            kv_neg,
                            use_cfg=use_cfg,
                            expected_global_end=expected_global,
                            expected_local_end=expected_local,
                        )
                        future_chunks.append(clean)
                        chunk_traces.append(
                            {
                                "chunk_index": chunk_index,
                                "future_frame_start": start,
                                "future_frame_end_exclusive": stop,
                                "rope_frame_start": current_start,
                                "rope_frame_end_exclusive": current_start
                                + profile.chunk_size,
                                "noise_sha256_per_sample": _per_sample_hashes(
                                    future_noise[:, start:stop]
                                ),
                                "clean_sha256_per_sample": _per_sample_hashes(clean),
                                "cache_before_global_end_index": before_global,
                                "cache_before_local_end_index": before_local,
                                "cache_after": cache_audit,
                                "timesteps": list(timetable),
                                "sigmas": list(sigmas),
                                "sigma_fp32_bits": _fp32_bits(sigmas),
                                "fresh_scheduler": True,
                                "denoising_forward_calls_per_branch": profile.sampling_steps,
                                "solver_update_calls": execution_trace[
                                    "solver_update_calls"
                                ],
                                "noisy_self_kv_commits": 0,
                                "clean_self_kv_commits_per_branch": 1,
                                "terminal_direct_x0": True,
                            }
                        )

                future_latents = torch.cat(future_chunks, dim=1)
                latents = torch.cat([initial_latent, future_latents], dim=1)
                video = pipeline.vae.decode_to_pixel(latents)
                video = (video * 0.5 + 0.5).clamp(0, 1)
                expected_pixel_frames = 1 + 4 * profile.generated_frames
                if (
                    video.ndim != 5
                    or video.shape[0] != initial_latent.shape[0]
                    or video.shape[1] != expected_pixel_frames
                ):
                    raise RuntimeError(
                        "Stage-1 rollout VAE output must be [B,97,C,H,W] for "
                        "one sink plus 24 future latents; got "
                        f"{tuple(video.shape)}"
                    )
                prompt_embeds = conditional_dict.get("prompt_embeds")
                negative_embeds = (
                    unconditional_dict.get("prompt_embeds") if use_cfg else None
                )
                provenance = dict(checkpoint_provenance or {})
                checkpoint_evidence = provenance.get("checkpoint")
                teacher_forcing_verified = bool(
                    provenance.get("runtime_mode") == "dynamic_stage1_lora"
                    and isinstance(checkpoint_evidence, Mapping)
                    and checkpoint_evidence.get("teacher_forcing") is True
                    and checkpoint_evidence.get("training_block_size") == 8
                )
                trace = {
                    "schema": STAGE1_ROLLOUT_TRACE_SCHEMA,
                    "profile": profile.to_canonical_dict(),
                    "profile_sha256": profile.canonical_sha256,
                    "training_context": {
                        "teacher_forcing_weights_verified": (teacher_forcing_verified),
                        "teacher_forcing_evidence": (
                            "checkpoint_preflight"
                            if teacher_forcing_verified
                            else "unverified"
                        ),
                        "rollout_semantics": "self_rollout",
                        "self_rollout_is_deployment_ood": (
                            True if teacher_forcing_verified else None
                        ),
                        "training_block_size": (
                            8 if teacher_forcing_verified else None
                        ),
                        "chunk_size_matches_training_block": (
                            profile.chunk_size == 8
                            if teacher_forcing_verified
                            else None
                        ),
                        "default_profile_topology": profile.default_topology,
                        "window_topology_was_teacher_forced": (
                            False if teacher_forcing_verified else None
                        ),
                    },
                    "cache_policy": {
                        "global_sink": "input_first_latent_t0_clean_commit",
                        "noisy_current_chunk_commit_self_kv": False,
                        "clean_x0_recache_commit_self_kv": True,
                        "history_source": "previous_model_generated_clean_x0",
                        "pinned_sink_enabled": False,
                    },
                    "cfg": {
                        "enabled": use_cfg,
                        "guidance_scale": profile.guidance_scale,
                        "formula": "uncond + guidance_scale * (cond - uncond)",
                        "conditional_cache_branches": 2 if use_cfg else 1,
                        "negative_prompt": profile.negative_prompt,
                        "negative_prompt_sha256": _text_sha256(profile.negative_prompt),
                    },
                    "prompts": list(prompts),
                    "prompt_sha256": [_text_sha256(prompt) for prompt in prompts],
                    "conditioning_sha256": (
                        tensor_identity_sha256(prompt_embeds)
                        if torch.is_tensor(prompt_embeds)
                        else None
                    ),
                    "negative_conditioning_sha256": (
                        tensor_identity_sha256(negative_embeds)
                        if torch.is_tensor(negative_embeds)
                        else None
                    ),
                    "input": {
                        "initial_latent_shape": list(initial_latent.shape),
                        "initial_latent_sha256_per_sample": _per_sample_hashes(
                            initial_latent
                        ),
                        "future_noise_shape": list(future_noise.shape),
                        "future_noise_sha256_per_sample": _per_sample_hashes(
                            future_noise
                        ),
                    },
                    "output": {
                        "future_latent_shape": list(future_latents.shape),
                        "future_latent_sha256_per_sample": _per_sample_hashes(
                            future_latents
                        ),
                        "full_latent_shape": list(latents.shape),
                        "full_latent_sha256_per_sample": _per_sample_hashes(latents),
                        "video_shape": list(video.shape),
                        "expected_pixel_frames": expected_pixel_frames,
                    },
                    "sink_cache_after_preload": sink_audit,
                    "chunks": chunk_traces,
                    "generator_call_budget": {
                        "per_branch": 1
                        + profile.num_chunks * (profile.sampling_steps + 1),
                        "all_branches": (2 if use_cfg else 1)
                        * (1 + profile.num_chunks * (profile.sampling_steps + 1)),
                    },
                    "checkpoint_provenance": provenance,
                }
                return Stage1RolloutResult(
                    future_latents=future_latents,
                    latents=latents,
                    video=video,
                    trace=trace,
                )
            finally:
                for name, value in original_fields.items():
                    if value is missing:
                        if hasattr(pipeline, name):
                            delattr(pipeline, name)
                    else:
                        setattr(pipeline, name, value)
                if original_dit_block is missing:
                    if hasattr(dit, "num_frame_per_block"):
                        delattr(dit, "num_frame_per_block")
                else:
                    dit.num_frame_per_block = original_dit_block


def run_stage1_rollout(
    pipeline,
    *,
    initial_latent: torch.Tensor,
    future_noise: torch.Tensor,
    prompts: Sequence[str],
    profile: Stage1RolloutProfile | Mapping[str, Any] | None = None,
    checkpoint_provenance: Mapping[str, Any] | None = None,
) -> Stage1RolloutResult:
    """Convenience entry point for one isolated Stage-1 rollout."""

    return Stage1RolloutPipeline(pipeline, profile).run(
        initial_latent=initial_latent,
        future_noise=future_noise,
        prompts=prompts,
        checkpoint_provenance=checkpoint_provenance,
    )


__all__ = [
    "STAGE1_ROLLOUT_TRACE_SCHEMA",
    "Stage1RolloutPipeline",
    "Stage1RolloutResult",
    "run_stage1_rollout",
]
