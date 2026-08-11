"""Stage-2 model roles, Wan adapters, and strict DMD/DFD loss math.

This module deliberately contains no rollout, optimizer, EMA, data loader, T5,
or VAE construction.  It owns the three independent DiT roles, their
shape-checked raw-flow/x0 calls, and the FP32 loss reductions used by the later
trainer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch

from torch import nn

from utils.stage2_score_math import (
    STAGE2_SCORE_DENOMINATOR_CLAMP,
    STAGE2_SCORE_FUTURE_FRAMES,
    STAGE2_SCORE_INPUT_FRAMES,
    Stage2NoisedScoreInput,
    Stage2NoisedScorePair,
    stage2_flow_to_x0,
    stage2_real_cfg_flow,
    validate_stage2_noised_score_input,
    validate_stage2_noised_score_pair,
)


class Stage2DiTRole(nn.Module):
    """A thin role wrapper around one independently-owned Wan DiT."""

    _ROLES = {"generator", "real_score", "fake_score"}

    def __init__(self, model: nn.Module, *, role: str, is_causal: bool):
        super().__init__()
        if role not in self._ROLES:
            raise ValueError(f"unknown Stage-2 role: {role!r}")
        if role == "generator" and not is_causal:
            raise ValueError("Stage-2 generator must be causal")
        if role != "generator" and is_causal:
            raise ValueError(f"Stage-2 {role} must be bidirectional")
        if not isinstance(model, nn.Module):
            raise TypeError("Stage2DiTRole.model must be torch.nn.Module")
        self.model = model
        self.role = role
        self.is_causal = bool(is_causal)
        self.uniform_timestep = not self.is_causal

    @staticmethod
    def _continuous_flow_to_x0_fp32(
        noisy_image_or_video: torch.Tensor,
        raw_flow: torch.Tensor,
        frame_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Use the Stage-2 continuous ``sigma=t/1000`` flow identity."""

        if raw_flow.shape != noisy_image_or_video.shape:
            raise ValueError("raw_flow and noisy input shapes differ")
        if tuple(frame_timestep.shape) != tuple(noisy_image_or_video.shape[:2]):
            raise ValueError("frame_timestep must have shape [B,F]")
        frame_timestep_fp32 = frame_timestep.to(device=raw_flow.device).float()
        if not bool(torch.isfinite(frame_timestep_fp32).all().item()):
            raise ValueError("frame_timestep contains non-finite values")
        if bool((frame_timestep_fp32 < 0).any().item()) or bool(
            (frame_timestep_fp32 > 1000).any().item()
        ):
            raise ValueError("frame_timestep must be in [0,1000]")
        sigma = (
            frame_timestep_fp32.view(
                frame_timestep_fp32.shape[0],
                frame_timestep_fp32.shape[1],
                1,
                1,
                1,
            )
            / 1000.0
        )
        return noisy_image_or_video.float() - sigma * raw_flow.float()

    @staticmethod
    def _generator_flow_to_x0_fp32(
        noisy_image_or_video: torch.Tensor,
        raw_flow: torch.Tensor,
        flow_sigma: torch.Tensor | float,
    ) -> torch.Tensor:
        """Use the caller-supplied UniPC sigma for a causal Generator call."""

        if raw_flow.shape != noisy_image_or_video.shape:
            raise ValueError("raw_flow and noisy input shapes differ")
        if isinstance(flow_sigma, bool) or not isinstance(
            flow_sigma, (torch.Tensor, float, int)
        ):
            raise TypeError("Generator flow_sigma must be a tensor or number")
        sigma = torch.as_tensor(
            flow_sigma,
            dtype=torch.float32,
            device=noisy_image_or_video.device,
        )
        if not bool(torch.isfinite(sigma).all().item()):
            raise ValueError("Generator flow_sigma contains non-finite values")
        if bool((sigma < 0).any().item()) or bool((sigma > 1).any().item()):
            raise ValueError("Generator flow_sigma must be in [0,1]")
        if sigma.ndim == 0:
            sigma = sigma.view(1, 1, 1, 1, 1)
        elif tuple(sigma.shape) == tuple(noisy_image_or_video.shape[:2]):
            sigma = sigma.view(
                int(sigma.shape[0]),
                int(sigma.shape[1]),
                1,
                1,
                1,
            )
        else:
            raise ValueError("Generator flow_sigma must be scalar or have shape [B,F]")
        return noisy_image_or_video.float() - sigma * raw_flow.float()

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: Mapping[str, torch.Tensor],
        timestep: torch.Tensor,
        flow_sigma: torch.Tensor | float | None = None,
        kv_cache: list[dict] | None = None,
        crossattn_cache: list[dict] | None = None,
        current_start: int | None = None,
        cache_start: int | None = None,
        commit_self_kv: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(raw_flow, x0_pred)`` for this independently owned DiT.

        Score x0 stays FP32 for later score math.  Generator x0 is safely cast
        back to the latent forward dtype so UniPC/clean recache can feed the
        BF16 causal DiT without a Conv3d dtype mismatch.
        """

        # Keep heavyweight Wan/T5/VAE imports out of init-only module import.
        from utils.wan_forward_adapter import (
            forward_stage2_causal_model,
            forward_stage2_i2v_score_model,
            wan_patch_embedding_dtype,
        )

        patch_size = tuple(int(value) for value in self.model.patch_size)
        if self.role == "generator":
            if flow_sigma is None:
                raise ValueError(
                    "Stage-2 causal Generator requires explicit UniPC flow_sigma"
                )
            raw_flow = forward_stage2_causal_model(
                self.model,
                cache_update_owner=self.model,
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=conditional_dict,
                frame_timestep=timestep,
                patch_size=patch_size,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                commit_self_kv=commit_self_kv,
                maximum_text_length=getattr(self.model, "text_len", None),
                expected_text_dim=getattr(self.model, "text_dim", None),
            )
        else:
            if flow_sigma is not None:
                raise ValueError(
                    "Stage-2 score roles forbid flow_sigma and use continuous t/1000"
                )
            if any(
                value is not None
                for value in (
                    kv_cache,
                    crossattn_cache,
                    current_start,
                    cache_start,
                    commit_self_kv,
                )
            ):
                raise ValueError(
                    "Stage-2 bidirectional score forward does not accept caches"
                )
            # Score noising remains FP32, while the actual Wan Conv3d patch
            # embedding is normally BF16.  Cast only the model input; x0 below
            # still uses the original FP32 noisy tensor and raw_flow.float().
            score_model_input = noisy_image_or_video.to(
                dtype=wan_patch_embedding_dtype(
                    self.model, fallback=noisy_image_or_video.dtype
                )
            )
            raw_flow = forward_stage2_i2v_score_model(
                self.model,
                noisy_image_or_video=score_model_input,
                conditional_dict=conditional_dict,
                frame_timestep=timestep,
                patch_size=patch_size,
                maximum_text_length=getattr(self.model, "text_len", None),
                expected_text_dim=getattr(self.model, "text_dim", None),
            )
        x0_pred_fp32 = (
            self._generator_flow_to_x0_fp32(
                noisy_image_or_video,
                raw_flow,
                flow_sigma,
            )
            if self.role == "generator"
            else self._continuous_flow_to_x0_fp32(
                noisy_image_or_video,
                raw_flow,
                timestep,
            )
        )
        x0_pred = (
            x0_pred_fp32.to(dtype=noisy_image_or_video.dtype)
            if self.role == "generator"
            else x0_pred_fp32
        )
        return raw_flow, x0_pred

    def forward_score(
        self,
        *,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: Mapping[str, torch.Tensor],
        frame_timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.role == "generator":
            raise RuntimeError("generator is not a bidirectional score role")
        return self(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=frame_timestep,
        )


def _parameter_storage_ids(module: nn.Module) -> set[tuple[str, int]]:
    storages: set[tuple[str, int]] = set()
    for parameter in module.parameters():
        if parameter.is_meta or parameter.numel() == 0:
            continue
        local = parameter
        to_local = getattr(parameter, "to_local", None)
        if callable(to_local):
            local = to_local()
        if local.numel() == 0:
            continue
        storage = local.untyped_storage()
        storages.add((str(local.device), int(storage.data_ptr())))
    return storages


@dataclass(frozen=True)
class Stage2GeneratorLossOutput:
    """FP32 Generator surrogate plus detached per-video diagnostics."""

    branch: str
    loss: torch.Tensor
    numerator: torch.Tensor
    count: int
    denominator: torch.Tensor
    denominator_clamped: torch.Tensor
    raw_score_difference_l2: torch.Tensor
    fake_x0_l2: torch.Tensor
    real_x0_l2: torch.Tensor


@dataclass(frozen=True)
class Stage2FakeScoreLossOutput:
    """FP32 fake-score raw-flow DSM loss plus detached diagnostics."""

    loss: torch.Tensor
    numerator: torch.Tensor
    count: int
    target_flow_l2: torch.Tensor
    prediction_l2: torch.Tensor


def _require_stage2_future(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if value.ndim != 5 or int(value.shape[1]) != STAGE2_SCORE_FUTURE_FRAMES:
        raise ValueError(f"{name} must have shape [B,24,C,H,W]")
    if int(value.shape[0]) < 1:
        raise ValueError(f"{name} batch must be non-empty")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains non-finite values")


def _require_score_tensor_for_future(
    value: torch.Tensor,
    name: str,
    generated_future: torch.Tensor,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    expected = (
        int(generated_future.shape[0]),
        STAGE2_SCORE_INPUT_FRAMES,
        *tuple(generated_future.shape[2:]),
    )
    if tuple(value.shape) != expected:
        raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
    if value.device != generated_future.device:
        raise ValueError(f"{name} and generated_future devices differ")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains non-finite values")


def _per_video_rms(value: torch.Tensor) -> torch.Tensor:
    reduction_dims = tuple(range(1, value.ndim))
    return value.float().square().mean(dim=reduction_dims).sqrt()


def compute_stage2_generator_distribution_matching_loss(
    *,
    branch: str,
    generated_future: torch.Tensor,
    noisy_fake_score: torch.Tensor,
    frame_timestep: torch.Tensor,
    fake_raw_flow: torch.Tensor,
    real_cond_raw_flow: torch.Tensor,
    real_uncond_raw_flow: torch.Tensor,
    noisy_real_score: torch.Tensor | None = None,
) -> Stage2GeneratorLossOutput:
    """Compute the exact future-only DMD or DFD Generator surrogate.

    ``dmd`` evaluates both teachers on ``noisy_fake_score``.  ``dfd`` changes
    only the real teacher input to ``noisy_real_score``; callers construct that
    tensor with :func:`utils.stage2_score_math.noise_stage2_score_pair` so the
    real/fake branches share the same video-global time and iid epsilon.

    All score-derived tensors are detached before the surrogate is formed.
    Therefore backward can reach ``generated_future`` directly, but never the
    fake/real score predictions or the noising graph.
    """

    if branch not in {"dmd", "dfd"}:
        raise ValueError("Stage-2 Generator branch must be 'dmd' or 'dfd'")
    _require_stage2_future(generated_future, "generated_future")
    if not generated_future.requires_grad:
        raise ValueError("generated_future must require grad for Generator update")

    score_tensors = {
        "noisy_fake_score": noisy_fake_score,
        "fake_raw_flow": fake_raw_flow,
        "real_cond_raw_flow": real_cond_raw_flow,
        "real_uncond_raw_flow": real_uncond_raw_flow,
    }
    for name, value in score_tensors.items():
        _require_score_tensor_for_future(value, name, generated_future)

    if branch == "dmd":
        if noisy_real_score is not None:
            raise ValueError("DMD must not receive noisy_real_score")
        real_teacher_input = noisy_fake_score
    else:
        if noisy_real_score is None:
            raise ValueError("DFD requires noisy_real_score")
        _require_score_tensor_for_future(
            noisy_real_score,
            "noisy_real_score",
            generated_future,
        )
        if not bool(
            torch.equal(
                noisy_fake_score[:, :1].float(),
                noisy_real_score[:, :1].float(),
            )
        ):
            raise ValueError("DFD noisy fake/real inputs must share the exact sink")
        real_teacher_input = noisy_real_score

    # Detach both prediction and teacher inputs before score-to-x0 math.  This
    # prevents accidental score-parameter or x_hat->score Jacobian gradients.
    fake_x0 = stage2_flow_to_x0(
        noisy_fake_score.detach(),
        fake_raw_flow.detach(),
        frame_timestep,
    )
    real_cfg_flow = stage2_real_cfg_flow(
        cond_flow=real_cond_raw_flow.detach(),
        uncond_flow=real_uncond_raw_flow.detach(),
    )
    real_x0 = stage2_flow_to_x0(
        real_teacher_input.detach(),
        real_cfg_flow,
        frame_timestep,
    )

    fake_future = fake_x0[:, 1:].detach()
    real_future = real_x0[:, 1:].detach()
    generated_fp32 = generated_future.float()
    raw_score_difference = fake_future - real_future
    reduction_dims = tuple(range(1, generated_fp32.ndim))
    denominator = (generated_fp32.detach() - real_future).abs().mean(dim=reduction_dims)
    if not bool(torch.isfinite(denominator).all().item()):
        raise ValueError("Stage-2 Generator denominator contains non-finite values")
    denominator_clamped = denominator.clamp_min(STAGE2_SCORE_DENOMINATOR_CLAMP)
    broadcast_shape = (int(generated_fp32.shape[0]),) + (1,) * (generated_fp32.ndim - 1)
    normalized_gradient = (
        raw_score_difference / denominator_clamped.view(broadcast_shape)
    ).detach()
    if not bool(torch.isfinite(normalized_gradient).all().item()):
        raise ValueError("Stage-2 Generator normalized gradient is non-finite")

    # The detached target realizes exactly 0.5*MSE with gradient g/N.
    surrogate_target = (generated_fp32 - normalized_gradient).detach()
    residual = generated_fp32 - surrogate_target
    numerator = 0.5 * residual.square().sum(dtype=torch.float32)
    count = int(generated_fp32.numel())
    loss = numerator / count
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("Stage-2 Generator surrogate loss is non-finite")

    return Stage2GeneratorLossOutput(
        branch=branch,
        loss=loss,
        numerator=numerator,
        count=count,
        denominator=denominator.detach(),
        denominator_clamped=denominator_clamped.detach(),
        raw_score_difference_l2=_per_video_rms(raw_score_difference).detach(),
        fake_x0_l2=_per_video_rms(fake_future).detach(),
        real_x0_l2=_per_video_rms(real_future).detach(),
    )


def compute_stage2_fake_score_flow_dsm_loss(
    *,
    generated_future: torch.Tensor,
    future_epsilon: torch.Tensor,
    fake_raw_flow: torch.Tensor,
) -> Stage2FakeScoreLossOutput:
    """Compute fake-score DSM directly in native raw-flow space.

    The sink prediction is deliberately sliced away.  There is no CFG,
    real-score call, real video, sigma/SNR multiplier, or x0 round-trip.
    """

    _require_stage2_future(generated_future, "generated_future")
    if generated_future.requires_grad:
        raise ValueError("fake-score update requires a detached no-grad rollout")
    _require_stage2_future(future_epsilon, "future_epsilon")
    if tuple(future_epsilon.shape) != tuple(generated_future.shape):
        raise ValueError("future_epsilon shape must match generated_future")
    if future_epsilon.device != generated_future.device:
        raise ValueError("future_epsilon and generated_future devices differ")
    _require_score_tensor_for_future(
        fake_raw_flow,
        "fake_raw_flow",
        generated_future,
    )

    target_flow = future_epsilon.float() - generated_future.detach().float()
    prediction = fake_raw_flow[:, 1:].float()
    residual = prediction - target_flow
    numerator = residual.square().sum(dtype=torch.float32)
    count = int(residual.numel())
    loss = numerator / count
    if not bool(torch.isfinite(loss).item()):
        raise ValueError("Stage-2 fake-score flow DSM loss is non-finite")
    return Stage2FakeScoreLossOutput(
        loss=loss,
        numerator=numerator,
        count=count,
        target_flow_l2=_per_video_rms(target_flow).mean().detach(),
        prediction_l2=_per_video_rms(prediction).mean().detach(),
    )


class Stage2DMD(nn.Module):
    """Three owned roles plus explicit, gradient-audited loss reductions."""

    def __init__(
        self,
        *,
        generator: Stage2DiTRole,
        real_score: Stage2DiTRole,
        fake_score: Stage2DiTRole,
    ):
        super().__init__()
        expected = {
            "generator": generator,
            "real_score": real_score,
            "fake_score": fake_score,
        }
        for role, value in expected.items():
            if not isinstance(value, Stage2DiTRole) or value.role != role:
                raise ValueError(f"Stage-2 {role} wrapper has the wrong role")
        self.generator = generator
        self.real_score = real_score
        self.fake_score = fake_score
        self.audit_independent_role_storage()

    def _roles(self) -> Iterable[tuple[str, Stage2DiTRole]]:
        return (
            ("generator", self.generator),
            ("real_score", self.real_score),
            ("fake_score", self.fake_score),
        )

    def audit_independent_role_storage(self) -> dict[str, object]:
        roles = dict(self._roles())
        if len({id(value) for value in roles.values()}) != 3:
            raise RuntimeError(
                "Stage-2 role wrappers must be three independent objects"
            )
        if len({id(value.model) for value in roles.values()}) != 3:
            raise RuntimeError("Stage-2 role DiTs must be three independent objects")

        parameter_ids = {
            role: {id(parameter) for parameter in wrapper.parameters()}
            for role, wrapper in roles.items()
        }
        storage_ids = {
            role: _parameter_storage_ids(wrapper) for role, wrapper in roles.items()
        }
        role_names = tuple(roles)
        for left_index, left in enumerate(role_names):
            for right in role_names[left_index + 1 :]:
                if parameter_ids[left] & parameter_ids[right]:
                    raise RuntimeError(
                        f"Stage-2 roles share Parameter objects: {left}/{right}"
                    )
                if storage_ids[left] & storage_ids[right]:
                    raise RuntimeError(
                        f"Stage-2 roles share parameter storage: {left}/{right}"
                    )
        return {
            "parameter_objects_disjoint": True,
            "parameter_storage_disjoint": True,
            "roles": role_names,
        }

    def forward_score(
        self,
        role: str,
        *,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: Mapping[str, torch.Tensor],
        frame_timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch one explicit real/fake bidirectional score model call."""

        if role not in {"real_score", "fake_score"}:
            raise ValueError("score role must be 'real_score' or 'fake_score'")
        return getattr(self, role).forward_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            frame_timestep=frame_timestep,
        )

    def audit_generator_teacher_state(self) -> dict[str, object]:
        """Require the immutable real teacher; fake isolation uses ``no_grad``.

        Fake-score LoRA stays trainable across the 5F->1G cycle.  Toggling its
        ``requires_grad`` flags around every G update is unnecessary and can be
        unsafe after FSDP wrapping.  The model-level G API instead evaluates
        both score roles under ``torch.no_grad`` and the pure reducer detaches
        every supplied score tensor.
        """

        trainable_real = [
            name
            for name, parameter in self.real_score.named_parameters()
            if parameter.requires_grad
        ]
        if trainable_real:
            preview = ", ".join(trainable_real[:3])
            if len(trainable_real) > 3:
                preview += ", ..."
            raise RuntimeError(
                "Stage-2 real_score must remain frozen; trainable parameters: "
                f"{preview}"
            )
        return {
            "real_score_frozen": True,
            "fake_score_isolated_by_no_grad": True,
        }

    def generator_distribution_matching_loss(
        self,
        *,
        branch: str,
        generated_future: torch.Tensor,
        noisy_fake_score: torch.Tensor,
        frame_timestep: torch.Tensor,
        fake_raw_flow: torch.Tensor,
        real_cond_raw_flow: torch.Tensor,
        real_uncond_raw_flow: torch.Tensor,
        noisy_real_score: torch.Tensor | None = None,
    ) -> Stage2GeneratorLossOutput:
        """Reduce already-evaluated frozen score outputs into a DMD/DFD loss."""

        self.audit_generator_teacher_state()
        return compute_stage2_generator_distribution_matching_loss(
            branch=branch,
            generated_future=generated_future,
            noisy_fake_score=noisy_fake_score,
            noisy_real_score=noisy_real_score,
            frame_timestep=frame_timestep,
            fake_raw_flow=fake_raw_flow,
            real_cond_raw_flow=real_cond_raw_flow,
            real_uncond_raw_flow=real_uncond_raw_flow,
        )

    def generator_distribution_matching_loss_from_models(
        self,
        *,
        branch: str,
        generated_future: torch.Tensor,
        noised_score: Stage2NoisedScoreInput | Stage2NoisedScorePair,
        conditional_dict: Mapping[str, torch.Tensor],
        real_unconditional_dict: Mapping[str, torch.Tensor],
        timing_callback: Callable[[str, Callable[[], object]], object] | None = None,
    ) -> Stage2GeneratorLossOutput:
        """Evaluate frozen score roles and reduce one DMD/DFD G loss.

        The Generator rollout is intentionally an input to this method: its
        random-exit/cache semantics live in the Stage-2 rollout pipeline.  This
        method calls fake-score once (positive conditioning, CFG1) and
        real-score twice.  Both real cond/uncond calls receive the exact same
        noisy tensor object and frame-timestep object.
        """

        self.audit_generator_teacher_state()
        _require_stage2_future(generated_future, "generated_future")
        if branch == "dmd":
            if not isinstance(noised_score, Stage2NoisedScoreInput):
                raise TypeError("DMD must consume one Stage2NoisedScoreInput")
            fake_noising = validate_stage2_noised_score_input(noised_score)
            real_noising = None
            real_teacher_input = fake_noising.noisy_score
        elif branch == "dfd":
            if not isinstance(noised_score, Stage2NoisedScorePair):
                raise TypeError("DFD must consume one Stage2NoisedScorePair")
            pair = validate_stage2_noised_score_pair(noised_score)
            fake_noising = pair.fake
            real_noising = pair.real
            real_teacher_input = real_noising.noisy_score
        else:
            raise ValueError("Stage-2 Generator branch must be 'dmd' or 'dfd'")
        noisy_fake_score = fake_noising.noisy_score
        frame_timestep = fake_noising.frame_timestep
        _require_score_tensor_for_future(
            noisy_fake_score,
            "noisy_fake_score",
            generated_future,
        )
        if real_noising is not None:
            _require_score_tensor_for_future(
                real_noising.noisy_score,
                "noisy_real_score",
                generated_future,
            )

        # These teachers are inference-only for a Generator update.  no_grad
        # also prevents an expensive x_hat->score input Jacobian from forming.
        def measured(label: str, callback: Callable[[], object]) -> object:
            return (
                callback()
                if timing_callback is None
                else timing_callback(label, callback)
            )

        with torch.no_grad():
            fake_raw_flow, _ = measured(
                "fake_score",
                lambda: self.fake_score.forward_score(
                    noisy_image_or_video=noisy_fake_score,
                    conditional_dict=conditional_dict,
                    frame_timestep=frame_timestep,
                ),
            )
            real_cond_raw_flow, _ = measured(
                "real_cond",
                lambda: self.real_score.forward_score(
                    noisy_image_or_video=real_teacher_input,
                    conditional_dict=conditional_dict,
                    frame_timestep=frame_timestep,
                ),
            )
            real_uncond_raw_flow, _ = measured(
                "real_uncond",
                lambda: self.real_score.forward_score(
                    noisy_image_or_video=real_teacher_input,
                    conditional_dict=real_unconditional_dict,
                    frame_timestep=frame_timestep,
                ),
            )

        return compute_stage2_generator_distribution_matching_loss(
            branch=branch,
            generated_future=generated_future,
            noisy_fake_score=noisy_fake_score,
            noisy_real_score=(
                None if real_noising is None else real_noising.noisy_score
            ),
            frame_timestep=frame_timestep,
            fake_raw_flow=fake_raw_flow,
            real_cond_raw_flow=real_cond_raw_flow,
            real_uncond_raw_flow=real_uncond_raw_flow,
        )

    def fake_score_flow_dsm_loss(
        self,
        *,
        generated_future: torch.Tensor,
        future_epsilon: torch.Tensor,
        fake_raw_flow: torch.Tensor,
    ) -> Stage2FakeScoreLossOutput:
        """Reduce one fake-score raw-flow prediction for an F update."""

        return compute_stage2_fake_score_flow_dsm_loss(
            generated_future=generated_future,
            future_epsilon=future_epsilon,
            fake_raw_flow=fake_raw_flow,
        )

    def fake_score_flow_dsm_loss_from_model(
        self,
        *,
        generated_future: torch.Tensor,
        noised_fake_score: Stage2NoisedScoreInput,
        conditional_dict: Mapping[str, torch.Tensor],
        timing_callback: Callable[[str, Callable[[], object]], object] | None = None,
    ) -> Stage2FakeScoreLossOutput:
        """Evaluate only fake-score and reduce its direct raw-flow DSM loss."""

        _require_stage2_future(generated_future, "generated_future")
        noising = validate_stage2_noised_score_input(noised_fake_score)
        noisy_fake_score = noising.noisy_score
        if generated_future.requires_grad or noisy_fake_score.requires_grad:
            raise ValueError("fake-score update requires a detached no-grad rollout")
        _require_score_tensor_for_future(
            noisy_fake_score,
            "noisy_fake_score",
            generated_future,
        )

        def callback():
            return self.fake_score.forward_score(
                noisy_image_or_video=noisy_fake_score,
                conditional_dict=conditional_dict,
                frame_timestep=noising.frame_timestep,
            )

        fake_raw_flow, _ = (
            callback()
            if timing_callback is None
            else timing_callback("fake_score", callback)
        )
        return compute_stage2_fake_score_flow_dsm_loss(
            generated_future=generated_future,
            future_epsilon=noising.future_epsilon,
            fake_raw_flow=fake_raw_flow,
        )

    def forward(self, *args, **kwargs):  # pragma: no cover - safety tripwire
        raise RuntimeError(
            "Stage-2 has no ambiguous combined forward; call one explicit role "
            "adapter and then the matching DMD/DFD or fake-score loss method."
        )
