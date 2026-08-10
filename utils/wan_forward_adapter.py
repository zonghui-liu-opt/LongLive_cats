"""Lightweight Wan call adapters shared by legacy and Stage-2 wrappers.

This module intentionally imports no Wan weights, Diffusers, T5, or VAE
implementation, so shape/cache contract tests can run on a CPU-only host.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import os
from typing import Any, Optional

import torch
from torch import nn


def wan_patch_embedding_dtype(
    model: nn.Module, *, fallback: torch.dtype
) -> torch.dtype:
    """Resolve Wan's patch-embedding compute dtype through PEFT/FSDP proxies."""

    patch_embedding = getattr(model, "patch_embedding", None)
    if patch_embedding is None:
        return fallback
    weight = getattr(patch_embedding, "weight", None)
    if isinstance(weight, torch.Tensor) and weight.is_floating_point():
        return weight.dtype
    if isinstance(patch_embedding, nn.Module):
        for parameter in patch_embedding.parameters():
            if parameter.is_floating_point():
                return parameter.dtype
    raise RuntimeError("Wan patch_embedding has no floating-point weight")


def legacy_wan_model_timestep(
    timestep: torch.Tensor, *, uniform_timestep: bool
) -> torch.Tensor:
    """Preserve the legacy noncausal scalar-time and causal frame-time split."""

    if uniform_timestep:
        return timestep[:, 0]
    return timestep


def _publish_causal_cache_grid_meta(kv_cache: Optional[list[dict]]) -> None:
    """Publish cache cursors before a potentially compiled causal forward."""

    if kv_cache is None or len(kv_cache) == 0:
        return
    try:
        from wan_5b.modules.causal_model import _CURRENT_GRID_META

        first_block_cache = kv_cache[0]
        _CURRENT_GRID_META["global_end_index"] = int(
            first_block_cache["global_end_index"].item()
        )
        _CURRENT_GRID_META["local_end_index"] = int(
            first_block_cache["local_end_index"].item()
        )
        pinned_start = first_block_cache.get("pinned_start", None)
        if pinned_start is not None and hasattr(pinned_start, "item"):
            _CURRENT_GRID_META["pinned_start"] = int(pinned_start.item())
            _CURRENT_GRID_META["pinned_len"] = int(
                first_block_cache["pinned_len"].item()
            )
        else:
            _CURRENT_GRID_META["pinned_start"] = -1
            _CURRENT_GRID_META["pinned_len"] = 0
    except (KeyError, AttributeError, ImportError):
        # Preserve the legacy wrapper's best-effort publication behavior.
        pass


def _detached_cache_update_infos(cache_update_infos: list) -> list:
    """Detach tensors carried by deferred self-KV update descriptions."""

    detached_infos = []
    for block_index, cache_update_info in cache_update_infos:
        current_end, local_end_index, update_info = cache_update_info
        if update_info is not None:
            update_info = {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in update_info.items()
            }
        detached_infos.append(
            (block_index, (current_end, local_end_index, update_info))
        )
    return detached_infos


def _assert_persistent_self_kv_detached(kv_cache: list[dict]) -> None:
    contaminated = []
    for block_index, cache in enumerate(kv_cache):
        for key in ("k", "v"):
            value = cache.get(key)
            if isinstance(value, torch.Tensor) and (
                value.requires_grad or value.grad_fn is not None
            ):
                contaminated.append(
                    (
                        block_index,
                        key,
                        bool(value.requires_grad),
                        (
                            type(value.grad_fn).__name__
                            if value.grad_fn is not None
                            else None
                        ),
                    )
                )
    if contaminated:
        raise RuntimeError(
            "committed persistent self-KV retains autograd state: "
            f"{contaminated[:8]}"
        )


def call_wan_model_with_cache_policy(
    model_call: Callable[..., Any],
    *model_args,
    cache_update_owner: nn.Module,
    commit_self_kv: bool | None = None,
    **model_kwargs,
):
    """Call Wan while making Stage-2 self-KV mutation explicit.

    ``commit_self_kv=None`` is the legacy behavior, including the existing
    ``LLV2_DEFER_KV_UPDATES`` experiment.  ``False`` computes against the
    supplied history but discards this call's updates.  ``True`` commits only
    detached updates and verifies that persistent K/V has no autograd state.
    """

    if commit_self_kv is not None and not isinstance(commit_self_kv, bool):
        raise TypeError("commit_self_kv must be bool or None")
    kv_cache = model_kwargs.get("kv_cache", None)
    if commit_self_kv is not None and kv_cache is None:
        raise ValueError("commit_self_kv requires a self kv_cache")

    _publish_causal_cache_grid_meta(kv_cache)
    legacy_defer = (
        commit_self_kv is None
        and os.environ.get("LLV2_DEFER_KV_UPDATES", "0") == "1"
        and kv_cache is not None
    )
    defer_updates = kv_cache is not None and (
        commit_self_kv is not None or legacy_defer
    )
    if defer_updates:
        model_kwargs["defer_cache_updates"] = True

    result = model_call(*model_args, **model_kwargs)
    if not defer_updates:
        return result
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "deferred Wan self-KV call expected " "(output, cache_update_infos)"
        )
    output, cache_update_infos = result

    if commit_self_kv is False:
        return output
    apply_updates = getattr(cache_update_owner, "_apply_cache_updates", None)
    if not callable(apply_updates):
        raise RuntimeError("Wan model cannot apply deferred self-KV updates")
    if commit_self_kv is True:
        cache_update_infos = _detached_cache_update_infos(cache_update_infos)
    if cache_update_infos:
        apply_updates(kv_cache, cache_update_infos)
    if commit_self_kv is True:
        _assert_persistent_self_kv_detached(kv_cache)
    return output


def _validate_prompt_conditioning(
    conditional_dict: Mapping[str, torch.Tensor],
    batch_size: int,
    *,
    device: torch.device,
    maximum_text_length: int | None = None,
    expected_text_dim: int | None = None,
) -> torch.Tensor:
    if not isinstance(conditional_dict, Mapping):
        raise TypeError("conditional_dict must be a mapping")
    if set(conditional_dict) != {"prompt_embeds"}:
        raise ValueError("Wan Stage-2 conditioning accepts only explicit prompt_embeds")
    prompt_embeds = conditional_dict["prompt_embeds"]
    if not isinstance(prompt_embeds, torch.Tensor) or prompt_embeds.ndim != 3:
        raise ValueError("prompt_embeds must have shape [B,L,D]")
    if int(prompt_embeds.shape[0]) != int(batch_size):
        raise ValueError("prompt_embeds batch size differs from latent batch")
    if maximum_text_length is not None and int(prompt_embeds.shape[1]) > int(
        maximum_text_length
    ):
        raise ValueError(
            "prompt_embeds sequence length exceeds the Wan text_len contract"
        )
    if expected_text_dim is not None and int(prompt_embeds.shape[2]) != int(
        expected_text_dim
    ):
        raise ValueError("prompt_embeds width differs from the Wan text_dim contract")
    if prompt_embeds.dtype != torch.bfloat16:
        raise TypeError("Wan Stage-2 prompt_embeds must be bfloat16")
    if prompt_embeds.device != device:
        raise ValueError("prompt_embeds and Stage-2 latent devices differ")
    return prompt_embeds


def forward_stage2_i2v_score_model(
    model_call: Callable[..., torch.Tensor],
    *,
    noisy_image_or_video: torch.Tensor,
    conditional_dict: Mapping[str, torch.Tensor],
    frame_timestep: torch.Tensor,
    patch_size: tuple[int, int, int],
    maximum_text_length: int | None = None,
    expected_text_dim: int | None = None,
) -> torch.Tensor:
    """Run the strict 25-frame bidirectional Stage-2 TI2V score adapter."""

    from utils.stage2_i2v_conditioning import (
        prepare_stage2_i2v_score_model_inputs,
    )

    prepared = prepare_stage2_i2v_score_model_inputs(
        noisy_image_or_video,
        frame_timestep,
        patch_size=patch_size,
    )
    prompt_embeds = _validate_prompt_conditioning(
        conditional_dict,
        int(noisy_image_or_video.shape[0]),
        device=noisy_image_or_video.device,
        maximum_text_length=maximum_text_length,
        expected_text_dim=expected_text_dim,
    )
    flow_pred = model_call(
        noisy_image_or_video.permute(0, 2, 1, 3, 4),
        t=prepared.token_timestep,
        context=prompt_embeds,
        seq_len=prepared.seq_len,
    )
    if not isinstance(flow_pred, torch.Tensor) or flow_pred.ndim != 5:
        raise RuntimeError("Wan score model must return a 5D tensor")
    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
    if flow_pred.shape != noisy_image_or_video.shape:
        raise RuntimeError(
            "Wan score output shape differs from its input: "
            f"{tuple(flow_pred.shape)} vs {tuple(noisy_image_or_video.shape)}"
        )
    return flow_pred


def forward_stage2_causal_model(
    model_call: Callable[..., torch.Tensor],
    *,
    cache_update_owner: nn.Module,
    noisy_image_or_video: torch.Tensor,
    conditional_dict: Mapping[str, torch.Tensor],
    frame_timestep: torch.Tensor,
    patch_size: tuple[int, int, int],
    kv_cache: Optional[list[dict]],
    crossattn_cache: Optional[list[dict]],
    current_start: Optional[int],
    cache_start: Optional[int] = None,
    commit_self_kv: bool | None = None,
    maximum_text_length: int | None = None,
    expected_text_dim: int | None = None,
) -> torch.Tensor:
    """Run a Stage-2 causal chunk with dynamic geometry and cache policy."""

    from utils.stage2_i2v_conditioning import (
        STAGE2_ALLOWED_LATENT_SPATIAL_SHAPES,
        STAGE2_LATENT_CHANNELS,
        STAGE2_LATENT_PATCH_SIZE,
        STAGE2_PATCH_TOKENS_PER_FRAME,
    )

    if not isinstance(noisy_image_or_video, torch.Tensor) or (
        noisy_image_or_video.ndim != 5
    ):
        raise ValueError("noisy_image_or_video must have shape [B,F,C,H,W]")
    if int(noisy_image_or_video.shape[2]) != STAGE2_LATENT_CHANNELS:
        raise ValueError("Stage-2 causal input must have 48 latent channels")
    spatial = tuple(int(value) for value in noisy_image_or_video.shape[3:])
    if spatial not in STAGE2_ALLOWED_LATENT_SPATIAL_SHAPES:
        raise ValueError(f"unsupported Stage-2 latent orientation: {spatial}")
    if tuple(int(value) for value in patch_size) != STAGE2_LATENT_PATCH_SIZE:
        raise ValueError("Stage-2 causal model requires patch_size=(1,2,2)")
    if not isinstance(frame_timestep, torch.Tensor):
        raise TypeError("causal frame_timestep must be a torch.Tensor")
    if frame_timestep.dtype != torch.float32:
        raise TypeError("causal frame_timestep must remain float32")
    expected_timestep_shape = tuple(noisy_image_or_video.shape[:2])
    if tuple(frame_timestep.shape) != expected_timestep_shape:
        raise ValueError(
            f"causal frame_timestep must have shape {expected_timestep_shape}"
        )
    if frame_timestep.device != noisy_image_or_video.device:
        raise ValueError("causal frame_timestep and latent devices differ")
    prompt_embeds = _validate_prompt_conditioning(
        conditional_dict,
        int(noisy_image_or_video.shape[0]),
        device=noisy_image_or_video.device,
        maximum_text_length=maximum_text_length,
        expected_text_dim=expected_text_dim,
    )
    if kv_cache is None or crossattn_cache is None:
        raise ValueError("Stage-2 causal forward requires self/cross KV caches")
    if (
        current_start is None
        or isinstance(current_start, bool)
        or not isinstance(current_start, int)
        or current_start < 0
    ):
        raise ValueError("current_start must be a non-negative token index")
    if cache_start is not None and (
        isinstance(cache_start, bool)
        or not isinstance(cache_start, int)
        or cache_start < 0
    ):
        raise ValueError("cache_start must be a non-negative token index or None")
    if current_start % STAGE2_PATCH_TOKENS_PER_FRAME:
        raise ValueError("current_start must be aligned to a latent frame")
    if cache_start is not None and (cache_start % STAGE2_PATCH_TOKENS_PER_FRAME):
        raise ValueError("cache_start must be aligned to a latent frame")
    seq_len = int(noisy_image_or_video.shape[1]) * STAGE2_PATCH_TOKENS_PER_FRAME
    flow_pred = call_wan_model_with_cache_policy(
        model_call,
        noisy_image_or_video.permute(0, 2, 1, 3, 4),
        cache_update_owner=cache_update_owner,
        commit_self_kv=commit_self_kv,
        t=frame_timestep,
        context=prompt_embeds,
        seq_len=seq_len,
        kv_cache=kv_cache,
        crossattn_cache=crossattn_cache,
        current_start=current_start,
        cache_start=cache_start,
    )
    if not isinstance(flow_pred, torch.Tensor) or flow_pred.ndim != 5:
        raise RuntimeError("Wan causal model must return a 5D tensor")
    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
    if flow_pred.shape != noisy_image_or_video.shape:
        raise RuntimeError("Wan causal output shape differs from its input")
    return flow_pred
