"""Strict Stage-2 TI2V score packing and timestep expansion.

Stage-2 deliberately does not reuse the legacy I2V helper that overwrites
frame zero in an existing clip.  The score clip is constructed explicitly as
one immutable image latent followed by all 24 newly generated video latents.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

STAGE2_INITIAL_LATENT_FRAMES = 1
STAGE2_FUTURE_LATENT_FRAMES = 24
STAGE2_SCORE_INPUT_FRAMES = 25
STAGE2_LATENT_CHANNELS = 48
STAGE2_LATENT_PATCH_SIZE = (1, 2, 2)
STAGE2_ALLOWED_LATENT_SPATIAL_SHAPES = ((30, 52), (52, 30))
STAGE2_PATCH_TOKENS_PER_FRAME = 390
STAGE2_SCORE_SEQ_LEN = 9_750
STAGE2_SCORE_TIMESTEP_MIN = 20.0
STAGE2_SCORE_TIMESTEP_MAX = 980.0


@dataclass(frozen=True)
class Stage2I2VScoreModelInputs:
    """Validated model-time inputs for one already-packed score clip."""

    frame_timestep: torch.Tensor
    token_timestep: torch.Tensor
    patch_tokens_per_frame: int
    seq_len: int


def _require_5d_latent(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 5:
        raise ValueError(
            f"{name} must have shape [B,F,C,H,W], got {tuple(value.shape)}"
        )


def _validate_locked_latent_geometry(value: torch.Tensor, name: str) -> None:
    if int(value.shape[2]) != STAGE2_LATENT_CHANNELS:
        raise ValueError(
            f"{name} must have {STAGE2_LATENT_CHANNELS} latent channels, "
            f"got {value.shape[2]}"
        )
    spatial = (int(value.shape[3]), int(value.shape[4]))
    if spatial not in STAGE2_ALLOWED_LATENT_SPATIAL_SHAPES:
        raise ValueError(
            f"{name} spatial shape must be one of "
            f"{STAGE2_ALLOWED_LATENT_SPATIAL_SHAPES}, got {spatial}"
        )


def pack_stage2_i2v_score_input(
    initial_latent: torch.Tensor,
    future_latent: torch.Tensor,
) -> torch.Tensor:
    """Return ``[explicit initial1, future24]`` without overwriting a slot.

    ``torch.cat`` is intentional: every future frame remains a distinct node
    in the Generator autograd graph, including future frame 0 and frame 23.
    """

    _require_5d_latent(initial_latent, "initial_latent")
    _require_5d_latent(future_latent, "future_latent")
    if int(initial_latent.shape[1]) != STAGE2_INITIAL_LATENT_FRAMES:
        raise ValueError(
            "Stage-2 score conditioning requires exactly one explicit "
            f"initial latent frame, got {initial_latent.shape[1]}"
        )
    if int(future_latent.shape[1]) != STAGE2_FUTURE_LATENT_FRAMES:
        raise ValueError(
            "Stage-2 score conditioning requires exactly 24 future latent "
            f"frames, got {future_latent.shape[1]}"
        )
    _validate_locked_latent_geometry(initial_latent, "initial_latent")
    _validate_locked_latent_geometry(future_latent, "future_latent")
    if initial_latent.shape[0] != future_latent.shape[0]:
        raise ValueError("initial_latent and future_latent batch sizes differ")
    if initial_latent.shape[2:] != future_latent.shape[2:]:
        raise ValueError("initial_latent and future_latent non-temporal shapes differ")
    if initial_latent.device != future_latent.device:
        raise ValueError("initial_latent and future_latent devices differ")
    if initial_latent.dtype != future_latent.dtype:
        raise TypeError("initial_latent and future_latent dtypes differ")

    packed = torch.cat((initial_latent, future_latent), dim=1)
    if tuple(packed.shape[:3]) != (
        int(initial_latent.shape[0]),
        STAGE2_SCORE_INPUT_FRAMES,
        STAGE2_LATENT_CHANNELS,
    ):
        raise AssertionError("Stage-2 score pack shape drifted")
    return packed


def _score_patch_geometry(
    score_input: torch.Tensor,
    patch_size: tuple[int, int, int],
) -> tuple[int, int]:
    _require_5d_latent(score_input, "score_input")
    if int(score_input.shape[1]) != STAGE2_SCORE_INPUT_FRAMES:
        raise ValueError(
            f"Stage-2 score_input must contain 25 frames, got {score_input.shape[1]}"
        )
    _validate_locked_latent_geometry(score_input, "score_input")
    normalized_patch_size = tuple(int(value) for value in patch_size)
    if normalized_patch_size != STAGE2_LATENT_PATCH_SIZE:
        raise ValueError(
            "Stage-2 TI2V score requires latent patch_size=(1,2,2), got "
            f"{normalized_patch_size}"
        )
    _, patch_height, patch_width = normalized_patch_size
    height, width = (int(score_input.shape[3]), int(score_input.shape[4]))
    if height % patch_height or width % patch_width:
        raise ValueError("score_input spatial shape is not patch divisible")
    patch_tokens_per_frame = (height // patch_height) * (width // patch_width)
    seq_len = int(score_input.shape[1]) * patch_tokens_per_frame
    if patch_tokens_per_frame != STAGE2_PATCH_TOKENS_PER_FRAME:
        raise ValueError(
            "Stage-2 orientation must produce 390 patch tokens per frame, "
            f"got {patch_tokens_per_frame}"
        )
    if seq_len != STAGE2_SCORE_SEQ_LEN:
        raise ValueError(f"Stage-2 score seq_len must be 9750, got {seq_len}")
    return patch_tokens_per_frame, seq_len


def prepare_stage2_i2v_score_model_inputs(
    score_input: torch.Tensor,
    frame_timestep: torch.Tensor,
    *,
    patch_size: tuple[int, int, int] = STAGE2_LATENT_PATCH_SIZE,
) -> Stage2I2VScoreModelInputs:
    """Validate frame time and expand it to Wan patch-token time.

    The first 390 tokens are the clean image sink at exactly ``t=0``.  Every
    one of the following 9,360 tokens uses the single video-global future
    timestep belonging to that sample.
    """

    patch_tokens_per_frame, seq_len = _score_patch_geometry(score_input, patch_size)
    if not isinstance(frame_timestep, torch.Tensor):
        raise TypeError("frame_timestep must be a torch.Tensor")
    expected_shape = (int(score_input.shape[0]), STAGE2_SCORE_INPUT_FRAMES)
    if tuple(frame_timestep.shape) != expected_shape:
        raise ValueError(
            f"frame_timestep must have shape {expected_shape}, got "
            f"{tuple(frame_timestep.shape)}"
        )
    if frame_timestep.device != score_input.device:
        raise ValueError("frame_timestep and score_input devices differ")
    if frame_timestep.dtype != torch.float32:
        raise TypeError("Stage-2 score frame_timestep must remain float32")
    if not bool(torch.isfinite(frame_timestep).all().item()):
        raise ValueError("frame_timestep contains non-finite values")
    if not bool(
        torch.equal(
            frame_timestep[:, :STAGE2_INITIAL_LATENT_FRAMES],
            torch.zeros_like(frame_timestep[:, :STAGE2_INITIAL_LATENT_FRAMES]),
        )
    ):
        raise ValueError("Stage-2 score sink timestep must be exactly zero")

    future_timestep = frame_timestep[:, STAGE2_INITIAL_LATENT_FRAMES:]
    expected_future = future_timestep[:, :1].expand_as(future_timestep)
    if not bool(torch.equal(future_timestep, expected_future)):
        raise ValueError(
            "Stage-2 score future timestep must be video-global per sample"
        )
    if bool((future_timestep < STAGE2_SCORE_TIMESTEP_MIN).any().item()) or bool(
        (future_timestep > STAGE2_SCORE_TIMESTEP_MAX).any().item()
    ):
        raise ValueError("Stage-2 score future timestep must be in [20, 980]")

    token_timestep = frame_timestep.repeat_interleave(patch_tokens_per_frame, dim=1)
    if tuple(token_timestep.shape) != (int(score_input.shape[0]), seq_len):
        raise AssertionError("Stage-2 token timestep shape drifted")
    return Stage2I2VScoreModelInputs(
        frame_timestep=frame_timestep,
        token_timestep=token_timestep,
        patch_tokens_per_frame=patch_tokens_per_frame,
        seq_len=seq_len,
    )
