"""Continuous FP32 score-time math for Stage-2 DMD/DFD.

This module is intentionally independent from the rollout scheduler.  Stage-2
score noising uses the exact continuous ``sigma = t / 1000`` contract below;
it must never look up a nearest discrete UniPC or training-scheduler sigma.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

STAGE2_SCORE_NUM_TRAIN_TIMESTEPS = 1000
STAGE2_SCORE_TIMESTEP_SHIFT = 5.0
STAGE2_SCORE_TIMESTEP_MIN = 20.0
STAGE2_SCORE_TIMESTEP_MAX = 980.0
STAGE2_SCORE_INPUT_FRAMES = 25
STAGE2_SCORE_FUTURE_FRAMES = 24
STAGE2_SCORE_REAL_CFG_SCALE = 5.0
STAGE2_SCORE_DENOMINATOR_CLAMP = 1.0e-6


@dataclass(frozen=True)
class Stage2ScoreTimesteps:
    """One independent video-global score-time sample per batch item."""

    uniform_integer: torch.Tensor
    future_timestep: torch.Tensor
    frame_timestep: torch.Tensor
    sigma: torch.Tensor


@dataclass(frozen=True)
class Stage2NoisedScoreInput:
    """FP32 continuous noising result for one ``[sink1, future24]`` clip."""

    noisy_score: torch.Tensor
    epsilon: torch.Tensor
    future_epsilon: torch.Tensor
    frame_timestep: torch.Tensor
    sigma: torch.Tensor


@dataclass(frozen=True)
class Stage2NoisedScorePair:
    """DFD fake/real teacher inputs built with exactly the same future noise."""

    fake: Stage2NoisedScoreInput
    real: Stage2NoisedScoreInput


def _require_floating_tensor(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")


def _require_finite(value: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} contains non-finite values")


def _require_score_input(value: torch.Tensor, name: str) -> None:
    _require_floating_tensor(value, name)
    if value.ndim != 5:
        raise ValueError(f"{name} must have shape [B,25,C,H,W]")
    if int(value.shape[0]) < 1:
        raise ValueError(f"{name} batch must be non-empty")
    if int(value.shape[1]) != STAGE2_SCORE_INPUT_FRAMES:
        raise ValueError(
            f"{name} must contain exactly {STAGE2_SCORE_INPUT_FRAMES} frames, "
            f"got {value.shape[1]}"
        )
    _require_finite(value, name)


def _require_future(value: torch.Tensor, name: str) -> None:
    _require_floating_tensor(value, name)
    if value.ndim != 5:
        raise ValueError(f"{name} must have shape [B,24,C,H,W]")
    if int(value.shape[0]) < 1:
        raise ValueError(f"{name} batch must be non-empty")
    if int(value.shape[1]) != STAGE2_SCORE_FUTURE_FRAMES:
        raise ValueError(
            f"{name} must contain exactly {STAGE2_SCORE_FUTURE_FRAMES} frames, "
            f"got {value.shape[1]}"
        )
    _require_finite(value, name)


def validate_stage2_frame_timestep(
    frame_timestep: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the strict ``[0, t x 24]`` frame time as FP32."""

    _require_floating_tensor(frame_timestep, "frame_timestep")
    expected_shape = (int(batch_size), STAGE2_SCORE_INPUT_FRAMES)
    if tuple(frame_timestep.shape) != expected_shape:
        raise ValueError(
            f"frame_timestep must have shape {expected_shape}, got "
            f"{tuple(frame_timestep.shape)}"
        )
    if frame_timestep.device != device:
        raise ValueError("frame_timestep and score tensors must use the same device")
    _require_finite(frame_timestep, "frame_timestep")
    value = frame_timestep.float()
    sink = value[:, :1]
    if not bool(torch.equal(sink, torch.zeros_like(sink))):
        raise ValueError("Stage-2 score sink timestep must be exactly zero")
    future = value[:, 1:]
    if not bool(torch.equal(future, future[:, :1].expand_as(future))):
        raise ValueError(
            "Stage-2 score future timestep must be video-global per sample"
        )
    if bool((future < STAGE2_SCORE_TIMESTEP_MIN).any().item()) or bool(
        (future > STAGE2_SCORE_TIMESTEP_MAX).any().item()
    ):
        raise ValueError("Stage-2 score future timestep must be in [20, 980]")
    return value


def stage2_score_timesteps_from_uniform_integers(
    uniform_integer: torch.Tensor,
) -> Stage2ScoreTimesteps:
    """Apply the locked shifted-integer mapping without a scheduler lookup."""

    if not isinstance(uniform_integer, torch.Tensor):
        raise TypeError("uniform_integer must be a torch.Tensor")
    if uniform_integer.ndim != 1:
        raise ValueError("uniform_integer must be one-dimensional")
    if uniform_integer.numel() < 1:
        raise ValueError("uniform_integer batch must be non-empty")
    if uniform_integer.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("uniform_integer must use an integer dtype")
    raw = uniform_integer.to(dtype=torch.int64)
    if bool((raw < 0).any().item()) or bool(
        (raw >= STAGE2_SCORE_NUM_TRAIN_TIMESTEPS).any().item()
    ):
        raise ValueError("uniform_integer values must be in [0, 1000)")

    u = raw.float() / float(STAGE2_SCORE_NUM_TRAIN_TIMESTEPS)
    shifted = (
        float(STAGE2_SCORE_NUM_TRAIN_TIMESTEPS)
        * (STAGE2_SCORE_TIMESTEP_SHIFT * u)
        / (1.0 + (STAGE2_SCORE_TIMESTEP_SHIFT - 1.0) * u)
    )
    future_timestep = shifted.clamp(
        min=STAGE2_SCORE_TIMESTEP_MIN,
        max=STAGE2_SCORE_TIMESTEP_MAX,
    )
    frame_timestep = torch.cat(
        (
            torch.zeros((raw.shape[0], 1), device=raw.device, dtype=torch.float32),
            future_timestep[:, None].expand(-1, STAGE2_SCORE_FUTURE_FRAMES),
        ),
        dim=1,
    )
    sigma = frame_timestep / float(STAGE2_SCORE_NUM_TRAIN_TIMESTEPS)
    return Stage2ScoreTimesteps(
        uniform_integer=raw,
        future_timestep=future_timestep,
        frame_timestep=frame_timestep,
        sigma=sigma,
    )


def sample_stage2_score_timesteps(
    *,
    batch_size: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> Stage2ScoreTimesteps:
    """Sample fresh per-sample integer score times on the requested device."""

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    device = torch.device(device)
    raw = torch.randint(
        0,
        STAGE2_SCORE_NUM_TRAIN_TIMESTEPS,
        (batch_size,),
        device=device,
        dtype=torch.int64,
        generator=generator,
    )
    return stage2_score_timesteps_from_uniform_integers(raw)


def _future_epsilon(
    future: torch.Tensor,
    *,
    supplied: torch.Tensor | None,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if supplied is not None and generator is not None:
        raise ValueError("provide future_epsilon or generator, not both")
    if supplied is None:
        return torch.randn(
            tuple(future.shape),
            device=future.device,
            dtype=torch.float32,
            generator=generator,
        )
    _require_future(supplied, "future_epsilon")
    if tuple(supplied.shape) != tuple(future.shape):
        raise ValueError(
            "future_epsilon shape must exactly match the 24-frame score future"
        )
    if supplied.device != future.device:
        raise ValueError("future_epsilon and score input devices differ")
    return supplied.float()


def noise_stage2_score_input(
    clean_score: torch.Tensor,
    frame_timestep: torch.Tensor,
    *,
    future_epsilon: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Stage2NoisedScoreInput:
    """Apply ``x_t=(1-sigma)x0+sigma*epsilon`` in FP32.

    The sink is concatenated back from the clean input rather than recovered by
    arithmetic, so it stays bit-exact even when future score time is near an
    endpoint.  The full epsilon tensor contains an exact zero sink and a fresh
    elementwise-iid future.
    """

    _require_score_input(clean_score, "clean_score")
    frame_timestep_fp32 = validate_stage2_frame_timestep(
        frame_timestep,
        batch_size=int(clean_score.shape[0]),
        device=clean_score.device,
    )
    clean_fp32 = clean_score.float()
    future = clean_fp32[:, 1:]
    epsilon_future = _future_epsilon(
        future,
        supplied=future_epsilon,
        generator=generator,
    )
    sigma = frame_timestep_fp32 / float(STAGE2_SCORE_NUM_TRAIN_TIMESTEPS)
    sigma_future = sigma[:, 1:].view(
        int(clean_score.shape[0]),
        STAGE2_SCORE_FUTURE_FRAMES,
        1,
        1,
        1,
    )
    noisy_future = (1.0 - sigma_future) * future + sigma_future * epsilon_future
    noisy_score = torch.cat((clean_fp32[:, :1], noisy_future), dim=1)
    epsilon = torch.cat((torch.zeros_like(clean_fp32[:, :1]), epsilon_future), dim=1)
    _require_finite(noisy_score, "noisy_score")
    return Stage2NoisedScoreInput(
        noisy_score=noisy_score,
        epsilon=epsilon,
        future_epsilon=epsilon_future,
        frame_timestep=frame_timestep_fp32,
        sigma=sigma,
    )


def noise_stage2_score_pair(
    fake_clean_score: torch.Tensor,
    real_clean_score: torch.Tensor,
    frame_timestep: torch.Tensor,
    *,
    future_epsilon: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Stage2NoisedScorePair:
    """Construct the DFD fake/real teacher pair with one shared ``(t, epsilon)``."""

    _require_score_input(fake_clean_score, "fake_clean_score")
    _require_score_input(real_clean_score, "real_clean_score")
    if tuple(fake_clean_score.shape) != tuple(real_clean_score.shape):
        raise ValueError("DFD fake and real score inputs must have the same shape")
    if fake_clean_score.device != real_clean_score.device:
        raise ValueError("DFD fake and real score inputs must use the same device")
    if fake_clean_score.dtype != real_clean_score.dtype:
        raise TypeError("DFD fake and real score inputs must use the same dtype")
    if not bool(
        torch.equal(
            fake_clean_score[:, :1].float(),
            real_clean_score[:, :1].float(),
        )
    ):
        raise ValueError(
            "DFD fake and real inputs must use the same exact initial sink"
        )

    shared_epsilon = _future_epsilon(
        fake_clean_score[:, 1:],
        supplied=future_epsilon,
        generator=generator,
    )
    fake = noise_stage2_score_input(
        fake_clean_score,
        frame_timestep,
        future_epsilon=shared_epsilon,
    )
    real = noise_stage2_score_input(
        real_clean_score,
        frame_timestep,
        future_epsilon=shared_epsilon,
    )
    if not bool(torch.equal(fake.future_epsilon, real.future_epsilon)):
        raise AssertionError("DFD shared future epsilon drifted")
    return Stage2NoisedScorePair(fake=fake, real=real)


def validate_stage2_noised_score_input(
    value: Stage2NoisedScoreInput,
) -> Stage2NoisedScoreInput:
    """Revalidate a noising bundle before a model/loss boundary consumes it."""

    if not isinstance(value, Stage2NoisedScoreInput):
        raise TypeError("score noising input must be Stage2NoisedScoreInput")
    _require_score_input(value.noisy_score, "noised_score.noisy_score")
    if value.noisy_score.dtype != torch.float32:
        raise TypeError("Stage-2 noised score tensor must be FP32")
    batch_size = int(value.noisy_score.shape[0])
    device = value.noisy_score.device
    frame_timestep = validate_stage2_frame_timestep(
        value.frame_timestep,
        batch_size=batch_size,
        device=device,
    )
    if value.frame_timestep.dtype != torch.float32:
        raise TypeError("Stage-2 score frame_timestep must be FP32")

    _require_score_input(value.epsilon, "noised_score.epsilon")
    _require_future(value.future_epsilon, "noised_score.future_epsilon")
    expected_shape = tuple(value.noisy_score.shape)
    if tuple(value.epsilon.shape) != expected_shape:
        raise ValueError("noised score epsilon shape differs from noisy score")
    if tuple(value.future_epsilon.shape) != (
        batch_size,
        STAGE2_SCORE_FUTURE_FRAMES,
        *expected_shape[2:],
    ):
        raise ValueError("noised score future epsilon has the wrong shape")
    for name, tensor in (
        ("epsilon", value.epsilon),
        ("future_epsilon", value.future_epsilon),
    ):
        if tensor.device != device:
            raise ValueError(f"noised score {name} device differs from noisy score")
        if tensor.dtype != torch.float32:
            raise TypeError(f"Stage-2 noised score {name} must be FP32")
    if not bool(
        torch.equal(value.epsilon[:, :1], torch.zeros_like(value.epsilon[:, :1]))
    ):
        raise ValueError("Stage-2 noised score sink epsilon must be exactly zero")
    if not bool(torch.equal(value.epsilon[:, 1:], value.future_epsilon)):
        raise ValueError("noised score epsilon and future_epsilon disagree")

    _require_floating_tensor(value.sigma, "noised_score.sigma")
    if tuple(value.sigma.shape) != (batch_size, STAGE2_SCORE_INPUT_FRAMES):
        raise ValueError("noised score sigma must have shape [B,25]")
    if value.sigma.device != device:
        raise ValueError("noised score sigma device differs from noisy score")
    if value.sigma.dtype != torch.float32:
        raise TypeError("Stage-2 noised score sigma must be FP32")
    _require_finite(value.sigma, "noised_score.sigma")
    if not bool(
        torch.equal(
            value.sigma,
            frame_timestep / float(STAGE2_SCORE_NUM_TRAIN_TIMESTEPS),
        )
    ):
        raise ValueError("noised score sigma does not equal continuous t/1000")
    return value


def validate_stage2_noised_score_pair(
    value: Stage2NoisedScorePair,
) -> Stage2NoisedScorePair:
    """Recheck DFD's exact shared time/noise/sink contract at call time."""

    if not isinstance(value, Stage2NoisedScorePair):
        raise TypeError("DFD noising input must be Stage2NoisedScorePair")
    fake = validate_stage2_noised_score_input(value.fake)
    real = validate_stage2_noised_score_input(value.real)
    if tuple(fake.noisy_score.shape) != tuple(real.noisy_score.shape):
        raise ValueError("DFD noised fake/real score shapes differ")
    if fake.noisy_score.device != real.noisy_score.device:
        raise ValueError("DFD noised fake/real score devices differ")
    if not bool(torch.equal(fake.frame_timestep, real.frame_timestep)):
        raise ValueError("DFD fake/real inputs must share exact frame timestep")
    if not bool(torch.equal(fake.sigma, real.sigma)):
        raise ValueError("DFD fake/real inputs must share exact continuous sigma")
    if not bool(torch.equal(fake.epsilon, real.epsilon)) or not bool(
        torch.equal(fake.future_epsilon, real.future_epsilon)
    ):
        raise ValueError("DFD fake/real inputs must share exact future epsilon")
    if not bool(torch.equal(fake.noisy_score[:, :1], real.noisy_score[:, :1])):
        raise ValueError("DFD noised fake/real inputs must share the exact sink")
    return value


def stage2_flow_to_x0(
    noisy_score: torch.Tensor,
    raw_flow: torch.Tensor,
    frame_timestep: torch.Tensor,
) -> torch.Tensor:
    """Convert native flow to ``x0`` with the same continuous FP32 sigma."""

    _require_score_input(noisy_score, "noisy_score")
    _require_score_input(raw_flow, "raw_flow")
    if tuple(noisy_score.shape) != tuple(raw_flow.shape):
        raise ValueError("noisy_score and raw_flow shapes differ")
    if noisy_score.device != raw_flow.device:
        raise ValueError("noisy_score and raw_flow devices differ")
    frame_timestep_fp32 = validate_stage2_frame_timestep(
        frame_timestep,
        batch_size=int(noisy_score.shape[0]),
        device=noisy_score.device,
    )
    sigma_future = (frame_timestep_fp32[:, 1:] / 1000.0).view(
        int(noisy_score.shape[0]),
        STAGE2_SCORE_FUTURE_FRAMES,
        1,
        1,
        1,
    )
    future_x0 = noisy_score[:, 1:].float() - sigma_future * raw_flow[:, 1:].float()
    x0 = torch.cat((noisy_score[:, :1].float(), future_x0), dim=1)
    _require_finite(x0, "x0_prediction")
    return x0


def stage2_real_cfg_flow(
    *,
    cond_flow: torch.Tensor,
    uncond_flow: torch.Tensor,
) -> torch.Tensor:
    """Apply unambiguous standard CFG5 to real-score native flow."""

    _require_floating_tensor(cond_flow, "cond_flow")
    _require_floating_tensor(uncond_flow, "uncond_flow")
    if tuple(cond_flow.shape) != tuple(uncond_flow.shape):
        raise ValueError("real-score cond/uncond flow shapes differ")
    if cond_flow.device != uncond_flow.device:
        raise ValueError("real-score cond/uncond flow devices differ")
    _require_finite(cond_flow, "cond_flow")
    _require_finite(uncond_flow, "uncond_flow")
    cond_fp32 = cond_flow.float()
    uncond_fp32 = uncond_flow.float()
    cfg = uncond_fp32 + STAGE2_SCORE_REAL_CFG_SCALE * (cond_fp32 - uncond_fp32)
    _require_finite(cfg, "real_cfg5_flow")
    return cfg
