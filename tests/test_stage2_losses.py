from __future__ import annotations

from dataclasses import replace
import subprocess
import sys

import pytest
import torch
from torch import nn

from model.stage2_dmd import (
    Stage2DMD,
    Stage2DiTRole,
    compute_stage2_fake_score_flow_dsm_loss,
    compute_stage2_generator_distribution_matching_loss,
)
from utils.stage2_score_math import (
    STAGE2_SCORE_DENOMINATOR_CLAMP,
    STAGE2_SCORE_REAL_CFG_SCALE,
    noise_stage2_score_input,
    noise_stage2_score_pair,
    sample_stage2_score_timesteps,
    stage2_flow_to_x0,
    stage2_real_cfg_flow,
    stage2_score_timesteps_from_uniform_integers,
)


def _frame_timestep(values: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        (torch.zeros_like(values[:, None]), values[:, None].repeat(1, 24)), dim=1
    )


def _score_input(future: torch.Tensor, *, sink_value: float = 0.0) -> torch.Tensor:
    sink = torch.full_like(future[:, :1], sink_value)
    return torch.cat((sink, future), dim=1)


def _raw_flow_for_x0(
    noisy_score: torch.Tensor,
    desired_x0: torch.Tensor,
    frame_timestep: torch.Tensor,
) -> torch.Tensor:
    sigma = (frame_timestep.float() / 1000.0).view(
        frame_timestep.shape[0], frame_timestep.shape[1], 1, 1, 1
    )
    flow = torch.zeros_like(noisy_score, dtype=torch.float32)
    flow[:, 1:] = (noisy_score[:, 1:].float() - desired_x0[:, 1:].float()) / sigma[
        :, 1:
    ]
    return flow


class _RecordingScoreRole(Stage2DiTRole):
    def __init__(self, *, role: str, raw_flow_value: float):
        super().__init__(nn.Linear(1, 1, bias=False), role=role, is_causal=False)
        with torch.no_grad():
            self.model.weight.fill_(raw_flow_value)
        self.calls: list[dict[str, object]] = []

    def forward_score(
        self,
        *,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict[str, torch.Tensor],
        frame_timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(
            {
                "noisy": noisy_image_or_video,
                "conditioning": conditional_dict,
                "frame_timestep": frame_timestep,
                "grad_enabled": torch.is_grad_enabled(),
            }
        )
        raw_flow = torch.ones_like(noisy_image_or_video) * self.model.weight[0, 0]
        return raw_flow, noisy_image_or_video.float()


class _BF16PatchEmbeddingScoreSpy(nn.Module):
    patch_size = (1, 2, 2)

    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv3d(
            48,
            48,
            kernel_size=1,
            groups=48,
            bias=False,
            dtype=torch.bfloat16,
        )
        self.input_dtype: torch.dtype | None = None

    def forward(self, latent, *, t, context, seq_len):
        del t, context, seq_len
        self.input_dtype = latent.dtype
        return self.patch_embedding(latent)


class _BF16CausalFlowSpy(nn.Module):
    patch_size = (1, 2, 2)

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(4.0, dtype=torch.bfloat16))

    def forward(
        self,
        latent,
        *,
        t,
        context,
        seq_len,
        kv_cache,
        crossattn_cache,
        current_start,
        cache_start,
    ):
        del t, context, seq_len, kv_cache, crossattn_cache, current_start, cache_start
        return torch.ones_like(latent) * self.anchor


def test_shifted_integer_timestep_formula_clamps_edges_and_builds_video_global_time():
    raw = torch.tensor([0, 1, 500, 999], dtype=torch.int64)
    sample = stage2_score_timesteps_from_uniform_integers(raw)
    u = raw.float() / 1000.0
    expected = (1000.0 * (5.0 * u) / (1.0 + 4.0 * u)).clamp(20.0, 980.0)

    assert sample.uniform_integer.dtype == torch.int64
    assert torch.equal(sample.uniform_integer, raw)
    assert torch.equal(sample.future_timestep, expected)
    assert sample.frame_timestep.shape == (4, 25)
    assert torch.count_nonzero(sample.frame_timestep[:, 0]).item() == 0
    assert torch.equal(sample.frame_timestep[:, 1:], expected[:, None].expand(-1, 24))
    assert sample.sigma.dtype == torch.float32
    assert torch.equal(sample.sigma, sample.frame_timestep / 1000.0)
    assert sample.future_timestep[0].item() == 20.0
    assert sample.future_timestep[-1].item() == 980.0


def test_score_timestep_sampling_is_seeded_integer_sampling_without_broadcasting_samples():
    left_generator = torch.Generator().manual_seed(1234)
    right_generator = torch.Generator().manual_seed(1234)
    left = sample_stage2_score_timesteps(
        batch_size=32, device="cpu", generator=left_generator
    )
    right = sample_stage2_score_timesteps(
        batch_size=32, device="cpu", generator=right_generator
    )
    assert torch.equal(left.uniform_integer, right.uniform_integer)
    assert torch.unique(left.uniform_integer).numel() > 1
    assert bool(((left.uniform_integer >= 0) & (left.uniform_integer < 1000)).all())


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        (torch.tensor([0.0]), "integer"),
        (torch.tensor([-1]), r"\[0, 1000\)"),
        (torch.tensor([1000]), r"\[0, 1000\)"),
        (torch.zeros(1, 1, dtype=torch.int64), "one-dimensional"),
    ],
)
def test_score_timestep_integer_contract_fails_closed(raw, match):
    with pytest.raises((TypeError, ValueError), match=match):
        stage2_score_timesteps_from_uniform_integers(raw)


def test_score_noising_uses_fp32_continuous_sigma_and_keeps_sink_exact():
    clean_future = torch.tensor([[[[[2.0]]], [[[4.0]]]]], dtype=torch.bfloat16).repeat(
        1, 12, 1, 1, 1
    )
    clean = _score_input(clean_future, sink_value=7.0)
    future_epsilon = torch.arange(24, dtype=torch.float32).view(1, 24, 1, 1, 1)
    frame_timestep = _frame_timestep(torch.tensor([500.0]))

    noised = noise_stage2_score_input(
        clean,
        frame_timestep,
        future_epsilon=future_epsilon,
    )
    expected_future = 0.5 * clean_future.float() + 0.5 * future_epsilon
    assert noised.noisy_score.dtype == torch.float32
    assert torch.equal(noised.noisy_score[:, :1], clean[:, :1].float())
    assert torch.equal(noised.epsilon[:, :1], torch.zeros_like(noised.epsilon[:, :1]))
    assert torch.equal(noised.epsilon[:, 1:], future_epsilon)
    assert torch.equal(noised.noisy_score[:, 1:], expected_future)
    assert torch.equal(noised.sigma[:, 1:], torch.full((1, 24), 0.5))


def test_random_future_noise_is_elementwise_not_one_frame_repeated_over_time():
    clean = torch.zeros(2, 25, 3, 2, 2)
    frame_timestep = _frame_timestep(torch.tensor([400.0, 700.0]))
    result = noise_stage2_score_input(
        clean,
        frame_timestep,
        generator=torch.Generator().manual_seed(9),
    )
    assert result.future_epsilon.shape == (2, 24, 3, 2, 2)
    assert not torch.equal(result.future_epsilon[:, 0], result.future_epsilon[:, 1])
    assert not torch.equal(result.future_epsilon[0], result.future_epsilon[1])


def test_score_role_casts_fp32_noising_only_for_real_bf16_patch_embedding():
    clean = torch.zeros(1, 25, 48, 30, 52)
    frame_timestep = _frame_timestep(torch.tensor([500.0]))
    noised = noise_stage2_score_input(
        clean,
        frame_timestep,
        future_epsilon=torch.ones(1, 24, 48, 30, 52),
    )
    assert noised.noisy_score.dtype == torch.float32
    spy = _BF16PatchEmbeddingScoreSpy()
    role = Stage2DiTRole(spy, role="fake_score", is_causal=False)

    raw_flow, x0_pred = role.forward_score(
        noisy_image_or_video=noised.noisy_score,
        conditional_dict={"prompt_embeds": torch.zeros(1, 2, 4, dtype=torch.bfloat16)},
        frame_timestep=frame_timestep,
    )

    assert spy.input_dtype == torch.bfloat16
    assert raw_flow.dtype == torch.bfloat16
    assert x0_pred.dtype == torch.float32
    expected = (
        noised.noisy_score.float()
        - (frame_timestep.view(1, 25, 1, 1, 1) / 1000.0) * raw_flow.float()
    )
    assert torch.equal(x0_pred, expected)
    assert torch.equal(x0_pred[:, :1], noised.noisy_score[:, :1])


def test_generator_x0_uses_explicit_unipc_sigma_not_integer_t_over_1000():
    role = Stage2DiTRole(_BF16CausalFlowSpy(), role="generator", is_causal=True)
    latent = torch.full((1, 1, 48, 30, 52), 3.0, dtype=torch.bfloat16)
    timestep = torch.full((1, 1), 999.0)
    flow_sigma = torch.tensor(0.9997999668121338)
    kwargs = dict(
        noisy_image_or_video=latent,
        conditional_dict={"prompt_embeds": torch.zeros(1, 2, 4, dtype=torch.bfloat16)},
        timestep=timestep,
        kv_cache=[{}],
        crossattn_cache=[{}],
        current_start=0,
        cache_start=0,
    )

    raw_flow, x0_pred = role(flow_sigma=flow_sigma, **kwargs)

    expected = latent.float() - flow_sigma * raw_flow.float()
    wrong_integer_sigma = latent.float() - 0.999 * raw_flow.float()
    assert x0_pred.dtype == latent.dtype
    assert torch.equal(x0_pred.float(), expected.to(latent.dtype).float())
    assert not torch.equal(
        x0_pred.float(), wrong_integer_sigma.to(latent.dtype).float()
    )
    with pytest.raises(ValueError, match="requires explicit UniPC flow_sigma"):
        role(**kwargs)

    score_role = Stage2DiTRole(
        _BF16PatchEmbeddingScoreSpy(), role="fake_score", is_causal=False
    )
    with pytest.raises(ValueError, match="score roles forbid flow_sigma"):
        score_role(
            noisy_image_or_video=torch.zeros(1, 25, 48, 30, 52),
            conditional_dict={"prompt_embeds": torch.zeros(1, 2, 4)},
            timestep=_frame_timestep(torch.tensor([500.0])),
            flow_sigma=0.5,
        )


def test_dfd_pair_uses_the_exact_same_future_noise_and_requires_the_same_sink():
    fake = _score_input(torch.full((1, 24, 1, 1, 1), 2.0), sink_value=11.0)
    real = _score_input(torch.full((1, 24, 1, 1, 1), 10.0), sink_value=11.0)
    epsilon = torch.arange(24, dtype=torch.float32).view(1, 24, 1, 1, 1)
    frame_timestep = _frame_timestep(torch.tensor([500.0]))
    pair = noise_stage2_score_pair(
        fake,
        real,
        frame_timestep,
        future_epsilon=epsilon,
    )

    assert torch.equal(pair.fake.epsilon, pair.real.epsilon)
    assert torch.equal(pair.fake.noisy_score[:, :1], fake[:, :1].float())
    assert torch.equal(pair.real.noisy_score[:, :1], real[:, :1].float())
    assert torch.equal(
        pair.real.noisy_score[:, 1:] - pair.fake.noisy_score[:, 1:],
        0.5 * (real[:, 1:].float() - fake[:, 1:].float()),
    )

    wrong_sink = real.clone()
    wrong_sink[:, 0] += 1
    with pytest.raises(ValueError, match="same exact initial sink"):
        noise_stage2_score_pair(
            fake, wrong_sink, frame_timestep, future_epsilon=epsilon
        )


@pytest.mark.parametrize("future_timestep", [20.0, 980.0])
def test_continuous_flow_to_x0_is_exact_at_score_timestep_boundaries(future_timestep):
    clean = _score_input(torch.full((1, 24, 1, 1, 1), 3.0), sink_value=13.0)
    epsilon = torch.full((1, 24, 1, 1, 1), 9.0)
    frame_timestep = _frame_timestep(torch.tensor([future_timestep]))
    noised = noise_stage2_score_input(clean, frame_timestep, future_epsilon=epsilon)
    target_flow = torch.cat(
        (torch.full_like(clean[:, :1], 999.0), epsilon - clean[:, 1:]), dim=1
    )
    recovered = stage2_flow_to_x0(
        noised.noisy_score,
        target_flow,
        frame_timestep,
    )
    assert recovered.dtype == torch.float32
    assert torch.allclose(recovered, clean.float(), atol=2e-5, rtol=0.0)


def test_real_cfg_is_standard_uncond_plus_five_times_cond_minus_uncond():
    uncond = torch.tensor([1.0, -2.0])
    cond = torch.tensor([4.0, 3.0])
    actual = stage2_real_cfg_flow(cond_flow=cond, uncond_flow=uncond)
    assert STAGE2_SCORE_REAL_CFG_SCALE == 5.0
    assert torch.equal(actual, uncond + 5.0 * (cond - uncond))
    assert not torch.equal(actual, cond + 5.0 * (cond - uncond))


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda clean, frame, eps: frame.__setitem__((slice(None), 0), 1.0),
            "sink timestep",
        ),
        (
            lambda clean, frame, eps: frame.__setitem__((slice(None), 2), 600.0),
            "video-global",
        ),
        (
            lambda clean, frame, eps: eps.__setitem__((slice(None), 0), float("nan")),
            "non-finite",
        ),
        (
            lambda clean, frame, eps: clean.__setitem__((slice(None), 2), float("inf")),
            "non-finite",
        ),
    ],
)
def test_score_noising_rejects_invalid_sink_time_global_time_noise_and_clean_values(
    mutation, match
):
    clean = torch.zeros(1, 25, 1, 1, 1)
    frame = _frame_timestep(torch.tensor([500.0]))
    epsilon = torch.zeros(1, 24, 1, 1, 1)
    mutation(clean, frame, epsilon)
    with pytest.raises(ValueError, match=match):
        noise_stage2_score_input(clean, frame, future_epsilon=epsilon)


def test_score_math_imports_no_discrete_scheduler_implementation():
    program = """
import sys
import utils.stage2_score_math
assert 'utils.scheduler' not in sys.modules
assert 'diffusers' not in sys.modules
print('continuous-score-math-only')
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "continuous-score-math-only"


def test_dmd_surrogate_uses_same_noisy_fake_input_cfg5_holistic_denom_and_only_g_grad():
    generated = torch.full((1, 24, 1, 1, 1), 2.0, requires_grad=True)
    clean_fake = _score_input(generated, sink_value=17.0)
    frame_timestep = _frame_timestep(torch.tensor([500.0]))
    noised = noise_stage2_score_input(
        clean_fake,
        frame_timestep,
        future_epsilon=torch.zeros_like(generated),
    )
    fake_x0 = _score_input(torch.full_like(generated, 5.0), sink_value=17.0)
    real_x0 = _score_input(torch.full_like(generated, 1.0), sink_value=17.0)
    fake_flow = (
        _raw_flow_for_x0(noised.noisy_score, fake_x0, frame_timestep)
        .detach()
        .requires_grad_()
    )
    real_flow = _raw_flow_for_x0(noised.noisy_score, real_x0, frame_timestep).detach()
    real_cond = real_flow.clone().requires_grad_()
    real_uncond = real_flow.clone().requires_grad_()
    with torch.no_grad():
        fake_flow[:, 0] = 1.0e6
        real_cond[:, 0] = -1.0e6
        real_uncond[:, 0] = 2.0e6

    output = compute_stage2_generator_distribution_matching_loss(
        branch="dmd",
        generated_future=generated,
        noisy_fake_score=noised.noisy_score,
        frame_timestep=frame_timestep,
        fake_raw_flow=fake_flow,
        real_cond_raw_flow=real_cond,
        real_uncond_raw_flow=real_uncond,
    )
    assert output.branch == "dmd"
    assert output.count == generated.numel()
    assert torch.equal(output.denominator, torch.tensor([1.0]))
    assert torch.equal(output.denominator_clamped, torch.tensor([1.0]))
    assert output.loss.item() == 8.0
    output.loss.backward()
    assert torch.allclose(
        generated.grad,
        torch.full_like(generated, 4.0 / generated.numel()),
        atol=1e-7,
        rtol=0.0,
    )
    assert fake_flow.grad is None
    assert real_cond.grad is None
    assert real_uncond.grad is None


def test_dfd_uses_real_teacher_input_with_shared_t_and_noise_not_fake_teacher_input():
    generated = torch.full((1, 24, 1, 1, 1), 2.0, requires_grad=True)
    real_future = torch.full_like(generated, 10.0)
    fake_clean = _score_input(generated, sink_value=19.0)
    real_clean = _score_input(real_future, sink_value=19.0)
    frame_timestep = _frame_timestep(torch.tensor([500.0]))
    epsilon = torch.full_like(generated, 3.0)
    pair = noise_stage2_score_pair(
        fake_clean,
        real_clean,
        frame_timestep,
        future_epsilon=epsilon,
    )
    fake_x0 = _score_input(torch.full_like(generated, 4.0), sink_value=19.0)
    real_x0 = _score_input(torch.full_like(generated, 8.0), sink_value=19.0)
    fake_flow = _raw_flow_for_x0(pair.fake.noisy_score, fake_x0, frame_timestep)
    real_flow = _raw_flow_for_x0(pair.real.noisy_score, real_x0, frame_timestep)

    output = compute_stage2_generator_distribution_matching_loss(
        branch="dfd",
        generated_future=generated,
        noisy_fake_score=pair.fake.noisy_score,
        noisy_real_score=pair.real.noisy_score,
        frame_timestep=frame_timestep,
        fake_raw_flow=fake_flow,
        real_cond_raw_flow=real_flow,
        real_uncond_raw_flow=real_flow,
    )
    assert output.branch == "dfd"
    assert torch.equal(output.denominator, torch.tensor([6.0]))
    output.loss.backward()
    assert torch.allclose(
        generated.grad,
        torch.full_like(generated, (-4.0 / 6.0) / generated.numel()),
        atol=1e-7,
        rtol=0.0,
    )


def test_generator_denom_is_one_scalar_per_video_and_zero_denom_uses_exact_clamp():
    generated = torch.cat(
        (
            torch.full((1, 24, 1, 1, 1), 2.0),
            torch.full((1, 24, 1, 1, 1), 6.0),
        ),
        dim=0,
    ).requires_grad_()
    frame_timestep = _frame_timestep(torch.tensor([500.0, 500.0]))
    noised = noise_stage2_score_input(
        _score_input(generated),
        frame_timestep,
        future_epsilon=torch.zeros_like(generated),
    )
    fake_x0 = _score_input(
        torch.cat(
            (torch.full_like(generated[:1], 3.0), torch.full_like(generated[1:], 9.0))
        )
    )
    real_x0 = _score_input(
        torch.cat(
            (torch.full_like(generated[:1], 2.0), torch.full_like(generated[1:], 4.0))
        )
    )
    fake_flow = _raw_flow_for_x0(noised.noisy_score, fake_x0, frame_timestep)
    real_flow = _raw_flow_for_x0(noised.noisy_score, real_x0, frame_timestep)
    output = compute_stage2_generator_distribution_matching_loss(
        branch="dmd",
        generated_future=generated,
        noisy_fake_score=noised.noisy_score,
        frame_timestep=frame_timestep,
        fake_raw_flow=fake_flow,
        real_cond_raw_flow=real_flow,
        real_uncond_raw_flow=real_flow,
    )
    assert STAGE2_SCORE_DENOMINATOR_CLAMP == 1.0e-6
    assert torch.equal(output.denominator, torch.tensor([0.0, 2.0]))
    assert torch.equal(
        output.denominator_clamped, torch.tensor([STAGE2_SCORE_DENOMINATOR_CLAMP, 2.0])
    )
    assert torch.isfinite(output.loss)


def test_generator_loss_branch_arguments_and_finite_predictions_fail_closed():
    generated = torch.zeros(1, 24, 1, 1, 1, requires_grad=True)
    score = torch.zeros(1, 25, 1, 1, 1)
    flow = torch.zeros_like(score)
    frame = _frame_timestep(torch.tensor([500.0]))
    common = dict(
        generated_future=generated,
        noisy_fake_score=score,
        frame_timestep=frame,
        fake_raw_flow=flow,
        real_cond_raw_flow=flow,
        real_uncond_raw_flow=flow,
    )
    with pytest.raises(ValueError, match="requires noisy_real_score"):
        compute_stage2_generator_distribution_matching_loss(branch="dfd", **common)
    with pytest.raises(ValueError, match="must not receive noisy_real_score"):
        compute_stage2_generator_distribution_matching_loss(
            branch="dmd", noisy_real_score=score, **common
        )
    bad = flow.clone()
    bad[:, 3] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        compute_stage2_generator_distribution_matching_loss(
            branch="dmd", **{**common, "fake_raw_flow": bad}
        )


def test_fake_score_dsm_uses_direct_raw_flow_future_only_and_detached_rollout():
    generated = torch.full((1, 24, 1, 1, 1), 2.0)
    epsilon = torch.full_like(generated, 5.0)
    raw_flow = torch.cat(
        (torch.full((1, 1, 1, 1, 1), 1.0e6), torch.full_like(generated, 4.0)),
        dim=1,
    ).requires_grad_()
    output = compute_stage2_fake_score_flow_dsm_loss(
        generated_future=generated,
        future_epsilon=epsilon,
        fake_raw_flow=raw_flow,
    )
    assert output.count == generated.numel()
    assert output.loss.item() == 1.0
    assert output.target_flow_l2.item() == 3.0
    output.loss.backward()
    assert raw_flow.grad[:, :1].count_nonzero().item() == 0
    assert torch.allclose(
        raw_flow.grad[:, 1:],
        torch.full_like(generated, 2.0 / generated.numel()),
        atol=1e-7,
        rtol=0.0,
    )

    with pytest.raises(ValueError, match="detached no-grad rollout"):
        compute_stage2_fake_score_flow_dsm_loss(
            generated_future=generated.requires_grad_(),
            future_epsilon=epsilon,
            fake_raw_flow=raw_flow.detach(),
        )


def test_model_level_dmd_calls_all_teachers_on_one_exact_noised_fake_object():
    generator_role = Stage2DiTRole(
        nn.Linear(1, 1, bias=False), role="generator", is_causal=True
    )
    with torch.no_grad():
        generator_role.model.weight.fill_(2.0)
    fake_role = _RecordingScoreRole(role="fake_score", raw_flow_value=4.0)
    real_role = _RecordingScoreRole(role="real_score", raw_flow_value=8.0)
    real_role.requires_grad_(False)
    model = Stage2DMD(
        generator=generator_role,
        real_score=real_role,
        fake_score=fake_role,
    )
    generated = torch.ones(1, 24, 1, 1, 1) * generator_role.model.weight[0, 0]
    frame_timestep = _frame_timestep(torch.tensor([500.0]))
    noised = noise_stage2_score_input(
        _score_input(generated, sink_value=21.0),
        frame_timestep,
        future_epsilon=torch.full_like(generated, 3.0),
    )
    positive = {"prompt_embeds": torch.tensor([1.0])}
    negative = {"prompt_embeds": torch.tensor([-1.0])}
    timed_labels = []

    def timing_callback(label, callback):
        timed_labels.append(label)
        return callback()

    output = model.generator_distribution_matching_loss_from_models(
        branch="dmd",
        generated_future=generated,
        noised_score=noised,
        conditional_dict=positive,
        real_unconditional_dict=negative,
        timing_callback=timing_callback,
    )

    calls = [fake_role.calls[0], *real_role.calls]
    assert len(fake_role.calls) == 1
    assert len(real_role.calls) == 2
    assert all(call["noisy"] is noised.noisy_score for call in calls)
    assert all(call["frame_timestep"] is noised.frame_timestep for call in calls)
    assert all(not call["grad_enabled"] for call in calls)
    assert timed_labels == ["fake_score", "real_cond", "real_uncond"]
    output.loss.backward()
    assert torch.isfinite(generator_role.model.weight.grad).all()
    assert fake_role.model.weight.grad is None
    assert real_role.model.weight.grad is None


def test_model_level_dfd_calls_teachers_no_grad_with_shared_objects_and_only_g_grad():
    generator_role = Stage2DiTRole(
        nn.Linear(1, 1, bias=False), role="generator", is_causal=True
    )
    with torch.no_grad():
        generator_role.model.weight.fill_(2.0)
    fake_role = _RecordingScoreRole(role="fake_score", raw_flow_value=4.0)
    real_role = _RecordingScoreRole(role="real_score", raw_flow_value=8.0)
    real_role.requires_grad_(False)
    model = Stage2DMD(
        generator=generator_role,
        real_score=real_role,
        fake_score=fake_role,
    )

    generated = torch.ones(1, 24, 1, 1, 1) * generator_role.model.weight[0, 0]
    fake_clean = _score_input(generated, sink_value=23.0)
    real_clean = _score_input(torch.full_like(generated, 10.0), sink_value=23.0)
    frame_timestep = _frame_timestep(torch.tensor([500.0]))
    pair = noise_stage2_score_pair(
        fake_clean,
        real_clean,
        frame_timestep,
        future_epsilon=torch.full_like(generated, 3.0),
    )
    positive = {"prompt_embeds": torch.tensor([1.0])}
    negative = {"prompt_embeds": torch.tensor([-1.0])}
    timed_labels = []

    def timing_callback(label, callback):
        timed_labels.append(label)
        return callback()

    output = model.generator_distribution_matching_loss_from_models(
        branch="dfd",
        generated_future=generated,
        noised_score=pair,
        conditional_dict=positive,
        real_unconditional_dict=negative,
        timing_callback=timing_callback,
    )

    assert len(fake_role.calls) == 1
    assert len(real_role.calls) == 2
    assert fake_role.calls[0]["noisy"] is pair.fake.noisy_score
    assert fake_role.calls[0]["conditioning"] is positive
    assert real_role.calls[0]["noisy"] is pair.real.noisy_score
    assert real_role.calls[1]["noisy"] is pair.real.noisy_score
    assert real_role.calls[0]["frame_timestep"] is frame_timestep
    assert real_role.calls[1]["frame_timestep"] is frame_timestep
    assert real_role.calls[0]["conditioning"] is positive
    assert real_role.calls[1]["conditioning"] is negative
    assert not fake_role.calls[0]["grad_enabled"]
    assert not real_role.calls[0]["grad_enabled"]
    assert not real_role.calls[1]["grad_enabled"]
    assert timed_labels == ["fake_score", "real_cond", "real_uncond"]

    output.loss.backward()
    assert torch.isfinite(generator_role.model.weight.grad).all()
    assert fake_role.model.weight.grad is None
    assert real_role.model.weight.grad is None


def test_model_level_dfd_revalidates_pair_shared_time_epsilon_and_sink():
    generator_role = Stage2DiTRole(
        nn.Linear(1, 1, bias=False), role="generator", is_causal=True
    )
    fake_role = _RecordingScoreRole(role="fake_score", raw_flow_value=4.0)
    real_role = _RecordingScoreRole(role="real_score", raw_flow_value=8.0)
    real_role.requires_grad_(False)
    model = Stage2DMD(
        generator=generator_role,
        real_score=real_role,
        fake_score=fake_role,
    )
    generated = torch.full((1, 24, 1, 1, 1), 2.0, requires_grad=True)
    frame_timestep = _frame_timestep(torch.tensor([500.0]))
    pair = noise_stage2_score_pair(
        _score_input(generated, sink_value=31.0),
        _score_input(torch.full_like(generated, 10.0), sink_value=31.0),
        frame_timestep,
        future_epsilon=torch.full_like(generated, 3.0),
    )
    common = dict(
        branch="dfd",
        generated_future=generated,
        conditional_dict={"prompt_embeds": torch.tensor([1.0])},
        real_unconditional_dict={"prompt_embeds": torch.tensor([-1.0])},
    )

    different_t = pair.real.frame_timestep.clone()
    different_t[:, 1:] = 600.0
    bad_time = replace(
        pair,
        real=replace(pair.real, frame_timestep=different_t, sigma=different_t / 1000.0),
    )
    with pytest.raises(ValueError, match="share exact frame timestep"):
        model.generator_distribution_matching_loss_from_models(
            noised_score=bad_time, **common
        )

    different_future_epsilon = pair.real.future_epsilon.clone()
    different_future_epsilon[:, 0] += 1.0
    different_epsilon = pair.real.epsilon.clone()
    different_epsilon[:, 1:] = different_future_epsilon
    bad_noise = replace(
        pair,
        real=replace(
            pair.real,
            epsilon=different_epsilon,
            future_epsilon=different_future_epsilon,
        ),
    )
    with pytest.raises(ValueError, match="share exact future epsilon"):
        model.generator_distribution_matching_loss_from_models(
            noised_score=bad_noise, **common
        )

    different_sink = pair.real.noisy_score.clone()
    different_sink[:, :1] += 1.0
    bad_sink = replace(pair, real=replace(pair.real, noisy_score=different_sink))
    with pytest.raises(ValueError, match="share the exact sink"):
        model.generator_distribution_matching_loss_from_models(
            noised_score=bad_sink, **common
        )
    assert len(fake_role.calls) == 0
    assert len(real_role.calls) == 0


def test_model_level_fake_dsm_calls_only_fake_score_and_only_f_gets_grad():
    generator_role = Stage2DiTRole(
        nn.Linear(1, 1, bias=False), role="generator", is_causal=True
    )
    fake_role = _RecordingScoreRole(role="fake_score", raw_flow_value=4.0)
    real_role = _RecordingScoreRole(role="real_score", raw_flow_value=8.0)
    generator_role.requires_grad_(False)
    real_role.requires_grad_(False)
    model = Stage2DMD(
        generator=generator_role,
        real_score=real_role,
        fake_score=fake_role,
    )
    generated = torch.full((1, 24, 1, 1, 1), 2.0)
    frame_timestep = _frame_timestep(torch.tensor([500.0]))
    noised = noise_stage2_score_input(
        _score_input(generated, sink_value=29.0),
        frame_timestep,
        future_epsilon=torch.full_like(generated, 5.0),
    )
    positive = {"prompt_embeds": torch.tensor([1.0])}
    timed_labels = []

    def timing_callback(label, callback):
        timed_labels.append(label)
        return callback()

    output = model.fake_score_flow_dsm_loss_from_model(
        generated_future=generated,
        noised_fake_score=noised,
        conditional_dict=positive,
        timing_callback=timing_callback,
    )

    assert output.loss.item() == 1.0
    assert len(fake_role.calls) == 1
    assert len(real_role.calls) == 0
    assert fake_role.calls[0]["noisy"] is noised.noisy_score
    assert fake_role.calls[0]["frame_timestep"] is frame_timestep
    assert fake_role.calls[0]["conditioning"] is positive
    assert fake_role.calls[0]["grad_enabled"]
    assert timed_labels == ["fake_score"]
    output.loss.backward()
    assert torch.isfinite(fake_role.model.weight.grad).all()
    assert generator_role.model.weight.grad is None
    assert real_role.model.weight.grad is None


def test_stage2_model_method_requires_only_real_teacher_permanently_frozen():
    generator_role = Stage2DiTRole(nn.Linear(1, 1), role="generator", is_causal=True)
    real_role = Stage2DiTRole(nn.Linear(1, 1), role="real_score", is_causal=False)
    fake_role = Stage2DiTRole(nn.Linear(1, 1), role="fake_score", is_causal=False)
    model = Stage2DMD(
        generator=generator_role,
        real_score=real_role,
        fake_score=fake_role,
    )
    generated = torch.zeros(1, 24, 1, 1, 1, requires_grad=True)
    score = torch.zeros(1, 25, 1, 1, 1)
    frame = _frame_timestep(torch.tensor([500.0]))
    kwargs = dict(
        branch="dmd",
        generated_future=generated,
        noisy_fake_score=score,
        frame_timestep=frame,
        fake_raw_flow=score,
        real_cond_raw_flow=score,
        real_uncond_raw_flow=score,
    )
    with pytest.raises(RuntimeError, match="real_score must remain frozen"):
        model.generator_distribution_matching_loss(**kwargs)
    real_role.requires_grad_(False)
    output = model.generator_distribution_matching_loss(**kwargs)
    assert torch.isfinite(output.loss)
    assert any(parameter.requires_grad for parameter in fake_role.parameters())
