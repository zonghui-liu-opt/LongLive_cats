from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import wan_5b.modules.causal_model as causal_model_module
from pipeline.stage2_rollout import (
    STAGE2_K2_SHIFT5_TIMESTEPS,
    STAGE2_K4_SHIFT5_TIMESTEPS,
    Stage2ExitRNGStreams,
    Stage2RolloutPipeline,
    Stage2SinkPrefixSnapshot,
    draw_stage2_exit_schedule,
)
from utils.stage2_config import load_stage2_config
from utils.stage2_inference import tensor_identity_sha256
from wan_5b.modules.causal_model import (
    CausalWanModel,
    CausalWanSelfAttention,
    MultiShotT2VCrossAttention,
)

FRAME_TOKENS = 390


class _FakeScheduler:
    def __init__(self):
        self.timesteps = torch.empty(0)
        self.sigmas = torch.empty(0)
        self.steps = []

    def set_timesteps(self, count, *, device, shift):
        assert shift == 5.0
        if count == 4:
            timetable = STAGE2_K4_SHIFT5_TIMESTEPS
            sigmas = [1.0, 0.937, 0.833, 0.624, 0.0]
        elif count == 2:
            timetable = STAGE2_K2_SHIFT5_TIMESTEPS
            sigmas = [1.0, 0.833, 0.0]
        else:
            raise AssertionError(count)
        self.timesteps = torch.tensor(timetable, dtype=torch.float32, device=device)
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32, device=device)

    def step(self, flow, timestep, sample, *, return_dict):
        assert return_dict is False
        self.steps.append(int(float(timestep)))
        return (sample - 0.01 * flow,)


class _TinyTransformer(nn.Module):
    def __init__(self, *, checkpointing=False):
        super().__init__()
        self.num_layers = 2
        self.num_heads = 2
        self.dim = 4
        self.text_len = 5
        self.local_attn_size = 16
        self.sink_size = 0
        self.global_sink_size = 0
        self.max_attention_size = 0
        self.is_gradient_checkpointing = checkpointing


class _FakeGenerator(nn.Module):
    def __init__(self, *, checkpointing=False):
        super().__init__()
        self.model = _TinyTransformer(checkpointing=checkpointing)
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.calls = []

    @staticmethod
    def _cursor(cache):
        return (
            int(cache["global_end_index"].item()),
            int(cache["local_end_index"].item()),
        )

    def forward(
        self,
        *,
        noisy_image_or_video,
        conditional_dict,
        timestep,
        kv_cache,
        crossattn_cache,
        current_start,
        cache_start,
        commit_self_kv,
        flow_sigma,
    ):
        assert cache_start == current_start
        before = self._cursor(kv_cache[0])
        self.calls.append(
            {
                "frames": noisy_image_or_video.shape[1],
                "timestep": tuple(float(value) for value in timestep[0].tolist()),
                "current_start": current_start,
                "commit": commit_self_kv,
                "before": before,
                "prompt": float(conditional_dict["prompt_embeds"].flatten()[0]),
                "flow_sigma": float(torch.as_tensor(flow_sigma).detach().cpu()),
                "runtime_local_attn_size": self.model.local_attn_size,
                "runtime_max_attention_size": self.model.max_attention_size,
                "runtime_sink_size": self.model.sink_size,
                "runtime_global_sink_size": self.model.global_sink_size,
            }
        )
        flow = noisy_image_or_video * self.scale
        sigma = torch.as_tensor(
            flow_sigma, dtype=torch.float32, device=noisy_image_or_video.device
        )
        while sigma.ndim < flow.ndim:
            sigma = sigma.unsqueeze(-1)
        x0 = noisy_image_or_video.float() - sigma * flow.float()
        x0 = x0.to(noisy_image_or_video.dtype)
        if commit_self_kv:
            token_count = noisy_image_or_video.shape[1] * FRAME_TOKENS
            current_end = current_start + token_count
            for cache in kv_cache:
                capacity = cache["k"].shape[1]
                old_local = int(cache["local_end_index"].item())
                marker = (
                    noisy_image_or_video.detach().float().mean().to(cache["k"].dtype)
                )
                if old_local + token_count <= capacity:
                    local_start = old_local
                    local_end = old_local + token_count
                else:
                    sink = self.model.global_sink_size * FRAME_TOKENS
                    history = capacity - sink - token_count
                    cache["k"][:, sink : sink + history].copy_(
                        cache["k"][:, old_local - history : old_local].clone()
                    )
                    cache["v"][:, sink : sink + history].copy_(
                        cache["v"][:, old_local - history : old_local].clone()
                    )
                    local_start = sink + history
                    local_end = capacity
                cache["k"][:, local_start:local_end].fill_(marker)
                cache["v"][:, local_start:local_end].fill_(marker + 1)
                cache["global_end_index"].fill_(current_end)
                cache["local_end_index"].fill_(local_end)
            for cache in crossattn_cache:
                cache["k"].fill_(float(conditional_dict["prompt_embeds"].flatten()[0]))
                cache["v"].fill_(2)
                cache["is_init"] = True
        return flow, x0


def _pipeline(generator=None, schedulers=None, *, spec=None):
    generator = generator or _FakeGenerator()
    schedulers = [] if schedulers is None else schedulers

    def factory():
        scheduler = _FakeScheduler()
        schedulers.append(scheduler)
        return scheduler

    return (
        Stage2RolloutPipeline(generator, spec=spec, scheduler_factory=factory),
        schedulers,
    )


def _inputs(*, prompt=3.0, batch=2):
    initial = torch.full((batch, 1, 1, 2, 2), 7.0, dtype=torch.bfloat16)
    noise = torch.linspace(0.1, 2.4, 24).view(1, 24, 1, 1, 1)
    noise = noise.expand(batch, -1, -1, 2, 2).clone().to(torch.bfloat16)
    conditioning = {
        "prompt_embeds": torch.full((batch, 2, 3), prompt, dtype=torch.bfloat16)
    }
    return initial, noise, conditioning


def test_real_unipc_k4_shift5_is_the_single_runtime_timetable():
    pipeline = Stage2RolloutPipeline(_FakeGenerator())
    scheduler, timetable = pipeline._new_scheduler(torch.device("cpu"))
    assert timetable == STAGE2_K4_SHIFT5_TIMESTEPS
    assert float(scheduler.sigmas[-1]) == 0.0
    # UniPC's exact sigma is the solver source of truth; integer display
    # timesteps are not a lossless substitute for x0 reconstruction.
    assert float(scheduler.sigmas[0]) != timetable[0] / 1000.0


@pytest.mark.parametrize(
    ("profile_name", "expected_timetable"),
    [
        ("baseline_c8w16k4s1", STAGE2_K4_SHIFT5_TIMESTEPS),
        ("c4w8k2s1", STAGE2_K2_SHIFT5_TIMESTEPS),
    ],
)
def test_real_unipc_native_timetable_and_terminal_fp32_x0_identity(
    profile_name, expected_timetable
):
    pipeline = Stage2RolloutPipeline(_FakeGenerator(), spec=profile_name)
    scheduler, timetable = pipeline._new_scheduler(torch.device("cpu"))
    assert timetable == expected_timetable
    assert len(scheduler.sigmas) == len(timetable) + 1
    assert float(scheduler.sigmas[-1]) == 0.0

    sample = torch.randn(2, 3, generator=torch.Generator().manual_seed(17))
    terminal_x0 = None
    terminal_step = None
    for timestep, sigma in zip(scheduler.timesteps, scheduler.sigmas[:-1]):
        flow = sample * 0.25 + 0.1
        terminal_x0 = sample - sigma * flow
        terminal_step = scheduler.step(flow, timestep, sample, return_dict=False)[0]
        sample = terminal_step
    assert terminal_x0 is not None and terminal_step is not None
    assert torch.equal(terminal_step, terminal_x0)


def test_rollout_contract_is_built_only_from_the_resolved_stage2_config():
    resolved = load_stage2_config("configs/train_i2v_stage2_600cats.yaml")
    schedulers = []

    def factory():
        scheduler = _FakeScheduler()
        schedulers.append(scheduler)
        return scheduler

    pipeline = Stage2RolloutPipeline.from_resolved_config(
        _FakeGenerator(), resolved, scheduler_factory=factory
    )
    assert pipeline.num_chunks == 3
    assert pipeline.history_frames == 8
    assert pipeline.physical_kv_capacity_frames == 17
    assert pipeline.cache_capacity_tokens == 17 * 390
    assert pipeline.frame_seq_length == resolved.patch_tokens_per_frame == 390


def test_stratified_exit_schedule_and_role_rng_resume_are_exact():
    generator = torch.Generator().manual_seed(123)
    schedule = draw_stage2_exit_schedule(
        accumulation_steps=8,
        num_denoising_steps=4,
        generator=generator,
        synchronize_ranks=False,
    )
    assert sorted(schedule[:4]) == [0, 1, 2, 3]
    assert sorted(schedule[4:]) == [0, 1, 2, 3]

    streams = Stage2ExitRNGStreams(91)
    initial_state = streams.state_dict()
    assert not torch.equal(initial_state["generator"], initial_state["fake_score"])
    g_first = streams.draw("generator", accumulation_steps=4, synchronize_ranks=False)
    f_first = streams.draw("fake_score", accumulation_steps=4, synchronize_ranks=False)
    assert sorted(g_first) == sorted(f_first) == [0, 1, 2, 3]
    state = streams.state_dict()
    expected_next = streams.draw(
        "generator", accumulation_steps=4, synchronize_ranks=False
    )
    restored = Stage2ExitRNGStreams(0)
    restored.load_state_dict(state)
    assert (
        restored.draw("generator", accumulation_steps=4, synchronize_ranks=False)
        == expected_next
    )


@pytest.mark.parametrize("accumulation", [4, 8])
def test_stratified_schedule_requires_complete_k4_strata(accumulation):
    values = draw_stage2_exit_schedule(
        accumulation_steps=accumulation,
        num_denoising_steps=4,
        generator=torch.Generator().manual_seed(4),
        synchronize_ranks=False,
    )
    for offset in range(0, accumulation, 4):
        assert sorted(values[offset : offset + 4]) == [0, 1, 2, 3]


def test_exit_schedule_supports_native_k2_strata_for_later_compression():
    values = draw_stage2_exit_schedule(
        accumulation_steps=4,
        num_denoising_steps=2,
        generator=torch.Generator().manual_seed(8),
        synchronize_ranks=False,
    )
    assert sorted(values[:2]) == [0, 1]
    assert sorted(values[2:]) == [0, 1]


def test_nonzero_rank_receives_the_rank0_exit_schedule(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "gloo")
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 3)

    def broadcast(value, src):
        assert src == 0
        value.copy_(torch.tensor([3, 1, 0, 2], device=value.device))

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    values = draw_stage2_exit_schedule(
        accumulation_steps=4,
        num_denoising_steps=4,
        generator=torch.Generator().manual_seed(999),
        synchronize_ranks=True,
    )
    assert values == (3, 1, 0, 2)


def test_nccl_exit_broadcast_rejects_cpu_tensor_before_collective(monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")
    monkeypatch.setattr(
        torch.distributed,
        "broadcast",
        lambda *_args, **_kwargs: pytest.fail("invalid CPU collective was attempted"),
    )
    with pytest.raises(ValueError, match="NCCL cannot broadcast.*CPU"):
        draw_stage2_exit_schedule(
            accumulation_steps=4,
            num_denoising_steps=4,
            generator=torch.Generator().manual_seed(999),
            device="cpu",
            synchronize_ranks=True,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"local_window_frames": 24},
        {"frame_seq_length": 391},
    ],
)
def test_rollout_constructor_cannot_bypass_locked_window_and_token_contract(override):
    with pytest.raises(ValueError, match="named spec|390 tokens/frame"):
        Stage2RolloutPipeline(_FakeGenerator(), **override)


@pytest.mark.parametrize(
    "profile_name",
    [
        "stress_c8w24k4s1",
        "lower_c8w8k4s1",
        "c8w16k4s4",
        "c8w16k4s8",
    ],
)
def test_deployment_only_profiles_cannot_enter_training_rollout(profile_name):
    pipeline, _ = _pipeline(spec=profile_name)
    initial, noise, conditioning = _inputs(batch=1)
    with pytest.raises(RuntimeError, match="deployment-only"):
        pipeline.rollout(
            initial_latent=initial,
            noise=noise,
            conditional_dict=conditioning,
            exit_step=0,
            requires_grad=True,
        )


def test_rollout_generates_24_new_latents_and_only_clean_forwards_commit_kv():
    generator = _FakeGenerator()
    pipeline, schedulers = _pipeline(generator)
    original_runtime = (
        generator.model.local_attn_size,
        generator.model.max_attention_size,
        generator.model.sink_size,
        generator.model.global_sink_size,
    )
    initial, noise, conditioning = _inputs()
    result, state = pipeline.rollout(
        initial_latent=initial,
        noise=noise,
        conditional_dict=conditioning,
        exit_step=2,
        requires_grad=True,
    )

    assert result.latents.shape == noise.shape
    assert result.latents.requires_grad
    assert result.exit_step == 2
    assert result.requires_grad is True
    assert result.scheduler_timesteps == STAGE2_K4_SHIFT5_TIMESTEPS
    assert result.chunk_timesteps == (STAGE2_K4_SHIFT5_TIMESTEPS[:3],) * 3
    assert result.chunk_sigmas == (result.scheduler_sigmas[:3],) * 3
    assert len(schedulers) == 3
    assert [scheduler.steps for scheduler in schedulers] == [
        [999, 937],
        [999, 937],
        [999, 937],
    ]
    assert generator.calls[1]["timestep"] == (999.0,) * 8
    assert generator.calls[1]["flow_sigma"] == 1.0
    assert generator.calls[1]["flow_sigma"] != 999.0 / 1000.0

    # One sink preload, then three noisy/exit forwards and one clean commit per chunk.
    assert len(generator.calls) == 1 + 3 * 4
    assert [call["commit"] for call in generator.calls] == [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        False,
        True,
    ]
    assert [call["current_start"] for call in generator.calls if call["commit"]] == [
        0,
        1 * FRAME_TOKENS,
        9 * FRAME_TOKENS,
        17 * FRAME_TOKENS,
    ]
    assert state.self_kv[0]["k"].shape[1] == 17 * FRAME_TOKENS
    assert result.cache_audit["capacity_frames"] == 17
    assert result.cache_audit["global_end_index"] == 25 * FRAME_TOKENS
    assert result.cache_audit["local_end_index"] == 17 * FRAME_TOKENS
    assert result.cache_audit["conditional_cache_branches"] == 1
    assert result.cache_audit["noisy_forward_calls"] == 9
    assert result.cache_audit["generator_forward_calls"] == 13
    assert result.rollout_mode == "random_exit"
    assert len(result.chunk_trace) == 3
    assert len(result.initial_latent_sha256_per_sample) == 2
    assert result.generated_latent_sha256_per_sample == ()
    assert result.prefix_snapshot is None
    assert all(call["runtime_local_attn_size"] == 17 for call in generator.calls)
    assert all(
        call["runtime_max_attention_size"] == 17 * FRAME_TOKENS
        for call in generator.calls
    )
    assert all(call["runtime_sink_size"] == 1 for call in generator.calls)
    assert all(call["runtime_global_sink_size"] == 1 for call in generator.calls)
    assert (
        generator.model.local_attn_size,
        generator.model.max_attention_size,
        generator.model.sink_size,
        generator.model.global_sink_size,
    ) == original_runtime
    assert all(
        cache["k"].grad_fn is None and cache["v"].grad_fn is None
        for cache in state.self_kv
    )

    # Both the first and final new latent retain a path to the Generator.
    (result.latents[:, 0].sum() + result.latents[:, 23].sum()).backward()
    assert generator.scale.grad is not None
    assert torch.isfinite(generator.scale.grad)
    assert generator.scale.grad.abs().item() > 0


def test_fake_score_rollout_is_fresh_and_graph_free():
    pipeline, _ = _pipeline()
    initial, noise, conditioning = _inputs(batch=1)
    result, _ = pipeline.rollout(
        initial_latent=initial,
        noise=noise,
        conditional_dict=conditioning,
        exit_step=0,
        requires_grad=False,
    )
    assert result.latents.shape[1] == 24
    assert result.latents.requires_grad is False
    assert result.requires_grad is False
    assert result.chunk_timesteps == ((999,),) * 3
    assert result.cache_audit["noisy_forward_calls"] == 3


def test_generate_full_episode_is_bitwise_the_baseline_exit3_kernel():
    initial, noise, conditioning = _inputs(batch=1)
    deploy_pipeline, deploy_schedulers = _pipeline()
    deploy, _ = deploy_pipeline.generate_full_episode(
        initial_latent=initial,
        noise=noise,
        conditional_dict=conditioning,
    )
    exit_pipeline, exit_schedulers = _pipeline()
    training_exit, _ = exit_pipeline.rollout(
        initial_latent=initial,
        noise=noise,
        conditional_dict=conditioning,
        exit_step=3,
        requires_grad=False,
    )

    assert torch.equal(deploy.latents, training_exit.latents)
    assert deploy.rollout_mode == "full_denoising"
    assert training_exit.rollout_mode == "random_exit"
    assert deploy.exit_step == 3
    assert deploy.requires_grad is False
    assert len(deploy.generated_latent_sha256_per_sample) == 1
    assert deploy.initial_latent_sha256_per_sample == (
        tensor_identity_sha256(initial[0]),
    )
    assert deploy.generated_latent_sha256_per_sample == (
        tensor_identity_sha256(deploy.latents[0]),
    )
    assert deploy.chunk_timesteps == (STAGE2_K4_SHIFT5_TIMESTEPS,) * 3
    assert deploy.cache_audit["generator_forward_calls"] == 16
    assert len(deploy_schedulers) == len(exit_schedulers) == 3
    assert [scheduler.steps for scheduler in deploy_schedulers] == [
        [999, 937, 833],
        [999, 937, 833],
        [999, 937, 833],
    ]
    assert [trace["chunk_index"] for trace in deploy.chunk_trace] == [0, 1, 2]
    assert [trace["rope_frame_start"] for trace in deploy.chunk_trace] == [1, 9, 17]
    assert [trace["cache_before_global_end_index"] for trace in deploy.chunk_trace] == [
        1 * FRAME_TOKENS,
        9 * FRAME_TOKENS,
        17 * FRAME_TOKENS,
    ]
    assert [trace["cache_after_global_end_index"] for trace in deploy.chunk_trace] == [
        9 * FRAME_TOKENS,
        17 * FRAME_TOKENS,
        25 * FRAME_TOKENS,
    ]
    assert [trace["cache_after_local_end_index"] for trace in deploy.chunk_trace] == [
        9 * FRAME_TOKENS,
        17 * FRAME_TOKENS,
        17 * FRAME_TOKENS,
    ]
    assert all(trace["fresh_scheduler"] is True for trace in deploy.chunk_trace)
    assert all(trace["noisy_self_kv_commits"] == 0 for trace in deploy.chunk_trace)
    assert all(trace["clean_self_kv_commits"] == 1 for trace in deploy.chunk_trace)
    assert all(
        len(trace["clean_latent_sha256_per_sample"]) == 1
        for trace in deploy.chunk_trace
    )
    json.dumps(
        {
            "cache_audit": deploy.cache_audit,
            "chunk_trace": deploy.chunk_trace,
        }
    )


@pytest.mark.parametrize(
    (
        "profile_name",
        "chunk_frames",
        "capacity_frames",
        "denoising_steps",
        "expected_calls",
    ),
    [
        ("baseline_c8w16k4s1", 8, 17, 4, 16),
        ("stress_c8w24k4s1", 8, 25, 4, 16),
        ("lower_c8w8k4s1", 8, 9, 4, 16),
        ("c4w12k4s1", 4, 13, 4, 31),
        ("c4w8k4s1", 4, 9, 4, 31),
        ("c4w8k2s1", 4, 9, 2, 19),
    ],
)
def test_named_s1_profiles_share_one_full_deploy_path(
    profile_name,
    chunk_frames,
    capacity_frames,
    denoising_steps,
    expected_calls,
):
    generator = _FakeGenerator()
    pipeline, schedulers = _pipeline(generator, spec=profile_name)
    initial, noise, conditioning = _inputs(batch=1)
    result, state = pipeline.generate_full_episode(
        initial_latent=initial,
        noise=noise,
        conditional_dict=conditioning,
    )

    chunks = 24 // chunk_frames
    assert result.latents.shape == noise.shape
    assert result.latents.requires_grad is False
    assert len(schedulers) == chunks
    assert len(result.chunk_trace) == chunks
    assert len(generator.calls) == expected_calls
    assert result.cache_audit["generator_forward_calls"] == expected_calls
    assert result.cache_audit["capacity_frames"] == capacity_frames
    assert result.cache_audit["global_end_index"] == 25 * FRAME_TOKENS
    assert result.cache_audit["local_end_index"] == capacity_frames * FRAME_TOKENS
    assert result.cache_audit["profile"]["name"] == profile_name
    assert state.self_kv[0]["k"].shape[1] == capacity_frames * FRAME_TOKENS
    assert [trace["rope_frame_start"] for trace in result.chunk_trace] == list(
        range(1, 25, chunk_frames)
    )
    assert all(
        trace["timestep_count"] == denoising_steps for trace in result.chunk_trace
    )
    assert all(
        call["runtime_max_attention_size"] == capacity_frames * FRAME_TOKENS
        for call in generator.calls
    )
    assert all(call["runtime_sink_size"] == 1 for call in generator.calls)
    assert generator.model.local_attn_size == 16
    assert generator.model.max_attention_size == 0
    assert generator.model.sink_size == 0
    assert generator.model.global_sink_size == 0


@pytest.mark.parametrize(
    ("target_profile", "sink_frames"),
    [("c8w16k4s4", 4), ("c8w16k4s8", 8)],
)
def test_episode1_prefix_snapshot_restores_detached_s4_s8_episode2(
    target_profile, sink_frames
):
    generator = _FakeGenerator()
    baseline, _ = _pipeline(generator)
    target, target_schedulers = _pipeline(generator, spec=target_profile)
    initial, noise, conditioning_a = _inputs(prompt=3.0, batch=2)
    noise[1].add_(0.5)
    result_a, state_a = baseline.generate_full_episode(
        initial_latent=initial,
        noise=noise,
        conditional_dict=conditioning_a,
        capture_prefix_sink_frames=sink_frames,
    )

    reference, _ = _pipeline()
    reference_a, _ = reference.generate_full_episode(
        initial_latent=initial,
        noise=noise,
        conditional_dict=conditioning_a,
    )
    assert torch.equal(result_a.latents, reference_a.latents)
    assert result_a.cache_audit["capacity_frames"] == 17
    snapshot = result_a.prefix_snapshot
    assert isinstance(snapshot, Stage2SinkPrefixSnapshot)
    assert snapshot.target_sink_frames == sink_frames
    assert snapshot.clean_latents.shape[1] == sink_frames - 1
    assert len(snapshot.clean_latent_sha256_per_sample) == 2
    assert len(set(snapshot.clean_latent_sha256_per_sample)) == 2
    assert torch.equal(
        snapshot.clean_latents,
        result_a.latents[:, : sink_frames - 1],
    )
    assert snapshot.clean_latents.requires_grad is False
    assert snapshot.clean_latents.grad_fn is None
    assert all(
        layer.k.requires_grad is False
        and layer.v.requires_grad is False
        and layer.k.grad_fn is None
        and layer.v.grad_fn is None
        and layer.k.shape[1] == sink_frames * FRAME_TOKENS
        for layer in snapshot.layers
    )
    assert result_a.chunk_trace[0]["captured_prefix_sink_frames"] == sink_frames
    assert result_a.chunk_trace[1]["captured_prefix_sink_frames"] == 0

    state_b = target.reset_for_new_episode(
        state_a,
        prefix_snapshot=snapshot,
    )
    assert state_b is not state_a
    assert state_b.profile_name == target_profile
    assert state_b.episode_index == 1
    assert state_b.episode_complete is False
    for cache, layer in zip(state_b.self_kv, snapshot.layers):
        assert cache["k"].shape[1] == (sink_frames + 16) * FRAME_TOKENS
        assert torch.equal(cache["k"][:, : target.sink_tokens], layer.k)
        assert torch.equal(cache["v"][:, : target.sink_tokens], layer.v)
        assert torch.count_nonzero(cache["k"][:, target.sink_tokens :]) == 0
        assert torch.count_nonzero(cache["v"][:, target.sink_tokens :]) == 0
        assert int(cache["global_end_index"].item()) == target.sink_tokens
        assert int(cache["local_end_index"].item()) == target.sink_tokens
    assert all(cache["is_init"] is False for cache in state_b.cross_kv)
    assert all(torch.count_nonzero(cache["k"]) == 0 for cache in state_b.cross_kv)

    calls_before_b = len(generator.calls)
    conditioning_b = {"prompt_embeds": torch.full((2, 2, 3), 9.0, dtype=torch.bfloat16)}
    result_b, state_b = target.generate_full_episode(
        initial_latent=initial,
        noise=noise + 1,
        conditional_dict=conditioning_b,
        state=state_b,
    )
    b_calls = generator.calls[calls_before_b:]
    assert len(target_schedulers) == 3
    assert len(b_calls) == 15
    assert result_b.cache_audit["sink_preload_forward_calls"] == 0
    assert result_b.cache_audit["generator_forward_calls"] == 15
    assert result_b.cache_audit["capacity_frames"] == sink_frames + 16
    assert result_b.cache_audit["global_end_index"] == (sink_frames + 24) * FRAME_TOKENS
    assert result_b.cache_audit["local_end_index"] == (sink_frames + 16) * FRAME_TOKENS
    assert result_b.chunk_trace[0]["rope_frame_start"] == sink_frames
    assert b_calls[0]["current_start"] == sink_frames * FRAME_TOKENS
    assert all(call["prompt"] == 9.0 for call in b_calls)
    assert all(call["runtime_sink_size"] == sink_frames for call in b_calls)
    assert all(
        call["runtime_max_attention_size"] == (sink_frames + 16) * FRAME_TOKENS
        for call in b_calls
    )
    assert state_b.episode_complete is True
    assert generator.model.local_attn_size == 16
    assert generator.model.max_attention_size == 0
    assert generator.model.sink_size == 0
    assert generator.model.global_sink_size == 0

    calls_before_baseline = len(generator.calls)
    baseline.generate_full_episode(
        initial_latent=initial,
        noise=noise,
        conditional_dict=conditioning_a,
    )
    later_baseline_calls = generator.calls[calls_before_baseline:]
    assert all(call["runtime_sink_size"] == 1 for call in later_baseline_calls)
    assert all(
        call["runtime_max_attention_size"] == 17 * FRAME_TOKENS
        for call in later_baseline_calls
    )


def test_multi_sink_profiles_fail_closed_without_a_prefix_snapshot():
    target, _ = _pipeline(spec="c8w16k4s4")
    initial, noise, conditioning = _inputs(batch=1)
    with pytest.raises(RuntimeError, match="cannot start a fresh episode"):
        target.generate_full_episode(
            initial_latent=initial,
            noise=noise,
            conditional_dict=conditioning,
        )


def test_attention_runtime_is_restored_when_scheduler_validation_fails():
    class _BadScheduler(_FakeScheduler):
        def set_timesteps(self, count, *, device, shift):
            super().set_timesteps(count, device=device, shift=shift)
            self.timesteps[0] = 998

    generator = _FakeGenerator()
    original_runtime = (
        generator.model.local_attn_size,
        generator.model.max_attention_size,
        generator.model.sink_size,
        generator.model.global_sink_size,
    )
    pipeline = Stage2RolloutPipeline(
        generator,
        scheduler_factory=_BadScheduler,
    )
    initial, noise, conditioning = _inputs(batch=1)
    with pytest.raises(RuntimeError, match="timetable drifted"):
        pipeline.generate_full_episode(
            initial_latent=initial,
            noise=noise,
            conditional_dict=conditioning,
        )
    assert (
        generator.model.local_attn_size,
        generator.model.max_attention_size,
        generator.model.sink_size,
        generator.model.global_sink_size,
    ) == original_runtime


def test_rollout_rejects_scheduler_instance_reuse_between_chunks():
    scheduler = _FakeScheduler()
    pipeline = Stage2RolloutPipeline(
        _FakeGenerator(),
        scheduler_factory=lambda: scheduler,
    )
    initial, noise, conditioning = _inputs(batch=1)
    with pytest.raises(RuntimeError, match="distinct UniPC scheduler"):
        pipeline.generate_full_episode(
            initial_latent=initial,
            noise=noise,
            conditional_dict=conditioning,
        )


def test_episode_reset_keeps_sink_clears_local_and_cross_and_restarts_rope():
    generator = _FakeGenerator()
    pipeline, _ = _pipeline(generator)
    initial, noise, conditioning_a = _inputs(prompt=3.0, batch=1)
    result_a, state = pipeline.rollout(
        initial_latent=initial,
        noise=noise,
        conditional_dict=conditioning_a,
        exit_step=0,
        requires_grad=True,
    )
    del result_a
    sink_before = [
        (cache["k"][:, :FRAME_TOKENS].clone(), cache["v"][:, :FRAME_TOKENS].clone())
        for cache in state.self_kv
    ]
    calls_before = len(generator.calls)
    assert pipeline.reset_for_new_episode(state) is state
    for cache, (sink_k, sink_v) in zip(state.self_kv, sink_before):
        assert torch.equal(cache["k"][:, :FRAME_TOKENS], sink_k)
        assert torch.equal(cache["v"][:, :FRAME_TOKENS], sink_v)
        assert torch.count_nonzero(cache["k"][:, FRAME_TOKENS:]) == 0
        assert torch.count_nonzero(cache["v"][:, FRAME_TOKENS:]) == 0
        assert int(cache["global_end_index"].item()) == FRAME_TOKENS
        assert int(cache["local_end_index"].item()) == FRAME_TOKENS
        assert int(cache["pinned_start"].item()) == -1
        assert int(cache["pinned_len"].item()) == 0
    assert all(cache["is_init"] is False for cache in state.cross_kv)
    assert all(torch.count_nonzero(cache["k"]) == 0 for cache in state.cross_kv)

    conditioning_b = {"prompt_embeds": torch.full((1, 2, 3), 9.0, dtype=torch.bfloat16)}
    result_b, state = pipeline.rollout(
        initial_latent=initial,
        noise=noise + 1,
        conditional_dict=conditioning_b,
        exit_step=0,
        requires_grad=True,
        state=state,
    )
    assert result_b.latents.shape[1] == 24
    assert result_b.cache_audit["sink_preload_forward_calls"] == 0
    assert result_b.cache_audit["generator_forward_calls"] == 6
    assert result_b.cache_audit["logical_query_tokens"] == 6 * 8 * FRAME_TOKENS
    second_calls = generator.calls[calls_before:]
    assert second_calls[0]["current_start"] == FRAME_TOKENS
    assert all(call["prompt"] == 9.0 for call in second_calls)
    for cache, (sink_k, sink_v) in zip(state.self_kv, sink_before):
        assert torch.equal(cache["k"][:, :FRAME_TOKENS], sink_k)
        assert torch.equal(cache["v"][:, :FRAME_TOKENS], sink_v)


def test_rollout_refuses_partial_reset_changed_sink_and_shape_fallbacks():
    pipeline, _ = _pipeline()
    initial, noise, conditioning = _inputs(batch=1)
    with pytest.raises(ValueError, match="exactly 24"):
        pipeline.rollout(
            initial_latent=initial,
            noise=noise[:, :-1],
            conditional_dict=conditioning,
            exit_step=0,
            requires_grad=True,
        )
    with pytest.raises(TypeError, match="bfloat16"):
        pipeline.rollout(
            initial_latent=initial.float(),
            noise=noise.float(),
            conditional_dict=conditioning,
            exit_step=0,
            requires_grad=True,
        )
    with pytest.raises(TypeError, match="prompt_embeds"):
        pipeline.rollout(
            initial_latent=initial,
            noise=noise,
            conditional_dict={"prompt_embeds": conditioning["prompt_embeds"].float()},
            exit_step=0,
            requires_grad=True,
        )
    with pytest.raises(ValueError, match="outside"):
        pipeline.rollout(
            initial_latent=initial,
            noise=noise,
            conditional_dict=conditioning,
            exit_step=4,
            requires_grad=True,
        )

    _, state = pipeline.rollout(
        initial_latent=initial,
        noise=noise,
        conditional_dict=conditioning,
        exit_step=0,
        requires_grad=True,
    )
    with pytest.raises(RuntimeError, match="reset_for_new_episode"):
        pipeline.rollout(
            initial_latent=initial,
            noise=noise,
            conditional_dict=conditioning,
            exit_step=0,
            requires_grad=True,
            state=state,
        )
    state.self_kv[0]["global_end_index"].sub_(FRAME_TOKENS)
    with pytest.raises(RuntimeError, match="cache cursor mismatch"):
        pipeline.reset_for_new_episode(state)
    state.self_kv[0]["global_end_index"].add_(FRAME_TOKENS)
    assert pipeline.reset_for_new_episode(state) is state
    changed = initial.clone()
    changed[:, :, :, 0, 0] += 1
    with pytest.raises(RuntimeError, match="sink changed"):
        pipeline.rollout(
            initial_latent=changed,
            noise=noise,
            conditional_dict=conditioning,
            exit_step=0,
            requires_grad=True,
            state=state,
        )


def test_generator_activation_checkpointing_is_rejected():
    with pytest.raises(RuntimeError, match="forbids activation checkpointing"):
        _pipeline(_FakeGenerator(checkpointing=True))


def test_real_causal_attention_last_chunk_sees_sink_history8_and_current8(
    monkeypatch,
):
    attention = CausalWanSelfAttention(
        dim=1,
        num_heads=1,
        local_attn_size=17,
        sink_size=1,
        qk_norm=False,
    )
    attention.global_sink_size = 1
    attention.max_attention_size = 17
    with torch.no_grad():
        for projection in (attention.q, attention.k, attention.v, attention.o):
            projection.weight.fill_(1)
            projection.bias.zero_()
    monkeypatch.setattr(
        causal_model_module,
        "causal_rope_apply",
        lambda value, *_args, **_kwargs: value,
    )
    attended = []

    def capture_attention(query, key, value):
        attended.append((key.detach().clone(), value.detach().clone()))
        return torch.zeros_like(query)

    monkeypatch.setattr(causal_model_module, "attention", capture_attention)
    cache = {
        "k": torch.zeros(1, 17, 1, 1),
        "v": torch.zeros(1, 17, 1, 1),
        "global_end_index": torch.tensor([0], dtype=torch.long),
        "local_end_index": torch.tensor([0], dtype=torch.long),
        "pinned_start": torch.tensor([-1], dtype=torch.long),
        "pinned_len": torch.tensor([0], dtype=torch.long),
    }
    cursor = 0
    for frames, marker in ((1, 1.0), (8, 2.0), (8, 3.0), (8, 4.0)):
        causal_model_module._CURRENT_GRID_META.clear()
        values = torch.full((1, frames, 1), marker)
        _, update = attention(
            values,
            seq_lens=torch.tensor([frames]),
            grid_sizes=torch.tensor([[frames, 1, 1]]),
            freqs=None,
            block_mask=None,
            kv_cache=cache,
            current_start=cursor,
            cache_start=cursor,
        )
        CausalWanModel._apply_cache_updates(SimpleNamespace(), [cache], [(0, update)])
        cursor += frames

    assert [item[0].shape[1] for item in attended] == [1, 9, 17, 17]
    last_keys = attended[-1][0][0, :, 0, 0]
    assert torch.equal(last_keys[:1], torch.tensor([1.0]))
    assert torch.equal(last_keys[1:9], torch.full((8,), 3.0))
    assert torch.equal(last_keys[9:], torch.full((8,), 4.0))
    assert int(cache["global_end_index"].item()) == 25
    assert int(cache["local_end_index"].item()) == 17


@pytest.mark.parametrize(
    ("window_frames", "expected_local_markers"),
    [
        (12, [5.0] * 4 + [6.0] * 4 + [7.0] * 4),
        (8, [6.0] * 4 + [7.0] * 4),
    ],
)
def test_real_causal_attention_c4_preserves_exact_named_history(
    monkeypatch,
    window_frames,
    expected_local_markers,
):
    capacity = 1 + window_frames
    attention = CausalWanSelfAttention(
        dim=1,
        num_heads=1,
        local_attn_size=capacity,
        sink_size=1,
        qk_norm=False,
    )
    attention.global_sink_size = 1
    attention.max_attention_size = capacity
    with torch.no_grad():
        for projection in (attention.q, attention.k, attention.v, attention.o):
            projection.weight.fill_(1)
            projection.bias.zero_()
    monkeypatch.setattr(
        causal_model_module,
        "causal_rope_apply",
        lambda value, *_args, **_kwargs: value,
    )
    attended = []

    def capture_attention(query, key, value):
        attended.append(key.detach().clone())
        return torch.zeros_like(query)

    monkeypatch.setattr(causal_model_module, "attention", capture_attention)
    cache = {
        "k": torch.zeros(1, capacity, 1, 1),
        "v": torch.zeros(1, capacity, 1, 1),
        "global_end_index": torch.tensor([0], dtype=torch.long),
        "local_end_index": torch.tensor([0], dtype=torch.long),
        "pinned_start": torch.tensor([-1], dtype=torch.long),
        "pinned_len": torch.tensor([0], dtype=torch.long),
    }
    cursor = 0
    for frames, marker in [(1, 1.0)] + [(4, float(value)) for value in range(2, 8)]:
        causal_model_module._CURRENT_GRID_META.clear()
        values = torch.full((1, frames, 1), marker)
        _, update = attention(
            values,
            seq_lens=torch.tensor([frames]),
            grid_sizes=torch.tensor([[frames, 1, 1]]),
            freqs=None,
            block_mask=None,
            kv_cache=cache,
            current_start=cursor,
            cache_start=cursor,
        )
        CausalWanModel._apply_cache_updates(SimpleNamespace(), [cache], [(0, update)])
        cursor += frames

    final_keys = attended[-1][0, :, 0, 0]
    assert torch.equal(final_keys[:1], torch.tensor([1.0]))
    assert torch.equal(final_keys[1:], torch.tensor(expected_local_markers))
    assert int(cache["global_end_index"].item()) == 25
    assert int(cache["local_end_index"].item()) == capacity


@pytest.mark.parametrize("sink_frames", [4, 8])
def test_real_causal_attention_preserves_restored_multi_sink_prefix(
    monkeypatch, sink_frames
):
    capacity = sink_frames + 16
    attention = CausalWanSelfAttention(
        dim=1,
        num_heads=1,
        local_attn_size=capacity,
        sink_size=sink_frames,
        qk_norm=False,
    )
    attention.global_sink_size = sink_frames
    attention.max_attention_size = capacity
    with torch.no_grad():
        for projection in (attention.q, attention.k, attention.v, attention.o):
            projection.weight.fill_(1)
            projection.bias.zero_()
    monkeypatch.setattr(
        causal_model_module,
        "causal_rope_apply",
        lambda value, *_args, **_kwargs: value,
    )
    attended = []

    def capture_attention(query, key, value):
        attended.append(key.detach().clone())
        return torch.zeros_like(query)

    monkeypatch.setattr(causal_model_module, "attention", capture_attention)
    cache = {
        "k": torch.zeros(1, capacity, 1, 1),
        "v": torch.zeros(1, capacity, 1, 1),
        "global_end_index": torch.tensor([sink_frames], dtype=torch.long),
        "local_end_index": torch.tensor([sink_frames], dtype=torch.long),
        "pinned_start": torch.tensor([-1], dtype=torch.long),
        "pinned_len": torch.tensor([0], dtype=torch.long),
    }
    prefix = torch.arange(1, sink_frames + 1, dtype=torch.float32)
    cache["k"][0, :sink_frames, 0, 0].copy_(prefix)
    cache["v"][0, :sink_frames, 0, 0].copy_(prefix)
    cursor = sink_frames
    for marker in (5.0, 6.0, 7.0):
        causal_model_module._CURRENT_GRID_META.clear()
        values = torch.full((1, 8, 1), marker)
        _, update = attention(
            values,
            seq_lens=torch.tensor([8]),
            grid_sizes=torch.tensor([[8, 1, 1]]),
            freqs=None,
            block_mask=None,
            kv_cache=cache,
            current_start=cursor,
            cache_start=cursor,
        )
        CausalWanModel._apply_cache_updates(SimpleNamespace(), [cache], [(0, update)])
        cursor += 8

    assert [keys.shape[1] for keys in attended] == [
        sink_frames + 8,
        capacity,
        capacity,
    ]
    final_keys = attended[-1][0, :, 0, 0]
    assert torch.equal(final_keys[:sink_frames], prefix)
    assert torch.equal(final_keys[sink_frames : sink_frames + 8], torch.full((8,), 6.0))
    assert torch.equal(final_keys[sink_frames + 8 :], torch.full((8,), 7.0))
    assert int(cache["global_end_index"].item()) == sink_frames + 24
    assert int(cache["local_end_index"].item()) == capacity


def test_stage2_cross_attention_cache_is_real_and_legacy_bypass_is_unchanged(
    monkeypatch,
):
    attention = MultiShotT2VCrossAttention(
        dim=4,
        num_heads=1,
        qk_norm=False,
    )
    with torch.no_grad():
        for projection in (attention.q, attention.k, attention.v, attention.o):
            projection.weight.copy_(torch.eye(4))
            projection.bias.zero_()
    attended_keys = []

    def capture_attention(query, key, value, **_kwargs):
        attended_keys.append(key.detach().clone())
        return query

    monkeypatch.setattr(causal_model_module, "flash_attention", capture_attention)
    query = torch.ones(1, 2, 4)
    first_context = torch.full((1, 3, 4), 2.0)
    changed_context = torch.full((1, 3, 4), 9.0)
    cache = {
        "k": torch.zeros(1, 3, 1, 4),
        "v": torch.zeros(1, 3, 1, 4),
        "is_init": False,
        "stage2_enabled": True,
    }

    attention(query, first_context, None, crossattn_cache=cache)
    assert cache["is_init"] is True
    first_cached = cache["k"].clone()
    attention(query, changed_context, None, crossattn_cache=cache)
    assert torch.equal(cache["k"], first_cached)
    assert torch.equal(attended_keys[0], attended_keys[1])
    assert cache["k"].grad_fn is None and cache["v"].grad_fn is None

    cache["k"].zero_()
    cache["v"].zero_()
    cache["is_init"] = False
    attention(query, changed_context, None, crossattn_cache=cache)
    assert cache["is_init"] is True
    assert not torch.equal(cache["k"], first_cached)

    legacy_cache = {
        "k": torch.zeros_like(cache["k"]),
        "v": torch.zeros_like(cache["v"]),
        "is_init": False,
    }
    attention(query, first_context, None, crossattn_cache=legacy_cache)
    assert legacy_cache["is_init"] is False


def test_exit_rng_state_rejects_missing_or_malformed_roles():
    streams = Stage2ExitRNGStreams(1)
    state = streams.state_dict()
    with pytest.raises(ValueError, match="generator/fake_score"):
        streams.load_state_dict({"generator": state["generator"]})
    broken = copy.deepcopy(state)
    broken["generator"] = torch.ones(2, dtype=torch.float32)
    with pytest.raises(TypeError, match="invalid"):
        streams.load_state_dict(broken)


def test_stratified_exit_rejects_incomplete_accumulation_group():
    with pytest.raises(ValueError, match="accumulation_steps %"):
        draw_stage2_exit_schedule(
            accumulation_steps=5,
            num_denoising_steps=4,
            generator=torch.Generator().manual_seed(1),
            synchronize_ranks=False,
        )
