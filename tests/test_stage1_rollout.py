from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import MethodType
from typing import ClassVar

import pytest
import torch

import pipeline.causal_diffusion_inference as causal_pipeline_module
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline
from pipeline.stage1_rollout import (
    STAGE1_ROLLOUT_TRACE_SCHEMA,
    Stage1RolloutPipeline,
    run_stage1_rollout,
)
from utils.config import DEFAULT_NEGATIVE_PROMPT


class CausalWanSelfAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_attention_size = 123
        self.sink_size = 3
        self.global_sink_size = 4


class _TinyDit(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.local_attn_size = -7
        self.t_scale = 1.25
        self.rope_method = "linear"
        self.original_seq_len = 99
        self.use_relative_rope = False
        self.rope_temporal_offset = 2.0
        self.num_frame_per_block = 6
        self.max_attention_size = 321
        self.sink_size = 3
        self.global_sink_size = 4
        self.attention = CausalWanSelfAttention()


class _RecordingGenerator(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _TinyDit()
        self.calls: list[dict[str, object]] = []
        self.raise_on_call: int | None = None
        self.failure_runtime: dict[str, object] | None = None

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
    ):
        attention = self.model.attention
        if self.raise_on_call is not None and len(self.calls) == self.raise_on_call:
            self.failure_runtime = {
                "dit_local_attn_size": self.model.local_attn_size,
                "dit_num_frame_per_block": self.model.num_frame_per_block,
                "attention_max_attention_size": attention.max_attention_size,
                "attention_sink_size": attention.sink_size,
                "attention_global_sink_size": attention.global_sink_size,
            }
            raise RuntimeError("injected rollout generator failure")

        cache = kv_cache[0]
        branch = cache["kind"]
        call = {
            "branch": branch,
            "commit": commit_self_kv,
            "before_global": int(cache["global_end_index"].item()),
            "before_local": int(cache["local_end_index"].item()),
            "current_start": int(current_start),
            "cache_start": int(cache_start),
            "input": noisy_image_or_video.detach().clone(),
            "timestep": timestep.detach().clone(),
            "dit_local_attn_size": self.model.local_attn_size,
            "dit_num_frame_per_block": self.model.num_frame_per_block,
            "attention_max_attention_size": attention.max_attention_size,
            "attention_sink_size": attention.sink_size,
            "attention_global_sink_size": attention.global_sink_size,
        }
        self.calls.append(call)

        if commit_self_kv:
            current_end = int(current_start) + noisy_image_or_video.shape[1]
            for layer_cache in kv_cache:
                local_before = int(layer_cache["local_end_index"].item())
                capacity = int(layer_cache["k"].shape[1])
                layer_cache["global_end_index"].fill_(current_end)
                layer_cache["local_end_index"].fill_(
                    min(capacity, local_before + noisy_image_or_video.shape[1])
                )
        crossattn_cache[0]["is_init"] = True

        prompt_value = float(conditional_dict["prompt_embeds"].reshape(-1)[0].item())
        raw_flow = torch.full_like(noisy_image_or_video, prompt_value)
        # Deliberately wrong: rollout must derive terminal x0 from raw flow and
        # the scheduler sigma, not trust this wrapper-side prediction.
        wrapper_x0 = torch.full_like(noisy_image_or_video, 999.0)
        return raw_flow, wrapper_x0


class _RecordingTextEncoder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, *, text_prompts):
        prompts = list(text_prompts)
        self.calls.append(prompts)
        values = [
            -2.0 if prompt == DEFAULT_NEGATIVE_PROMPT else 2.0 for prompt in prompts
        ]
        return {
            "prompt_embeds": torch.tensor(values, dtype=torch.float32).reshape(
                len(values), 1, 1
            )
        }


class _RecordingVAE:
    def __init__(self) -> None:
        self.decode_inputs: list[torch.Tensor] = []

    def decode_to_pixel(self, latents):
        self.decode_inputs.append(latents.detach().clone())
        pixel_frames = 1 + 4 * (latents.shape[1] - 1)
        return torch.zeros(
            latents.shape[0],
            pixel_frames,
            3,
            2,
            2,
            dtype=latents.dtype,
            device=latents.device,
        )


class _FourStepScheduler:
    instances: ClassVar[list[_FourStepScheduler]] = []

    def __init__(self, **kwargs) -> None:
        self.constructor_kwargs = kwargs
        self.timesteps: torch.Tensor | None = None
        self.sigmas: torch.Tensor | None = None
        self.set_call: tuple[int, torch.device, float] | None = None
        self.step_calls: list[float] = []
        self.__class__.instances.append(self)

    def set_timesteps(self, steps, *, device, shift) -> None:
        self.set_call = (steps, device, shift)
        assert steps == 4
        self.timesteps = torch.tensor([40, 30, 20, 10], device=device)
        self.sigmas = torch.tensor([1.0, 0.75, 0.5, 0.25, 0.0], device=device)

    def step(self, _flow, timestep, sample, *, return_dict):
        assert return_dict is False
        self.step_calls.append(float(timestep.item()))
        return (sample + 1.0,)


def _self_kv_cache(kind: str, *, layers: int, dtype, device, capacity: int):
    return [
        {
            "kind": kind,
            "k": torch.zeros(1, capacity, 1, 1, dtype=dtype, device=device),
            "v": torch.zeros(1, capacity, 1, 1, dtype=dtype, device=device),
            "quantized": False,
            "block_token_size": 8,
            "max_blocks": 2,
            "num_heads": 1,
            "num_filled_blocks": 0,
            "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
            "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
        }
        for _ in range(layers)
    ]


def _cross_cache(*, layers: int, dtype, device):
    return [
        {
            "k": torch.zeros(1, 1, 1, 1, dtype=dtype, device=device),
            "v": torch.zeros(1, 1, 1, 1, dtype=dtype, device=device),
            "is_init": False,
        }
        for _ in range(layers)
    ]


def _make_pipeline(monkeypatch, *, layers: int = 1):
    _FourStepScheduler.instances.clear()
    monkeypatch.setattr(
        causal_pipeline_module,
        "FlowUniPCMultistepScheduler",
        _FourStepScheduler,
    )
    pipeline = CausalDiffusionInferencePipeline.__new__(
        CausalDiffusionInferencePipeline
    )
    torch.nn.Module.__init__(pipeline)
    pipeline.frame_seq_length = 1
    pipeline.num_frame_per_block = 6
    pipeline.num_transformer_blocks = layers
    pipeline.num_train_timesteps = 1000
    pipeline.sampling_steps = 7
    pipeline.sample_solver = "legacy-solver"
    pipeline.shift = 2.5
    pipeline.guidance_scale = 1.75
    pipeline.negative_prompt = "sentinel-negative"
    pipeline.local_attn_size = 23
    pipeline.global_sink_size = 0
    pipeline.sink_size = 0
    pipeline.timesteps = torch.tensor([-123.0])
    pipeline.multi_shot_sink = False
    pipeline.shot_clean_recache = False
    pipeline.streaming_vae = False
    pipeline.vae_device = None
    pipeline.async_vae = False
    pipeline.quantize_kv = False
    pipeline.inference_t_scale = None
    pipeline.use_relative_rope = False
    pipeline._rope_method_override = None
    pipeline._original_seq_len_override = None
    pipeline.multi_shot_rope_offset = 0.0
    pipeline._continuation_lock = None
    pipeline._active_continuation_session = None
    pipeline.generator = _RecordingGenerator()
    pipeline.text_encoder = _RecordingTextEncoder()
    pipeline.vae = _RecordingVAE()
    pipeline.built_kv_pairs = []

    def build_kv(self, batch_size, dtype, device):
        assert batch_size == 1
        pair = (
            _self_kv_cache(
                "positive",
                layers=layers,
                dtype=dtype,
                device=device,
                capacity=self.local_attn_size,
            ),
            _self_kv_cache(
                "negative",
                layers=layers,
                dtype=dtype,
                device=device,
                capacity=self.local_attn_size,
            ),
        )
        self.built_kv_pairs.append(pair)
        return pair

    def build_cross(self, batch_size, dtype, device):
        assert batch_size == 1
        return _cross_cache(layers=layers, dtype=dtype, device=device), _cross_cache(
            layers=layers, dtype=dtype, device=device
        )

    pipeline._build_kv_cache = MethodType(build_kv, pipeline)
    pipeline._build_crossattn_cache = MethodType(build_cross, pipeline)
    return pipeline


def _inputs():
    initial = torch.full((1, 1, 1, 1, 1), 100.0)
    noise = torch.arange(24, dtype=torch.float32).reshape(1, 24, 1, 1, 1)
    return initial, noise


def _original_runtime_state(pipeline):
    attention = pipeline.generator.model.attention
    return {
        "num_frame_per_block": pipeline.num_frame_per_block,
        "local_attn_size": pipeline.local_attn_size,
        "sampling_steps": pipeline.sampling_steps,
        "sample_solver": pipeline.sample_solver,
        "shift": pipeline.shift,
        "guidance_scale": pipeline.guidance_scale,
        "negative_prompt": pipeline.negative_prompt,
        "global_sink_size": pipeline.global_sink_size,
        "sink_size": pipeline.sink_size,
        "timesteps_id": id(pipeline.timesteps),
        "timesteps": tuple(float(value) for value in pipeline.timesteps),
        "dit_num_frame_per_block": pipeline.generator.model.num_frame_per_block,
        "dit_local_attn_size": pipeline.generator.model.local_attn_size,
        "dit_max_attention_size": pipeline.generator.model.max_attention_size,
        "dit_sink_size": pipeline.generator.model.sink_size,
        "dit_global_sink_size": pipeline.generator.model.global_sink_size,
        "attention_max_attention_size": attention.max_attention_size,
        "attention_sink_size": attention.sink_size,
        "attention_global_sink_size": attention.global_sink_size,
    }


def test_cfg_rollout_matches_stage2_cache_policy_and_decodes_once(monkeypatch):
    pipeline = _make_pipeline(monkeypatch)
    initial, noise = _inputs()
    original_runtime = _original_runtime_state(pipeline)

    result = run_stage1_rollout(
        pipeline,
        initial_latent=initial,
        future_noise=noise,
        prompts=["a cat runs"],
        profile={"sampling_steps": 4},
        checkpoint_provenance={
            "runtime_mode": "dynamic_stage1_lora",
            "adapter": {"variant": "ema"},
            "checkpoint": {
                "teacher_forcing": True,
                "training_block_size": 8,
            },
        },
    )

    assert result.future_latents.shape == (1, 24, 1, 1, 1)
    assert result.latents.shape == (1, 25, 1, 1, 1)
    assert result.video.shape == (1, 97, 3, 2, 2)
    # Three scheduler updates add 3, then exact terminal x0 subtracts
    # sigma=0.25 times CFG-combined flow (-2 + 5 * (2 - -2) = 18).
    assert torch.equal(result.future_latents, noise - 1.5)
    assert torch.equal(result.latents[:, :1], initial)
    assert len(pipeline.vae.decode_inputs) == 1
    assert torch.equal(pipeline.vae.decode_inputs[0], result.latents)

    calls = pipeline.generator.calls
    assert len(calls) == 32
    assert [call["branch"] for call in calls[:2]] == ["positive", "negative"]
    for branch in ("positive", "negative"):
        branch_calls = [call for call in calls if call["branch"] == branch]
        assert len(branch_calls) == 16
        assert [call["commit"] for call in branch_calls] == [
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            False,
            False,
            False,
            True,
        ]
        assert [call["current_start"] for call in branch_calls] == [
            0,
            *([1] * 5),
            *([9] * 5),
            *([17] * 5),
        ]
        assert [call["cache_start"] for call in branch_calls] == [
            0,
            *([1] * 5),
            *([9] * 5),
            *([17] * 5),
        ]
        assert [call["before_global"] for call in branch_calls] == [
            0,
            *([1] * 5),
            *([9] * 5),
            *([17] * 5),
        ]
        assert [call["before_local"] for call in branch_calls] == [
            0,
            *([1] * 5),
            *([9] * 5),
            *([17] * 5),
        ]
        assert all(call["dit_local_attn_size"] == 17 for call in branch_calls)
        assert all(call["dit_num_frame_per_block"] == 8 for call in branch_calls)
        assert all(call["attention_max_attention_size"] == 17 for call in branch_calls)
        assert all(call["attention_sink_size"] == 1 for call in branch_calls)
        assert all(call["attention_global_sink_size"] == 1 for call in branch_calls)

    positive_calls = [call for call in calls if call["branch"] == "positive"]
    noisy_first_calls = [positive_calls[index] for index in (1, 6, 11)]
    clean_calls = [positive_calls[index] for index in (5, 10, 15)]
    for chunk_index, call in enumerate(noisy_first_calls):
        start = chunk_index * 8
        assert torch.equal(call["input"], noise[:, start : start + 8])
        assert torch.equal(
            call["timestep"], torch.full((1, 8), 40.0, dtype=torch.float32)
        )
    for chunk_index, call in enumerate(clean_calls):
        start = chunk_index * 8
        assert torch.equal(call["input"], result.future_latents[:, start : start + 8])
        assert torch.count_nonzero(call["timestep"]) == 0

    assert len(_FourStepScheduler.instances) == 3
    assert len({id(scheduler) for scheduler in _FourStepScheduler.instances}) == 3
    assert [scheduler.set_call[0] for scheduler in _FourStepScheduler.instances] == [
        4,
        4,
        4,
    ]
    assert [scheduler.set_call[2] for scheduler in _FourStepScheduler.instances] == [
        5.0,
        5.0,
        5.0,
    ]
    assert [scheduler.step_calls for scheduler in _FourStepScheduler.instances] == [
        [40.0, 30.0, 20.0],
        [40.0, 30.0, 20.0],
        [40.0, 30.0, 20.0],
    ]

    kv_pos, kv_neg = pipeline.built_kv_pairs[0]
    for cache in (kv_pos[0], kv_neg[0]):
        assert cache["k"].shape[1] == cache["v"].shape[1] == 17
        assert int(cache["global_end_index"].item()) == 25
        assert int(cache["local_end_index"].item()) == 17
        assert int(cache["pinned_start"].item()) == -1
        assert int(cache["pinned_len"].item()) == 0

    trace = result.trace
    assert trace["schema"] == STAGE1_ROLLOUT_TRACE_SCHEMA
    assert trace["profile"]["chunk_size"] == 8
    assert trace["profile"]["window_size"] == 17
    assert trace["profile"]["global_sink_size"] == 1
    assert trace["profile"]["sampling_steps"] == 4
    assert len(trace["profile_sha256"]) == 64
    assert trace["cache_policy"] == {
        "global_sink": "input_first_latent_t0_clean_commit",
        "noisy_current_chunk_commit_self_kv": False,
        "clean_x0_recache_commit_self_kv": True,
        "history_source": "previous_model_generated_clean_x0",
        "pinned_sink_enabled": False,
    }
    assert trace["cfg"]["enabled"] is True
    assert trace["cfg"]["conditional_cache_branches"] == 2
    assert trace["cfg"]["negative_prompt"] == DEFAULT_NEGATIVE_PROMPT
    assert trace["generator_call_budget"] == {"per_branch": 16, "all_branches": 32}
    assert trace["training_context"] == {
        "teacher_forcing_weights_verified": True,
        "teacher_forcing_evidence": "checkpoint_preflight",
        "rollout_semantics": "self_rollout",
        "self_rollout_is_deployment_ood": True,
        "training_block_size": 8,
        "chunk_size_matches_training_block": True,
        "default_profile_topology": True,
        "window_topology_was_teacher_forced": False,
    }
    assert trace["checkpoint_provenance"] == {
        "runtime_mode": "dynamic_stage1_lora",
        "adapter": {"variant": "ema"},
        "checkpoint": {
            "teacher_forcing": True,
            "training_block_size": 8,
        },
    }
    assert trace["output"]["full_latent_shape"] == [1, 25, 1, 1, 1]
    assert trace["output"]["video_shape"] == [1, 97, 3, 2, 2]
    assert trace["sink_cache_after_preload"]["positive"]["global_end_index"] == 1
    assert trace["sink_cache_after_preload"]["positive"]["local_end_index"] == 1
    assert [chunk["cache_before_global_end_index"] for chunk in trace["chunks"]] == [
        1,
        9,
        17,
    ]
    assert [chunk["cache_before_local_end_index"] for chunk in trace["chunks"]] == [
        1,
        9,
        17,
    ]
    assert [
        chunk["cache_after"]["positive"]["global_end_index"]
        for chunk in trace["chunks"]
    ] == [9, 17, 25]
    assert [
        chunk["cache_after"]["positive"]["local_end_index"] for chunk in trace["chunks"]
    ] == [9, 17, 17]
    assert all(
        chunk["cache_after"]["positive"]["capacity_frames"] == 17
        for chunk in trace["chunks"]
    )
    assert all(chunk["solver_update_calls"] == 3 for chunk in trace["chunks"])
    assert all(chunk["terminal_direct_x0"] is True for chunk in trace["chunks"])
    assert all(chunk["noisy_self_kv_commits"] == 0 for chunk in trace["chunks"])
    assert all(
        chunk["clean_self_kv_commits_per_branch"] == 1 for chunk in trace["chunks"]
    )
    assert _original_runtime_state(pipeline) == original_runtime


def test_cfg_one_uses_only_positive_branch_and_sixteen_generator_calls(monkeypatch):
    pipeline = _make_pipeline(monkeypatch)
    initial, noise = _inputs()

    result = run_stage1_rollout(
        pipeline,
        initial_latent=initial,
        future_noise=noise,
        prompts=["a cat runs"],
        profile={"sampling_steps": 4, "guidance_scale": 1.0},
    )

    assert len(pipeline.generator.calls) == 16
    assert {call["branch"] for call in pipeline.generator.calls} == {"positive"}
    assert pipeline.text_encoder.calls == [["a cat runs"]]
    assert torch.equal(result.future_latents, noise + 2.5)
    assert result.trace["cfg"]["enabled"] is False
    assert result.trace["cfg"]["conditional_cache_branches"] == 1
    assert result.trace["negative_conditioning_sha256"] is None
    assert result.trace["sink_cache_after_preload"]["negative"] is None
    assert result.trace["generator_call_budget"] == {
        "per_branch": 16,
        "all_branches": 16,
    }
    assert result.trace["training_context"] == {
        "teacher_forcing_weights_verified": False,
        "teacher_forcing_evidence": "unverified",
        "rollout_semantics": "self_rollout",
        "self_rollout_is_deployment_ood": None,
        "training_block_size": None,
        "chunk_size_matches_training_block": None,
        "default_profile_topology": True,
        "window_topology_was_teacher_forced": None,
    }
    kv_pos, kv_neg = pipeline.built_kv_pairs[0]
    assert int(kv_pos[0]["global_end_index"].item()) == 25
    assert int(kv_neg[0]["global_end_index"].item()) == 0


def test_rollout_restores_originally_missing_timesteps_attribute(monkeypatch):
    pipeline = _make_pipeline(monkeypatch)
    delattr(pipeline, "timesteps")
    initial, noise = _inputs()

    run_stage1_rollout(
        pipeline,
        initial_latent=initial,
        future_noise=noise,
        prompts=["a cat runs"],
        profile={"sampling_steps": 4, "guidance_scale": 1.0},
    )

    assert not hasattr(pipeline, "timesteps")


def test_generator_failure_restores_pipeline_dit_and_attention_runtime(monkeypatch):
    pipeline = _make_pipeline(monkeypatch)
    initial, noise = _inputs()
    original_runtime = _original_runtime_state(pipeline)
    pipeline.generator.raise_on_call = 3

    with pytest.raises(RuntimeError, match="injected rollout generator failure"):
        run_stage1_rollout(
            pipeline,
            initial_latent=initial,
            future_noise=noise,
            prompts=["a cat runs"],
            profile={"sampling_steps": 4},
        )

    assert pipeline.generator.failure_runtime == {
        "dit_local_attn_size": 17,
        "dit_num_frame_per_block": 8,
        "attention_max_attention_size": 17,
        "attention_sink_size": 1,
        "attention_global_sink_size": 1,
    }
    assert _original_runtime_state(pipeline) == original_runtime
    assert len(pipeline.vae.decode_inputs) == 0


def test_concurrent_profiles_snapshot_and_restore_inside_rollout_lock(monkeypatch):
    pipeline = _make_pipeline(monkeypatch)
    initial, noise = _inputs()
    original_runtime = _original_runtime_state(pipeline)
    entered_encoder = threading.Event()
    release_encoder = threading.Event()

    class TrackingLock:
        def __init__(self):
            self._lock = threading.RLock()
            self._state_lock = threading.Lock()
            self._owner = None
            self.contended = threading.Event()

        def __enter__(self):
            identity = threading.get_ident()
            with self._state_lock:
                if self._owner is not None and self._owner != identity:
                    self.contended.set()
            self._lock.acquire()
            with self._state_lock:
                self._owner = identity
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            with self._state_lock:
                self._owner = None
            self._lock.release()

    delegate_encoder = pipeline.text_encoder

    class BlockingTextEncoder:
        def __call__(self, *, text_prompts):
            if list(text_prompts) == ["first"]:
                entered_encoder.set()
                if not release_encoder.wait(timeout=5):
                    raise RuntimeError("timed out waiting to release first rollout")
            return delegate_encoder(text_prompts=text_prompts)

    tracking_lock = TrackingLock()
    pipeline._continuation_lock = tracking_lock
    pipeline.text_encoder = BlockingTextEncoder()

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(
            run_stage1_rollout,
            pipeline,
            initial_latent=initial,
            future_noise=noise,
            prompts=["first"],
            profile={"sampling_steps": 4},
        )
        assert entered_encoder.wait(timeout=5)
        second = executor.submit(
            run_stage1_rollout,
            pipeline,
            initial_latent=initial,
            future_noise=noise,
            prompts=["second"],
            profile={
                "chunk_size": 4,
                "window_size": 9,
                "sampling_steps": 4,
            },
        )
        try:
            assert tracking_lock.contended.wait(timeout=5)
        finally:
            release_encoder.set()
        assert first.result(timeout=5).video.shape[1] == 97
        assert second.result(timeout=5).video.shape[1] == 97

    assert _original_runtime_state(pipeline) == original_runtime


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (
            lambda initial, noise, prompts: (
                initial.repeat(1, 2, 1, 1, 1),
                noise,
                prompts,
            ),
            ValueError,
            "exactly the one-frame sink",
        ),
        (
            lambda initial, noise, prompts: (initial, noise[:, :-1], prompts),
            ValueError,
            "future_noise shape mismatch",
        ),
        (
            lambda initial, noise, prompts: (initial, noise.double(), prompts),
            TypeError,
            "dtype/device must match",
        ),
        (
            lambda initial, noise, prompts: (
                initial,
                noise.clone().index_fill_(1, torch.tensor([0]), float("nan")),
                prompts,
            ),
            ValueError,
            "non-finite",
        ),
        (
            lambda initial, noise, _prompts: (initial, noise, []),
            ValueError,
            "one non-empty string",
        ),
        (
            lambda initial, noise, _prompts: (initial, noise, "a cat runs"),
            TypeError,
            "one string per batch item",
        ),
    ],
)
def test_input_counterexamples_fail_before_cache_or_scheduler(
    monkeypatch, mutate, error, message
):
    pipeline = _make_pipeline(monkeypatch)
    initial, noise = _inputs()
    initial, noise, prompts = mutate(initial, noise, ["a cat runs"])

    with pytest.raises(error, match=message):
        Stage1RolloutPipeline(pipeline, {"sampling_steps": 4}).run(
            initial_latent=initial,
            future_noise=noise,
            prompts=prompts,
        )

    assert pipeline.generator.calls == []
    assert pipeline.built_kv_pairs == []
    assert _FourStepScheduler.instances == []


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("quantize_kv", True, "unquantized self-KV"),
        ("multi_shot_sink", True, "incompatible with pipeline.multi_shot_sink"),
        ("streaming_vae", True, "incompatible with pipeline.streaming_vae"),
        ("_active_continuation_session", object(), "during continuation"),
    ],
)
def test_incompatible_runtime_is_rejected_before_mutation(
    monkeypatch, attribute, value, message
):
    pipeline = _make_pipeline(monkeypatch)
    initial, noise = _inputs()
    original_runtime = _original_runtime_state(pipeline)
    setattr(pipeline, attribute, value)

    with pytest.raises((ValueError, RuntimeError), match=message):
        run_stage1_rollout(
            pipeline,
            initial_latent=initial,
            future_noise=noise,
            prompts=["a cat runs"],
            profile={"sampling_steps": 4},
        )

    assert _original_runtime_state(pipeline) == original_runtime
    assert pipeline.generator.calls == []
    assert pipeline.built_kv_pairs == []
