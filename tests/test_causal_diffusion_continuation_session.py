from __future__ import annotations

import threading
from types import MethodType, SimpleNamespace

import pytest
import torch

import pipeline.causal_diffusion_inference as causal_pipeline_module
from pipeline.causal_diffusion_continuation import (
    audit_untruncated_prompt,
    tensor_identity_sha256,
)
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline

FRAME_SEQ_LENGTH = 3
LATENT_HEIGHT = 2
LATENT_WIDTH = 6


class CausalWanSelfAttention(torch.nn.Module):
    def __init__(self, frame_seq_length: int):
        super().__init__()
        self.local_attn_size = 24
        self.sink_size = 0
        self.global_sink_size = 0
        self.max_attention_size = 24 * frame_seq_length


class _TinyDit(torch.nn.Module):
    def __init__(self, layers: int, frame_seq_length: int):
        super().__init__()
        self.local_attn_size = -1
        self.t_scale = 1.0
        self.rope_method = "linear"
        self.original_seq_len = None
        self.use_relative_rope = False
        self.rope_temporal_offset = 0.0
        self.sink_size = 0
        self.global_sink_size = 0
        self.attentions = torch.nn.ModuleList(
            [CausalWanSelfAttention(frame_seq_length) for _ in range(layers)]
        )


class _RecordingGenerator(torch.nn.Module):
    def __init__(self, layers: int, frame_seq_length: int):
        super().__init__()
        self.model = _TinyDit(layers, frame_seq_length)
        self.frame_seq_length = frame_seq_length
        self.calls: list[dict[str, object]] = []
        self.raise_on_call: int | None = None

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
    ):
        del crossattn_cache
        if self.raise_on_call is not None and len(self.calls) == self.raise_on_call:
            raise RuntimeError("injected generator failure")
        branch = kv_cache[0]["kind"]
        self.calls.append(
            {
                "branch": branch,
                "before_global": int(kv_cache[0]["global_end_index"].item()),
                "before_local": int(kv_cache[0]["local_end_index"].item()),
                "current_start": int(current_start),
                "cache_start": int(cache_start),
                "prompt_value": int(
                    conditional_dict["prompt_embeds"].reshape(-1)[0].item()
                ),
                "prompt_tensor_id": id(conditional_dict["prompt_embeds"]),
                "kv_tensor_id": id(kv_cache[0]["k"]),
                "input": noisy_image_or_video.detach().clone(),
                "timestep": timestep.detach().clone(),
            }
        )

        num_new_frames = int(noisy_image_or_video.shape[1])
        num_new = num_new_frames * self.frame_seq_length
        current_end = int(current_start) + num_new
        sink_tokens = int(self.model.attentions[0].sink_size) * self.frame_seq_length
        values = (
            noisy_image_or_video[:, :, :1, :1, :1]
            .reshape(1, num_new_frames, 1, 1)
            .repeat_interleave(self.frame_seq_length, dim=1)
        )
        capacity = 24 * self.frame_seq_length
        for cache in kv_cache:
            local_end_before = int(cache["local_end_index"].item())
            global_end_before = int(cache["global_end_index"].item())
            if (
                current_end > global_end_before
                and local_end_before + num_new > capacity
            ):
                evicted = local_end_before + num_new - capacity
                rolled = local_end_before - evicted - sink_tokens
                if rolled > 0:
                    cache["k"][:, sink_tokens : sink_tokens + rolled] = cache["k"][
                        :, sink_tokens + evicted : sink_tokens + evicted + rolled
                    ].clone()
                    cache["v"][:, sink_tokens : sink_tokens + rolled] = cache["v"][
                        :, sink_tokens + evicted : sink_tokens + evicted + rolled
                    ].clone()
                local_end = capacity
            elif current_end > global_end_before:
                local_end = local_end_before + num_new
            else:
                local_end = local_end_before
            local_start = local_end - num_new
            cache["k"][:, local_start:local_end] = values
            cache["v"][:, local_start:local_end] = values
            cache["global_end_index"].fill_(current_end)
            cache["local_end_index"].fill_(local_end)
        return torch.zeros_like(noisy_image_or_video), None


class _TokenizerBackend:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def __call__(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        lengths = []
        for text in texts:
            if text.startswith("tokens="):
                lengths.append(int(text.split("=", 1)[1]))
            else:
                lengths.append(max(2, len(text)))
        return SimpleNamespace(input_ids=[list(range(length)) for length in lengths])


class _TokenizerWrapper:
    def __init__(self):
        self.seq_len = 512
        self.clean = "whitespace"
        self.tokenizer = _TokenizerBackend()

    def _clean(self, text):
        return " ".join(text.split())


class _RecordingTextEncoder:
    VALUES = {"action-a": 11, "hold": 22, "action-b": 33, "negative": -1}

    def __init__(self):
        self.tokenizer = _TokenizerWrapper()
        self.calls: list[list[str]] = []

    def __call__(self, *, text_prompts):
        prompts = list(text_prompts)
        self.calls.append(prompts)
        values = [self.VALUES.get(prompt, 44) for prompt in prompts]
        return {
            "prompt_embeds": torch.tensor(values, dtype=torch.float32).reshape(
                len(values), 1, 1
            )
        }


class _TinyScheduler:
    instances: list["_TinyScheduler"] = []

    def __init__(self, **kwargs):
        self.constructor_kwargs = kwargs
        self.timesteps = None
        self.set_call = None
        self.__class__.instances.append(self)

    def set_timesteps(self, steps, *, device, shift):
        self.set_call = (steps, device, shift)
        self.timesteps = torch.tensor([2.0, 1.0], device=device)

    def step(self, _flow, _timestep, sample, *, return_dict):
        assert return_dict is False
        return (sample + 1.0,)


class _RecordingVAE:
    def __init__(self):
        self.decode_inputs: list[torch.Tensor] = []

    def decode_to_pixel(self, latents):
        self.decode_inputs.append(latents.detach().clone())
        frames = 1 + (latents.shape[1] - 1) * 4
        return torch.zeros(
            latents.shape[0],
            frames,
            3,
            2,
            2,
            dtype=latents.dtype,
            device=latents.device,
        )


def _cache(kind: str, layers: int, dtype, device, frame_seq_length: int):
    caches = []
    for _ in range(layers):
        caches.append(
            {
                "kind": kind,
                "k": torch.zeros(
                    1, 24 * frame_seq_length, 1, 1, dtype=dtype, device=device
                ),
                "v": torch.zeros(
                    1, 24 * frame_seq_length, 1, 1, dtype=dtype, device=device
                ),
                "quantized": False,
                "block_token_size": 8 * frame_seq_length,
                "max_blocks": 3,
                "num_heads": 1,
                "num_filled_blocks": 0,
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
                "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
            }
        )
    return caches


def _cross_cache(layers: int, dtype, device):
    return [
        {
            "k": torch.zeros(1, 512, 1, 1, dtype=dtype, device=device),
            "v": torch.zeros(1, 512, 1, 1, dtype=dtype, device=device),
            "is_init": False,
        }
        for _ in range(layers)
    ]


def _make_pipeline(monkeypatch, *, layers=2):
    _TinyScheduler.instances.clear()
    monkeypatch.setattr(
        causal_pipeline_module,
        "FlowUniPCMultistepScheduler",
        _TinyScheduler,
    )
    pipeline = CausalDiffusionInferencePipeline.__new__(
        CausalDiffusionInferencePipeline
    )
    torch.nn.Module.__init__(pipeline)
    pipeline.frame_seq_length = FRAME_SEQ_LENGTH
    pipeline.num_frame_per_block = 8
    pipeline.num_transformer_blocks = layers
    pipeline.num_train_timesteps = 1000
    pipeline.sampling_steps = 50
    pipeline.sample_solver = "unipc"
    pipeline.shift = 5.0
    pipeline.independent_first_frame = True
    pipeline.guidance_scale = 5.0
    pipeline.negative_prompt = "negative"
    pipeline.local_attn_size = -1
    pipeline.sink_size = 0
    pipeline.multi_shot_sink = False
    pipeline.shot_clean_recache = False
    pipeline.global_sink_size = 0
    pipeline.scene_cut_prefix = "The scene transitions. "
    pipeline.multi_shot_rope_offset = 0.0
    pipeline.inference_t_scale = None
    pipeline.use_relative_rope = False
    pipeline._rope_method_override = None
    pipeline._original_seq_len_override = None
    pipeline.streaming_vae = False
    pipeline.vae_device = None
    pipeline.async_vae = False
    pipeline.quantize_kv = False
    pipeline.kv_cache_pos = None
    pipeline.kv_cache_neg = None
    pipeline.crossattn_cache_pos = None
    pipeline.crossattn_cache_neg = None
    pipeline._continuation_lock = None
    pipeline._active_continuation_session = None
    pipeline.generator = _RecordingGenerator(layers, FRAME_SEQ_LENGTH)
    pipeline.text_encoder = _RecordingTextEncoder()
    pipeline.vae = _RecordingVAE()

    def build_kv(self, batch_size, dtype, device):
        assert batch_size == 1
        return _cache("positive", layers, dtype, device, FRAME_SEQ_LENGTH), _cache(
            "negative", layers, dtype, device, FRAME_SEQ_LENGTH
        )

    def build_cross(self, batch_size, dtype, device):
        assert batch_size == 1
        return _cross_cache(layers, dtype, device), _cross_cache(layers, dtype, device)

    pipeline._build_kv_cache = MethodType(build_kv, pipeline)
    pipeline._build_crossattn_cache = MethodType(build_cross, pipeline)
    return pipeline


def _run_three_segments(pipeline, *, sink_size):
    initial = torch.full((1, 1, 1, LATENT_HEIGHT, LATENT_WIDTH), 99.0)
    noise_plan = torch.arange(64, dtype=torch.float32).reshape(1, 64, 1, 1, 1)
    noise_plan = noise_plan.expand(1, 64, 1, LATENT_HEIGHT, LATENT_WIDTH).clone()
    session = pipeline.begin_session(
        initial_latent=initial.clone(),
        sink_size=sink_size,
        noise_plan=noise_plan.clone(),
    )
    action_a = session.generate_segment(
        "action-a",
        noise=noise_plan[:, 0:24].clone(),
        segment_name="action_a",
    )
    assert session.cursor == 24
    hold = session.generate_segment(
        "hold",
        noise=noise_plan[:, 24:40].clone(),
        segment_name="hold",
    )
    assert session.cursor == 40
    action_b = session.generate_segment(
        "action-b",
        noise=noise_plan[:, 40:64].clone(),
        carry_last_latent_as_anchor=True,
        segment_name="action_b",
    )
    assert session.cursor == 64
    return session, initial, noise_plan, action_a, hold, action_b


def test_three_segment_session_preserves_cursor_prompts_anchors_and_decodes_once(
    monkeypatch,
):
    pipeline = _make_pipeline(monkeypatch)
    session, initial, noise, action_a, hold, action_b = _run_three_segments(
        pipeline, sink_size=0
    )

    assert action_a.shape[1] == 24
    assert hold.shape[1] == 16
    assert action_b.shape[1] == 24
    assert torch.equal(action_a[:, :1], initial)
    assert torch.equal(action_a[:, 1:], noise[:, 1:24] + 2)
    assert torch.equal(hold, noise[:, 24:40] + 2)
    assert torch.equal(action_b[:, :1], hold[:, -1:])
    assert torch.equal(action_b[:, 1:], noise[:, 41:64] + 2)

    result = session.finish()
    assert session.state == "finished"
    assert result.latents.shape == (
        1,
        64,
        1,
        LATENT_HEIGHT,
        LATENT_WIDTH,
    )
    assert result.video.shape == (1, 253, 3, 2, 2)
    assert len(pipeline.vae.decode_inputs) == 1
    assert pipeline.vae.decode_inputs[0].shape[1] == 64
    assert pipeline._active_continuation_session is None

    blocks = result.trace["blocks"]
    assert [block["global_end_frames"] for block in blocks] == [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
    ]
    assert [block["local_end_frames"] for block in blocks] == [
        8,
        16,
        24,
        24,
        24,
        24,
        24,
        24,
    ]
    assert [block["global_end_tokens"] for block in blocks] == [
        FRAME_SEQ_LENGTH * value for value in range(8, 65, 8)
    ]
    assert [block["local_end_tokens"] for block in blocks] == [
        FRAME_SEQ_LENGTH * value for value in (8, 16, 24, 24, 24, 24, 24, 24)
    ]
    assert all(
        block["cache_capacity_tokens"] == 24 * FRAME_SEQ_LENGTH for block in blocks
    )
    assert result.trace["anchors"] == [
        {"kind": "initial", "source": "initial_latent", "destination": 0},
        {"kind": "soft_reanchor", "source": 39, "destination": 40},
    ]
    assert result.trace["noise_identity_sha256"] == tensor_identity_sha256(noise)

    positive = [
        call for call in pipeline.generator.calls if call["branch"] == "positive"
    ]
    negative = [
        call for call in pipeline.generator.calls if call["branch"] == "negative"
    ]
    assert {call["prompt_value"] for call in positive} == {11, 22, 33}
    assert {call["prompt_value"] for call in negative} == {-1}
    assert len({call["prompt_tensor_id"] for call in negative}) == 1
    assert len({call["kv_tensor_id"] for call in positive}) == 1
    assert len({call["kv_tensor_id"] for call in negative}) == 1
    assert [
        next(
            call["before_global"]
            for call in positive
            if call["current_start"] == block_start
        )
        for block_start in range(0, 64 * FRAME_SEQ_LENGTH, 8 * FRAME_SEQ_LENGTH)
    ] == list(range(0, 64 * FRAME_SEQ_LENGTH, 8 * FRAME_SEQ_LENGTH))
    assert sorted(
        {(call["current_start"], call["cache_start"]) for call in positive}
    ) == [
        (value, value)
        for value in range(0, 64 * FRAME_SEQ_LENGTH, 8 * FRAME_SEQ_LENGTH)
    ]
    assert pipeline.text_encoder.calls == [
        ["negative"],
        ["action-a"],
        ["hold"],
        ["action-b"],
    ]


def test_initial_and_soft_anchor_are_clamped_before_and_after_every_step(
    monkeypatch,
):
    original_overwrite = causal_pipeline_module._overwrite_i2v_context
    overwrite_calls = []

    def recording_overwrite(value, anchor, context_frames):
        output = original_overwrite(value, anchor, context_frames)
        overwrite_calls.append(
            {
                "anchor": anchor.detach().clone(),
                "output": output.detach().clone(),
                "context_frames": context_frames,
            }
        )
        return output

    monkeypatch.setattr(
        causal_pipeline_module,
        "_overwrite_i2v_context",
        recording_overwrite,
    )
    pipeline = _make_pipeline(monkeypatch)
    session, initial, _noise, _action_a, hold, _action_b = _run_three_segments(
        pipeline, sink_size=0
    )

    # Two anchored blocks × (2 pre-step + 2 post-step + 1 final) clamps.
    assert len(overwrite_calls) == 10
    initial_calls = [
        call for call in overwrite_calls if torch.equal(call["anchor"], initial)
    ]
    soft_anchor = hold[:, -1:]
    soft_calls = [
        call for call in overwrite_calls if torch.equal(call["anchor"], soft_anchor)
    ]
    assert len(initial_calls) == 5
    assert len(soft_calls) == 5
    for call in overwrite_calls:
        assert call["context_frames"] == 1
        assert torch.equal(call["output"][:, :1], call["anchor"])

    positive_calls = [
        call for call in pipeline.generator.calls if call["branch"] == "positive"
    ]
    for block_index in range(8):
        current_start = block_index * 8 * FRAME_SEQ_LENGTH
        block_calls = [
            call for call in positive_calls if call["current_start"] == current_start
        ]
        assert len(block_calls) == 3
        for sampling_call in block_calls[:2]:
            timestep = sampling_call["timestep"]
            if block_index in (0, 5):
                assert torch.equal(timestep[:, :1], torch.zeros_like(timestep[:, :1]))
                assert torch.all(timestep[:, 1:] > 0)
            else:
                assert torch.all(timestep > 0)
        assert torch.equal(
            block_calls[-1]["timestep"],
            torch.zeros_like(block_calls[-1]["timestep"]),
        )
    session.finish()


def test_sink_variants_share_noise_identity_and_sink1_keeps_initial_cache(
    monkeypatch,
):
    pipeline0 = _make_pipeline(monkeypatch)
    session0, initial0, noise0, *_ = _run_three_segments(pipeline0, sink_size=0)
    sink0_cache_start = session0._bundle.kv_pos[0]["k"][:, :FRAME_SEQ_LENGTH].clone()
    trace0 = session0.trace_snapshot()
    session0.finish()

    pipeline1 = _make_pipeline(monkeypatch)
    session1, initial1, noise1, *_ = _run_three_segments(pipeline1, sink_size=1)
    trace1 = session1.trace_snapshot()
    expected_initial0 = (
        initial0[:, 0, 0, 0, 0].reshape(1, 1, 1, 1).expand_as(sink0_cache_start)
    )
    expected_initial1 = (
        initial1[:, 0, 0, 0, 0].reshape(1, 1, 1, 1).expand(1, FRAME_SEQ_LENGTH, 1, 1)
    )
    assert not torch.equal(sink0_cache_start, expected_initial0)
    for caches in (session1._bundle.kv_pos, session1._bundle.kv_neg):
        for cache in caches:
            assert torch.equal(cache["k"][:, :FRAME_SEQ_LENGTH], expected_initial1)
            assert torch.equal(cache["v"][:, :FRAME_SEQ_LENGTH], expected_initial1)
            assert int(cache["pinned_start"].item()) == -1
            assert int(cache["pinned_len"].item()) == 0
    assert trace0["noise_identity_sha256"] == trace1["noise_identity_sha256"]
    assert tensor_identity_sha256(noise0) == tensor_identity_sha256(noise1)
    assert all(block["pinned_start"] == -1 for block in trace1["blocks"])
    session1.finish()
    assert all(
        attention.sink_size == 0 for attention in pipeline1.generator.model.attentions
    )


@pytest.mark.parametrize("token_count", [512, 513])
def test_prompt_audit_rejects_truncation_boundary(token_count):
    text_encoder = _RecordingTextEncoder()
    with pytest.raises(ValueError, match="truncation boundary"):
        audit_untruncated_prompt(text_encoder, f"tokens={token_count}")
    call = text_encoder.tokenizer.tokenizer.calls[-1]
    assert call["add_special_tokens"] is True
    assert call["padding"] is False
    assert call["truncation"] is False


def test_prompt_audit_accepts_511_untruncated_tokens():
    text_encoder = _RecordingTextEncoder()
    audit = audit_untruncated_prompt(text_encoder, "tokens=511")
    assert audit["token_count"] == 511
    assert audit["truncation"] is False


def test_active_session_blocks_ordinary_inference_clear_and_second_session(
    monkeypatch,
):
    pipeline = _make_pipeline(monkeypatch)
    initial = torch.zeros(1, 1, 1, LATENT_HEIGHT, LATENT_WIDTH)
    plan = torch.zeros(1, 8, 1, LATENT_HEIGHT, LATENT_WIDTH)
    session = pipeline.begin_session(
        initial_latent=initial,
        sink_size=0,
        noise_plan=plan,
    )
    with pytest.raises(RuntimeError, match="ordinary inference"):
        pipeline.inference(
            noise=plan,
            text_prompts=[["action-a"]],
            initial_latent=initial,
            return_latents=True,
        )
    with pytest.raises(RuntimeError, match="cannot clear"):
        pipeline.clear_cache()
    with pytest.raises(RuntimeError, match="already has an active"):
        pipeline.begin_session(
            initial_latent=initial,
            sink_size=0,
            noise_plan=plan,
        )
    session.abort("test cleanup")
    assert session.state == "failed"
    assert pipeline._active_continuation_session is None

    replacement = pipeline.begin_session(
        initial_latent=initial,
        sink_size=0,
        noise_plan=plan,
    )
    replacement.abort("replacement cleanup")


def test_ordinary_inference_holds_exclusive_lock_until_it_finishes(monkeypatch):
    pipeline = _make_pipeline(monkeypatch)
    initial = torch.zeros(1, 1, 1, LATENT_HEIGHT, LATENT_WIDTH)
    plan = torch.zeros(1, 8, 1, LATENT_HEIGHT, LATENT_WIDTH)
    ordinary_started = threading.Event()
    release_ordinary = threading.Event()
    session_started = threading.Event()
    errors = []
    sessions = []

    def blocking_independent(self, **kwargs):
        ordinary_started.set()
        if not release_ordinary.wait(timeout=2):
            raise RuntimeError("test timed out waiting to release ordinary inference")
        return kwargs["noise"]

    pipeline._inference_independent = MethodType(blocking_independent, pipeline)

    def run_ordinary():
        try:
            pipeline.inference(
                noise=plan,
                text_prompts=[["action-a"]],
                initial_latent=initial,
                return_latents=True,
            )
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    def begin_after_ordinary():
        try:
            session = pipeline.begin_session(
                initial_latent=initial,
                sink_size=0,
                noise_plan=plan,
            )
            sessions.append(session)
            session_started.set()
        except Exception as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    ordinary_thread = threading.Thread(target=run_ordinary)
    session_thread = threading.Thread(target=begin_after_ordinary)
    ordinary_thread.start()
    assert ordinary_started.wait(timeout=1)
    session_thread.start()
    assert not session_started.wait(timeout=0.05)
    release_ordinary.set()
    ordinary_thread.join(timeout=2)
    session_thread.join(timeout=2)

    assert not ordinary_thread.is_alive()
    assert not session_thread.is_alive()
    assert errors == []
    assert len(sessions) == 1
    sessions[0].abort("test cleanup")


@pytest.mark.parametrize(
    "failure_kind",
    [
        "non_aligned",
        "wrong_shape",
        "soft_anchor_empty",
        "cfg_changed",
        "sink_changed",
        "sampling_steps_changed",
        "negative_prompt_changed",
        "t_scale_changed",
        "original_seq_len_changed",
        "rope_offset_changed",
        "cache_drift",
        "negative_cache_drift",
        "local_cache_drift",
        "pinned_cache_drift",
        "wrong_noise_slice",
        "generator_failure",
    ],
)
def test_session_mismatches_and_generation_errors_poison_state(
    monkeypatch,
    failure_kind,
):
    pipeline = _make_pipeline(monkeypatch)
    initial = torch.zeros(1, 1, 1, LATENT_HEIGHT, LATENT_WIDTH)
    plan = torch.arange(16, dtype=torch.float32).reshape(1, 16, 1, 1, 1)
    plan = plan.expand(1, 16, 1, LATENT_HEIGHT, LATENT_WIDTH).clone()
    session = pipeline.begin_session(
        initial_latent=initial,
        sink_size=0,
        noise_plan=plan,
    )
    if failure_kind == "non_aligned":
        noise = plan[:, :7]
        kwargs = {}
    elif failure_kind == "wrong_shape":
        noise = torch.zeros(1, 8, 2, LATENT_HEIGHT, LATENT_WIDTH)
        kwargs = {}
    elif failure_kind == "soft_anchor_empty":
        noise = plan[:, :8]
        kwargs = {"carry_last_latent_as_anchor": True}
    elif failure_kind == "cfg_changed":
        pipeline.guidance_scale = 4.0
        noise = plan[:, :8]
        kwargs = {}
    elif failure_kind == "sink_changed":
        pipeline.sink_size = 1
        noise = plan[:, :8]
        kwargs = {}
    elif failure_kind == "sampling_steps_changed":
        pipeline.sampling_steps = 49
        noise = plan[:, :8]
        kwargs = {}
    elif failure_kind == "negative_prompt_changed":
        pipeline.negative_prompt = "changed"
        noise = plan[:, :8]
        kwargs = {}
    elif failure_kind == "t_scale_changed":
        pipeline._dit_model.t_scale = 2.0
        noise = plan[:, :8]
        kwargs = {}
    elif failure_kind == "original_seq_len_changed":
        pipeline._dit_model.original_seq_len = 24
        noise = plan[:, :8]
        kwargs = {}
    elif failure_kind == "rope_offset_changed":
        pipeline._dit_model.rope_temporal_offset = 1.0
        noise = plan[:, :8]
        kwargs = {}
    elif failure_kind == "cache_drift":
        session._bundle.kv_pos[1]["global_end_index"].fill_(FRAME_SEQ_LENGTH)
        noise = plan[:, :8]
        kwargs = {}
    elif failure_kind == "negative_cache_drift":
        session._bundle.kv_neg[1]["global_end_index"].fill_(FRAME_SEQ_LENGTH)
        noise = plan[:, :8]
        kwargs = {}
    elif failure_kind == "local_cache_drift":
        session._bundle.kv_pos[1]["local_end_index"].fill_(FRAME_SEQ_LENGTH)
        noise = plan[:, :8]
        kwargs = {}
    elif failure_kind == "pinned_cache_drift":
        session._bundle.kv_pos[1]["pinned_start"].fill_(0)
        session._bundle.kv_pos[1]["pinned_len"].fill_(FRAME_SEQ_LENGTH)
        noise = plan[:, :8]
        kwargs = {}
    elif failure_kind == "wrong_noise_slice":
        noise = plan[:, 8:16]
        kwargs = {}
    elif failure_kind == "generator_failure":
        pipeline.generator.raise_on_call = 1
        noise = plan[:, :8]
        kwargs = {}
    else:  # pragma: no cover
        raise AssertionError(failure_kind)

    with pytest.raises((ValueError, RuntimeError)):
        session.generate_segment("action-a", noise=noise, **kwargs)
    assert session.state == "failed"
    assert session.trace_snapshot()["failure"] is not None
    assert pipeline._active_continuation_session is None
    with pytest.raises(RuntimeError, match="session is failed"):
        session.generate_segment("action-a", noise=plan[:, :8])


def test_finish_empty_session_poisoned_and_finished_session_cannot_continue(
    monkeypatch,
):
    pipeline = _make_pipeline(monkeypatch)
    initial = torch.zeros(1, 1, 1, LATENT_HEIGHT, LATENT_WIDTH)
    plan = torch.zeros(1, 8, 1, LATENT_HEIGHT, LATENT_WIDTH)
    empty = pipeline.begin_session(
        initial_latent=initial,
        sink_size=0,
        noise_plan=plan,
    )
    with pytest.raises(RuntimeError, match="empty"):
        empty.finish()
    assert empty.state == "failed"

    complete = pipeline.begin_session(
        initial_latent=initial,
        sink_size=0,
        noise_plan=plan,
    )
    complete.generate_segment("action-a", noise=plan)
    complete.finish()
    with pytest.raises(RuntimeError, match="session is finished"):
        complete.generate_segment("action-a", noise=plan)


def test_finish_rejects_incomplete_noise_plan_and_decode_failure_poisoned(
    monkeypatch,
):
    pipeline = _make_pipeline(monkeypatch)
    initial = torch.zeros(1, 1, 1, LATENT_HEIGHT, LATENT_WIDTH)
    plan = torch.zeros(1, 16, 1, LATENT_HEIGHT, LATENT_WIDTH)
    incomplete = pipeline.begin_session(
        initial_latent=initial,
        sink_size=0,
        noise_plan=plan,
    )
    incomplete.generate_segment("action-a", noise=plan[:, :8])
    with pytest.raises(RuntimeError, match="complete locked noise_plan"):
        incomplete.finish()
    assert incomplete.state == "failed"
    assert pipeline._active_continuation_session is None

    short_plan = plan[:, :8].clone()
    decode_failure = pipeline.begin_session(
        initial_latent=initial,
        sink_size=0,
        noise_plan=short_plan,
    )
    decode_failure.generate_segment("action-a", noise=short_plan)

    def raise_decode(_latents):
        raise RuntimeError("injected decode failure")

    monkeypatch.setattr(pipeline.vae, "decode_to_pixel", raise_decode)
    with pytest.raises(RuntimeError, match="injected decode failure"):
        decode_failure.finish()
    assert decode_failure.state == "failed"
    assert pipeline._active_continuation_session is None


def test_finished_or_failed_session_restores_ordinary_inference(monkeypatch):
    pipeline = _make_pipeline(monkeypatch)
    initial = torch.zeros(1, 1, 1, LATENT_HEIGHT, LATENT_WIDTH)
    plan = torch.zeros(1, 8, 1, LATENT_HEIGHT, LATENT_WIDTH)

    finished = pipeline.begin_session(
        initial_latent=initial,
        sink_size=0,
        noise_plan=plan,
    )
    finished.generate_segment("action-a", noise=plan)
    finished.finish()
    ordinary_after_finish = pipeline.inference(
        noise=plan,
        text_prompts=[["action-a"]],
        initial_latent=initial,
        return_latents=True,
    )
    assert ordinary_after_finish.shape == plan.shape

    failed = pipeline.begin_session(
        initial_latent=initial,
        sink_size=0,
        noise_plan=plan,
    )
    failed.abort("intentional failure")
    ordinary_after_failure = pipeline.inference(
        noise=plan,
        text_prompts=[["action-a"]],
        initial_latent=initial,
        return_latents=True,
    )
    assert ordinary_after_failure.shape == plan.shape


def test_tensor_identity_hash_binds_dtype_shape_bytes_and_session_clones_inputs(
    monkeypatch,
):
    base = torch.arange(8, dtype=torch.float32)
    assert tensor_identity_sha256(base) != tensor_identity_sha256(base + 1)
    assert tensor_identity_sha256(base) != tensor_identity_sha256(base.reshape(2, 4))
    assert tensor_identity_sha256(base) != tensor_identity_sha256(
        base.view(torch.int32)
    )

    pipeline = _make_pipeline(monkeypatch)
    initial = torch.zeros(1, 1, 1, LATENT_HEIGHT, LATENT_WIDTH)
    plan = torch.zeros(1, 8, 1, LATENT_HEIGHT, LATENT_WIDTH)
    expected_initial_hash = tensor_identity_sha256(initial)
    expected_noise_hash = tensor_identity_sha256(plan)
    session = pipeline.begin_session(
        initial_latent=initial,
        sink_size=0,
        noise_plan=plan,
    )
    initial.fill_(7)
    plan.fill_(8)
    locked_noise = session._noise_plan.clone()
    generated = session.generate_segment("action-a", noise=locked_noise)
    result = session.finish()
    assert torch.equal(generated[:, :1], torch.zeros_like(generated[:, :1]))
    assert result.trace["initial_latent_sha256"] == expected_initial_hash
    assert result.trace["noise_identity_sha256"] == expected_noise_hash


@pytest.mark.parametrize(
    "initial",
    [
        torch.zeros(2, 1, 1, LATENT_HEIGHT, LATENT_WIDTH),
        torch.zeros(1, 2, 1, LATENT_HEIGHT, LATENT_WIDTH),
        torch.full((1, 1, 1, LATENT_HEIGHT, LATENT_WIDTH), float("nan")),
    ],
)
def test_begin_session_rejects_batch_shape_or_nonfinite_initial(monkeypatch, initial):
    pipeline = _make_pipeline(monkeypatch)
    with pytest.raises(ValueError):
        pipeline.begin_session(initial_latent=initial, sink_size=0)
    assert pipeline._active_continuation_session is None
