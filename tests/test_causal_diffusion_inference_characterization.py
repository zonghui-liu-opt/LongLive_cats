from __future__ import annotations

import inspect
from types import MethodType

import torch

import pipeline.causal_diffusion_inference as causal_pipeline_module
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline


class _TinyDit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.local_attn_size = -1
        self.t_scale = 1.0
        self.rope_method = "linear"
        self.original_seq_len = None
        self.use_relative_rope = False
        self.rope_temporal_offset = 0.0
        self.max_attention_size = 24
        self.sink_size = 0
        self.global_sink_size = 0


class _RecordingGenerator(torch.nn.Module):
    def __init__(self, frame_seq_length: int):
        super().__init__()
        self.model = _TinyDit()
        self.frame_seq_length = frame_seq_length
        self.calls: list[dict[str, object]] = []

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
        cache = kv_cache[0]
        self.calls.append(
            {
                "kind": cache["kind"],
                "before_global": int(cache["global_end_index"].item()),
                "before_local": int(cache["local_end_index"].item()),
                "current_start": int(current_start),
                "cache_start": int(cache_start),
                "input": noisy_image_or_video.detach().clone(),
                "timestep": timestep.detach().clone(),
                "prompt_id": int(
                    conditional_dict["prompt_embeds"].reshape(-1)[0].item()
                ),
                "crossattn_was_initialized": bool(crossattn_cache[0]["is_init"]),
            }
        )
        current_end = current_start + (
            noisy_image_or_video.shape[1] * self.frame_seq_length
        )
        cache["global_end_index"].fill_(current_end)
        cache["local_end_index"].fill_(min(current_end, 4 * self.frame_seq_length))
        crossattn_cache[0]["is_init"] = True
        return torch.zeros_like(noisy_image_or_video), None


class _RecordingTextEncoder:
    _PROMPT_IDS = {"action-a": 11, "action-b": 22, "negative": -1}

    def __init__(self):
        self.calls: list[list[str]] = []

    def __call__(self, *, text_prompts):
        prompts = list(text_prompts)
        self.calls.append(prompts)
        values = [self._PROMPT_IDS[prompt] for prompt in prompts]
        return {
            "prompt_embeds": torch.tensor(values, dtype=torch.float32).reshape(
                len(values), 1, 1
            )
        }


class _RecordingScheduler:
    instances: list["_RecordingScheduler"] = []

    def __init__(self, **kwargs):
        self.constructor_kwargs = kwargs
        self.set_timesteps_call = None
        self.timesteps = None
        self.step_timesteps: list[float] = []
        self.__class__.instances.append(self)

    def set_timesteps(self, steps, *, device, shift):
        self.set_timesteps_call = {
            "steps": steps,
            "device": device,
            "shift": shift,
        }
        # Keep the characterization cheap while recording the requested
        # production schedule exactly.
        self.timesteps = torch.tensor([2.0, 1.0], device=device)

    def step(self, _flow_pred, timestep, sample, *, return_dict):
        assert return_dict is False
        self.step_timesteps.append(float(timestep.item()))
        return (sample + 1.0,)


def _cache(kind: str, device: torch.device) -> dict[str, object]:
    return {
        "kind": kind,
        "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
        "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
        "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
    }


def _make_tiny_pipeline():
    pipeline = CausalDiffusionInferencePipeline.__new__(
        CausalDiffusionInferencePipeline
    )
    torch.nn.Module.__init__(pipeline)
    pipeline.frame_seq_length = 1
    pipeline.num_frame_per_block = 2
    pipeline.num_transformer_blocks = 1
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
    pipeline.kv_cache_pos = None
    pipeline.kv_cache_neg = None
    pipeline.crossattn_cache_pos = None
    pipeline.crossattn_cache_neg = None
    pipeline.generator = _RecordingGenerator(pipeline.frame_seq_length)
    pipeline.text_encoder = _RecordingTextEncoder()

    def initialize_kv_cache(self, batch_size, dtype, device):
        del batch_size, dtype
        self.kv_cache_pos = [_cache("positive", device)]
        self.kv_cache_neg = [_cache("negative", device)]

    def initialize_crossattn_cache(self, batch_size, dtype, device):
        del batch_size, dtype, device
        self.crossattn_cache_pos = [{"is_init": False}]
        self.crossattn_cache_neg = [{"is_init": False}]

    pipeline._initialize_kv_cache = MethodType(initialize_kv_cache, pipeline)
    pipeline._initialize_crossattn_cache = MethodType(
        initialize_crossattn_cache, pipeline
    )
    return pipeline


def test_ordinary_inference_public_signature_is_stable():
    signature = inspect.signature(CausalDiffusionInferencePipeline.inference)
    assert list(signature.parameters) == [
        "self",
        "noise",
        "text_prompts",
        "initial_latent",
        "return_latents",
        "start_frame_index",
    ]
    assert signature.parameters["initial_latent"].default is None
    assert signature.parameters["return_latents"].default is False
    assert signature.parameters["start_frame_index"].default == 0


def test_ordinary_inference_resets_state_and_preserves_block_semantics(monkeypatch):
    _RecordingScheduler.instances.clear()
    monkeypatch.setattr(
        causal_pipeline_module,
        "FlowUniPCMultistepScheduler",
        _RecordingScheduler,
    )
    pipeline = _make_tiny_pipeline()
    noise = torch.zeros(1, 4, 1, 1, 1)
    initial_latent = torch.full((1, 1, 1, 1, 1), 9.0)
    kwargs = {
        "noise": noise,
        "text_prompts": [["action-a", "action-b"]],
        "initial_latent": initial_latent,
        "return_latents": True,
    }

    first = pipeline.inference(**kwargs)
    first_call_count = len(pipeline.generator.calls)
    second = pipeline.inference(**kwargs)

    assert first.shape == second.shape == (1, 4, 1, 1, 1)
    assert torch.equal(first, second)
    assert torch.equal(first[:, :1], initial_latent)
    assert torch.equal(first[:, 1:], torch.full_like(first[:, 1:], 2.0))

    # One fresh two-step scheduler per block, on every ordinary inference call.
    schedulers = _RecordingScheduler.instances
    assert len(schedulers) == 4
    assert len({id(scheduler) for scheduler in schedulers}) == 4
    assert [scheduler.constructor_kwargs for scheduler in schedulers] == [
        {
            "num_train_timesteps": 1000,
            "shift": 1,
            "use_dynamic_shifting": False,
        }
    ] * 4
    assert [scheduler.set_timesteps_call for scheduler in schedulers] == [
        {"steps": 50, "device": torch.device("cpu"), "shift": 5.0}
    ] * 4
    assert [scheduler.step_timesteps for scheduler in schedulers] == [
        [2.0, 1.0],
        [2.0, 1.0],
        [2.0, 1.0],
        [2.0, 1.0],
    ]

    first_calls = pipeline.generator.calls[:first_call_count]
    second_calls = pipeline.generator.calls[first_call_count:]
    assert second_calls[0]["kind"] == "positive"
    assert second_calls[0]["before_global"] == 0
    assert second_calls[0]["before_local"] == 0
    second_negative = next(call for call in second_calls if call["kind"] == "negative")
    assert second_negative["before_global"] == 0
    assert second_negative["before_local"] == 0

    # Positive prompts switch by block; the negative embedding stays unchanged.
    for calls in (first_calls, second_calls):
        positive = [call for call in calls if call["kind"] == "positive"]
        negative = [call for call in calls if call["kind"] == "negative"]
        assert {
            call["prompt_id"] for call in positive if call["current_start"] == 0
        } == {11}
        assert {
            call["prompt_id"] for call in positive if call["current_start"] == 2
        } == {22}
        assert {call["prompt_id"] for call in negative} == {-1}
        assert {(call["current_start"], call["cache_start"]) for call in positive} == {
            (0, 0),
            (2, 2),
        }

        # The I2V anchor is clamped for every model call in the first block.
        first_block = [call for call in calls if call["current_start"] == 0]
        assert first_block
        assert all(float(call["input"][:, 0].item()) == 9.0 for call in first_block)
        assert all(float(call["timestep"][:, 0].item()) == 0.0 for call in first_block)

        # Each block ends with one positive and one negative clean t=0 recache.
        for block_start in (0, 2):
            block_calls = [
                call for call in calls if call["current_start"] == block_start
            ]
            clean_recache = [
                call
                for call in block_calls
                if bool(torch.all(call["timestep"] == 0).item())
            ]
            assert len(clean_recache) == 2
            assert clean_recache[0] is block_calls[-2]
            assert clean_recache[1] is block_calls[-1]
            assert [call["kind"] for call in clean_recache] == [
                "positive",
                "negative",
            ]

    assert pipeline.text_encoder.calls == [
        ["action-a", "action-b"],
        ["negative"],
        ["action-a", "action-b"],
        ["negative"],
    ]


def test_ordinary_inference_keeps_legacy_initial_latent_dtype_conversion(
    monkeypatch,
):
    _RecordingScheduler.instances.clear()
    monkeypatch.setattr(
        causal_pipeline_module,
        "FlowUniPCMultistepScheduler",
        _RecordingScheduler,
    )
    pipeline = _make_tiny_pipeline()
    noise = torch.zeros(1, 2, 1, 1, 1, dtype=torch.float32)
    initial_latent = torch.full(
        (1, 1, 1, 1, 1),
        9.0,
        dtype=torch.float64,
    )

    output = pipeline.inference(
        noise=noise,
        text_prompts=[["action-a"]],
        initial_latent=initial_latent,
        return_latents=True,
    )

    assert output.dtype == noise.dtype
    assert torch.equal(output[:, :1], initial_latent.to(dtype=noise.dtype))
    first_block_calls = [
        call for call in pipeline.generator.calls if call["current_start"] == 0
    ]
    assert first_block_calls
    assert all(call["input"].dtype == noise.dtype for call in first_block_calls)
