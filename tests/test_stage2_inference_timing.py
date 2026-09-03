from __future__ import annotations

import itertools
import json
import math

import pytest
import torch

from tests.test_stage2_inference import _FakeVAE
from tests.test_stage2_rollout import _FakeGenerator, _inputs, _pipeline
from utils import stage2_inference_timing as timing
from utils.stage2_inference import (
    generate_stage2_single_action,
    generate_stage2_two_action,
)
from utils.stage2_inference_timing import (
    Stage2InferenceTiming,
    stage2_timing_span,
)


def _clock(monkeypatch, values=None):
    ticks = iter(values) if values is not None else itertools.count()
    monkeypatch.setattr(timing, "perf_counter", lambda: float(next(ticks)))


def test_disabled_spans_do_not_read_clock_or_synchronize(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled timing performed a timing operation")

    monkeypatch.setattr(timing, "perf_counter", unexpected)
    monkeypatch.setattr(timing, "_synchronize", unexpected)
    with stage2_timing_span("dit", "cuda:7"):
        with stage2_timing_span("vae_decode", "cuda:7"):
            value = 42
    assert value == 42


def test_cpu_stage_and_total_durations_have_a_json_safe_schema(monkeypatch):
    _clock(monkeypatch, [0, 1, 3, 4, 7, 8, 12, 15])

    def unexpected(*args, **kwargs):
        raise AssertionError("CPU timing called CUDA synchronize")

    monkeypatch.setattr(torch.cuda, "synchronize", unexpected)
    collector = Stage2InferenceTiming(torch.device("cpu"))
    with collector.record():
        with stage2_timing_span("dit", "cpu"):
            pass
        with stage2_timing_span("vae_decode", "cpu"):
            pass
        with stage2_timing_span("video_postprocess", "cpu"):
            pass

    result = collector.to_dict()
    assert result == {
        "schema": "longlive_stage2_inference_timing/v1",
        "method": "cpu_wall",
        "device": "cpu",
        "dit_seconds": 2.0,
        "dit_calls": 1,
        "vae_decode_seconds": 3.0,
        "vae_decode_calls": 1,
        "video_postprocess_seconds": 4.0,
        "video_postprocess_calls": 1,
        "total_seconds": 15.0,
        "other_seconds": 6.0,
    }
    assert json.loads(json.dumps(result, allow_nan=False)) == result


def test_cuda_work_completes_before_each_stop_clock_read(monkeypatch):
    events = []
    ticks = iter([0.0, 2.0, 8.0, 10.0])

    def clock():
        value = next(ticks)
        events.append(("clock", value))
        return value

    monkeypatch.setattr(timing, "perf_counter", clock)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: events.append(("sync", str(device)))
    )
    collector = Stage2InferenceTiming("cuda:3")
    with collector.record():
        with stage2_timing_span("dit", torch.device("cuda:3")):
            events.append(("work", None))

    assert events == [
        ("sync", "cuda:3"),
        ("clock", 0.0),
        ("sync", "cuda:3"),
        ("clock", 2.0),
        ("work", None),
        ("sync", "cuda:3"),
        ("clock", 8.0),
        ("sync", "cuda:3"),
        ("clock", 10.0),
    ]
    assert collector.to_dict()["method"] == "cuda_synchronized_wall"
    assert collector.to_dict()["dit_seconds"] == 6.0
    assert collector.to_dict()["other_seconds"] == 4.0


def test_nested_spans_are_counted_once_under_the_outer_stage(monkeypatch):
    _clock(monkeypatch, [0, 1, 4, 5])
    collector = Stage2InferenceTiming("cpu")
    with collector.record():
        with stage2_timing_span("dit", "cpu"):
            with stage2_timing_span("dit", "cpu"):
                with stage2_timing_span("vae_decode", "cpu"):
                    pass

    result = collector.to_dict()
    assert result["dit_seconds"] == 3.0
    assert result["dit_calls"] == 1
    assert result["vae_decode_calls"] == 0
    assert result["total_seconds"] == 5.0
    assert result["other_seconds"] == 2.0


def test_nested_collectors_restore_the_previous_scope_after_failure(monkeypatch):
    _clock(monkeypatch)
    outer = Stage2InferenceTiming("cpu")
    inner = Stage2InferenceTiming("cpu")
    with outer.record():
        with pytest.raises(RuntimeError, match="failed model"):
            with inner.record():
                with stage2_timing_span("dit", "cpu"):
                    raise RuntimeError("failed model")
        with stage2_timing_span("vae_decode", "cpu"):
            pass
    assert inner.to_dict()["dit_calls"] == 1
    assert outer.to_dict()["dit_calls"] == 0
    assert outer.to_dict()["vae_decode_calls"] == 1
    assert timing._ACTIVE_COLLECTOR.get() is None
    assert timing._ACTIVE_SPAN.get() is None

    def unexpected(*args, **kwargs):
        raise AssertionError("timing leaked after the failed operation")

    monkeypatch.setattr(timing, "perf_counter", unexpected)
    with stage2_timing_span("dit", "cpu"):
        pass


def test_failed_boundary_synchronization_still_restores_context(monkeypatch):
    _clock(monkeypatch)
    sync_calls = []

    def synchronize(device):
        sync_calls.append(device)
        if len(sync_calls) == 3:
            raise RuntimeError("CUDA failure")

    monkeypatch.setattr(timing, "_synchronize", synchronize)
    collector = Stage2InferenceTiming("cuda:0")
    with pytest.raises(RuntimeError, match="CUDA failure"):
        with collector.record():
            with stage2_timing_span("dit", "cuda:0"):
                pass
    assert timing._ACTIVE_COLLECTOR.get() is None
    assert timing._ACTIVE_SPAN.get() is None
    assert collector.to_dict()["dit_calls"] == 0


def test_record_can_accumulate_disjoint_scopes_but_rejects_recursion(monkeypatch):
    _clock(monkeypatch)
    collector = Stage2InferenceTiming("cpu")
    for _ in range(2):
        with collector.record():
            with pytest.raises(RuntimeError, match="recursively"):
                with collector.record():
                    pass
            with stage2_timing_span("dit", "cpu"):
                pass
    result = collector.to_dict()
    assert result["dit_calls"] == 2
    assert result["dit_seconds"] == 2.0
    assert result["total_seconds"] == 6.0


@pytest.mark.parametrize(("episodes", "expected_dit_calls"), [(1, 31), (2, 61)])
def test_c4w16k4s1_counts_every_generator_forward_and_decode(
    monkeypatch, episodes, expected_dit_calls
):
    _clock(monkeypatch)

    def generate():
        generator = _FakeGenerator()
        pipeline, _ = _pipeline(generator, spec="c4w16k4s1")
        initial, _, action_a = _inputs(prompt=1.0, batch=1)
        if episodes == 1:
            result = generate_stage2_single_action(
                pipeline,
                _FakeVAE(),
                initial_latent=initial,
                conditional_dict=action_a,
                seeds=[1],
            )
        else:
            action_b = {
                "prompt_embeds": torch.full((1, 2, 3), 2.0, dtype=torch.bfloat16)
            }
            result = generate_stage2_two_action(
                pipeline,
                pipeline,
                _FakeVAE(),
                initial_latent=initial,
                action_a_conditional_dict=action_a,
                action_b_conditional_dict=action_b,
                seeds=[1],
            )
        return result, generator

    unmeasured, _ = generate()
    collector = Stage2InferenceTiming("cpu")
    with collector.record():
        measured, generator = generate()

    timings = collector.to_dict()
    assert timings["dit_calls"] == expected_dit_calls == len(generator.calls)
    assert timings["dit_calls"] == sum(
        episode["cache_audit"]["generator_forward_calls"]
        for episode in measured.trace["episodes"]
    )
    # Each episode has 6 chunks x 4 denoising calls, 6 clean recache calls,
    # plus one initial-sink prefill in episode A.  Episode B reuses that sink.
    assert sum(not call["commit"] for call in generator.calls) == episodes * 24
    assert sum(call["commit"] for call in generator.calls) == episodes * 6 + 1
    assert timings["vae_decode_calls"] == episodes
    assert timings["video_postprocess_calls"] == (1 if episodes == 1 else 3)
    for key, value in timings.items():
        if key.endswith("_seconds"):
            assert math.isfinite(value) and value >= 0
    assert measured.video.shape[1] == episodes * 96
    assert torch.equal(measured.video, unmeasured.video)
    assert measured.trace == unmeasured.trace
    for actual, expected in zip(measured.episode_latents, unmeasured.episode_latents):
        assert torch.equal(actual, expected)
