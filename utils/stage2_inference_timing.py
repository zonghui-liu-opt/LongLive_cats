"""Opt-in synchronized stage timing for Stage-2 deployment inference.

The rollout and decode primitives use ``stage2_timing_span`` without depending
on runtime configuration.  With no active collector it does not read a clock,
import torch, or synchronize a device.  Nested spans belong to their outermost
stage so a measured operation is never counted twice in one collector.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Any

STAGE2_INFERENCE_TIMING_SCHEMA = "longlive_stage2_inference_timing/v1"
_STAGES = ("dit", "vae_decode", "video_postprocess")
_ACTIVE_COLLECTOR: ContextVar[Stage2InferenceTiming | None] = ContextVar(
    "stage2_inference_timing_collector", default=None
)
_ACTIVE_SPAN: ContextVar[Stage2InferenceTiming | None] = ContextVar(
    "stage2_inference_timing_span", default=None
)


def _synchronize(device: Any) -> None:
    if str(device).split(":", 1)[0] == "cuda":
        import torch

        torch.cuda.synchronize(device)


def _elapsed(start: float) -> float:
    elapsed = perf_counter() - start
    if not math.isfinite(elapsed):
        raise RuntimeError("Stage-2 inference timer returned a non-finite duration")
    return max(0.0, elapsed)


class Stage2InferenceTiming:
    """Accumulate stage and total times inside one or more ``record`` scopes.

    CUDA wall times synchronize the selected device at each boundary, making
    asynchronous GPU work part of its stage.  Total time also contains work
    between spans, including scheduler/cache work and trace construction.
    Call counts include entered spans whose body raises an exception.
    """

    def __init__(self, device: Any):
        self.device = str(device)
        self._seconds = dict.fromkeys(_STAGES, 0.0)
        self._calls = dict.fromkeys(_STAGES, 0)
        self._total_seconds = 0.0
        self._recording = False

    @contextmanager
    def record(self) -> Iterator[Stage2InferenceTiming]:
        """Activate this collector and measure the full enclosed operation."""

        if self._recording:
            raise RuntimeError("a Stage-2 timing collector cannot record recursively")
        _synchronize(self.device)
        start = perf_counter()
        token = _ACTIVE_COLLECTOR.set(self)
        self._recording = True
        try:
            yield self
        finally:
            try:
                _synchronize(self.device)
                self._total_seconds += _elapsed(start)
            finally:
                self._recording = False
                _ACTIVE_COLLECTOR.reset(token)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe batch durations after all recording scopes finish."""

        if self._recording:
            raise RuntimeError("finish Stage-2 timing recording before exporting it")
        payload: dict[str, Any] = {
            "schema": STAGE2_INFERENCE_TIMING_SCHEMA,
            "method": (
                "cuda_synchronized_wall"
                if self.device.split(":", 1)[0] == "cuda"
                else "cpu_wall"
            ),
            "device": self.device,
        }
        for stage in _STAGES:
            payload[f"{stage}_seconds"] = self._seconds[stage]
            payload[f"{stage}_calls"] = self._calls[stage]
        payload["total_seconds"] = self._total_seconds
        payload["other_seconds"] = max(
            0.0, self._total_seconds - sum(self._seconds.values())
        )
        return payload


@contextmanager
def stage2_timing_span(stage: str, device: Any) -> Iterator[None]:
    """Measure a stage when enabled; otherwise run with no timing operations."""

    collector = _ACTIVE_COLLECTOR.get()
    if collector is None or _ACTIVE_SPAN.get() is collector:
        yield
        return
    if stage not in _STAGES:
        raise ValueError(f"unknown Stage-2 inference timing stage: {stage!r}")
    _synchronize(device)
    start = perf_counter()
    token = _ACTIVE_SPAN.set(collector)
    try:
        yield
    finally:
        try:
            _synchronize(device)
            collector._seconds[stage] += _elapsed(start)
            collector._calls[stage] += 1
        finally:
            _ACTIVE_SPAN.reset(token)


__all__ = [
    "STAGE2_INFERENCE_TIMING_SCHEMA",
    "Stage2InferenceTiming",
    "stage2_timing_span",
]
