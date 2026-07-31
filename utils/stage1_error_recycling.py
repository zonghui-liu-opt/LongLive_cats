"""SP-synchronized logical gates for Stage-1 error recycling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import random
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class ErrorRecyclingGate:
    active: bool
    context: bool
    latent: bool
    noise: bool
    update_buffer: bool

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


def _value(schedule_values: Any, name: str) -> Any:
    if isinstance(schedule_values, dict):
        return schedule_values[name]
    return getattr(schedule_values, name)


def _probability(schedule_values: Any, name: str) -> float:
    value = float(_value(schedule_values, name))
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite probability, got {value}.")
    return value


def sample_error_recycling_gate(
    schedule_values: Any,
    *,
    clean_buffer_update_prob: float,
    rng: Any = random,
) -> ErrorRecyclingGate:
    """Sample the hierarchy once on an SP root for one logical sample."""
    mode = str(_value(schedule_values, "error_recycling_mode"))
    clean_update = float(clean_buffer_update_prob)
    if not math.isfinite(clean_update) or not 0.0 <= clean_update <= 1.0:
        raise ValueError("clean_buffer_update_prob must be in [0, 1].")
    if mode == "collect_only":
        return ErrorRecyclingGate(
            active=False,
            context=False,
            latent=False,
            noise=False,
            update_buffer=True,
        )
    if mode != "collect_and_inject":
        raise ValueError(f"Unsupported error recycling mode: {mode!r}.")

    active = rng.random() < _probability(schedule_values, "active_probability")
    if not active:
        return ErrorRecyclingGate(
            active=False,
            context=False,
            latent=False,
            noise=False,
            update_buffer=rng.random() < clean_update,
        )
    return ErrorRecyclingGate(
        active=True,
        context=rng.random()
        < _probability(schedule_values, "context_probability_given_active"),
        latent=rng.random()
        < _probability(schedule_values, "latent_probability_given_active"),
        noise=rng.random()
        < _probability(schedule_values, "noise_probability_given_active"),
        update_buffer=True,
    )


def broadcast_error_recycling_gate(
    schedule_values: Any,
    *,
    clean_buffer_update_prob: float,
    group=None,
    root_global_rank: int = 0,
    device: torch.device | str = "cpu",
    rng: Any = random,
) -> ErrorRecyclingGate:
    """Generate on the SP group root and broadcast exactly five booleans."""
    distributed = dist.is_available() and dist.is_initialized()
    is_root = not distributed or dist.get_rank() == int(root_global_rank)
    if is_root:
        gate = sample_error_recycling_gate(
            schedule_values,
            clean_buffer_update_prob=clean_buffer_update_prob,
            rng=rng,
        )
        values = [
            gate.active,
            gate.context,
            gate.latent,
            gate.noise,
            gate.update_buffer,
        ]
    else:
        values = [False] * 5
    tensor = torch.tensor(values, dtype=torch.uint8, device=device)
    if distributed:
        dist.broadcast(tensor, src=int(root_global_rank), group=group)
    result = [bool(value) for value in tensor.cpu().tolist()]
    return ErrorRecyclingGate(*result)
