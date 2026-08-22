"""Named, immutable rollout topologies for Stage-2 training and inference.

The release training config remains the sole serialized baseline contract.  This
module only describes the small set of explicitly approved rollout/compression
topologies, so inference experiments cannot silently invent a C/W/S/K mix.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

_PROFILE_FIELDS: dict[str, tuple[int, int, int, int]] = {
    # name: (chunk C, local window W, global sink S, denoising steps K)
    "baseline_c8w16k4s1": (8, 16, 1, 4),
    "stress_c8w24k4s1": (8, 24, 1, 4),
    "lower_c8w8k4s1": (8, 8, 1, 4),
    "c4w12k4s1": (4, 12, 1, 4),
    "c4w8k4s1": (4, 8, 1, 4),
    "c4w8k2s1": (4, 8, 1, 2),
    "c8w16k4s4": (8, 16, 4, 4),
    "c8w16k4s8": (8, 16, 8, 4),
}
_DEPLOYMENT_ONLY_PROFILES = frozenset(
    {
        "stress_c8w24k4s1",
        "lower_c8w8k4s1",
        "c8w16k4s4",
        "c8w16k4s8",
    }
)

STAGE2_ROLLOUT_PROFILE_NAMES = tuple(_PROFILE_FIELDS)

_STAGE2_GENERATED_EPISODE_FRAMES = 24
_STAGE2_NUM_TRAIN_TIMESTEPS = 1000
_STAGE2_TIMESTEP_SHIFT = 5.0
_STAGE2_MIN_DEPLOYMENT_STEPS = 1
_STAGE2_MAX_DEPLOYMENT_STEPS = 8
_DEPLOYMENT_PROFILE_NAME = re.compile(
    r"^deploy_c(?P<chunk>[0-9]+)w(?P<window>[0-9]+)"
    r"k(?P<steps>[0-9]+)s1_(?P<digest>[0-9a-f]{64})$"
)


def _plain_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _float32_bits(value: float) -> str:
    """Return one platform-independent IEEE-754 binary32 identity."""

    return struct.pack(">f", float(value)).hex()


def _float32_from_bits(value: str) -> float:
    return struct.unpack(">f", bytes.fromhex(value))[0]


@dataclass(frozen=True, slots=True)
class Stage2Shift5Schedule:
    """Deterministic 1000-step UniPC shift=5 deployment schedule."""

    timesteps: tuple[int, ...]
    sigma_fp32_bits: tuple[str, ...]

    @property
    def sigmas(self) -> tuple[float, ...]:
        return tuple(_float32_from_bits(value) for value in self.sigma_fp32_bits)


def resolve_stage2_shift5_schedule(
    num_denoising_steps: int,
) -> Stage2Shift5Schedule:
    """Resolve the exact native K-step schedule used by Stage-2 deployment.

    This deliberately mirrors ``FlowUniPCMultistepScheduler.set_timesteps``:
    its endpoints come from the scheduler's FP32 1000-step base schedule,
    interpolation and time shifting happen in binary64, and emitted sigmas are
    rounded to binary32. Keeping the reference independent makes scheduler
    drift observable rather than self-confirming.
    """

    steps = _plain_positive_int(num_denoising_steps, "num_denoising_steps")
    if not _STAGE2_MIN_DEPLOYMENT_STEPS <= steps <= _STAGE2_MAX_DEPLOYMENT_STEPS:
        raise ValueError("Stage-2 deployment num_denoising_steps must be in [1, 8]")

    sigma_max = _float32_from_bits(
        _float32_bits(1.0 - 1.0 / _STAGE2_NUM_TRAIN_TIMESTEPS)
    )
    shifted_sigmas: list[float] = []
    for index in range(steps):
        unshifted = sigma_max + (0.0 - sigma_max) * (index / steps)
        shifted_sigmas.append(
            _STAGE2_TIMESTEP_SHIFT
            * unshifted
            / (1.0 + (_STAGE2_TIMESTEP_SHIFT - 1.0) * unshifted)
        )
    timesteps = tuple(
        int(value * _STAGE2_NUM_TRAIN_TIMESTEPS) for value in shifted_sigmas
    )
    sigma_fp32_bits = tuple(_float32_bits(value) for value in (*shifted_sigmas, 0.0))
    if len(set(timesteps)) != len(timesteps) or any(
        left <= right for left, right in pairwise(timesteps)
    ):
        raise AssertionError(
            f"Stage-2 shift=5 reference timetable is not strictly decreasing: {timesteps}"
        )
    return Stage2Shift5Schedule(
        timesteps=timesteps,
        sigma_fp32_bits=sigma_fp32_bits,
    )


def _validate_deployment_dimensions(
    *,
    chunk_frames: Any,
    local_window_frames: Any,
    global_sink_frames: Any,
    num_denoising_steps: Any,
) -> tuple[int, int, int, int]:
    chunk = _plain_positive_int(chunk_frames, "chunk_frames")
    window = _plain_positive_int(local_window_frames, "local_window_frames")
    sink = _plain_positive_int(global_sink_frames, "global_sink_frames")
    steps = _plain_positive_int(num_denoising_steps, "num_denoising_steps")
    if chunk < 2 or _STAGE2_GENERATED_EPISODE_FRAMES % chunk:
        raise ValueError("deployment chunk_frames must be >=2 and divide 24")
    if window < chunk or window % chunk or window > 24:
        raise ValueError(
            "deployment local_window_frames must be >= chunk_frames, a multiple "
            "of chunk_frames, and <=24"
        )
    if sink != 1:
        raise ValueError("dynamic Stage-2 deployment profiles require S=1")
    resolve_stage2_shift5_schedule(steps)
    return chunk, window, sink, steps


def _deployment_profile_name(
    *,
    chunk_frames: int,
    local_window_frames: int,
    num_denoising_steps: int,
) -> str:
    schedule = resolve_stage2_shift5_schedule(num_denoising_steps)
    identity = {
        "generated_episode_frames": _STAGE2_GENERATED_EPISODE_FRAMES,
        "chunk_frames": chunk_frames,
        "local_window_frames": local_window_frames,
        "global_sink_frames": 1,
        "num_denoising_steps": num_denoising_steps,
        "solver": "unipc",
        "timestep_shift": _STAGE2_TIMESTEP_SHIFT,
        "num_train_timesteps": _STAGE2_NUM_TRAIN_TIMESTEPS,
        "timesteps": list(schedule.timesteps),
        "sigma_fp32_bits": list(schedule.sigma_fp32_bits),
    }
    payload = json.dumps(
        identity,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256(payload).hexdigest()
    return (
        f"deploy_c{chunk_frames}w{local_window_frames}"
        f"k{num_denoising_steps}s1_{digest}"
    )


@dataclass(frozen=True, slots=True)
class Stage2RolloutSpec:
    """One approved C/W/S/K topology with all other values derived."""

    name: str
    chunk_frames: int
    local_window_frames: int
    global_sink_frames: int
    num_denoising_steps: int
    generated_episode_frames: int = 24
    solver: str = "unipc"
    timestep_shift: float = 5.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Stage-2 rollout profile name must be non-empty")
        values = (
            _plain_positive_int(self.chunk_frames, "chunk_frames"),
            _plain_positive_int(self.local_window_frames, "local_window_frames"),
            _plain_positive_int(self.global_sink_frames, "global_sink_frames"),
            _plain_positive_int(self.num_denoising_steps, "num_denoising_steps"),
        )
        if self.name in _PROFILE_FIELDS:
            if values != _PROFILE_FIELDS[self.name]:
                raise ValueError(
                    f"Stage-2 rollout profile {self.name!r} does not match its named "
                    f"contract: expected={_PROFILE_FIELDS[self.name]}, actual={values}"
                )
        elif _DEPLOYMENT_PROFILE_NAME.fullmatch(self.name):
            chunk, window, _sink, steps = _validate_deployment_dimensions(
                chunk_frames=self.chunk_frames,
                local_window_frames=self.local_window_frames,
                global_sink_frames=self.global_sink_frames,
                num_denoising_steps=self.num_denoising_steps,
            )
            expected_name = _deployment_profile_name(
                chunk_frames=chunk,
                local_window_frames=window,
                num_denoising_steps=steps,
            )
            if self.name != expected_name:
                raise ValueError(
                    "dynamic Stage-2 deployment profile canonical name mismatch"
                )
        else:
            raise ValueError(f"unknown Stage-2 rollout profile: {self.name!r}")
        _plain_positive_int(self.generated_episode_frames, "generated_episode_frames")
        if self.generated_episode_frames != _STAGE2_GENERATED_EPISODE_FRAMES:
            raise ValueError("Stage-2 rollout profiles must generate exactly 24 frames")
        if self.generated_episode_frames % self.chunk_frames:
            raise ValueError("generated_episode_frames must divide into whole chunks")
        if self.local_window_frames < self.chunk_frames:
            raise ValueError("local_window_frames must include the current chunk")
        if self.local_window_frames % self.chunk_frames:
            raise ValueError("local_window_frames must be a multiple of chunk_frames")
        if self.solver != "unipc" or float(self.timestep_shift) != 5.0:
            raise ValueError("Stage-2 rollout profiles require UniPC with shift=5")

    @property
    def num_chunks(self) -> int:
        return self.generated_episode_frames // self.chunk_frames

    @property
    def history_frames(self) -> int:
        return self.local_window_frames - self.chunk_frames

    @property
    def physical_kv_capacity_frames(self) -> int:
        return self.global_sink_frames + self.local_window_frames

    @property
    def fresh_deploy_dit_calls(self) -> int:
        return 1 + self.num_chunks * (self.num_denoising_steps + 1)

    @property
    def training_allowed(self) -> bool:
        return (
            self.name in _PROFILE_FIELDS and self.name not in _DEPLOYMENT_ONLY_PROFILES
        )

    @property
    def fresh_episode_allowed(self) -> bool:
        return self.global_sink_frames == 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "generated_episode_frames": self.generated_episode_frames,
            "chunk_frames": self.chunk_frames,
            "num_chunks": self.num_chunks,
            "local_window_frames": self.local_window_frames,
            "history_frames": self.history_frames,
            "global_sink_frames": self.global_sink_frames,
            "physical_kv_capacity_frames": self.physical_kv_capacity_frames,
            "num_denoising_steps": self.num_denoising_steps,
            "solver": self.solver,
            "timestep_shift": self.timestep_shift,
            "fresh_deploy_dit_calls": self.fresh_deploy_dit_calls,
            "training_allowed": self.training_allowed,
            "fresh_episode_allowed": self.fresh_episode_allowed,
        }


def resolve_stage2_rollout_profile(name: str) -> Stage2RolloutSpec:
    """Resolve a frozen named contract or a self-authenticating deployment id."""

    if not isinstance(name, str):
        raise TypeError(f"Stage-2 rollout profile name must be a string: {name!r}")
    if name in _PROFILE_FIELDS:
        chunk, window, sink, steps = _PROFILE_FIELDS[name]
        return Stage2RolloutSpec(
            name=name,
            chunk_frames=chunk,
            local_window_frames=window,
            global_sink_frames=sink,
            num_denoising_steps=steps,
        )
    match = _DEPLOYMENT_PROFILE_NAME.fullmatch(name)
    if match is None:
        raise ValueError(f"unknown Stage-2 rollout profile: {name!r}")
    spec = build_stage2_deployment_rollout_spec(
        chunk_frames=int(match.group("chunk")),
        local_window_frames=int(match.group("window")),
        num_denoising_steps=int(match.group("steps")),
    )
    if spec.name != name:
        raise ValueError("dynamic Stage-2 deployment profile canonical name mismatch")
    return spec


def build_stage2_deployment_rollout_spec(
    *,
    chunk_frames: int,
    local_window_frames: int,
    num_denoising_steps: int,
) -> Stage2RolloutSpec:
    """Build one canonical, deployment-only S1 C/W/K sweep topology."""

    chunk, window, sink, steps = _validate_deployment_dimensions(
        chunk_frames=chunk_frames,
        local_window_frames=local_window_frames,
        global_sink_frames=1,
        num_denoising_steps=num_denoising_steps,
    )
    return Stage2RolloutSpec(
        name=_deployment_profile_name(
            chunk_frames=chunk,
            local_window_frames=window,
            num_denoising_steps=steps,
        ),
        chunk_frames=chunk,
        local_window_frames=window,
        global_sink_frames=sink,
        num_denoising_steps=steps,
    )


__all__ = [
    "STAGE2_ROLLOUT_PROFILE_NAMES",
    "Stage2RolloutSpec",
    "Stage2Shift5Schedule",
    "build_stage2_deployment_rollout_spec",
    "resolve_stage2_rollout_profile",
    "resolve_stage2_shift5_schedule",
]
