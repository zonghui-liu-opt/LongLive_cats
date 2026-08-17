"""Named, immutable rollout topologies for Stage-2 training and inference.

The release training config remains the sole serialized baseline contract.  This
module only describes the small set of explicitly approved rollout/compression
topologies, so inference experiments cannot silently invent a C/W/S/K mix.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def _plain_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


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
        if self.name not in _PROFILE_FIELDS:
            raise ValueError(f"unknown Stage-2 rollout profile: {self.name!r}")
        values = (
            _plain_positive_int(self.chunk_frames, "chunk_frames"),
            _plain_positive_int(self.local_window_frames, "local_window_frames"),
            _plain_positive_int(self.global_sink_frames, "global_sink_frames"),
            _plain_positive_int(self.num_denoising_steps, "num_denoising_steps"),
        )
        if values != _PROFILE_FIELDS[self.name]:
            raise ValueError(
                f"Stage-2 rollout profile {self.name!r} does not match its named "
                f"contract: expected={_PROFILE_FIELDS[self.name]}, actual={values}"
            )
        _plain_positive_int(self.generated_episode_frames, "generated_episode_frames")
        if self.generated_episode_frames != 24:
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
        return self.name not in _DEPLOYMENT_ONLY_PROFILES

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
    """Return a fresh immutable value for one exact named topology."""

    if not isinstance(name, str) or name not in _PROFILE_FIELDS:
        raise ValueError(f"unknown Stage-2 rollout profile: {name!r}")
    chunk, window, sink, steps = _PROFILE_FIELDS[name]
    return Stage2RolloutSpec(
        name=name,
        chunk_frames=chunk,
        local_window_frames=window,
        global_sink_frames=sink,
        num_denoising_steps=steps,
    )


__all__ = [
    "STAGE2_ROLLOUT_PROFILE_NAMES",
    "Stage2RolloutSpec",
    "resolve_stage2_rollout_profile",
]
