"""Strict Stage-1 teacher-forcing LoRA rollout profile contract.

``window_size`` uses the user-facing physical-cache convention throughout this
module: it includes the current noisy chunk, generated-history KV, and the
permanent global sink.  The first version supports the conservative C2/C4/C8
deployment family.  C8 is the formal Stage-1 training block size; C8/W17/S1
is only this rollout's default deployment topology, because teacher forcing
did not train a rolling W17 self-generated-history topology.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Real
from typing import Any

from utils.config import DEFAULT_NEGATIVE_PROMPT
from utils.stage1_io import canonical_json_sha256

STAGE1_ROLLOUT_PROFILE_SCHEMA = "longlive.stage1_rollout_profile/v1"

_CONFIG_FIELDS = frozenset(
    {
        "chunk_size",
        "window_size",
        "global_sink_size",
        "generated_frames",
        "sampling_steps",
        "timestep_shift",
        "guidance_scale",
        "solver",
        "kv_quant",
    }
)


def _plain_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer (bool is not accepted)")
    return value


def _finite_float_at_least(value: Any, name: str, minimum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        # Config callers intentionally receive one ValueError family for every
        # invalid profile scalar, including an incorrect scalar type.
        raise ValueError(  # noqa: TRY004
            f"{name} must be a finite real number (bool is not accepted)"
        )
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    if normalized < minimum:
        comparator = ">0" if minimum == 0.0 else f">={minimum:g}"
        raise ValueError(f"{name} must be {comparator}")
    if minimum == 0.0 and normalized == 0.0:
        raise ValueError(f"{name} must be >0")
    return normalized


@dataclass(frozen=True, slots=True)
class Stage1RolloutProfile:
    """Immutable v1 runtime profile, measured in latent frames.

    Fixed fields remain constructor arguments so an explicitly serialized
    profile can be validated rather than silently normalized.  The negative
    prompt is deliberately ``init=False``: runtime configuration cannot
    replace it, even with a value that happens to match today.
    """

    chunk_size: int = 8
    window_size: int = 17
    global_sink_size: int = 1
    generated_frames: int = 24
    sampling_steps: int = 50
    timestep_shift: float = 5.0
    guidance_scale: float = 5.0
    solver: str = "unipc"
    kv_quant: bool = False
    negative_prompt: str = field(
        default=DEFAULT_NEGATIVE_PROMPT,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        chunk_size = _plain_positive_int(self.chunk_size, "chunk_size")
        window_size = _plain_positive_int(self.window_size, "window_size")
        global_sink_size = _plain_positive_int(
            self.global_sink_size, "global_sink_size"
        )
        generated_frames = _plain_positive_int(
            self.generated_frames, "generated_frames"
        )
        sampling_steps = _plain_positive_int(self.sampling_steps, "sampling_steps")

        if chunk_size not in {2, 4, 8}:
            raise ValueError(
                "Stage1RolloutProfile v1 requires chunk_size to be one of {2, 4, 8}"
            )
        if global_sink_size != 1:
            raise ValueError("Stage1RolloutProfile v1 requires global_sink_size=1")
        if generated_frames != 24:
            raise ValueError("Stage1RolloutProfile v1 requires generated_frames=24")
        history_size = window_size - global_sink_size - chunk_size
        if history_size < chunk_size:
            raise ValueError(
                "window_size must retain at least one full generated-history chunk: "
                "window_size >= global_sink_size + 2 * chunk_size"
            )
        if history_size % chunk_size:
            raise ValueError(
                "generated history (window_size - global_sink_size - chunk_size) "
                "must be a multiple of chunk_size"
            )
        if window_size - global_sink_size > 24:
            raise ValueError(
                "local_window_size (window_size - global_sink_size) must be <=24"
            )
        if sampling_steps not in {4, 50}:
            raise ValueError("sampling_steps must be exactly 4 or 50")

        timestep_shift = _finite_float_at_least(
            self.timestep_shift, "timestep_shift", 0.0
        )
        guidance_scale = _finite_float_at_least(
            self.guidance_scale, "guidance_scale", 1.0
        )
        object.__setattr__(self, "timestep_shift", timestep_shift)
        object.__setattr__(self, "guidance_scale", guidance_scale)

        if not isinstance(self.solver, str) or self.solver != "unipc":
            raise ValueError("Stage1RolloutProfile v1 requires solver='unipc'")
        if not isinstance(self.kv_quant, bool):
            raise ValueError(  # noqa: TRY004
                "kv_quant must be a bool and must be false"
            )
        if self.kv_quant:
            raise ValueError(
                "Stage1RolloutProfile v1 requires kv_quant=false; the 17-frame "
                "sink-aware cache is not quantization-safe"
            )
        if self.negative_prompt is not DEFAULT_NEGATIVE_PROMPT:
            raise AssertionError(
                "negative_prompt is not bound to DEFAULT_NEGATIVE_PROMPT"
            )

        if generated_frames % chunk_size:
            raise ValueError("generated_frames must divide into whole chunks")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Stage1RolloutProfile:
        """Resolve a strict profile mapping without accepting prompt overrides."""

        if not isinstance(value, Mapping):
            raise TypeError("Stage-1 rollout profile must be a mapping")
        raw = dict(value)
        if "negative_prompt" in raw:
            raise ValueError(
                "negative_prompt cannot be configured for Stage1RolloutProfile v1; "
                "it is fixed to utils.config.DEFAULT_NEGATIVE_PROMPT"
            )

        schema = raw.pop("schema_version", STAGE1_ROLLOUT_PROFILE_SCHEMA)
        if schema != STAGE1_ROLLOUT_PROFILE_SCHEMA:
            raise ValueError(
                "schema_version must be " f"{STAGE1_ROLLOUT_PROFILE_SCHEMA!r}"
            )
        unknown = sorted(set(raw) - _CONFIG_FIELDS)
        if unknown:
            raise ValueError(
                "unknown Stage-1 rollout profile fields: " + ", ".join(unknown)
            )
        return cls(**raw)

    @property
    def history_size(self) -> int:
        """Generated-history KV frames retained beside the current chunk."""

        return self.window_size - self.global_sink_size - self.chunk_size

    @property
    def kv_cache_size(self) -> int:
        """Alias matching the inference-profile terminology."""

        return self.history_size

    @property
    def local_window_size(self) -> int:
        """Current plus rolling history, excluding the permanent sink."""

        return self.window_size - self.global_sink_size

    @property
    def num_chunks(self) -> int:
        return self.generated_frames // self.chunk_size

    @property
    def default_block_size(self) -> bool:
        """Whether the profile uses the formal Stage-1 C8 block size."""

        return self.chunk_size == 8

    @property
    def default_topology(self) -> bool:
        """Whether C/W equal this rollout's default deployment topology."""

        return self.chunk_size == 8 and self.window_size == 17

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return the complete, unit-explicit v1 identity used for hashing."""

        return {
            "schema_version": STAGE1_ROLLOUT_PROFILE_SCHEMA,
            "chunk_size": self.chunk_size,
            "window_size": self.window_size,
            "global_sink_size": self.global_sink_size,
            "history_size": self.history_size,
            "local_window_size": self.local_window_size,
            "generated_frames": self.generated_frames,
            "num_chunks": self.num_chunks,
            "default_block_size": self.default_block_size,
            "default_topology": self.default_topology,
            "sampling_steps": self.sampling_steps,
            "timestep_shift": self.timestep_shift,
            "guidance_scale": self.guidance_scale,
            "solver": self.solver,
            "kv_quant": self.kv_quant,
            "negative_prompt": self.negative_prompt,
        }

    def to_dict(self) -> dict[str, Any]:
        """Compatibility alias for trace serializers."""

        return self.to_canonical_dict()

    @property
    def canonical_dict(self) -> dict[str, Any]:
        return self.to_canonical_dict()

    @property
    def canonical_sha256(self) -> str:
        return canonical_json_sha256(self.to_canonical_dict())


def resolve_stage1_rollout_profile(
    value: Stage1RolloutProfile | Mapping[str, Any] | None = None,
) -> Stage1RolloutProfile:
    """Resolve the default, an already-resolved profile, or a strict mapping."""

    if value is None:
        return Stage1RolloutProfile()
    if isinstance(value, Stage1RolloutProfile):
        return value
    if isinstance(value, Mapping):
        return Stage1RolloutProfile.from_mapping(value)
    raise TypeError(
        "Stage-1 rollout profile must be None, Stage1RolloutProfile, or a mapping"
    )


def resolve_stage1_rollout_profile_overrides(
    value: Mapping[str, Any] | None,
    overrides: Mapping[str, Any],
) -> tuple[Stage1RolloutProfile, dict[str, Any]]:
    """Apply non-None CLI-style overrides and return the canonical raw source.

    The returned raw mapping intentionally excludes derived trace fields so it
    can be written back to ``stage1_rollout.profile`` and parsed again by the
    checkpoint preflight without identity drift.
    """

    if value is None:
        raw: dict[str, Any] = {}
    elif isinstance(value, Mapping):
        raw = dict(value)
    else:
        raise TypeError("Stage-1 rollout profile overrides require a mapping")
    if not isinstance(overrides, Mapping):
        raise TypeError("Stage-1 rollout overrides must be a mapping")
    unknown = sorted(set(overrides) - _CONFIG_FIELDS)
    if unknown:
        raise ValueError("unknown Stage-1 rollout overrides: " + ", ".join(unknown))
    for name, override in overrides.items():
        if override is not None:
            raw[name] = override
    return Stage1RolloutProfile.from_mapping(raw), raw


__all__ = [
    "STAGE1_ROLLOUT_PROFILE_SCHEMA",
    "Stage1RolloutProfile",
    "resolve_stage1_rollout_profile",
    "resolve_stage1_rollout_profile_overrides",
]
