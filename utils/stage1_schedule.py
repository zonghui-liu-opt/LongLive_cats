"""Resolved, stateless schedule for causal I2V Stage-1 training.

The schedule deliberately derives optimizer-step counts from the real local
DataLoader length.  It never keeps mutable scheduler state: a checkpoint only
needs the resolved configuration and the number of completed updates in order
to reproduce the next learning rate and error-recycling probabilities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


SCHEDULE_SCHEMA_VERSION = 1


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _require_finite(name: str, value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    result = int(value)
    if result <= 0 or float(value) != result:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return result


def _integer_product(name: str, left: int, right: Any) -> int:
    product = left * _require_finite(name, right)
    rounded = round(product)
    if not math.isclose(product, rounded, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            f"{name} must resolve to an integer number of updates, got {product}."
        )
    if rounded <= 0:
        raise ValueError(f"{name} must resolve to at least one update, got {rounded}.")
    return int(rounded)


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    epochs: float
    start_update: int
    end_update: int
    lr_start: float
    lr_end: float
    error_recycling_mode: str
    ramp_updates: int = 0
    max_active_prob: float = 0.0
    effective_context_prob: float = 0.0
    effective_latent_prob: float = 0.0
    effective_noise_prob: float = 0.0

    @property
    def updates(self) -> int:
        return self.end_update - self.start_update


@dataclass(frozen=True)
class ScheduleValues:
    update_index: int
    optimizer_step: int
    phase_index: int
    phase_name: str
    lr: float
    ramp_u: float
    ramp_s: float
    error_recycling_mode: str
    active_probability: float
    context_probability_given_active: float
    latent_probability_given_active: float
    noise_probability_given_active: float
    effective_context_probability: float
    effective_latent_probability: float
    effective_noise_probability: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Stage1Schedule:
    micro_batches_per_epoch: int
    gradient_accumulation_steps: int
    updates_per_epoch: int
    phases: tuple[PhaseSpec, ...]
    total_epochs: float
    total_updates: int
    checkpoint_interval_updates: int
    schema_version: int = SCHEDULE_SCHEMA_VERSION

    @classmethod
    def from_config(cls, config: Any, dataloader_length: int) -> "Stage1Schedule":
        training = _get(config, "training", config)
        data = _get(config, "data", None)
        infra = _get(config, "infra", None)
        checkpointing = _get(config, "checkpointing", None)
        if checkpointing is None:
            checkpointing = config

        micro_batches = _positive_int("len(dataloader)", dataloader_length)
        accumulation = _positive_int(
            "training.gradient_accumulation_steps",
            _get(training, "gradient_accumulation_steps", 1),
        )
        if micro_batches % accumulation:
            raise ValueError(
                f"len(dataloader) ({micro_batches}) must be divisible by "
                f"gradient_accumulation_steps ({accumulation}); partial updates "
                "are forbidden."
            )
        updates_per_epoch = micro_batches // accumulation

        raw_phases = _get(training, "phases", None)
        if not raw_phases or not isinstance(raw_phases, Sequence):
            raise ValueError("training.phases must be a non-empty sequence.")
        legacy_max_iters = _get(training, "max_iters", _get(config, "max_iters", None))
        if legacy_max_iters is not None:
            raise ValueError(
                "Phase-derived Stage-1 schedules cannot also define max_iters; "
                "there must be exactly one source of truth."
            )

        # Cross-check the real DataLoader length against the locked logical
        # sample topology when those values are present. This detects sampler
        # padding/dropping before a single optimizer update is attempted.
        topology_values = {
            "data.expected_num_samples": _get(data, "expected_num_samples", None),
            "infra.data_parallel_size": _get(infra, "data_parallel_size", None),
            "data.batch_size": _get(data, "batch_size", None),
        }
        if any(value is not None for value in topology_values.values()):
            if any(value is None for value in topology_values.values()):
                missing = [name for name, value in topology_values.items() if value is None]
                raise ValueError(
                    f"Stage-1 DataLoader topology is incomplete; missing {missing}."
                )
            sample_count = _positive_int(
                "data.expected_num_samples", topology_values["data.expected_num_samples"]
            )
            dp_size = _positive_int(
                "infra.data_parallel_size", topology_values["infra.data_parallel_size"]
            )
            batch_size = _positive_int("data.batch_size", topology_values["data.batch_size"])
            divisor = dp_size * batch_size
            if sample_count % divisor:
                raise ValueError(
                    f"data.expected_num_samples ({sample_count}) must be divisible by "
                    f"DP×batch ({divisor}); dropping or oversampling is forbidden."
                )
            expected_micro_batches = sample_count // divisor
            if micro_batches != expected_micro_batches:
                raise ValueError(
                    f"len(dataloader)={micro_batches} does not match exact epoch "
                    f"accounting {sample_count}/({dp_size}×{batch_size})="
                    f"{expected_micro_batches}."
                )

        phases: list[PhaseSpec] = []
        cursor = 0
        for phase_index, raw_phase in enumerate(raw_phases):
            name = str(_get(raw_phase, "name", "")).strip()
            if not name:
                raise ValueError(f"training.phases[{phase_index}].name must be non-empty.")
            epochs = _require_finite(
                f"training.phases[{phase_index}].epochs",
                _get(raw_phase, "epochs", None),
            )
            phase_updates = _integer_product(
                f"training.phases[{phase_index}].epochs * updates_per_epoch",
                updates_per_epoch,
                epochs,
            )

            lr_cfg = _get(raw_phase, "lr", {})
            lr_schedule = _get(lr_cfg, "schedule", None)
            lr_start = _require_finite(
                f"training.phases[{phase_index}].lr.start",
                _get(lr_cfg, "start", None),
            )
            lr_end = _require_finite(
                f"training.phases[{phase_index}].lr.end", lr_start
                if _get(lr_cfg, "end", None) is None
                else _get(lr_cfg, "end"),
            )
            if lr_start < 0 or lr_end < 0:
                raise ValueError("Learning rates must be non-negative.")
            if lr_schedule is not None and str(lr_schedule) != "constant":
                raise ValueError(
                    f"Explicit phase lr.schedule currently supports only 'constant', "
                    f"got {lr_schedule!r}; phase transitions belong under transition."
                )
            if lr_schedule == "constant" and lr_start != lr_end:
                raise ValueError(
                    f"Phase {name} declares a constant LR but start/end differ: "
                    f"{lr_start} != {lr_end}."
                )

            er_cfg = _get(raw_phase, "error_recycling", {})
            er_mode = str(_get(er_cfg, "mode", "collect_only"))
            if er_mode not in {"collect_only", "collect_and_inject"}:
                raise ValueError(
                    f"Unsupported error_recycling mode {er_mode!r} in phase {name}."
                )

            ramp_updates = 0
            max_active = context = latent = noise = 0.0
            if er_mode == "collect_and_inject":
                transition = _get(raw_phase, "transition", {})
                schedule = str(_get(transition, "schedule", "smoothstep"))
                if schedule != "smoothstep":
                    raise ValueError(
                        f"Phase {name} transition.schedule must be 'smoothstep', "
                        f"got {schedule!r}."
                    )
                ramp_fraction = _require_finite(
                    f"training.phases[{phase_index}].transition.ramp_fraction",
                    _get(transition, "ramp_fraction", None),
                )
                if not 0.0 < ramp_fraction <= 1.0:
                    raise ValueError(
                        f"ramp_fraction must be in (0, 1], got {ramp_fraction}."
                    )
                ramp_updates = _integer_product(
                    f"training.phases[{phase_index}] ramp updates",
                    phase_updates,
                    ramp_fraction,
                )
                max_active = _probability(
                    f"training.phases[{phase_index}].error_recycling.max_active_prob",
                    _get(er_cfg, "max_active_prob", 0.0),
                )
                context = _probability(
                    f"training.phases[{phase_index}].error_recycling.effective_context_prob",
                    _get(er_cfg, "effective_context_prob", 0.0),
                )
                latent = _probability(
                    f"training.phases[{phase_index}].error_recycling.effective_latent_prob",
                    _get(er_cfg, "effective_latent_prob", 0.0),
                )
                noise = _probability(
                    f"training.phases[{phase_index}].error_recycling.effective_noise_prob",
                    _get(er_cfg, "effective_noise_prob", 0.0),
                )
                for label, probability in (
                    ("context", context), ("latent", latent), ("noise", noise)
                ):
                    if probability > max_active:
                        raise ValueError(
                            f"Effective {label} probability ({probability}) cannot "
                            f"exceed max_active_prob ({max_active})."
                        )

            phases.append(
                PhaseSpec(
                    name=name,
                    epochs=epochs,
                    start_update=cursor,
                    end_update=cursor + phase_updates,
                    lr_start=lr_start,
                    lr_end=lr_end,
                    error_recycling_mode=er_mode,
                    ramp_updates=ramp_updates,
                    max_active_prob=max_active,
                    effective_context_prob=context,
                    effective_latent_prob=latent,
                    effective_noise_prob=noise,
                )
            )
            cursor += phase_updates

        checkpoint_epochs = _require_finite(
            "checkpointing.every_epochs", _get(checkpointing, "every_epochs", None)
        )
        checkpoint_updates = _integer_product(
            "checkpointing.every_epochs * updates_per_epoch",
            updates_per_epoch,
            checkpoint_epochs,
        )
        total_epochs = sum(phase.epochs for phase in phases)
        return cls(
            micro_batches_per_epoch=micro_batches,
            gradient_accumulation_steps=accumulation,
            updates_per_epoch=updates_per_epoch,
            phases=tuple(phases),
            total_epochs=total_epochs,
            total_updates=cursor,
            checkpoint_interval_updates=checkpoint_updates,
        )

    def values_at(self, update_index: int) -> ScheduleValues:
        if isinstance(update_index, bool) or int(update_index) != update_index:
            raise ValueError(f"update_index must be an integer, got {update_index!r}.")
        update_index = int(update_index)
        if update_index < 0 or update_index >= self.total_updates:
            raise IndexError(
                f"update_index {update_index} outside [0, {self.total_updates - 1}]."
            )

        for phase_index, phase in enumerate(self.phases):
            if phase.start_update <= update_index < phase.end_update:
                break
        else:  # pragma: no cover - guarded by total_updates bounds above
            raise AssertionError("Resolved schedule has a gap.")

        if phase.error_recycling_mode == "collect_and_inject":
            local_index = update_index - phase.start_update
            if phase.ramp_updates == 1:
                ramp_u = 1.0
            else:
                ramp_u = min(max(local_index / (phase.ramp_updates - 1), 0.0), 1.0)
            ramp_s = smoothstep(ramp_u)
        else:
            ramp_u = ramp_s = 0.0

        lr = phase.lr_start + (phase.lr_end - phase.lr_start) * ramp_s
        active = phase.max_active_prob * ramp_s
        effective_context = phase.effective_context_prob * ramp_s
        effective_latent = phase.effective_latent_prob * ramp_s
        effective_noise = phase.effective_noise_prob * ramp_s

        def conditional(effective: float) -> float:
            if phase.max_active_prob == 0.0:
                return 0.0
            return effective / phase.max_active_prob

        return ScheduleValues(
            update_index=update_index,
            optimizer_step=update_index + 1,
            phase_index=phase_index,
            phase_name=phase.name,
            lr=lr,
            ramp_u=ramp_u,
            ramp_s=ramp_s,
            error_recycling_mode=phase.error_recycling_mode,
            active_probability=active,
            context_probability_given_active=conditional(
                phase.effective_context_prob
            ),
            latent_probability_given_active=conditional(
                phase.effective_latent_prob
            ),
            noise_probability_given_active=conditional(
                phase.effective_noise_prob
            ),
            effective_context_probability=effective_context,
            effective_latent_probability=effective_latent,
            effective_noise_probability=effective_noise,
        )

    def is_checkpoint_step(self, completed_optimizer_steps: int) -> bool:
        if completed_optimizer_steps <= 0 or completed_optimizer_steps > self.total_updates:
            return False
        return completed_optimizer_steps % self.checkpoint_interval_updates == 0

    def epoch_position(self, completed_micro_batches: int) -> tuple[int, int, float]:
        if completed_micro_batches < 0:
            raise ValueError("completed_micro_batches must be non-negative.")
        epoch, cursor = divmod(completed_micro_batches, self.micro_batches_per_epoch)
        progress = cursor / self.micro_batches_per_epoch
        return epoch, cursor, progress

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phases"] = [asdict(phase) for phase in self.phases]
        return value

    def resolved_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _probability(name: str, value: Any) -> float:
    result = _require_finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {result}.")
    return result


def smoothstep(value: float) -> float:
    value = min(max(float(value), 0.0), 1.0)
    return 3.0 * value * value - 2.0 * value * value * value


def resolve_stage1_schedule(config: Any, dataloader_length: int) -> Stage1Schedule:
    """Compatibility-friendly functional entry point used by the trainer/tests."""
    return Stage1Schedule.from_config(config, dataloader_length)
