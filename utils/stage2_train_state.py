"""Pure Stage-2 training schedule, clock, and loss-scaling primitives.

The trainer owns model execution.  This module owns the smaller contract that
must never be implicit: a cycle is exactly five fake-score updates followed by
one generator update, phase/DFD decisions come from the successful generator
clock, and the generator clock is committed only after its required EMA action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Mapping

import torch
import torch.distributed as dist

STAGE2_TRAIN_STATE_SCHEMA = "longlive_stage2_train_state/v1"
STAGE2_CYCLE_SUBSTEPS = ("F1", "F2", "F3", "F4", "F5", "G")
STAGE2_GENERATOR_BRANCHES = ("dmd", "dfd")
STAGE2_EMA_ACTIONS = ("skipped", "initialized", "updated")

_SUBSTEP_FAKE_OFFSET = {
    substep: index for index, substep in enumerate(STAGE2_CYCLE_SUBSTEPS)
}
_STATE_KEYS = {
    "schema",
    "completed_g",
    "completed_f",
    "cycle",
    "next_substep",
    "successful_attempts",
    "nonfinite_attempts",
    "nonfinite_attempts_by_role",
}


def _plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}, got {value!r}")
    return value


def _plain_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number, got {value!r}")
    return value


@dataclass(frozen=True)
class Stage2GeneratorPosition:
    """Schedule values for the next not-yet-committed G update."""

    completed_g_before_update: int
    generator_update: int
    global_epoch: int
    phase: str
    phase_epoch: int
    epoch_update_index: int
    dfd_probability: float


@dataclass(frozen=True)
class Stage2GeneratorBranchDecision:
    """Auditable result of the rank-0 DFD branch draw."""

    branch: str
    probability: float
    generator_update: int


@dataclass(frozen=True)
class Stage2TrainingSchedule:
    """Immutable schedule derived from a resolved Stage-2 config."""

    generator_updates_per_epoch: int
    fake_updates_per_generator_update: int
    phase_a_generator_updates: int
    phase_b_generator_updates: int
    total_generator_updates: int
    phase_b_mode: str
    phase_b_dfd_probability_max: float
    ema_initialize_at_completed_generator_update: int
    ema_first_decay_completed_generator_update: int
    nonfinite_max_attempts_per_update: int

    def __post_init__(self) -> None:
        self.validate()

    @classmethod
    def from_resolved_config(cls, resolved: Any) -> "Stage2TrainingSchedule":
        schedule = cls(
            generator_updates_per_epoch=int(resolved.generator_updates_per_epoch),
            fake_updates_per_generator_update=int(
                resolved.fake_updates_per_generator_update
            ),
            phase_a_generator_updates=int(resolved.phase_a_generator_updates),
            phase_b_generator_updates=int(resolved.phase_b_generator_updates),
            total_generator_updates=int(resolved.total_generator_updates),
            phase_b_mode=str(resolved.phase_b_mode),
            phase_b_dfd_probability_max=float(resolved.phase_b_dfd_probability_max),
            ema_initialize_at_completed_generator_update=int(
                resolved.ema_initialize_at_completed_generator_update
            ),
            ema_first_decay_completed_generator_update=int(
                resolved.ema_first_decay_completed_generator_update
            ),
            nonfinite_max_attempts_per_update=int(
                resolved.nonfinite_max_attempts_per_update
            ),
        )
        expected_fake_a = (
            schedule.phase_a_generator_updates
            * schedule.fake_updates_per_generator_update
        )
        expected_fake_b = (
            schedule.phase_b_generator_updates
            * schedule.fake_updates_per_generator_update
        )
        if int(resolved.phase_a_fake_updates) != expected_fake_a:
            raise ValueError(
                "resolved phase_a_fake_updates disagrees with the 5F schedule"
            )
        if int(resolved.phase_b_fake_updates) != expected_fake_b:
            raise ValueError(
                "resolved phase_b_fake_updates disagrees with the 5F schedule"
            )
        if int(resolved.total_fake_updates) != expected_fake_a + expected_fake_b:
            raise ValueError("resolved total_fake_updates disagrees with the schedule")
        return schedule

    @property
    def phase_a_fake_updates(self) -> int:
        return self.phase_a_generator_updates * self.fake_updates_per_generator_update

    @property
    def phase_b_fake_updates(self) -> int:
        return self.phase_b_generator_updates * self.fake_updates_per_generator_update

    @property
    def total_fake_updates(self) -> int:
        return self.total_generator_updates * self.fake_updates_per_generator_update

    def validate(self) -> None:
        per_epoch = _plain_int(
            self.generator_updates_per_epoch,
            "generator_updates_per_epoch",
            minimum=2,
        )
        fake_ratio = _plain_int(
            self.fake_updates_per_generator_update,
            "fake_updates_per_generator_update",
            minimum=1,
        )
        if fake_ratio != 5:
            raise ValueError(
                "Stage-2 fake_updates_per_generator_update must be exactly 5"
            )
        phase_a = _plain_int(
            self.phase_a_generator_updates,
            "phase_a_generator_updates",
            minimum=1,
        )
        phase_b = _plain_int(
            self.phase_b_generator_updates,
            "phase_b_generator_updates",
        )
        total = _plain_int(
            self.total_generator_updates,
            "total_generator_updates",
            minimum=1,
        )
        if phase_a % per_epoch or phase_b % per_epoch:
            raise ValueError("Stage-2 phase update counts must contain whole epochs")
        if total != phase_a + phase_b:
            raise ValueError(
                "total_generator_updates must equal Phase-A plus Phase-B updates"
            )
        probability = _plain_float(
            self.phase_b_dfd_probability_max,
            "phase_b_dfd_probability_max",
        )
        if probability < 0.0 or probability > 1.0:
            raise ValueError("phase_b_dfd_probability_max must be in [0, 1]")
        if phase_b == 0:
            if self.phase_b_mode != "disabled" or probability != 0.0:
                raise ValueError(
                    "an A-only Stage-2 schedule requires disabled Phase B and p_DFD=0"
                )
        elif self.phase_b_mode == "dmd_only":
            if probability != 0.0:
                raise ValueError("the matched DMD control requires p_DFD=0")
        elif self.phase_b_mode != "dmd_dfd":
            raise ValueError("Phase B must use mode 'dmd_dfd' or 'dmd_only'")
        ema_start = _plain_int(
            self.ema_initialize_at_completed_generator_update,
            "ema_initialize_at_completed_generator_update",
            minimum=1,
        )
        ema_decay_start = _plain_int(
            self.ema_first_decay_completed_generator_update,
            "ema_first_decay_completed_generator_update",
            minimum=1,
        )
        if ema_decay_start != ema_start + 1:
            raise ValueError("the first EMA decay must be the update after EMA init")
        if ema_start > total:
            raise ValueError("EMA initialization lies after the end of training")
        _plain_int(
            self.nonfinite_max_attempts_per_update,
            "nonfinite_max_attempts_per_update",
            minimum=1,
        )

    def next_generator_position(self, completed_g: int) -> Stage2GeneratorPosition:
        completed_g = _plain_int(completed_g, "completed_g")
        if completed_g >= self.total_generator_updates:
            raise StopIteration(
                "Stage-2 generator schedule is complete; there is no next G update"
            )
        global_epoch = completed_g // self.generator_updates_per_epoch + 1
        if completed_g < self.phase_a_generator_updates:
            phase = "A"
            phase_offset = completed_g
            probability = 0.0
        else:
            phase = "B"
            phase_offset = completed_g - self.phase_a_generator_updates
            if self.phase_b_mode == "dmd_only":
                probability = 0.0
            elif phase_offset < self.generator_updates_per_epoch:
                probability = (
                    self.phase_b_dfd_probability_max
                    * (phase_offset % self.generator_updates_per_epoch)
                    / (self.generator_updates_per_epoch - 1)
                )
            else:
                probability = self.phase_b_dfd_probability_max
        return Stage2GeneratorPosition(
            completed_g_before_update=completed_g,
            generator_update=completed_g + 1,
            global_epoch=global_epoch,
            phase=phase,
            phase_epoch=phase_offset // self.generator_updates_per_epoch + 1,
            epoch_update_index=phase_offset % self.generator_updates_per_epoch,
            dfd_probability=float(probability),
        )

    def expected_ema_action(self, completed_g: int) -> str:
        completed_g = _plain_int(completed_g, "completed_g", minimum=1)
        if completed_g > self.total_generator_updates:
            raise ValueError("completed_g lies after the configured training schedule")
        if completed_g < self.ema_initialize_at_completed_generator_update:
            return "skipped"
        if completed_g == self.ema_initialize_at_completed_generator_update:
            return "initialized"
        return "updated"


@dataclass
class Stage2TrainingState:
    """Committed successful clocks; no partial microbatch state lives here."""

    completed_g: int = 0
    completed_f: int = 0
    cycle: int = 0
    next_substep: str = "F1"
    successful_attempts: int = 0
    nonfinite_attempts: int = 0
    nonfinite_attempts_by_role: dict[str, int] = field(
        default_factory=lambda: {"generator": 0, "fake_score": 0}
    )

    def __post_init__(self) -> None:
        self._validate_fields()

    @property
    def completed_generator_updates(self) -> int:
        return self.completed_g

    @property
    def completed_fake_score_updates(self) -> int:
        return self.completed_f

    @property
    def is_cycle_boundary(self) -> bool:
        return self.next_substep == "F1" and self.completed_f == 5 * self.completed_g

    @property
    def current_role(self) -> str:
        return "generator" if self.next_substep == "G" else "fake_score"

    def _validate_fields(self) -> None:
        for name in (
            "completed_g",
            "completed_f",
            "cycle",
            "successful_attempts",
            "nonfinite_attempts",
        ):
            _plain_int(getattr(self, name), name)
        if not isinstance(self.nonfinite_attempts_by_role, Mapping):
            raise TypeError("nonfinite_attempts_by_role must be a mapping")
        if set(self.nonfinite_attempts_by_role) != {"generator", "fake_score"}:
            raise ValueError(
                "nonfinite_attempts_by_role must contain exactly generator/fake_score"
            )
        for role, count in self.nonfinite_attempts_by_role.items():
            _plain_int(count, f"nonfinite_attempts_by_role.{role}")
        if sum(self.nonfinite_attempts_by_role.values()) != self.nonfinite_attempts:
            raise ValueError(
                "nonfinite_attempts must equal the sum of the per-role counters"
            )
        if self.next_substep not in STAGE2_CYCLE_SUBSTEPS:
            raise ValueError(
                f"next_substep must be one of {STAGE2_CYCLE_SUBSTEPS}, "
                f"got {self.next_substep!r}"
            )

    def validate(self, schedule: Stage2TrainingSchedule) -> None:
        self._validate_fields()
        schedule.validate()
        if self.completed_g > schedule.total_generator_updates:
            raise ValueError("completed_g exceeds the configured schedule")
        if self.cycle != self.completed_g:
            raise ValueError("cycle must equal completed_g at every committed substep")
        expected_f = 5 * self.completed_g + _SUBSTEP_FAKE_OFFSET[self.next_substep]
        if self.completed_f != expected_f:
            raise ValueError(
                "completed_f is inconsistent with completed_g/next_substep: "
                f"expected {expected_f}, got {self.completed_f}"
            )
        if self.successful_attempts != self.completed_f + self.completed_g:
            raise ValueError("successful_attempts must equal completed_f + completed_g")
        if (
            self.completed_g == schedule.total_generator_updates
            and not self.is_cycle_boundary
        ):
            raise ValueError("a complete Stage-2 run must end at an F1 cycle boundary")

    def is_complete(self, schedule: Stage2TrainingSchedule) -> bool:
        self.validate(schedule)
        return self.completed_g == schedule.total_generator_updates

    def assert_checkpointable(self, schedule: Stage2TrainingSchedule) -> None:
        self.validate(schedule)
        if not self.is_cycle_boundary:
            raise RuntimeError(
                "Stage-2 checkpoints are only legal at next_substep=F1 cycle boundaries"
            )

    def record_nonfinite_attempt(self, role: str | None = None) -> None:
        role = self.current_role if role is None else role
        if role not in {"generator", "fake_score"}:
            raise ValueError(f"invalid nonfinite role {role!r}")
        self.nonfinite_attempts += 1
        self.nonfinite_attempts_by_role[role] += 1

    def commit_successful_fake_update(
        self,
        substep: str,
        *,
        schedule: Stage2TrainingSchedule,
    ) -> None:
        self.validate(schedule)
        if self.completed_g >= schedule.total_generator_updates:
            raise RuntimeError("cannot commit F after Stage-2 training is complete")
        if substep != self.next_substep or substep == "G":
            raise RuntimeError(
                f"expected successful {self.next_substep}, got fake update {substep!r}"
            )
        index = STAGE2_CYCLE_SUBSTEPS.index(substep)
        self.completed_f += 1
        self.successful_attempts += 1
        self.next_substep = STAGE2_CYCLE_SUBSTEPS[index + 1]
        self.validate(schedule)

    def commit_successful_generator_update(
        self,
        *,
        ema_action: str,
        schedule: Stage2TrainingSchedule,
    ) -> None:
        self.validate(schedule)
        if self.next_substep != "G":
            raise RuntimeError(
                f"generator update is illegal while next_substep={self.next_substep}"
            )
        completed_g = self.completed_g + 1
        expected_ema = schedule.expected_ema_action(completed_g)
        if ema_action not in STAGE2_EMA_ACTIONS or ema_action != expected_ema:
            raise RuntimeError(
                "EMA action must complete before committing G: "
                f"expected {expected_ema!r}, got {ema_action!r}"
            )
        self.completed_g = completed_g
        self.cycle = completed_g
        self.successful_attempts += 1
        self.next_substep = "F1"
        self.validate(schedule)

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": STAGE2_TRAIN_STATE_SCHEMA,
            "completed_g": self.completed_g,
            "completed_f": self.completed_f,
            "cycle": self.cycle,
            "next_substep": self.next_substep,
            "successful_attempts": self.successful_attempts,
            "nonfinite_attempts": self.nonfinite_attempts,
            "nonfinite_attempts_by_role": dict(self.nonfinite_attempts_by_role),
        }

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, Any],
        *,
        schedule: Stage2TrainingSchedule,
    ) -> "Stage2TrainingState":
        if not isinstance(state_dict, Mapping):
            raise TypeError("Stage-2 training state must be a mapping")
        missing = sorted(_STATE_KEYS - set(state_dict))
        extra = sorted(set(state_dict) - _STATE_KEYS)
        if missing or extra:
            raise ValueError(
                f"Stage-2 training state keys mismatch: missing={missing}, extra={extra}"
            )
        if state_dict["schema"] != STAGE2_TRAIN_STATE_SCHEMA:
            raise ValueError(
                f"unsupported Stage-2 train state: {state_dict['schema']!r}"
            )
        state = cls(
            completed_g=state_dict["completed_g"],
            completed_f=state_dict["completed_f"],
            cycle=state_dict["cycle"],
            next_substep=state_dict["next_substep"],
            successful_attempts=state_dict["successful_attempts"],
            nonfinite_attempts=state_dict["nonfinite_attempts"],
            nonfinite_attempts_by_role=dict(state_dict["nonfinite_attempts_by_role"]),
        )
        state.validate(schedule)
        return state


def draw_stage2_generator_branch(
    *,
    schedule: Stage2TrainingSchedule,
    completed_g: int,
    generator: torch.Generator,
    device: torch.device | str | None = None,
    synchronize_ranks: bool = True,
) -> Stage2GeneratorBranchDecision:
    """Draw one rank-consistent DMD/DFD choice from the dedicated CPU RNG.

    Rank 0 always consumes exactly one scalar, including when ``p_DFD=0``.
    This makes branch-stream progression independent of conditional shortcuts.
    """

    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an explicit torch.Generator")
    if torch.device(generator.device).type != "cpu":
        raise ValueError("the Stage-2 DFD branch generator must be a CPU generator")
    position = schedule.next_generator_position(completed_g)
    distributed = synchronize_ranks and dist.is_available() and dist.is_initialized()
    backend = str(dist.get_backend()).lower() if distributed else ""
    if device is None:
        if "nccl" in backend:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "NCCL Stage-2 branch broadcast requires a visible CUDA device"
                )
            target_device = torch.device("cuda", torch.cuda.current_device())
        else:
            target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)
    if distributed and "nccl" in backend and target_device.type != "cuda":
        raise ValueError(
            "NCCL cannot broadcast the Stage-2 branch flag from CPU; "
            "use the current-rank CUDA device or leave device=None"
        )
    rank = dist.get_rank() if distributed else 0
    if rank == 0:
        draw = torch.rand((), generator=generator, device="cpu")
        branch_flag = torch.tensor(
            int(float(draw.item()) < position.dfd_probability),
            dtype=torch.uint8,
            device=target_device,
        )
    else:
        branch_flag = torch.empty((), dtype=torch.uint8, device=target_device)
    if distributed:
        dist.broadcast(branch_flag, src=0)
    value = int(branch_flag.to(device="cpu").item())
    if value not in (0, 1):
        raise RuntimeError(f"broadcast Stage-2 branch flag is invalid: {value}")
    return Stage2GeneratorBranchDecision(
        branch=STAGE2_GENERATOR_BRANCHES[value],
        probability=position.dfd_probability,
        generator_update=position.generator_update,
    )


def stage2_global_mean_loss_for_backward(
    local_numerator: torch.Tensor,
    *,
    global_count: torch.Tensor | float | int,
    world_size: int,
) -> torch.Tensor:
    """Scale one local numerator for an exact WORLD-global mean gradient.

    FSDP2 averages synchronized gradients over ``world_size``.  Multiplying
    each local numerator by ``world_size / global_count`` therefore yields the
    gradient of ``sum(all numerators) / sum(all counts)`` after reduce-scatter.
    The same formula works for unequal local counts and any accumulation depth.
    """

    if not torch.is_tensor(local_numerator) or local_numerator.numel() != 1:
        raise ValueError("local_numerator must be a scalar tensor")
    world_size = _plain_int(world_size, "world_size", minimum=1)
    denominator = torch.as_tensor(
        global_count,
        device=local_numerator.device,
        dtype=local_numerator.dtype,
    )
    if denominator.numel() != 1:
        raise ValueError("global_count must be one positive finite scalar")
    if not bool(torch.isfinite(denominator).item()) or not bool(
        (denominator > 0).item()
    ):
        raise ValueError("global_count must be one positive finite scalar")
    return local_numerator * (float(world_size) / denominator)


__all__ = [
    "STAGE2_CYCLE_SUBSTEPS",
    "STAGE2_EMA_ACTIONS",
    "STAGE2_GENERATOR_BRANCHES",
    "STAGE2_TRAIN_STATE_SCHEMA",
    "Stage2GeneratorBranchDecision",
    "Stage2GeneratorPosition",
    "Stage2TrainingSchedule",
    "Stage2TrainingState",
    "draw_stage2_generator_branch",
    "stage2_global_mean_loss_for_backward",
]
