from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from utils.stage2_config import load_stage2_config
from utils.stage2_train_state import (
    STAGE2_CYCLE_SUBSTEPS,
    Stage2TrainingSchedule,
    Stage2TrainingState,
    draw_stage2_generator_branch,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "train_i2v_stage2_600cats.yaml"


def _schedule() -> Stage2TrainingSchedule:
    return Stage2TrainingSchedule.from_resolved_config(load_stage2_config(CONFIG_PATH))


def test_state_machine_is_exactly_five_fake_updates_then_one_generator_update():
    schedule = _schedule()
    state = Stage2TrainingState()
    observed = []

    assert schedule.phase_a_generator_updates == 240
    assert schedule.phase_b_generator_updates == 40
    assert schedule.phase_a_fake_updates == 1_200
    assert schedule.phase_b_fake_updates == 200

    for _ in range(schedule.total_generator_updates):
        for substep in STAGE2_CYCLE_SUBSTEPS[:-1]:
            observed.append(state.next_substep)
            state.commit_successful_fake_update(substep, schedule=schedule)
        observed.append(state.next_substep)
        completed_g = state.completed_g + 1
        state.commit_successful_generator_update(
            ema_action=schedule.expected_ema_action(completed_g),
            schedule=schedule,
        )

    assert tuple(observed[:12]) == STAGE2_CYCLE_SUBSTEPS * 2
    assert state.completed_g == 280
    assert state.completed_f == 1_400
    assert state.cycle == 280
    assert state.successful_attempts == 1_680
    assert state.next_substep == "F1"
    assert state.is_cycle_boundary
    assert state.is_complete(schedule)


def test_phase_and_b1_probability_are_derived_only_from_completed_g():
    schedule = _schedule()

    last_a = schedule.next_generator_position(239)
    assert (last_a.phase, last_a.phase_epoch, last_a.epoch_update_index) == (
        "A",
        24,
        9,
    )
    assert last_a.dfd_probability == 0.0

    b1 = [schedule.next_generator_position(240 + index) for index in range(10)]
    assert [position.phase for position in b1] == ["B"] * 10
    assert [position.phase_epoch for position in b1] == [1] * 10
    assert [position.epoch_update_index for position in b1] == list(range(10))
    assert [position.dfd_probability for position in b1] == [
        0.25 * index / 9 for index in range(10)
    ]

    for completed_g in range(250, 280):
        assert schedule.next_generator_position(completed_g).dfd_probability == 0.25

    # Resume uses the successful G clock directly; no implicit epoch cursor exists.
    resumed = Stage2TrainingState(
        completed_g=246,
        completed_f=1_230,
        cycle=246,
        next_substep="F1",
        successful_attempts=1_476,
    )
    resumed.validate(schedule)
    assert schedule.next_generator_position(
        resumed.completed_g
    ).dfd_probability == pytest.approx(0.25 * 6 / 9)


def test_a_only_and_matched_dmd_control_never_select_dfd():
    baseline = load_stage2_config(CONFIG_PATH)

    a_only = _schedule()
    a_only = replace(
        a_only,
        phase_b_generator_updates=0,
        total_generator_updates=baseline.phase_a_generator_updates,
        phase_b_mode="disabled",
        phase_b_dfd_probability_max=0.0,
    )
    a_only.validate()
    assert a_only.next_generator_position(239).phase == "A"
    with pytest.raises(StopIteration, match="complete"):
        a_only.next_generator_position(240)

    dmd_only = replace(
        _schedule(),
        phase_b_mode="dmd_only",
        phase_b_dfd_probability_max=0.0,
    )
    dmd_only.validate()
    assert all(
        dmd_only.next_generator_position(completed_g).dfd_probability == 0.0
        for completed_g in range(240, 280)
    )


def test_nonfinite_attempt_does_not_advance_success_clocks_or_substep():
    schedule = _schedule()
    state = Stage2TrainingState()
    before = state.state_dict()
    state.record_nonfinite_attempt()

    assert state.completed_g == before["completed_g"]
    assert state.completed_f == before["completed_f"]
    assert state.cycle == before["cycle"]
    assert state.next_substep == before["next_substep"]
    assert state.successful_attempts == before["successful_attempts"]
    assert state.nonfinite_attempts == 1
    state.validate(schedule)


def test_generator_commit_requires_matching_ema_action_and_cycle_boundary():
    schedule = _schedule()
    state = Stage2TrainingState(
        completed_g=39,
        completed_f=200,
        cycle=39,
        next_substep="G",
        successful_attempts=239,
    )

    with pytest.raises(RuntimeError, match="EMA action"):
        state.commit_successful_generator_update(
            ema_action="skipped", schedule=schedule
        )
    assert state.next_substep == "G"
    assert state.completed_g == 39

    state.commit_successful_generator_update(
        ema_action="initialized", schedule=schedule
    )
    assert (state.completed_g, state.completed_f, state.cycle) == (40, 200, 40)
    assert state.next_substep == "F1"


def test_dfd_branch_draw_is_seeded_and_reports_the_locked_probability():
    schedule = _schedule()
    left = torch.Generator(device="cpu").manual_seed(1234)
    right = torch.Generator(device="cpu").manual_seed(1234)

    left_decisions = [
        draw_stage2_generator_branch(
            schedule=schedule,
            completed_g=completed_g,
            generator=left,
            synchronize_ranks=False,
        )
        for completed_g in range(240, 280)
    ]
    right_decisions = [
        draw_stage2_generator_branch(
            schedule=schedule,
            completed_g=completed_g,
            generator=right,
            synchronize_ranks=False,
        )
        for completed_g in range(240, 280)
    ]

    assert left_decisions == right_decisions
    assert left_decisions[0].probability == 0.0
    assert left_decisions[9].probability == 0.25
    assert {decision.branch for decision in left_decisions} <= {"dmd", "dfd"}
