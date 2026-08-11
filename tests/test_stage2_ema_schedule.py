from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from utils.distributed import TrainableShardedEMA
from utils.stage2_config import load_stage2_config
from utils.stage2_train_state import Stage2TrainingSchedule, Stage2TrainingState

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "train_i2v_stage2_600cats.yaml"


class TinyGeneratorLora(nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.frozen = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        self.layer = nn.Module()
        self.layer.lora_A = nn.ParameterDict(
            {"default": nn.Parameter(torch.full((2, 3), value))}
        )
        self.layer.lora_B = nn.ParameterDict(
            {"default": nn.Parameter(torch.full((3, 2), value))}
        )


def _fill_trainable(model, value):
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data.fill_(value)


def test_stage2_ema_skips_through_g39_copies_g40_and_first_decays_at_g41():
    model = TinyGeneratorLora(1.0)
    ema = TrainableShardedEMA(
        model,
        decay=0.99,
        start_step=40,
        topology={"rank_layout": tuple(range(8)), "mesh_dim_names": ("shard",)},
    )

    for completed_g in range(1, 40):
        _fill_trainable(model, float(completed_g))
        assert ema.update_after_step(model, completed_g) == "skipped"
    assert not ema.initialized
    assert ema.last_completed_step == 39

    _fill_trainable(model, 40.0)
    assert ema.update_after_step(model, 40) == "initialized"
    assert all(
        torch.equal(value, torch.full_like(value, 40.0))
        for value in ema.shadow.values()
    )

    # Five F updates and a non-finite G attempt must not call EMA at all.
    state_before_f_and_retry = ema.state_dict()
    assert ema.last_completed_step == 40
    assert state_before_f_and_retry["last_completed_step"] == 40

    _fill_trainable(model, 50.0)
    assert ema.update_after_step(model, 41) == "updated"
    assert all(
        torch.allclose(value, torch.full_like(value, 40.1))
        for value in ema.shadow.values()
    )


def test_state_commit_and_ema_share_the_same_successful_g_clock():
    schedule = Stage2TrainingSchedule.from_resolved_config(
        load_stage2_config(CONFIG_PATH)
    )
    state = Stage2TrainingState(
        completed_g=40,
        completed_f=205,
        cycle=40,
        next_substep="G",
        successful_attempts=245,
    )

    with pytest.raises(RuntimeError, match="EMA action"):
        state.commit_successful_generator_update(
            ema_action="initialized", schedule=schedule
        )
    assert state.completed_g == 40
    assert state.next_substep == "G"

    state.commit_successful_generator_update(ema_action="updated", schedule=schedule)
    assert state.completed_g == 41
    assert state.completed_f == 205
    assert state.next_substep == "F1"
