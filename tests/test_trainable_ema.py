from __future__ import annotations

import pytest
import torch
from torch import nn

from utils.distributed import TrainableShardedEMA


class TinyLocalLora(nn.Module):
    def __init__(self, value: float = 0.0):
        super().__init__()
        self.base = nn.Parameter(torch.full((2, 2), -7.0), requires_grad=False)
        self.layer = nn.Module()
        self.layer.lora_A = nn.ParameterDict(
            {"default": nn.Parameter(torch.full((2, 3), value, dtype=torch.float32))}
        )
        self.layer.lora_B = nn.ParameterDict(
            {"default": nn.Parameter(torch.full((4, 2), value, dtype=torch.float32))}
        )


def trainable_values(module):
    return {
        name: parameter.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    }


def fill_trainable(module, value):
    for parameter in module.parameters():
        if parameter.requires_grad:
            parameter.data.fill_(value)


def test_ema_initializes_at_start_step_then_decays_in_cpu_fp32():
    model = TinyLocalLora(value=1.0)
    ema = TrainableShardedEMA(
        model,
        decay=0.5,
        start_step=3,
        topology={"sp_size": 3, "dp_size": 2},
    )

    assert ema.update_after_step(model, 1) == "skipped"
    assert ema.update_after_step(model, 2) == "skipped"
    fill_trainable(model, 3.0)
    assert ema.update_after_step(model, 3) == "initialized"
    assert ema.initialized
    assert all(tensor.device.type == "cpu" for tensor in ema.shadow.values())
    assert all(tensor.dtype == torch.float32 for tensor in ema.shadow.values())
    assert all(torch.equal(tensor, torch.full_like(tensor, 3.0)) for tensor in ema.shadow.values())

    fill_trainable(model, 5.0)
    assert ema.update_after_step(model, 4) == "updated"
    assert all(torch.equal(tensor, torch.full_like(tensor, 4.0)) for tensor in ema.shadow.values())
    with pytest.raises(ValueError, match="must increase"):
        ema.update_after_step(model, 4)


def test_ema_refuses_to_silently_skip_the_start_step():
    model = TinyLocalLora(value=1.0)
    ema = TrainableShardedEMA(model, decay=0.9, start_step=3)
    with pytest.raises(RuntimeError, match="start step was skipped"):
        ema.update_after_step(model, 4)


def test_swap_into_restores_raw_shards_after_exception():
    model = TinyLocalLora(value=2.0)
    ema = TrainableShardedEMA(model, decay=0.9, start_step=1)
    assert ema.update_after_step(model, 1) == "initialized"
    fill_trainable(model, 9.0)
    raw = trainable_values(model)

    with pytest.raises(RuntimeError, match="gather failed"):
        with ema.swap_into(model):
            for name, parameter in model.named_parameters():
                if parameter.requires_grad:
                    cleaned = ema._clean_param_name(name)
                    assert torch.equal(parameter, ema.shadow[cleaned])
            raise RuntimeError("gather failed")

    restored = trainable_values(model)
    assert raw.keys() == restored.keys()
    for name in raw:
        assert torch.equal(restored[name], raw[name])


def test_ema_state_roundtrip_and_topology_validation():
    model = TinyLocalLora(value=2.0)
    ema = TrainableShardedEMA(
        model,
        decay=0.9,
        start_step=1,
        topology={"sp_size": 3, "dp_size": 2},
    )
    ema.update_after_step(model, 1)
    fill_trainable(model, 4.0)
    ema.update_after_step(model, 2)
    state = ema.state_dict()

    restored_model = TinyLocalLora(value=-1.0)
    restored = TrainableShardedEMA(
        restored_model,
        decay=0.9,
        start_step=1,
        topology={"sp_size": 3, "dp_size": 2},
    )
    restored.load_state_dict(state, restored_model)
    assert restored.last_completed_step == 2
    assert restored.initialized
    assert state["shadow"].keys() == restored.shadow.keys()
    for name in restored.shadow:
        assert torch.equal(restored.shadow[name], state["shadow"][name])

    wrong_topology = TrainableShardedEMA(
        TinyLocalLora(),
        decay=0.9,
        start_step=1,
        topology={"sp_size": 2, "dp_size": 3},
    )
    with pytest.raises(ValueError, match="logical topology mismatch"):
        wrong_topology.load_state_dict(state, TinyLocalLora())


def test_ema_rejects_non_lora_trainables_and_nonfinite_updates():
    model = TinyLocalLora()
    model.base.requires_grad_(True)
    with pytest.raises(ValueError, match="only accepts LoRA"):
        TrainableShardedEMA(model)

    model = TinyLocalLora()
    ema = TrainableShardedEMA(model, start_step=1)
    next(parameter for parameter in model.parameters() if parameter.requires_grad).data.fill_(
        torch.nan
    )
    with pytest.raises(ValueError, match="non-finite"):
        ema.update_after_step(model, 1)
