import copy
from functools import lru_cache
import random

import numpy as np
import pytest
import torch

from utils.lora_utils import LocalLoraShard, LoraTensorSpec
from utils.stage2_checkpoint import (
    STAGE2_CHECKPOINT_MILESTONES,
    Stage2CollectiveOps,
    audit_stage2_lora_optimizer,
    build_stage2_trainer_state,
    capture_stage2_rng_state,
    consolidate_stage2_lora_shards,
    derive_stage2_phase_state,
    gather_stage2_optimizer_state,
    restore_stage2_optimizer_state,
    restore_stage2_rng_state,
    validate_stage2_cycle_boundary,
    validate_stage2_full_optimizer_state,
    validate_stage2_rng_state,
    validate_stage2_trainer_state,
)
from utils.stage2_sampler import build_stage2_role_samplers


@lru_cache(maxsize=None)
def _cached_sampler_state(completed_f: int, completed_g: int) -> dict:
    actions = ("a", "b", "c")
    streams = build_stage2_role_samplers(
        [action for action in actions for _ in range(200)],
        [(30, 52)] * 600,
        action_order=actions,
        base_seed=17,
    )
    for _ in range(completed_f):
        streams.fake_score.next_global_batch()
    for _ in range(completed_g):
        streams.generator.next_global_batch()
    return streams.state_dict()


def _sampler_state(completed_f: int, completed_g: int) -> dict:
    return copy.deepcopy(_cached_sampler_state(completed_f, completed_g))


def _loader_states() -> dict[str, torch.Tensor]:
    return {
        role: torch.Generator().manual_seed(seed).get_state()
        for role, seed in (("fake_score", 101), ("generator", 202))
    }


def _trainer_state(completed_g: int = 10) -> dict:
    return build_stage2_trainer_state(
        completed_generator_updates=completed_g,
        completed_fake_updates=5 * completed_g,
        successful_attempts={
            "generator": completed_g,
            "fake_score": 5 * completed_g,
        },
        nonfinite_attempts={"generator": 1, "fake_score": 2},
        sampler_state=_sampler_state(5 * completed_g, completed_g),
        dataloader_generator_states=_loader_states(),
        run_id="run-a",
        next_attempt_index=6 * completed_g + 3,
        contract_hash="a" * 64,
        launch_hash="b" * 64,
    )


def test_cycle_boundary_is_exactly_five_f_then_g_and_f1():
    boundary = validate_stage2_cycle_boundary(
        completed_generator_updates=7,
        completed_fake_updates=35,
        completed_cycles=7,
        next_substep="F1",
        accumulation_cursor=0,
        pending_state={
            "gradients": False,
            "batch": False,
            "branch": False,
            "kv": False,
            "optimizer_step": False,
        },
    )
    assert boundary == {
        "completed_generator_updates": 7,
        "completed_fake_updates": 35,
        "completed_cycles": 7,
        "next_substep": "F1",
        "accumulation_cursor": 0,
        "pending_state": {
            "gradients": False,
            "batch": False,
            "branch": False,
            "kv": False,
            "optimizer_step": False,
        },
    }


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"completed_fake_updates": 34}, "5\\*completed_generator"),
        ({"completed_cycles": 6}, "completed_cycles"),
        ({"next_substep": "F2"}, "next_substep"),
        ({"accumulation_cursor": 1}, "accumulation_cursor"),
        ({"pending_state": {"gradients": True}}, "pending_state"),
    ],
)
def test_cycle_boundary_rejects_partial_or_pending_state(updates, match):
    kwargs = {
        "completed_generator_updates": 7,
        "completed_fake_updates": 35,
        "completed_cycles": 7,
        "next_substep": "F1",
        "accumulation_cursor": 0,
        "pending_state": {
            "gradients": False,
            "batch": False,
            "branch": False,
            "kv": False,
            "optimizer_step": False,
        },
    }
    if "pending_state" in updates:
        kwargs["pending_state"] = {
            **kwargs["pending_state"],
            **updates["pending_state"],
        }
    else:
        kwargs.update(updates)
    with pytest.raises((RuntimeError, ValueError), match=match):
        validate_stage2_cycle_boundary(**kwargs)


def test_trainer_state_derives_phase_and_refuses_redundant_drift():
    state = _trainer_state(240)
    assert state["completed_cycles"] == 240
    assert state["next_logical_substep_id"] == 1440
    assert state["phase_state"] == derive_stage2_phase_state(240)
    assert state["phase_state"]["phase"] == "B"
    assert state["phase_state"]["next_dfd_probability"] == 0.0
    assert state["phase_state"]["milestone"] == "A24"
    assert 240 in STAGE2_CHECKPOINT_MILESTONES

    broken = copy.deepcopy(state)
    broken["phase_state"]["next_dfd_probability"] = 0.25
    with pytest.raises(RuntimeError, match="phase_state"):
        validate_stage2_trainer_state(broken)


def test_checkpoint_phase_state_uses_the_resolved_config_mode_names():
    control = derive_stage2_phase_state(
        245,
        phase_b_mode="dmd_only",
    )
    assert control["phase"] == "B"
    assert control["next_dfd_probability"] == 0.0

    a_only = derive_stage2_phase_state(
        240,
        phase_b_generator_updates=0,
        phase_b_mode="disabled",
    )
    assert a_only["phase"] == "complete"
    with pytest.raises(ValueError, match="disabled"):
        derive_stage2_phase_state(
            240,
            phase_b_generator_updates=0,
            phase_b_mode="dmd_dfd",
        )


def test_longrun_phase_boundaries_do_not_reuse_legacy_g280_labels():
    kwargs = {
        "phase_a_generator_updates": 3_600,
        "phase_b_generator_updates": 400,
        "phase_b_mode": "dmd_dfd",
    }
    assert derive_stage2_phase_state(280, **kwargs)["milestone"] is None
    assert derive_stage2_phase_state(280, **kwargs)["phase"] == "A"
    phase_b_entry = derive_stage2_phase_state(3_600, **kwargs)
    assert phase_b_entry["phase"] == "B"
    assert phase_b_entry["milestone"] == "A360"
    terminal = derive_stage2_phase_state(4_000, **kwargs)
    assert terminal["phase"] == "complete"
    assert terminal["milestone"] == "B40"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda state: state["successful_attempts"].__setitem__("generator", 9),
            "successful_attempts",
        ),
        (
            lambda state: state["sampler_state"]["fake_score"].__setitem__(
                "completed_batches", 49
            ),
            "sampler",
        ),
        (
            lambda state: state["dataloader_generator_states"].__setitem__(
                "generator", torch.zeros(2, dtype=torch.int64)
            ),
            "DataLoader",
        ),
        (lambda state: state.__setitem__("contract_hash", "bad"), "contract_hash"),
        (lambda state: state.__setitem__("next_substep", "G"), "next_substep"),
    ],
)
def test_trainer_state_rejects_counter_sampler_rng_and_hash_drift(mutate, match):
    state = _trainer_state()
    mutate(state)
    with pytest.raises((RuntimeError, ValueError, TypeError), match=match):
        validate_stage2_trainer_state(state)


def test_rank_rng_roundtrip_covers_general_dedicated_and_rank0_control():
    random.seed(11)
    np.random.seed(12)
    torch.manual_seed(13)
    dedicated = {
        "generator_score": torch.Generator().manual_seed(14),
        "fake_score_noise": torch.Generator().manual_seed(15),
    }
    controls = {
        "generator_exit": torch.Generator().manual_seed(16),
        "fake_score_exit": torch.Generator().manual_seed(17),
        "dfd_branch": torch.Generator().manual_seed(18),
    }
    state = capture_stage2_rng_state(
        rank=0,
        dedicated_generators=dedicated,
        rank0_control_generators=controls,
        include_cuda=False,
    )
    expected = (
        random.random(),
        np.random.rand(),
        torch.rand(2),
        *(torch.rand(2, generator=value) for value in dedicated.values()),
        *(torch.rand(2, generator=value) for value in controls.values()),
    )
    restore_stage2_rng_state(
        state,
        rank=0,
        dedicated_generators=dedicated,
        rank0_control_generators=controls,
        require_cuda_topology=False,
    )
    actual = (
        random.random(),
        np.random.rand(),
        torch.rand(2),
        *(torch.rand(2, generator=value) for value in dedicated.values()),
        *(torch.rand(2, generator=value) for value in controls.values()),
    )
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    for left, right in zip(actual[2:], expected[2:]):
        assert torch.equal(left, right)


def test_rank_rng_rejects_missing_control_without_partial_generator_mutation():
    dedicated = {"score": torch.Generator().manual_seed(1)}
    controls = {
        "generator_exit": torch.Generator().manual_seed(2),
        "fake_score_exit": torch.Generator().manual_seed(3),
        "dfd_branch": torch.Generator().manual_seed(4),
    }
    state = capture_stage2_rng_state(
        rank=0,
        dedicated_generators=dedicated,
        rank0_control_generators=controls,
        include_cuda=False,
    )
    before = dedicated["score"].get_state().clone()
    del state["rank0_control"]["dfd_branch"]
    with pytest.raises(RuntimeError, match="rank0 control"):
        restore_stage2_rng_state(
            state,
            rank=0,
            dedicated_generators=dedicated,
            rank0_control_generators=controls,
            require_cuda_topology=False,
        )
    assert torch.equal(dedicated["score"].get_state(), before)


def test_rank0_structural_rng_audit_never_instantiates_foreign_cuda(
    monkeypatch,
):
    state = capture_stage2_rng_state(
        rank=7,
        dedicated_generators={"score": torch.Generator().manual_seed(9)},
        rank0_control_generators=None,
        include_cuda=False,
    )
    state["dedicated"]["score"]["device"] = "cuda:7"
    original_generator = torch.Generator
    constructed_devices = []

    def guarded_generator(*args, **kwargs):
        device = kwargs.get("device", args[0] if args else "cpu")
        constructed_devices.append(str(device))
        if str(device).startswith("cuda"):
            raise AssertionError("foreign CUDA generator was instantiated")
        return original_generator(*args, **kwargs)

    monkeypatch.setattr(torch, "Generator", guarded_generator)
    validate_stage2_rng_state(state, expected_rank=7)
    assert not any(device.startswith("cuda") for device in constructed_devices)


class TinyRole(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float32))
        self.lora_B = torch.nn.Parameter(torch.ones(4, 2, dtype=torch.float32))


class OuterTinyRole(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyRole()


def _tiny_optimizer_state(step: int = 5):
    names = ("lora_A", "lora_B")
    return {
        "state": {
            name: {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.zeros((2, 3) if name == "lora_A" else (4, 2)),
                "exp_avg_sq": torch.ones((2, 3) if name == "lora_A" else (4, 2)),
            }
            for name in names
        },
        "param_groups": [
            {
                "params": list(names),
                "lr": 2e-6,
                "betas": (0.0, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
            }
        ],
    }


def test_optimizer_state_is_role_exact_fp32_and_step_exact():
    state = _tiny_optimizer_state()
    assert (
        validate_stage2_full_optimizer_state(
            state,
            role="generator",
            expected_parameter_names=("lora_A", "lora_B"),
            expected_completed_updates=5,
        )
        is state
    )
    bad_step = copy.deepcopy(state)
    bad_step["state"]["lora_A"]["step"] = torch.tensor(4.0)
    with pytest.raises(ValueError, match="AdamW step"):
        validate_stage2_full_optimizer_state(
            bad_step,
            role="generator",
            expected_parameter_names=("lora_A", "lora_B"),
            expected_completed_updates=5,
        )
    wrong_dtype = copy.deepcopy(state)
    wrong_dtype["state"]["lora_B"]["exp_avg"] = torch.zeros(4, 2).bfloat16()
    with pytest.raises(TypeError, match="FP32"):
        validate_stage2_full_optimizer_state(
            wrong_dtype,
            role="generator",
            expected_parameter_names=("lora_A", "lora_B"),
            expected_completed_updates=5,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lr", 123.0),
        ("betas", (0.9, 0.9)),
        ("eps", 1.0),
        ("weight_decay", 99.0),
    ],
)
def test_optimizer_state_rejects_any_adamw_hyperparameter_drift(field, value):
    state = _tiny_optimizer_state()
    state["param_groups"][0][field] = value
    with pytest.raises(ValueError, match=field):
        validate_stage2_full_optimizer_state(
            state,
            role="generator",
            expected_parameter_names=("lora_A", "lora_B"),
            expected_completed_updates=5,
        )


def test_fake_score_optimizer_uses_its_distinct_locked_learning_rate():
    state = _tiny_optimizer_state()
    state["param_groups"][0]["lr"] = 4e-7
    assert (
        validate_stage2_full_optimizer_state(
            state,
            role="fake_score",
            expected_parameter_names=("lora_A", "lora_B"),
            expected_completed_updates=5,
        )
        is state
    )


@pytest.mark.parametrize(
    ("role", "learning_rate"),
    (("generator", 1.0e-5), ("fake_score", 2.0e-6)),
)
def test_optimizer_state_accepts_explicit_longrun_resolved_spec(role, learning_rate):
    state = _tiny_optimizer_state()
    state["param_groups"][0]["lr"] = learning_rate
    spec = {
        "role": role,
        "optimizer_type": "adamw",
        "learning_rate": learning_rate,
        "betas": [0.0, 0.999],
        "eps": 1.0e-8,
        "weight_decay": 0.0,
        "schedule": "constant",
    }
    assert (
        validate_stage2_full_optimizer_state(
            state,
            role=role,
            expected_parameter_names=("lora_A", "lora_B"),
            expected_completed_updates=5,
            expected_optimizer_spec=spec,
        )
        is state
    )
    with pytest.raises(ValueError, match="lr mismatch"):
        validate_stage2_full_optimizer_state(
            state,
            role=role,
            expected_parameter_names=("lora_A", "lora_B"),
            expected_completed_updates=5,
        )


def test_optimizer_restore_interface_broadcasts_then_audits_local_moments():
    module = TinyRole()
    optimizer = torch.optim.AdamW(
        [module.lora_A, module.lora_B],
        lr=2e-6,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    state = _tiny_optimizer_state(step=5)
    schema = {
        "lora_A.weight": LoraTensorSpec("lora_A", (2, 3), torch.float32),
        "lora_B.weight": LoraTensorSpec("lora_B", (4, 2), torch.float32),
    }
    calls = []

    def set_state(model, target_optimizer, rank_state, *, options):
        assert model is module
        calls.append((rank_state, options))
        parameters = dict(model.named_parameters())
        for name, values in rank_state["state"].items():
            target_optimizer.state[parameters[name]] = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in values.items()
            }

    options = []

    def options_factory(**kwargs):
        options.append(kwargs)
        return kwargs

    ops = Stage2CollectiveOps(
        get_rank=lambda: 0,
        get_world_size=lambda: 8,
        barrier=lambda: None,
        consensus=lambda success: success,
        broadcast_object=lambda value, src: value,
    )
    restore_stage2_optimizer_state(
        module,
        optimizer,
        state,
        role="generator",
        expected_schema=schema,
        expected_completed_updates=5,
        collectives=ops,
        set_optimizer_state_dict_fn=set_state,
        options_factory=options_factory,
    )
    assert options == [
        {
            "full_state_dict": True,
            "cpu_offload": True,
            "broadcast_from_rank0": True,
        }
    ]
    assert calls[0][0] is state
    assert all(
        parameter in optimizer.state for parameter in (module.lora_A, module.lora_B)
    )


def test_optimizer_gather_canonicalizes_outer_runtime_fqns_to_schema_names():
    module = OuterTinyRole()
    optimizer = torch.optim.AdamW(
        [module.model.lora_A, module.model.lora_B],
        lr=2e-6,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    for parameter in optimizer.param_groups[0]["params"]:
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    schema = {
        "lora_A.weight": LoraTensorSpec("lora_A", (2, 3), torch.float32),
        "lora_B.weight": LoraTensorSpec("lora_B", (4, 2), torch.float32),
    }
    runtime_state = _tiny_optimizer_state(step=1)
    runtime_state["state"] = {
        f"model.{name}": value for name, value in runtime_state["state"].items()
    }
    runtime_state["param_groups"][0]["params"] = ["model.lora_A", "model.lora_B"]

    operations = Stage2CollectiveOps(
        get_rank=lambda: 0,
        get_world_size=lambda: 8,
        barrier=lambda: None,
        consensus=lambda success: success,
        broadcast_object=lambda value, src: value,
    )
    gathered = gather_stage2_optimizer_state(
        module,
        optimizer,
        role="generator",
        expected_schema=schema,
        expected_completed_updates=1,
        collectives=operations,
        get_optimizer_state_dict_fn=lambda *args, **kwargs: runtime_state,
        options_factory=lambda **kwargs: kwargs,
    )

    assert set(gathered["state"]) == {"lora_A", "lora_B"}
    assert gathered["param_groups"][0]["params"] == ["lora_A", "lora_B"]


def test_optimizer_restore_maps_canonical_checkpoint_names_to_outer_runtime_fqns():
    module = OuterTinyRole()
    optimizer = torch.optim.AdamW(
        [module.model.lora_A, module.model.lora_B],
        lr=2e-6,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    schema = {
        "lora_A.weight": LoraTensorSpec("lora_A", (2, 3), torch.float32),
        "lora_B.weight": LoraTensorSpec("lora_B", (4, 2), torch.float32),
    }
    canonical_state = _tiny_optimizer_state(step=5)
    installed_names = []

    def set_state(model, target_optimizer, rank_state, *, options):
        del options
        installed_names.extend(rank_state["param_groups"][0]["params"])
        parameters = dict(model.named_parameters())
        for name, values in rank_state["state"].items():
            target_optimizer.state[parameters[name]] = {
                key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in values.items()
            }

    operations = Stage2CollectiveOps(
        get_rank=lambda: 0,
        get_world_size=lambda: 8,
        barrier=lambda: None,
        consensus=lambda success: success,
        broadcast_object=lambda value, src: value,
    )
    restore_stage2_optimizer_state(
        module,
        optimizer,
        canonical_state,
        role="generator",
        expected_schema=schema,
        expected_completed_updates=5,
        collectives=operations,
        set_optimizer_state_dict_fn=set_state,
        options_factory=lambda **kwargs: kwargs,
    )

    assert installed_names == ["model.lora_A", "model.lora_B"]
    assert all(
        parameter in optimizer.state
        for parameter in (module.model.lora_A, module.model.lora_B)
    )


def test_local_optimizer_audit_rejects_non_lora_and_role_state():
    module = TinyRole()
    optimizer = torch.optim.AdamW(
        [module.lora_A, module.lora_B],
        lr=2e-6,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    assert audit_stage2_lora_optimizer(module, optimizer, role="generator") == (
        "lora_A",
        "lora_B",
    )
    optimizer.add_param_group({"params": [module.base]})
    with pytest.raises(ValueError, match="identities"):
        audit_stage2_lora_optimizer(module, optimizer, role="generator")


def _one_dimensional_shards(world_size=8):
    schema = {
        "block.lora_A.weight": LoraTensorSpec(
            "block.lora_A.default.weight", (10, 2), torch.float32
        )
    }
    full = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    records = []
    chunk = (full.shape[0] + world_size - 1) // world_size
    ranks = tuple(range(world_size))
    for rank in ranks:
        start_row = min(rank * chunk, 10)
        end_row = min(start_row + chunk, 10)
        local = full[start_row:end_row].clone()
        record = LocalLoraShard(
            tensor=local,
            global_shape=(10, 2),
            intra_param_start=start_row * 2,
            shard_rank=rank,
            shard_world_size=world_size,
            shard_group_ranks=ranks,
            fsdp_unit_fingerprint="same",
            local_shape=tuple(local.shape),
            mesh_shape=(world_size,),
            mesh_dim_names=("shard",),
            placements=("shard:0",),
            mesh_coordinate=(rank,),
            mesh_ranks=(ranks,),
            replica_group_ranks=(rank,),
            authoritative_shard_group_ranks=ranks,
            shard_dim=0,
            shard_offset=start_row,
            global_rank=rank,
            is_dtensor=True,
        )
        records.append({"block.lora_A.weight": record})
    return schema, full, records


@pytest.mark.parametrize("world_size", (4, 8))
def test_one_dimensional_selective_lora_consolidation(world_size):
    schema, full, records = _one_dimensional_shards(world_size)
    result = consolidate_stage2_lora_shards(records, expected_schema=schema)
    assert list(result) == ["block.lora_A.weight"]
    assert torch.equal(result["block.lora_A.weight"], full)

    corrupt = copy.deepcopy(records)
    corrupt[-1]["block.lora_A.weight"] = copy.copy(corrupt[-1]["block.lora_A.weight"])
    object.__setattr__(corrupt[-1]["block.lora_A.weight"], "mesh_shape", (2, 2))
    with pytest.raises(ValueError, match="one-dimensional"):
        consolidate_stage2_lora_shards(corrupt, expected_schema=schema)


def test_stage2_collectives_accept_four_or_eight_and_reject_other_world_sizes():
    for world_size in (4, 8):
        Stage2CollectiveOps(
            get_rank=lambda: 0,
            get_world_size=lambda world_size=world_size: world_size,
            barrier=lambda: None,
            consensus=lambda success: success,
            broadcast_object=lambda value, src: value,
        ).validate()
    ops = Stage2CollectiveOps(
        get_rank=lambda: 0,
        get_world_size=lambda: 7,
        barrier=lambda: None,
        consensus=lambda success: success,
        broadcast_object=lambda value, src: value,
    )
    with pytest.raises(RuntimeError, match="WORLD_SIZE"):
        ops.validate()
