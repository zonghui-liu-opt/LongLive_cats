import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import fully_shard

from utils.distributed import TrainableShardedEMA
from utils.lora_utils import LoraTensorSpec
from utils.stage1_checkpoint import (
    Stage1AdapterPair,
    Stage1CollectiveOps,
    apply_resume_retention,
    apply_stage1_resume_retention_collective,
    audit_stage1_lora_optimizer,
    build_stage1_trainer_state,
    build_checkpoint_manifest,
    capture_rng_state,
    checkpoint_directory,
    finalize_stage1_checkpoint,
    find_latest_resumable_checkpoint,
    gather_stage1_optimizer_state,
    gather_stage1_raw_and_ema_adapters,
    heavy_required_files,
    load_stage1_resume_payloads,
    restore_rng_state,
    restore_stage1_optimizer_state,
    validate_checkpoint,
    validate_stage1_adapter_state,
    validate_stage1_full_optimizer_state,
    write_stage1_adapter_pair,
    write_checkpoint_manifest,
    write_stage1_resume_payloads,
    write_success_marker,
)
from utils.stage1_io import atomic_torch_save, atomic_write_json


def _checkpoint(root, step, resumable=True):
    directory = checkpoint_directory(root, step)
    directory.mkdir(parents=True)
    for name in [
        "adapter_raw.safetensors",
        "adapter_ema.safetensors",
        "resolved_config.yaml",
    ]:
        (directory / name).write_bytes(name.encode())
    atomic_write_json(directory / "base_reference.json", {"base_sha256": "base"})
    if resumable:
        for name in heavy_required_files() - {"_RESUMABLE_SUCCESS"}:
            (directory / name).write_bytes(name.encode())
    write_checkpoint_manifest(
        directory,
        completed_step=step,
        world_size=6,
        sequence_parallel_size=3,
        data_parallel_size=2,
        resumable=resumable,
    )
    write_success_marker(directory, resumable=False)
    if resumable:
        write_success_marker(directory, resumable=True)
    return directory


def test_latest_resume_rejects_corrupted_newer_marker(tmp_path):
    _checkpoint(tmp_path, 75, resumable=True)
    incomplete = _checkpoint(tmp_path, 150, resumable=True)
    (incomplete / "rng_state_rank00003.pt").unlink()
    _checkpoint(tmp_path, 225, resumable=False)
    with pytest.raises(RuntimeError, match="incomplete"):
        find_latest_resumable_checkpoint(
            tmp_path, expected_topology=(6, 3, 2), expected_base_sha256="base"
        )


def test_latest_resume_ignores_valid_artifact_only_checkpoint(tmp_path):
    latest = _checkpoint(tmp_path, 75, resumable=True)
    _checkpoint(tmp_path, 225, resumable=False)
    found = find_latest_resumable_checkpoint(
        tmp_path, expected_topology=(6, 3, 2), expected_base_sha256="base"
    )
    assert found == latest


def test_retention_keeps_adapters_success_and_permanent_step(tmp_path):
    for step in (75, 150, 300, 375):
        _checkpoint(tmp_path, step, resumable=True)
    removed = apply_stage1_resume_retention_collective(
        tmp_path,
        keep_last=2,
        keep_steps=[300],
        collectives=_fake_collectives(),
    )
    assert removed == [75, 150]
    for step in removed:
        directory = checkpoint_directory(tmp_path, step)
        assert (directory / "_SUCCESS").is_file()
        assert (directory / "adapter_raw.safetensors").is_file()
        assert (directory / "adapter_ema.safetensors").is_file()
        assert not (directory / "_RESUMABLE_SUCCESS").exists()
        assert validate_checkpoint(directory)["resumable"] is False
    assert apply_resume_retention(tmp_path, keep_last=2, keep_steps=[300]) == []


def test_rng_state_roundtrip_cpu():
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    state = capture_rng_state(include_cuda=False)
    expected = (random.random(), np.random.rand(), torch.rand(1))
    restore_rng_state(state, require_cuda_topology=False)
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_rng_state_roundtrip_touches_only_current_cuda_device(monkeypatch):
    cuda_state = torch.tensor([1, 2, 3], dtype=torch.uint8)
    calls = {"get": [], "set": []}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 6)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state",
        lambda device: calls["get"].append(device) or cuda_state.clone(),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: (_ for _ in ()).throw(AssertionError("must not read all devices")),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda state, device: calls["set"].append((state.clone(), device)),
    )

    state = capture_rng_state(include_cuda=True)
    assert state["schema_version"] == 2
    assert state["cuda_device_count"] == 6
    assert state["cuda_device_index"] == 2
    assert torch.equal(state["torch_cuda"], cuda_state)
    assert calls["get"] == [2]

    restore_rng_state(state, require_cuda_topology=True)
    assert len(calls["set"]) == 1
    assert torch.equal(calls["set"][0][0], cuda_state)
    assert calls["set"][0][1] == 2


def test_manifest_or_topology_tampering_fails_fast(tmp_path):
    directory = _checkpoint(tmp_path, 75, resumable=True)
    path = directory / "checkpoint_manifest.json"
    value = json.loads(path.read_text())
    value["topology"]["world_size"] = 5
    path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        validate_checkpoint(directory, require_resumable=True)


class TinyCheckpointLora(nn.Module):
    def __init__(self, value=1.0):
        super().__init__()
        self.frozen = nn.Parameter(torch.full((3,), -7.0), requires_grad=False)
        self.block = nn.Module()
        self.block.lora_A = nn.ModuleDict(
            {"default": nn.Linear(3, 2, bias=False, dtype=torch.float32)}
        )
        self.block.lora_B = nn.ModuleDict(
            {"default": nn.Linear(2, 4, bias=False, dtype=torch.float32)}
        )
        self.block.lora_A["default"].weight.data.fill_(value)
        self.block.lora_B["default"].weight.data.fill_(value)

    def forward(self):
        return (
            self.block.lora_A["default"].weight.sum()
            + self.block.lora_B["default"].weight.sum()
        )


def _tiny_schema():
    return {
        "block.lora_A.weight": LoraTensorSpec(
            "block.lora_A.default.weight", (2, 3), torch.float32
        ),
        "block.lora_B.weight": LoraTensorSpec(
            "block.lora_B.default.weight", (4, 2), torch.float32
        ),
    }


def _fake_collectives(rank=0, events=None):
    events = [] if events is None else events
    return Stage1CollectiveOps(
        get_rank=lambda: rank,
        get_world_size=lambda: 6,
        barrier=lambda: events.append("barrier"),
        consensus=lambda success: events.append(("consensus", success)) or success,
    )


def _tiny_selective_state(module, **_kwargs):
    named = dict(module.named_parameters())
    return {
        key: named[spec.raw_parameter_name].detach().cpu().clone()
        for key, spec in _tiny_schema().items()
    }


def _identity_directed_gather(local, **_kwargs):
    return local


def _fsdp2_checkpoint_worker(rank, init_file):
    dist.init_process_group(
        "gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=6,
    )
    try:
        torch.manual_seed(2026)
        model = TinyCheckpointLora(value=1.0)
        model.frozen.data = model.frozen.data.to(torch.bfloat16)
        schema = _tiny_schema()
        mesh = DeviceMesh(
            "cpu",
            torch.arange(6, dtype=torch.int32).reshape(2, 3),
            mesh_dim_names=("replicate", "shard"),
        )
        fully_shard(model, mesh=mesh, reshard_after_forward=False)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-3,
        )
        model().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema = TrainableShardedEMA(
            model,
            start_step=1,
            topology={"mesh_shape": (2, 3)},
        )
        ema.update_after_step(model, 1)

        pair = gather_stage1_raw_and_ema_adapters(
            model,
            ema,
            expected_schema=schema,
            authoritative_shard_group=mesh.get_group("shard"),
            replica_group=mesh.get_group("replicate"),
            expected_adapter_tensors=2,
            expected_global_numel=14,
        )
        if rank == 0:
            assert pair.raw is not None and pair.ema is not None
            assert tuple(pair.raw) == tuple(sorted(schema))
        else:
            assert pair.raw is None and pair.ema is None

        full_optimizer = gather_stage1_optimizer_state(
            model,
            optimizer,
            expected_schema=schema,
        )
        optimizer.state.clear()
        restore_stage1_optimizer_state(
            model,
            optimizer,
            full_optimizer,
            expected_schema=schema,
        )
        audit_stage1_lora_optimizer(
            model,
            optimizer,
            expected_schema=schema,
            require_initialized_moments=True,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_fsdp2_six_rank_selective_pair_and_dcp_optimizer_roundtrip(tmp_path):
    init_file = tmp_path / "checkpoint_fsdp2_gloo_init"
    mp.spawn(
        _fsdp2_checkpoint_worker,
        args=(str(init_file),),
        nprocs=6,
        join=True,
    )


def test_collective_raw_ema_gather_writes_canonical_and_restores(tmp_path, monkeypatch):
    from safetensors.torch import load_file

    model = TinyCheckpointLora(value=1.0)
    ema = TrainableShardedEMA(model, start_step=1)
    ema.update_after_step(model, 1)

    def forbidden_model_state(*args, **kwargs):
        raise AssertionError("adapter checkpoint must never request full model state")

    monkeypatch.setattr(model, "state_dict", forbidden_model_state)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data.fill_(9.0)
    raw_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    events = []
    pair = gather_stage1_raw_and_ema_adapters(
        model,
        ema,
        expected_schema=_tiny_schema(),
        authoritative_shard_group=object(),
        replica_group=object(),
        expected_adapter_tensors=2,
        expected_global_numel=14,
        collectives=_fake_collectives(events=events),
        get_local_shards_fn=_tiny_selective_state,
        gather_fn=_identity_directed_gather,
    )
    assert pair.raw is not None and pair.ema is not None
    assert all(torch.equal(value, torch.full_like(value, 9.0)) for value in pair.raw.values())
    assert all(torch.equal(value, torch.full_like(value, 1.0)) for value in pair.ema.values())
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter, raw_before[name])

    paths = write_stage1_adapter_pair(
        tmp_path,
        pair,
        expected_schema=_tiny_schema(),
        completed_step=75,
        expected_adapter_tensors=2,
        expected_global_numel=14,
        collectives=_fake_collectives(events=events),
    )
    assert paths is not None
    assert set(load_file(str(paths[0]))) == set(_tiny_schema())
    assert set(load_file(str(paths[1]))) == set(_tiny_schema())
    assert events.count("barrier") >= 6


def test_ema_gather_failure_restores_raw_and_never_writes_artifacts(tmp_path):
    model = TinyCheckpointLora(value=2.0)
    ema = TrainableShardedEMA(model, start_step=1)
    ema.update_after_step(model, 1)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data.fill_(8.0)
    raw = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    calls = 0

    def fail_second_gather(local, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected EMA gather failure")
        return local

    with pytest.raises(RuntimeError, match="EMA selective directed gather"):
        gather_stage1_raw_and_ema_adapters(
            model,
            ema,
            expected_schema=_tiny_schema(),
            authoritative_shard_group=object(),
            replica_group=object(),
            expected_adapter_tensors=2,
            expected_global_numel=14,
            collectives=_fake_collectives(),
            get_local_shards_fn=_tiny_selective_state,
            gather_fn=fail_second_gather,
        )
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert torch.equal(parameter, raw[name])
    assert not list(tmp_path.iterdir())


def _adam_step(model, optimizer):
    loss = sum(parameter.sum() for parameter in model.parameters() if parameter.requires_grad)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)


def test_dcp_optimizer_only_get_set_contract_is_injectable_and_fp32():
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
    )

    model = TinyCheckpointLora()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    _adam_step(model, optimizer)
    seen = []

    def get_state(module, optim, *, options):
        seen.append((options.full_state_dict, options.cpu_offload))
        return get_optimizer_state_dict(module, optim, options=options)

    full_state = gather_stage1_optimizer_state(
        model,
        optimizer,
        expected_schema=_tiny_schema(),
        expected_completed_step=1,
        collectives=_fake_collectives(),
        get_optimizer_state_dict_fn=get_state,
        options_factory=StateDictOptions,
    )
    assert full_state is not None
    assert seen == [(True, True)]
    assert set(full_state["state"]) == {
        "block.lora_A.default.weight",
        "block.lora_B.default.weight",
    }
    assert all(
        values["exp_avg"].dtype == torch.float32
        and values["exp_avg_sq"].dtype == torch.float32
        for values in full_state["state"].values()
    )

    restored = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    restore_seen = []

    def set_state(module, optim, state, *, options):
        restore_seen.append(
            (options.full_state_dict, options.cpu_offload, options.broadcast_from_rank0)
        )
        named = dict(module.named_parameters())
        for name, values in state["state"].items():
            optim.state[named[name]] = {
                key: value.detach().clone() if isinstance(value, torch.Tensor) else value
                for key, value in values.items()
            }

    restore_stage1_optimizer_state(
        model,
        restored,
        full_state,
        expected_schema=_tiny_schema(),
        expected_completed_step=1,
        collectives=_fake_collectives(),
        set_optimizer_state_dict_fn=set_state,
        options_factory=StateDictOptions,
    )
    assert restore_seen == [(True, True, True)]
    audit_stage1_lora_optimizer(
        model,
        restored,
        expected_schema=_tiny_schema(),
        require_initialized_moments=True,
        expected_completed_step=1,
    )

    with pytest.raises(ValueError, match="AdamW step mismatch"):
        validate_stage1_full_optimizer_state(
            full_state,
            expected_parameter_names=tuple(full_state["state"]),
            expected_completed_step=2,
        )

    broken = {
        "state": {
            name: dict(values) for name, values in full_state["state"].items()
        },
        "param_groups": full_state["param_groups"],
    }
    first = next(iter(broken["state"].values()))
    first["exp_avg"] = first["exp_avg"].to(torch.bfloat16)
    with pytest.raises(RuntimeError, match="optimizer-only state validation"):
        gather_stage1_optimizer_state(
            model,
            optimizer,
            expected_schema=_tiny_schema(),
            collectives=_fake_collectives(),
            get_optimizer_state_dict_fn=lambda *args, **kwargs: broken,
            options_factory=StateDictOptions,
        )


def _trainer_state(step=75):
    return build_stage1_trainer_state(
        optimizer_state={"state": {}, "param_groups": []},
        completed_step=step,
        global_epoch=0,
        committed_microbatch_cursor_in_epoch=150,
        phase_derivation={"phase": "A"},
        sampler_state={"epoch": 0},
        dataloader_generator_state=torch.Generator().manual_seed(7).get_state(),
        next_attempt_index=step,
        nonfinite_attempt_count=0,
        resolved_config_sha256="a" * 64,
    )


def test_heavy_payload_commit_resume_and_marker_order(tmp_path, monkeypatch):
    directory = checkpoint_directory(tmp_path, 75)
    directory.mkdir()
    for name in (
        "adapter_raw.safetensors",
        "adapter_ema.safetensors",
        "resolved_config.yaml",
    ):
        (directory / name).write_bytes(name.encode())
    atomic_write_json(directory / "base_reference.json", {"base_sha256": "base"})

    for rank in range(6):
        written = write_stage1_resume_payloads(
            directory,
            trainer_state=_trainer_state() if rank == 0 else None,
            ema_state={"rank": rank, "shadow": {}},
            rng_state={"rank": rank},
            error_buffer_state={"sp": rank} if rank < 3 else {"ignored": rank},
            collectives=_fake_collectives(rank=rank),
        )
        assert f"ema_local_rank{rank:05d}.pt" in written

    manifest = finalize_stage1_checkpoint(
        directory,
        completed_step=75,
        resumable=True,
        collectives=_fake_collectives(rank=0),
    )
    assert manifest is not None
    assert (directory / "_SUCCESS").is_file()
    assert (directory / "_RESUMABLE_SUCCESS").is_file()
    validate_checkpoint(
        directory,
        require_resumable=True,
        expected_topology=(6, 3, 2),
        expected_base_sha256="base",
    )

    payload = load_stage1_resume_payloads(
        directory,
        replica_group=object(),
        expected_base_sha256="base",
        expected_resolved_config_sha256="a" * 64,
        collectives=_fake_collectives(rank=0),
        broadcast_object_list_fn=lambda holder, **kwargs: None,
        get_process_group_ranks_fn=lambda group: (0, 3),
    )
    assert payload.trainer_state["completed_step"] == 75
    assert "optimizer_state" not in payload.trainer_state
    assert payload.optimizer_state == {"state": {}, "param_groups": []}
    assert payload.ema_state["rank"] == 0
    assert payload.error_buffer_state == {"sp": 0}

    def forbidden_nonzero_hash_scan(*args, **kwargs):
        raise AssertionError("nonzero rank repeated checkpoint/base hash validation")

    monkeypatch.setattr(
        "utils.stage1_checkpoint.validate_checkpoint", forbidden_nonzero_hash_scan
    )

    def inject_rank4(holder, *, src, group=None):
        if group is None:
            trainer = torch.load(
                directory / "trainer_state.pt", map_location="cpu", weights_only=False
            )
            trainer.pop("optimizer_state")
            holder[0] = trainer
        else:
            holder[0] = torch.load(
                directory / "error_buffer_sp1.pt", map_location="cpu", weights_only=False
            )

    replica_payload = load_stage1_resume_payloads(
        directory,
        replica_group=object(),
        expected_base_sha256="base",
        collectives=_fake_collectives(rank=4),
        broadcast_object_list_fn=inject_rank4,
        get_process_group_ranks_fn=lambda group: (1, 4),
    )
    assert replica_payload.trainer_state["completed_step"] == 75
    assert "optimizer_state" not in replica_payload.trainer_state
    assert replica_payload.optimizer_state is None
    assert replica_payload.ema_state["rank"] == 4
    assert replica_payload.error_buffer_state == {"sp": 1}


def test_finalize_failure_never_publishes_a_marker(tmp_path):
    directory = checkpoint_directory(tmp_path, 75)
    directory.mkdir()
    for name in (
        "adapter_raw.safetensors",
        "adapter_ema.safetensors",
        "resolved_config.yaml",
    ):
        (directory / name).write_bytes(b"x")
    atomic_write_json(directory / "base_reference.json", {"base_sha256": "base"})
    with pytest.raises(RuntimeError, match="checkpoint marker preflight"):
        finalize_stage1_checkpoint(
            directory,
            completed_step=75,
            resumable=True,
            collectives=_fake_collectives(),
        )
    assert not (directory / "_SUCCESS").exists()
    assert not (directory / "_RESUMABLE_SUCCESS").exists()
    assert not (directory / "checkpoint_manifest.json").exists()


def test_resumable_marker_failure_rolls_back_artifact_marker(tmp_path, monkeypatch):
    directory = checkpoint_directory(tmp_path, 75)
    directory.mkdir()
    for name in (
        "adapter_raw.safetensors",
        "adapter_ema.safetensors",
        "resolved_config.yaml",
    ):
        (directory / name).write_bytes(b"x")
    atomic_write_json(directory / "base_reference.json", {"base_sha256": "base"})
    for name in heavy_required_files() - {"_RESUMABLE_SUCCESS"}:
        (directory / name).write_bytes(b"heavy")

    real_write_marker = write_success_marker

    def fail_resumable(path, *, resumable):
        if resumable:
            raise OSError("injected marker failure")
        return real_write_marker(path, resumable=resumable)

    monkeypatch.setattr("utils.stage1_checkpoint.write_success_marker", fail_resumable)
    with pytest.raises(RuntimeError, match="resumable success marker"):
        finalize_stage1_checkpoint(
            directory,
            completed_step=75,
            resumable=True,
            collectives=_fake_collectives(),
        )
    assert (directory / "checkpoint_manifest.json").is_file()
    assert not (directory / "_SUCCESS").exists()
    assert not (directory / "_RESUMABLE_SUCCESS").exists()
