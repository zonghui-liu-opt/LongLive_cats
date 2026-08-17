import copy
import json
import random
from functools import lru_cache

import numpy as np
import pytest
import torch

import utils.stage2_checkpoint as stage2_checkpoint_module
from utils.lora_utils import LoraTensorSpec
from utils.stage1_io import canonical_json_sha256, sha256_file
from utils.stage2_checkpoint import (
    STAGE2_CHECKPOINT_MILESTONES,
    Stage2CollectiveOps,
    apply_stage2_checkpoint_retention,
    build_stage2_trainer_state,
    capture_stage2_rng_state,
    checkpoint_directory,
    find_latest_stage2_checkpoint,
    load_stage2_checkpoint,
    load_stage2_checkpoint_collective,
    load_stage2_generator_ema_checkpoint,
    save_stage2_checkpoint,
    save_stage2_checkpoint_from_payloads,
    validate_stage2_checkpoint,
)
from utils.stage2_sampler import build_stage2_role_samplers


def _schema(prefix: str):
    return {
        f"{prefix}.lora_A.weight": LoraTensorSpec(
            f"{prefix}.lora_A.default.weight", (2, 3), torch.float32
        ),
        f"{prefix}.lora_B.weight": LoraTensorSpec(
            f"{prefix}.lora_B.default.weight", (4, 2), torch.float32
        ),
    }


GENERATOR_SCHEMA = _schema("generator")
FAKE_SCHEMA = _schema("fake_score")


def _adapter(schema, value):
    return {
        key: torch.full(spec.global_shape, value, dtype=torch.float32)
        for key, spec in schema.items()
    }


def _optimizer(schema, step):
    names = [spec.raw_parameter_name for spec in schema.values()]
    role = "generator" if schema is GENERATOR_SCHEMA else "fake_score"
    return {
        "state": {
            name: {
                "step": torch.tensor(float(step)),
                "exp_avg": torch.zeros(
                    next(
                        spec.global_shape
                        for spec in schema.values()
                        if spec.raw_parameter_name == name
                    ),
                    dtype=torch.float32,
                ),
                "exp_avg_sq": torch.ones(
                    next(
                        spec.global_shape
                        for spec in schema.values()
                        if spec.raw_parameter_name == name
                    ),
                    dtype=torch.float32,
                ),
            }
            for name in names
        },
        "param_groups": [
            {
                "params": names,
                "lr": 2e-6 if role == "generator" else 4e-7,
                "betas": (0.0, 0.999),
                "eps": 1e-8,
                "weight_decay": 0.0,
            }
        ],
    }


@lru_cache(maxsize=None)
def _cached_sampler_state(g):
    actions = ("a", "b", "c")
    streams = build_stage2_role_samplers(
        [action for action in actions for _ in range(200)],
        [(30, 52)] * 600,
        action_order=actions,
        base_seed=17,
    )
    for _ in range(5 * g):
        streams.fake_score.next_global_batch()
    for _ in range(g):
        streams.generator.next_global_batch()
    return streams.state_dict()


def _sampler_state(g):
    return copy.deepcopy(_cached_sampler_state(g))


def _loader_generators(rank: int) -> dict[str, torch.Generator]:
    seeds = {
        "generator_loader": 1 if rank == 0 else 1_000 + rank,
        "fake_score_loader": 2 if rank == 0 else 2_000 + rank,
        "generator_rollout": 3_000 + rank,
        "fake_score_rollout": 4_000 + rank,
        "generator_timestep": 5_000 + rank,
        "fake_score_timestep": 6_000 + rank,
        "generator_noise": 7_000 + rank,
        "fake_score_noise": 8_000 + rank,
    }
    return {
        name: torch.Generator(device="cpu").manual_seed(seed)
        for name, seed in seeds.items()
    }


def _provenance(*, parent: str | None = None, smoke_probe=None) -> dict:
    return {
        "assets": {
            "generator": {"checkpoint_sha256": "d" * 64},
            "real_score": {"checkpoint_sha256": "e" * 64},
            "fake_score": {"checkpoint_sha256": "e" * 64},
        },
        "data": {
            "stage2_manifest_sha256": "f" * 64,
            "source_manifest_sha256": "1" * 64,
            "negative_manifest_sha256": "2" * 64,
            "negative_artifact_sha256": "3" * 64,
        },
        "lineage": {
            "parent_checkpoint": parent,
            "parent_checkpoint_manifest_sha256": (None if parent is None else "4" * 64),
        },
        "smoke_probe": smoke_probe,
    }


def _metrics_snapshot(g: int, *, run_id: str | None = None) -> bytes:
    owner = run_id or f"run-{g}"
    records = [
        {
            "record_type": "run_start",
            "run_id": owner,
            "parent_run_id": None,
        },
        *(
            {
                "record_type": "train_step",
                "run_id": owner,
                "attempt_index": logical,
                "logical_substep_id": logical,
            }
            for logical in range(6 * g)
        ),
    ]
    return b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        for record in records
    )


def _trainer(g, *, phase_b_mode="dmd_dfd"):
    return build_stage2_trainer_state(
        completed_generator_updates=g,
        completed_fake_updates=5 * g,
        successful_attempts={"generator": g, "fake_score": 5 * g},
        nonfinite_attempts={"generator": 0, "fake_score": 0},
        sampler_state=_sampler_state(g),
        dataloader_generator_states={
            "generator": torch.Generator().manual_seed(1).get_state(),
            "fake_score": torch.Generator().manual_seed(2).get_state(),
        },
        run_id=f"run-{g}",
        next_attempt_index=6 * g + 1,
        contract_hash="a" * 64,
        launch_hash="b" * 64,
        cache_audit_launch_hash="7" * 64,
        phase_b_mode=phase_b_mode,
    )


def _rank_states(g, *, ema_initialized=None):
    if ema_initialized is None:
        ema_initialized = g >= 40
    ema = {}
    rng = {}
    parameter_shapes = {
        spec.raw_parameter_name: tuple(spec.global_shape)
        for spec in GENERATOR_SCHEMA.values()
    }
    for rank in range(8):
        local_shapes = {}
        for name, shape in parameter_shapes.items():
            chunk = (shape[0] + 7) // 8
            local_first = max(0, min(chunk, shape[0] - rank * chunk))
            local_shapes[name] = (local_first, *shape[1:])
        shard_metadata = {
            name: {
                "kind": "dtensor",
                "global_shape": parameter_shapes[name],
                "local_shape": local_shapes[name],
                "dtype": "torch.float32",
                "mesh_device_type": "cuda",
                "mesh_dim_names": ("shard",),
                "mesh_shape": (8,),
                "rank_layout": tuple((mesh_rank,) for mesh_rank in range(8)),
                "coordinate": (rank,),
                "placements": ("S(0)",),
            }
            for name in parameter_shapes
        }
        ema[rank] = {
            "schema_version": 2,
            "decay": 0.99,
            "start_step": 40,
            "initialized": ema_initialized,
            "last_completed_step": g,
            "rank": rank,
            "world_size": 8,
            "topology": {
                "rank_layout": tuple(range(8)),
                "mesh_dim_names": ("shard",),
            },
            "local_shapes": local_shapes,
            "global_shapes": parameter_shapes,
            "shard_metadata": shard_metadata,
            "shadow": (
                {
                    name: torch.ones(local_shapes[name], dtype=torch.float32)
                    for name in parameter_shapes
                }
                if ema_initialized
                else {}
            ),
        }
        dedicated = _loader_generators(rank)
        controls = (
            {
                "generator_exit": torch.Generator().manual_seed(100),
                "fake_score_exit": torch.Generator().manual_seed(101),
                "dfd_branch": torch.Generator().manual_seed(102),
            }
            if rank == 0
            else None
        )
        rng[rank] = capture_stage2_rng_state(
            rank=rank,
            dedicated_generators=dedicated,
            rank0_control_generators=controls,
            include_cuda=False,
        )
    return ema, rng


def _topology():
    return {
        "world_size": 8,
        "nodes": 1,
        "fsdp_backend": "fsdp2",
        "sharding_strategy": "FULL_SHARD",
        "mesh_shape": [8],
        "mesh_dim_names": ["shard"],
        "rank_layout": list(range(8)),
        "microbatch_size_per_device": 2,
        "gradient_accumulation_steps": 4,
        "global_batch_size": 64,
    }


def _save(
    root,
    g,
    *,
    fail_at=None,
    provenance=None,
    phase_b_mode="dmd_dfd",
    metrics_snapshot=None,
):
    ema_states, rng_states = _rank_states(g)
    return save_stage2_checkpoint_from_payloads(
        root,
        trainer_state=_trainer(g, phase_b_mode=phase_b_mode),
        metrics_lineage_snapshot=(
            _metrics_snapshot(g) if metrics_snapshot is None else metrics_snapshot
        ),
        generator_raw=_adapter(GENERATOR_SCHEMA, float(g)),
        fake_score_raw=_adapter(FAKE_SCHEMA, float(g + 1)),
        generator_ema=(_adapter(GENERATOR_SCHEMA, float(g + 2)) if g >= 40 else None),
        generator_schema=GENERATOR_SCHEMA,
        fake_score_schema=FAKE_SCHEMA,
        generator_optimizer_state=_optimizer(GENERATOR_SCHEMA, g),
        fake_score_optimizer_state=_optimizer(FAKE_SCHEMA, 5 * g),
        rank_ema_states=ema_states,
        rank_rng_states=rng_states,
        resolved_config={"contract": "fixture", "g": g},
        provenance=_provenance() if provenance is None else provenance,
        topology=_topology(),
        io_include_cuda=False,
        failure_injector=(
            (
                lambda stage: (
                    (_ for _ in ()).throw(OSError(stage)) if stage == fail_at else None
                )
            )
            if fail_at
            else None
        ),
    )


def test_atomic_checkpoint_roundtrip_and_exact_file_mapping(tmp_path):
    directory = _save(tmp_path, 40)
    assert directory == checkpoint_directory(tmp_path, 40)
    assert (directory / "_SUCCESS").is_file()
    manifest = validate_stage2_checkpoint(
        directory,
        expected_contract_hash="a" * 64,
        expected_topology=_topology(),
        expected_generator_parameter_names=tuple(
            spec.raw_parameter_name for spec in GENERATOR_SCHEMA.values()
        ),
        expected_fake_score_parameter_names=tuple(
            spec.raw_parameter_name for spec in FAKE_SCHEMA.values()
        ),
    )
    assert manifest["completed_generator_updates"] == 40
    assert manifest["completed_fake_updates"] == 200
    assert manifest["config"] == {
        "contract_hash": "a" * 64,
        "launch_hash": "b" * 64,
        "cache_audit_launch_hash": "7" * 64,
    }
    assert manifest["ema"] == {
        "initialized": True,
        "canonical_artifact": "generator_ema.safetensors",
        "last_completed_generator_update": 40,
    }
    expected_files = {
        "generator_raw.safetensors",
        "fake_score_raw.safetensors",
        "generator_ema.safetensors",
        "optimizer_generator.pt",
        "optimizer_fake_score.pt",
        "trainer_state.pt",
        "metrics_lineage.jsonl",
        "resolved_config.json",
        "provenance.json",
        *(f"ema_state_rank{rank:05d}.pt" for rank in range(8)),
        *(f"rng_state_rank{rank:05d}.pt" for rank in range(8)),
    }
    assert {entry["name"] for entry in manifest["files"]} == expected_files
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    assert manifest["manifest_sha256"] == canonical_json_sha256(body)

    payload = load_stage2_checkpoint(
        directory,
        expected_contract_hash="a" * 64,
        expected_world_size=8,
        expected_topology=_topology(),
    )
    assert payload.directory == directory
    assert payload.trainer_state["next_substep"] == "F1"
    assert payload.trainer_state["cache_audit_launch_hash"] == "7" * 64
    assert payload.generator_ema is not None
    assert torch.equal(
        payload.generator_raw["generator.lora_A.weight"],
        torch.full((2, 3), 40.0),
    )
    assert set(payload.rank_rng_states) == set(range(8))


def test_next_f1_probe_roundtrips_inside_manifest_bound_provenance(tmp_path):
    probe = {
        "schema": "longlive_stage2_next_f1_probe/v1",
        "state": {
            "next_substep": "F1",
            "current_role": "fake_score",
            "completed_fake_updates": 200,
            "completed_generator_updates": 40,
            "completed_cycles": 40,
            "successful_attempts": 240,
            "nonfinite_attempts": 0,
            "nonfinite_attempts_by_role": {"generator": 0, "fake_score": 0},
        },
        "sampler": {
            "state_sha256": "5" * 64,
            "completed_batches": 200,
            "stream_epoch": 20,
            "batch_cursor": 0,
            "extra_slot_cursor": 2,
            "global_batch_ids": list(range(64)),
        },
        "exit_schedule": [2, 0, 3, 1],
        "rank_payloads": [
            {
                "rank": rank,
                "attempt_state_sha256": f"{rank:x}" * 64,
                "stream_state_sha256": {
                    "fake_score_loader": "1" * 64,
                    "fake_score_exit": "2" * 64,
                    "fake_score_rollout": "3" * 64,
                    "fake_score_timestep": "4" * 64,
                    "fake_score_noise": "5" * 64,
                },
                "microbatches": [],
            }
            for rank in range(8)
        ],
    }
    smoke_probe = {
        "smoke_mode": "C0",
        "status": "PASS",
        "parent_next_f1_probe_consumed": False,
        "next_f1_probe": probe,
    }
    directory = _save(
        tmp_path,
        40,
        provenance=_provenance(smoke_probe=smoke_probe),
    )

    payload = load_stage2_checkpoint(
        directory,
        expected_contract_hash="a" * 64,
        expected_world_size=8,
        expected_topology=_topology(),
    )

    assert payload.provenance["smoke_probe"] == smoke_probe
    provenance_path = directory / "provenance.json"
    tampered = json.loads(provenance_path.read_text(encoding="utf-8"))
    tampered["smoke_probe"]["next_f1_probe"]["sampler"]["batch_cursor"] = 1
    provenance_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash/size|self binding"):
        validate_stage2_checkpoint(directory)


def test_metrics_lineage_snapshot_is_semantic_and_manifest_bound(tmp_path):
    invalid = b'{"record_type":"run_start","run_id":"run-10"}\n'
    with pytest.raises(RuntimeError, match="exact committed logical prefix"):
        _save(tmp_path / "invalid", 10, metrics_snapshot=invalid)
    assert not checkpoint_directory(tmp_path / "invalid", 10).exists()

    directory = _save(tmp_path / "valid", 10)
    (directory / "metrics_lineage.jsonl").write_bytes(
        _metrics_snapshot(10) + b'{"tampered":true}\n'
    )
    with pytest.raises(RuntimeError, match="hash/size"):
        validate_stage2_checkpoint(directory)


def test_generator_ema_inference_gate_never_deserializes_training_pickles(
    tmp_path, monkeypatch
):
    directory = _save(tmp_path, 40)

    def forbidden_torch_load(*args, **kwargs):
        raise AssertionError("inference EMA gate must not call torch.load")

    monkeypatch.setattr(torch, "load", forbidden_torch_load)
    payload = load_stage2_generator_ema_checkpoint(
        directory,
        expected_contract_hash="a" * 64,
        expected_launch_hash="b" * 64,
        expected_topology=_topology(),
        expected_resolved_config={"contract": "fixture", "g": 40},
        expected_generator_schema=GENERATOR_SCHEMA,
    )
    assert payload.directory == directory
    assert payload.manifest["ema"]["initialized"] is True
    assert payload.resolved_config == {"contract": "fixture", "g": 40}
    assert payload.provenance["schema"] == "longlive_stage2_checkpoint_provenance"
    assert set(payload.provenance["code_version"]) == {"stage2_source_sha256"}
    assert torch.equal(
        payload.generator_ema["generator.lora_A.weight"],
        torch.full((2, 3), 42.0),
    )


def test_generator_ema_inference_gate_hash_scans_opaque_training_pickles(
    tmp_path, monkeypatch
):
    directory = _save(tmp_path, 40)
    (directory / "optimizer_fake_score.pt").write_bytes(b"tampered")

    def forbidden_torch_load(*args, **kwargs):
        raise AssertionError("inference EMA gate must not call torch.load")

    monkeypatch.setattr(torch, "load", forbidden_torch_load)
    with pytest.raises(RuntimeError, match="hash/size mismatch"):
        load_stage2_generator_ema_checkpoint(
            directory,
            expected_contract_hash="a" * 64,
            expected_launch_hash="b" * 64,
        )


def test_generator_ema_inference_gate_rejects_swap_after_manifest_scan(
    tmp_path, monkeypatch
):
    directory = _save(tmp_path, 40)
    ema_path = directory / "generator_ema.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    from safetensors.torch import save_file

    save_file(_adapter(GENERATOR_SCHEMA, 99.0), str(replacement))
    replacement_bytes = replacement.read_bytes()
    validate_manifest_and_files = stage2_checkpoint_module._validate_manifest_and_files

    def validate_then_swap(*args, **kwargs):
        manifest = validate_manifest_and_files(*args, **kwargs)
        ema_path.write_bytes(replacement_bytes)
        return manifest

    monkeypatch.setattr(
        stage2_checkpoint_module,
        "_validate_manifest_and_files",
        validate_then_swap,
    )
    with pytest.raises(RuntimeError, match="Generator EMA snapshot hash/size mismatch"):
        load_stage2_generator_ema_checkpoint(
            directory,
            expected_contract_hash="a" * 64,
            expected_launch_hash="b" * 64,
            expected_generator_schema=GENERATOR_SCHEMA,
        )


def test_generator_ema_inference_deserializes_the_authenticated_snapshot(
    tmp_path, monkeypatch
):
    directory = _save(tmp_path, 40)
    ema_path = directory / "generator_ema.safetensors"
    replacement = tmp_path / "replacement.safetensors"
    import safetensors.torch

    safetensors.torch.save_file(_adapter(GENERATOR_SCHEMA, 99.0), str(replacement))
    replacement_bytes = replacement.read_bytes()
    deserialize_snapshot = safetensors.torch.load
    calls = 0

    def swap_live_path_then_deserialize(snapshot):
        nonlocal calls
        calls += 1
        ema_path.write_bytes(replacement_bytes)
        return deserialize_snapshot(snapshot)

    monkeypatch.setattr(
        safetensors.torch,
        "load",
        swap_live_path_then_deserialize,
    )
    payload = load_stage2_generator_ema_checkpoint(
        directory,
        expected_contract_hash="a" * 64,
        expected_launch_hash="b" * 64,
        expected_generator_schema=GENERATOR_SCHEMA,
    )
    assert calls == 1
    assert torch.equal(
        payload.generator_ema["generator.lora_A.weight"],
        torch.full((2, 3), 42.0),
    )
    assert torch.equal(
        safetensors.torch.load_file(str(ema_path))["generator.lora_A.weight"],
        torch.full((2, 3), 99.0),
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_contract_hash", "0" * 64, "contract hash"),
        ("expected_launch_hash", "0" * 64, "launch hash"),
        ("expected_resolved_config", {"wrong": True}, "resolved config"),
    ],
)
def test_generator_ema_inference_gate_rejects_any_resolved_launch_drift(
    tmp_path, field, value, message
):
    directory = _save(tmp_path, 40)
    kwargs = {
        "expected_contract_hash": "a" * 64,
        "expected_launch_hash": "b" * 64,
        "expected_resolved_config": {"contract": "fixture", "g": 40},
        field: value,
    }
    with pytest.raises(RuntimeError, match=message):
        load_stage2_generator_ema_checkpoint(directory, **kwargs)


def test_generator_ema_inference_gate_requires_initialized_ema(tmp_path):
    directory = _save(tmp_path, 10)
    with pytest.raises(RuntimeError, match="initialized G>=40"):
        load_stage2_generator_ema_checkpoint(
            directory,
            expected_contract_hash="a" * 64,
            expected_launch_hash="b" * 64,
        )


def test_generator_ema_inference_gate_rejects_rehashed_invalid_provenance(tmp_path):
    directory = _save(tmp_path, 40)
    provenance_path = directory / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["unexpected"] = True
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    manifest_path = directory / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance_sha256"] = canonical_json_sha256(provenance)
    entry = next(
        item for item in manifest["files"] if item["name"] == "provenance.json"
    )
    entry["size"] = provenance_path.stat().st_size
    entry["sha256"] = sha256_file(provenance_path)
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = canonical_json_sha256(body)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance.*keys mismatch"):
        load_stage2_generator_ema_checkpoint(
            directory,
            expected_contract_hash="a" * 64,
            expected_launch_hash="b" * 64,
        )


def test_child_checkpoint_lineage_is_explicit_and_strict(tmp_path):
    lineage = _provenance(parent="/checkpoints/checkpoint_stage2_g000240")
    directory = _save(tmp_path, 250, provenance=lineage)
    payload = load_stage2_generator_ema_checkpoint(
        directory,
        expected_contract_hash="a" * 64,
        expected_launch_hash="b" * 64,
    )
    assert payload.provenance["lineage"] == lineage["lineage"]

    invalid = _provenance()
    invalid["lineage"]["parent_checkpoint_manifest_sha256"] = "4" * 64
    with pytest.raises(ValueError, match="cold Stage-2 lineage"):
        _save(tmp_path / "invalid", 40, provenance=invalid)
    assert not checkpoint_directory(tmp_path / "invalid", 40).exists()


def test_a24_can_fork_b0_b1_but_post_a24_resume_is_arm_exact(tmp_path):
    parent_a24 = _save(tmp_path / "parent", 240, phase_b_mode="dmd_dfd")
    # A24 itself contains no arm-specific update, so both matched children may
    # consume the same byte-identical parent.
    validate_stage2_checkpoint(parent_a24, expected_phase_b_mode="dmd_only")
    validate_stage2_checkpoint(parent_a24, expected_phase_b_mode="dmd_dfd")

    b1 = _save(tmp_path / "b1", 250, phase_b_mode="dmd_dfd")
    with pytest.raises(RuntimeError, match="different Phase-B arm"):
        validate_stage2_checkpoint(b1, expected_phase_b_mode="dmd_only")
    b0 = _save(tmp_path / "b0", 250, phase_b_mode="dmd_only")
    validate_stage2_checkpoint(b0, expected_phase_b_mode="dmd_only")


def test_collective_load_rank0_returns_full_optimizer_and_local_rank_state(tmp_path):
    directory = _save(tmp_path, 40)
    operations = Stage2CollectiveOps(
        get_rank=lambda: 0,
        get_world_size=lambda: 8,
        barrier=lambda: None,
        consensus=lambda success: success,
        broadcast_object=lambda value, src: value,
    )
    payload = load_stage2_checkpoint_collective(
        directory,
        expected_contract_hash="a" * 64,
        expected_phase_b_mode="dmd_dfd",
        expected_world_size=8,
        expected_topology=_topology(),
        collectives=operations,
        io_include_cuda=False,
    )
    assert payload.generator_optimizer_state_rank0 is not None
    assert payload.fake_score_optimizer_state_rank0 is not None
    assert payload.generator_ema_rank0 is not None
    assert payload.local_ema_state["rank"] == 0
    assert payload.local_rng_state["rank"] == 0
    assert payload.trainer_state["cache_audit_launch_hash"] == "7" * 64


def test_symlinked_checkpoint_and_root_are_rejected_before_resolution(tmp_path):
    real_root = tmp_path / "real"
    target = _save(real_root, 10)
    link_root = tmp_path / "link-root"
    link_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(RuntimeError, match="regular directory"):
        find_latest_stage2_checkpoint(link_root)

    link_checkpoint = tmp_path / "checkpoint_stage2_g000010"
    link_checkpoint.symlink_to(target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="regular directory"):
        validate_stage2_checkpoint(link_checkpoint)


def test_high_level_save_collects_two_roles_and_all_rank_local_state(
    tmp_path, monkeypatch
):
    ema_states, rng_states = _rank_states(10)
    generator_module = torch.nn.Linear(1, 1)
    fake_score_module = torch.nn.Linear(1, 1)
    generator_optimizer = torch.optim.AdamW(
        generator_module.parameters(),
        lr=2e-6,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    fake_score_optimizer = torch.optim.AdamW(
        fake_score_module.parameters(),
        lr=4e-7,
        betas=(0.0, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    dedicated_generators = _loader_generators(0)
    control_generators = {
        "generator_exit": torch.Generator().manual_seed(89),
        "fake_score_exit": torch.Generator().manual_seed(90),
        "dfd_branch": torch.Generator().manual_seed(91),
    }
    entry_torch_state = torch.get_rng_state().clone()
    entry_rollout_state = dedicated_generators["generator_rollout"].get_state().clone()

    class FakeEma:
        def state_dict(self):
            return ema_states[0]

    gathered_roles = []

    def gather_adapter(module, *, expected_schema, role, **kwargs):
        gathered_roles.append((module, role))
        # Deliberately consume both default and dedicated RNG after save entry.
        # The persisted rank payload and post-save runtime must still reflect
        # the entry snapshot, not this checkpoint-I/O work.
        torch.rand(3)
        torch.rand(3, generator=dedicated_generators["generator_rollout"])
        value = 1.0 if role == "generator" else 2.0
        return _adapter(expected_schema, value)

    def gather_optimizer(
        module,
        optimizer,
        *,
        role,
        expected_schema,
        expected_completed_updates,
        **kwargs,
    ):
        del module, optimizer
        assert expected_completed_updates == (10 if role == "generator" else 50)
        return _optimizer(expected_schema, expected_completed_updates)

    monkeypatch.setattr(
        stage2_checkpoint_module,
        "gather_stage2_lora_state_dict",
        gather_adapter,
    )
    monkeypatch.setattr(
        stage2_checkpoint_module,
        "gather_stage2_optimizer_state",
        gather_optimizer,
    )

    def gather_rank_state(local, *, object_gather_list, dst):
        assert dst == 0
        assert local[0] == 0
        assert object_gather_list is not None
        assert torch.equal(local[2]["general"]["torch_cpu"], entry_torch_state)
        assert torch.equal(
            local[2]["dedicated"]["generator_rollout"]["state"],
            entry_rollout_state,
        )
        object_gather_list[:] = [local] + [
            (rank, ema_states[rank], rng_states[rank]) for rank in range(1, 8)
        ]

    operations = Stage2CollectiveOps(
        get_rank=lambda: 0,
        get_world_size=lambda: 8,
        barrier=lambda: None,
        consensus=lambda success: success,
        broadcast_object=lambda value, src: value,
    )
    directory = save_stage2_checkpoint(
        tmp_path,
        trainer_state=_trainer(10),
        metrics_lineage_snapshot=_metrics_snapshot(10),
        resolved_config={"fixture": True},
        generator_module=generator_module,
        fake_score_module=fake_score_module,
        generator_optimizer=generator_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        generator_ema=FakeEma(),
        generator_schema=GENERATOR_SCHEMA,
        fake_score_schema=FAKE_SCHEMA,
        dedicated_generators=dedicated_generators,
        rank0_control_generators=control_generators,
        provenance=_provenance(),
        topology=_topology(),
        collectives=operations,
        gather_rank_object_fn=gather_rank_state,
        io_include_cuda=False,
        apply_retention=False,
    )
    assert directory == checkpoint_directory(tmp_path, 10).resolve()
    assert gathered_roles == [
        (generator_module, "generator"),
        (fake_score_module, "fake_score"),
    ]
    assert torch.equal(torch.get_rng_state(), entry_torch_state)
    assert torch.equal(
        dedicated_generators["generator_rollout"].get_state(), entry_rollout_state
    )
    validate_stage2_checkpoint(directory, expected_contract_hash="a" * 64)


@pytest.mark.parametrize("corruption", ["pending_gradient", "loader_rng_duplicate"])
def test_high_level_save_rejects_nonquiescent_live_boundary(tmp_path, corruption):
    generator_module = torch.nn.Linear(1, 1)
    fake_score_module = torch.nn.Linear(1, 1)
    if corruption == "pending_gradient":
        generator_module.weight.grad = torch.ones_like(generator_module.weight)
    dedicated = _loader_generators(0)
    if corruption == "loader_rng_duplicate":
        torch.rand(1, generator=dedicated["generator_loader"])
    controls = {
        "generator_exit": torch.Generator().manual_seed(89),
        "fake_score_exit": torch.Generator().manual_seed(90),
        "dfd_branch": torch.Generator().manual_seed(91),
    }
    operations = Stage2CollectiveOps(
        get_rank=lambda: 0,
        get_world_size=lambda: 8,
        barrier=lambda: None,
        consensus=lambda success: success,
        broadcast_object=lambda value, src: value,
    )
    with pytest.raises(RuntimeError, match="live cycle-boundary audit"):
        save_stage2_checkpoint(
            tmp_path,
            trainer_state=_trainer(10),
            metrics_lineage_snapshot=_metrics_snapshot(10),
            resolved_config={"fixture": True},
            generator_module=generator_module,
            fake_score_module=fake_score_module,
            generator_optimizer=torch.optim.AdamW(
                generator_module.parameters(),
                lr=2e-6,
                betas=(0.0, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            ),
            fake_score_optimizer=torch.optim.AdamW(
                fake_score_module.parameters(),
                lr=4e-7,
                betas=(0.0, 0.999),
                eps=1e-8,
                weight_decay=0.0,
            ),
            generator_ema=object(),
            generator_schema=GENERATOR_SCHEMA,
            fake_score_schema=FAKE_SCHEMA,
            dedicated_generators=dedicated,
            rank0_control_generators=controls,
            provenance=_provenance(),
            topology=_topology(),
            collectives=operations,
            io_include_cuda=False,
            apply_retention=False,
        )
    assert not checkpoint_directory(tmp_path, 10).exists()


def test_pre_ema_checkpoint_is_explicit_null_but_keeps_all_local_states(tmp_path):
    directory = _save(tmp_path, 10)
    manifest = validate_stage2_checkpoint(directory)
    assert manifest["ema"] == {
        "initialized": False,
        "canonical_artifact": None,
        "last_completed_generator_update": 10,
    }
    assert not (directory / "generator_ema.safetensors").exists()
    assert all(
        (directory / f"ema_state_rank{rank:05d}.pt").is_file() for rank in range(8)
    )


def test_failure_before_rename_cleans_hidden_temp_and_never_publishes(tmp_path):
    with pytest.raises(OSError, match="before_rename"):
        _save(tmp_path, 10, fail_at="before_rename")
    assert not checkpoint_directory(tmp_path, 10).exists()
    assert not list(tmp_path.glob(".checkpoint_stage2_g000010.*"))


def test_marker_failure_is_ignored_by_latest_and_same_g_can_be_republished(tmp_path):
    older = _save(tmp_path, 10)
    with pytest.raises(OSError, match="before_success_marker"):
        _save(tmp_path, 20, fail_at="before_success_marker")
    directory = checkpoint_directory(tmp_path, 20)
    assert directory.is_dir()
    assert not (directory / "_SUCCESS").exists()
    assert find_latest_stage2_checkpoint(tmp_path) == older
    with pytest.raises(RuntimeError, match="incomplete uncommitted"):
        validate_stage2_checkpoint(directory)

    replacement = _save(tmp_path, 20)
    assert replacement == directory
    assert (replacement / "_SUCCESS").is_file()
    assert find_latest_stage2_checkpoint(tmp_path) == replacement
    assert not list(tmp_path.glob(".checkpoint_stage2_g000020.uncommitted.*"))


def test_newest_corruption_fails_closed_instead_of_resuming_older(tmp_path):
    older = _save(tmp_path, 10)
    newer = _save(tmp_path, 20)
    assert find_latest_stage2_checkpoint(tmp_path) == newer
    (newer / "optimizer_generator.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="hash/size"):
        find_latest_stage2_checkpoint(tmp_path)
    assert older.is_dir()


@pytest.mark.parametrize("extra", ["extra.bin", "nested/file.bin"])
def test_extra_or_nested_checkpoint_files_are_rejected(tmp_path, extra):
    directory = _save(tmp_path, 10)
    path = directory / extra
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"extra")
    with pytest.raises(RuntimeError, match="file set"):
        validate_stage2_checkpoint(directory)


def test_retention_keeps_latest_two_and_every_milestone_full(tmp_path):
    steps = (10, 20, 30, 80, 90, 100, 120)
    for step in steps:
        _save(tmp_path, step)
    removed = apply_stage2_checkpoint_retention(tmp_path, keep_last=2)
    assert removed == [10, 20, 30, 90]
    assert {
        step for step in steps if checkpoint_directory(tmp_path, step).exists()
    } == {
        80,
        100,
        120,
    }
    assert {80, 120}.issubset(STAGE2_CHECKPOINT_MILESTONES)
    for step in (80, 100, 120):
        validate_stage2_checkpoint(checkpoint_directory(tmp_path, step))


def test_retention_validates_every_candidate_before_deleting_anything(tmp_path):
    for step in (10, 20, 30):
        _save(tmp_path, step)
    (checkpoint_directory(tmp_path, 30) / "trainer_state.pt").write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="hash/size"):
        apply_stage2_checkpoint_retention(tmp_path, keep_last=2)
    assert all(checkpoint_directory(tmp_path, step).exists() for step in (10, 20, 30))


def test_checkpoint_io_preserves_python_numpy_and_torch_rng(tmp_path):
    random.seed(901)
    np.random.seed(902)
    torch.manual_seed(903)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()

    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    expected = (random.random(), np.random.rand(), torch.rand(3))

    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    _save(tmp_path, 10)
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])


def test_manifest_self_hash_and_topology_tampering_fail_before_torch_load(tmp_path):
    directory = _save(tmp_path, 10)
    path = directory / "checkpoint_manifest.json"
    manifest = json.loads(path.read_text())
    manifest["topology"]["world_size"] = 7
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="self hash"):
        validate_stage2_checkpoint(directory)


def test_rank_ema_state_cannot_claim_initialization_before_g40(tmp_path):
    ema_states, rng_states = _rank_states(10, ema_initialized=True)
    kwargs = dict(
        root=tmp_path,
        trainer_state=_trainer(10),
        generator_raw=_adapter(GENERATOR_SCHEMA, 1.0),
        fake_score_raw=_adapter(FAKE_SCHEMA, 2.0),
        generator_ema=None,
        generator_schema=GENERATOR_SCHEMA,
        fake_score_schema=FAKE_SCHEMA,
        generator_optimizer_state=_optimizer(GENERATOR_SCHEMA, 10),
        fake_score_optimizer_state=_optimizer(FAKE_SCHEMA, 50),
        rank_ema_states=ema_states,
        rank_rng_states=rng_states,
        resolved_config={"fixture": True},
        provenance=_provenance(),
        topology=_topology(),
        io_include_cuda=False,
    )
    with pytest.raises(RuntimeError, match="EMA initialization"):
        save_stage2_checkpoint_from_payloads(**kwargs)
    assert not checkpoint_directory(tmp_path, 10).exists()


@pytest.mark.parametrize("payload_kind", ["sampler", "ema"])
def test_publication_requires_full_sampler_and_ema_schema_before_marker(
    tmp_path, payload_kind
):
    trainer = _trainer(40)
    ema_states, rng_states = _rank_states(40)
    if payload_kind == "sampler":
        del trainer["sampler_state"]["generator"]["action_states"]
    else:
        del ema_states[3]["shard_metadata"]
    with pytest.raises((RuntimeError, ValueError), match="sampler|EMA"):
        save_stage2_checkpoint_from_payloads(
            tmp_path,
            trainer_state=trainer,
            generator_raw=_adapter(GENERATOR_SCHEMA, 1.0),
            fake_score_raw=_adapter(FAKE_SCHEMA, 2.0),
            generator_ema=_adapter(GENERATOR_SCHEMA, 3.0),
            generator_schema=GENERATOR_SCHEMA,
            fake_score_schema=FAKE_SCHEMA,
            generator_optimizer_state=_optimizer(GENERATOR_SCHEMA, 40),
            fake_score_optimizer_state=_optimizer(FAKE_SCHEMA, 200),
            rank_ema_states=ema_states,
            rank_rng_states=rng_states,
            resolved_config={"fixture": True},
            provenance=_provenance(),
            topology=_topology(),
            io_include_cuda=False,
        )
    assert not checkpoint_directory(tmp_path, 40).exists()
