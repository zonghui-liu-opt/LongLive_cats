import json
import random

import numpy as np
import pytest
import torch

import utils.stage2_checkpoint as stage2_checkpoint_module
from utils.lora_utils import LoraTensorSpec
from utils.stage1_io import canonical_json_sha256
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
    save_stage2_checkpoint,
    save_stage2_checkpoint_from_payloads,
    validate_stage2_checkpoint,
)


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
        "param_groups": [{"params": names, "lr": 1e-6}],
    }


def _sampler_state(g):
    return {
        "schema": "longlive_stage2_sampler_streams/v2",
        "fake_score": {"role": "fake_score", "completed_batches": 5 * g},
        "generator": {"role": "generator", "completed_batches": g},
    }


def _trainer(g):
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
        next_attempt_index=6 * g,
        contract_hash="a" * 64,
        launch_hash="b" * 64,
        cache_audit_launch_hash="7" * 64,
    )


def _rank_states(g, *, ema_initialized=None):
    if ema_initialized is None:
        ema_initialized = g >= 40
    ema = {}
    rng = {}
    for rank in range(8):
        ema[rank] = {
            "schema_version": 2,
            "initialized": ema_initialized,
            "last_completed_step": g,
            "rank": rank,
            "world_size": 8,
            "shadow": ({"x": torch.ones(1)} if ema_initialized else {}),
        }
        dedicated = {"score": torch.Generator().manual_seed(10 + rank)}
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


def _save(root, g, *, fail_at=None):
    ema_states, rng_states = _rank_states(g)
    return save_stage2_checkpoint_from_payloads(
        root,
        trainer_state=_trainer(g),
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
        provenance={
            "git_commit": "c" * 40,
            "generator_base_sha256": "d" * 64,
            "real_score_base_sha256": "e" * 64,
            "fake_score_base_sha256": "e" * 64,
            "cache_manifest_sha256": "f" * 64,
            "negative_manifest_sha256": "1" * 64,
            "negative_prompt_sha256": "2" * 64,
            "negative_embedding_sha256": "3" * 64,
            "negative_padding_sha256": "4" * 64,
            "generator_role_schema_sha256": "5" * 64,
            "fake_score_role_schema_sha256": "6" * 64,
        },
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
    generator_optimizer = torch.optim.AdamW(generator_module.parameters())
    fake_score_optimizer = torch.optim.AdamW(fake_score_module.parameters())

    class FakeEma:
        def state_dict(self):
            return ema_states[0]

    gathered_roles = []

    def gather_adapter(module, *, expected_schema, role, **kwargs):
        gathered_roles.append((module, role))
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
        object_gather_list[:] = [
            (rank, ema_states[rank], rng_states[rank]) for rank in range(8)
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
        resolved_config={"fixture": True},
        generator_module=generator_module,
        fake_score_module=fake_score_module,
        generator_optimizer=generator_optimizer,
        fake_score_optimizer=fake_score_optimizer,
        generator_ema=FakeEma(),
        generator_schema=GENERATOR_SCHEMA,
        fake_score_schema=FAKE_SCHEMA,
        dedicated_generators={"score": torch.Generator().manual_seed(88)},
        rank0_control_generators={
            "generator_exit": torch.Generator().manual_seed(89),
            "fake_score_exit": torch.Generator().manual_seed(90),
            "dfd_branch": torch.Generator().manual_seed(91),
        },
        provenance={"git_commit": "c" * 40},
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
    validate_stage2_checkpoint(directory, expected_contract_hash="a" * 64)


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


def test_marker_failure_leaves_uncommitted_final_and_discovery_fails_closed(tmp_path):
    with pytest.raises(OSError, match="before_success_marker"):
        _save(tmp_path, 10, fail_at="before_success_marker")
    directory = checkpoint_directory(tmp_path, 10)
    assert directory.is_dir()
    assert not (directory / "_SUCCESS").exists()
    with pytest.raises(RuntimeError, match="incomplete uncommitted"):
        find_latest_stage2_checkpoint(tmp_path)


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
        provenance={"git_commit": "c" * 40},
        topology=_topology(),
        io_include_cuda=False,
    )
    with pytest.raises(RuntimeError, match="EMA initialization"):
        save_stage2_checkpoint_from_payloads(**kwargs)
    assert not checkpoint_directory(tmp_path, 10).exists()
