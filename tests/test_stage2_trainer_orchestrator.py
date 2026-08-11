from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from trainer.stage2_distillation import Trainer
import trainer.stage2_distillation as trainer_module
import utils.distributed as distributed_utils
import utils.stage2_checkpoint as stage2_checkpoint
import utils.stage2_metrics as stage2_metrics
from utils.stage2_config import load_stage2_config
from utils.stage2_train_state import Stage2TrainingSchedule, Stage2TrainingState
import utils.stage2_train_state as train_state_module

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "train_i2v_stage2_600cats.yaml"


class _CapturedLogger:
    instances = []

    def __init__(self, path, **options):
        self.path = Path(path)
        self.options = options
        self.run_id = options["run_id"]
        self.records = []
        self.__class__.instances.append(self)

    def append(self, record_type, fields):
        self.records.append((record_type, dict(fields)))


def _bare_trainer(**attributes):
    trainer = Trainer.__new__(Trainer)
    for name, value in attributes.items():
        setattr(trainer, name, value)
    return trainer


@pytest.mark.parametrize(
    ("smoke_mode", "no_save", "auto_resume", "expected_dry_run"),
    [
        pytest.param(None, False, True, False, id="formal"),
        pytest.param("C0", False, False, True, id="C0-cold-save"),
        pytest.param("C1", False, True, True, id="C1-resume-save"),
        pytest.param("C2", True, True, True, id="C2-resume-discard"),
    ],
)
def test_smoke_options_encode_one_formal_or_c0_c1_c2_contract(
    tmp_path,
    smoke_mode,
    no_save,
    auto_resume,
    expected_dry_run,
):
    trainer = Trainer(
        load_stage2_config(CONFIG_PATH),
        output_dir=tmp_path / (smoke_mode or "formal"),
        no_save=no_save,
        auto_resume=auto_resume,
        smoke_mode=smoke_mode,
    )

    assert trainer.options.smoke_mode == smoke_mode
    assert trainer.options.no_save is no_save
    assert trainer.options.auto_resume is auto_resume
    assert trainer.options.dry_run is expected_dry_run


@pytest.mark.parametrize("smoke_mode", ["C1", "C2"])
def test_resume_smoke_requires_a_preceding_complete_checkpoint(
    monkeypatch, tmp_path, smoke_mode
):
    trainer = _bare_trainer(
        resolved=SimpleNamespace(resume_stage2_checkpoint=None),
        options=SimpleNamespace(
            output_dir=tmp_path,
            auto_resume=True,
            smoke_mode=smoke_mode,
        ),
    )
    monkeypatch.setattr(
        stage2_checkpoint,
        "find_latest_stage2_checkpoint",
        lambda _root: None,
    )

    with pytest.raises(RuntimeError, match=f"smoke {smoke_mode} requires"):
        trainer._discover_resume_checkpoint()


def test_c0_smoke_rejects_an_explicit_resume_checkpoint(tmp_path):
    trainer = _bare_trainer(
        resolved=SimpleNamespace(
            resume_stage2_checkpoint=str(tmp_path / "previous-checkpoint")
        ),
        options=SimpleNamespace(
            output_dir=tmp_path,
            auto_resume=False,
            smoke_mode="C0",
        ),
    )

    with pytest.raises(RuntimeError, match="C0 must start from the cold"):
        trainer._discover_resume_checkpoint()


@pytest.mark.parametrize(
    ("smoke_mode", "parent_probe", "error_match"),
    [
        pytest.param(None, None, None, id="formal-resumes-formal"),
        pytest.param(
            None,
            {"smoke_mode": "C1", "status": "PASS"},
            "refuses to resume a C0/C1 smoke checkpoint",
            id="formal-rejects-smoke",
        ),
        pytest.param(
            "C1",
            {"smoke_mode": "C0", "status": "PASS"},
            None,
            id="C1-resumes-C0",
        ),
        pytest.param(
            "C2",
            {"smoke_mode": "C1", "status": "PASS"},
            None,
            id="C2-resumes-C1",
        ),
    ],
)
def test_resume_provenance_separates_formal_and_smoke_lineages(
    smoke_mode,
    parent_probe,
    error_match,
):
    git_identity = {
        "commit": "a" * 40,
        "worktree_clean": True,
        "ignored_files_absent": True,
    }
    data_provenance = {
        "stage2_manifest_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64,
        "negative_manifest_sha256": "d" * 64,
        "negative_artifact_sha256": "e" * 64,
    }
    trainer = _bare_trainer(
        options=SimpleNamespace(smoke_mode=smoke_mode),
        dataset=SimpleNamespace(
            manifest={"manifest_sha256": data_provenance["stage2_manifest_sha256"]},
            source_manifest={
                "manifest_sha256": data_provenance["source_manifest_sha256"]
            },
            negative_conditioning={
                "manifest": {
                    "manifest_sha256": data_provenance["negative_manifest_sha256"],
                    "artifact": {"sha256": data_provenance["negative_artifact_sha256"]},
                }
            },
        ),
    )
    payload = SimpleNamespace(
        provenance={
            "git": git_identity,
            "data": data_provenance,
            "smoke_probe": parent_probe,
        }
    )

    context = (
        pytest.raises(RuntimeError, match=error_match)
        if error_match is not None
        else nullcontext()
    )
    with context:
        trainer._audit_resume_runtime_bindings(payload, git_identity)


def test_distributed_checkpoint_dataclass_drives_resume_state_sampler_loader_and_rng(
    monkeypatch, tmp_path
):
    resolved = load_stage2_config(CONFIG_PATH)
    schedule = Stage2TrainingSchedule.from_resolved_config(resolved)
    loader_states = {
        "generator": torch.Generator().manual_seed(101).get_state(),
        "fake_score": torch.Generator().manual_seed(202).get_state(),
    }
    sampler_state = {
        "schema": "longlive_stage2_sampler_streams/v2",
        "fake_score": {"role": "fake_score", "completed_batches": 10},
        "generator": {"role": "generator", "completed_batches": 2},
    }
    trainer_state = stage2_checkpoint.build_stage2_trainer_state(
        completed_generator_updates=2,
        completed_fake_updates=10,
        successful_attempts={"generator": 2, "fake_score": 10},
        nonfinite_attempts={"generator": 3, "fake_score": 4},
        sampler_state=sampler_state,
        dataloader_generator_states=loader_states,
        run_id="resume-run",
        next_attempt_index=19,
        contract_hash=resolved.contract_hash(),
        launch_hash=resolved.launch_hash(),
        cache_audit_launch_hash="c" * 64,
        phase_a_generator_updates=resolved.phase_a_generator_updates,
        phase_b_generator_updates=resolved.phase_b_generator_updates,
        phase_b_mode=resolved.phase_b_mode,
    )
    local_rng_state = {"exact": "rank-zero-rng-payload"}
    payload = stage2_checkpoint.Stage2DistributedCheckpointPayload(
        directory=tmp_path / "checkpoint_stage2_g000002",
        manifest={"manifest_sha256": "d" * 64},
        trainer_state=trainer_state,
        generator_raw=OrderedDict([("generator.lora", torch.ones(1))]),
        fake_score_raw=OrderedDict([("fake_score.lora", torch.zeros(1))]),
        generator_ema_rank0=OrderedDict([("generator.lora", torch.full((1,), 2.0))]),
        generator_optimizer_state_rank0={"state": "generator-optimizer"},
        fake_score_optimizer_state_rank0={"state": "fake-score-optimizer"},
        local_ema_state={"state": "local-ema"},
        local_rng_state=local_rng_state,
        resolved_config={"config_schema": "longlive_stage2_train/v1"},
        provenance={"source": "checkpoint"},
    )
    load_calls = []

    def load_collective(
        directory,
        *,
        expected_contract_hash,
        expected_world_size,
        expected_topology,
    ):
        load_calls.append(
            {
                "directory": directory,
                "expected_contract_hash": expected_contract_hash,
                "expected_world_size": expected_world_size,
                "expected_topology": expected_topology,
            }
        )
        return payload

    monkeypatch.setattr(
        stage2_checkpoint, "load_stage2_checkpoint_collective", load_collective
    )
    restored_sampler_states = []
    generator_loader = torch.Generator().manual_seed(301)
    fake_score_loader = torch.Generator().manual_seed(302)
    generator_rollout = torch.Generator().manual_seed(401)
    fake_score_rollout = torch.Generator().manual_seed(402)
    generator_exit = torch.Generator().manual_seed(501)
    fake_score_exit = torch.Generator().manual_seed(502)
    branch_rng = torch.Generator().manual_seed(503)
    exit_controls = {
        "generator": generator_exit,
        "fake_score": fake_score_exit,
    }
    trainer = _bare_trainer(
        resume_checkpoint=payload.directory,
        resolved=resolved,
        schedule=schedule,
        world_size=8,
        rank=0,
        is_main_process=True,
        samplers=SimpleNamespace(
            load_state_dict=lambda state: restored_sampler_states.append(state)
        ),
        dedicated_generators={
            "generator_rollout": generator_rollout,
            "fake_score_rollout": fake_score_rollout,
        },
        dataloader_generators={
            "generator": generator_loader,
            "fake_score": fake_score_loader,
        },
        exit_rng=SimpleNamespace(generator=lambda role: exit_controls[role]),
        branch_rng=branch_rng,
    )

    loaded = trainer._load_resume_before_roles()
    assert loaded is payload
    assert load_calls == [
        {
            "directory": payload.directory,
            "expected_contract_hash": resolved.contract_hash(),
            "expected_world_size": 8,
            "expected_topology": trainer._checkpoint_topology(),
        }
    ]

    trainer._restore_logical_state(loaded)
    assert trainer.state.completed_g == 2
    assert trainer.state.completed_f == 10
    assert trainer.state.successful_attempts == 12
    assert trainer.state.nonfinite_attempts == 7
    assert trainer.state.nonfinite_attempts_by_role == {
        "generator": 3,
        "fake_score": 4,
    }
    assert restored_sampler_states == [trainer_state["sampler_state"]]
    assert torch.equal(
        trainer_state["dataloader_generator_states"]["generator"],
        loader_states["generator"],
    )
    assert torch.equal(
        trainer_state["dataloader_generator_states"]["fake_score"],
        loader_states["fake_score"],
    )

    rng_calls = []

    def restore_rng(
        state,
        *,
        rank,
        dedicated_generators,
        rank0_control_generators,
        require_cuda_topology,
    ):
        rng_calls.append(
            {
                "state": state,
                "rank": rank,
                "dedicated_generators": dedicated_generators,
                "rank0_control_generators": rank0_control_generators,
                "require_cuda_topology": require_cuda_topology,
            }
        )

    monkeypatch.setattr(stage2_checkpoint, "restore_stage2_rng_state", restore_rng)
    trainer._restore_rng_last(loaded)

    assert len(rng_calls) == 1
    call = rng_calls[0]
    assert call["state"] is local_rng_state
    assert call["rank"] == 0
    assert call["require_cuda_topology"] is True
    assert call["dedicated_generators"] == {
        "generator_rollout": generator_rollout,
        "fake_score_rollout": fake_score_rollout,
        "generator_loader": generator_loader,
        "fake_score_loader": fake_score_loader,
    }
    assert call["rank0_control_generators"] == {
        "generator_exit": generator_exit,
        "fake_score_exit": fake_score_exit,
        "dfd_branch": branch_rng,
    }


def test_checkpoint_bridge_calls_current_builder_and_saver_contract(
    monkeypatch, tmp_path
):
    resolved = load_stage2_config(CONFIG_PATH)
    schedule = Stage2TrainingSchedule.from_resolved_config(resolved)
    state = Stage2TrainingState(
        completed_g=2,
        completed_f=10,
        cycle=2,
        next_substep="F1",
        successful_attempts=12,
        nonfinite_attempts=5,
        nonfinite_attempts_by_role={"generator": 2, "fake_score": 3},
    )
    state.validate(schedule)
    sampler_state = {
        "schema": "longlive_stage2_sampler_streams/v2",
        "fake_score": {"role": "fake_score", "completed_batches": 10},
        "generator": {"role": "generator", "completed_batches": 2},
    }
    generator_loader = torch.Generator().manual_seed(601)
    fake_score_loader = torch.Generator().manual_seed(602)
    generator_rollout = torch.Generator().manual_seed(603)
    fake_score_rollout = torch.Generator().manual_seed(604)
    generator_exit = torch.Generator().manual_seed(605)
    fake_score_exit = torch.Generator().manual_seed(606)
    branch_rng = torch.Generator().manual_seed(607)
    exit_controls = {
        "generator": generator_exit,
        "fake_score": fake_score_exit,
    }
    generator_module = object()
    fake_score_module = object()
    generator_optimizer = object()
    fake_score_optimizer = object()
    generator_schema = {"generator.lora": object()}
    fake_score_schema = {"fake_score.lora": object()}
    generator_ema = object()
    shard_group = object()
    provenance = {"provenance": "exact"}
    cache_audit_launch_hash = "e" * 64
    old_directory = tmp_path / "checkpoint_stage2_g000001"
    old_directory.mkdir()
    destination = tmp_path / "checkpoint_stage2_g000002"
    builder_calls = []
    saver_calls = []
    real_builder = stage2_checkpoint.build_stage2_trainer_state

    def capture_builder(
        *,
        completed_generator_updates,
        completed_fake_updates,
        successful_attempts,
        nonfinite_attempts,
        sampler_state,
        dataloader_generator_states,
        run_id,
        next_attempt_index,
        contract_hash,
        launch_hash,
        cache_audit_launch_hash,
        phase_a_generator_updates,
        phase_b_generator_updates,
        phase_b_mode,
    ):
        arguments = {
            "completed_generator_updates": completed_generator_updates,
            "completed_fake_updates": completed_fake_updates,
            "successful_attempts": successful_attempts,
            "nonfinite_attempts": nonfinite_attempts,
            "sampler_state": sampler_state,
            "dataloader_generator_states": dataloader_generator_states,
            "run_id": run_id,
            "next_attempt_index": next_attempt_index,
            "contract_hash": contract_hash,
            "launch_hash": launch_hash,
            "cache_audit_launch_hash": cache_audit_launch_hash,
            "phase_a_generator_updates": phase_a_generator_updates,
            "phase_b_generator_updates": phase_b_generator_updates,
            "phase_b_mode": phase_b_mode,
        }
        builder_calls.append(arguments)
        return real_builder(**arguments)

    def capture_saver(
        root,
        *,
        trainer_state,
        resolved_config,
        generator_module,
        fake_score_module,
        generator_optimizer,
        fake_score_optimizer,
        generator_ema,
        generator_schema,
        fake_score_schema,
        dedicated_generators,
        rank0_control_generators,
        provenance,
        topology,
        shard_group,
        keep_last,
    ):
        saver_calls.append(
            {
                "root": root,
                "trainer_state": trainer_state,
                "resolved_config": resolved_config,
                "generator_module": generator_module,
                "fake_score_module": fake_score_module,
                "generator_optimizer": generator_optimizer,
                "fake_score_optimizer": fake_score_optimizer,
                "generator_ema": generator_ema,
                "generator_schema": generator_schema,
                "fake_score_schema": fake_score_schema,
                "dedicated_generators": dedicated_generators,
                "rank0_control_generators": rank0_control_generators,
                "provenance": provenance,
                "topology": topology,
                "shard_group": shard_group,
                "keep_last": keep_last,
            }
        )
        old_directory.rmdir()
        destination.mkdir()
        (destination / "checkpoint_manifest.json").write_text(
            json.dumps({"manifest_sha256": "f" * 64}), encoding="utf-8"
        )
        (destination / "payload.bin").write_bytes(b"stage2-checkpoint")
        return destination

    monkeypatch.setattr(
        stage2_checkpoint, "build_stage2_trainer_state", capture_builder
    )
    monkeypatch.setattr(stage2_checkpoint, "save_stage2_checkpoint", capture_saver)
    trainer = _bare_trainer(
        state=state,
        schedule=schedule,
        resolved=resolved,
        options=SimpleNamespace(output_dir=tmp_path),
        logger=SimpleNamespace(run_id="checkpoint-run", next_attempt_index=41),
        samplers=SimpleNamespace(state_dict=lambda: sampler_state),
        dataloader_generators={
            "generator": generator_loader,
            "fake_score": fake_score_loader,
        },
        dedicated_generators={
            "generator_rollout": generator_rollout,
            "fake_score_rollout": fake_score_rollout,
        },
        exit_rng=SimpleNamespace(generator=lambda role: exit_controls[role]),
        branch_rng=branch_rng,
        is_main_process=True,
        rank=0,
        world_size=8,
        cache_audit_launch_hash=cache_audit_launch_hash,
        model=SimpleNamespace(
            generator=generator_module,
            fake_score=fake_score_module,
        ),
        optimizers={
            "generator": generator_optimizer,
            "fake_score": fake_score_optimizer,
        },
        generator_ema=generator_ema,
        lora_schemas={
            "generator": generator_schema,
            "fake_score": fake_score_schema,
        },
        mesh=SimpleNamespace(
            get_group=lambda name: (
                shard_group
                if name == "shard"
                else pytest.fail(f"unexpected mesh group {name}")
            )
        ),
        _runtime_rank0_checked=lambda _label, callback: callback(),
        _checkpoint_provenance=lambda identity: (
            provenance
            if identity == {"commit": "a" * 40}
            else pytest.fail("checkpoint provenance received the wrong Git identity")
        ),
    )

    event = trainer._save_checkpoint({"commit": "a" * 40})

    assert len(builder_calls) == 1
    builder = builder_calls[0]
    assert builder["completed_generator_updates"] == 2
    assert builder["completed_fake_updates"] == 10
    assert builder["successful_attempts"] == {"generator": 2, "fake_score": 10}
    assert builder["nonfinite_attempts"] == {"generator": 2, "fake_score": 3}
    assert builder["sampler_state"] is sampler_state
    assert builder["run_id"] == "checkpoint-run"
    assert builder["next_attempt_index"] == 42
    assert builder["contract_hash"] == resolved.contract_hash()
    assert builder["launch_hash"] == resolved.launch_hash()
    assert builder["cache_audit_launch_hash"] == cache_audit_launch_hash
    assert torch.equal(
        builder["dataloader_generator_states"]["generator"],
        generator_loader.get_state(),
    )
    assert torch.equal(
        builder["dataloader_generator_states"]["fake_score"],
        fake_score_loader.get_state(),
    )

    assert len(saver_calls) == 1
    saver = saver_calls[0]
    assert saver["root"] == tmp_path
    assert saver["trainer_state"]["cache_audit_launch_hash"] == cache_audit_launch_hash
    assert saver["resolved_config"] is resolved
    assert saver["generator_module"] is generator_module
    assert saver["fake_score_module"] is fake_score_module
    assert saver["generator_optimizer"] is generator_optimizer
    assert saver["fake_score_optimizer"] is fake_score_optimizer
    assert saver["generator_ema"] is generator_ema
    assert saver["generator_schema"] is generator_schema
    assert saver["fake_score_schema"] is fake_score_schema
    assert saver["dedicated_generators"] == {
        "generator_rollout": generator_rollout,
        "fake_score_rollout": fake_score_rollout,
        "generator_loader": generator_loader,
        "fake_score_loader": fake_score_loader,
    }
    assert saver["rank0_control_generators"] == {
        "generator_exit": generator_exit,
        "fake_score_exit": fake_score_exit,
        "dfd_branch": branch_rng,
    }
    assert saver["provenance"] is provenance
    assert saver["topology"] == trainer._checkpoint_topology()
    assert saver["shard_group"] is shard_group
    assert saver["keep_last"] == resolved.keep_last_resumable

    assert event["path"] == str(destination)
    assert event["manifest_sha256"] == "f" * 64
    assert event["total_bytes"] == sum(
        child.stat().st_size for child in destination.iterdir() if child.is_file()
    )
    assert event["retained"] == [destination.name]
    assert event["removed"] == [old_directory.name]


def test_logger_run_id_is_created_once_on_rank0_and_broadcast_to_every_rank(
    monkeypatch, tmp_path
):
    broadcast_value = []

    def fake_broadcast(holder, src):
        assert src == 0
        if holder[0] is not None:
            broadcast_value[:] = holder
        else:
            holder[:] = broadcast_value

    monkeypatch.setattr(trainer_module.dist, "broadcast_object_list", fake_broadcast)
    monkeypatch.setattr(
        trainer_module.uuid, "uuid4", lambda: SimpleNamespace(hex="shared-run-id")
    )
    monkeypatch.setattr(stage2_metrics, "Stage2MetricsLogger", _CapturedLogger)
    _CapturedLogger.instances.clear()

    resolved = SimpleNamespace(
        jsonl_path="metrics/stage2.jsonl",
        fsync_every_steps=1,
        contract_hash=lambda: "a" * 64,
    )
    state = SimpleNamespace(successful_attempts=7)
    common = {
        "resolved": resolved,
        "state": state,
        "options": SimpleNamespace(output_dir=tmp_path),
        "_run_metadata": lambda _identity: {"kind": "tiny"},
    }
    rank0 = _bare_trainer(is_main_process=True, **common)
    rank1 = _bare_trainer(is_main_process=False, **common)

    rank0._build_logger(None, {"commit": "b" * 40})
    rank1._build_logger(None, {"commit": "b" * 40})

    assert rank0.logger.run_id == rank1.logger.run_id == "shared-run-id"
    assert rank0.logger.options["enabled"] is True
    assert rank1.logger.options["enabled"] is False
    assert rank0.logger.options["resume_from_logical_substep"] == 7


def test_metric_clock_tracks_every_f1_through_g_success_commit():
    resolved = load_stage2_config(CONFIG_PATH)
    schedule = Stage2TrainingSchedule.from_resolved_config(resolved)
    state = Stage2TrainingState()
    trainer = _bare_trainer(state=state)
    observed = []

    for logical, substep in enumerate(("F1", "F2", "F3", "F4", "F5")):
        state.commit_successful_fake_update(substep, schedule=schedule)
        observed.append(trainer._metric_clock(substep, logical))
    state.commit_successful_generator_update(ema_action="skipped", schedule=schedule)
    observed.append(trainer._metric_clock("G", 5))

    assert [item["cycle_substep"] for item in observed] == [
        "F1",
        "F2",
        "F3",
        "F4",
        "F5",
        "G",
    ]
    assert [item["logical_substep_id"] for item in observed] == list(range(6))
    assert [item["completed_fake_updates"] for item in observed] == [1, 2, 3, 4, 5, 5]
    assert [item["completed_generator_updates"] for item in observed] == [
        0,
        0,
        0,
        0,
        0,
        1,
    ]
    assert [item["completed_cycles"] for item in observed] == [0, 0, 0, 0, 0, 1]


@pytest.mark.parametrize(
    ("smoke_mode", "starting_cycles", "expect_checkpoint"),
    [
        pytest.param("C0", 0, True, id="C0-saves-cold-boundary"),
        pytest.param("C1", 1, True, id="C1-saves-resumed-boundary"),
        pytest.param("C2", 2, False, id="C2-discards-forced-dfd-result"),
    ],
)
def test_smoke_loop_runs_one_complete_cycle_then_saves_or_discards_at_boundary(
    tmp_path,
    smoke_mode,
    starting_cycles,
    expect_checkpoint,
):
    resolved = load_stage2_config(CONFIG_PATH)
    schedule = Stage2TrainingSchedule.from_resolved_config(resolved)
    state = Stage2TrainingState()

    def commit_cycle():
        for substep in ("F1", "F2", "F3", "F4", "F5"):
            state.commit_successful_fake_update(substep, schedule=schedule)
        next_completed_g = state.completed_g + 1
        state.commit_successful_generator_update(
            ema_action=schedule.expected_ema_action(next_completed_g),
            schedule=schedule,
        )

    for _ in range(starting_cycles):
        commit_cycle()

    executed = []
    checkpoint_calls = []
    records = []

    def run_one_substep():
        substep = state.next_substep
        executed.append(substep)
        if substep == "G":
            next_completed_g = state.completed_g + 1
            state.commit_successful_generator_update(
                ema_action=schedule.expected_ema_action(next_completed_g),
                schedule=schedule,
            )
        else:
            state.commit_successful_fake_update(substep, schedule=schedule)
        return {"elapsed": 1.0}

    def save_checkpoint(identity):
        checkpoint_calls.append(identity)
        return {
            "path": tmp_path / f"checkpoint-{state.completed_g}",
            "manifest_sha256": "a" * 64,
            "total_bytes": 123,
            "retained": [],
            "removed": [],
        }

    trainer = _bare_trainer(
        resolved=resolved,
        schedule=schedule,
        state=state,
        options=SimpleNamespace(
            smoke_mode=smoke_mode,
            dry_run=True,
            no_save=smoke_mode == "C2",
        ),
        _append_metric=lambda record_type, fields: records.append(
            (record_type, dict(fields))
        ),
        _run_one_logical_substep=run_one_substep,
        _save_checkpoint=save_checkpoint,
    )

    trainer._train_loop({"commit": "b" * 40})

    assert executed == ["F1", "F2", "F3", "F4", "F5", "G"]
    assert state.cycle == starting_cycles + 1
    assert bool(checkpoint_calls) is expect_checkpoint
    assert [record_type for record_type, _fields in records] == (
        ["cycle_summary", "checkpoint_event"]
        if expect_checkpoint
        else ["cycle_summary"]
    )
    assert records[0][1]["dry_run"] is True


@pytest.mark.parametrize("sample_count", [1, 2])
def test_update_attempt_scales_global_loss_and_generator_tokens_by_microbatch(
    monkeypatch,
    sample_count,
):
    parameter = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    target = nn.Module()
    target.register_parameter("lora_weight", parameter)
    frozen = nn.Linear(1, 1).requires_grad_(False)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scaling_calls = []

    class TinyLoss:
        def __init__(self):
            self.numerator = parameter.square() * (12.0 * sample_count)
            self.count = 12 * sample_count
            self.loss = self.numerator / self.count

    micro = {
        "output": TinyLoss(),
        "diagnostic": {"finite_metric": 1.0},
        "sample_count": sample_count,
        "rollout_result": SimpleNamespace(
            cache_audit={
                "generator_forward_calls": 3,
                "logical_query_tokens": 100,
            }
        ),
        "score_calls": 1,
        "frame_time": torch.tensor(
            [[0.0, 100.0 + index] for index in range(sample_count)]
        ),
    }
    resolved = SimpleNamespace(
        global_batch_size=64,
        score_seq_len=9_750,
        generator_optimizer=SimpleNamespace(max_grad_norm=10.0),
        fake_score_optimizer=SimpleNamespace(max_grad_norm=10.0),
    )
    trainer = _bare_trainer(
        model=SimpleNamespace(
            generator=frozen,
            real_score=frozen,
            fake_score=target,
        ),
        optimizers={"fake_score": optimizer},
        resolved=resolved,
        world_size=8,
        device=torch.device("cpu"),
        _compute_micro_loss=lambda **_kwargs: micro,
        _world_consensus=lambda condition: bool(condition),
        _gradients_finite=lambda _module: True,
        _other_role_gradients_absent=lambda _role: True,
        _clip_grad_norm=lambda _module, _maximum: 1.0,
        _reduce_loss=lambda numerator, count: (float(numerator), count),
    )

    def capture_scaling(numerator, *, global_count, world_size):
        scaling_calls.append((global_count, world_size))
        return numerator

    monkeypatch.setattr(
        train_state_module,
        "stage2_global_mean_loss_for_backward",
        capture_scaling,
    )
    monkeypatch.setattr(
        distributed_utils, "fsdp2_accumulation", lambda *_a, **_k: nullcontext()
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)

    result = trainer._run_update_attempt(
        role="fake_score",
        batches=[{"sample_id": torch.arange(sample_count)}],
        exits=[2],
        branch="flow_dsm",
    )

    assert result["success"] is True
    assert scaling_calls == [(768, 8)]
    assert result["loss_count"] == 12 * sample_count
    assert result["rollout_tokens"] == 100 * sample_count


@pytest.mark.parametrize(
    ("local_error", "consensus_results", "expected_error"),
    [
        pytest.param(None, [True], None, id="all-ranks-ready"),
        pytest.param(
            None,
            [False, True],
            ValueError,
            id="peer-rank-nonfinite-retries-before-score",
        ),
        pytest.param(
            TypeError("bad score shape"),
            [False, False],
            RuntimeError,
            id="contract-error-fails-all-ranks-before-score",
        ),
    ],
)
def test_pre_score_failures_choose_one_collective_branch_on_every_rank(
    local_error,
    consensus_results,
    expected_error,
):
    observed_conditions = []
    results = iter(consensus_results)
    trainer = _bare_trainer(
        _world_consensus=lambda condition: observed_conditions.append(condition)
        or next(results)
    )

    context = pytest.raises(expected_error) if expected_error else nullcontext()
    with context:
        trainer._align_pre_score_failure("score noising/setup", local_error)

    assert observed_conditions == (
        [local_error is None]
        if consensus_results == [True]
        else [
            local_error is None,
            local_error is None or isinstance(local_error, ValueError),
        ]
    )


def test_nonfinite_prediction_uses_world_gate_and_skips_optimizer(monkeypatch):
    parameter = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    target = nn.Module()
    target.register_parameter("lora_weight", parameter)

    class CountingOptimizer:
        def __init__(self):
            self.zero_grad_calls = 0
            self.step_calls = 0

        def zero_grad(self, *, set_to_none):
            assert set_to_none is True
            self.zero_grad_calls += 1

        def step(self):
            self.step_calls += 1

    optimizer = CountingOptimizer()
    consensus_inputs = []
    trainer = _bare_trainer(
        model=SimpleNamespace(fake_score=target),
        optimizers={"fake_score": optimizer},
        resolved=SimpleNamespace(),
        device=torch.device("cpu"),
        _compute_micro_loss=lambda **_kwargs: (_ for _ in ()).throw(
            FloatingPointError("non-finite prediction")
        ),
        _world_consensus=lambda condition: consensus_inputs.append(condition) or False,
    )
    monkeypatch.setattr(
        distributed_utils, "fsdp2_accumulation", lambda *_a, **_k: nullcontext()
    )

    result = trainer._run_update_attempt(
        role="fake_score",
        batches=[{}],
        exits=[0],
        branch="flow_dsm",
    )

    assert result == {
        "success": False,
        "reason": "nonfinite prediction/loss: non-finite prediction",
    }
    assert consensus_inputs == [False]
    assert optimizer.step_calls == 0
    assert optimizer.zero_grad_calls == 2


def test_nonfinite_attempt_is_replayed_then_only_the_success_clock_commits(monkeypatch):
    resolved_config = load_stage2_config(CONFIG_PATH)
    schedule = Stage2TrainingSchedule.from_resolved_config(resolved_config)
    state = Stage2TrainingState()
    records = []
    restores = []
    attempts = iter(
        [
            {"success": False, "reason": "peer rank nonfinite"},
            {
                "success": True,
                "loss": 1.0,
                "loss_numerator": 24.0,
                "loss_count": 24,
                "preclip_grad_norm": 1.0,
                "diagnostics": {},
                "rollout_forward_calls": 3,
                "rollout_tokens": 100,
                "score_forward_calls": 1,
                "score_tokens": 9_750,
                "exit_values": [0],
                "timestep_values": [100.0],
                "compute_seconds": 0.2,
                "optimizer_seconds": 0.1,
            },
        ]
    )
    timing = {
        "step_seconds_max": 1.0,
        "step_seconds_mean": 0.9,
        "straggler_ratio": 1.0 / 0.9,
        "data_seconds_max": 0.1,
        "h2d_seconds_max": 0.1,
        "compute_seconds_max": 0.2,
        "optimizer_seconds_max": 0.1,
        "ema_seconds_max": 0.0,
        "timing_closure_error_seconds": 0.5,
    }
    trainer = _bare_trainer(
        state=state,
        schedule=schedule,
        resolved=resolved_config,
        options=SimpleNamespace(smoke_mode=None, dry_run=False),
        device=torch.device("cpu"),
        world_size=8,
        is_main_process=True,
        logger=SimpleNamespace(
            append=lambda record_type, fields: records.append(
                (record_type, dict(fields))
            )
        ),
        exit_rng=SimpleNamespace(draw=lambda *_args, **_kwargs: [0]),
        _snapshot_attempt=lambda: {"snapshot": True},
        _restore_attempt=lambda snapshot: restores.append(snapshot),
        _materialize_batches=lambda _role: [{"sample_id": torch.tensor([7])}],
        _to_device=lambda batch: batch,
        _run_update_attempt=lambda **_kwargs: next(attempts),
        _timing_summary=lambda *_args, **_kwargs: timing,
        _memory_fields=lambda: {
            "gpu_memory_allocated_gib_max": 1.0,
            "gpu_memory_reserved_gib_max": 2.0,
        },
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda *_args, **_kwargs: None
    )

    result = trainer._run_one_logical_substep()

    assert [record_type for record_type, _fields in records] == [
        "nonfinite_attempt",
        "train_step",
    ]
    assert records[0][1]["completed_fake_updates"] == 0
    assert records[0][1]["logical_substep_id"] == 0
    assert result["fields"]["completed_fake_updates"] == 1
    assert result["fields"]["cycle_substep"] == "F1"
    assert state.nonfinite_attempts == 1
    assert state.completed_f == 1
    assert state.completed_g == 0
    assert len(restores) == 2


@pytest.mark.parametrize(
    ("smoke_mode", "expected_branch"),
    [
        pytest.param("C1", "dmd", id="C1-forces-pure-dmd"),
        pytest.param("C2", "dfd", id="C2-forces-dfd"),
    ],
)
def test_resume_smoke_forces_its_generator_branch(
    monkeypatch, smoke_mode, expected_branch
):
    resolved = load_stage2_config(CONFIG_PATH)
    schedule = Stage2TrainingSchedule.from_resolved_config(resolved)
    state = Stage2TrainingState()
    for substep in ("F1", "F2", "F3", "F4", "F5"):
        state.commit_successful_fake_update(substep, schedule=schedule)

    captured_branches = []
    captured_probabilities = []
    timing = {
        "step_seconds_max": 1.0,
        "step_seconds_mean": 1.0,
        "straggler_ratio": 1.0,
        "data_seconds_max": 0.1,
        "h2d_seconds_max": 0.1,
        "compute_seconds_max": 0.2,
        "optimizer_seconds_max": 0.1,
        "ema_seconds_max": 0.1,
        "timing_closure_error_seconds": 0.4,
    }

    def run_update_attempt(**options):
        captured_branches.append(options["branch"])
        return {
            "success": True,
            "loss": 1.0,
            "loss_numerator": 24.0,
            "loss_count": 24,
            "preclip_grad_norm": 1.0,
            "diagnostics": {},
            "rollout_forward_calls": 3,
            "rollout_tokens": 100,
            "score_forward_calls": 1,
            "score_tokens": 9_750,
            "exit_values": [0],
            "timestep_values": [100.0],
            "compute_seconds": 0.2,
            "optimizer_seconds": 0.1,
        }

    def draw_branch(probability):
        captured_probabilities.append(probability)
        return "dfd" if probability == 1.0 else "dmd"

    trainer = _bare_trainer(
        state=state,
        schedule=schedule,
        resolved=resolved,
        options=SimpleNamespace(smoke_mode=smoke_mode, dry_run=True),
        device=torch.device("cpu"),
        world_size=8,
        is_main_process=False,
        logger=SimpleNamespace(append=lambda *_args, **_kwargs: None),
        model=SimpleNamespace(generator=object()),
        generator_ema=SimpleNamespace(
            update_after_step=lambda _model, completed_g: schedule.expected_ema_action(
                completed_g
            )
        ),
        exit_rng=SimpleNamespace(draw=lambda *_args, **_kwargs: [0]),
        _snapshot_attempt=lambda: {"snapshot": True},
        _restore_attempt=lambda _snapshot: None,
        _materialize_batches=lambda _role: [{"sample_id": torch.tensor([7])}],
        _to_device=lambda batch: batch,
        _draw_branch=draw_branch,
        _run_update_attempt=run_update_attempt,
        _timing_summary=lambda *_args, **_kwargs: timing,
        _memory_fields=lambda: {
            "gpu_memory_allocated_gib_max": 1.0,
            "gpu_memory_reserved_gib_max": 2.0,
        },
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda *_args, **_kwargs: None
    )

    result = trainer._run_one_logical_substep()

    assert captured_branches == [expected_branch]
    assert captured_probabilities == [1.0 if smoke_mode == "C2" else 0.0]
    assert result["fields"]["branch"] == expected_branch
    assert result["fields"]["branch_is_dfd"] == int(expected_branch == "dfd")


def test_timing_summary_reports_positive_slowest_rank_fields_and_closure(monkeypatch):
    trainer = _bare_trainer(world_size=2, device=torch.device("cpu"))

    def fake_all_gather(outputs, local):
        outputs[0].copy_(local)
        outputs[1].copy_(
            torch.tensor(
                [
                    2.0,
                    0.2,
                    0.3,
                    0.2,
                    0.1,
                    0.1,
                    0.1,
                    0.1,
                    0.2,
                    0.1,
                    0.05,
                ],
                dtype=local.dtype,
            )
        )

    monkeypatch.setattr(trainer_module.dist, "all_gather", fake_all_gather)
    result = trainer._timing_summary(
        1.5,
        {"data": 0.1, "h2d": 0.2, "compute": 0.7, "optimizer": 0.1, "ema": 0.05},
    )

    assert result["step_seconds_max"] == pytest.approx(2.0)
    assert result["step_seconds_mean"] == pytest.approx(1.75)
    assert result["straggler_ratio"] > 1.0
    for field in (
        "data_seconds_max",
        "h2d_seconds_max",
        "rollout_seconds_max",
        "fake_score_seconds_max",
        "real_cond_seconds_max",
        "real_uncond_seconds_max",
        "loss_build_seconds_max",
        "backward_seconds_max",
        "clip_optimizer_seconds_max",
        "compute_seconds_max",
        "optimizer_seconds_max",
        "ema_seconds_max",
        "timing_closure_error_seconds",
    ):
        assert result[field] > 0.0


def test_generator_and_fake_score_optimizers_use_their_locked_specs():
    resolved = load_stage2_config(CONFIG_PATH)

    def module():
        result = nn.Module()
        result.register_parameter(
            "lora_weight", nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        )
        return result

    generator = Trainer._optimizer(module(), resolved.generator_optimizer)
    fake_score = Trainer._optimizer(module(), resolved.fake_score_optimizer)

    assert isinstance(generator, torch.optim.AdamW)
    assert isinstance(fake_score, torch.optim.AdamW)
    assert generator.param_groups[0]["lr"] == pytest.approx(2.0e-6)
    assert fake_score.param_groups[0]["lr"] == pytest.approx(4.0e-7)
    for optimizer in (generator, fake_score):
        group = optimizer.param_groups[0]
        assert group["betas"] == (0.0, 0.999)
        assert group["eps"] == pytest.approx(1.0e-8)
        assert group["weight_decay"] == pytest.approx(0.0)
