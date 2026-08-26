from __future__ import annotations

import ast
import hashlib
import inspect
import json
import random
import copy
import textwrap
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import create_autospec

import numpy as np
import pytest
import torch
from torch import nn

import trainer.stage2_distillation as trainer_module
import utils.distributed as distributed_utils
import utils.stage2_checkpoint as stage2_checkpoint
import utils.stage2_metrics as stage2_metrics
import utils.stage2_train_state as train_state_module
from model.stage2_dmd import Stage2DMD
from trainer.stage2_distillation import (
    Trainer,
    _STAGE2_CHECKPOINT_RNG_RUNTIME_API_VERSION,
    _STAGE2_DMD_RUNTIME_METHODS,
    _audit_stage2_checkpoint_rng_runtime_api,
    _audit_stage2_dmd_runtime_api,
    _should_apply_stage2_checkpoint_retention,
)
from utils.stage2_config import load_stage2_config
from utils.stage2_sampler import build_stage2_role_samplers
from utils.stage2_train_state import Stage2TrainingSchedule, Stage2TrainingState

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "train_i2v_stage2_600cats.yaml"
C4W16_CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "train_i2v_stage2_600cats_c4w16s1_micro1_acc8.yaml"
)


def test_c4w16_keep_all_profile_skips_quadratic_checkpoint_retention_scan():
    baseline = load_stage2_config(CONFIG_PATH)
    c4w16 = load_stage2_config(C4W16_CONFIG_PATH)

    assert _should_apply_stage2_checkpoint_retention(baseline) is True
    assert c4w16.checkpoint_interval_generator_updates == 10
    assert c4w16.keep_last_resumable == 400
    assert c4w16.milestone_generator_updates == ()
    assert _should_apply_stage2_checkpoint_retention(c4w16) is False


def test_stage2_dmd_runtime_api_rejects_stale_loss_signatures(monkeypatch):
    audit = _audit_stage2_dmd_runtime_api(Stage2DMD)
    assert audit["api_version"] == Stage2DMD.RUNTIME_API_VERSION
    assert audit["timing_fields"] == stage2_metrics.STAGE2_TIMING_FIELDS
    assert set(audit["methods"]) == {
        "fake_score_flow_dsm_loss_from_model",
        "generator_distribution_matching_loss_from_models",
    }

    class _UnversionedStage2DMD:
        pass

    with pytest.raises(RuntimeError, match=r"version mismatch.*actual=None"):
        _audit_stage2_dmd_runtime_api(_UnversionedStage2DMD)

    class _StaleStage2DMD:
        RUNTIME_API_VERSION = Stage2DMD.RUNTIME_API_VERSION

        def fake_score_flow_dsm_loss_from_model(
            self,
            *,
            generated_future,
            noised_fake_score,
            conditional_dict,
        ):
            del generated_future, noised_fake_score, conditional_dict

        def generator_distribution_matching_loss_from_models(
            self,
            *,
            branch,
            generated_future,
            noised_score,
            conditional_dict,
            real_unconditional_dict,
            timing_callback=None,
        ):
            del (
                branch,
                generated_future,
                noised_score,
                conditional_dict,
                real_unconditional_dict,
                timing_callback,
            )

    with pytest.raises(RuntimeError, match=r"fake_score.*timing_callback"):
        _audit_stage2_dmd_runtime_api(_StaleStage2DMD)

    monkeypatch.setattr(
        stage2_metrics,
        "STAGE2_TIMING_FIELDS",
        tuple(
            field
            for field in stage2_metrics.STAGE2_TIMING_FIELDS
            if field != "orchestration_seconds_max"
        ),
    )
    with pytest.raises(RuntimeError, match=r"timing runtime API mismatch"):
        _audit_stage2_dmd_runtime_api(Stage2DMD)


def test_stage2_checkpoint_rng_runtime_api_accepts_current_source():
    audit = _audit_stage2_checkpoint_rng_runtime_api(stage2_checkpoint)

    assert audit["api_version"] == _STAGE2_CHECKPOINT_RNG_RUNTIME_API_VERSION
    assert Path(audit["source_file"]) == Path(stage2_checkpoint.__file__).resolve()
    assert audit["world_sizes"] == (4, 8)
    assert "expected_world_size" in audit["signatures"]["restore_stage2_rng_state"]


def test_stage2_checkpoint_rng_runtime_api_rejects_legacy_restore_signature():
    def capture_stage2_rng_state(
        *,
        rank,
        world_size=8,
        dedicated_generators,
        rank0_control_generators,
        include_cuda=True,
    ):
        del (
            rank,
            world_size,
            dedicated_generators,
            rank0_control_generators,
            include_cuda,
        )

    def validate_stage2_rng_state(
        state,
        *,
        expected_rank,
        expected_world_size=8,
        expected_dedicated_names=None,
    ):
        del state, expected_rank, expected_world_size, expected_dedicated_names

    def restore_stage2_rng_state(
        state,
        *,
        rank,
        dedicated_generators,
        rank0_control_generators,
        require_cuda_topology=True,
    ):
        del (
            state,
            rank,
            dedicated_generators,
            rank0_control_generators,
            require_cuda_topology,
        )

    legacy = SimpleNamespace(
        __file__=__file__,
        STAGE2_WORLD_SIZES=(4, 8),
        STAGE2_CHECKPOINT_RNG_RUNTIME_API_VERSION=(
            _STAGE2_CHECKPOINT_RNG_RUNTIME_API_VERSION
        ),
        capture_stage2_rng_state=capture_stage2_rng_state,
        validate_stage2_rng_state=validate_stage2_rng_state,
        restore_stage2_rng_state=restore_stage2_rng_state,
    )

    with pytest.raises(
        RuntimeError,
        match=r"restore_stage2_rng_state.*expected_world_size.*Deploy one complete",
    ):
        _audit_stage2_checkpoint_rng_runtime_api(
            legacy,
            expected_source=Path(__file__),
        )


def test_trainer_micro_loss_call_keywords_match_audited_runtime_api():
    tree = ast.parse(textwrap.dedent(inspect.getsource(Trainer._compute_micro_loss)))
    calls = {
        node.func.attr: tuple(
            keyword.arg for keyword in node.keywords if keyword.arg is not None
        )
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _STAGE2_DMD_RUNTIME_METHODS
    }

    assert calls == _STAGE2_DMD_RUNTIME_METHODS


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


def test_resume_role_initialization_reaudits_assets_before_any_model_load(
    monkeypatch,
):
    import utils.stage2_role_init as role_init_module

    recorded_assets = {"snapshot": "previous-job"}
    fresh_assets = {"snapshot": "current-rank0-audit"}
    resume_payload = SimpleNamespace(
        provenance={"assets": recorded_assets},
        generator_raw={"generator": "adapter"},
        fake_score_raw={"fake_score": "adapter"},
        manifest={"manifest_sha256": "a" * 64},
    )
    rank0_labels = []
    refresh_calls = []
    initialize_calls = []

    def rank0_checked(label, callback):
        rank0_labels.append(label)
        return callback()

    def refresh(assets, *, architecture_root):
        refresh_calls.append((assets, architecture_root))
        return fresh_assets

    initialized_model = object()

    def initialize(resolved, **kwargs):
        initialize_calls.append((resolved, kwargs))
        assert kwargs["assets"] is fresh_assets
        assert kwargs["resume_adapter_states"] == {
            "generator": resume_payload.generator_raw,
            "fake_score": resume_payload.fake_score_raw,
        }
        return SimpleNamespace(
            model=initialized_model,
            role_audits={"roles": "audited"},
            lora_schemas={"schemas": "immutable"},
        )

    monkeypatch.setattr(
        role_init_module, "refresh_stage2_role_asset_identities", refresh
    )
    monkeypatch.setattr(role_init_module, "initialize_stage2_roles", initialize)
    monkeypatch.setattr(
        trainer_module,
        "_audit_stage2_dmd_runtime_api",
        lambda model_type: {"model_type": model_type.__name__},
    )
    expected_runtime_audit = {"model_type": "object"}
    trainer = _bare_trainer(
        resolved=SimpleNamespace(architecture_root="/current/architecture"),
        is_main_process=True,
        _rank0_checked=rank0_checked,
        _world_checked=lambda _label, callback: callback(),
        model_runtime_api_audit=expected_runtime_audit,
    )

    trainer._initialize_roles(mesh="mesh", resume_payload=resume_payload)

    assert rank0_labels == ["refresh immutable role assets for resume"]
    assert refresh_calls == [(recorded_assets, "/current/architecture")]
    assert len(initialize_calls) == 1
    assert trainer.assets is fresh_assets
    assert trainer.model is initialized_model
    assert trainer.role_audits == {"roles": "audited"}
    assert trainer.lora_schemas == {"schemas": "immutable"}

    trainer.dataset = SimpleNamespace(
        manifest={"manifest_sha256": "b" * 64},
        source_manifest={"manifest_sha256": "c" * 64},
        negative_conditioning={
            "manifest": {
                "manifest_sha256": "d" * 64,
                "artifact": {"sha256": "e" * 64},
            }
        },
    )
    trainer.resume_checkpoint = "/checkpoint_stage2_g000001"
    trainer.resume_payload = resume_payload
    trainer._last_cycle_smoke_probe = None
    child_provenance = trainer._checkpoint_provenance()
    assert child_provenance["assets"] is fresh_assets
    assert child_provenance["lineage"] == {
        "parent_checkpoint": "/checkpoint_stage2_g000001",
        "parent_checkpoint_manifest_sha256": "a" * 64,
    }


def _sampler_state(*, completed_f: int, completed_g: int):
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


def _next_f1_probe_trainer(*, smoke_mode="C1"):
    from pipeline.stage2_rollout import Stage2ExitRNGStreams

    resolved = load_stage2_config(CONFIG_PATH)
    schedule = Stage2TrainingSchedule.from_resolved_config(resolved)
    state = Stage2TrainingState()
    for substep in ("F1", "F2", "F3", "F4", "F5"):
        state.commit_successful_fake_update(substep, schedule=schedule)
    state.commit_successful_generator_update(
        ema_action=schedule.expected_ema_action(1),
        schedule=schedule,
    )
    actions = ("a", "b", "c")
    spatial_shapes = [(30, 52)] * 600
    samplers = build_stage2_role_samplers(
        [action for action in actions for _ in range(200)],
        spatial_shapes,
        action_order=actions,
        base_seed=resolved.training_seed,
        microbatch_size_per_device=resolved.microbatch_size_per_device,
    )
    for _ in range(state.completed_f):
        samplers.fake_score.next_global_batch()
    for _ in range(state.completed_g):
        samplers.generator.next_global_batch()
    dataloader_generators = {
        role: torch.Generator(device="cpu").manual_seed(seed)
        for role, seed in (("fake_score", 101), ("generator", 102))
    }
    dedicated_generators = {
        f"{role}_{kind}": torch.Generator(device="cpu").manual_seed(seed)
        for role, base in (("fake_score", 200), ("generator", 300))
        for kind, seed in zip(("rollout", "timestep", "noise"), range(base, base + 3))
    }
    return _bare_trainer(
        resolved=resolved,
        schedule=schedule,
        state=state,
        options=SimpleNamespace(smoke_mode=smoke_mode),
        rank=0,
        world_size=8,
        is_main_process=True,
        device=torch.device("cpu"),
        dataset=SimpleNamespace(spatial_shapes=spatial_shapes),
        samplers=samplers,
        dataloader_generators=dataloader_generators,
        dedicated_generators=dedicated_generators,
        exit_rng=Stage2ExitRNGStreams(resolved.training_seed),
        branch_rng=torch.Generator(device="cpu").manual_seed(401),
        resume_payload=None,
        _smoke_parent_probe_consumed=False,
    )


def test_next_f1_probe_predicts_exact_draws_without_consuming_any_state():
    from utils.stage2_sampler import partition_stage2_global_batch
    from utils.stage2_score_math import sample_stage2_score_timesteps

    trainer = _next_f1_probe_trainer()
    snapshot = trainer._snapshot_attempt()
    before = trainer._attempt_state_sha256(snapshot)

    probe = trainer._capture_next_f1_probe()

    assert trainer._attempt_state_sha256(trainer._snapshot_attempt()) == before
    assert probe["schema"] == "longlive_stage2_next_f1_probe/v1"
    assert probe["state"] == {
        "next_substep": "F1",
        "current_role": "fake_score",
        "completed_fake_updates": 5,
        "completed_generator_updates": 1,
        "completed_cycles": 1,
        "successful_attempts": 6,
        "nonfinite_attempts": 0,
        "nonfinite_attempts_by_role": {"generator": 0, "fake_score": 0},
    }
    assert probe["sampler"]["completed_batches"] == 5
    assert probe["sampler"]["batch_cursor"] == 5
    assert len(probe["sampler"]["global_batch_ids"]) == 64
    assert sorted(probe["exit_schedule"]) == [0, 1, 2, 3]
    assert [payload["rank"] for payload in probe["rank_payloads"]] == [0]
    rank_payload = probe["rank_payloads"][0]
    assert rank_payload["attempt_state_sha256"] == before
    assert set(rank_payload["stream_state_sha256"]) == {
        "fake_score_loader",
        "fake_score_exit",
        "fake_score_rollout",
        "fake_score_timestep",
        "fake_score_noise",
    }
    assert len(rank_payload["microbatches"]) == 4

    # Independently replay the production draw order from the saved boundary:
    # sampler -> exit -> per-micro rollout noise -> timestep -> score epsilon.
    trainer._restore_attempt(snapshot)
    global_batch = trainer.samplers.fake_score.next_global_batch()
    local_microbatches = partition_stage2_global_batch(
        global_batch,
        rank=trainer.rank,
        world_size=trainer.world_size,
        microbatch_size_per_device=trainer.resolved.microbatch_size_per_device,
        gradient_accumulation_steps=trainer.resolved.gradient_accumulation_steps,
        spatial_shapes=trainer.dataset.spatial_shapes,
    )
    exits = trainer.exit_rng.draw(
        "fake_score",
        accumulation_steps=trainer.resolved.gradient_accumulation_steps,
        num_denoising_steps=trainer.resolved.num_denoising_steps,
        mode=trainer.resolved.exit_sampling,
        device=trainer.device,
        synchronize_ranks=True,
    )
    assert list(global_batch) == probe["sampler"]["global_batch_ids"]
    assert list(exits) == probe["exit_schedule"]
    for sample_ids, predicted in zip(local_microbatches, rank_payload["microbatches"]):
        shape = (
            len(sample_ids),
            trainer.resolved.generated_episode_frames,
            trainer.resolved.latent_channels,
            *trainer.dataset.spatial_shapes[sample_ids[0]],
        )
        rollout_noise = torch.randn(
            shape,
            device=trainer.device,
            dtype=torch.bfloat16,
            generator=trainer.dedicated_generators["fake_score_rollout"],
        )
        timesteps = sample_stage2_score_timesteps(
            batch_size=len(sample_ids),
            device=trainer.device,
            generator=trainer.dedicated_generators["fake_score_timestep"],
        )
        score_noise = torch.randn(
            shape,
            device=trainer.device,
            dtype=torch.float32,
            generator=trainer.dedicated_generators["fake_score_noise"],
        )
        assert predicted == {
            "sample_ids": list(sample_ids),
            "spatial_shape": list(trainer.dataset.spatial_shapes[sample_ids[0]]),
            "rollout_noise_sha256": trainer._tensor_sha256(rollout_noise),
            "score_uniform_integers": timesteps.uniform_integer.tolist(),
            "score_frame_timestep_sha256": trainer._tensor_sha256(
                timesteps.frame_timestep
            ),
            "score_noise_sha256": trainer._tensor_sha256(score_noise),
        }
    trainer._restore_attempt(snapshot)
    assert trainer._attempt_state_sha256(trainer._snapshot_attempt()) == before


def test_resume_probe_accepts_exact_parent_then_rejects_every_bound_field():
    trainer = _next_f1_probe_trainer()
    state_before = trainer._attempt_state_sha256(trainer._snapshot_attempt())
    expected = trainer._capture_next_f1_probe()
    trainer.resume_payload = SimpleNamespace(
        provenance={"smoke_probe": {"next_f1_probe": expected}}
    )

    trainer._verify_parent_next_f1_probe()

    assert trainer._smoke_parent_probe_consumed is True
    assert trainer._attempt_state_sha256(trainer._snapshot_attempt()) == state_before
    trainer._capture_next_f1_probe = lambda: (_ for _ in ()).throw(
        AssertionError("a consumed parent probe must never run twice")
    )
    trainer._verify_parent_next_f1_probe()

    mutations = {
        "counter": lambda value: value["state"].__setitem__(
            "completed_fake_updates", 4
        ),
        "sampler state": lambda value: value["sampler"].__setitem__(
            "state_sha256", "0" * 64
        ),
        "sampler cursor": lambda value: value["sampler"].__setitem__("batch_cursor", 4),
        "batch id": lambda value: value["sampler"]["global_batch_ids"].__setitem__(
            0, (value["sampler"]["global_batch_ids"][0] + 1) % 600
        ),
        "exit": lambda value: value["exit_schedule"].__setitem__(0, 9),
        "attempt state": lambda value: value["rank_payloads"][0].__setitem__(
            "attempt_state_sha256", "1" * 64
        ),
        "loader stream": lambda value: value["rank_payloads"][0][
            "stream_state_sha256"
        ].__setitem__("fake_score_loader", "2" * 64),
        "exit stream": lambda value: value["rank_payloads"][0][
            "stream_state_sha256"
        ].__setitem__("fake_score_exit", "3" * 64),
        "rollout stream": lambda value: value["rank_payloads"][0][
            "stream_state_sha256"
        ].__setitem__("fake_score_rollout", "4" * 64),
        "timestep stream": lambda value: value["rank_payloads"][0][
            "stream_state_sha256"
        ].__setitem__("fake_score_timestep", "5" * 64),
        "score-noise stream": lambda value: value["rank_payloads"][0][
            "stream_state_sha256"
        ].__setitem__("fake_score_noise", "6" * 64),
        "rollout prediction": lambda value: value["rank_payloads"][0]["microbatches"][
            0
        ].__setitem__("rollout_noise_sha256", "7" * 64),
        "timestep draw": lambda value: value["rank_payloads"][0]["microbatches"][0][
            "score_uniform_integers"
        ].__setitem__(0, 999),
        "timestep projection": lambda value: value["rank_payloads"][0]["microbatches"][
            0
        ].__setitem__("score_frame_timestep_sha256", "8" * 64),
        "score-noise prediction": lambda value: value["rank_payloads"][0][
            "microbatches"
        ][0].__setitem__("score_noise_sha256", "9" * 64),
    }
    trainer._capture_next_f1_probe = lambda: expected
    for label, mutate in mutations.items():
        tampered = copy.deepcopy(expected)
        mutate(tampered)
        trainer.resume_payload = SimpleNamespace(
            provenance={"smoke_probe": {"next_f1_probe": tampered}}
        )
        trainer._smoke_parent_probe_consumed = False
        with pytest.raises(RuntimeError, match="next F1 probe mismatch") as error:
            trainer._verify_parent_next_f1_probe()
        assert trainer._smoke_parent_probe_consumed is False
        assert "mismatch" in str(error.value), label


def test_resume_probe_requires_a_parent_probe_and_exact_f1_boundary():
    trainer = _next_f1_probe_trainer()
    trainer.resume_payload = SimpleNamespace(provenance={"smoke_probe": {}})
    with pytest.raises(RuntimeError, match="parent has no next F1 probe"):
        trainer._verify_parent_next_f1_probe()

    trainer.state.next_substep = "F2"
    with pytest.raises(RuntimeError, match="did not resume at F1"):
        trainer._verify_parent_next_f1_probe()


def test_c2_parent_probe_is_consumed_exactly_once():
    trainer = _next_f1_probe_trainer(smoke_mode="C2")
    expected = {"schema": "test-probe"}
    calls = []
    trainer.resume_payload = SimpleNamespace(
        provenance={"smoke_probe": {"next_f1_probe": expected}}
    )
    trainer._capture_next_f1_probe = lambda: calls.append(True) or expected

    trainer._verify_parent_next_f1_probe()
    trainer._verify_parent_next_f1_probe()

    assert calls == [True]
    assert trainer._smoke_parent_probe_consumed is True


@pytest.mark.parametrize(
    ("gather_mode", "message"),
    [
        pytest.param("missing", "every rank payload", id="missing-ranks"),
        pytest.param("duplicate", "incomplete or duplicated", id="duplicate-ranks"),
    ],
)
def test_next_f1_probe_rejects_incomplete_distributed_rank_payloads(
    monkeypatch, gather_mode, message
):
    trainer = _next_f1_probe_trainer()
    trainer.resolved = SimpleNamespace(
        microbatch_size_per_device=2,
        gradient_accumulation_steps=4,
        generated_episode_frames=1,
        latent_channels=1,
        num_denoising_steps=4,
        exit_sampling="stratified_uniform",
    )
    trainer._runtime_world_checked = lambda _label, callback: callback()
    trainer._world_consensus = bool

    monkeypatch.setattr(trainer_module.dist, "is_available", lambda: True)
    monkeypatch.setattr(trainer_module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(trainer_module.dist, "get_backend", lambda: "gloo")
    monkeypatch.setattr(trainer_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(
        trainer_module.dist, "broadcast", lambda *_args, **_kwargs: None
    )

    def all_gather_object(outputs, local):
        if gather_mode == "missing":
            outputs[0] = local
        else:
            for index in range(len(outputs)):
                outputs[index] = copy.deepcopy(local)

    monkeypatch.setattr(trainer_module.dist, "all_gather_object", all_gather_object)

    with pytest.raises(RuntimeError, match=message):
        trainer._capture_next_f1_probe()


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
    assert trainer.model_runtime_api_audit["api_version"] == (
        Stage2DMD.RUNTIME_API_VERSION
    )


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


def _branch_resume_trainer(tmp_path, anchor):
    return _bare_trainer(
        resolved=SimpleNamespace(
            resume_stage2_checkpoint=str(anchor),
            contract_hash=lambda: "a" * 64,
            phase_b_mode="dmd_only",
            fsdp_backend="fully_shard",
            microbatch_size_per_device=2,
            gradient_accumulation_steps=4,
            global_batch_size=64,
        ),
        options=SimpleNamespace(
            output_dir=tmp_path,
            auto_resume=True,
            smoke_mode=None,
        ),
        world_size=8,
    )


def _write_branch_lineage(path, *, parent, parent_sha256):
    from utils.stage1_io import canonical_json_sha256

    path.mkdir(parents=True)
    provenance = {
        "lineage": {
            "parent_checkpoint": None if parent is None else str(parent.resolve()),
            "parent_checkpoint_manifest_sha256": parent_sha256,
        }
    }
    (path / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return canonical_json_sha256(provenance)


def test_explicit_a24_is_used_when_branch_output_has_no_local_checkpoint(
    monkeypatch, tmp_path
):
    anchor = tmp_path / "parent" / "checkpoint_stage2_g000240"
    trainer = _branch_resume_trainer(tmp_path / "b0", anchor)
    monkeypatch.setattr(
        stage2_checkpoint, "find_latest_stage2_checkpoint", lambda _root: None
    )

    assert trainer._discover_resume_checkpoint() == anchor.resolve()


def test_explicit_a24_becomes_anchor_for_strict_local_child_auto_resume(
    monkeypatch, tmp_path
):
    output = tmp_path / "b0"
    anchor = tmp_path / "b1" / "checkpoint_stage2_g000240"
    g250 = output / "checkpoint_stage2_g000250"
    g260 = output / "checkpoint_stage2_g000260"
    anchor.mkdir(parents=True)
    hashes = {
        anchor.resolve(): "1" * 64,
        g250.resolve(): "2" * 64,
        g260.resolve(): "3" * 64,
    }
    provenance_hashes = {
        g250.resolve(): _write_branch_lineage(
            g250, parent=anchor, parent_sha256=hashes[anchor.resolve()]
        ),
        g260.resolve(): _write_branch_lineage(
            g260, parent=g250, parent_sha256=hashes[g250.resolve()]
        ),
    }
    trainer = _branch_resume_trainer(output, anchor)
    monkeypatch.setattr(
        stage2_checkpoint, "find_latest_stage2_checkpoint", lambda _root: g260
    )
    validated = []

    def validate(path, **expected):
        path = Path(path).resolve()
        validated.append(path)
        assert expected["expected_contract_hash"] == "a" * 64
        assert expected["expected_phase_b_mode"] == "dmd_only"
        assert expected["expected_topology"] == trainer._checkpoint_topology()
        return {
            "manifest_sha256": hashes[path],
            "provenance_sha256": provenance_hashes.get(path, "f" * 64),
        }

    monkeypatch.setattr(stage2_checkpoint, "validate_stage2_checkpoint", validate)

    assert trainer._discover_resume_checkpoint() == g260.resolve()
    assert validated == [g260.resolve(), g250.resolve(), anchor.resolve()]


@pytest.mark.parametrize("corruption", ["unrelated_parent", "parent_hash"])
def test_explicit_branch_anchor_rejects_unrelated_or_drifted_local_lineage(
    monkeypatch, tmp_path, corruption
):
    output = tmp_path / "b0"
    anchor = tmp_path / "b1" / "checkpoint_stage2_g000240"
    unrelated = tmp_path / "other" / "checkpoint_stage2_g000240"
    child = output / "checkpoint_stage2_g000250"
    anchor.mkdir(parents=True)
    unrelated.mkdir(parents=True)
    parent = unrelated if corruption == "unrelated_parent" else anchor
    parent_hash = "9" * 64 if corruption == "parent_hash" else "4" * 64
    provenance_hash = _write_branch_lineage(
        child,
        parent=parent,
        parent_sha256=parent_hash,
    )
    trainer = _branch_resume_trainer(output, anchor)
    monkeypatch.setattr(
        stage2_checkpoint, "find_latest_stage2_checkpoint", lambda _root: child
    )
    manifests = {
        child.resolve(): {
            "manifest_sha256": "2" * 64,
            "provenance_sha256": provenance_hash,
        },
        anchor.resolve(): {
            "manifest_sha256": "1" * 64,
            "provenance_sha256": "e" * 64,
        },
    }
    monkeypatch.setattr(
        stage2_checkpoint,
        "validate_stage2_checkpoint",
        lambda path, **_expected: manifests[Path(path).resolve()],
    )

    match = "left --logdir" if corruption == "unrelated_parent" else "hash drifted"
    with pytest.raises(RuntimeError, match=match):
        trainer._discover_resume_checkpoint()


def test_incomplete_newer_branch_directory_does_not_mask_complete_child(
    monkeypatch, tmp_path
):
    output = tmp_path / "b0"
    anchor = tmp_path / "b1" / "checkpoint_stage2_g000240"
    child = output / "checkpoint_stage2_g000250"
    incomplete = output / "checkpoint_stage2_g000260"
    anchor.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    provenance_hash = _write_branch_lineage(
        child,
        parent=anchor,
        parent_sha256="1" * 64,
    )
    trainer = _branch_resume_trainer(output, anchor)
    # ``find_latest_stage2_checkpoint`` owns the committed-marker filter; this
    # return value models its already-tested decision to ignore ``incomplete``.
    monkeypatch.setattr(
        stage2_checkpoint, "find_latest_stage2_checkpoint", lambda _root: child
    )
    manifests = {
        child.resolve(): {
            "manifest_sha256": "2" * 64,
            "provenance_sha256": provenance_hash,
        },
        anchor.resolve(): {
            "manifest_sha256": "1" * 64,
            "provenance_sha256": "e" * 64,
        },
    }
    monkeypatch.setattr(
        stage2_checkpoint,
        "validate_stage2_checkpoint",
        lambda path, **_expected: manifests[Path(path).resolve()],
    )

    assert trainer._discover_resume_checkpoint() == child.resolve()


def test_metrics_lineage_resume_refuses_existing_nonprefix_or_changed_snapshot(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint_stage2_g000240"
    checkpoint.mkdir()
    source = checkpoint / "metrics_lineage.jsonl"
    snapshot = b'{"record_type":"run_start","run_id":"parent"}\n'
    source.write_bytes(snapshot)
    entry = {
        "name": "metrics_lineage.jsonl",
        "size": len(snapshot),
        "sha256": hashlib.sha256(snapshot).hexdigest(),
    }
    payload = SimpleNamespace(
        directory=checkpoint,
        manifest={"files": [entry]},
    )
    destination = tmp_path / "child" / "metrics.jsonl"
    destination.parent.mkdir()
    destination.write_bytes(b'{"run_id":"unrelated"}\n')
    trainer = _bare_trainer(
        resolved=SimpleNamespace(jsonl_path=str(destination)),
        options=SimpleNamespace(output_dir=destination.parent),
    )

    with pytest.raises(RuntimeError, match="exact authenticated metrics prefix"):
        trainer._prepare_metrics_lineage(payload)

    destination.unlink()
    source.write_bytes(snapshot + b'{"tampered":true}\n')
    with pytest.raises(RuntimeError, match="changed after checkpoint load"):
        trainer._prepare_metrics_lineage(payload)


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
        trainer._audit_resume_runtime_bindings(payload)


def test_distributed_checkpoint_dataclass_drives_resume_state_sampler_loader_and_rng(
    monkeypatch, tmp_path
):
    resolved = load_stage2_config(CONFIG_PATH)
    schedule = Stage2TrainingSchedule.from_resolved_config(resolved)
    loader_states = {
        "generator": torch.Generator().manual_seed(101).get_state(),
        "fake_score": torch.Generator().manual_seed(202).get_state(),
    }
    sampler_state = _sampler_state(completed_f=10, completed_g=2)
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
        expected_phase_b_mode,
    ):
        load_calls.append(
            {
                "directory": directory,
                "expected_contract_hash": expected_contract_hash,
                "expected_world_size": expected_world_size,
                "expected_topology": expected_topology,
                "expected_phase_b_mode": expected_phase_b_mode,
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
            "expected_phase_b_mode": resolved.phase_b_mode,
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

    restore_rng = create_autospec(stage2_checkpoint.restore_stage2_rng_state)
    monkeypatch.setattr(stage2_checkpoint, "restore_stage2_rng_state", restore_rng)
    trainer._restore_rng_last(loaded)

    restore_rng.assert_called_once()
    call = restore_rng.call_args
    assert call.args == (local_rng_state,)
    assert call.kwargs["rank"] == 0
    assert call.kwargs["expected_world_size"] == 8
    assert call.kwargs["require_cuda_topology"] is True
    assert call.kwargs["dedicated_generators"] == {
        "generator_rollout": generator_rollout,
        "fake_score_rollout": fake_score_rollout,
        "generator_loader": generator_loader,
        "fake_score_loader": fake_score_loader,
    }
    assert call.kwargs["rank0_control_generators"] == {
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
    sampler_state = _sampler_state(completed_f=10, completed_g=2)
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
    generator_module = nn.Linear(1, 1)
    fake_score_module = nn.Linear(1, 1)
    real_score_module = nn.Linear(1, 1).requires_grad_(False)
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
        metrics_lineage_snapshot,
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
        milestone_updates,
        apply_retention,
    ):
        saver_calls.append(
            {
                "root": root,
                "trainer_state": trainer_state,
                "metrics_lineage_snapshot": metrics_lineage_snapshot,
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
                "milestone_updates": milestone_updates,
                "apply_retention": apply_retention,
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
    metric_path = tmp_path / "metrics" / "stage2.jsonl"
    metric_path.parent.mkdir()
    metric_path.write_bytes(b'{"record_type":"cycle_summary"}\n')
    trainer = _bare_trainer(
        state=state,
        schedule=schedule,
        resolved=resolved,
        options=SimpleNamespace(output_dir=tmp_path),
        logger=SimpleNamespace(run_id="checkpoint-run", next_attempt_index=41),
        metric_path=metric_path,
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
            real_score=real_score_module,
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
        _checkpoint_provenance=lambda: provenance,
    )

    event = trainer._save_checkpoint()

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
    assert saver["metrics_lineage_snapshot"] == metric_path.read_bytes()
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
    assert saver["milestone_updates"] == resolved.milestone_generator_updates
    assert saver["apply_retention"] is True

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
        "_run_metadata": lambda: {"kind": "tiny"},
    }
    rank0 = _bare_trainer(is_main_process=True, **common)
    rank1 = _bare_trainer(is_main_process=False, **common)

    rank0._build_logger(None)
    rank1._build_logger(None)

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

    def save_checkpoint():
        checkpoint_calls.append(True)
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
        _capture_next_f1_probe=lambda: {"schema": "test-next-f1-probe"},
        _save_checkpoint=save_checkpoint,
        _smoke_parent_probe_consumed=smoke_mode in {"C1", "C2"},
    )

    trainer._train_loop()

    assert executed == ["F1", "F2", "F3", "F4", "F5", "G"]
    assert state.cycle == starting_cycles + 1
    assert bool(checkpoint_calls) is expect_checkpoint
    assert [record_type for record_type, _fields in records] == (
        ["cycle_summary", "checkpoint_event"]
        if expect_checkpoint
        else ["cycle_summary"]
    )
    assert records[0][1]["dry_run"] is True
    smoke_acceptance = records[0][1]["smoke_acceptance"]
    assert smoke_acceptance["parent_next_f1_probe_consumed"] is (
        smoke_mode in {"C1", "C2"}
    )
    if smoke_mode in {"C0", "C1"}:
        assert smoke_acceptance["next_f1_probe"] == {"schema": "test-next-f1-probe"}
    else:
        assert "next_f1_probe" not in smoke_acceptance


def test_cpu_smoke_gate_requires_zero_nonfinite_attempts_in_cycle():
    trainer = _bare_trainer(
        options=SimpleNamespace(dry_run=True, smoke_mode="C0"),
        device=torch.device("cpu"),
    )

    probe = trainer._validate_smoke_cycle(
        [{} for _ in range(6)],
        nonfinite_attempts=0,
    )

    assert probe == {
        "smoke_mode": "C0",
        "status": "PASS_CPU_TEST_SEAM",
        "live_allocated_gib_max": 0.0,
        "nonfinite_attempts": 0,
    }
    with pytest.raises(RuntimeError, match=r"failures=\['nonfinite_attempts'\]"):
        trainer._validate_smoke_cycle(
            [{} for _ in range(6)],
            nonfinite_attempts=1,
        )


def test_train_loop_passes_one_nonfinite_delta_to_gate_and_cycle_summary():
    resolved = load_stage2_config(CONFIG_PATH)
    schedule = Stage2TrainingSchedule.from_resolved_config(resolved)
    state = Stage2TrainingState()
    gate_calls = []
    records = []
    executed = []

    def run_one_substep():
        substep = state.next_substep
        if not executed:
            state.record_nonfinite_attempt()
        executed.append(substep)
        if substep == "G":
            state.commit_successful_generator_update(
                ema_action=schedule.expected_ema_action(state.completed_g + 1),
                schedule=schedule,
            )
        else:
            state.commit_successful_fake_update(substep, schedule=schedule)
        return {"elapsed": 1.0}

    def validate(step_fields, *, nonfinite_attempts):
        gate_calls.append(
            {
                "steps": len(step_fields),
                "nonfinite_attempts": nonfinite_attempts,
                "next_substep": state.next_substep,
                "completed_cycles": state.cycle,
            }
        )
        return {"smoke_mode": "C2", "status": "TEST_GATE"}

    trainer = _bare_trainer(
        resolved=resolved,
        schedule=schedule,
        state=state,
        options=SimpleNamespace(smoke_mode="C2", dry_run=True, no_save=True),
        _append_metric=lambda record_type, fields: records.append(
            (record_type, dict(fields))
        ),
        _run_one_logical_substep=run_one_substep,
        _validate_smoke_cycle=validate,
        _smoke_parent_probe_consumed=True,
    )

    trainer._train_loop()

    assert gate_calls == [
        {
            "steps": 6,
            "nonfinite_attempts": 1,
            "next_substep": "F1",
            "completed_cycles": 1,
        }
    ]
    assert records[0][0] == "cycle_summary"
    assert records[0][1]["nonfinite_attempts"] == 1


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
    assert result["attempt_orchestration_seconds"] >= 0.0
    assert result["attempt_seconds"] == pytest.approx(
        sum(result["phase_timings"].values())
        + result["optimizer_seconds"]
        + result["attempt_orchestration_seconds"]
    )


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


def test_production_nonfinite_retry_clears_partial_backward_and_replays_every_stream(
    monkeypatch,
):
    """Exercise Trainer's real snapshot/attempt/restore path after one backward."""

    from pipeline.stage2_rollout import Stage2ExitRNGStreams

    resolved = load_stage2_config(CONFIG_PATH)
    schedule = Stage2TrainingSchedule.from_resolved_config(resolved)
    state = Stage2TrainingState()
    for substep in ("F1", "F2", "F3", "F4", "F5"):
        state.commit_successful_fake_update(substep, schedule=schedule)
    assert state.next_substep == "G"

    class ReplaySamplers:
        def __init__(self):
            self.generator_cursor = 0

        def draw_generator_batch(self):
            value = self.generator_cursor
            self.generator_cursor += 1
            return value

        def state_dict(self):
            return {"generator_cursor": self.generator_cursor}

        def load_state_dict(self, value):
            self.generator_cursor = int(value["generator_cursor"])

    parameter = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
    generator_role = nn.Module()
    generator_role.register_parameter("lora_weight", parameter)
    frozen_role = nn.Linear(1, 1).requires_grad_(False)

    class CountingOptimizer:
        def __init__(self):
            self.step_calls = 0
            self.zero_grad_calls = 0

        def zero_grad(self, *, set_to_none):
            assert set_to_none is True
            self.zero_grad_calls += 1
            parameter.grad = None

        def step(self):
            self.step_calls += 1
            assert parameter.grad is not None
            with torch.no_grad():
                parameter.add_(parameter.grad, alpha=-0.05)

    optimizer = CountingOptimizer()
    ema_calls = []
    generator_ema = SimpleNamespace(
        update_after_step=lambda _module, completed_g: ema_calls.append(completed_g)
        or schedule.expected_ema_action(completed_g)
    )
    samplers = ReplaySamplers()
    dataloader_generators = {
        role: torch.Generator(device="cpu").manual_seed(seed)
        for role, seed in (("generator", 101), ("fake_score", 102))
    }
    dedicated_generators = {
        f"{role}_{kind}": torch.Generator(device="cpu").manual_seed(seed)
        for role, base in (("generator", 200), ("fake_score", 300))
        for kind, seed in zip(("rollout", "timestep", "noise"), range(base, base + 3))
    }
    exit_rng = Stage2ExitRNGStreams(401)
    branch_rng = torch.Generator(device="cpu").manual_seed(402)
    initial_branch_state = branch_rng.get_state().clone()
    expected_branch_rng = torch.Generator(device="cpu")
    expected_branch_rng.set_state(initial_branch_state.clone())
    torch.rand((), generator=expected_branch_rng)
    expected_branch_state_after_one_draw = expected_branch_rng.get_state().clone()

    random.seed(501)
    np.random.seed(502)
    torch.manual_seed(503)
    materializations = []
    micro_draws = []
    partial_gradient_before_failure = []
    metric_records = []

    def materialize(role):
        assert role == "generator"
        batch_id = samplers.draw_generator_batch()
        loader_draws = torch.rand(
            resolved.gradient_accumulation_steps,
            generator=dataloader_generators[role],
        )
        record = (batch_id, tuple(float(value) for value in loader_draws))
        materializations.append(record)
        return [
            {
                "sample_id": torch.tensor([batch_id * 10 + index]),
                "loader_draw": loader_draws[index].clone(),
            }
            for index in range(resolved.gradient_accumulation_steps)
        ]

    def compute_micro_loss(*, role, batch, exit_step, branch):
        assert role == "generator"
        call_index = len(micro_draws)
        if call_index == 1:
            assert parameter.grad is not None
            partial_gradient_before_failure.append(parameter.grad.detach().clone())
        record = {
            "sample_id": int(batch["sample_id"].item()),
            "loader_draw": float(batch["loader_draw"].item()),
            "exit_step": int(exit_step),
            "branch": branch,
            "python": random.random(),
            "numpy": float(np.random.random()),
            "torch_default": float(torch.rand(()).item()),
            "rollout": float(
                torch.rand((), generator=dedicated_generators["generator_rollout"])
            ),
            "timestep": float(
                torch.rand((), generator=dedicated_generators["generator_timestep"])
            ),
            "noise": float(
                torch.rand((), generator=dedicated_generators["generator_noise"])
            ),
        }
        micro_draws.append(record)
        numerator = parameter.square()
        output = SimpleNamespace(
            numerator=numerator,
            count=1,
            loss=(
                torch.full((), float("nan"), dtype=torch.float32)
                if call_index == 1
                else numerator
            ),
        )
        return {
            "output": output,
            "diagnostic": {"finite_metric": 1.0},
            "sample_count": 1,
            "rollout_result": SimpleNamespace(
                latents=torch.zeros(1),
                cache_audit={
                    "generator_forward_calls": 1,
                    "logical_query_tokens": 390,
                },
            ),
            "score_calls": 3,
            "fake_score_calls": 1,
            "real_score_calls": 2,
            "frame_time": torch.tensor([[0.0, 100.0]]),
            "phase_timings": {},
        }

    def append_metric(record_type, fields):
        metric_records.append((record_type, dict(fields)))
        if record_type == "nonfinite_attempt":
            assert parameter.grad is None
            assert parameter.detach().item() == 1.0
            assert optimizer.step_calls == 0
            assert ema_calls == []
            assert state.completed_f == 5
            assert state.completed_g == 0
            assert state.cycle == 0
            assert state.next_substep == "G"
            assert samplers.generator_cursor == 0
            assert torch.equal(branch_rng.get_state(), initial_branch_state)

    timing = {
        "step_seconds_max": 1.0,
        "step_seconds_mean": 1.0,
        "straggler_ratio": 1.0,
        "data_seconds_max": 0.1,
        "h2d_seconds_max": 0.1,
        "rollout_seconds_max": 0.1,
        "fake_score_seconds_max": 0.1,
        "real_cond_seconds_max": 0.1,
        "real_uncond_seconds_max": 0.1,
        "loss_build_seconds_max": 0.1,
        "backward_seconds_max": 0.1,
        "clip_optimizer_seconds_max": 0.1,
        "compute_seconds_max": 0.6,
        "optimizer_seconds_max": 0.1,
        "ema_seconds_max": 0.0,
        "timing_closure_error_seconds": 0.0,
    }
    trainer = _bare_trainer(
        state=state,
        schedule=schedule,
        resolved=resolved,
        options=SimpleNamespace(smoke_mode=None, dry_run=False),
        device=torch.device("cpu"),
        world_size=8,
        rank=0,
        is_main_process=True,
        logger=SimpleNamespace(append=append_metric),
        model=SimpleNamespace(
            generator=generator_role,
            fake_score=frozen_role,
            real_score=frozen_role,
        ),
        optimizers={"generator": optimizer},
        generator_ema=generator_ema,
        samplers=samplers,
        dataloader_generators=dataloader_generators,
        dedicated_generators=dedicated_generators,
        exit_rng=exit_rng,
        branch_rng=branch_rng,
        _materialize_batches=materialize,
        _to_device=lambda batch: batch,
        _compute_micro_loss=compute_micro_loss,
        _world_consensus=lambda condition: bool(condition),
        _reduce_loss=lambda numerator, count: (float(numerator), count),
        _timing_summary=lambda *_args, **_kwargs: timing,
        _memory_fields=lambda: {
            "gpu_memory_allocated_gib_max": 1.0,
            "gpu_memory_reserved_gib_max": 2.0,
            "gpu_memory_free_gib_min": 70.0,
            "gpu_memory_total_gib_min": 80.0,
            "gpu_step_seconds_mean": 1.0,
            "gpu_step_seconds_max": 1.0,
        },
    )
    monkeypatch.setattr(
        distributed_utils, "fsdp2_accumulation", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(
        trainer_module.dist, "broadcast", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        torch.cuda, "reset_peak_memory_stats", lambda *_args, **_kwargs: None
    )

    result = trainer._run_one_logical_substep()

    assert partial_gradient_before_failure
    assert bool(torch.isfinite(partial_gradient_before_failure[0]).all())
    assert materializations[0] == materializations[1]
    assert micro_draws[0] == micro_draws[2]
    assert micro_draws[1] == micro_draws[3]
    assert [name for name, _fields in metric_records] == [
        "nonfinite_attempt",
        "train_step",
    ]
    assert optimizer.step_calls == 1
    assert ema_calls == [1]
    assert state.nonfinite_attempts_by_role == {"generator": 1, "fake_score": 0}
    assert state.completed_f == 5
    assert state.completed_g == 1
    assert state.cycle == 1
    assert state.next_substep == "F1"
    assert result["fields"]["phase"] == "A"
    assert torch.equal(branch_rng.get_state(), expected_branch_state_after_one_draw)


def test_checkpoint_live_quiescence_rejects_active_substep_gradients_and_runtime():
    roles = {
        role: nn.Linear(1, 1).requires_grad_(role != "real_score")
        for role in ("generator", "real_score", "fake_score")
    }
    trainer = _bare_trainer(
        model=SimpleNamespace(**roles),
        _active_logical_substep=None,
    )
    assert trainer._assert_live_checkpoint_quiescence() == {
        "pending_gradients": False,
        "pending_batch": False,
        "pending_branch": False,
        "pending_rollout_kv": False,
        "pending_transaction": False,
    }

    trainer._active_logical_substep = {"cycle_substep": "G"}
    with pytest.raises(RuntimeError, match="logical substep is active"):
        trainer._assert_live_checkpoint_quiescence()
    trainer._active_logical_substep = None

    roles["generator"].weight.grad = torch.ones_like(roles["generator"].weight)
    with pytest.raises(RuntimeError, match="retained parameter gradients"):
        trainer._assert_live_checkpoint_quiescence()
    roles["generator"].weight.grad = None

    trainer.pending_rollout_state = object()
    with pytest.raises(RuntimeError, match="attempt-local runtime state"):
        trainer._assert_live_checkpoint_quiescence()


def test_tracked_logical_substep_clears_the_live_marker_on_failure():
    state = SimpleNamespace(
        current_role="fake_score", next_substep="F1", successful_attempts=0
    )
    trainer = _bare_trainer(state=state, _active_logical_substep=None)

    def fail_inside_substep():
        assert trainer._active_logical_substep == {
            "role": "fake_score",
            "cycle_substep": "F1",
            "logical_substep_id": 0,
        }
        raise FloatingPointError("stop")

    trainer._run_one_logical_substep = fail_inside_substep
    with pytest.raises(FloatingPointError, match="stop"):
        trainer._run_tracked_logical_substep()
    assert trainer._active_logical_substep is None


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
    captured_timing_categories = []
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
            "attempt_orchestration_seconds": 0.3,
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
        _timing_summary=lambda _elapsed, categories: captured_timing_categories.append(
            dict(categories)
        )
        or timing,
        _memory_fields=lambda: {
            "gpu_memory_allocated_gib_max": 1.0,
            "gpu_memory_reserved_gib_max": 2.0,
        },
        _smoke_parent_probe_consumed=True,
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
    assert captured_timing_categories[0]["orchestration"] >= 0.3


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
                    0.15,
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
        "orchestration_seconds_max",
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
