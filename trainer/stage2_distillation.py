"""Stage-2-only 5F -> 1G DMD/DFD trainer.

This orchestrator deliberately owns no diffusion math.  It wires the audited
F25 dataset, independent G/F samplers, random-exit rollout, the explicit
``Stage2DMD`` role methods, two LoRA-only optimizers, CPU EMA, JSONL metrics,
and cycle-boundary checkpoints into the one legal state transition sequence.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import random
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class _TrainerOptions:
    output_dir: Path
    no_save: bool
    no_visualize: bool
    auto_resume: bool
    smoke_mode: str | None

    @property
    def dry_run(self) -> bool:
        return self.smoke_mode is not None


def _local_tensor(value: torch.Tensor) -> torch.Tensor:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_STAGE2_DMD_RUNTIME_API_VERSION = "longlive_stage2_dmd_runtime/v2"
_STAGE2_DMD_RUNTIME_REPAIR = (
    "Run scripts/apply_stage2_innernet_hotfix.py from the project root, or "
    "deploy one complete stage-2 source snapshot; do not mix trainer and model files."
)
_STAGE2_TIMING_RUNTIME_FIELDS = (
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
    "ema_seconds_max",
)
_STAGE2_DMD_RUNTIME_METHODS = {
    "fake_score_flow_dsm_loss_from_model": (
        "generated_future",
        "noised_fake_score",
        "conditional_dict",
        "timing_callback",
    ),
    "generator_distribution_matching_loss_from_models": (
        "branch",
        "generated_future",
        "noised_score",
        "conditional_dict",
        "real_unconditional_dict",
        "timing_callback",
    ),
}


def _audit_stage2_dmd_runtime_api(model_type: type) -> dict[str, Any]:
    """Fail before training when trainer and Stage2DMD source are out of sync."""

    if not isinstance(model_type, type):
        raise TypeError("Stage-2 DMD runtime API audit requires a model type")
    source_file = inspect.getsourcefile(model_type) or "<unknown>"
    actual_version = getattr(model_type, "RUNTIME_API_VERSION", None)
    if actual_version != _STAGE2_DMD_RUNTIME_API_VERSION:
        raise RuntimeError(
            "Stage-2 DMD runtime API version mismatch: "
            f"expected={_STAGE2_DMD_RUNTIME_API_VERSION!r}, "
            f"actual={actual_version!r}, source={source_file}. "
            f"{_STAGE2_DMD_RUNTIME_REPAIR}"
        )

    methods: dict[str, str] = {}
    for method_name, expected_names in _STAGE2_DMD_RUNTIME_METHODS.items():
        method = getattr(model_type, method_name, None)
        if not callable(method):
            raise RuntimeError(
                f"Stage-2 DMD runtime API lacks callable {method_name}; "
                f"source={source_file}"
            )
        signature = inspect.signature(method)
        parameters = tuple(signature.parameters.values())
        if not parameters or parameters[0].name != "self":
            raise RuntimeError(
                f"Stage-2 DMD runtime API mismatch for {method_name}: "
                f"missing self parameter; source={source_file}"
            )
        exposed = parameters[1:]
        actual_names = tuple(parameter.name for parameter in exposed)
        if actual_names != expected_names:
            missing = [name for name in expected_names if name not in actual_names]
            unexpected = [name for name in actual_names if name not in expected_names]
            raise RuntimeError(
                f"Stage-2 DMD runtime API mismatch for {method_name}: "
                f"missing={missing}, unexpected={unexpected}, "
                f"expected_order={list(expected_names)}, "
                f"actual_order={list(actual_names)}, source={source_file}. "
                f"{_STAGE2_DMD_RUNTIME_REPAIR}"
            )
        invalid_kinds = [
            parameter.name
            for parameter in exposed
            if parameter.kind is not inspect.Parameter.KEYWORD_ONLY
        ]
        invalid_defaults = [
            parameter.name
            for parameter in exposed
            if (parameter.name == "timing_callback" and parameter.default is not None)
            or (
                parameter.name != "timing_callback"
                and parameter.default is not inspect.Parameter.empty
            )
        ]
        if invalid_kinds or invalid_defaults:
            raise RuntimeError(
                f"Stage-2 DMD runtime API mismatch for {method_name}: "
                f"non_keyword_only={invalid_kinds}, "
                f"invalid_defaults={invalid_defaults}, source={source_file}"
            )
        methods[method_name] = str(signature)
    from utils.stage2_metrics import STAGE2_TIMING_FIELDS

    actual_timing_fields = tuple(STAGE2_TIMING_FIELDS)
    if actual_timing_fields != _STAGE2_TIMING_RUNTIME_FIELDS:
        raise RuntimeError(
            "Stage-2 timing runtime API mismatch: "
            f"expected={list(_STAGE2_TIMING_RUNTIME_FIELDS)}, "
            f"actual={list(actual_timing_fields)}. {_STAGE2_DMD_RUNTIME_REPAIR}"
        )
    return {
        "api_version": actual_version,
        "model_type": f"{model_type.__module__}.{model_type.__qualname__}",
        "source_file": str(Path(source_file).expanduser().resolve()),
        "methods": methods,
        "timing_fields": actual_timing_fields,
    }


class Trainer:
    """Production Stage-2 trainer; construction stays CPU/light until ``train``."""

    def __init__(
        self,
        resolved_config: Any,
        *,
        output_dir: str | os.PathLike[str],
        no_save: bool = False,
        no_visualize: bool = False,
        auto_resume: bool = True,
        smoke_mode: str | None = None,
    ) -> None:
        if getattr(resolved_config, "trainer", None) != "stage2_distillation":
            raise ValueError(
                "Stage-2 Trainer requires the strict Stage2ResolvedConfig."
            )
        from model.stage2_dmd import Stage2DMD

        self.model_runtime_api_audit = _audit_stage2_dmd_runtime_api(Stage2DMD)
        if not output_dir:
            raise ValueError("Stage-2 requires an explicit --logdir.")
        if smoke_mode not in {None, "C0", "C1", "C2"}:
            raise ValueError("Stage-2 smoke_mode must be one of C0/C1/C2 or None.")
        self.resolved = resolved_config
        from utils.stage2_train_state import Stage2TrainingSchedule

        self.schedule = Stage2TrainingSchedule.from_resolved_config(resolved_config)
        self.options = _TrainerOptions(
            output_dir=Path(output_dir).expanduser().resolve(),
            no_save=bool(no_save),
            no_visualize=bool(no_visualize),
            auto_resume=bool(auto_resume),
            smoke_mode=smoke_mode,
        )
        if self.resolved.torch_compile:
            raise ValueError(
                "Stage-2 torch.compile remains a post-eager profiling candidate; "
                "the production trainer refuses to enable it before H100 parity."
            )
        self._initialized_process_group = False
        self.logger = None
        self.resume_payload = None
        self._last_cycle_smoke_probe = None
        self._smoke_parent_probe_consumed = False
        self._active_logical_substep = None

    def _initialize_distributed(self) -> None:
        if not dist.is_available():
            raise RuntimeError("torch.distributed is unavailable.")
        if not dist.is_initialized():
            if "LOCAL_RANK" not in os.environ:
                raise RuntimeError("Stage-2 training must run under torchrun.")
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            dist.init_process_group("nccl", timeout=timedelta(minutes=60))
            self._initialized_process_group = True
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
        self.device = torch.device("cuda", self.local_rank)
        self.is_main_process = self.rank == 0
        if self.world_size != self.resolved.expected_world_size:
            raise RuntimeError(
                f"Stage-2 requires WORLD_SIZE={self.resolved.expected_world_size}, "
                f"got {self.world_size}."
            )

    def _world_consensus(self, condition: bool) -> bool:
        value = torch.tensor(
            [1 if condition else 0], dtype=torch.uint8, device=self.device
        )
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
        return bool(value.item())

    @staticmethod
    def _is_retryable_nonfinite_error(error: BaseException) -> bool:
        message = str(error).lower()
        return isinstance(error, (ValueError, FloatingPointError)) and (
            "finite" in message or "nan" in message
        )

    def _align_pre_score_failure(
        self, label: str, local_error: BaseException | None
    ) -> None:
        """Make every rank take the same branch before the next FSDP forward.

        Rollout output validation and score-noising are local CUDA/tensor work.
        A single-rank error must be resolved collectively *before* healthy
        ranks enter the bidirectional score model, otherwise collective order
        diverges and the retry path can hang.  A finite/NaN error remains an
        exact-replay candidate; contract/programming errors fail the job on all
        ranks together.
        """

        if self._world_consensus(local_error is None):
            return
        retryable_here = local_error is None or self._is_retryable_nonfinite_error(
            local_error
        )
        if self._world_consensus(retryable_here):
            raise ValueError(f"non-finite {label} on one or more ranks")
        raise RuntimeError(
            f"Stage-2 {label} failed on one or more ranks before the aligned "
            "score forward."
        )

    def _world_checked(self, label: str, callback):
        value = None
        error = None
        try:
            value = callback()
        except Exception as exc:  # turn rank-local errors into one visible failure
            error = f"{type(exc).__name__}: {exc}"
        statuses: list[dict[str, Any] | None] = [None] * self.world_size
        dist.all_gather_object(
            statuses, {"rank": self.rank, "label": label, "error": error}
        )
        failures = [
            f"rank{item['rank']}: {item['error']}"
            for item in statuses
            if item is not None and item["error"] is not None
        ]
        if failures or any(item is None for item in statuses):
            raise RuntimeError(f"Stage-2 WORLD stage failed ({label}): {failures}")
        return value

    def _rank0_checked(self, label: str, callback):
        holder: list[Any] = [None]
        if self.is_main_process:
            try:
                holder[0] = {"ok": True, "value": callback()}
            except Exception as exc:
                holder[0] = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        dist.broadcast_object_list(holder, src=0)
        result = holder[0]
        if not isinstance(result, Mapping) or not result.get("ok"):
            raise RuntimeError(
                f"Stage-2 rank-0 stage failed ({label}): "
                f"{result.get('error') if isinstance(result, Mapping) else result}"
            )
        return result.get("value")

    def _runtime_world_checked(self, label: str, callback):
        if dist.is_available() and dist.is_initialized():
            return self._world_checked(label, callback)
        return callback()

    def _runtime_rank0_checked(self, label: str, callback):
        if dist.is_available() and dist.is_initialized():
            return self._rank0_checked(label, callback)
        return callback() if self.is_main_process else None

    def _append_metric(self, record_type: str, fields: Mapping[str, Any]) -> int | None:
        return self._runtime_rank0_checked(
            f"JSONL {record_type}",
            lambda: self.logger.append(record_type, fields),
        )

    def _seed_training_rngs(self) -> None:
        seed = int(self.resolved.training_seed) + self.rank * 100_003
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        self.dataloader_generators = {}
        self.dedicated_generators = {}
        offsets = {
            "fake_score_loader": 10_001,
            "generator_loader": 10_002,
            "fake_score_rollout": 20_001,
            "generator_rollout": 20_002,
            "fake_score_timestep": 30_001,
            "generator_timestep": 30_002,
            "fake_score_noise": 40_001,
            "generator_noise": 40_002,
        }
        for role in ("fake_score", "generator"):
            loader = torch.Generator(device="cpu")
            loader.manual_seed(seed + offsets[f"{role}_loader"])
            self.dataloader_generators[role] = loader
            for kind in ("rollout", "timestep", "noise"):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed + offsets[f"{role}_{kind}"])
                self.dedicated_generators[f"{role}_{kind}"] = generator

        from pipeline.stage2_rollout import Stage2ExitRNGStreams

        self.exit_rng = Stage2ExitRNGStreams(int(self.resolved.training_seed))
        self.branch_rng = torch.Generator(device="cpu")
        self.branch_rng.manual_seed(int(self.resolved.training_seed) + 0x444644)

    def _discover_resume_checkpoint(self):
        from utils.stage2_checkpoint import find_latest_stage2_checkpoint

        explicit = self.resolved.resume_stage2_checkpoint
        if self.options.smoke_mode == "C0" and explicit is not None:
            raise RuntimeError(
                "Stage-2 smoke C0 must start from the cold initialization."
            )
        discovered = (
            find_latest_stage2_checkpoint(self.options.output_dir)
            if self.options.auto_resume and self.options.output_dir.exists()
            else None
        )
        if explicit is not None:
            explicit_candidate = Path(explicit).expanduser()
            if explicit_candidate.is_symlink():
                raise RuntimeError(
                    "Explicit Stage-2 resume checkpoint cannot be a symlink: "
                    f"{explicit_candidate}"
                )
            explicit_path = explicit_candidate.resolve()
            if discovered is None:
                selected = explicit_path
            else:
                discovered_path = Path(discovered).resolve()
                if explicit_path != discovered_path:
                    self._assert_checkpoint_descends_from_anchor(
                        discovered_path,
                        explicit_path,
                    )
                selected = discovered_path
        else:
            selected = Path(discovered).resolve() if discovered is not None else None
        if self.options.smoke_mode in {"C1", "C2"} and selected is None:
            raise RuntimeError(
                f"Stage-2 smoke {self.options.smoke_mode} requires a complete "
                "checkpoint from the preceding smoke cycle."
            )
        return selected

    def _assert_checkpoint_descends_from_anchor(
        self,
        descendant: Path,
        anchor: Path,
    ) -> None:
        """Prove an auto-resume child reaches one exact explicit parent.

        ``checkpoints.resume_stage2`` is an immutable ancestry anchor for the
        matched B0/B1 fork, not a request to keep reloading A24 forever.  Every
        edge is bound by both the canonical parent path and the parent's
        self-hashed manifest.  Intermediate children must remain in this
        launch's output directory; an unrelated local run therefore cannot be
        silently preferred merely because it has a larger G clock.
        """

        from utils.stage1_io import canonical_json_sha256
        from utils.stage2_checkpoint import validate_stage2_checkpoint

        output_root = self.options.output_dir.resolve()
        cursor = descendant.resolve()
        anchor = anchor.resolve()
        if cursor.parent != output_root:
            raise RuntimeError(
                "Newest Stage-2 checkpoint is outside --logdir and cannot be "
                f"proved as a local child: latest={cursor}, logdir={output_root}."
            )

        expected_manifest_sha256: str | None = None
        visited: set[Path] = set()
        while True:
            if cursor in visited:
                raise RuntimeError(
                    f"Stage-2 checkpoint ancestry contains a cycle at {cursor}."
                )
            visited.add(cursor)
            if cursor != anchor and cursor.parent != output_root:
                raise RuntimeError(
                    "Stage-2 checkpoint ancestry left --logdir before reaching "
                    f"the explicit anchor: checkpoint={cursor}, anchor={anchor}."
                )

            manifest = validate_stage2_checkpoint(
                cursor,
                expected_contract_hash=self.resolved.contract_hash(),
                expected_topology=self._checkpoint_topology(),
                expected_phase_b_mode=self.resolved.phase_b_mode,
            )
            actual_manifest_sha256 = manifest["manifest_sha256"]
            if (
                expected_manifest_sha256 is not None
                and actual_manifest_sha256 != expected_manifest_sha256
            ):
                raise RuntimeError(
                    "Stage-2 checkpoint ancestry parent manifest hash drifted: "
                    f"checkpoint={cursor}, expected={expected_manifest_sha256}, "
                    f"actual={actual_manifest_sha256}."
                )
            if cursor == anchor:
                return

            provenance_path = cursor / "provenance.json"
            if provenance_path.is_symlink() or not provenance_path.is_file():
                raise RuntimeError(
                    "Stage-2 checkpoint ancestry has no regular provenance: "
                    f"{provenance_path}."
                )
            with provenance_path.open("r", encoding="utf-8") as handle:
                provenance = json.load(handle)
            if (
                not isinstance(provenance, Mapping)
                or canonical_json_sha256(dict(provenance))
                != manifest["provenance_sha256"]
            ):
                raise RuntimeError(
                    "Stage-2 checkpoint ancestry provenance changed after "
                    f"validation: {cursor}."
                )
            lineage = provenance.get("lineage")
            if not isinstance(lineage, Mapping):
                raise RuntimeError(
                    f"Stage-2 checkpoint ancestry is missing lineage: {cursor}."
                )
            parent_value = lineage.get("parent_checkpoint")
            parent_manifest_sha256 = lineage.get("parent_checkpoint_manifest_sha256")
            if not isinstance(parent_value, str) or not parent_value:
                raise RuntimeError(
                    "Stage-2 local checkpoint does not descend from the explicit "
                    f"anchor: child={cursor}, anchor={anchor}."
                )
            parent_candidate = Path(parent_value).expanduser()
            parent_path = parent_candidate.resolve()
            if (
                not parent_candidate.is_absolute()
                or parent_candidate != parent_path
                or parent_candidate.is_symlink()
            ):
                raise RuntimeError(
                    "Stage-2 checkpoint ancestry parent path is not one canonical "
                    f"absolute directory: {parent_value!r}."
                )
            if (
                not isinstance(parent_manifest_sha256, str)
                or len(parent_manifest_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in parent_manifest_sha256
                )
            ):
                raise RuntimeError(
                    "Stage-2 checkpoint ancestry parent manifest hash is invalid: "
                    f"{cursor}."
                )
            expected_manifest_sha256 = parent_manifest_sha256
            cursor = parent_path

    def _build_data_runtime(self) -> None:
        from utils.stage2_i2v_data import Stage2I2VCacheDataset
        from utils.stage2_sampler import build_stage2_role_samplers

        self.dataset = Stage2I2VCacheDataset(
            self.resolved.cache_dir,
            metadata_path=self.resolved.metadata_path,
            source_cache_manifest_path=self.resolved.source_cache_manifest,
            negative_conditioning_manifest_path=(
                self.resolved.negative_conditioning_manifest
            ),
            config_contract_sha256=self.resolved.contract_hash(),
            config_launch_sha256=self.cache_audit_launch_hash,
            expected_num_samples=self.resolved.expected_num_samples,
        )
        self.samplers = build_stage2_role_samplers(
            self.dataset.action_ids,
            self.dataset.spatial_shapes,
            action_order=self.dataset.action_order,
            base_seed=self.resolved.training_seed,
            microbatch_size_per_device=self.resolved.microbatch_size_per_device,
        )

    def _audit_resume_runtime_bindings(self, resume_payload: Any) -> None:
        if resume_payload is None:
            return
        provenance = resume_payload.provenance
        expected_data = {
            "stage2_manifest_sha256": self.dataset.manifest["manifest_sha256"],
            "source_manifest_sha256": self.dataset.source_manifest["manifest_sha256"],
            "negative_manifest_sha256": self.dataset.negative_conditioning["manifest"][
                "manifest_sha256"
            ],
            "negative_artifact_sha256": self.dataset.negative_conditioning["manifest"][
                "artifact"
            ]["sha256"],
        }
        if provenance.get("data") != expected_data:
            raise RuntimeError(
                "Stage-2 resume checkpoint data/negative provenance differs from "
                "the audited runtime cache."
            )
        smoke_probe = provenance.get("smoke_probe")
        if self.options.smoke_mode is None and smoke_probe is not None:
            raise RuntimeError(
                "Formal Stage-2 training refuses to resume a C0/C1 smoke "
                "checkpoint. Use a separate formal --logdir and start from the "
                "audited cold initialization or a formal checkpoint."
            )
        expected_parent_mode = {"C1": "C0", "C2": "C1"}.get(self.options.smoke_mode)
        if expected_parent_mode is not None:
            if (
                not isinstance(smoke_probe, Mapping)
                or smoke_probe.get("smoke_mode") != expected_parent_mode
            ):
                raise RuntimeError(
                    f"Stage-2 smoke {self.options.smoke_mode} must resume the "
                    f"preceding {expected_parent_mode} checkpoint."
                )

    def _load_resume_before_roles(self):
        if self.resume_checkpoint is None:
            return None
        from utils.stage2_checkpoint import load_stage2_checkpoint_collective

        return load_stage2_checkpoint_collective(
            self.resume_checkpoint,
            expected_contract_hash=self.resolved.contract_hash(),
            expected_world_size=self.world_size,
            expected_topology=self._checkpoint_topology(),
            expected_phase_b_mode=self.resolved.phase_b_mode,
        )

    def _checkpoint_topology(self) -> dict[str, Any]:
        return {
            "world_size": self.world_size,
            "nodes": 1,
            "fsdp_backend": self.resolved.fsdp_backend,
            "sharding_strategy": "FULL_SHARD",
            "mesh_shape": [self.world_size],
            "mesh_dim_names": ["shard"],
            "rank_layout": list(range(self.world_size)),
            "microbatch_size_per_device": self.resolved.microbatch_size_per_device,
            "gradient_accumulation_steps": self.resolved.gradient_accumulation_steps,
            "global_batch_size": self.resolved.global_batch_size,
        }

    def _initialize_roles(self, mesh: Any, resume_payload: Any) -> None:
        from utils.stage2_role_init import initialize_stage2_roles
        from utils.stage2_role_manifest import audit_stage2_init_assets

        if resume_payload is None:
            assets = self._rank0_checked(
                "immutable role assets", lambda: audit_stage2_init_assets(self.resolved)
            )
            adapter_states = None
        else:
            assets = resume_payload.provenance["assets"]
            adapter_states = {
                "generator": resume_payload.generator_raw,
                "fake_score": resume_payload.fake_score_raw,
            }
        initialized = initialize_stage2_roles(
            self.resolved,
            assets=assets,
            mesh=mesh,
            stage_runner=self._world_checked,
            is_main_process=self.is_main_process,
            resume_adapter_states=adapter_states,
        )
        self.assets = assets
        self.model = initialized.model
        actual_runtime_api = _audit_stage2_dmd_runtime_api(type(self.model))
        if actual_runtime_api != self.model_runtime_api_audit:
            raise RuntimeError(
                "Stage-2 DMD runtime API changed during role initialization"
            )
        self.role_audits = initialized.role_audits
        self.lora_schemas = initialized.lora_schemas

    @staticmethod
    def _optimizer(module: torch.nn.Module, spec: Any) -> torch.optim.AdamW:
        if spec.optimizer_type != "adamw" or spec.schedule != "constant":
            raise ValueError(f"unsupported Stage-2 optimizer spec: {spec}")
        parameters = [
            parameter for parameter in module.parameters() if parameter.requires_grad
        ]
        if not parameters or any(
            parameter.dtype != torch.float32 for parameter in parameters
        ):
            raise TypeError("Stage-2 optimizers require non-empty FP32 LoRA masters.")
        return torch.optim.AdamW(
            parameters,
            lr=spec.learning_rate,
            betas=tuple(spec.betas),
            eps=spec.eps,
            weight_decay=spec.weight_decay,
        )

    def _build_optimizers_and_ema(self, resume_payload: Any) -> None:
        from utils.distributed import TrainableShardedEMA
        from utils.stage2_checkpoint import audit_stage2_lora_optimizer

        self.optimizers = {
            "generator": self._optimizer(
                self.model.generator, self.resolved.generator_optimizer
            ),
            "fake_score": self._optimizer(
                self.model.fake_score, self.resolved.fake_score_optimizer
            ),
        }
        for role in ("generator", "fake_score"):
            audit_stage2_lora_optimizer(
                getattr(self.model, role),
                self.optimizers[role],
                role=role,
                expected_schema=self.lora_schemas[role],
                expected_optimizer_spec=(
                    self.resolved.generator_optimizer
                    if role == "generator"
                    else self.resolved.fake_score_optimizer
                ),
            )
        self.generator_ema = TrainableShardedEMA(
            self.model.generator,
            decay=self.resolved.ema_decay,
            start_step=self.resolved.ema_initialize_at_completed_generator_update,
            expected_parameter_names=tuple(
                spec.raw_parameter_name
                for spec in self.lora_schemas["generator"].values()
            ),
            topology={
                "rank_layout": tuple(range(self.world_size)),
                "mesh_dim_names": ("shard",),
            },
        )
        if resume_payload is not None:
            from utils.stage2_checkpoint import restore_stage2_optimizer_state

            restore_stage2_optimizer_state(
                module=self.model.generator,
                optimizer=self.optimizers["generator"],
                optimizer_state=resume_payload.generator_optimizer_state_rank0,
                role="generator",
                expected_schema=self.lora_schemas["generator"],
                expected_completed_updates=resume_payload.trainer_state[
                    "completed_generator_updates"
                ],
                expected_optimizer_spec=self.resolved.generator_optimizer,
            )
            restore_stage2_optimizer_state(
                module=self.model.fake_score,
                optimizer=self.optimizers["fake_score"],
                optimizer_state=resume_payload.fake_score_optimizer_state_rank0,
                role="fake_score",
                expected_schema=self.lora_schemas["fake_score"],
                expected_completed_updates=resume_payload.trainer_state[
                    "completed_fake_updates"
                ],
                expected_optimizer_spec=self.resolved.fake_score_optimizer,
            )
            self.generator_ema.load_state_dict(
                resume_payload.local_ema_state, self.model.generator
            )

    def _build_rollout(self) -> None:
        from pipeline.stage2_rollout import Stage2RolloutPipeline

        self.rollout = Stage2RolloutPipeline.from_resolved_config(
            self.model.generator, self.resolved
        )

    def _materialize_batches(self, role: str) -> list[dict[str, Any]]:
        from utils.stage2_i2v_data import stage2_i2v_cache_collate
        from utils.stage2_sampler import partition_stage2_global_batch

        sampler = getattr(self.samplers, role)
        global_batch = sampler.next_global_batch()
        local_microbatches = partition_stage2_global_batch(
            global_batch,
            rank=self.rank,
            world_size=self.world_size,
            microbatch_size_per_device=self.resolved.microbatch_size_per_device,
            gradient_accumulation_steps=self.resolved.gradient_accumulation_steps,
            spatial_shapes=self.dataset.spatial_shapes,
        )
        loader_kwargs: dict[str, Any] = {
            "dataset": self.dataset,
            "batch_sampler": local_microbatches,
            "num_workers": self.resolved.num_workers,
            "pin_memory": False,
            "persistent_workers": False,
            "collate_fn": stage2_i2v_cache_collate,
            "generator": self.dataloader_generators[role],
        }
        if self.resolved.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 1
        batches = list(torch.utils.data.DataLoader(**loader_kwargs))
        if len(batches) != self.resolved.gradient_accumulation_steps:
            raise RuntimeError(
                "Stage-2 DataLoader did not materialize one accumulation."
            )
        return batches

    def _to_device(self, batch: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(batch)
        for key in (
            "sample_id",
            "action_index",
            "initial_latent",
            "real_future",
            "prompt_embeds",
            "prompt_mask",
            "height",
            "width",
        ):
            result[key] = batch[key].to(device=self.device, non_blocking=False)
        return result

    def _draw_branch(self, probability: float) -> str:
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"invalid DFD probability {probability}")
        value = torch.empty(1, dtype=torch.float32, device=self.device)
        if self.is_main_process:
            cpu_value = torch.rand((), generator=self.branch_rng, device="cpu")
            value.fill_(float(cpu_value.item()))
        dist.broadcast(value, src=0)
        return "dfd" if float(value.item()) < probability else "dmd"

    def _snapshot_attempt(self) -> dict[str, Any]:
        from utils.stage1_checkpoint import capture_rng_state

        return {
            "global_rng": capture_rng_state(include_cuda=True),
            "samplers": copy.deepcopy(self.samplers.state_dict()),
            "dataloader_generators": {
                role: generator.get_state().clone()
                for role, generator in self.dataloader_generators.items()
            },
            "exit_rng": copy.deepcopy(self.exit_rng.state_dict()),
            "branch_rng": self.branch_rng.get_state().clone(),
            "dedicated_generators": {
                name: generator.get_state().clone()
                for name, generator in self.dedicated_generators.items()
            },
        }

    def _restore_attempt(self, snapshot: Mapping[str, Any]) -> None:
        from utils.stage1_checkpoint import restore_rng_state

        self.samplers.load_state_dict(snapshot["samplers"])
        for role, value in snapshot["dataloader_generators"].items():
            self.dataloader_generators[role].set_state(value.clone())
        self.exit_rng.load_state_dict(snapshot["exit_rng"])
        self.branch_rng.set_state(snapshot["branch_rng"].clone())
        for name, value in snapshot["dedicated_generators"].items():
            self.dedicated_generators[name].set_state(value.clone())
        restore_rng_state(snapshot["global_rng"], require_cuda_topology=True)

    @staticmethod
    def _tensor_sha256(value: torch.Tensor) -> str:
        if not isinstance(value, torch.Tensor):
            raise TypeError("Stage-2 probe tensor identity requires a Tensor")
        tensor = value.detach().to(device="cpu").contiguous()
        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    @classmethod
    def _attempt_state_sha256(cls, value: Any) -> str:
        """Hash nested RNG/sampler state without serialization side effects."""

        digest = hashlib.sha256()

        def update(item: Any) -> None:
            if isinstance(item, torch.Tensor):
                digest.update(b"tensor:")
                digest.update(cls._tensor_sha256(item).encode("ascii"))
            elif isinstance(item, np.ndarray):
                array = np.ascontiguousarray(item)
                digest.update(b"ndarray:")
                digest.update(str(array.dtype).encode("ascii"))
                digest.update(str(tuple(array.shape)).encode("ascii"))
                digest.update(array.tobytes())
            elif isinstance(item, np.generic):
                update(item.item())
            elif isinstance(item, Mapping):
                digest.update(b"mapping{")
                for key in sorted(item, key=lambda candidate: repr(candidate)):
                    update(key)
                    update(item[key])
                digest.update(b"}")
            elif isinstance(item, (tuple, list)):
                digest.update(b"tuple[" if isinstance(item, tuple) else b"list[")
                for child in item:
                    update(child)
                digest.update(b"]")
            elif isinstance(item, (str, int, float, bool)) or item is None:
                digest.update(type(item).__name__.encode("ascii"))
                digest.update(b":")
                digest.update(repr(item).encode("utf-8"))
            else:
                raise TypeError(
                    "Unsupported Stage-2 probe state value: " f"{type(item).__name__}"
                )

        update(value)
        return digest.hexdigest()

    def _next_f1_state_projection(self) -> dict[str, Any]:
        return {
            "next_substep": self.state.next_substep,
            "current_role": self.state.current_role,
            "completed_fake_updates": self.state.completed_f,
            "completed_generator_updates": self.state.completed_g,
            "completed_cycles": self.state.cycle,
            "successful_attempts": self.state.successful_attempts,
            "nonfinite_attempts": self.state.nonfinite_attempts,
            "nonfinite_attempts_by_role": dict(self.state.nonfinite_attempts_by_role),
        }

    def _capture_next_f1_probe(self) -> dict[str, Any]:
        """Predict the next F1 draws, then restore every mutable stream exactly."""

        from utils.stage1_io import canonical_json_sha256
        from utils.stage2_sampler import partition_stage2_global_batch
        from utils.stage2_score_math import sample_stage2_score_timesteps

        state_projection = self._next_f1_state_projection()
        if (
            state_projection["next_substep"] != "F1"
            or state_projection["current_role"] != "fake_score"
        ):
            raise RuntimeError(
                "Stage-2 next-state probe is legal only at a committed F1 boundary"
            )
        snapshot = self._snapshot_attempt()
        before_sha256 = self._attempt_state_sha256(snapshot)
        local_envelope: dict[str, Any] | None = None
        try:

            def predict_sampler():
                global_batch = self.samplers.fake_score.next_global_batch()
                local_microbatches = partition_stage2_global_batch(
                    global_batch,
                    rank=self.rank,
                    world_size=self.world_size,
                    microbatch_size_per_device=(
                        self.resolved.microbatch_size_per_device
                    ),
                    gradient_accumulation_steps=(
                        self.resolved.gradient_accumulation_steps
                    ),
                    spatial_shapes=self.dataset.spatial_shapes,
                )
                return global_batch, local_microbatches

            global_batch, local_microbatches = self._runtime_world_checked(
                "predict next F1 sampler batch",
                predict_sampler,
            )
            exits = self.exit_rng.draw(
                "fake_score",
                accumulation_steps=self.resolved.gradient_accumulation_steps,
                num_denoising_steps=self.resolved.num_denoising_steps,
                mode=self.resolved.exit_sampling,
                device=self.device,
                synchronize_ranks=True,
            )

            def predict_explicit_streams() -> list[dict[str, Any]]:
                micro_probes: list[dict[str, Any]] = []
                for sample_ids in local_microbatches:
                    spatial_shapes = {
                        tuple(self.dataset.spatial_shapes[index])
                        for index in sample_ids
                    }
                    if len(spatial_shapes) != 1:
                        raise RuntimeError(
                            "Stage-2 next F1 probe microbatch mixes spatial shapes"
                        )
                    height, width = next(iter(spatial_shapes))
                    future_shape = (
                        len(sample_ids),
                        self.resolved.generated_episode_frames,
                        self.resolved.latent_channels,
                        int(height),
                        int(width),
                    )
                    rollout_noise = torch.randn(
                        future_shape,
                        device=self.device,
                        dtype=torch.bfloat16,
                        generator=self.dedicated_generators["fake_score_rollout"],
                    )
                    timesteps = sample_stage2_score_timesteps(
                        batch_size=len(sample_ids),
                        device=self.device,
                        generator=self.dedicated_generators["fake_score_timestep"],
                    )
                    score_noise = torch.randn(
                        future_shape,
                        device=self.device,
                        dtype=torch.float32,
                        generator=self.dedicated_generators["fake_score_noise"],
                    )
                    micro_probes.append(
                        {
                            "sample_ids": list(sample_ids),
                            "spatial_shape": [int(height), int(width)],
                            "rollout_noise_sha256": self._tensor_sha256(rollout_noise),
                            "score_uniform_integers": (
                                timesteps.uniform_integer.detach()
                                .to(device="cpu")
                                .tolist()
                            ),
                            "score_frame_timestep_sha256": self._tensor_sha256(
                                timesteps.frame_timestep
                            ),
                            "score_noise_sha256": self._tensor_sha256(score_noise),
                        }
                    )
                    del rollout_noise, timesteps, score_noise
                return micro_probes

            micro_probes = self._runtime_world_checked(
                "predict next F1 explicit RNG streams",
                predict_explicit_streams,
            )
            fake_sampler_state = snapshot["samplers"]["fake_score"]
            local_envelope = {
                "common": {
                    "state": state_projection,
                    "sampler": {
                        "state_sha256": self._attempt_state_sha256(fake_sampler_state),
                        "completed_batches": fake_sampler_state["completed_batches"],
                        "stream_epoch": fake_sampler_state["stream_epoch"],
                        "batch_cursor": fake_sampler_state["batch_cursor"],
                        "extra_slot_cursor": fake_sampler_state["extra_slot_cursor"],
                        "global_batch_ids": list(global_batch),
                    },
                    "exit_schedule": list(exits),
                },
                "rank": {
                    "rank": self.rank,
                    "attempt_state_sha256": before_sha256,
                    "stream_state_sha256": {
                        "fake_score_loader": self._tensor_sha256(
                            snapshot["dataloader_generators"]["fake_score"]
                        ),
                        "fake_score_exit": self._tensor_sha256(
                            snapshot["exit_rng"]["fake_score"]
                        ),
                        "fake_score_rollout": self._tensor_sha256(
                            snapshot["dedicated_generators"]["fake_score_rollout"]
                        ),
                        "fake_score_timestep": self._tensor_sha256(
                            snapshot["dedicated_generators"]["fake_score_timestep"]
                        ),
                        "fake_score_noise": self._tensor_sha256(
                            snapshot["dedicated_generators"]["fake_score_noise"]
                        ),
                    },
                    "microbatches": micro_probes,
                },
            }
        finally:
            self._restore_attempt(snapshot)

        after_sha256 = self._attempt_state_sha256(self._snapshot_attempt())
        state_restored = before_sha256 == after_sha256
        if dist.is_available() and dist.is_initialized():
            state_restored = self._world_consensus(state_restored)
        if not state_restored:
            raise RuntimeError("Stage-2 next F1 probe changed sampler or RNG state")
        if local_envelope is None:
            raise RuntimeError("Stage-2 next F1 probe produced no local payload")

        if dist.is_available() and dist.is_initialized():
            gathered: list[dict[str, Any] | None] = [None] * self.world_size
            dist.all_gather_object(gathered, local_envelope)
            envelopes = [item for item in gathered if item is not None]
            if len(envelopes) != self.world_size:
                raise RuntimeError(
                    "Stage-2 next F1 probe did not gather every rank payload"
                )
        else:
            envelopes = [local_envelope]
        if not envelopes:
            raise RuntimeError("Stage-2 next F1 probe gathered no rank payloads")
        common = envelopes[0]["common"]
        common_sha256 = canonical_json_sha256(common)
        if any(
            canonical_json_sha256(item["common"]) != common_sha256 for item in envelopes
        ):
            raise RuntimeError(
                "Stage-2 next F1 sampler/exit prediction differs across ranks"
            )
        rank_payloads = sorted(
            (item["rank"] for item in envelopes), key=lambda item: item["rank"]
        )
        ranks = [item["rank"] for item in rank_payloads]
        expected_ranks = (
            list(range(self.world_size))
            if dist.is_available() and dist.is_initialized()
            else [self.rank]
        )
        if ranks != expected_ranks:
            raise RuntimeError(
                "Stage-2 next F1 probe rank payloads are incomplete or duplicated: "
                f"expected={expected_ranks}, actual={ranks}"
            )
        probe = {
            "schema": "longlive_stage2_next_f1_probe/v1",
            **common,
            "rank_payloads": rank_payloads,
        }
        canonical_json_sha256(probe)
        return probe

    @staticmethod
    def _probe_difference(expected: Any, actual: Any, path: str = "probe") -> str:
        if type(expected) is not type(actual):
            return f"{path}: type {type(expected).__name__} != {type(actual).__name__}"
        if isinstance(expected, Mapping):
            if set(expected) != set(actual):
                return f"{path}: keys {sorted(expected)} != {sorted(actual)}"
            for key in sorted(expected):
                difference = Trainer._probe_difference(
                    expected[key], actual[key], f"{path}.{key}"
                )
                if difference:
                    return difference
            return ""
        if isinstance(expected, list):
            if len(expected) != len(actual):
                return f"{path}: length {len(expected)} != {len(actual)}"
            for index, (left, right) in enumerate(zip(expected, actual)):
                difference = Trainer._probe_difference(left, right, f"{path}[{index}]")
                if difference:
                    return difference
            return ""
        return "" if expected == actual else f"{path}: {expected!r} != {actual!r}"

    def _verify_parent_next_f1_probe(self) -> None:
        smoke_mode = getattr(self.options, "smoke_mode", None)
        if smoke_mode not in {"C1", "C2"}:
            return
        if getattr(self, "_smoke_parent_probe_consumed", False):
            return
        if self.state.next_substep != "F1" or self.state.current_role != "fake_score":
            raise RuntimeError(f"Stage-2 smoke {smoke_mode} did not resume at F1")
        parent_smoke_probe = (
            None
            if self.resume_payload is None
            else self.resume_payload.provenance.get("smoke_probe")
        )
        expected = (
            None
            if not isinstance(parent_smoke_probe, Mapping)
            else parent_smoke_probe.get("next_f1_probe")
        )
        if not isinstance(expected, Mapping):
            raise RuntimeError(
                f"Stage-2 smoke {smoke_mode} parent has no next F1 probe"
            )
        actual = self._capture_next_f1_probe()
        difference = self._probe_difference(dict(expected), actual)
        if difference:
            raise RuntimeError(
                f"Stage-2 smoke {smoke_mode} next F1 probe mismatch: " f"{difference}"
            )
        self._smoke_parent_probe_consumed = True

    def _negative_conditioning(self, batch_size: int) -> dict[str, torch.Tensor]:
        negative = self.dataset.negative_conditioning
        embeds = negative["prompt_embeds"].to(device=self.device)
        if embeds.ndim != 2:
            raise RuntimeError("negative prompt_embeds must be [512,4096].")
        return {"prompt_embeds": embeds.unsqueeze(0).expand(batch_size, -1, -1)}

    def _sample_rollout_noise(self, role: str, initial: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            (
                int(initial.shape[0]),
                self.resolved.generated_episode_frames,
                *tuple(initial.shape[2:]),
            ),
            device=self.device,
            dtype=torch.bfloat16,
            generator=self.dedicated_generators[f"{role}_rollout"],
        )

    def _sample_score_noising(
        self,
        *,
        role: str,
        fake_clean: torch.Tensor,
        real_clean: torch.Tensor | None,
    ):
        from utils.stage2_score_math import (
            noise_stage2_score_input,
            noise_stage2_score_pair,
            sample_stage2_score_timesteps,
        )

        timesteps = sample_stage2_score_timesteps(
            batch_size=int(fake_clean.shape[0]),
            device=self.device,
            generator=self.dedicated_generators[f"{role}_timestep"],
        )
        epsilon = torch.randn(
            tuple(fake_clean[:, 1:].shape),
            device=self.device,
            dtype=torch.float32,
            generator=self.dedicated_generators[f"{role}_noise"],
        )
        if real_clean is None:
            return noise_stage2_score_input(
                fake_clean, timesteps.frame_timestep, future_epsilon=epsilon
            )
        return noise_stage2_score_pair(
            fake_clean,
            real_clean,
            timesteps.frame_timestep,
            future_epsilon=epsilon,
        )

    def _gradients_finite(self, module: torch.nn.Module) -> bool:
        found = False
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            found = True
            if parameter.grad is None:
                return False
            local = _local_tensor(parameter.grad)
            if local.numel() and not bool(torch.isfinite(local).all().item()):
                return False
        return found

    def _other_role_gradients_absent(self, target_role: str) -> bool:
        for role in ("generator", "real_score", "fake_score"):
            if role == target_role:
                continue
            if any(
                parameter.grad is not None
                for parameter in getattr(self.model, role).parameters()
            ):
                return False
        return True

    def _clip_grad_norm(self, module: torch.nn.Module, maximum: float) -> float:
        value = torch.nn.utils.clip_grad_norm_(
            module.parameters(), max_norm=maximum, error_if_nonfinite=False
        )
        full_tensor = getattr(value, "full_tensor", None)
        if callable(full_tensor):
            value = full_tensor()
        return float(value.item())

    def _reduce_loss(self, numerator: torch.Tensor, count: int) -> tuple[float, int]:
        packed = torch.tensor(
            [float(numerator.detach().float().item()), float(count)],
            device=self.device,
            dtype=torch.float64,
        )
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)
        return float(packed[0].item()), int(packed[1].item())

    def _reduce_diagnostics(
        self, diagnostics: Sequence[Mapping[str, float]]
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        distributed = dist.is_available() and dist.is_initialized()
        for key in sorted({name for item in diagnostics for name in item}):
            values = [float(item[key]) for item in diagnostics if key in item]
            if not values:
                continue
            if key.endswith("_min"):
                tensor = torch.tensor(
                    min(values), device=self.device, dtype=torch.float64
                )
                if distributed:
                    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
                result[key] = float(tensor.item())
            elif key.endswith("_max"):
                tensor = torch.tensor(
                    max(values), device=self.device, dtype=torch.float64
                )
                if distributed:
                    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
                result[key] = float(tensor.item())
            else:
                packed = torch.tensor(
                    [sum(values), len(values)],
                    device=self.device,
                    dtype=torch.float64,
                )
                if distributed:
                    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                result[key] = float((packed[0] / packed[1]).item())
        return result

    def _global_values(self, values: Sequence[float | int]) -> list[float]:
        if not (dist.is_available() and dist.is_initialized()):
            return [float(value) for value in values]
        gathered: list[list[float] | None] = [None] * self.world_size
        dist.all_gather_object(gathered, [float(value) for value in values])
        return [value for row in gathered if row is not None for value in row]

    def _world_max_seconds(self, value: float) -> float:
        if not (dist.is_available() and dist.is_initialized()):
            return float(value)
        tensor = torch.tensor(value, device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
        return float(tensor.item())

    def _global_batch_identity(self, batches: Sequence[Mapping[str, Any]]) -> str:
        local = [batch["sample_id"].detach().cpu().tolist() for batch in batches]
        if dist.is_available() and dist.is_initialized():
            gathered: list[list[list[int]] | None] = [None] * self.world_size
            dist.all_gather_object(gathered, local)
            payload = [row for row in gathered if row is not None]
        else:
            payload = [local]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _assert_state_consensus(self) -> None:
        payload = self.state.state_dict()
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if not (dist.is_available() and dist.is_initialized()):
            return
        gathered: list[str | None] = [None] * self.world_size
        dist.all_gather_object(gathered, digest)
        if len(set(gathered)) != 1:
            raise RuntimeError(
                f"Stage-2 committed state diverged across ranks: {gathered}"
            )

    @staticmethod
    def _score_timestep_histogram(values: Sequence[float]) -> dict[str, int]:
        boundaries = (
            20.0,
            100.0,
            200.0,
            300.0,
            400.0,
            500.0,
            600.0,
            700.0,
            800.0,
            900.0,
            980.000001,
        )
        result: dict[str, int] = {}
        for lower, upper in zip(boundaries[:-1], boundaries[1:]):
            right = "]" if upper > 980.0 else ")"
            label = f"[{int(lower)},{980 if upper > 980 else int(upper)}{right}"
            result[label] = sum(lower <= value < upper for value in values)
        return result

    def _timing_summary(
        self, elapsed: float, categories: Mapping[str, float]
    ) -> dict[str, float]:
        names = (
            "data",
            "h2d",
            "rollout",
            "fake_score",
            "real_cond",
            "real_uncond",
            "loss_build",
            "backward",
            "clip_optimizer",
            "orchestration",
            "ema",
        )
        # Keep the old coarse seam usable by tiny CPU tests while production
        # always emits the detailed Stage-2 timing contract.
        normalized = {name: float(categories.get(name, 0.0)) for name in names}
        if "compute" in categories and not any(
            normalized[name]
            for name in (
                "rollout",
                "fake_score",
                "real_cond",
                "real_uncond",
                "loss_build",
                "backward",
            )
        ):
            normalized["rollout"] = float(categories["compute"])
        if "optimizer" in categories and not normalized["clip_optimizer"]:
            normalized["clip_optimizer"] = float(categories["optimizer"])
        value = torch.tensor(
            [elapsed, *(normalized[name] for name in names)],
            device=self.device,
            dtype=torch.float64,
        )
        values = [torch.zeros_like(value) for _ in range(self.world_size)]
        dist.all_gather(values, value)
        rows = [item.detach().cpu().tolist() for item in values]
        wall_values = [float(row[0]) for row in rows]
        maximum = max(wall_values)
        mean = sum(wall_values) / len(wall_values)
        slowest = rows[wall_values.index(maximum)]
        selected = {name: float(slowest[index + 1]) for index, name in enumerate(names)}
        closure = maximum - sum(selected.values())
        compute = sum(
            selected[name]
            for name in (
                "rollout",
                "fake_score",
                "real_cond",
                "real_uncond",
                "loss_build",
                "backward",
            )
        )
        return {
            "step_seconds_max": maximum,
            "step_seconds_mean": mean,
            "straggler_ratio": maximum / mean,
            "data_seconds_max": selected["data"],
            "h2d_seconds_max": selected["h2d"],
            "rollout_seconds_max": selected["rollout"],
            "fake_score_seconds_max": selected["fake_score"],
            "real_cond_seconds_max": selected["real_cond"],
            "real_uncond_seconds_max": selected["real_uncond"],
            "loss_build_seconds_max": selected["loss_build"],
            "backward_seconds_max": selected["backward"],
            "clip_optimizer_seconds_max": selected["clip_optimizer"],
            "orchestration_seconds_max": selected["orchestration"],
            "compute_seconds_max": compute,
            "optimizer_seconds_max": selected["clip_optimizer"],
            "ema_seconds_max": selected["ema"],
            "timing_closure_error_seconds": closure,
        }

    def _cuda_timed(self, callback):
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        value = callback()
        torch.cuda.synchronize(self.device)
        return value, time.perf_counter() - started

    def _compute_micro_loss(
        self,
        *,
        role: str,
        batch: Mapping[str, Any],
        exit_step: int,
        branch: str,
    ) -> dict[str, Any]:
        """Run all aligned forwards for one microbatch, without backward."""

        from utils.stage2_i2v_conditioning import pack_stage2_i2v_score_input

        phase_timings = {
            "rollout": 0.0,
            "fake_score": 0.0,
            "real_cond": 0.0,
            "real_uncond": 0.0,
            "loss_build": 0.0,
        }

        def timed_model_call(label: str, callback):
            value, elapsed = self._cuda_timed(callback)
            phase_timings[label] += elapsed
            return value

        initial = batch["initial_latent"]
        condition = {"prompt_embeds": batch["prompt_embeds"]}
        noise = self._sample_rollout_noise(role, initial)
        save_context = (
            torch.autograd.graph.save_on_cpu(pin_memory=True)
            if role == "generator" and self.resolved.saved_tensor_cpu_offload
            else nullcontext()
        )

        def execute_rollout():
            with save_context:
                return self.rollout.rollout(
                    initial_latent=initial,
                    noise=noise,
                    conditional_dict=condition,
                    exit_step=exit_step,
                    requires_grad=role == "generator",
                )

        rollout_value = None
        rollout_result = None
        generated = None
        rollout_error: BaseException | None = None
        try:
            rollout_value, phase_timings["rollout"] = self._cuda_timed(execute_rollout)
            rollout_result, _ = rollout_value
            generated = rollout_result.latents
            if not bool(torch.isfinite(generated.detach()).all().item()):
                rollout_error = ValueError("non-finite Stage-2 rollout prediction")
        except (TypeError, ValueError, FloatingPointError) as exc:
            rollout_error = exc
        self._align_pre_score_failure("rollout/output validation", rollout_error)
        assert rollout_result is not None and generated is not None
        torch.cuda.synchronize(self.device)
        post_rollout_started = time.perf_counter()
        noised = None
        setup_error: BaseException | None = None
        try:
            fake_clean = pack_stage2_i2v_score_input(initial, generated)
            real_clean = (
                pack_stage2_i2v_score_input(initial, batch["real_future"])
                if role == "generator" and branch == "dfd"
                else None
            )
            noised = self._sample_score_noising(
                role=role,
                fake_clean=fake_clean,
                real_clean=real_clean,
            )
        except Exception as exc:  # pure local setup; align before any score forward
            setup_error = exc
        self._align_pre_score_failure("score noising/setup", setup_error)
        assert noised is not None
        if role == "fake_score":
            output = self.model.fake_score_flow_dsm_loss_from_model(
                generated_future=generated.detach(),
                noised_fake_score=noised,
                conditional_dict=condition,
                timing_callback=timed_model_call,
            )
            diagnostic = {
                "target_flow_l2": float(output.target_flow_l2.item()),
                "prediction_l2": float(output.prediction_l2.item()),
            }
            score_calls = 1
            fake_score_calls = 1
            real_score_calls = 0
        else:
            output = self.model.generator_distribution_matching_loss_from_models(
                branch=branch,
                generated_future=generated,
                noised_score=noised,
                conditional_dict=condition,
                real_unconditional_dict=self._negative_conditioning(
                    int(initial.shape[0])
                ),
                timing_callback=timed_model_call,
            )
            diagnostic = {
                "denominator_min": float(output.denominator.min().item()),
                "denominator_mean": float(output.denominator.mean().item()),
                "denominator_max": float(output.denominator.max().item()),
                "raw_score_difference_l2": float(
                    output.raw_score_difference_l2.mean().item()
                ),
                "fake_x0_l2": float(output.fake_x0_l2.mean().item()),
                "real_x0_l2": float(output.real_x0_l2.mean().item()),
            }
            score_calls = 3
            fake_score_calls = 1
            real_score_calls = 2
        torch.cuda.synchronize(self.device)
        post_rollout_elapsed = time.perf_counter() - post_rollout_started
        phase_timings["loss_build"] = max(
            0.0,
            post_rollout_elapsed
            - phase_timings["fake_score"]
            - phase_timings["real_cond"]
            - phase_timings["real_uncond"],
        )
        frame_time = (
            noised.fake.frame_timestep
            if hasattr(noised, "fake")
            else noised.frame_timestep
        )
        return {
            "output": output,
            "diagnostic": diagnostic,
            "score_calls": score_calls,
            "fake_score_calls": fake_score_calls,
            "real_score_calls": real_score_calls,
            "rollout_result": rollout_result,
            "frame_time": frame_time,
            "sample_count": int(initial.shape[0]),
            "phase_timings": phase_timings,
        }

    def _run_update_attempt(
        self,
        *,
        role: str,
        batches: Sequence[Mapping[str, Any]],
        exits: Sequence[int],
        branch: str,
    ) -> dict[str, Any]:
        from utils.distributed import fsdp2_accumulation

        attempt_started = time.perf_counter()
        module = getattr(self.model, role)
        optimizer = self.optimizers[role]
        optimizer.zero_grad(set_to_none=True)
        local_numerator = torch.zeros((), device=self.device, dtype=torch.float32)
        local_count = 0
        diagnostics: list[dict[str, float]] = []
        rollout_forward_calls = 0
        rollout_tokens = 0
        score_forward_calls = 0
        score_tokens = 0
        fake_score_forward_calls = 0
        fake_score_tokens = 0
        real_score_forward_calls = 0
        real_score_tokens = 0
        exit_values: list[int] = []
        timestep_values: list[float] = []
        phase_timings = {
            "rollout": 0.0,
            "fake_score": 0.0,
            "real_cond": 0.0,
            "real_uncond": 0.0,
            "loss_build": 0.0,
            "backward": 0.0,
        }
        compute_started = time.perf_counter()

        for micro_index, batch in enumerate(batches):
            exit_step = int(exits[micro_index])
            sync = micro_index == len(batches) - 1
            with fsdp2_accumulation(module, sync_gradients=sync):
                local_nonfinite: Exception | None = None
                micro: dict[str, Any] | None = None
                try:
                    micro = self._compute_micro_loss(
                        role=role,
                        batch=batch,
                        exit_step=exit_step,
                        branch=branch,
                    )
                except (ValueError, FloatingPointError) as exc:
                    if not self._is_retryable_nonfinite_error(exc):
                        raise
                    local_nonfinite = exc
                # Pre-score failures were already aligned before the score
                # forward.  Any retryable error reaching this point therefore
                # occurred after every rank completed the same model forwards.
                if not self._world_consensus(local_nonfinite is None):
                    optimizer.zero_grad(set_to_none=True)
                    return {
                        "success": False,
                        "reason": "nonfinite prediction/loss: "
                        + (str(local_nonfinite) if local_nonfinite else "peer rank"),
                    }
                assert micro is not None
                output = micro["output"]
                diagnostic = micro["diagnostic"]
                diagnostic_values = tuple(diagnostic.values())
                locally_finite = bool(
                    torch.isfinite(output.loss.detach()).item()
                ) and all(math.isfinite(value) for value in diagnostic_values)
                if not self._world_consensus(locally_finite):
                    optimizer.zero_grad(set_to_none=True)
                    return {"success": False, "reason": "nonfinite loss/diagnostic"}
                from utils.stage2_train_state import (
                    stage2_global_mean_loss_for_backward,
                )

                sample_count = micro["sample_count"]
                generated_latents = getattr(micro["rollout_result"], "latents", None)
                expected_count = (
                    int(generated_latents.numel())
                    if isinstance(generated_latents, torch.Tensor)
                    else int(output.count)
                )
                if int(output.count) != expected_count or expected_count % sample_count:
                    raise RuntimeError(
                        "Stage-2 loss count must equal every generated future element: "
                        f"expected={expected_count}, actual={int(output.count)}."
                    )
                global_count = (
                    expected_count // sample_count
                ) * self.resolved.global_batch_size
                backward_loss = stage2_global_mean_loss_for_backward(
                    output.numerator,
                    global_count=global_count,
                    world_size=self.world_size,
                )
                _, backward_elapsed = self._cuda_timed(backward_loss.backward)
                phase_timings["backward"] += backward_elapsed
            local_numerator = local_numerator + output.numerator.detach().float()
            local_count += int(output.count)
            diagnostics.append(diagnostic)
            rollout_result = micro["rollout_result"]
            rollout_forward_calls += int(
                rollout_result.cache_audit["generator_forward_calls"]
            )
            # The rollout audit reports the token schedule for one sample.  A
            # Wan forward processes the whole local microbatch, so throughput
            # accounting must scale it here (exactly once) by the actual batch
            # size.  This keeps both locked launch shapes correct: micro=2 and
            # the memory fallback micro=1.
            rollout_tokens += (
                int(rollout_result.cache_audit["logical_query_tokens"]) * sample_count
            )
            score_forward_calls += int(micro["score_calls"])
            score_tokens += (
                int(micro["score_calls"]) * sample_count * self.resolved.score_seq_len
            )
            fake_score_forward_calls += int(micro.get("fake_score_calls", 0))
            fake_score_tokens += (
                int(micro.get("fake_score_calls", 0))
                * sample_count
                * self.resolved.score_seq_len
            )
            real_score_forward_calls += int(micro.get("real_score_calls", 0))
            real_score_tokens += (
                int(micro.get("real_score_calls", 0))
                * sample_count
                * self.resolved.score_seq_len
            )
            exit_values.append(exit_step)
            frame_time = micro["frame_time"]
            timestep_values.extend(frame_time[:, 1].detach().float().cpu().tolist())
            for name, value in micro.get("phase_timings", {}).items():
                phase_timings[name] += float(value)

        torch.cuda.synchronize(self.device)
        compute_seconds = time.perf_counter() - compute_started
        if not self._world_consensus(self._gradients_finite(module)):
            optimizer.zero_grad(set_to_none=True)
            return {"success": False, "reason": "nonfinite or missing gradients"}
        if not self._world_consensus(self._other_role_gradients_absent(role)):
            optimizer.zero_grad(set_to_none=True)
            raise RuntimeError(
                f"Stage-2 {role} update leaked gradients into another role."
            )
        optimizer_started = time.perf_counter()
        grad_norm = self._runtime_world_checked(
            f"{role} gradient clipping",
            lambda: self._clip_grad_norm(
                module,
                (
                    self.resolved.generator_optimizer.max_grad_norm
                    if role == "generator"
                    else self.resolved.fake_score_optimizer.max_grad_norm
                ),
            ),
        )
        if not self._world_consensus(math.isfinite(grad_norm)):
            optimizer.zero_grad(set_to_none=True)
            return {"success": False, "reason": "nonfinite preclip grad norm"}
        self._runtime_world_checked(f"{role} optimizer step", optimizer.step)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(self.device)
        optimizer_seconds = time.perf_counter() - optimizer_started
        # A post-step non-finite value is not safely retryable: optimizer state
        # already mutated. Fail the job and resume the previous complete cycle.
        parameters_finite = all(
            not _local_tensor(parameter).numel()
            or bool(torch.isfinite(_local_tensor(parameter)).all().item())
            for parameter in module.parameters()
            if parameter.requires_grad
        )
        if not self._world_consensus(parameters_finite):
            raise FloatingPointError(
                f"Stage-2 {role} optimizer produced non-finite parameters; "
                "restart from the previous complete-cycle checkpoint."
            )

        if hasattr(self, "lora_schemas") and hasattr(self, "state"):
            from utils.stage2_checkpoint import audit_stage2_lora_optimizer

            expected_completed = (
                self.state.completed_g + 1
                if role == "generator"
                else self.state.completed_f + 1
            )
            self._runtime_world_checked(
                f"{role} optimizer post-step state audit",
                lambda: audit_stage2_lora_optimizer(
                    module,
                    optimizer,
                    role=role,
                    expected_schema=self.lora_schemas[role],
                    require_initialized_moments=True,
                    expected_completed_updates=expected_completed,
                    expected_optimizer_spec=(
                        self.resolved.generator_optimizer
                        if role == "generator"
                        else self.resolved.fake_score_optimizer
                    ),
                ),
            )

        numerator, count = self._reduce_loss(local_numerator, local_count)
        merged = self._reduce_diagnostics(diagnostics)
        torch.cuda.synchronize(self.device)
        attempt_seconds = time.perf_counter() - attempt_started
        classified_attempt_seconds = sum(phase_timings.values()) + optimizer_seconds
        attempt_orchestration_seconds = max(
            0.0, attempt_seconds - classified_attempt_seconds
        )
        return {
            "success": True,
            "loss_numerator": numerator,
            "loss_count": count,
            "loss": numerator / count,
            "preclip_grad_norm": grad_norm,
            "diagnostics": merged,
            "rollout_forward_calls": rollout_forward_calls,
            "rollout_tokens": rollout_tokens,
            "score_forward_calls": score_forward_calls,
            "score_tokens": score_tokens,
            "fake_score_forward_calls": fake_score_forward_calls,
            "fake_score_tokens": fake_score_tokens,
            "real_score_forward_calls": real_score_forward_calls,
            "real_score_tokens": real_score_tokens,
            "exit_values": exit_values,
            "timestep_values": timestep_values,
            "compute_seconds": compute_seconds,
            "optimizer_seconds": optimizer_seconds,
            "attempt_seconds": attempt_seconds,
            "attempt_orchestration_seconds": attempt_orchestration_seconds,
            "phase_timings": phase_timings,
        }

    def _memory_fields(self) -> dict[str, float]:
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        allocated = torch.tensor(
            [torch.cuda.max_memory_allocated(self.device) / 1024**3],
            device=self.device,
            dtype=torch.float64,
        )
        reserved = torch.tensor(
            [torch.cuda.max_memory_reserved(self.device) / 1024**3],
            device=self.device,
            dtype=torch.float64,
        )
        free = torch.tensor(
            [free_bytes / 1024**3], device=self.device, dtype=torch.float64
        )
        total = torch.tensor(
            [total_bytes / 1024**3], device=self.device, dtype=torch.float64
        )
        dist.all_reduce(allocated, op=dist.ReduceOp.MAX)
        dist.all_reduce(reserved, op=dist.ReduceOp.MAX)
        dist.all_reduce(free, op=dist.ReduceOp.MIN)
        dist.all_reduce(total, op=dist.ReduceOp.MIN)
        result = {
            "gpu_memory_allocated_gib_max": float(allocated.item()),
            "gpu_memory_reserved_gib_max": float(reserved.item()),
            "gpu_memory_free_gib_min": float(free.item()),
            "gpu_memory_total_gib_min": float(total.item()),
        }
        if self.options.dry_run:
            result["nvml_gpu_memory_free_gib_min"] = self._nvml_free_gib_min()
        return result

    def _nvml_free_gib_min(self) -> float:
        if getattr(self, "device", torch.device("cpu")).type != "cuda":
            return math.inf
        try:
            import pynvml
        except ImportError as exc:
            raise RuntimeError(
                "Stage-2 H100 smoke requires nvidia-ml-py for the NVML free-memory gate."
            ) from exc
        pynvml.nvmlInit()
        try:
            properties = torch.cuda.get_device_properties(self.device)
            uuid_value = getattr(properties, "uuid", None)
            handle = None
            if uuid_value:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByUUID(
                        str(uuid_value).encode("utf-8")
                    )
                except (pynvml.NVMLError, TypeError, ValueError):
                    handle = None
            if handle is None:
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.local_rank)
            local_free = pynvml.nvmlDeviceGetMemoryInfo(handle).free / 1024**3
        finally:
            pynvml.nvmlShutdown()
        tensor = torch.tensor(local_free, device=self.device, dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
        return float(tensor.item())

    def _validate_smoke_cycle(
        self,
        step_fields: Sequence[Mapping[str, Any]],
        *,
        nonfinite_attempts: int,
    ) -> dict[str, Any] | None:
        if not self.options.dry_run:
            return None
        if len(step_fields) != 6:
            raise RuntimeError("Stage-2 smoke must contain exactly 5F+1G train steps.")
        if (
            isinstance(nonfinite_attempts, bool)
            or not isinstance(nonfinite_attempts, int)
            or nonfinite_attempts < 0
        ):
            raise RuntimeError(
                "Stage-2 smoke cycle nonfinite_attempts must be an integer >= 0."
            )
        if nonfinite_attempts:
            raise RuntimeError(
                "Stage-2 H100 smoke acceptance gate failed: "
                "failures=['nonfinite_attempts'], "
                f"measurements={{'nonfinite_attempts': {nonfinite_attempts}}}."
            )
        if getattr(self, "device", torch.device("cpu")).type != "cuda":
            return {
                "smoke_mode": self.options.smoke_mode,
                "status": "PASS_CPU_TEST_SEAM",
                "live_allocated_gib_max": 0.0,
                "nonfinite_attempts": nonfinite_attempts,
            }
        total_gib = min(float(item["gpu_memory_total_gib_min"]) for item in step_fields)
        max_allocated = max(
            float(item["gpu_memory_allocated_gib_max"]) for item in step_fields
        )
        max_reserved = max(
            float(item["gpu_memory_reserved_gib_max"]) for item in step_fields
        )
        cuda_free = min(float(item["gpu_memory_free_gib_min"]) for item in step_fields)
        nvml_free = min(
            float(item["nvml_gpu_memory_free_gib_min"]) for item in step_fields
        )
        min_required_free = max(
            self.resolved.preflight_min_free_gib,
            self.resolved.preflight_min_free_fraction * total_gib,
        )
        max_straggler = max(float(item["straggler_ratio"]) for item in step_fields)
        local_live = (
            torch.cuda.memory_allocated(self.device) / 1024**3
            if getattr(self, "device", torch.device("cpu")).type == "cuda"
            else 0.0
        )
        live = torch.tensor(local_live, device=self.device, dtype=torch.float64)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(live, op=dist.ReduceOp.MAX)
        live_max = float(live.item())
        checks = {
            "nonfinite_attempts": nonfinite_attempts,
            "allocated_fraction": max_allocated / total_gib,
            "reserved_fraction": max_reserved / total_gib,
            "cuda_free_gib_min": cuda_free,
            "nvml_free_gib_min": nvml_free,
            "straggler_ratio_max": max_straggler,
            "live_allocated_gib_max": live_max,
        }
        failures = []
        if (
            checks["allocated_fraction"]
            > self.resolved.preflight_max_allocated_fraction
        ):
            failures.append("max_allocated")
        if checks["reserved_fraction"] > self.resolved.preflight_max_reserved_fraction:
            failures.append("max_reserved")
        if min(cuda_free, nvml_free) < min_required_free:
            failures.append("min_free")
        if max_straggler > self.resolved.preflight_max_straggler_ratio:
            failures.append("straggler")
        parent_probe = (
            None
            if self.resume_payload is None
            else self.resume_payload.provenance.get("smoke_probe")
        )
        if isinstance(parent_probe, Mapping):
            growth = live_max - float(parent_probe["live_allocated_gib_max"])
            checks["live_allocated_growth_gib"] = growth
            if growth > self.resolved.preflight_max_live_allocated_growth_gib:
                failures.append("live_allocated_growth")
        else:
            checks["live_allocated_growth_gib"] = 0.0
        if failures:
            raise RuntimeError(
                "Stage-2 H100 smoke acceptance gate failed: "
                f"failures={failures}, measurements={checks}."
            )
        return {
            "smoke_mode": self.options.smoke_mode,
            "status": "PASS",
            **checks,
        }

    def _metric_clock(
        self, executed_substep: str, logical_substep_id: int
    ) -> dict[str, Any]:
        return {
            "logical_substep_id": logical_substep_id,
            "completed_fake_updates": self.state.completed_f,
            "completed_generator_updates": self.state.completed_g,
            "completed_cycles": self.state.cycle,
            "cycle_substep": executed_substep,
        }

    def _run_one_logical_substep(self) -> dict[str, Any]:
        role = self.state.current_role
        substep = self.state.next_substep
        logical = self.state.successful_attempts
        self._verify_parent_next_f1_probe()
        position = self.schedule.next_generator_position(self.state.completed_g)
        probability = position.dfd_probability if role == "generator" else 0.0
        if role == "generator" and self.options.smoke_mode == "C1":
            probability = 0.0
        elif role == "generator" and self.options.smoke_mode == "C2":
            probability = 1.0
        snapshot = self._snapshot_attempt()
        retry_seconds = 0.0
        for attempt in range(1, self.resolved.nonfinite_max_attempts_per_update + 1):
            if attempt > 1:
                self._restore_attempt(snapshot)
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            started = time.perf_counter()
            batches = self._materialize_batches(role)
            data_seconds = time.perf_counter() - started
            h2d_started = time.perf_counter()
            batches = [self._to_device(batch) for batch in batches]
            torch.cuda.synchronize(self.device)
            h2d_seconds = time.perf_counter() - h2d_started
            control_started = time.perf_counter()
            exits = self.exit_rng.draw(
                role,
                accumulation_steps=self.resolved.gradient_accumulation_steps,
                num_denoising_steps=self.resolved.num_denoising_steps,
                mode=self.resolved.exit_sampling,
                device=self.device,
                synchronize_ranks=True,
            )
            branch = (
                self._draw_branch(probability) if role == "generator" else "flow_dsm"
            )
            control_seconds = time.perf_counter() - control_started
            result = self._run_update_attempt(
                role=role, batches=batches, exits=exits, branch=branch
            )
            post_attempt_control_started = time.perf_counter()
            torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            if not result["success"]:
                attempt_seconds = self._world_max_seconds(elapsed)
                retry_seconds += attempt_seconds
                self._restore_attempt(snapshot)
                self.state.record_nonfinite_attempt(role)
                memory = self._memory_fields()
                batch_identity = self._global_batch_identity(batches)
                self._append_metric(
                    "nonfinite_attempt",
                    {
                        **self._metric_clock(substep, logical),
                        "role": role,
                        "attempt_number_for_substep": attempt,
                        "reason": result["reason"],
                        "elapsed_seconds": attempt_seconds,
                        "batch_identity": batch_identity,
                        "nonfinite": True,
                        "skipped": True,
                        "dry_run": self.options.dry_run,
                        "smoke_mode": self.options.smoke_mode,
                        **memory,
                    },
                )
                if attempt == self.resolved.nonfinite_max_attempts_per_update:
                    raise FloatingPointError(
                        "Stage-2 logical substep remained non-finite for two exact "
                        f"replays: logical_substep_id={logical}, role={role}."
                    )
                continue

            ema_action = "not_applicable"
            ema_seconds = 0.0
            if role == "generator":
                next_completed_g = self.state.completed_g + 1
                ema_started = time.perf_counter()
                control_seconds += ema_started - post_attempt_control_started
                ema_action = self._runtime_world_checked(
                    "generator EMA update",
                    lambda: self.generator_ema.update_after_step(
                        self.model.generator, next_completed_g
                    ),
                )
                ema_seconds = time.perf_counter() - ema_started
                post_attempt_control_started = time.perf_counter()
                self._runtime_world_checked(
                    "commit generator clock",
                    lambda: self.state.commit_successful_generator_update(
                        ema_action=ema_action, schedule=self.schedule
                    ),
                )
            else:
                self._runtime_world_checked(
                    "commit fake-score clock",
                    lambda: self.state.commit_successful_fake_update(
                        substep, schedule=self.schedule
                    ),
                )
            self._assert_state_consensus()
            torch.cuda.synchronize(self.device)
            control_seconds += time.perf_counter() - post_attempt_control_started
            elapsed = time.perf_counter() - started
            timing = self._timing_summary(
                elapsed,
                {
                    "data": data_seconds,
                    "h2d": h2d_seconds,
                    **result.get(
                        "phase_timings", {"rollout": result["compute_seconds"]}
                    ),
                    "clip_optimizer": result["optimizer_seconds"],
                    "orchestration": result.get("attempt_orchestration_seconds", 0.0)
                    + control_seconds,
                    "ema": ema_seconds,
                },
            )
            score_t = self._global_values(result["timestep_values"])
            # Exit schedules are rank-0 drawn and broadcast, so one local copy
            # is the logical accumulation schedule; gathering would count the
            # same four decisions eight times.
            exits_flat = [int(value) for value in result["exit_values"]]
            memory = self._memory_fields()
            fake_score_calls = result.get(
                "fake_score_forward_calls",
                result["score_forward_calls"] if role == "fake_score" else 0,
            )
            fake_score_tokens = result.get(
                "fake_score_tokens",
                result["score_tokens"] if role == "fake_score" else 0,
            )
            real_score_calls = result.get("real_score_forward_calls", 0)
            real_score_tokens = result.get("real_score_tokens", 0)
            fields = {
                **self._metric_clock(substep, logical),
                "role": role,
                "phase": position.phase,
                "generator_update_target": position.generator_update,
                "global_epoch": position.global_epoch,
                "phase_epoch": position.phase_epoch,
                "epoch_update_index": position.epoch_update_index,
                "branch": branch,
                "loss": result["loss"],
                "loss_numerator": result["loss_numerator"],
                "loss_count": result["loss_count"],
                "learning_rate": (
                    self.resolved.generator_optimizer.learning_rate
                    if role == "generator"
                    else self.resolved.fake_score_optimizer.learning_rate
                ),
                "preclip_grad_norm": result["preclip_grad_norm"],
                "ema_action": ema_action,
                "dfd_probability": probability,
                "branch_is_dfd": int(branch == "dfd"),
                "score_timestep_min": min(score_t),
                "score_timestep_mean": sum(score_t) / len(score_t),
                "score_timestep_max": max(score_t),
                "score_timestep_histogram": self._score_timestep_histogram(score_t),
                "score_timestep_edge_mass": sum(
                    value
                    in {
                        self.resolved.score_timestep_min,
                        self.resolved.score_timestep_max,
                    }
                    for value in score_t
                )
                / len(score_t),
                "exit_step_mean": sum(exits_flat) / len(exits_flat),
                "exit_histogram": {
                    str(index): exits_flat.count(index)
                    for index in range(self.resolved.num_denoising_steps)
                },
                "generator_forward_calls": result["rollout_forward_calls"],
                "generator_logical_query_tokens": result["rollout_tokens"]
                * self.world_size,
                "score_forward_calls": result["score_forward_calls"],
                "score_logical_query_tokens": result["score_tokens"] * self.world_size,
                "fake_score_forward_calls": fake_score_calls,
                "fake_score_logical_query_tokens": fake_score_tokens * self.world_size,
                "real_score_forward_calls": real_score_calls,
                "real_score_logical_query_tokens": real_score_tokens * self.world_size,
                "samples_per_second": self.resolved.global_batch_size
                / timing["step_seconds_max"],
                "generated_latents_per_second": self.resolved.global_batch_size
                * self.resolved.generated_episode_frames
                / timing["step_seconds_max"],
                "generator_tokens_per_second": result["rollout_tokens"]
                * self.world_size
                / timing["step_seconds_max"],
                "score_tokens_per_second": result["score_tokens"]
                * self.world_size
                / timing["step_seconds_max"],
                "fake_score_tokens_per_second": fake_score_tokens
                * self.world_size
                / timing["step_seconds_max"],
                "real_score_tokens_per_second": real_score_tokens
                * self.world_size
                / timing["step_seconds_max"],
                "dry_run": self.options.dry_run,
                "smoke_mode": self.options.smoke_mode,
                **result["diagnostics"],
                **timing,
                **memory,
            }
            self._append_metric("train_step", fields)
            return {
                "fields": fields,
                "elapsed": timing["step_seconds_max"],
                "total_elapsed": retry_seconds + timing["step_seconds_max"],
                "retry_seconds": retry_seconds,
            }
        raise AssertionError("Stage-2 retry loop exited without success or exception")

    def _restore_logical_state(self, resume_payload: Any) -> None:
        from utils.stage2_train_state import Stage2TrainingState

        if resume_payload is None:
            self.state = Stage2TrainingState()
            return
        checkpoint = resume_payload.trainer_state
        successful = checkpoint["successful_attempts"]
        nonfinite = checkpoint["nonfinite_attempts"]
        self.state = Stage2TrainingState(
            completed_g=checkpoint["completed_generator_updates"],
            completed_f=checkpoint["completed_fake_updates"],
            cycle=checkpoint["completed_cycles"],
            next_substep=checkpoint["next_substep"],
            successful_attempts=sum(int(value) for value in successful.values()),
            nonfinite_attempts=sum(int(value) for value in nonfinite.values()),
            nonfinite_attempts_by_role=dict(nonfinite),
        )
        self.state.validate(self.schedule)
        self.samplers.load_state_dict(checkpoint["sampler_state"])

    def _checkpoint_dedicated_generators(self) -> dict[str, torch.Generator]:
        return {
            **self.dedicated_generators,
            "generator_loader": self.dataloader_generators["generator"],
            "fake_score_loader": self.dataloader_generators["fake_score"],
        }

    def _rank0_control_generators(
        self,
    ) -> dict[str, torch.Generator] | None:
        if not self.is_main_process:
            return None
        return {
            "generator_exit": self.exit_rng.generator("generator"),
            "fake_score_exit": self.exit_rng.generator("fake_score"),
            "dfd_branch": self.branch_rng,
        }

    def _restore_rng_last(self, resume_payload: Any) -> None:
        if resume_payload is None:
            return
        from utils.stage2_checkpoint import restore_stage2_rng_state

        restore_stage2_rng_state(
            resume_payload.local_rng_state,
            rank=self.rank,
            expected_world_size=self.world_size,
            dedicated_generators=self._checkpoint_dedicated_generators(),
            rank0_control_generators=self._rank0_control_generators(),
            require_cuda_topology=True,
        )

    def _run_metadata(self) -> dict[str, Any]:
        return {
            "config_contract_sha256": self.resolved.contract_hash(),
            "config_launch_sha256": self.resolved.launch_hash(),
            "resolved_config": self.resolved.to_dict(),
            "world_size": self.world_size,
            "topology": {
                "fsdp_backend": self.resolved.fsdp_backend,
                "sharding_strategy": self.resolved.sharding_strategy,
                "microbatch_size_per_device": self.resolved.microbatch_size_per_device,
                "gradient_accumulation_steps": self.resolved.gradient_accumulation_steps,
                "global_batch_size": self.resolved.global_batch_size,
            },
            "role_hashes": {
                role: audit.base_checkpoint_sha256
                for role, audit in self.role_audits.items()
            },
            "terminal_counts": {
                "fake_score_updates": self.resolved.total_fake_updates,
                "generator_updates": self.resolved.total_generator_updates,
                "cycles": self.resolved.total_cycles,
            },
            "phase_boundaries": [
                {
                    "label": "Phase A end",
                    "generator_update": self.resolved.phase_a_generator_updates,
                },
                *(
                    [
                        {
                            "label": "Phase B end",
                            "generator_update": self.resolved.total_generator_updates,
                        }
                    ]
                    if self.resolved.phase_b_epochs
                    else []
                ),
            ],
            "workload": {
                "patch_tokens_per_frame": self.resolved.patch_tokens_per_frame,
                "score_input_frames": self.resolved.score_input_frames,
                "loss_future_frames": self.resolved.future_latent_frames,
                "sink_in_score_compute": True,
                "sink_in_loss": False,
            },
            "dry_run": self.options.dry_run,
            "smoke_mode": self.options.smoke_mode,
        }

    def _metric_output_path(self) -> Path:
        metric_path = Path(self.resolved.jsonl_path)
        if not metric_path.is_absolute():
            metric_path = self.options.output_dir / metric_path
        return metric_path.expanduser().resolve()

    def _prepare_metrics_lineage(self, resume_payload: Any) -> None:
        """Import or authenticate the immutable JSONL prefix in a checkpoint."""

        self.metric_path = self._metric_output_path()
        if resume_payload is None:
            return
        entries = {entry["name"]: entry for entry in resume_payload.manifest["files"]}
        entry = entries.get("metrics_lineage.jsonl")
        if not isinstance(entry, Mapping):
            raise RuntimeError(
                "Stage-2 resume checkpoint has no bound metrics lineage snapshot."
            )
        source = resume_payload.directory / "metrics_lineage.jsonl"
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(
                f"Stage-2 metrics lineage snapshot is not a regular file: {source}"
            )
        snapshot = source.read_bytes()
        if (
            len(snapshot) != entry["size"]
            or hashlib.sha256(snapshot).hexdigest() != entry["sha256"]
        ):
            raise RuntimeError(
                "Stage-2 metrics lineage snapshot changed after checkpoint load."
            )

        destination = self.metric_path
        checkpoint_directory = Path(resume_payload.directory).resolve()
        if (
            destination == source.resolve()
            or checkpoint_directory == destination
            or checkpoint_directory in destination.parents
        ):
            raise RuntimeError(
                "Stage-2 metrics output cannot mutate the immutable resume "
                f"checkpoint: {destination}"
            )
        if destination.is_symlink() or (
            destination.exists() and not destination.is_file()
        ):
            raise RuntimeError(
                f"Stage-2 metrics destination is not a regular file: {destination}"
            )
        if destination.exists():
            existing = destination.read_bytes()
            if not existing.startswith(snapshot):
                raise RuntimeError(
                    "Existing Stage-2 JSONL does not contain the checkpoint's "
                    "exact authenticated metrics prefix. Refusing to overwrite "
                    f"or splice {destination}."
                )
            return

        from utils.stage1_io import atomic_write_bytes

        destination.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(destination, snapshot)

    def _build_logger(self, resume_payload: Any) -> None:
        from utils.stage2_metrics import Stage2MetricsLogger

        metric_path = getattr(self, "metric_path", self._metric_output_path())
        lineage = (
            None
            if resume_payload is None
            else resume_payload.trainer_state["jsonl_lineage"]
        )
        parent = None if lineage is None else lineage["run_id"]
        next_attempt = 0 if lineage is None else lineage["next_attempt_index"]
        run_id_holder = [uuid.uuid4().hex if self.is_main_process else None]
        dist.broadcast_object_list(run_id_holder, src=0)
        run_id = run_id_holder[0]
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError("Stage-2 failed to broadcast one shared JSONL run_id.")
        self.logger = Stage2MetricsLogger(
            metric_path,
            experiment_id=f"stage2-{self.resolved.contract_hash()[:12]}",
            run_id=run_id,
            parent_run_id=parent,
            resume_from_logical_substep=self.state.successful_attempts,
            checkpoint_next_attempt_index=next_attempt,
            fsync_every_steps=self.resolved.fsync_every_steps,
            enabled=self.is_main_process,
            run_metadata=self._run_metadata(),
        )
        self.metric_path = metric_path

    def _checkpoint_provenance(self) -> dict[str, Any]:
        return {
            "assets": self.assets,
            "data": {
                "stage2_manifest_sha256": self.dataset.manifest["manifest_sha256"],
                "source_manifest_sha256": self.dataset.source_manifest[
                    "manifest_sha256"
                ],
                "negative_manifest_sha256": self.dataset.negative_conditioning[
                    "manifest"
                ]["manifest_sha256"],
                "negative_artifact_sha256": self.dataset.negative_conditioning[
                    "manifest"
                ]["artifact"]["sha256"],
            },
            "lineage": {
                "parent_checkpoint": (
                    None
                    if self.resume_checkpoint is None
                    else str(self.resume_checkpoint)
                ),
                "parent_checkpoint_manifest_sha256": (
                    None
                    if self.resume_payload is None
                    else self.resume_payload.manifest["manifest_sha256"]
                ),
            },
            "smoke_probe": self._last_cycle_smoke_probe,
        }

    def _assert_live_checkpoint_quiescence(self) -> dict[str, bool]:
        """Prove that no partial substep state can enter a committed checkpoint."""

        active = getattr(self, "_active_logical_substep", None)
        if active is not None:
            raise RuntimeError(
                "Stage-2 checkpoint attempted while a logical substep is active: "
                f"{active}"
            )
        leaked_gradients: list[str] = []
        for role in ("generator", "real_score", "fake_score"):
            module = getattr(self.model, role)
            leaked_gradients.extend(
                f"{role}.{name}"
                for name, parameter in module.named_parameters()
                if parameter.grad is not None
            )
        if leaked_gradients:
            raise RuntimeError(
                "Stage-2 checkpoint boundary retained parameter gradients: "
                f"{leaked_gradients[:8]}"
            )

        # Batches, the sampled branch, and the per-attempt rollout/KV state are
        # deliberately local to `_run_one_logical_substep`.  The active marker
        # spans that entire call.  Also reject any future refactor that stores
        # one of those objects on Trainer and accidentally extends its lifetime.
        pending_runtime = []
        for name, value in vars(self).items():
            lowered = name.lower()
            if name == "_active_logical_substep":
                continue
            matches_pending_name = any(
                token in lowered
                for token in (
                    "pending_batch",
                    "active_batch",
                    "pending_branch",
                    "active_branch",
                    "rollout_state",
                    "active_transaction",
                    "pending_transaction",
                )
            )
            is_empty = (
                value is None
                or value is False
                or (isinstance(value, (tuple, list, dict)) and len(value) == 0)
            )
            if matches_pending_name and not is_empty:
                pending_runtime.append(name)
        if pending_runtime:
            raise RuntimeError(
                "Stage-2 checkpoint boundary retained attempt-local runtime state: "
                f"{sorted(pending_runtime)}"
            )
        return {
            "pending_gradients": False,
            "pending_batch": False,
            "pending_branch": False,
            "pending_rollout_kv": False,
            "pending_transaction": False,
        }

    def _run_tracked_logical_substep(self) -> dict[str, Any]:
        if getattr(self, "_active_logical_substep", None) is not None:
            raise RuntimeError("Stage-2 logical substeps cannot be nested")
        self._active_logical_substep = {
            "role": self.state.current_role,
            "cycle_substep": self.state.next_substep,
            "logical_substep_id": self.state.successful_attempts,
        }
        try:
            return self._run_one_logical_substep()
        finally:
            self._active_logical_substep = None

    def _snapshot_checkpoint_metrics_lineage(self) -> dict[str, Any]:
        snapshot = self.metric_path.read_bytes()
        if not snapshot or not snapshot.endswith(b"\n"):
            raise RuntimeError(
                "Stage-2 checkpoint requires a non-empty, newline-terminated "
                "metrics lineage snapshot."
            )
        self._checkpoint_metrics_lineage_snapshot = snapshot
        return {
            "run_id": self.logger.run_id,
            # The snapshot is taken before checkpoint_event; reserve exactly
            # that post-publication attempt in the checkpoint cursor.
            "next_attempt_index": self.logger.next_attempt_index + 1,
        }

    def _save_checkpoint(self) -> dict[str, Any]:
        from utils.stage2_checkpoint import (
            build_stage2_trainer_state,
            save_stage2_checkpoint,
        )

        self.state.assert_checkpointable(self.schedule)
        self._assert_live_checkpoint_quiescence()
        lineage = self._runtime_rank0_checked(
            "snapshot Stage-2 JSONL lineage",
            self._snapshot_checkpoint_metrics_lineage,
        )
        trainer_state = build_stage2_trainer_state(
            completed_generator_updates=self.state.completed_g,
            completed_fake_updates=self.state.completed_f,
            successful_attempts={
                "generator": self.state.completed_g,
                "fake_score": self.state.completed_f,
            },
            nonfinite_attempts=self.state.nonfinite_attempts_by_role,
            sampler_state=self.samplers.state_dict(),
            dataloader_generator_states={
                role: generator.get_state().clone()
                for role, generator in self.dataloader_generators.items()
            },
            run_id=lineage["run_id"],
            next_attempt_index=lineage["next_attempt_index"],
            contract_hash=self.resolved.contract_hash(),
            launch_hash=self.resolved.launch_hash(),
            cache_audit_launch_hash=self.cache_audit_launch_hash,
            phase_a_generator_updates=self.resolved.phase_a_generator_updates,
            phase_b_generator_updates=self.resolved.phase_b_generator_updates,
            phase_b_mode=self.resolved.phase_b_mode,
        )
        before = self._runtime_rank0_checked(
            "snapshot Stage-2 checkpoint retention set",
            lambda: sorted(
                child.name
                for child in self.options.output_dir.glob("checkpoint_stage2_g*")
                if child.is_dir()
            ),
        )
        destination = save_stage2_checkpoint(
            self.options.output_dir,
            trainer_state=trainer_state,
            metrics_lineage_snapshot=(
                self._checkpoint_metrics_lineage_snapshot
                if self.is_main_process
                else None
            ),
            resolved_config=self.resolved,
            generator_module=self.model.generator,
            fake_score_module=self.model.fake_score,
            generator_optimizer=self.optimizers["generator"],
            fake_score_optimizer=self.optimizers["fake_score"],
            generator_ema=self.generator_ema,
            generator_schema=self.lora_schemas["generator"],
            fake_score_schema=self.lora_schemas["fake_score"],
            dedicated_generators=self._checkpoint_dedicated_generators(),
            rank0_control_generators=self._rank0_control_generators(),
            provenance=self._checkpoint_provenance(),
            topology=self._checkpoint_topology(),
            shard_group=self.mesh.get_group("shard"),
            keep_last=self.resolved.keep_last_resumable,
            milestone_updates=self.resolved.milestone_generator_updates,
        )
        return self._runtime_rank0_checked(
            "read committed Stage-2 checkpoint event",
            lambda: self._checkpoint_event_payload(destination, before),
        )

    def _checkpoint_event_payload(
        self, destination: Path, before: Sequence[str]
    ) -> dict[str, Any]:
        manifest_path = destination / "checkpoint_manifest.json"
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        after = sorted(
            child.name
            for child in self.options.output_dir.glob("checkpoint_stage2_g*")
            if child.is_dir()
        )
        return {
            "path": str(destination),
            "manifest_sha256": manifest["manifest_sha256"],
            "total_bytes": sum(
                child.stat().st_size
                for child in destination.iterdir()
                if child.is_file()
            ),
            "retained": after,
            "removed": sorted(set(before) - set(after)),
        }

    def _maybe_visualize(self, *, require_complete: bool) -> None:
        if self.options.no_visualize:
            return
        self._runtime_rank0_checked(
            "Stage-2 metrics visualization",
            lambda: self._plot_metrics(require_complete=require_complete),
        )

    def _plot_metrics(self, *, require_complete: bool) -> list[Path]:
        from scripts.plot_stage2_training import plot_stage2_training

        return plot_stage2_training(
            self.metric_path,
            self.options.output_dir / "plots",
            include_dry_run=self.options.dry_run,
            require_complete=require_complete,
        )

    def _train_loop(self) -> None:
        cycle_elapsed: list[float] = []
        cycle_retry_elapsed: list[float] = []
        cycle_step_fields: list[Mapping[str, Any]] = []
        start_cycle = self.state.cycle
        cycle_nonfinite_start = self.state.nonfinite_attempts
        while self.state.completed_g < self.resolved.total_generator_updates:
            result = self._run_tracked_logical_substep()
            cycle_elapsed.append(float(result.get("total_elapsed", result["elapsed"])))
            cycle_retry_elapsed.append(float(result.get("retry_seconds", 0.0)))
            cycle_step_fields.append(result.get("fields", {}))
            if (
                self.state.current_role != "fake_score"
                or self.state.next_substep != "F1"
            ):
                continue
            cycle_nonfinite_attempts = (
                self.state.nonfinite_attempts - cycle_nonfinite_start
            )
            smoke_probe = self._validate_smoke_cycle(
                cycle_step_fields,
                nonfinite_attempts=cycle_nonfinite_attempts,
            )
            if smoke_probe is not None:
                smoke_probe = {
                    **smoke_probe,
                    "parent_next_f1_probe_consumed": bool(
                        getattr(self, "_smoke_parent_probe_consumed", False)
                    ),
                }
                if self.options.smoke_mode in {"C0", "C1"}:
                    smoke_probe["next_f1_probe"] = self._capture_next_f1_probe()
            self._last_cycle_smoke_probe = smoke_probe
            cycle_fields = {
                **self._metric_clock("G", self.state.successful_attempts - 1),
                "cycle_seconds": sum(cycle_elapsed),
                "fake_score_seconds": sum(cycle_elapsed[:5]),
                "generator_seconds": cycle_elapsed[5],
                "fake_score_time_fraction": sum(cycle_elapsed[:5]) / sum(cycle_elapsed),
                "generator_time_fraction": cycle_elapsed[5] / sum(cycle_elapsed),
                "cycle_samples_per_second": (
                    6 * self.resolved.global_batch_size / sum(cycle_elapsed)
                ),
                "fake_score_samples_per_second": (
                    5 * self.resolved.global_batch_size / sum(cycle_elapsed[:5])
                ),
                "generator_samples_per_second": (
                    self.resolved.global_batch_size / cycle_elapsed[5]
                ),
                "successful_substeps": 6,
                "nonfinite_attempts": cycle_nonfinite_attempts,
                "retry_seconds": sum(cycle_retry_elapsed),
                "dry_run": self.options.dry_run,
                "smoke_mode": self.options.smoke_mode,
                "smoke_acceptance": smoke_probe,
            }
            self._append_metric("cycle_summary", cycle_fields)
            cycle_elapsed = []
            cycle_retry_elapsed = []
            cycle_step_fields = []
            cycle_nonfinite_start = self.state.nonfinite_attempts
            should_save = not self.options.no_save and (
                self.options.smoke_mode in {"C0", "C1"}
                or self.state.completed_g in self.resolved.milestone_generator_updates
                or self.state.completed_g
                % self.resolved.checkpoint_interval_generator_updates
                == 0
            )
            if should_save:
                started = time.perf_counter()
                event = self._save_checkpoint()
                elapsed = time.perf_counter() - started
                self._append_metric(
                    "checkpoint_event",
                    {
                        **self._metric_clock("G", self.state.successful_attempts - 1),
                        "checkpoint_path": str(event["path"]),
                        "checkpoint_sha256": event["manifest_sha256"],
                        "checkpoint_bytes": event["total_bytes"],
                        "success_marker": True,
                        "retained": event.get("retained", []),
                        "removed": event.get("removed", []),
                        "elapsed_seconds": elapsed,
                        "dry_run": self.options.dry_run,
                        "smoke_mode": self.options.smoke_mode,
                    },
                )
            if self.options.dry_run and self.state.cycle == start_cycle + 1:
                return

    def train(self) -> None:
        self._initialize_distributed()
        try:
            from utils.stage2_fsdp2 import build_stage2_fsdp2_device_mesh

            self._rank0_checked(
                "create Stage-2 output directory",
                lambda: self.options.output_dir.mkdir(parents=True, exist_ok=True),
            )
            self._world_checked("seed Stage-2 RNG streams", self._seed_training_rngs)
            self.resume_checkpoint = self._rank0_checked(
                "discover Stage-2 resume checkpoint",
                self._discover_resume_checkpoint,
            )
            mesh, self.runtime_audit = build_stage2_fsdp2_device_mesh()
            self.mesh = mesh
            resume_payload = self._load_resume_before_roles()
            self.resume_payload = resume_payload
            self.cache_audit_launch_hash = (
                self.resolved.launch_hash()
                if resume_payload is None
                else resume_payload.trainer_state["cache_audit_launch_hash"]
            )
            self._world_checked("Stage-2 data runtime", self._build_data_runtime)
            self._world_checked(
                "Stage-2 resume runtime provenance",
                lambda: self._audit_resume_runtime_bindings(resume_payload),
            )
            self._initialize_roles(mesh, resume_payload)
            self._world_checked(
                "build/restore Stage-2 optimizers and EMA",
                lambda: self._build_optimizers_and_ema(resume_payload),
            )
            self._world_checked("build Stage-2 rollout", self._build_rollout)
            self._world_checked(
                "restore Stage-2 logical state",
                lambda: self._restore_logical_state(resume_payload),
            )
            # RNG restore is intentionally last, after roles, optimizers, EMA,
            # counters, samplers, and loader generators are fully constructed.
            self._world_checked(
                "restore Stage-2 RNG state last",
                lambda: self._restore_rng_last(resume_payload),
            )
            self._rank0_checked(
                "prepare Stage-2 JSONL checkpoint lineage",
                lambda: self._prepare_metrics_lineage(resume_payload),
            )
            self._world_checked(
                "build Stage-2 JSONL logger",
                lambda: self._build_logger(resume_payload),
            )
            self._train_loop()
            complete = (
                not self.options.dry_run
                and self.state.completed_g == self.resolved.total_generator_updates
            )
            self._append_metric(
                "run_end",
                {
                    **self._metric_clock("G", self.state.successful_attempts - 1),
                    "status": "complete" if complete else "smoke_complete",
                    "dry_run": self.options.dry_run,
                    "smoke_mode": self.options.smoke_mode,
                },
            )
            self._runtime_rank0_checked("close Stage-2 JSONL", self.logger.close)
            self._maybe_visualize(require_complete=complete)
            dist.barrier()
        except Exception:
            if self.logger is not None and getattr(self, "is_main_process", False):
                self.logger.close()
            # Never convert a failed training job into exit status zero.
            raise
        finally:
            if self._initialized_process_group and dist.is_initialized():
                dist.destroy_process_group()


__all__ = ["Trainer"]
