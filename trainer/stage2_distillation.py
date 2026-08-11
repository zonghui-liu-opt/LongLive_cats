"""Stage-2-only 5F -> 1G DMD/DFD trainer.

This orchestrator deliberately owns no diffusion math.  It wires the audited
F25 dataset, independent G/F samplers, random-exit rollout, the explicit
``Stage2DMD`` role methods, two LoRA-only optimizers, CPU EMA, JSONL metrics,
and cycle-boundary checkpoints into the one legal state transition sequence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
import copy
from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import time
from typing import Any
import uuid

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


def _repo_identity(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ("git", *args),
            cwd=project_root,
            text=True,
            capture_output=True,
            check=False,
            env={
                key: value
                for key, value in os.environ.items()
                if key not in {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"}
            },
        )
        if result.returncode:
            raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
        return result.stdout

    top = Path(run("rev-parse", "--show-toplevel").strip()).resolve()
    if top != project_root.resolve():
        raise RuntimeError(
            f"Git top-level mismatch: expected={project_root}, actual={top}"
        )
    commit = run("rev-parse", "HEAD").strip()
    if len(commit) != 40:
        raise RuntimeError("Stage-2 requires one full Git commit identity.")
    dirty = run("status", "--porcelain=v1", "--untracked-files=all").splitlines()
    ignored = run(
        "ls-files", "--others", "--ignored", "--exclude-standard"
    ).splitlines()
    if dirty or ignored:
        raise RuntimeError(
            "Stage-2 formal training requires a clean checkout with no ignored "
            f"runtime files; dirty={dirty[:8]}, ignored={ignored[:8]}."
        )
    return {"commit": commit, "worktree_clean": True, "ignored_files_absent": True}


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
        if not output_dir:
            raise ValueError(
                "Stage-2 requires an explicit --logdir outside the clean checkout."
            )
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
        self.project_root = Path(__file__).resolve().parents[1]
        try:
            self.options.output_dir.relative_to(self.project_root)
        except ValueError:
            pass
        else:
            raise ValueError("Stage-2 --logdir must be outside the clean Git checkout.")
        if self.resolved.torch_compile:
            raise ValueError(
                "Stage-2 torch.compile remains a post-eager profiling candidate; "
                "the production trainer refuses to enable it before H100 parity."
            )
        self._initialized_process_group = False
        self.logger = None
        self.resume_payload = None
        self._last_cycle_smoke_probe = None

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
            explicit_path = Path(explicit).expanduser().resolve()
            if discovered is not None and explicit_path != Path(discovered).resolve():
                raise RuntimeError(
                    "Explicit Stage-2 resume checkpoint differs from the newest "
                    f"checkpoint in --logdir: explicit={explicit_path}, latest={discovered}."
                )
            selected = explicit_path
        else:
            selected = Path(discovered).resolve() if discovered is not None else None
        if self.options.smoke_mode in {"C1", "C2"} and selected is None:
            raise RuntimeError(
                f"Stage-2 smoke {self.options.smoke_mode} requires a complete "
                "checkpoint from the preceding smoke cycle."
            )
        return selected

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

    def _audit_resume_runtime_bindings(
        self, resume_payload: Any, git_identity: Mapping[str, Any]
    ) -> None:
        if resume_payload is None:
            return
        provenance = resume_payload.provenance
        if provenance.get("git") != dict(git_identity):
            raise RuntimeError(
                "Stage-2 resume checkpoint Git identity differs from this clean checkout."
            )
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
            )
        self.generator_ema = TrainableShardedEMA(
            self.model.generator,
            decay=self.resolved.ema_decay,
            start_step=self.resolved.ema_initialize_at_completed_generator_update,
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
                ),
            )

        numerator, count = self._reduce_loss(local_numerator, local_count)
        merged = self._reduce_diagnostics(diagnostics)
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
        self, step_fields: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any] | None:
        if not self.options.dry_run:
            return None
        if len(step_fields) != 6:
            raise RuntimeError("Stage-2 smoke must contain exactly 5F+1G train steps.")
        if getattr(self, "device", torch.device("cpu")).type != "cuda":
            return {
                "smoke_mode": self.options.smoke_mode,
                "status": "PASS_CPU_TEST_SEAM",
                "live_allocated_gib_max": 0.0,
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
            result = self._run_update_attempt(
                role=role, batches=batches, exits=exits, branch=branch
            )
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
                ema_action = self._runtime_world_checked(
                    "generator EMA update",
                    lambda: self.generator_ema.update_after_step(
                        self.model.generator, next_completed_g
                    ),
                )
                ema_seconds = time.perf_counter() - ema_started
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
            dedicated_generators=self._checkpoint_dedicated_generators(),
            rank0_control_generators=self._rank0_control_generators(),
            require_cuda_topology=True,
        )

    def _run_metadata(self, git_identity: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "git_commit": git_identity["commit"],
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

    def _build_logger(
        self, resume_payload: Any, git_identity: Mapping[str, Any]
    ) -> None:
        from utils.stage2_metrics import Stage2MetricsLogger

        metric_path = Path(self.resolved.jsonl_path)
        if not metric_path.is_absolute():
            metric_path = self.options.output_dir / metric_path
        metric_path = metric_path.expanduser().resolve()
        project_root = getattr(
            self, "project_root", Path(__file__).resolve().parents[1]
        )
        try:
            metric_path.relative_to(project_root)
        except ValueError:
            pass
        else:
            raise ValueError(
                "Stage-2 JSONL path must stay outside the clean Git checkout."
            )
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
            run_metadata=self._run_metadata(git_identity),
        )
        self.metric_path = metric_path

    def _checkpoint_provenance(self, git_identity: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "git": dict(git_identity),
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

    def _save_checkpoint(self, git_identity: Mapping[str, Any]) -> dict[str, Any]:
        from utils.stage2_checkpoint import (
            build_stage2_trainer_state,
            save_stage2_checkpoint,
        )

        self.state.assert_checkpointable(self.schedule)
        lineage = self._runtime_rank0_checked(
            "snapshot Stage-2 JSONL lineage",
            lambda: {
                "run_id": self.logger.run_id,
                # Reserve the checkpoint_event appended only after _SUCCESS.
                "next_attempt_index": self.logger.next_attempt_index + 1,
            },
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
            provenance=self._checkpoint_provenance(git_identity),
            topology=self._checkpoint_topology(),
            shard_group=self.mesh.get_group("shard"),
            keep_last=self.resolved.keep_last_resumable,
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

    def _train_loop(self, git_identity: Mapping[str, Any]) -> None:
        cycle_elapsed: list[float] = []
        cycle_retry_elapsed: list[float] = []
        cycle_step_fields: list[Mapping[str, Any]] = []
        start_cycle = self.state.cycle
        cycle_nonfinite_start = self.state.nonfinite_attempts
        while self.state.completed_g < self.resolved.total_generator_updates:
            result = self._run_one_logical_substep()
            cycle_elapsed.append(float(result.get("total_elapsed", result["elapsed"])))
            cycle_retry_elapsed.append(float(result.get("retry_seconds", 0.0)))
            cycle_step_fields.append(result.get("fields", {}))
            if (
                self.state.current_role != "fake_score"
                or self.state.next_substep != "F1"
            ):
                continue
            smoke_probe = self._validate_smoke_cycle(cycle_step_fields)
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
                "nonfinite_attempts": (
                    self.state.nonfinite_attempts - cycle_nonfinite_start
                ),
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
                or self.state.completed_g
                % self.resolved.checkpoint_interval_generator_updates
                == 0
            )
            if should_save:
                started = time.perf_counter()
                event = self._save_checkpoint(git_identity)
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

            git_identity = self._rank0_checked(
                "clean Git identity", lambda: _repo_identity(self.project_root)
            )
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
                lambda: self._audit_resume_runtime_bindings(
                    resume_payload, git_identity
                ),
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
            self._world_checked(
                "build Stage-2 JSONL logger",
                lambda: self._build_logger(resume_payload, git_identity),
            )
            self._train_loop(git_identity)
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
