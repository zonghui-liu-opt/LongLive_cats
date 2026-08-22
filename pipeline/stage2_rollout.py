"""Stage-2 on-policy rollout with an explicit, autograd-safe KV contract.

This module owns only the generator rollout used by both fake-score and
generator updates.  It deliberately does not choose losses, optimizers,
phases, or checkpoints.  The caller must provide one ``exit_step`` for an
entire microbatch; sampling that step twice inside the pipeline is forbidden.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, ClassVar

import torch
import torch.distributed as dist

from pipeline.stage2_rollout_profile import (
    Stage2RolloutSpec,
    resolve_stage2_rollout_profile,
    resolve_stage2_shift5_schedule,
)
from utils.stage2_cross_kv import Stage2CrossKVInitState

STAGE2_K2_SHIFT5_TIMESTEPS = (999, 833)
STAGE2_K4_SHIFT5_TIMESTEPS = (999, 937, 833, 624)
if resolve_stage2_shift5_schedule(2).timesteps != STAGE2_K2_SHIFT5_TIMESTEPS:
    raise AssertionError("Stage-2 K2 shift=5 golden timetable drifted")
if resolve_stage2_shift5_schedule(4).timesteps != STAGE2_K4_SHIFT5_TIMESTEPS:
    raise AssertionError("Stage-2 K4 shift=5 golden timetable drifted")
_EXIT_ROLES = ("generator", "fake_score")


def _strict_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _per_sample_sha256(value: torch.Tensor) -> tuple[str, ...]:
    if not isinstance(value, torch.Tensor) or value.ndim == 0:
        raise ValueError("per-sample tensor identity requires a batched tensor")
    return tuple(_tensor_sha256(value[index]) for index in range(value.shape[0]))


def draw_stage2_exit_schedule(
    *,
    accumulation_steps: int,
    num_denoising_steps: int,
    generator: torch.Generator,
    mode: str = "stratified_uniform",
    device: torch.device | str | None = None,
    synchronize_ranks: bool = True,
) -> tuple[int, ...]:
    """Draw one explicit exit per accumulation microbatch.

    In stratified mode every consecutive group of ``K`` microbatches is one
    independent permutation of ``0..K-1``.  Rank 0 draws and broadcasts the
    finished schedule, so every rank, sample, and chunk uses the same exit for
    a given microbatch.
    """

    accumulation_steps = _strict_nonnegative_int(
        accumulation_steps, "accumulation_steps"
    )
    num_denoising_steps = _strict_nonnegative_int(
        num_denoising_steps, "num_denoising_steps"
    )
    if accumulation_steps == 0 or num_denoising_steps == 0:
        raise ValueError("accumulation_steps and num_denoising_steps must be positive")
    if not isinstance(generator, torch.Generator):
        raise TypeError("generator must be an explicit torch.Generator")
    if mode not in {"stratified_uniform", "iid_uniform"}:
        raise ValueError(f"unsupported Stage-2 exit mode: {mode!r}")
    if mode == "stratified_uniform" and accumulation_steps % num_denoising_steps:
        raise ValueError(
            "stratified exit sampling requires accumulation_steps % "
            "num_denoising_steps == 0"
        )

    distributed = synchronize_ranks and dist.is_available() and dist.is_initialized()
    backend = str(dist.get_backend()).lower() if distributed else ""
    if device is None:
        if "nccl" in backend:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "NCCL Stage-2 exit broadcast requires a visible CUDA device"
                )
            target_device = torch.device("cuda", torch.cuda.current_device())
        else:
            target_device = torch.device("cpu")
    else:
        target_device = torch.device(device)
    if distributed and "nccl" in backend and target_device.type != "cuda":
        raise ValueError(
            "NCCL cannot broadcast the Stage-2 exit schedule from CPU; "
            "use the current-rank CUDA device or leave device=None"
        )
    rank = dist.get_rank() if distributed else 0
    if rank == 0:
        if mode == "iid_uniform":
            schedule = torch.randint(
                num_denoising_steps,
                (accumulation_steps,),
                generator=generator,
                dtype=torch.long,
                device="cpu",
            )
        else:
            groups = [
                torch.randperm(
                    num_denoising_steps,
                    generator=generator,
                    dtype=torch.long,
                    device="cpu",
                )
                for _ in range(accumulation_steps // num_denoising_steps)
            ]
            schedule = torch.cat(groups)
    else:
        schedule = torch.empty(accumulation_steps, dtype=torch.long, device="cpu")

    schedule = schedule.to(device=target_device)
    if distributed:
        dist.broadcast(schedule, src=0)
    values = tuple(int(item) for item in schedule.to(device="cpu").tolist())
    if any(item < 0 or item >= num_denoising_steps for item in values):
        raise RuntimeError("broadcast Stage-2 exit schedule is out of range")
    if mode == "stratified_uniform":
        expected = list(range(num_denoising_steps))
        for offset in range(0, accumulation_steps, num_denoising_steps):
            if sorted(values[offset : offset + num_denoising_steps]) != expected:
                raise RuntimeError("stratified Stage-2 exit schedule lost a stratum")
    return values


class Stage2ExitRNGStreams:
    """Two independent CPU RNG streams for G and F exit schedules."""

    _OFFSETS: ClassVar[dict[str, int]] = {
        "generator": 0x475F45584954,
        "fake_score": 0x465F45584954,
    }

    def __init__(self, seed: int):
        seed = _strict_nonnegative_int(seed, "seed")
        self._generators: dict[str, torch.Generator] = {}
        modulus = (1 << 63) - 1
        for role in _EXIT_ROLES:
            stream = torch.Generator(device="cpu")
            stream.manual_seed((seed + self._OFFSETS[role]) % modulus)
            self._generators[role] = stream

    def draw(
        self,
        role: str,
        *,
        accumulation_steps: int,
        num_denoising_steps: int = 4,
        mode: str = "stratified_uniform",
        device: torch.device | str | None = None,
        synchronize_ranks: bool = True,
    ) -> tuple[int, ...]:
        if role not in self._generators:
            raise ValueError(f"unknown Stage-2 exit RNG role: {role!r}")
        return draw_stage2_exit_schedule(
            accumulation_steps=accumulation_steps,
            num_denoising_steps=num_denoising_steps,
            generator=self._generators[role],
            mode=mode,
            device=device,
            synchronize_ranks=synchronize_ranks,
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            role: generator.get_state().clone()
            for role, generator in self._generators.items()
        }

    def generator(self, role: str) -> torch.Generator:
        """Return the owned CPU stream for exact checkpoint capture only."""

        if role not in self._generators:
            raise ValueError(f"unknown Stage-2 exit RNG role: {role!r}")
        return self._generators[role]

    def load_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        if set(state) != set(_EXIT_ROLES):
            raise ValueError("Stage-2 exit RNG state must contain generator/fake_score")
        for role in _EXIT_ROLES:
            value = state[role]
            if (
                not torch.is_tensor(value)
                or value.dtype != torch.uint8
                or value.ndim != 1
            ):
                raise TypeError(f"invalid Stage-2 exit RNG state for {role}")
            self._generators[role].set_state(value.detach().cpu().clone())


@dataclass
class Stage2RolloutState:
    self_kv: list[dict[str, Any]]
    cross_kv: list[dict[str, Any]]
    initial_latent_sha256: str
    batch_size: int
    dtype: torch.dtype
    device: torch.device
    initial_latent_sha256_per_sample: tuple[str, ...]
    profile_name: str
    episode_index: int = 0
    episode_complete: bool = False


@dataclass(frozen=True)
class Stage2LayerKVSnapshot:
    k: torch.Tensor
    v: torch.Tensor


@dataclass(frozen=True)
class Stage2SinkPrefixSnapshot:
    """Detached episode-1 prefix used as the permanent episode-2 sink."""

    source_profile_name: str
    target_sink_frames: int
    frame_seq_length: int
    layers: tuple[Stage2LayerKVSnapshot, ...]
    clean_latents: torch.Tensor
    clean_latent_sha256_per_sample: tuple[str, ...]
    initial_latent_sha256: str
    initial_latent_sha256_per_sample: tuple[str, ...]
    batch_size: int
    dtype: torch.dtype
    device: torch.device
    source_episode_index: int


@dataclass(frozen=True)
class Stage2RolloutResult:
    latents: torch.Tensor
    exit_step: int
    requires_grad: bool
    scheduler_timesteps: tuple[int, ...]
    scheduler_sigmas: tuple[float, ...]
    chunk_timesteps: tuple[tuple[int, ...], ...]
    chunk_sigmas: tuple[tuple[float, ...], ...]
    cache_audit: Mapping[str, Any]
    rollout_mode: str
    chunk_trace: tuple[Mapping[str, Any], ...]
    initial_latent_sha256_per_sample: tuple[str, ...]
    generated_latent_sha256_per_sample: tuple[str, ...]
    prefix_snapshot: Stage2SinkPrefixSnapshot | None


class Stage2RolloutPipeline:
    """Generate exactly 24 new latent frames under one named rollout spec."""

    def __init__(
        self,
        generator: Any,
        *,
        spec: Stage2RolloutSpec | str | None = None,
        generated_episode_frames: int | None = None,
        chunk_frames: int | None = None,
        local_window_frames: int | None = None,
        global_sink_frames: int | None = None,
        num_denoising_steps: int | None = None,
        timestep_shift: float | None = None,
        num_train_timesteps: int = 1000,
        frame_seq_length: int = 390,
        scheduler_factory: Callable[[], Any] | None = None,
    ):
        baseline = resolve_stage2_rollout_profile("baseline_c8w16k4s1")
        if spec is None:
            legacy_values = {
                "generated_episode_frames": generated_episode_frames,
                "chunk_frames": chunk_frames,
                "local_window_frames": local_window_frames,
                "global_sink_frames": global_sink_frames,
                "num_denoising_steps": num_denoising_steps,
                "timestep_shift": timestep_shift,
            }
            expected = {
                "generated_episode_frames": baseline.generated_episode_frames,
                "chunk_frames": baseline.chunk_frames,
                "local_window_frames": baseline.local_window_frames,
                "global_sink_frames": baseline.global_sink_frames,
                "num_denoising_steps": baseline.num_denoising_steps,
                "timestep_shift": baseline.timestep_shift,
            }
            wrong = {
                name: {"expected": expected[name], "actual": value}
                for name, value in legacy_values.items()
                if value is not None and value != expected[name]
            }
            if wrong:
                raise ValueError(
                    "non-baseline Stage-2 rollout dimensions require an explicit "
                    f"named spec; mismatches={wrong}"
                )
            selected_spec = baseline
        elif isinstance(spec, str):
            selected_spec = resolve_stage2_rollout_profile(spec)
        elif isinstance(spec, Stage2RolloutSpec):
            selected_spec = spec
        else:
            raise TypeError("spec must be a Stage2RolloutSpec, profile name, or None")
        if spec is not None and any(
            value is not None
            for value in (
                generated_episode_frames,
                chunk_frames,
                local_window_frames,
                global_sink_frames,
                num_denoising_steps,
                timestep_shift,
            )
        ):
            raise ValueError(
                "named Stage-2 rollout specs cannot be mixed with overrides"
            )

        self.generator = generator
        self.spec = selected_spec
        self.generated_episode_frames = selected_spec.generated_episode_frames
        self.chunk_frames = selected_spec.chunk_frames
        self.local_window_frames = selected_spec.local_window_frames
        self.global_sink_frames = selected_spec.global_sink_frames
        self.num_denoising_steps = selected_spec.num_denoising_steps
        self.timestep_shift = selected_spec.timestep_shift
        self.num_train_timesteps = int(num_train_timesteps)
        self.frame_seq_length = int(frame_seq_length)
        self._scheduler_factory = scheduler_factory
        self.history_frames = self.local_window_frames - self.chunk_frames
        self.physical_kv_capacity_frames = (
            self.global_sink_frames + self.local_window_frames
        )
        self.cache_capacity_tokens = (
            self.physical_kv_capacity_frames * self.frame_seq_length
        )
        self.sink_tokens = self.global_sink_frames * self.frame_seq_length
        self._validate_contract()
        self.num_chunks = selected_spec.num_chunks
        model = self._causal_backbone()
        if bool(getattr(model, "is_gradient_checkpointing", False)) or bool(
            getattr(model, "gradient_checkpointing", False)
        ):
            raise RuntimeError(
                "Stage-2 generator cache rollout forbids activation checkpointing"
            )

    @classmethod
    def from_resolved_config(
        cls,
        generator: Any,
        resolved: Any,
        *,
        scheduler_factory: Callable[[], Any] | None = None,
    ) -> Stage2RolloutPipeline:
        spec = resolve_stage2_rollout_profile("baseline_c8w16k4s1")
        actual = (
            int(resolved.generated_episode_frames),
            int(resolved.chunk_frames),
            int(resolved.local_window_frames),
            int(resolved.global_sink_frames),
            int(resolved.num_denoising_steps),
            float(resolved.rollout_timestep_shift),
        )
        expected = (
            spec.generated_episode_frames,
            spec.chunk_frames,
            spec.local_window_frames,
            spec.global_sink_frames,
            spec.num_denoising_steps,
            spec.timestep_shift,
        )
        if actual != expected:
            raise ValueError(
                "resolved Stage-2 training config is not the named baseline rollout: "
                f"expected={expected}, actual={actual}"
            )
        return cls(
            generator,
            spec=spec,
            num_train_timesteps=resolved.num_train_timesteps,
            frame_seq_length=resolved.patch_tokens_per_frame,
            scheduler_factory=scheduler_factory,
        )

    def _validate_contract(self) -> None:
        integer_fields = {
            "generated_episode_frames": self.generated_episode_frames,
            "chunk_frames": self.chunk_frames,
            "local_window_frames": self.local_window_frames,
            "global_sink_frames": self.global_sink_frames,
            "num_denoising_steps": self.num_denoising_steps,
            "num_train_timesteps": self.num_train_timesteps,
            "frame_seq_length": self.frame_seq_length,
        }
        if any(value <= 0 for value in integer_fields.values()):
            raise ValueError(
                f"Stage-2 rollout dimensions must be positive: {integer_fields}"
            )
        if self.generated_episode_frames != 24 or self.frame_seq_length != 390:
            raise ValueError(
                "Stage-2 rollout requires 24 generated frames and 390 tokens/frame"
            )
        if self.num_train_timesteps != 1000:
            raise ValueError("Stage-2 rollout requires 1000 training timesteps")
        try:
            resolve_stage2_shift5_schedule(self.num_denoising_steps)
        except ValueError as exc:
            raise ValueError(
                "Stage-2 rollout supports only native K1-K8 UniPC"
            ) from exc
        if self.timestep_shift != 5.0:
            raise ValueError("Stage-2 rollout requires UniPC shift=5")
        derived = (
            self.history_frames,
            self.physical_kv_capacity_frames,
            self.num_chunks if hasattr(self, "num_chunks") else self.spec.num_chunks,
        )
        expected = (
            self.spec.history_frames,
            self.spec.physical_kv_capacity_frames,
            self.spec.num_chunks,
        )
        if derived != expected:
            raise AssertionError(
                f"Stage-2 rollout derived topology drifted: expected={expected}, actual={derived}"
            )

    def _transformer(self) -> Any:
        model = getattr(self.generator, "model", None)
        return model if model is not None else self.generator

    def _causal_backbone(self) -> Any:
        model = self._transformer()
        candidates = model.modules() if hasattr(model, "modules") else (model,)
        for candidate in candidates:
            if all(
                hasattr(candidate, name)
                for name in ("num_layers", "num_heads", "dim", "text_len")
            ):
                return candidate
        raise ValueError("generator does not expose one causal Wan backbone")

    @contextmanager
    def _generator_attention_runtime(self) -> Iterator[None]:
        """Install this profile only for one episode and restore shared state."""

        model = self._causal_backbone()
        if bool(getattr(model, "is_gradient_checkpointing", False)) or bool(
            getattr(model, "gradient_checkpointing", False)
        ):
            raise RuntimeError(
                "Stage-2 generator cache rollout forbids activation checkpointing"
            )
        missing = object()
        saved: list[tuple[Any, str, Any]] = []
        seen: set[tuple[int, str]] = set()

        def install(
            target: Any, name: str, value: Any, *, create: bool = False
        ) -> None:
            key = (id(target), name)
            if key in seen or (not create and not hasattr(target, name)):
                return
            seen.add(key)
            saved.append((target, name, getattr(target, name, missing)))
            setattr(target, name, value)

        try:
            # W excludes the sink; physical attention/cache span is S+W.
            install(
                model,
                "local_attn_size",
                self.physical_kv_capacity_frames,
                create=True,
            )
            install(model, "sink_size", self.global_sink_frames, create=True)
            install(model, "global_sink_size", self.global_sink_frames, create=True)
            for module in model.modules() if hasattr(model, "modules") else ():
                install(module, "local_attn_size", self.physical_kv_capacity_frames)
                install(module, "max_attention_size", self.cache_capacity_tokens)
                install(module, "sink_size", self.global_sink_frames)
                install(module, "global_sink_size", self.global_sink_frames)
            yield
        finally:
            for target, name, previous in reversed(saved):
                if previous is missing:
                    delattr(target, name)
                else:
                    setattr(target, name, previous)

    def _new_scheduler(self, device: torch.device) -> tuple[Any, tuple[int, ...]]:
        if self._scheduler_factory is None:
            from wan_5b.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

            scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False,
            )
        else:
            scheduler = self._scheduler_factory()
        scheduler.set_timesteps(
            self.num_denoising_steps,
            device=device,
            shift=self.timestep_shift,
        )
        raw_timesteps = tuple(
            float(value) for value in scheduler.timesteps.detach().cpu().tolist()
        )
        reference = resolve_stage2_shift5_schedule(self.num_denoising_steps)
        expected_timesteps = reference.timesteps
        if raw_timesteps != tuple(float(value) for value in expected_timesteps):
            raise RuntimeError(
                "Stage-2 UniPC timetable drifted: "
                f"expected={expected_timesteps}, actual={raw_timesteps}"
            )
        timesteps = tuple(int(value) for value in raw_timesteps)
        terminal_sigma = float(scheduler.sigmas[-1].detach().cpu())
        if terminal_sigma != 0.0:
            raise RuntimeError(
                f"Stage-2 UniPC terminal sigma must be 0, got {terminal_sigma}"
            )
        sigma_values = tuple(
            float(value) for value in scheduler.sigmas.detach().float().cpu().tolist()
        )
        sigma_fp32_bits = tuple(
            struct.pack(">f", value).hex() for value in sigma_values
        )
        if sigma_fp32_bits != reference.sigma_fp32_bits:
            raise RuntimeError(
                "Stage-2 UniPC sigma schedule drifted: "
                f"expected_fp32_bits={reference.sigma_fp32_bits}, "
                f"actual_fp32_bits={sigma_fp32_bits}"
            )
        if len(sigma_values) != self.num_denoising_steps + 1:
            raise RuntimeError(
                "Stage-2 UniPC must expose one sigma per denoising step plus "
                f"the terminal zero, got {sigma_values}"
            )
        if not all(
            torch.isfinite(torch.tensor(value)).item() for value in sigma_values
        ):
            raise RuntimeError(
                "Stage-2 UniPC sigma schedule contains non-finite values"
            )
        if any(left <= right for left, right in pairwise(sigma_values)):
            raise RuntimeError(
                f"Stage-2 UniPC sigmas must strictly decrease to zero: {sigma_values}"
            )
        return scheduler, timesteps

    def _allocate_state(self, initial_latent: torch.Tensor) -> Stage2RolloutState:
        if initial_latent.ndim != 5 or initial_latent.shape[1] != 1:
            raise ValueError("initial_latent must have shape [B,1,C,H,W]")
        if not initial_latent.is_floating_point():
            raise TypeError("initial_latent must be floating point")
        if initial_latent.dtype != torch.bfloat16:
            raise TypeError(
                "Stage-2 rollout initial_latent and persistent KV must be bfloat16"
            )
        return self._allocate_state_from_metadata(
            batch_size=int(initial_latent.shape[0]),
            dtype=initial_latent.dtype,
            device=initial_latent.device,
            initial_latent_sha256=_tensor_sha256(initial_latent),
            initial_latent_sha256_per_sample=_per_sample_sha256(initial_latent),
            episode_index=0,
        )

    def _allocate_state_from_metadata(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        initial_latent_sha256: str,
        initial_latent_sha256_per_sample: tuple[str, ...],
        episode_index: int,
    ) -> Stage2RolloutState:
        model = self._causal_backbone()
        num_layers = int(
            getattr(model, "num_layers", len(getattr(model, "blocks", ())))
        )
        num_heads = int(getattr(model, "num_heads", 0))
        dim = int(getattr(model, "dim", 0))
        if num_layers <= 0 or num_heads <= 0 or dim <= 0 or dim % num_heads:
            raise ValueError(
                "generator model does not expose a valid layer/head contract"
            )
        head_dim = dim // num_heads
        if batch_size <= 0 or len(initial_latent_sha256_per_sample) != batch_size:
            raise ValueError("Stage-2 state hash count differs from its batch size")
        if dtype != torch.bfloat16:
            raise TypeError("Stage-2 persistent KV state must be bfloat16")
        self_kv = []
        for _ in range(num_layers):
            self_kv.append(
                {
                    "k": torch.zeros(
                        batch_size,
                        self.cache_capacity_tokens,
                        num_heads,
                        head_dim,
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        batch_size,
                        self.cache_capacity_tokens,
                        num_heads,
                        head_dim,
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.zeros(1, dtype=torch.long, device=device),
                    "local_end_index": torch.zeros(1, dtype=torch.long, device=device),
                    "pinned_start": torch.full(
                        (1,), -1, dtype=torch.long, device=device
                    ),
                    "pinned_len": torch.zeros(1, dtype=torch.long, device=device),
                }
            )
        text_len = int(getattr(model, "text_len", 512))
        cross_kv = [
            {
                "k": torch.zeros(
                    batch_size,
                    text_len,
                    num_heads,
                    head_dim,
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.zeros(
                    batch_size,
                    text_len,
                    num_heads,
                    head_dim,
                    dtype=dtype,
                    device=device,
                ),
                "stage2_state": Stage2CrossKVInitState(),
                "stage2_enabled": True,
            }
            for _ in range(num_layers)
        ]
        return Stage2RolloutState(
            self_kv=self_kv,
            cross_kv=cross_kv,
            initial_latent_sha256=initial_latent_sha256,
            batch_size=batch_size,
            dtype=dtype,
            device=device,
            initial_latent_sha256_per_sample=initial_latent_sha256_per_sample,
            profile_name=self.spec.name,
            episode_index=episode_index,
        )

    @staticmethod
    def _generator_outputs(value: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(value, tuple) or len(value) < 2:
            raise TypeError("Stage-2 generator must return (raw_flow, x0_pred)")
        raw_flow, x0_pred = value[:2]
        if not torch.is_tensor(raw_flow) or not torch.is_tensor(x0_pred):
            raise TypeError("Stage-2 generator outputs must be tensors")
        if raw_flow.shape != x0_pred.shape:
            raise ValueError("Stage-2 raw_flow/x0_pred shapes differ")
        return raw_flow, x0_pred

    def _call_generator(
        self,
        latent: torch.Tensor,
        conditioning: Mapping[str, Any],
        timestep: torch.Tensor,
        state: Stage2RolloutState,
        *,
        current_start_frame: int,
        commit_self_kv: bool,
        flow_sigma: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        value = self.generator(
            noisy_image_or_video=latent,
            conditional_dict=dict(conditioning),
            timestep=timestep,
            kv_cache=state.self_kv,
            crossattn_cache=state.cross_kv,
            current_start=current_start_frame * self.frame_seq_length,
            cache_start=current_start_frame * self.frame_seq_length,
            commit_self_kv=commit_self_kv,
            flow_sigma=flow_sigma,
        )
        raw_flow, x0_pred = self._generator_outputs(value)
        if raw_flow.shape != latent.shape:
            raise ValueError("Stage-2 generator changed the rollout latent shape")
        return raw_flow, x0_pred

    def _audit_cache(
        self,
        state: Stage2RolloutState,
        *,
        expected_global_end: int,
        expected_local_end: int,
        expected_cross_initialized: bool = True,
    ) -> dict[str, Any]:
        if len(state.self_kv) == 0 or len(state.self_kv) != len(state.cross_kv):
            raise RuntimeError("Stage-2 self/cross cache layer count mismatch")
        for layer_index, cache in enumerate(state.self_kv):
            k = cache.get("k")
            v = cache.get("v")
            if not torch.is_tensor(k) or not torch.is_tensor(v) or k.shape != v.shape:
                raise RuntimeError(f"Stage-2 layer {layer_index} has invalid K/V")
            if k.shape[1] != self.cache_capacity_tokens:
                raise RuntimeError(
                    f"Stage-2 layer {layer_index} cache capacity drifted"
                )
            if (
                k.requires_grad
                or v.requires_grad
                or k.grad_fn is not None
                or v.grad_fn is not None
            ):
                raise RuntimeError(
                    f"Stage-2 layer {layer_index} persistent K/V retained autograd"
                )
            global_end = int(cache["global_end_index"].item())
            local_end = int(cache["local_end_index"].item())
            if global_end != expected_global_end or local_end != expected_local_end:
                raise RuntimeError(
                    f"Stage-2 layer {layer_index} cache cursor mismatch: "
                    f"expected=({expected_global_end},{expected_local_end}), "
                    f"actual=({global_end},{local_end})"
                )
            if (
                int(cache["pinned_start"].item()) != -1
                or int(cache["pinned_len"].item()) != 0
            ):
                raise RuntimeError(
                    "Stage-2 rollout must not use a floating pinned sink"
                )
        for layer_index, cache in enumerate(state.cross_kv):
            k = cache.get("k")
            v = cache.get("v")
            init_state = cache.get("stage2_state")
            if cache.get("stage2_enabled") is not True:
                raise RuntimeError(
                    f"Stage-2 layer {layer_index} cross-KV branch is not enabled"
                )
            if not isinstance(init_state, Stage2CrossKVInitState):
                raise RuntimeError(
                    f"Stage-2 layer {layer_index} has invalid cross-KV state"
                )
            if (
                not torch.is_tensor(k)
                or not torch.is_tensor(v)
                or k.shape != v.shape
                or k.requires_grad
                or v.requires_grad
                or k.grad_fn is not None
                or v.grad_fn is not None
            ):
                raise RuntimeError(
                    f"Stage-2 layer {layer_index} has invalid persistent cross K/V"
                )
            if init_state.initialized is not expected_cross_initialized:
                raise RuntimeError(
                    f"Stage-2 layer {layer_index} cross-KV init state mismatch"
                )
        return {
            "layers": len(state.self_kv),
            "capacity_frames": self.physical_kv_capacity_frames,
            "capacity_tokens": self.cache_capacity_tokens,
            "global_end_index": expected_global_end,
            "local_end_index": expected_local_end,
            "persistent_kv_detached": True,
            "conditional_cache_branches": 1,
            "cross_kv_active": True,
            "cross_kv_initialized": expected_cross_initialized,
        }

    def _preload_sink(
        self,
        initial_latent: torch.Tensor,
        conditioning: Mapping[str, Any],
        state: Stage2RolloutState,
    ) -> None:
        if self.global_sink_frames != 1 or initial_latent.shape[1] != 1:
            raise RuntimeError(
                "fresh Stage-2 episodes can preload only the original S1 sink; "
                "S4/S8 must restore an episode-1 prefix snapshot"
            )
        timestep = torch.zeros(
            state.batch_size,
            self.global_sink_frames,
            dtype=torch.float32,
            device=state.device,
        )
        with torch.no_grad():
            self._call_generator(
                initial_latent,
                conditioning,
                timestep,
                state,
                current_start_frame=0,
                commit_self_kv=True,
                flow_sigma=0.0,
            )
        self._audit_cache(
            state,
            expected_global_end=self.sink_tokens,
            expected_local_end=self.sink_tokens,
        )

    def _capture_sink_prefix(
        self,
        state: Stage2RolloutState,
        clean_chunk: torch.Tensor,
        *,
        target_sink_frames: int,
    ) -> Stage2SinkPrefixSnapshot:
        """Snapshot original sink + earliest clean episode-1 latents once."""

        target_sink_frames = _strict_nonnegative_int(
            target_sink_frames, "capture_prefix_sink_frames"
        )
        if target_sink_frames not in {4, 8}:
            raise ValueError("Stage-2 prefix snapshots support only S4 or S8")
        if self.spec.name != "baseline_c8w16k4s1":
            raise RuntimeError(
                "Stage-2 prefix snapshots must be captured by the baseline S1 path"
            )
        if state.profile_name != self.spec.name or state.episode_index != 0:
            raise RuntimeError(
                "Stage-2 prefix snapshots are valid only for baseline episode 1"
            )
        if target_sink_frames > 1 + self.chunk_frames:
            raise ValueError("requested sink prefix exceeds the first clean chunk")
        expected_clean_shape = (state.batch_size, self.chunk_frames)
        if (
            clean_chunk.ndim != 5
            or tuple(clean_chunk.shape[:2]) != expected_clean_shape
            or clean_chunk.dtype != state.dtype
            or clean_chunk.device != state.device
        ):
            raise RuntimeError(
                "first clean chunk cannot back the requested sink prefix"
            )

        prefix_tokens = target_sink_frames * self.frame_seq_length
        layers: list[Stage2LayerKVSnapshot] = []
        with torch.no_grad():
            for layer_index, cache in enumerate(state.self_kv):
                k = cache["k"][:, :prefix_tokens].detach().clone()
                v = cache["v"][:, :prefix_tokens].detach().clone()
                if (
                    k.requires_grad
                    or v.requires_grad
                    or k.grad_fn is not None
                    or v.grad_fn is not None
                ):
                    raise RuntimeError(
                        f"Stage-2 prefix layer {layer_index} retained autograd"
                    )
                layers.append(Stage2LayerKVSnapshot(k=k, v=v))
            clean_latents = clean_chunk[:, : target_sink_frames - 1].detach().clone()
        if clean_latents.requires_grad or clean_latents.grad_fn is not None:
            raise RuntimeError("Stage-2 clean prefix latent retained autograd")
        return Stage2SinkPrefixSnapshot(
            source_profile_name=self.spec.name,
            target_sink_frames=target_sink_frames,
            frame_seq_length=self.frame_seq_length,
            layers=tuple(layers),
            clean_latents=clean_latents,
            clean_latent_sha256_per_sample=_per_sample_sha256(clean_latents),
            initial_latent_sha256=state.initial_latent_sha256,
            initial_latent_sha256_per_sample=(state.initial_latent_sha256_per_sample),
            batch_size=state.batch_size,
            dtype=state.dtype,
            device=state.device,
            source_episode_index=state.episode_index,
        )

    @staticmethod
    def _clear_cross_kv(state: Stage2RolloutState) -> None:
        with torch.no_grad():
            for layer_index, cache in enumerate(state.cross_kv):
                init_state = cache.get("stage2_state")
                if not isinstance(init_state, Stage2CrossKVInitState):
                    raise RuntimeError(
                        f"Stage-2 layer {layer_index} has invalid cross-KV state"
                    )
                cache["k"].zero_()
                cache["v"].zero_()
                cache["k"].detach_()
                cache["v"].detach_()
                init_state.clear()

    def _validate_prefix_snapshot(
        self,
        state: Stage2RolloutState,
        snapshot: Stage2SinkPrefixSnapshot,
    ) -> None:
        if not isinstance(snapshot, Stage2SinkPrefixSnapshot):
            raise TypeError("prefix_snapshot must be a Stage2SinkPrefixSnapshot")
        if self.global_sink_frames not in {4, 8}:
            raise ValueError("prefix snapshots may be restored only by S4/S8 profiles")
        if snapshot.target_sink_frames != self.global_sink_frames:
            raise RuntimeError(
                "Stage-2 prefix snapshot sink length differs from the target profile"
            )
        if (
            snapshot.source_profile_name != "baseline_c8w16k4s1"
            or state.profile_name != snapshot.source_profile_name
        ):
            raise RuntimeError("Stage-2 prefix snapshot is not from the baseline path")
        if (
            snapshot.source_episode_index != state.episode_index
            or snapshot.source_episode_index != 0
        ):
            raise RuntimeError("Stage-2 prefix snapshot episode identity changed")
        if snapshot.frame_seq_length != self.frame_seq_length:
            raise RuntimeError("Stage-2 prefix snapshot tokenization changed")
        metadata = (
            snapshot.batch_size,
            snapshot.dtype,
            snapshot.device,
            snapshot.initial_latent_sha256,
            snapshot.initial_latent_sha256_per_sample,
        )
        expected_metadata = (
            state.batch_size,
            state.dtype,
            state.device,
            state.initial_latent_sha256,
            state.initial_latent_sha256_per_sample,
        )
        if metadata != expected_metadata:
            raise RuntimeError("Stage-2 prefix snapshot source metadata changed")
        if len(snapshot.layers) != len(state.self_kv):
            raise RuntimeError("Stage-2 prefix snapshot layer count changed")
        if len(state.self_kv) == 0 or len(state.self_kv) != len(state.cross_kv):
            raise RuntimeError("Stage-2 prefix source cache layer count changed")
        source_spec = resolve_stage2_rollout_profile(snapshot.source_profile_name)
        source_capacity_tokens = (
            source_spec.physical_kv_capacity_frames * self.frame_seq_length
        )
        source_final_global = (
            source_spec.global_sink_frames + source_spec.generated_episode_frames
        ) * self.frame_seq_length
        clean = snapshot.clean_latents
        if (
            not torch.is_tensor(clean)
            or clean.ndim != 5
            or tuple(clean.shape[:2]) != (state.batch_size, self.global_sink_frames - 1)
            or clean.dtype != state.dtype
            or clean.device != state.device
            or clean.requires_grad
            or clean.grad_fn is not None
            or _per_sample_sha256(clean) != snapshot.clean_latent_sha256_per_sample
        ):
            raise RuntimeError("Stage-2 prefix snapshot clean latents are invalid")
        prefix_tokens = self.sink_tokens
        original_sink_tokens = self.frame_seq_length
        for layer_index, (source_cache, layer) in enumerate(
            zip(state.self_kv, snapshot.layers)
        ):
            source_k = source_cache.get("k")
            source_v = source_cache.get("v")
            if (
                not torch.is_tensor(source_k)
                or not torch.is_tensor(source_v)
                or source_k.shape != source_v.shape
                or source_k.shape[1] != source_capacity_tokens
                or source_k.requires_grad
                or source_v.requires_grad
                or source_k.grad_fn is not None
                or source_v.grad_fn is not None
                or int(source_cache["global_end_index"].item()) != source_final_global
                or int(source_cache["local_end_index"].item()) != source_capacity_tokens
                or int(source_cache["pinned_start"].item()) != -1
                or int(source_cache["pinned_len"].item()) != 0
            ):
                raise RuntimeError(
                    f"Stage-2 prefix source layer {layer_index} is not complete"
                )
            expected_shape = (
                state.batch_size,
                prefix_tokens,
                source_k.shape[2],
                source_k.shape[3],
            )
            for name, value in (("k", layer.k), ("v", layer.v)):
                if (
                    not torch.is_tensor(value)
                    or tuple(value.shape) != expected_shape
                    or value.dtype != state.dtype
                    or value.device != state.device
                    or value.requires_grad
                    or value.grad_fn is not None
                ):
                    raise RuntimeError(
                        f"Stage-2 prefix layer {layer_index} {name} is invalid"
                    )
            if not torch.equal(
                layer.k[:, :original_sink_tokens],
                source_cache["k"][:, :original_sink_tokens],
            ) or not torch.equal(
                layer.v[:, :original_sink_tokens],
                source_cache["v"][:, :original_sink_tokens],
            ):
                raise RuntimeError("Stage-2 prefix snapshot original sink changed")
        for layer_index, cache in enumerate(state.cross_kv):
            k = cache.get("k")
            v = cache.get("v")
            init_state = cache.get("stage2_state")
            if (
                cache.get("stage2_enabled") is not True
                or not isinstance(init_state, Stage2CrossKVInitState)
                or init_state.initialized is not True
                or not torch.is_tensor(k)
                or not torch.is_tensor(v)
                or k.shape != v.shape
                or k.requires_grad
                or v.requires_grad
                or k.grad_fn is not None
                or v.grad_fn is not None
            ):
                raise RuntimeError(
                    f"Stage-2 prefix source cross layer {layer_index} is invalid"
                )

    def reset_for_new_episode(
        self,
        state: Stage2RolloutState,
        *,
        prefix_snapshot: Stage2SinkPrefixSnapshot | None = None,
    ) -> Stage2RolloutState:
        """Reset prompt/local state and retain the selected permanent sink."""

        if not isinstance(state, Stage2RolloutState):
            raise TypeError("state must be a Stage2RolloutState")
        if not state.episode_complete:
            raise RuntimeError("cannot reset a partial Stage-2 episode")
        if prefix_snapshot is not None:
            self._validate_prefix_snapshot(state, prefix_snapshot)
            restored = self._allocate_state_from_metadata(
                batch_size=state.batch_size,
                dtype=state.dtype,
                device=state.device,
                initial_latent_sha256=state.initial_latent_sha256,
                initial_latent_sha256_per_sample=(
                    state.initial_latent_sha256_per_sample
                ),
                episode_index=state.episode_index + 1,
            )
            with torch.no_grad():
                for cache, layer in zip(restored.self_kv, prefix_snapshot.layers):
                    cache["k"][:, : self.sink_tokens].copy_(layer.k)
                    cache["v"][:, : self.sink_tokens].copy_(layer.v)
                    cache["k"].detach_()
                    cache["v"].detach_()
                    cache["global_end_index"].fill_(self.sink_tokens)
                    cache["local_end_index"].fill_(self.sink_tokens)
                    cache["pinned_start"].fill_(-1)
                    cache["pinned_len"].zero_()
            self._clear_cross_kv(restored)
            self._audit_cache(
                restored,
                expected_global_end=self.sink_tokens,
                expected_local_end=self.sink_tokens,
                expected_cross_initialized=False,
            )
            return restored

        if self.global_sink_frames != 1:
            raise RuntimeError("S4/S8 episode reset requires a prefix snapshot")
        if state.profile_name != self.spec.name:
            raise RuntimeError("Stage-2 reset profile differs from the state profile")
        self._audit_cache(
            state,
            expected_global_end=(
                self.global_sink_frames + self.generated_episode_frames
            )
            * self.frame_seq_length,
            expected_local_end=self.cache_capacity_tokens,
        )
        with torch.no_grad():
            for cache in state.self_kv:
                cache["k"][:, self.sink_tokens :].zero_()
                cache["v"][:, self.sink_tokens :].zero_()
                cache["k"].detach_()
                cache["v"].detach_()
                cache["global_end_index"].fill_(self.sink_tokens)
                cache["local_end_index"].fill_(self.sink_tokens)
                cache["pinned_start"].fill_(-1)
                cache["pinned_len"].zero_()
        self._clear_cross_kv(state)
        state.episode_index += 1
        state.episode_complete = False
        self._audit_cache(
            state,
            expected_global_end=self.sink_tokens,
            expected_local_end=self.sink_tokens,
            expected_cross_initialized=False,
        )
        return state

    def _run_episode(
        self,
        *,
        initial_latent: torch.Tensor,
        noise: torch.Tensor,
        conditional_dict: Mapping[str, Any],
        exit_step: int,
        requires_grad: bool,
        state: Stage2RolloutState | None = None,
        rollout_mode: str,
        capture_prefix_sink_frames: int | None = None,
        record_identity_hashes: bool,
    ) -> tuple[Stage2RolloutResult, Stage2RolloutState]:
        """Shared 24-latent kernel for training exits and full deployment."""

        exit_step = _strict_nonnegative_int(exit_step, "exit_step")
        if not isinstance(requires_grad, bool):
            raise TypeError("requires_grad must explicitly select the G or F rollout")
        if rollout_mode not in {"random_exit", "full_denoising"}:
            raise ValueError(f"unknown Stage-2 rollout mode: {rollout_mode!r}")
        if rollout_mode == "full_denoising" and (
            requires_grad or exit_step != self.num_denoising_steps - 1
        ):
            raise AssertionError("full deployment must be graph-free and use all steps")
        if record_identity_hashes is not (rollout_mode == "full_denoising"):
            raise AssertionError(
                "per-sample output hashing must be enabled only for deployment"
            )
        if rollout_mode == "random_exit" and capture_prefix_sink_frames is not None:
            raise ValueError("prefix snapshots are deployment-only")
        if exit_step >= self.num_denoising_steps:
            raise ValueError("exit_step is outside the Stage-2 UniPC schedule")
        if initial_latent.ndim != 5 or initial_latent.shape[1] != 1:
            raise ValueError("initial_latent must have shape [B,1,C,H,W]")
        if initial_latent.dtype != torch.bfloat16:
            raise TypeError("Stage-2 rollout inputs must be bfloat16")
        if not bool(torch.isfinite(initial_latent).all().item()):
            raise ValueError("initial_latent contains non-finite values")
        expected_noise_shape = (
            initial_latent.shape[0],
            self.generated_episode_frames,
            *initial_latent.shape[2:],
        )
        if tuple(noise.shape) != tuple(expected_noise_shape):
            raise ValueError(
                f"noise must contain exactly 24 new latents: "
                f"expected={expected_noise_shape}, actual={tuple(noise.shape)}"
            )
        if noise.dtype != initial_latent.dtype or noise.device != initial_latent.device:
            raise TypeError("noise and initial_latent dtype/device must match")
        if not bool(torch.isfinite(noise).all().item()):
            raise ValueError("noise contains non-finite values")
        if "prompt_embeds" not in conditional_dict:
            raise KeyError("conditional_dict.prompt_embeds is required")
        prompt_embeds = conditional_dict["prompt_embeds"]
        if not isinstance(prompt_embeds, torch.Tensor) or prompt_embeds.ndim != 3:
            raise ValueError("conditional_dict.prompt_embeds must have shape [B,L,D]")
        if prompt_embeds.dtype != torch.bfloat16:
            raise TypeError("Stage-2 rollout prompt_embeds must be bfloat16")
        if (
            int(prompt_embeds.shape[0]) != int(initial_latent.shape[0])
            or prompt_embeds.device != initial_latent.device
        ):
            raise ValueError(
                "prompt_embeds batch/device must match the Stage-2 rollout latents"
            )
        if not bool(torch.isfinite(prompt_embeds).all().item()):
            raise ValueError("prompt_embeds contains non-finite values")
        initial_hash = _tensor_sha256(initial_latent)
        initial_hashes = _per_sample_sha256(initial_latent)
        if state is not None and not isinstance(state, Stage2RolloutState):
            raise TypeError("state must be a Stage2RolloutState or None")
        preloaded_this_rollout = state is None
        if state is None:
            if self.global_sink_frames != 1:
                raise RuntimeError(
                    "S4/S8 cannot start a fresh episode; restore an episode-1 "
                    "prefix snapshot first"
                )
            state = self._allocate_state(initial_latent)
            self._preload_sink(initial_latent, conditional_dict, state)
        else:
            if state.episode_complete:
                raise RuntimeError("call reset_for_new_episode before the next action")
            if state.profile_name != self.spec.name:
                raise RuntimeError("Stage-2 rollout profile differs from state profile")
            if (
                state.initial_latent_sha256 != initial_hash
                or state.initial_latent_sha256_per_sample != initial_hashes
            ):
                raise RuntimeError("the permanent Stage-2 initial sink changed")
            if state.batch_size != initial_latent.shape[0]:
                raise RuntimeError(
                    "Stage-2 rollout batch size changed within a session"
                )
            if (
                state.dtype != initial_latent.dtype
                or state.device != initial_latent.device
            ):
                raise RuntimeError(
                    "Stage-2 rollout dtype/device changed within a session"
                )
            self._audit_cache(
                state,
                expected_global_end=self.sink_tokens,
                expected_local_end=self.sink_tokens,
                expected_cross_initialized=False,
            )

        outputs: list[torch.Tensor] = []
        traces: list[tuple[int, ...]] = []
        sigma_traces: list[tuple[float, ...]] = []
        chunk_trace: list[Mapping[str, Any]] = []
        full_timetable: tuple[int, ...] | None = None
        full_sigmas: tuple[float, ...] | None = None
        prefix_snapshot: Stage2SinkPrefixSnapshot | None = None
        scheduler_instances: list[Any] = []
        for chunk_index in range(self.num_chunks):
            start = chunk_index * self.chunk_frames
            latent = noise[:, start : start + self.chunk_frames]
            cache_before_global = int(state.self_kv[0]["global_end_index"].item())
            cache_before_local = int(state.self_kv[0]["local_end_index"].item())
            scheduler, timetable = self._new_scheduler(state.device)
            if any(scheduler is previous for previous in scheduler_instances):
                raise RuntimeError(
                    "Stage-2 requires a distinct UniPC scheduler for every chunk"
                )
            scheduler_instances.append(scheduler)
            scheduler_sigmas = tuple(
                float(value)
                for value in scheduler.sigmas.detach().float().cpu().tolist()
            )
            if full_timetable is None:
                full_timetable = timetable
                full_sigmas = scheduler_sigmas
            elif timetable != full_timetable or scheduler_sigmas != full_sigmas:
                raise RuntimeError("Stage-2 UniPC schedule changed between chunks")
            traces.append(timetable[: exit_step + 1])
            sigma_traces.append(scheduler_sigmas[: exit_step + 1])
            current_start_frame = self.global_sink_frames + start
            x0_pred = None
            for step_index, timestep_value in enumerate(scheduler.timesteps):
                if step_index > exit_step:
                    break
                timestep = torch.full(
                    (state.batch_size, self.chunk_frames),
                    float(timestep_value),
                    dtype=torch.float32,
                    device=state.device,
                )
                if step_index < exit_step:
                    with torch.no_grad():
                        raw_flow, _ = self._call_generator(
                            latent,
                            conditional_dict,
                            timestep,
                            state,
                            current_start_frame=current_start_frame,
                            commit_self_kv=False,
                            flow_sigma=scheduler.sigmas[step_index],
                        )
                        latent = scheduler.step(
                            raw_flow,
                            timestep_value,
                            latent,
                            return_dict=False,
                        )[0]
                else:
                    context = nullcontext() if requires_grad else torch.no_grad()
                    with context:
                        # Exit semantics always return Stage2DiTRole's raw x0.
                        # At K-1 this is the mathematical terminal UniPC value;
                        # keeping the role's FP32 flow->x0 conversion (then BF16
                        # cast) also preserves the established training/deploy
                        # bit pattern instead of redoing it in scheduler.step.
                        _, x0_pred = self._call_generator(
                            latent,
                            conditional_dict,
                            timestep,
                            state,
                            current_start_frame=current_start_frame,
                            commit_self_kv=False,
                            flow_sigma=scheduler.sigmas[step_index],
                        )
            if x0_pred is None:
                raise AssertionError("Stage-2 exit forward did not run")
            if x0_pred.shape != latent.shape:
                raise RuntimeError("Stage-2 exit x0 shape drifted")
            clean_timestep = torch.zeros(
                state.batch_size,
                self.chunk_frames,
                dtype=torch.float32,
                device=state.device,
            )
            with torch.no_grad():
                self._call_generator(
                    x0_pred.detach(),
                    conditional_dict,
                    clean_timestep,
                    state,
                    current_start_frame=current_start_frame,
                    commit_self_kv=True,
                    flow_sigma=0.0,
                )
            outputs.append(x0_pred)
            absolute_end = (
                current_start_frame + self.chunk_frames
            ) * self.frame_seq_length
            local_frames = min(
                self.physical_kv_capacity_frames,
                self.global_sink_frames + (chunk_index + 1) * self.chunk_frames,
            )
            chunk_audit = self._audit_cache(
                state,
                expected_global_end=absolute_end,
                expected_local_end=local_frames * self.frame_seq_length,
            )
            if chunk_index == 0 and capture_prefix_sink_frames is not None:
                prefix_snapshot = self._capture_sink_prefix(
                    state,
                    x0_pred,
                    target_sink_frames=capture_prefix_sink_frames,
                )
            chunk_trace.append(
                {
                    "profile_name": self.spec.name,
                    "episode_index": state.episode_index,
                    "chunk_index": chunk_index,
                    "generated_frame_start": start,
                    "generated_frame_end_exclusive": start + self.chunk_frames,
                    "rope_frame_start": current_start_frame,
                    "rope_frame_end_exclusive": (
                        current_start_frame + self.chunk_frames
                    ),
                    "timestep_count": exit_step + 1,
                    "exit_step": exit_step,
                    "denoising_forward_calls": exit_step + 1,
                    "solver_update_calls": exit_step,
                    "clean_recache_forward_calls": 1,
                    "rollout_mode": rollout_mode,
                    "timesteps": list(timetable[: exit_step + 1]),
                    "sigmas": list(scheduler_sigmas[: exit_step + 1]),
                    "terminal_sigma": scheduler_sigmas[-1],
                    "fresh_scheduler": True,
                    "noisy_self_kv_commits": 0,
                    "clean_self_kv_commits": 1,
                    "cache_before_global_end_index": cache_before_global,
                    "cache_before_local_end_index": cache_before_local,
                    "cache_after_global_end_index": chunk_audit["global_end_index"],
                    "cache_after_local_end_index": chunk_audit["local_end_index"],
                    "cache_capacity_frames": chunk_audit["capacity_frames"],
                    "cache_capacity_tokens": chunk_audit["capacity_tokens"],
                    "sink_frames": self.global_sink_frames,
                    "initial_latent_sha256_per_sample": list(initial_hashes),
                    "clean_latent_sha256_per_sample": (
                        list(_per_sample_sha256(x0_pred))
                        if record_identity_hashes
                        else []
                    ),
                    "captured_prefix_sink_frames": (
                        prefix_snapshot.target_sink_frames
                        if chunk_index == 0 and prefix_snapshot is not None
                        else 0
                    ),
                    "captured_prefix_clean_latent_sha256_per_sample": (
                        list(prefix_snapshot.clean_latent_sha256_per_sample)
                        if chunk_index == 0 and prefix_snapshot is not None
                        else []
                    ),
                }
            )

        generated = torch.cat(outputs, dim=1)
        if generated.shape[1] != 24:
            raise AssertionError("Stage-2 rollout did not produce 24 new latent frames")
        if requires_grad and not generated.requires_grad:
            raise RuntimeError("Stage-2 on-policy rollout lost the Generator gradient")
        if not requires_grad and generated.requires_grad:
            raise RuntimeError(
                "Stage-2 fake-score rollout unexpectedly retained a graph"
            )
        final_audit = self._audit_cache(
            state,
            expected_global_end=(
                self.global_sink_frames + self.generated_episode_frames
            )
            * self.frame_seq_length,
            expected_local_end=self.cache_capacity_tokens,
        )
        if full_timetable is None or full_sigmas is None:
            raise AssertionError("Stage-2 rollout created no UniPC chunk schedule")
        noisy_forward_calls = self.num_chunks * (exit_step + 1)
        final_audit = {
            **final_audit,
            "profile": self.spec.to_dict(),
            "scheduler_instances": self.num_chunks,
            "solver_update_calls": self.num_chunks * exit_step,
            "terminal_x0_direct": rollout_mode == "full_denoising",
            "noisy_forward_calls": noisy_forward_calls,
            "clean_recache_forward_calls": self.num_chunks,
            "sink_preload_forward_calls": int(preloaded_this_rollout),
            "generator_forward_calls": int(preloaded_this_rollout)
            + noisy_forward_calls
            + self.num_chunks,
            "logical_query_tokens": int(preloaded_this_rollout) * self.sink_tokens
            + (noisy_forward_calls + self.num_chunks)
            * self.chunk_frames
            * self.frame_seq_length,
        }
        result = Stage2RolloutResult(
            latents=generated,
            exit_step=exit_step,
            requires_grad=requires_grad,
            scheduler_timesteps=full_timetable,
            scheduler_sigmas=full_sigmas,
            chunk_timesteps=tuple(traces),
            chunk_sigmas=tuple(sigma_traces),
            cache_audit=final_audit,
            rollout_mode=rollout_mode,
            chunk_trace=tuple(chunk_trace),
            initial_latent_sha256_per_sample=initial_hashes,
            generated_latent_sha256_per_sample=(
                # Deployment trace requires byte identities.  Training
                # deliberately avoids a full-latent GPU->CPU hash sync.
                _per_sample_sha256(generated)
                if record_identity_hashes
                else ()
            ),
            prefix_snapshot=prefix_snapshot,
        )
        state.episode_complete = True
        return result, state

    def rollout(
        self,
        *,
        initial_latent: torch.Tensor,
        noise: torch.Tensor,
        conditional_dict: Mapping[str, Any],
        exit_step: int,
        requires_grad: bool,
        state: Stage2RolloutState | None = None,
    ) -> tuple[Stage2RolloutResult, Stage2RolloutState]:
        """Run one training/fake-score episode at one caller-selected exit."""

        if not self.spec.training_allowed:
            raise RuntimeError(f"Stage-2 profile {self.spec.name!r} is deployment-only")
        with self._generator_attention_runtime():
            return self._run_episode(
                initial_latent=initial_latent,
                noise=noise,
                conditional_dict=conditional_dict,
                exit_step=exit_step,
                requires_grad=requires_grad,
                state=state,
                rollout_mode="random_exit",
                record_identity_hashes=False,
            )

    def generate_full_episode(
        self,
        *,
        initial_latent: torch.Tensor,
        noise: torch.Tensor,
        conditional_dict: Mapping[str, Any],
        state: Stage2RolloutState | None = None,
        capture_prefix_sink_frames: int | None = None,
    ) -> tuple[Stage2RolloutResult, Stage2RolloutState]:
        """Deploy one full native K-step episode through the shared kernel."""

        with self._generator_attention_runtime():
            return self._run_episode(
                initial_latent=initial_latent,
                noise=noise,
                conditional_dict=conditional_dict,
                exit_step=self.num_denoising_steps - 1,
                requires_grad=False,
                state=state,
                rollout_mode="full_denoising",
                capture_prefix_sink_frames=capture_prefix_sink_frames,
                record_identity_hashes=True,
            )


__all__ = [
    "STAGE2_K2_SHIFT5_TIMESTEPS",
    "STAGE2_K4_SHIFT5_TIMESTEPS",
    "Stage2ExitRNGStreams",
    "Stage2LayerKVSnapshot",
    "Stage2RolloutPipeline",
    "Stage2RolloutResult",
    "Stage2RolloutState",
    "Stage2SinkPrefixSnapshot",
    "draw_stage2_exit_schedule",
]
