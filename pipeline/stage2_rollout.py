"""Stage-2 on-policy rollout with an explicit, autograd-safe KV contract.

This module owns only the generator rollout used by both fake-score and
generator updates.  It deliberately does not choose losses, optimizers,
phases, or checkpoints.  The caller must provide one ``exit_step`` for an
entire microbatch; sampling that step twice inside the pipeline is forbidden.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from typing import Any, Callable, Mapping

import torch
import torch.distributed as dist

STAGE2_K4_SHIFT5_TIMESTEPS = (999, 937, 833, 624)
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

    _OFFSETS = {"generator": 0x475F45584954, "fake_score": 0x465F45584954}

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
    episode_index: int = 0
    episode_complete: bool = False


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


class Stage2RolloutPipeline:
    """Generate exactly 24 new latent frames from one permanent sink."""

    def __init__(
        self,
        generator: Any,
        *,
        generated_episode_frames: int = 24,
        chunk_frames: int = 8,
        local_window_frames: int = 16,
        global_sink_frames: int = 1,
        num_denoising_steps: int = 4,
        timestep_shift: float = 5.0,
        num_train_timesteps: int = 1000,
        frame_seq_length: int = 390,
        scheduler_factory: Callable[[], Any] | None = None,
    ):
        self.generator = generator
        self.generated_episode_frames = int(generated_episode_frames)
        self.chunk_frames = int(chunk_frames)
        self.local_window_frames = int(local_window_frames)
        self.global_sink_frames = int(global_sink_frames)
        self.num_denoising_steps = int(num_denoising_steps)
        self.timestep_shift = float(timestep_shift)
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
        self.num_chunks = self.generated_episode_frames // self.chunk_frames
        self._configure_generator_attention()

    @classmethod
    def from_resolved_config(
        cls,
        generator: Any,
        resolved: Any,
        *,
        scheduler_factory: Callable[[], Any] | None = None,
    ) -> "Stage2RolloutPipeline":
        return cls(
            generator,
            generated_episode_frames=resolved.generated_episode_frames,
            chunk_frames=resolved.chunk_frames,
            local_window_frames=resolved.local_window_frames,
            global_sink_frames=resolved.global_sink_frames,
            num_denoising_steps=resolved.num_denoising_steps,
            timestep_shift=resolved.rollout_timestep_shift,
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
        if self.generated_episode_frames % self.chunk_frames:
            raise ValueError("generated_episode_frames must divide into whole chunks")
        if self.local_window_frames < self.chunk_frames:
            raise ValueError("local_window_frames must include the current chunk")
        if self.local_window_frames % self.chunk_frames:
            raise ValueError("local_window_frames must be a multiple of chunk_frames")
        if self.global_sink_frames != 1:
            raise ValueError("the Stage-2 baseline requires exactly one global sink")
        if self.num_denoising_steps != 4 or self.timestep_shift != 5.0:
            raise ValueError("the Stage-2 baseline requires native K4/shift5 UniPC")
        if self.generated_episode_frames != 24 or self.chunk_frames != 8:
            raise ValueError("the Stage-2 baseline must generate 3x8 = 24 new latents")
        if (
            self.local_window_frames != 16
            or self.history_frames != 8
            or self.frame_seq_length != 390
            or self.physical_kv_capacity_frames != 17
        ):
            raise ValueError(
                "the Stage-2 baseline requires C8/W16/H8/S1, 390 tokens/frame, "
                "and physical KV capacity 17 frames"
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

    def _configure_generator_attention(self) -> None:
        model = self._causal_backbone()
        if bool(getattr(model, "is_gradient_checkpointing", False)) or bool(
            getattr(model, "gradient_checkpointing", False)
        ):
            raise RuntimeError(
                "Stage-2 generator cache rollout forbids activation checkpointing"
            )
        # W excludes the sink.  The causal attention's internal total span and
        # physical cache both include it, hence S+W=17 rather than legacy 16.
        setattr(model, "local_attn_size", self.physical_kv_capacity_frames)
        setattr(model, "sink_size", self.global_sink_frames)
        setattr(model, "global_sink_size", self.global_sink_frames)
        for module in model.modules() if hasattr(model, "modules") else ():
            if hasattr(module, "local_attn_size"):
                module.local_attn_size = self.physical_kv_capacity_frames
            if hasattr(module, "max_attention_size"):
                module.max_attention_size = self.cache_capacity_tokens
            if hasattr(module, "sink_size"):
                module.sink_size = self.global_sink_frames
            if hasattr(module, "global_sink_size"):
                module.global_sink_size = self.global_sink_frames

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
        timesteps = tuple(int(float(value)) for value in scheduler.timesteps.tolist())
        if timesteps != STAGE2_K4_SHIFT5_TIMESTEPS:
            raise RuntimeError(
                "Stage-2 UniPC timetable drifted: "
                f"expected={STAGE2_K4_SHIFT5_TIMESTEPS}, actual={timesteps}"
            )
        terminal_sigma = float(scheduler.sigmas[-1].detach().cpu())
        if terminal_sigma != 0.0:
            raise RuntimeError(
                f"Stage-2 UniPC terminal sigma must be 0, got {terminal_sigma}"
            )
        sigma_values = tuple(
            float(value) for value in scheduler.sigmas.detach().float().cpu().tolist()
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
        if any(left <= right for left, right in zip(sigma_values, sigma_values[1:])):
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
        batch_size = int(initial_latent.shape[0])
        dtype = initial_latent.dtype
        device = initial_latent.device
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
                "is_init": False,
                "stage2_enabled": True,
            }
            for _ in range(num_layers)
        ]
        return Stage2RolloutState(
            self_kv=self_kv,
            cross_kv=cross_kv,
            initial_latent_sha256=_tensor_sha256(initial_latent),
            batch_size=batch_size,
            dtype=dtype,
            device=device,
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
                    "Stage-2 baseline must not use a floating pinned sink"
                )
        for layer_index, cache in enumerate(state.cross_kv):
            k = cache.get("k")
            v = cache.get("v")
            if cache.get("stage2_enabled") is not True:
                raise RuntimeError(
                    f"Stage-2 layer {layer_index} cross-KV branch is not enabled"
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
            if bool(cache.get("is_init", False)) is not expected_cross_initialized:
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

    def reset_for_new_episode(self, state: Stage2RolloutState) -> None:
        """Keep only the original sink and clear all prompt/local state."""

        if not state.episode_complete:
            raise RuntimeError("cannot reset a partial Stage-2 episode")
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
            for cache in state.cross_kv:
                cache["k"].zero_()
                cache["v"].zero_()
                cache["k"].detach_()
                cache["v"].detach_()
                cache["is_init"] = False
        state.episode_index += 1
        state.episode_complete = False
        self._audit_cache(
            state,
            expected_global_end=self.sink_tokens,
            expected_local_end=self.sink_tokens,
            expected_cross_initialized=False,
        )

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
        """Run one 24-new-latent episode with one caller-selected exit."""

        exit_step = _strict_nonnegative_int(exit_step, "exit_step")
        if not isinstance(requires_grad, bool):
            raise TypeError("requires_grad must explicitly select the G or F rollout")
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
        preloaded_this_rollout = state is None
        if state is None:
            state = self._allocate_state(initial_latent)
            self._preload_sink(initial_latent, conditional_dict, state)
        else:
            if state.episode_complete:
                raise RuntimeError("call reset_for_new_episode before the next action")
            if state.initial_latent_sha256 != initial_hash:
                raise RuntimeError("the permanent Stage-2 initial sink changed")
            if state.batch_size != initial_latent.shape[0]:
                raise RuntimeError(
                    "Stage-2 rollout batch size changed within a session"
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
        full_timetable: tuple[int, ...] | None = None
        full_sigmas: tuple[float, ...] | None = None
        for chunk_index in range(self.num_chunks):
            start = chunk_index * self.chunk_frames
            latent = noise[:, start : start + self.chunk_frames]
            scheduler, timetable = self._new_scheduler(state.device)
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
            self._audit_cache(
                state,
                expected_global_end=absolute_end,
                expected_local_end=local_frames * self.frame_seq_length,
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
        state.episode_complete = True
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
        return (
            Stage2RolloutResult(
                latents=generated,
                exit_step=exit_step,
                requires_grad=requires_grad,
                scheduler_timesteps=full_timetable,
                scheduler_sigmas=full_sigmas,
                chunk_timesteps=tuple(traces),
                chunk_sigmas=tuple(sigma_traces),
                cache_audit=final_audit,
            ),
            state,
        )


__all__ = [
    "STAGE2_K4_SHIFT5_TIMESTEPS",
    "Stage2ExitRNGStreams",
    "Stage2RolloutPipeline",
    "Stage2RolloutResult",
    "Stage2RolloutState",
    "draw_stage2_exit_schedule",
]
