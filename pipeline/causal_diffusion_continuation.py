"""Stateful single-sample continuation sessions for the causal pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import threading
from typing import Any, Optional

import torch

CONTINUATION_TRACE_SCHEMA_VERSION = 1
CONTINUATION_CACHE_FRAMES = 24
CONTINUATION_BLOCK_FRAMES = 8
CONTINUATION_TEXT_LIMIT = 512


@dataclass
class ContinuationCacheBundle:
    kv_pos: list[dict[str, Any]]
    kv_neg: list[dict[str, Any]]
    cross_pos: list[dict[str, Any]]
    cross_neg: list[dict[str, Any]]

    def release(self) -> None:
        self.kv_pos.clear()
        self.kv_neg.clear()
        self.cross_pos.clear()
        self.cross_neg.clear()


@dataclass(frozen=True)
class ContinuationResult:
    latents: torch.Tensor
    video: torch.Tensor
    trace: dict[str, Any]


def tensor_identity_sha256(value: torch.Tensor) -> str:
    """Hash tensor dtype, shape and contiguous CPU bytes (not merely a seed)."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor identity requires a torch.Tensor")
    contiguous = value.detach().contiguous().cpu()
    raw_bytes = contiguous.view(torch.uint8).numpy().tobytes(order="C")
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).removeprefix("torch.").encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(int(size)) for size in contiguous.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(raw_bytes)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_untruncated_prompt(
    text_encoder,
    prompt: str,
    *,
    token_limit: int = CONTINUATION_TEXT_LIMIT,
) -> dict[str, Any]:
    """Count exact tokens before the normal fixed-512 encoding path."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if prompt != prompt.strip():
        raise ValueError("prompt must not contain surrounding whitespace")
    wrapper = getattr(text_encoder, "tokenizer", None)
    backend = getattr(wrapper, "tokenizer", None)
    if wrapper is None or backend is None:
        raise RuntimeError("text encoder does not expose the production tokenizer")
    configured_limit = getattr(wrapper, "seq_len", None)
    if configured_limit != token_limit:
        raise RuntimeError(
            f"production tokenizer must use seq_len={token_limit}, got {configured_limit}"
        )
    cleaned = wrapper._clean(prompt) if getattr(wrapper, "clean", None) else prompt
    encoded = backend(
        [cleaned],
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    input_ids = (
        encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
    )
    first = input_ids[0]
    token_count = int(first.numel()) if torch.is_tensor(first) else len(first)
    if token_count >= token_limit:
        raise ValueError(
            f"prompt has {token_count} untruncated tokens and reaches the "
            f"{token_limit}-token truncation boundary"
        )
    return {
        "sha256": text_sha256(prompt),
        "token_count": token_count,
        "token_limit": token_limit,
        "cleaning": getattr(wrapper, "clean", None),
        "add_special_tokens": True,
        "truncation": False,
    }


def _cache_scalar(cache: dict[str, Any], name: str) -> int:
    value = cache.get(name)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise RuntimeError(f"cache {name} must be a scalar tensor")
    return int(value.item())


def audit_cache_bundle(
    pipeline,
    bundle: ContinuationCacheBundle,
    *,
    cursor_frames: int,
    sink_size: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, Any]:
    """Assert every positive/negative transformer cache is in lockstep."""

    frame_seq_length = int(pipeline.frame_seq_length)
    expected_layers = int(pipeline.num_transformer_blocks)
    expected_capacity = CONTINUATION_CACHE_FRAMES * frame_seq_length
    expected_global = cursor_frames * frame_seq_length
    expected_local = min(cursor_frames, CONTINUATION_CACHE_FRAMES) * frame_seq_length
    cache_sets = {
        "positive": bundle.kv_pos,
        "negative": bundle.kv_neg,
    }
    if any(len(caches) != expected_layers for caches in cache_sets.values()):
        raise RuntimeError("positive/negative KV cache layer count mismatch")
    if (
        len(bundle.cross_pos) != expected_layers
        or len(bundle.cross_neg) != expected_layers
    ):
        raise RuntimeError(
            "positive/negative cross-attention cache layer count mismatch"
        )

    for branch, caches in cache_sets.items():
        for layer_index, cache in enumerate(caches):
            if cache.get("quantized", False):
                raise RuntimeError("continuation does not support quantized KV caches")
            k = cache.get("k")
            v = cache.get("v")
            if not torch.is_tensor(k) or not torch.is_tensor(v):
                raise RuntimeError(
                    f"{branch} cache layer {layer_index} has no tensor K/V"
                )
            if k.shape != v.shape or k.ndim != 4:
                raise RuntimeError(
                    f"{branch} cache layer {layer_index} K/V shape mismatch"
                )
            if k.shape[0] != batch_size or k.shape[1] != expected_capacity:
                raise RuntimeError(
                    f"{branch} cache layer {layer_index} capacity mismatch: "
                    f"expected batch/capacity {batch_size}/{expected_capacity}, "
                    f"got {tuple(k.shape[:2])}"
                )
            if (
                k.dtype != dtype
                or v.dtype != dtype
                or k.device != device
                or v.device != device
            ):
                raise RuntimeError(
                    f"{branch} cache layer {layer_index} dtype/device mismatch"
                )
            if int(cache.get("block_token_size", -1)) != (
                CONTINUATION_BLOCK_FRAMES * frame_seq_length
            ):
                raise RuntimeError(
                    f"{branch} cache layer {layer_index} block size mismatch"
                )
            if int(cache.get("max_blocks", -1)) != 3:
                raise RuntimeError(
                    f"{branch} cache layer {layer_index} max_blocks must be 3"
                )
            global_end = _cache_scalar(cache, "global_end_index")
            local_end = _cache_scalar(cache, "local_end_index")
            pinned_start = _cache_scalar(cache, "pinned_start")
            pinned_len = _cache_scalar(cache, "pinned_len")
            if global_end != expected_global or local_end != expected_local:
                raise RuntimeError(
                    f"{branch} cache layer {layer_index} cursor mismatch: "
                    f"expected global/local {expected_global}/{expected_local}, "
                    f"got {global_end}/{local_end}"
                )
            if pinned_start != -1 or pinned_len != 0:
                raise RuntimeError(
                    f"{branch} cache layer {layer_index} must keep pinned_start=-1/pinned_len=0"
                )

    attention_modules = [
        module
        for _, module in pipeline.generator.model.named_modules()
        if module.__class__.__name__ == "CausalWanSelfAttention"
    ]
    if len(attention_modules) != expected_layers:
        raise RuntimeError(
            f"expected {expected_layers} causal self-attention modules, "
            f"found {len(attention_modules)}"
        )
    for layer_index, module in enumerate(attention_modules):
        if int(module.local_attn_size) != CONTINUATION_CACHE_FRAMES:
            raise RuntimeError(
                f"attention layer {layer_index} effective local size must be 24"
            )
        if int(module.sink_size) != sink_size:
            raise RuntimeError(f"attention layer {layer_index} sink mismatch")
        if int(getattr(module, "global_sink_size", -1)) != 0:
            raise RuntimeError(f"attention layer {layer_index} global sink must be 0")
        if int(module.max_attention_size) < expected_capacity:
            raise RuntimeError(
                f"attention layer {layer_index} max attention is below cache capacity"
            )

    return {
        "global_end_tokens": expected_global,
        "local_end_tokens": expected_local,
        "global_end_frames": cursor_frames,
        "local_end_frames": min(cursor_frames, CONTINUATION_CACHE_FRAMES),
        "cache_capacity_tokens": expected_capacity,
        "effective_attention_local_size": CONTINUATION_CACHE_FRAMES,
        "pinned_start": -1,
        "pinned_len": 0,
    }


class ContinuationSession:
    """Exclusive stateful continuation session owned by one pipeline."""

    @classmethod
    def begin(
        cls,
        pipeline,
        *,
        initial_latent: torch.Tensor,
        sink_size: int,
        noise_plan: Optional[torch.Tensor] = None,
    ) -> "ContinuationSession":
        session = cls(
            pipeline,
            initial_latent=initial_latent,
            sink_size=sink_size,
            noise_plan=noise_plan,
        )
        session._activate()
        return session

    def __init__(
        self,
        pipeline,
        *,
        initial_latent: torch.Tensor,
        sink_size: int,
        noise_plan: Optional[torch.Tensor],
    ):
        self._pipeline = pipeline
        self._initial_latent = (
            initial_latent.clone()
            if isinstance(initial_latent, torch.Tensor)
            else initial_latent
        )
        self._sink_size = int(sink_size)
        self._noise_plan = (
            noise_plan.clone() if isinstance(noise_plan, torch.Tensor) else noise_plan
        )
        self._bundle: Optional[ContinuationCacheBundle] = None
        self._runtime_context = None
        self._negative_conditioning = None
        self._latents: list[torch.Tensor] = []
        self._noise_parts: list[dict[str, Any]] = []
        self._segments: list[dict[str, Any]] = []
        self._blocks: list[dict[str, Any]] = []
        self._anchors: list[dict[str, Any]] = []
        self._cursor = 0
        self._state = "initializing"
        self._failure: Optional[dict[str, str]] = None
        self._locked: dict[str, Any] = {}
        self._initial_identity: Optional[str] = None
        self._noise_identity: Optional[str] = None
        self._negative_prompt_audit: Optional[dict[str, Any]] = None

    @property
    def state(self) -> str:
        return self._state

    @property
    def cursor(self) -> int:
        return self._cursor

    def _lock(self):
        get_lock = getattr(self._pipeline, "_get_continuation_lock", None)
        if callable(get_lock):
            return get_lock()
        lock = getattr(self._pipeline, "_continuation_lock", None)
        if lock is None:
            lock = threading.RLock()
            self._pipeline._continuation_lock = lock
        return lock

    def _validate_begin_contract(self) -> None:
        pipeline = self._pipeline
        initial = self._initial_latent
        if not isinstance(initial, torch.Tensor) or initial.ndim != 5:
            raise ValueError("initial_latent must be a [1,1,C,H,W] tensor")
        if initial.shape[0] != 1 or initial.shape[1] != 1:
            raise ValueError(
                "continuation requires batch_size=1 and one initial latent"
            )
        if not initial.is_floating_point() or not torch.isfinite(initial).all().item():
            raise ValueError("initial_latent must be finite floating point")
        if self._sink_size not in (0, 1):
            raise ValueError("continuation sink_size must be exactly 0 or 1")
        if int(pipeline.num_frame_per_block) != CONTINUATION_BLOCK_FRAMES:
            raise ValueError("continuation requires num_frame_per_block=8")
        if str(pipeline.sample_solver).lower() != "unipc":
            raise ValueError("continuation requires the UniPC sample solver")
        if int(pipeline.sampling_steps) != 50:
            raise ValueError("continuation requires exactly 50 sampling steps")
        if float(pipeline.guidance_scale) != 5.0:
            raise ValueError("continuation requires guidance_scale=5.0")
        if float(pipeline.shift) != 5.0:
            raise ValueError("continuation requires timestep shift=5.0")
        if int(pipeline.num_train_timesteps) <= 0:
            raise ValueError("continuation requires positive num_train_timesteps")
        if int(pipeline.local_attn_size) != -1:
            raise ValueError("continuation requires configured local_attn_size=-1")
        if bool(pipeline.quantize_kv):
            raise ValueError("continuation requires unquantized KV caches")
        if bool(pipeline.multi_shot_sink) or int(pipeline.global_sink_size) != 0:
            raise ValueError("continuation disables multi-shot and global sink modes")
        if bool(pipeline.shot_clean_recache):
            raise ValueError("continuation disables shot_clean_recache")
        if bool(pipeline.use_relative_rope):
            raise ValueError("continuation disables relative RoPE")
        if float(pipeline.multi_shot_rope_offset) != 0.0:
            raise ValueError("continuation disables multi-shot RoPE offsets")
        if bool(pipeline.streaming_vae) or bool(pipeline.async_vae):
            raise ValueError("continuation requires one final non-streaming VAE decode")
        if not bool(pipeline.independent_first_frame):
            raise ValueError("continuation requires independent_first_frame=true")
        if int(pipeline.frame_seq_length) != (
            int(initial.shape[-2]) * int(initial.shape[-1]) // 4
        ):
            raise ValueError("initial latent geometry does not match frame_seq_length")
        if self._noise_plan is not None:
            self._validate_noise(self._noise_plan, require_block_aligned=True)
            if self._noise_plan.shape[1] <= 0:
                raise ValueError("noise_plan must be non-empty")

    def _validate_noise(
        self,
        noise: torch.Tensor,
        *,
        require_block_aligned: bool,
    ) -> None:
        if not isinstance(noise, torch.Tensor) or noise.ndim != 5:
            raise ValueError("noise must be a [1,T,C,H,W] tensor")
        if noise.shape[0] != 1:
            raise ValueError("continuation requires batch_size=1 noise")
        if noise.shape[2:] != self._initial_latent.shape[2:]:
            raise ValueError("noise shape must match initial_latent channels/geometry")
        if (
            noise.dtype != self._initial_latent.dtype
            or noise.device != self._initial_latent.device
        ):
            raise ValueError("noise dtype/device must match initial_latent")
        if noise.shape[1] <= 0 or (
            require_block_aligned and noise.shape[1] % CONTINUATION_BLOCK_FRAMES
        ):
            raise ValueError(
                "continuation noise length must be positive and divisible by 8"
            )
        if not torch.isfinite(noise).all().item():
            raise ValueError("noise contains NaN or Inf")

    def _activate(self) -> None:
        with self._lock():
            if (
                getattr(self._pipeline, "_active_continuation_session", None)
                is not None
            ):
                raise RuntimeError(
                    "pipeline already has an active continuation session"
                )
            self._validate_begin_contract()
            # Release any independent-video cache allocation before reserving
            # a distinct session-owned cache bundle.
            self._pipeline.kv_cache_pos = None
            self._pipeline.kv_cache_neg = None
            self._pipeline.crossattn_cache_pos = None
            self._pipeline.crossattn_cache_neg = None
            self._pipeline._active_continuation_session = self
            try:
                self._runtime_context = self._pipeline._inference_runtime_overrides(
                    sink_size=self._sink_size,
                    global_sink_size=0,
                )
                self._runtime_context.__enter__()
                kv_pos, kv_neg = self._pipeline._build_kv_cache(
                    batch_size=1,
                    dtype=self._initial_latent.dtype,
                    device=self._initial_latent.device,
                )
                cross_pos, cross_neg = self._pipeline._build_crossattn_cache(
                    batch_size=1,
                    dtype=self._initial_latent.dtype,
                    device=self._initial_latent.device,
                )
                self._bundle = ContinuationCacheBundle(
                    kv_pos=kv_pos,
                    kv_neg=kv_neg,
                    cross_pos=cross_pos,
                    cross_neg=cross_neg,
                )
                initial_cache_audit = audit_cache_bundle(
                    self._pipeline,
                    self._bundle,
                    cursor_frames=0,
                    sink_size=self._sink_size,
                    batch_size=1,
                    dtype=self._initial_latent.dtype,
                    device=self._initial_latent.device,
                )
                negative_prompt = self._pipeline.negative_prompt
                self._negative_prompt_audit = audit_untruncated_prompt(
                    self._pipeline.text_encoder,
                    negative_prompt,
                )
                self._negative_conditioning = self._pipeline.text_encoder(
                    text_prompts=[negative_prompt]
                )
                self._locked = {
                    "batch_size": 1,
                    "latent_shape": [
                        int(size) for size in self._initial_latent.shape[2:]
                    ],
                    "dtype": str(self._initial_latent.dtype).removeprefix("torch."),
                    "device": str(self._initial_latent.device),
                    "guidance_scale": float(self._pipeline.guidance_scale),
                    "sample_solver": str(self._pipeline.sample_solver).lower(),
                    "sampling_steps": int(self._pipeline.sampling_steps),
                    "timestep_shift": float(self._pipeline.shift),
                    "num_train_timesteps": int(self._pipeline.num_train_timesteps),
                    "negative_prompt_sha256": text_sha256(negative_prompt),
                    "block_size": CONTINUATION_BLOCK_FRAMES,
                    "sink_size": self._sink_size,
                    "pipeline_sink_size": int(self._pipeline.sink_size),
                    "global_sink_size": int(self._pipeline.global_sink_size),
                    "multi_shot_sink": bool(self._pipeline.multi_shot_sink),
                    "shot_clean_recache": bool(self._pipeline.shot_clean_recache),
                    "multi_shot_rope_offset": float(
                        self._pipeline.multi_shot_rope_offset
                    ),
                    "quantize_kv": bool(self._pipeline.quantize_kv),
                    "independent_first_frame": bool(
                        self._pipeline.independent_first_frame
                    ),
                    "streaming_vae": bool(self._pipeline.streaming_vae),
                    "async_vae": bool(self._pipeline.async_vae),
                    "local_attn_size_config": int(self._pipeline.local_attn_size),
                    "use_relative_rope": bool(self._pipeline.use_relative_rope),
                    "rope_method": str(self._pipeline._dit_model.rope_method),
                    "effective_t_scale": float(self._pipeline._dit_model.t_scale),
                    "effective_local_attn_size": int(
                        self._pipeline._dit_model.local_attn_size
                    ),
                    "effective_attention_local_size": initial_cache_audit[
                        "effective_attention_local_size"
                    ],
                    "effective_use_relative_rope": bool(
                        self._pipeline._dit_model.use_relative_rope
                    ),
                    "effective_original_seq_len": getattr(
                        self._pipeline._dit_model, "original_seq_len", None
                    ),
                    "effective_rope_temporal_offset": float(
                        self._pipeline._dit_model.rope_temporal_offset
                    ),
                    "frame_seq_length": int(self._pipeline.frame_seq_length),
                }
                self._initial_identity = tensor_identity_sha256(self._initial_latent)
                if self._noise_plan is not None:
                    self._noise_identity = tensor_identity_sha256(self._noise_plan)
                self._state = "active"
            except Exception as exc:
                self._state = "failed"
                self._failure = {"type": type(exc).__name__, "message": str(exc)}
                self._cleanup()
                raise

    def _ensure_active(self) -> None:
        if self._state != "active":
            raise RuntimeError(
                f"continuation session is {self._state}; expected active"
            )
        if getattr(self._pipeline, "_active_continuation_session", None) is not self:
            raise RuntimeError("continuation session no longer owns its pipeline")

    def _assert_locked_state(self) -> None:
        pipeline = self._pipeline
        initial = self._initial_latent
        if (
            not isinstance(initial, torch.Tensor)
            or initial.shape[0] != self._locked["batch_size"]
            or [int(size) for size in initial.shape[2:]] != self._locked["latent_shape"]
            or str(initial.dtype).removeprefix("torch.") != self._locked["dtype"]
            or str(initial.device) != self._locked["device"]
        ):
            raise RuntimeError(
                "initial latent shape/dtype/device changed during continuation"
            )
        if float(pipeline.guidance_scale) != self._locked["guidance_scale"]:
            raise RuntimeError("guidance_scale changed during continuation")
        if str(pipeline.sample_solver).lower() != self._locked["sample_solver"]:
            raise RuntimeError("sample solver changed during continuation")
        if int(pipeline.sampling_steps) != self._locked["sampling_steps"]:
            raise RuntimeError("sampling steps changed during continuation")
        if float(pipeline.shift) != self._locked["timestep_shift"]:
            raise RuntimeError("timestep shift changed during continuation")
        if int(pipeline.num_train_timesteps) != self._locked["num_train_timesteps"]:
            raise RuntimeError("num_train_timesteps changed during continuation")
        if (
            text_sha256(pipeline.negative_prompt)
            != self._locked["negative_prompt_sha256"]
        ):
            raise RuntimeError("negative prompt changed during continuation")
        if int(pipeline.num_frame_per_block) != self._locked["block_size"]:
            raise RuntimeError("block size changed during continuation")
        if int(pipeline.local_attn_size) != self._locked["local_attn_size_config"]:
            raise RuntimeError("local attention config changed during continuation")
        if self._sink_size != self._locked["sink_size"]:
            raise RuntimeError("session sink changed during continuation")
        if int(pipeline.sink_size) != self._locked["pipeline_sink_size"]:
            raise RuntimeError("pipeline sink changed during continuation")
        if int(pipeline.global_sink_size) != self._locked["global_sink_size"]:
            raise RuntimeError("global sink changed during continuation")
        if bool(pipeline.multi_shot_sink) != self._locked["multi_shot_sink"]:
            raise RuntimeError("multi-shot sink mode changed during continuation")
        if bool(pipeline.shot_clean_recache) != self._locked["shot_clean_recache"]:
            raise RuntimeError("shot clean recache mode changed during continuation")
        if (
            float(pipeline.multi_shot_rope_offset)
            != self._locked["multi_shot_rope_offset"]
        ):
            raise RuntimeError("multi-shot RoPE offset changed during continuation")
        if bool(pipeline.quantize_kv) != self._locked["quantize_kv"]:
            raise RuntimeError("KV quantization mode changed during continuation")
        if (
            bool(pipeline.independent_first_frame)
            != self._locked["independent_first_frame"]
        ):
            raise RuntimeError(
                "independent-first-frame mode changed during continuation"
            )
        if bool(pipeline.streaming_vae) != self._locked["streaming_vae"]:
            raise RuntimeError("streaming VAE mode changed during continuation")
        if bool(pipeline.async_vae) != self._locked["async_vae"]:
            raise RuntimeError("async VAE mode changed during continuation")
        if bool(pipeline.use_relative_rope) != self._locked["use_relative_rope"]:
            raise RuntimeError("RoPE mode changed during continuation")
        dit = pipeline._dit_model
        if str(dit.rope_method) != self._locked["rope_method"]:
            raise RuntimeError("RoPE method changed during continuation")
        effective_state = {
            "effective_t_scale": float(dit.t_scale),
            "effective_local_attn_size": int(dit.local_attn_size),
            "effective_use_relative_rope": bool(dit.use_relative_rope),
            "effective_original_seq_len": getattr(dit, "original_seq_len", None),
            "effective_rope_temporal_offset": float(dit.rope_temporal_offset),
        }
        for name, value in effective_state.items():
            if value != self._locked[name]:
                raise RuntimeError(
                    f"effective model {name.removeprefix('effective_')} changed "
                    "during continuation"
                )

    def _audit_cursor(self) -> dict[str, Any]:
        if self._bundle is None:
            raise RuntimeError("continuation cache bundle has been released")
        return audit_cache_bundle(
            self._pipeline,
            self._bundle,
            cursor_frames=self._cursor,
            sink_size=self._sink_size,
            batch_size=1,
            dtype=self._initial_latent.dtype,
            device=self._initial_latent.device,
        )

    def _poison(self, exc: BaseException) -> None:
        if self._state == "active":
            self._state = "failed"
            self._failure = {"type": type(exc).__name__, "message": str(exc)}
            self._cleanup()

    def generate_segment(
        self,
        prompt: str,
        *,
        noise: torch.Tensor,
        carry_last_latent_as_anchor: bool = False,
        segment_name: Optional[str] = None,
    ) -> torch.Tensor:
        """Generate one N×8 segment while preserving session self-KV state."""

        with self._lock():
            try:
                self._ensure_active()
                self._assert_locked_state()
                # Reject cache/cursor drift before tokenization or T5 work;
                # a poisoned cache must never be consumed by a new segment.
                self._audit_cursor()
                self._validate_noise(noise, require_block_aligned=True)
                segment_frames = int(noise.shape[1])
                if self._noise_plan is not None:
                    end = self._cursor + segment_frames
                    if end > self._noise_plan.shape[1]:
                        raise ValueError(
                            "segment consumes beyond the locked noise_plan"
                        )
                    expected_noise = self._noise_plan[:, self._cursor : end]
                    if not torch.equal(noise, expected_noise):
                        raise ValueError(
                            "segment noise is not bitwise equal to its global noise_plan slice"
                        )
                if carry_last_latent_as_anchor and not self._latents:
                    raise ValueError(
                        "soft re-anchor requires previously generated latents"
                    )
                if self._cursor == 0 and carry_last_latent_as_anchor:
                    raise ValueError(
                        "the initial segment cannot request soft re-anchor"
                    )

                prompt_audit = audit_untruncated_prompt(
                    self._pipeline.text_encoder,
                    prompt,
                )
                positive_conditioning = self._pipeline.text_encoder(
                    text_prompts=[prompt]
                )
                segment_index = len(self._segments)
                name = segment_name or f"segment_{segment_index}"
                segment_start = self._cursor
                segment_record = {
                    "segment_index": segment_index,
                    "name": name,
                    "start_latent": segment_start,
                    "end_latent": segment_start + segment_frames - 1,
                    "latent_frames": segment_frames,
                    "blocks": segment_frames // CONTINUATION_BLOCK_FRAMES,
                    "prompt": prompt_audit,
                    "carry_last_latent_as_anchor": bool(carry_last_latent_as_anchor),
                }
                self._noise_parts.append(
                    {
                        "segment_index": segment_index,
                        "start_latent": segment_start,
                        "end_latent_exclusive": segment_start + segment_frames,
                        "sha256": tensor_identity_sha256(noise),
                    }
                )

                generated_blocks: list[torch.Tensor] = []
                for block_offset in range(0, segment_frames, CONTINUATION_BLOCK_FRAMES):
                    self._audit_cursor()
                    global_start = self._cursor
                    block_noise = noise[
                        :, block_offset : block_offset + CONTINUATION_BLOCK_FRAMES
                    ].clone()
                    anchor = None
                    anchor_kind = None
                    anchor_source = None
                    if self._cursor == 0:
                        anchor = self._initial_latent
                        anchor_kind = "initial"
                        anchor_source = "initial_latent"
                    elif carry_last_latent_as_anchor and block_offset == 0:
                        anchor = self._latents[-1][:, -1:].clone()
                        anchor_kind = "soft_reanchor"
                        anchor_source = self._cursor - 1

                    if self._bundle is None:  # defensive after the audit above
                        raise RuntimeError("continuation cache bundle is unavailable")
                    clean = self._pipeline._denoise_and_recache_block(
                        noise_block=block_noise,
                        conditional_dict=positive_conditioning,
                        unconditional_dict=self._negative_conditioning,
                        use_cfg=self._locked["guidance_scale"] != 1.0,
                        global_start_frame=global_start,
                        cache_start_frame=global_start,
                        kv_cache_pos=self._bundle.kv_pos,
                        kv_cache_neg=self._bundle.kv_neg,
                        crossattn_cache_pos=self._bundle.cross_pos,
                        crossattn_cache_neg=self._bundle.cross_neg,
                        anchor_latent=anchor,
                        zero_kv_before_recache=False,
                    )
                    if clean.shape != block_noise.shape:
                        raise RuntimeError(
                            "block kernel returned an unexpected latent shape"
                        )
                    if (
                        clean.dtype != block_noise.dtype
                        or clean.device != block_noise.device
                    ):
                        raise RuntimeError(
                            "block kernel returned an unexpected latent dtype/device"
                        )
                    if not torch.isfinite(clean).all().item():
                        raise RuntimeError("block kernel returned NaN or Inf")
                    if anchor is not None and not torch.equal(clean[:, :1], anchor):
                        raise RuntimeError(
                            "block anchor was not preserved in the clean output"
                        )

                    self._cursor += CONTINUATION_BLOCK_FRAMES
                    cache_trace = self._audit_cursor()
                    block_record = {
                        "block_index": len(self._blocks),
                        "segment_index": segment_index,
                        "global_start_latent": global_start,
                        "global_end_latent": self._cursor - 1,
                        "positive_prompt_sha256": prompt_audit["sha256"],
                        "negative_prompt_sha256": self._locked[
                            "negative_prompt_sha256"
                        ],
                        "anchor_kind": anchor_kind,
                        "anchor_source": anchor_source,
                        "anchor_destination": (
                            global_start if anchor is not None else None
                        ),
                        **cache_trace,
                    }
                    if anchor is not None:
                        self._anchors.append(
                            {
                                "kind": anchor_kind,
                                "source": anchor_source,
                                "destination": global_start,
                            }
                        )
                    self._blocks.append(block_record)
                    self._latents.append(clean)
                    generated_blocks.append(clean)

                self._segments.append(segment_record)
                return torch.cat(generated_blocks, dim=1)
            except Exception as exc:
                self._poison(exc)
                raise

    def finish(self) -> ContinuationResult:
        """Validate the generic session, concatenate latents, and decode once."""

        with self._lock():
            try:
                self._ensure_active()
                self._assert_locked_state()
                if not self._latents:
                    raise RuntimeError("cannot finish an empty continuation session")
                if self._cursor % CONTINUATION_BLOCK_FRAMES:
                    raise RuntimeError("continuation cursor is not block aligned")
                if (
                    self._noise_plan is not None
                    and self._cursor != self._noise_plan.shape[1]
                ):
                    raise RuntimeError(
                        "continuation did not consume the complete locked noise_plan"
                    )
                self._audit_cursor()
                latents = torch.cat(self._latents, dim=1)
                if latents.shape[1] != self._cursor:
                    raise RuntimeError("accumulated latent length differs from cursor")
                video = self._pipeline.vae.decode_to_pixel(latents)
                video = (video * 0.5 + 0.5).clamp(0, 1)
                expected_pixel_frames = 1 + (self._cursor - 1) * 4
                if video.ndim != 5 or video.shape[0] != 1:
                    raise RuntimeError("VAE decode returned an invalid video shape")
                if video.shape[1] != expected_pixel_frames:
                    raise RuntimeError(
                        f"VAE decode returned {video.shape[1]} frames; "
                        f"expected {expected_pixel_frames}"
                    )
                if not torch.isfinite(video).all().item():
                    raise RuntimeError("VAE decode returned NaN or Inf")
                self._state = "finished"
                trace = self.trace_snapshot()
                result = ContinuationResult(latents=latents, video=video, trace=trace)
                self._cleanup()
                return result
            except Exception as exc:
                self._poison(exc)
                raise

    def abort(self, reason: str) -> None:
        with self._lock():
            self._ensure_active()
            exc = RuntimeError(reason)
            self._poison(exc)

    def trace_snapshot(self) -> dict[str, Any]:
        return {
            "schema": "longlive_stage1_continuation_session",
            "schema_version": CONTINUATION_TRACE_SCHEMA_VERSION,
            "status": self._state,
            "cursor_frames": self._cursor,
            "locked": dict(self._locked),
            "initial_latent_sha256": self._initial_identity,
            "noise_identity_sha256": self._noise_identity,
            "noise_slices": list(self._noise_parts),
            "negative_prompt": self._negative_prompt_audit,
            "segments": list(self._segments),
            "blocks": list(self._blocks),
            "anchors": list(self._anchors),
            "failure": None if self._failure is None else dict(self._failure),
        }

    def _cleanup(self) -> None:
        try:
            if self._bundle is not None:
                self._bundle.release()
                self._bundle = None
            self._negative_conditioning = None
            self._latents.clear()
            self._noise_plan = None
            self._initial_latent = torch.empty(0)
            if self._runtime_context is not None:
                runtime_context = self._runtime_context
                self._runtime_context = None
                runtime_context.__exit__(None, None, None)
        finally:
            # Even a restoration failure must not strand the pipeline behind
            # a stale active-session guard.
            if getattr(self._pipeline, "_active_continuation_session", None) is self:
                self._pipeline._active_continuation_session = None
