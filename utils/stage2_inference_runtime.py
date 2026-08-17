"""Distributed, resumable Stage-2 Generator-EMA deployment inference.

The runtime is intentionally separate from every legacy inference pipeline.  It
uses rank-stride data parallelism, the shared Stage-2 rollout kernel, and the
strict artifact validators.  Heavy model construction is behind a small
dependency boundary so orchestration and failure semantics remain CPU-testable.
"""

from __future__ import annotations

import gc
import json
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from pipeline.stage2_rollout import Stage2RolloutPipeline
from pipeline.stage2_rollout_profile import resolve_stage2_rollout_profile
from utils.inference_utils import save_video
from utils.stage1_causal_validation import probe_video
from utils.stage1_i2v_data import load_stage1_input_image
from utils.stage1_io import atomic_output_path, canonical_json_sha256, sha256_file
from utils.stage2_inference import (
    STAGE2_INFERENCE_FPS,
    Stage2InferenceResult,
    clear_stage2_vae_cache,
    generate_stage2_single_action,
    generate_stage2_two_action,
)
from utils.stage2_inference_artifacts import (
    STAGE2_INFERENCE_MANIFEST_NAME,
    STAGE2_REVIEW_INDEX_NAME,
    build_stage2_inference_config_identity,
    build_stage2_inference_manifest,
    build_stage2_sample_trace,
    validate_stage2_inference_artifact_set,
    validate_stage2_inference_config_identity,
    validate_stage2_inference_manifest,
    validate_stage2_sample_trace,
    validate_stage2_video_artifact,
    write_stage2_inference_manifest,
    write_stage2_review_index,
    write_stage2_sample_trace,
)
from utils.stage2_inference_assets import (
    assert_stage2_runtime_asset_identities,
    build_stage2_runtime_assets,
    validate_stage2_runtime_assets,
)
from utils.stage2_inference_batch import (
    STAGE2_SINGLE_DATASET,
    STAGE2_TWO_ACTION_DATASET,
    Stage2InferenceSample,
    build_stage2_inference_samples,
    shard_stage2_inference_samples,
)
from utils.stage2_inference_config import ResolvedStage2InferenceConfig
from utils.stage2_inference_loader import load_stage2_ema_generator_for_inference

_GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_OUTPUT_ROOT_GUARD_SCHEMA = "longlive_stage2_output_root_guard/v1"


@dataclass(frozen=True)
class Stage2InferenceDistributedContext:
    """One torchrun process and its exclusive CUDA device."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    def __post_init__(self) -> None:
        for name, value in (
            ("rank", self.rank),
            ("local_rank", self.local_rank),
            ("world_size", self.world_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be a plain integer")
        if self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.local_rank < 0:
            raise ValueError("local_rank must be non-negative")
        if not isinstance(self.device, torch.device):
            raise TypeError("device must be a torch.device")


@dataclass(frozen=True, slots=True)
class _Stage2OutputRootGuard:
    """Authenticated identity of the canonical output directory."""

    path: str
    device: int
    inode: int
    mode: int

    def __post_init__(self) -> None:
        if (
            type(self.path) is not str
            or not self.path
            or not Path(self.path).is_absolute()
            or os.path.normpath(self.path) != self.path
        ):
            raise TypeError("Stage-2 output root guard path must be canonical")
        for name, value in (
            ("device", self.device),
            ("inode", self.inode),
            ("mode", self.mode),
        ):
            if type(value) is not int or value < 0:
                raise TypeError(
                    f"Stage-2 output root guard {name} must be a plain integer"
                )
        if not stat.S_ISDIR(self.mode) or stat.S_ISLNK(self.mode):
            raise ValueError("Stage-2 output root guard must identify a directory")

    @property
    def root(self) -> Path:
        return Path(self.path)


def _environment_integer(name: str) -> int:
    value = os.environ.get(name)
    if value is None or not value.isascii() or not value.isdecimal():
        raise RuntimeError(f"Stage-2 inference requires canonical torchrun {name}")
    return int(value)


def initialize_stage2_inference_distributed() -> Stage2InferenceDistributedContext:
    """Initialize the NCCL process group from a mandatory torchrun environment."""

    rank = _environment_integer("RANK")
    local_rank = _environment_integer("LOCAL_RANK")
    world_size = _environment_integer("WORLD_SIZE")
    if world_size <= 0 or rank >= world_size:
        raise RuntimeError("invalid Stage-2 torchrun rank/world size")
    if not torch.cuda.is_available():
        raise RuntimeError("formal Stage-2 inference requires CUDA")
    visible_devices = torch.cuda.device_count()
    if local_rank >= visible_devices:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is outside {visible_devices} visible CUDA devices"
        )
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise RuntimeError("torchrun environment differs from the process group")
    if "nccl" not in str(dist.get_backend()).lower():
        raise RuntimeError("formal Stage-2 inference requires the NCCL backend")
    if torch.cuda.current_device() != local_rank:
        raise RuntimeError("current CUDA device differs from LOCAL_RANK")
    return Stage2InferenceDistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device("cuda", local_rank),
    )


def _default_text_encoder_builder(
    config: ResolvedStage2InferenceConfig,
    device: torch.device,
) -> torch.nn.Module:
    from utils.wan_5b_wrapper import WanTextEncoder

    return WanTextEncoder(
        t5_checkpoint=config.t5_checkpoint,
        tokenizer_dir=config.tokenizer_dir,
        device=device,
    )


def _default_vae_builder(
    config: ResolvedStage2InferenceConfig,
    device: torch.device,
) -> torch.nn.Module:
    from utils.wan_5b_wrapper import WanVAEWrapper

    vae = WanVAEWrapper(vae_checkpoint=config.vae_checkpoint)
    vae.to(device=device, dtype=torch.bfloat16)
    vae.requires_grad_(False)
    vae.eval()
    return vae


def _default_generator_loader(
    config: ResolvedStage2InferenceConfig,
    device: torch.device,
    runtime_assets: Mapping[str, Any],
) -> Any:
    return load_stage2_ema_generator_for_inference(
        config.stage2_checkpoint,
        architecture_root=config.architecture_root,
        device=device,
        trusted_generator_asset=runtime_assets["generator_asset"],
        expected_recorded_generator_asset_sha256=runtime_assets["checkpoint"][
            "generator_recorded_asset_sha256"
        ],
    )


def _default_pipeline_builder(
    generator: torch.nn.Module,
    loaded_generator: Any,
    profile: str,
) -> Stage2RolloutPipeline:
    resolved = loaded_generator.resolved_training_config
    return Stage2RolloutPipeline(
        generator,
        spec=resolve_stage2_rollout_profile(profile),
        num_train_timesteps=int(resolved.num_train_timesteps),
        frame_seq_length=int(resolved.patch_tokens_per_frame),
    )


def _default_all_gather_object(
    context: Stage2InferenceDistributedContext,
    value: Any,
) -> tuple[Any, ...]:
    if context.world_size == 1:
        return (value,)
    if not dist.is_initialized():
        raise RuntimeError("Stage-2 distributed collective has no process group")
    values: list[Any] = [None] * context.world_size
    dist.all_gather_object(values, value)
    return tuple(values)


def _default_barrier(context: Stage2InferenceDistributedContext) -> None:
    if context.world_size > 1:
        if not dist.is_initialized():
            raise RuntimeError("Stage-2 distributed barrier has no process group")
        dist.barrier()


def _capture_clean_git_code_version() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    try:
        commit_result = subprocess.run(
            ["git", "-C", os.fspath(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status_result = subprocess.run(
            ["git", "-C", os.fspath(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("formal Stage-2 inference requires Git metadata") from exc
    commit = commit_result.stdout.strip().lower()
    if _GIT_COMMIT_RE.fullmatch(commit) is None:
        raise RuntimeError("formal Stage-2 inference Git commit is invalid")
    return {"git_commit": commit, "dirty": bool(status_result.stdout.strip())}


@dataclass(frozen=True)
class Stage2InferenceRuntimeOps:
    """Production operations with injectable heavy/I/O boundaries for CPU tests."""

    build_samples: Callable[..., tuple[Stage2InferenceSample, ...]] = (
        build_stage2_inference_samples
    )
    shard_samples: Callable[..., tuple[Stage2InferenceSample, ...]] = (
        shard_stage2_inference_samples
    )
    build_runtime_assets: Callable[..., Mapping[str, Any]] = build_stage2_runtime_assets
    validate_runtime_assets: Callable[..., Mapping[str, Any]] = (
        validate_stage2_runtime_assets
    )
    assert_runtime_asset_identities: Callable[..., None] = (
        assert_stage2_runtime_asset_identities
    )
    build_text_encoder: Callable[..., Any] = _default_text_encoder_builder
    load_generator: Callable[..., Any] = _default_generator_loader
    build_vae: Callable[..., Any] = _default_vae_builder
    build_pipeline: Callable[..., Any] = _default_pipeline_builder
    load_image: Callable[..., torch.Tensor] = load_stage1_input_image
    generate_single: Callable[..., Stage2InferenceResult] = (
        generate_stage2_single_action
    )
    generate_two: Callable[..., Stage2InferenceResult] = generate_stage2_two_action
    save_video: Callable[..., None] = save_video
    probe_video: Callable[..., Mapping[str, Any]] = probe_video
    build_sample_trace: Callable[..., Mapping[str, Any]] = build_stage2_sample_trace
    validate_sample_trace: Callable[..., Mapping[str, Any]] = (
        validate_stage2_sample_trace
    )
    write_sample_trace: Callable[..., Path] = write_stage2_sample_trace
    validate_artifact_set: Callable[..., None] = validate_stage2_inference_artifact_set
    build_manifest: Callable[..., Mapping[str, Any]] = build_stage2_inference_manifest
    validate_manifest: Callable[..., Mapping[str, Any]] = (
        validate_stage2_inference_manifest
    )
    write_manifest: Callable[..., Path] = write_stage2_inference_manifest
    write_review_index: Callable[..., Path] = write_stage2_review_index
    capture_code_version: Callable[[], Mapping[str, Any]] = (
        _capture_clean_git_code_version
    )
    all_gather_object: Callable[..., tuple[Any, ...]] = _default_all_gather_object
    barrier: Callable[..., None] = _default_barrier


def _strict_json_object(
    guard: _Stage2OutputRootGuard,
    path: Path,
) -> dict[str, Any]:
    _assert_regular_parents(guard, path, allow_missing=False)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"Stage-2 JSON artifact is not a regular file: {path}")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            value[key] = child
        return value

    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid Stage-2 JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"Stage-2 JSON artifact must contain an object: {path}")
    _assert_regular_parents(guard, path, allow_missing=False)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Stage-2 JSON artifact changed while reading: {path}")
    return value


def _lstat_directory_identity(path: Path) -> tuple[int, int, int]:
    try:
        status = path.lstat()
    except OSError as exc:
        raise RuntimeError(
            f"Stage-2 output root is not an accessible directory: {path}"
        ) from exc
    mode = int(status.st_mode)
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise RuntimeError(f"Stage-2 output root is not a regular directory: {path}")
    return int(status.st_dev), int(status.st_ino), mode


def _assert_output_root_guard(guard: _Stage2OutputRootGuard) -> Path:
    """Fail closed when the canonical root no longer has its captured identity.

    This is a conventional pre/post TOCTOU guard, not a claim of capability
    security against an actively malicious local process switching paths in the
    remaining check-to-open micro-window.  That stronger guarantee needs dirfd
    anchored openat/mkdirat operations throughout the artifact writers.
    """

    if type(guard) is not _Stage2OutputRootGuard:
        raise TypeError("Stage-2 output root guard has an invalid type")
    try:
        guard_path = guard.path
        guard_values = (guard.device, guard.inode, guard.mode)
    except AttributeError as exc:
        raise TypeError("Stage-2 output root guard has invalid fields") from exc
    if (
        type(guard_path) is not str
        or not guard_path
        or not Path(guard_path).is_absolute()
        or os.path.normpath(guard_path) != guard_path
        or any(type(value) is not int or value < 0 for value in guard_values)
        or not stat.S_ISDIR(guard_values[2])
        or stat.S_ISLNK(guard_values[2])
    ):
        raise TypeError("Stage-2 output root guard has invalid fields")
    root = Path(guard_path)
    before = _lstat_directory_identity(root)
    expected = guard_values
    if before != expected:
        raise RuntimeError(f"Stage-2 output root identity changed: {root}")
    try:
        resolved = root.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(f"Stage-2 output root cannot be resolved: {root}") from exc
    after = _lstat_directory_identity(root)
    if after != expected or resolved != root:
        raise RuntimeError(f"Stage-2 output root identity changed: {root}")
    return root


def _output_root_guard_identity(
    guard: _Stage2OutputRootGuard,
) -> dict[str, Any]:
    root = _assert_output_root_guard(guard)
    return {
        "schema": _OUTPUT_ROOT_GUARD_SCHEMA,
        "path": os.fspath(root),
        "device": guard.device,
        "inode": guard.inode,
        "mode": guard.mode,
    }


def _validate_output_root_guard_identity(value: Any) -> dict[str, Any]:
    expected_keys = {"schema", "path", "device", "inode", "mode"}
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("Stage-2 output root guard identity schema mismatch")
    if value.get("schema") != _OUTPUT_ROOT_GUARD_SCHEMA:
        raise ValueError("Stage-2 output root guard identity version mismatch")
    path = value.get("path")
    if (
        type(path) is not str
        or not path
        or not Path(path).is_absolute()
        or os.path.normpath(path) != path
    ):
        raise TypeError("Stage-2 output root guard identity path is invalid")
    result: dict[str, Any] = {"schema": _OUTPUT_ROOT_GUARD_SCHEMA, "path": path}
    for name in ("device", "inode", "mode"):
        item = value.get(name)
        if type(item) is not int or item < 0:
            raise TypeError(
                f"Stage-2 output root guard identity {name} must be a plain integer"
            )
        result[name] = item
    if not stat.S_ISDIR(result["mode"]) or stat.S_ISLNK(result["mode"]):
        raise ValueError("Stage-2 output root guard identity mode is invalid")
    return result


def _prepare_output_root(
    path: str | os.PathLike[str],
) -> _Stage2OutputRootGuard:
    root = Path(path).expanduser()
    if root.is_symlink() or (root.exists() and not root.is_dir()):
        raise RuntimeError(f"Stage-2 output root is not a regular directory: {root}")
    resolved = root.resolve()
    checkout = Path(__file__).resolve().parents[1]
    try:
        resolved.relative_to(checkout)
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "Stage-2 inference output root must be outside the Git checkout: "
            f"{resolved}"
        )
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir() or root.resolve() != resolved:
        raise RuntimeError(
            f"Stage-2 output root changed during directory creation: {root}"
        )
    device, inode, mode = _lstat_directory_identity(resolved)
    guard = _Stage2OutputRootGuard(
        path=os.fspath(resolved),
        device=device,
        inode=inode,
        mode=mode,
    )
    _assert_output_root_guard(guard)
    return guard


def _metadata_identity(
    config: ResolvedStage2InferenceConfig,
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for dataset, value in (
        (STAGE2_SINGLE_DATASET, config.single_metadata),
        (STAGE2_TWO_ACTION_DATASET, config.two_action_metadata),
    ):
        source = Path(value).expanduser()
        if source.is_symlink() or not source.is_file():
            raise RuntimeError(
                f"Stage-2 {dataset} metadata is not a regular file: {source}"
            )
        path = source.resolve()
        result[dataset] = {
            "path": os.fspath(path),
            "sha256": sha256_file(path),
        }
    return result


def _relative_artifact_parts(
    guard: _Stage2OutputRootGuard,
    destination: Path,
) -> tuple[Path, tuple[str, ...]]:
    root = _assert_output_root_guard(guard)
    if not isinstance(destination, Path) or not destination.is_absolute():
        raise TypeError("Stage-2 artifact destination must be an absolute Path")
    try:
        relative = destination.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"Stage-2 artifact escapes its output root: {destination}"
        ) from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise RuntimeError(f"Stage-2 artifact path is not canonical: {destination}")
    return root, relative.parts


def _assert_regular_output_directory(
    guard: _Stage2OutputRootGuard,
    directory: Path,
) -> None:
    root = _assert_output_root_guard(guard)
    try:
        directory.relative_to(root)
        before = directory.lstat()
    except (ValueError, OSError) as exc:
        raise RuntimeError(
            f"Stage-2 artifact parent is not a regular directory: {directory}"
        ) from exc
    before_identity = (
        int(before.st_dev),
        int(before.st_ino),
        int(before.st_mode),
    )
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
        raise RuntimeError(
            f"Stage-2 artifact parent is not a regular directory: {directory}"
        )
    try:
        resolved = directory.resolve(strict=True)
        resolved.relative_to(root)
        after = directory.lstat()
    except (ValueError, OSError) as exc:
        raise RuntimeError(
            f"Stage-2 artifact parent escapes its output root: {directory}"
        ) from exc
    after_identity = (int(after.st_dev), int(after.st_ino), int(after.st_mode))
    if resolved != directory or after_identity != before_identity:
        raise RuntimeError(
            f"Stage-2 artifact parent changed during validation: {directory}"
        )
    _assert_output_root_guard(guard)


def _assert_regular_parents(
    guard: _Stage2OutputRootGuard,
    destination: Path,
    *,
    allow_missing: bool,
) -> None:
    root, parts = _relative_artifact_parts(guard, destination)
    current = root
    for part in parts[:-1]:
        _assert_output_root_guard(guard)
        current /= part
        try:
            current.lstat()
        except FileNotFoundError:
            if allow_missing:
                break
            raise RuntimeError(
                f"Stage-2 artifact parent is missing: {current}"
            ) from None
        _assert_regular_output_directory(guard, current)
    _assert_output_root_guard(guard)


def _ensure_regular_parents(
    guard: _Stage2OutputRootGuard,
    destination: Path,
) -> None:
    root, parts = _relative_artifact_parts(guard, destination)
    current = root
    for part in parts[:-1]:
        _assert_output_root_guard(guard)
        current /= part
        try:
            current.lstat()
        except FileNotFoundError:
            pass
        else:
            _assert_regular_output_directory(guard, current)
        _assert_output_root_guard(guard)
        try:
            current.mkdir(exist_ok=True)
        except (FileExistsError, NotADirectoryError) as exc:
            raise RuntimeError(
                f"Stage-2 artifact parent is not a regular directory: {current}"
            ) from exc
        # This check is deliberately after mkdir(exist_ok=True): concurrent ranks
        # may all create the same profile/dataset directory, while a file or
        # symlink winner must never be accepted as an artifact parent.
        _assert_output_root_guard(guard)
        _assert_regular_output_directory(guard, current)
    _assert_regular_parents(guard, destination, allow_missing=False)


def _preencode_prompts(
    samples: Sequence[Stage2InferenceSample],
    *,
    config: ResolvedStage2InferenceConfig,
    device: torch.device,
    ops: Stage2InferenceRuntimeOps,
    runtime_assets: Mapping[str, Any],
) -> dict[str, Mapping[str, torch.Tensor]]:
    prompts = tuple(
        dict.fromkeys(prompt for sample in samples for prompt in sample.prompts)
    )
    if not prompts:
        return {}
    asset_names = ("t5_checkpoint", "tokenizer_dir")
    ops.assert_runtime_asset_identities(runtime_assets, names=asset_names)
    encoder: Any = None
    try:
        encoder = ops.build_text_encoder(config, device)
        ops.assert_runtime_asset_identities(runtime_assets, names=asset_names)
        if isinstance(encoder, torch.nn.Module):
            encoder.requires_grad_(False)
            encoder.eval()
        with torch.inference_mode():
            encoded = encoder(list(prompts))
        # WanTextEncoder tokenizes inside forward(), so this closes both the
        # checkpoint construction and the tokenizer/encoder forward window.
        ops.assert_runtime_asset_identities(runtime_assets, names=asset_names)
        if not isinstance(encoded, Mapping) or set(encoded) != {"prompt_embeds"}:
            raise RuntimeError(
                "WanTextEncoder must return exactly one conditional prompt cache"
            )
        embeddings = encoded["prompt_embeds"]
        if (
            not isinstance(embeddings, torch.Tensor)
            or embeddings.ndim != 3
            or embeddings.shape[0] != len(prompts)
            or not embeddings.is_floating_point()
            or not bool(torch.isfinite(embeddings).all().item())
        ):
            raise RuntimeError("WanTextEncoder returned invalid prompt embeddings")
        embeddings = (
            embeddings.detach().to(device=device, dtype=torch.bfloat16).contiguous()
        )
        return {
            prompt: {"prompt_embeds": embeddings[index : index + 1]}
            for index, prompt in enumerate(prompts)
        }
    finally:
        encoder = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        ops.assert_runtime_asset_identities(runtime_assets, names=asset_names)


def _initial_latent(
    sample: Stage2InferenceSample,
    *,
    vae: Any,
    device: torch.device,
    ops: Stage2InferenceRuntimeOps,
) -> torch.Tensor:
    image_path = Path(sample.input_image_path)
    if image_path.is_symlink() or not image_path.is_file():
        raise RuntimeError(f"Stage-2 source image is not a regular file: {image_path}")
    if sha256_file(image_path) != sample.image_sha256:
        raise RuntimeError(
            f"Stage-2 source image changed before loading: {sample.sample_key}"
        )
    image = ops.load_image(sample)
    if sha256_file(image_path) != sample.image_sha256:
        raise RuntimeError(
            f"Stage-2 source image changed while loading: {sample.sample_key}"
        )
    if not isinstance(image, torch.Tensor):
        raise TypeError("strict Stage-1 image loader returned a non-tensor")
    expected_image = (1, 3, 1, sample.height, sample.width)
    if tuple(image.shape) != expected_image or image.dtype != torch.float32:
        raise RuntimeError(
            "strict Stage-1 image tensor mismatch: "
            f"expected={expected_image}/float32, actual={tuple(image.shape)}/{image.dtype}"
        )
    image = image.to(device=device, dtype=torch.bfloat16)
    clear_stage2_vae_cache(vae)
    with torch.inference_mode():
        latent = vae.encode_to_latent(image)
    expected_latent = (
        1,
        1,
        48,
        sample.height // 16,
        sample.width // 16,
    )
    if (
        not isinstance(latent, torch.Tensor)
        or tuple(latent.shape) != expected_latent
        or latent.device != device
        or latent.dtype != torch.float32
        or not bool(torch.isfinite(latent).all().item())
    ):
        actual = (
            type(latent).__name__,
            tuple(latent.shape) if isinstance(latent, torch.Tensor) else None,
            getattr(latent, "dtype", None),
            getattr(latent, "device", None),
        )
        raise RuntimeError(
            f"Stage-2 source latent contract mismatch: expected={expected_latent}, actual={actual}"
        )
    # WanVAEWrapper intentionally emits FP32 even when its parameters and
    # input are BF16.  The causal Generator/KV contract is BF16, so perform the
    # one explicit boundary cast after validating the real VAE output.
    latent = latent.detach().to(dtype=torch.bfloat16).contiguous()
    if not bool(torch.isfinite(latent).all().item()):
        raise RuntimeError("Stage-2 source latent became non-finite after BF16 cast")
    return latent


def _validate_video_tensor(
    sample: Stage2InferenceSample,
    result: Stage2InferenceResult,
) -> None:
    expected_frames = 96 if sample.dataset == STAGE2_SINGLE_DATASET else 192
    expected_shape = (1, expected_frames, 3, sample.height, sample.width)
    video = getattr(result, "video", None)
    if (
        not isinstance(video, torch.Tensor)
        or tuple(video.shape) != expected_shape
        or not video.is_floating_point()
        or not bool(torch.isfinite(video).all().item())
    ):
        raise RuntimeError(
            f"Stage-2 generated video tensor mismatch for {sample.sample_key}"
        )
    if bool((video < 0).any().item()) or bool((video > 1).any().item()):
        raise RuntimeError("Stage-2 generated video must stay in [0,1]")


def _validate_existing_pair(
    guard: _Stage2OutputRootGuard,
    *,
    sample: Stage2InferenceSample,
    checkpoint: Mapping[str, Any],
    inference_config: Mapping[str, Any],
    ops: Stage2InferenceRuntimeOps,
) -> dict[str, Any] | None:
    root = _assert_output_root_guard(guard)
    video_path = root / sample.output_relative_path
    trace_path = root / sample.trace_relative_path
    _assert_regular_parents(guard, video_path, allow_missing=True)
    _assert_regular_parents(guard, trace_path, allow_missing=True)
    video_exists = video_path.exists() or video_path.is_symlink()
    trace_exists = trace_path.exists() or trace_path.is_symlink()
    _assert_output_root_guard(guard)
    if video_exists != trace_exists:
        raise RuntimeError(
            "Stage-2 resumability requires a complete video+trace pair; found only "
            f"one artifact for {sample.sample_key}"
        )
    if not video_exists:
        _assert_output_root_guard(guard)
        return None
    trace = dict(
        ops.validate_sample_trace(
            _strict_json_object(guard, trace_path),
            sample=sample,
            inference_config=inference_config,
        )
    )
    _assert_output_root_guard(guard)
    if trace.get("checkpoint") != dict(checkpoint):
        raise RuntimeError(
            f"existing Stage-2 sample uses another checkpoint: {sample.sample_key}"
        )
    if trace.get("inference_config") != dict(inference_config):
        raise RuntimeError(
            "existing Stage-2 sample uses another inference config: "
            f"{sample.sample_key}"
        )
    technical = dict(
        validate_stage2_video_artifact(
            sample,
            video_path,
            probe_fn=ops.probe_video,
        )
    )
    _assert_regular_parents(guard, video_path, allow_missing=False)
    expected_output = {
        "relative_path": sample.output_relative_path.as_posix(),
        **technical,
    }
    if trace.get("output") != expected_output:
        raise RuntimeError(
            f"existing Stage-2 video differs from its trace: {sample.sample_key}"
        )
    _assert_output_root_guard(guard)
    return trace


def _generate_one_sample(
    guard: _Stage2OutputRootGuard,
    *,
    sample: Stage2InferenceSample,
    initial_latent: torch.Tensor,
    prompt_cache: Mapping[str, Mapping[str, torch.Tensor]],
    pipelines: Mapping[str, Any],
    vae: Any,
    checkpoint: Mapping[str, Any],
    inference_config: Mapping[str, Any],
    ops: Stage2InferenceRuntimeOps,
) -> dict[str, Any]:
    root = _assert_output_root_guard(guard)
    existing = _validate_existing_pair(
        guard,
        sample=sample,
        checkpoint=checkpoint,
        inference_config=inference_config,
        ops=ops,
    )
    if existing is not None:
        return existing

    target_spec = resolve_stage2_rollout_profile(sample.profile)
    target_pipeline = pipelines[sample.profile]
    episode1_pipeline = (
        pipelines["baseline_c8w16k4s1"]
        if target_spec.global_sink_frames > 1
        else target_pipeline
    )
    with torch.inference_mode():
        if sample.dataset == STAGE2_SINGLE_DATASET:
            result = ops.generate_single(
                episode1_pipeline,
                vae,
                initial_latent=initial_latent,
                conditional_dict=prompt_cache[sample.prompts[0]],
                seeds=(sample.seed,),
            )
        elif sample.dataset == STAGE2_TWO_ACTION_DATASET:
            result = ops.generate_two(
                episode1_pipeline,
                target_pipeline,
                vae,
                initial_latent=initial_latent,
                action_a_conditional_dict=prompt_cache[sample.prompts[0]],
                action_b_conditional_dict=prompt_cache[sample.prompts[1]],
                seeds=(sample.seed,),
            )
        else:  # pragma: no cover - guarded by the strict batch planner
            raise ValueError(f"unknown Stage-2 dataset {sample.dataset!r}")
    _validate_video_tensor(sample, result)

    video_path = root / sample.output_relative_path
    trace_path = root / sample.trace_relative_path
    _ensure_regular_parents(guard, video_path)
    _ensure_regular_parents(guard, trace_path)
    _assert_output_root_guard(guard)
    if video_path.exists() or video_path.is_symlink():
        raise FileExistsError(video_path)
    _assert_regular_parents(guard, video_path, allow_missing=False)
    # These pre/post checks fail closed for ordinary replacement races.  They
    # intentionally do not claim to defeat a malicious local process that can
    # switch a path inside the unavoidable check-to-open micro-window.
    with atomic_output_path(video_path, suffix=".mp4") as temporary:
        _assert_regular_parents(guard, temporary, allow_missing=False)
        ops.save_video(result.video, temporary, fps=STAGE2_INFERENCE_FPS)
        _assert_regular_parents(guard, temporary, allow_missing=False)
        validate_stage2_video_artifact(
            sample,
            temporary,
            probe_fn=ops.probe_video,
        )
        _assert_regular_parents(guard, temporary, allow_missing=False)
        if video_path.exists() or video_path.is_symlink():
            raise FileExistsError(video_path)
        _assert_output_root_guard(guard)
    _assert_regular_parents(guard, video_path, allow_missing=False)
    trace = dict(
        # The injected builder probes and hashes the committed video.
        ops.build_sample_trace(
            sample=sample,
            generation_trace=result.trace,
            output_root=root,
            video_path=video_path,
            checkpoint=checkpoint,
            inference_config=inference_config,
            probe_fn=ops.probe_video,
        )
    )
    _assert_regular_parents(guard, video_path, allow_missing=False)
    trace = dict(
        ops.validate_sample_trace(
            trace,
            sample=sample,
            inference_config=inference_config,
        )
    )
    if trace.get("inference_config") != dict(inference_config):
        raise RuntimeError(
            f"Stage-2 sample trace lost its inference config: {sample.sample_key}"
        )
    _assert_regular_parents(guard, trace_path, allow_missing=False)
    ops.write_sample_trace(root, sample=sample, trace=trace)
    _assert_regular_parents(guard, trace_path, allow_missing=False)
    if not trace_path.is_file() or trace_path.is_symlink():
        raise RuntimeError(f"Stage-2 trace commit failed: {trace_path}")
    _assert_output_root_guard(guard)
    return trace


def _render_review_index_bytes(
    manifest: Mapping[str, Any],
    *,
    ops: Stage2InferenceRuntimeOps,
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="stage2-index-") as directory:
        path = ops.write_review_index(directory, manifest=manifest)
        path = Path(path)
        if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
            raise RuntimeError("Stage-2 review index writer produced no regular file")
        return path.read_bytes()


def _finalize_artifacts(
    guard: _Stage2OutputRootGuard,
    *,
    samples: Sequence[Stage2InferenceSample],
    checkpoint: Mapping[str, Any],
    inference_config: Mapping[str, Any],
    metadata: Mapping[str, Mapping[str, Any]],
    code_version: Mapping[str, Any],
    ops: Stage2InferenceRuntimeOps,
) -> tuple[Path, Path]:
    root = _assert_output_root_guard(guard)
    ops.validate_artifact_set(root, samples=samples)
    _assert_output_root_guard(guard)
    traces: dict[str, Mapping[str, Any]] = {}
    trace_paths: dict[str, Path] = {}
    for sample in samples:
        image_path = Path(sample.input_image_path)
        if (
            image_path.is_symlink()
            or not image_path.is_file()
            or sha256_file(image_path) != sample.image_sha256
        ):
            raise RuntimeError(
                f"Stage-2 source image changed before finalization: {sample.sample_key}"
            )
        trace_path = root / sample.trace_relative_path
        video_path = root / sample.output_relative_path
        _assert_regular_parents(guard, trace_path, allow_missing=False)
        _assert_regular_parents(guard, video_path, allow_missing=False)
        trace = dict(
            ops.validate_sample_trace(
                _strict_json_object(guard, trace_path),
                sample=sample,
                inference_config=inference_config,
            )
        )
        _assert_regular_parents(guard, trace_path, allow_missing=False)
        # Re-probe and hash every MP4 on rank 0; a valid JSON sidecar alone is
        # never accepted as a complete batch.
        technical = dict(
            validate_stage2_video_artifact(
                sample,
                video_path,
                probe_fn=ops.probe_video,
            )
        )
        _assert_regular_parents(guard, video_path, allow_missing=False)
        if trace.get("output") != {
            "relative_path": sample.output_relative_path.as_posix(),
            **technical,
        }:
            raise RuntimeError(
                f"Stage-2 rank-0 artifact revalidation failed: {sample.sample_key}"
            )
        if trace.get("checkpoint") != dict(checkpoint):
            raise RuntimeError(
                f"Stage-2 trace checkpoint differs at finalization: {sample.sample_key}"
            )
        if trace.get("inference_config") != dict(inference_config):
            raise RuntimeError(
                "Stage-2 trace inference config differs at finalization: "
                f"{sample.sample_key}"
            )
        traces[sample.sample_key] = trace
        trace_paths[sample.sample_key] = trace_path

    _assert_output_root_guard(guard)
    manifest = dict(
        ops.build_manifest(
            output_root=root,
            samples=samples,
            traces=traces,
            trace_paths=trace_paths,
            checkpoint=checkpoint,
            inference_config=inference_config,
            metadata=metadata,
            code_version=code_version,
        )
    )
    _assert_output_root_guard(guard)
    manifest = dict(ops.validate_manifest(manifest, inference_config=inference_config))
    _assert_output_root_guard(guard)
    if manifest.get("inference_config") != dict(inference_config):
        raise RuntimeError("Stage-2 manifest lost its inference config identity")
    manifest_path = root / STAGE2_INFERENCE_MANIFEST_NAME
    index_path = root / STAGE2_REVIEW_INDEX_NAME
    _assert_output_root_guard(guard)
    expected_index = _render_review_index_bytes(manifest, ops=ops)
    _assert_output_root_guard(guard)

    manifest_exists = manifest_path.exists() or manifest_path.is_symlink()
    index_exists = index_path.exists() or index_path.is_symlink()
    _assert_output_root_guard(guard)
    if manifest_exists:
        existing_manifest = dict(
            ops.validate_manifest(
                _strict_json_object(guard, manifest_path),
                inference_config=inference_config,
            )
        )
        _assert_output_root_guard(guard)
        if existing_manifest != manifest:
            raise RuntimeError("existing Stage-2 manifest differs from live artifacts")
        if not index_exists:
            raise RuntimeError("complete Stage-2 manifest is missing its review index")
    if index_exists:
        _assert_output_root_guard(guard)
        if index_path.is_symlink() or not index_path.is_file():
            raise RuntimeError("Stage-2 review index is not a regular file")
        if index_path.read_bytes() != expected_index:
            raise RuntimeError(
                "existing Stage-2 review index differs from its manifest"
            )
        _assert_output_root_guard(guard)
    else:
        _assert_output_root_guard(guard)
        written_index = Path(ops.write_review_index(root, manifest=manifest))
        _assert_output_root_guard(guard)
        if written_index != index_path or written_index.read_bytes() != expected_index:
            raise RuntimeError("Stage-2 review index commit failed")
        _assert_output_root_guard(guard)

    # The complete manifest is the final commit marker.  No preceding failure
    # can therefore masquerade as a completed inference batch.
    if not manifest_exists:
        _assert_output_root_guard(guard)
        written_manifest = Path(ops.write_manifest(root, manifest=manifest))
        _assert_output_root_guard(guard)
        if written_manifest != manifest_path:
            raise RuntimeError("Stage-2 inference manifest path drifted")
        if (
            dict(
                ops.validate_manifest(
                    _strict_json_object(guard, manifest_path),
                    inference_config=inference_config,
                )
            )
            != manifest
        ):
            raise RuntimeError("Stage-2 inference manifest commit failed")
        _assert_output_root_guard(guard)
    _assert_output_root_guard(guard)
    return manifest_path, index_path


def _collective_phase_result(
    context: Stage2InferenceDistributedContext,
    *,
    payload: Mapping[str, Any],
    local_exception: Exception | None,
    ops: Stage2InferenceRuntimeOps,
    phase: str,
) -> tuple[Mapping[str, Any], ...]:
    gathered = ops.all_gather_object(context, dict(payload))
    if len(gathered) != context.world_size or any(
        not isinstance(item, Mapping) for item in gathered
    ):
        raise RuntimeError(f"Stage-2 {phase} collective returned invalid payloads")
    errors = [
        f"rank{index}: {item.get('error')}"
        for index, item in enumerate(gathered)
        if item.get("error") is not None
    ]
    if errors:
        if local_exception is not None:
            raise local_exception
        raise RuntimeError(f"Stage-2 {phase} failed on another rank: {errors}")
    return tuple(gathered)


def run_stage2_inference(
    config: ResolvedStage2InferenceConfig,
    *,
    context: Stage2InferenceDistributedContext | None = None,
    ops: Stage2InferenceRuntimeOps | None = None,
) -> dict[str, Any]:
    """Run every configured profile and publish a manifest only after full QA."""

    if not isinstance(config, ResolvedStage2InferenceConfig):
        raise TypeError("run_stage2_inference requires a resolved strict config")
    runtime_ops = ops or Stage2InferenceRuntimeOps()
    distributed = context or initialize_stage2_inference_distributed()
    asset_exception: Exception | None = None
    runtime_assets: Mapping[str, Any] | None = None
    if distributed.rank == 0:
        try:
            runtime_assets = dict(runtime_ops.build_runtime_assets(config))
        except Exception as exc:  # noqa: BLE001 - synchronize rank-0 attestation
            asset_exception = exc
    asset_startup = _collective_phase_result(
        distributed,
        payload={
            "error": (
                None
                if asset_exception is None
                else f"{type(asset_exception).__name__}: {asset_exception}"
            ),
            "runtime_assets": runtime_assets,
        },
        local_exception=asset_exception,
        ops=runtime_ops,
        phase="runtime asset authentication",
    )
    startup_exception: Exception | None = None
    code_version: Mapping[str, Any] | None = None
    inference_config: Mapping[str, Any] | None = None
    root: Path | None = None
    root_guard: _Stage2OutputRootGuard | None = None
    root_guard_identity: Mapping[str, Any] | None = None
    metadata: Mapping[str, Mapping[str, str]] | None = None
    samples: tuple[Stage2InferenceSample, ...] | None = None
    local_samples: tuple[Stage2InferenceSample, ...] | None = None
    sample_plan_sha256: str | None = None
    try:
        if asset_startup[0].get("runtime_assets") is None or any(
            item.get("runtime_assets") is not None for item in asset_startup[1:]
        ):
            raise RuntimeError(
                "Stage-2 runtime assets must be authenticated exactly once on rank 0"
            )
        runtime_assets = dict(
            runtime_ops.validate_runtime_assets(asset_startup[0].get("runtime_assets"))
        )
        runtime_ops.assert_runtime_asset_identities(
            runtime_assets,
            include_source_manifest=True,
        )
        inference_config = build_stage2_inference_config_identity(
            config,
            runtime_assets=runtime_assets["identity"],
        )
        root_guard = _prepare_output_root(config.output_root)
        root = root_guard.root
        root_guard_identity = _output_root_guard_identity(root_guard)
        metadata_before_planning = _metadata_identity(config)
        samples = runtime_ops.build_samples(
            single_metadata=config.single_metadata,
            two_action_metadata=config.two_action_metadata,
            seeds=config.seeds,
            profiles=config.profiles,
        )
        if not isinstance(samples, tuple) or any(
            not isinstance(sample, Stage2InferenceSample) for sample in samples
        ):
            raise TypeError("Stage-2 inference planner must return a sample tuple")
        local_samples = runtime_ops.shard_samples(
            samples,
            rank=distributed.rank,
            world_size=distributed.world_size,
        )
        if (
            not isinstance(local_samples, tuple)
            or local_samples != samples[distributed.rank :: distributed.world_size]
        ):
            raise RuntimeError("Stage-2 inference rank-stride shard plan mismatch")
        metadata = _metadata_identity(config)
        if metadata != metadata_before_planning:
            raise RuntimeError(
                "Stage-2 inference metadata changed while planning samples"
            )
        sample_plan_sha256 = canonical_json_sha256(
            [sample.to_manifest_source() for sample in samples]
        )
        code_version = dict(runtime_ops.capture_code_version())
        if set(code_version) != {"git_commit", "dirty"}:
            raise ValueError("Stage-2 inference code version schema mismatch")
        if code_version["dirty"] is not False:
            raise RuntimeError("formal Stage-2 inference requires a clean Git checkout")
        if _GIT_COMMIT_RE.fullmatch(str(code_version["git_commit"])) is None:
            raise RuntimeError("formal Stage-2 inference Git commit is invalid")
    except Exception as exc:  # noqa: BLE001 - synchronize every rank before raising
        startup_exception = exc
    startup = _collective_phase_result(
        distributed,
        payload={
            "error": (
                None
                if startup_exception is None
                else f"{type(startup_exception).__name__}: {startup_exception}"
            ),
            "code_version": code_version,
            "inference_config": inference_config,
            "output_root": os.fspath(root) if root is not None else None,
            "output_root_guard": root_guard_identity,
            "metadata": metadata,
            "sample_plan_sha256": sample_plan_sha256,
            "local_sample_keys": (
                [sample.sample_key for sample in local_samples]
                if local_samples is not None
                else None
            ),
        },
        local_exception=startup_exception,
        ops=runtime_ops,
        phase="startup",
    )
    config_identities = [
        validate_stage2_inference_config_identity(item.get("inference_config"))
        for item in startup
    ]
    if any(value != config_identities[0] for value in config_identities[1:]):
        raise RuntimeError("Stage-2 ranks resolved different inference configs")
    inference_config = config_identities[0]
    root_values = [item.get("output_root") for item in startup]
    if any(not isinstance(value, str) or not value for value in root_values) or any(
        value != root_values[0] for value in root_values[1:]
    ):
        raise RuntimeError("Stage-2 ranks prepared different output roots")
    guard_identities = [
        _validate_output_root_guard_identity(item.get("output_root_guard"))
        for item in startup
    ]
    if any(value != guard_identities[0] for value in guard_identities[1:]):
        raise RuntimeError("Stage-2 ranks prepared different output root identities")
    if root_guard is None or root_guard_identity != guard_identities[0]:
        raise AssertionError("successful Stage-2 startup lost its output root guard")
    root = _assert_output_root_guard(root_guard)
    if root != Path(root_values[0]):
        raise RuntimeError("Stage-2 local output root differs from rank consensus")
    versions = [dict(item["code_version"]) for item in startup]
    if any(value != versions[0] for value in versions[1:]):
        raise RuntimeError("Stage-2 ranks do not share one clean Git commit")
    code_version = versions[0]
    metadata_values = [dict(item["metadata"]) for item in startup]
    if any(value != metadata_values[0] for value in metadata_values[1:]):
        raise RuntimeError("Stage-2 ranks do not share one metadata snapshot")
    plan_hashes = [item["sample_plan_sha256"] for item in startup]
    if any(value != plan_hashes[0] for value in plan_hashes[1:]):
        raise RuntimeError("Stage-2 ranks built different inference sample plans")
    if samples is None or local_samples is None or runtime_assets is None:
        raise AssertionError("successful Stage-2 startup lost its sample plan")
    sample_keys = [sample.sample_key for sample in samples]
    for rank, item in enumerate(startup):
        expected_keys = sample_keys[rank :: distributed.world_size]
        if item.get("local_sample_keys") != expected_keys:
            raise RuntimeError(f"Stage-2 rank {rank} published a different shard plan")
    metadata = metadata_values[0]

    local_exception: Exception | None = None
    checkpoint: Mapping[str, Any] | None = None
    generated = 0
    skipped = 0
    resources: list[Any] = []
    prompt_cache: dict[str, Mapping[str, torch.Tensor]] | None = None
    loaded: Any = None
    generator: Any = None
    vae: Any = None
    pipelines: dict[str, Any] | None = None
    latent_cache: dict[tuple[str, int, int], torch.Tensor] | None = None
    try:
        bootstrap_exception: Exception | None = None
        if local_samples:
            try:
                prompt_cache = _preencode_prompts(
                    local_samples,
                    config=config,
                    device=distributed.device,
                    ops=runtime_ops,
                    runtime_assets=runtime_assets,
                )
                generator_assets = ("architecture_config", "generator_base")
                runtime_ops.assert_runtime_asset_identities(
                    runtime_assets,
                    names=generator_assets,
                )
                loaded = runtime_ops.load_generator(
                    config,
                    distributed.device,
                    runtime_assets,
                )
                resources.append(loaded)
                runtime_ops.assert_runtime_asset_identities(
                    runtime_assets,
                    names=generator_assets,
                )
                checkpoint = dict(loaded.checkpoint_identity())
                generator = loaded.generator
            except Exception as exc:  # noqa: BLE001 - synchronize model bootstrap
                bootstrap_exception = exc
        bootstrap = _collective_phase_result(
            distributed,
            payload={
                "error": (
                    None
                    if bootstrap_exception is None
                    else (
                        f"{type(bootstrap_exception).__name__}: "
                        f"{bootstrap_exception}"
                    )
                ),
                "checkpoint": checkpoint,
            },
            local_exception=bootstrap_exception,
            ops=runtime_ops,
            phase="Generator checkpoint bootstrap",
        )
        checkpoint_identities = [
            dict(item["checkpoint"])
            for item in bootstrap
            if item.get("checkpoint") is not None
        ]
        if not checkpoint_identities:
            raise RuntimeError("no Stage-2 rank loaded the Generator EMA checkpoint")
        if any(
            value != checkpoint_identities[0] for value in checkpoint_identities[1:]
        ):
            raise RuntimeError("Stage-2 ranks loaded different Generator checkpoints")
        checkpoint = checkpoint_identities[0]
        authenticated_checkpoint = runtime_assets["checkpoint"]
        if (
            checkpoint.get("directory") != authenticated_checkpoint["directory"]
            or checkpoint.get("manifest_sha256")
            != authenticated_checkpoint["manifest_sha256"]
        ):
            raise RuntimeError(
                "Stage-2 loaded checkpoint differs from the rank-0 runtime "
                "attestation"
            )

        existing_sample_keys: set[str] = set()
        preflight_exception: Exception | None = None
        try:
            for sample in local_samples:
                if (
                    _validate_existing_pair(
                        root_guard,
                        sample=sample,
                        checkpoint=checkpoint,
                        inference_config=inference_config,
                        ops=runtime_ops,
                    )
                    is not None
                ):
                    existing_sample_keys.add(sample.sample_key)
        except Exception as exc:  # noqa: BLE001 - synchronize resume preflight
            preflight_exception = exc
        _collective_phase_result(
            distributed,
            payload={
                "error": (
                    None
                    if preflight_exception is None
                    else (
                        f"{type(preflight_exception).__name__}: "
                        f"{preflight_exception}"
                    )
                )
            },
            local_exception=preflight_exception,
            ops=runtime_ops,
            phase="resume preflight",
        )

        if local_samples:
            vae_assets = ("vae_checkpoint",)
            runtime_ops.assert_runtime_asset_identities(
                runtime_assets,
                names=vae_assets,
            )
            vae = runtime_ops.build_vae(config, distributed.device)
            resources.append(vae)
            runtime_ops.assert_runtime_asset_identities(
                runtime_assets,
                names=vae_assets,
            )
            required_profiles = set(config.profiles)
            if any(
                resolve_stage2_rollout_profile(name).global_sink_frames > 1
                for name in required_profiles
            ):
                required_profiles.add("baseline_c8w16k4s1")
            pipelines = {
                name: runtime_ops.build_pipeline(generator, loaded, name)
                for name in sorted(required_profiles)
            }
            resources.extend(pipelines.values())
            latent_cache = {}
            for sample in local_samples:
                prior = _validate_existing_pair(
                    root_guard,
                    sample=sample,
                    checkpoint=checkpoint,
                    inference_config=inference_config,
                    ops=runtime_ops,
                )
                was_existing = sample.sample_key in existing_sample_keys
                if was_existing and prior is None:
                    raise RuntimeError(
                        "Stage-2 existing artifact pair disappeared after resume "
                        f"preflight: {sample.sample_key}"
                    )
                if not was_existing and prior is not None:
                    raise RuntimeError(
                        "Stage-2 artifact pair appeared after resume preflight: "
                        f"{sample.sample_key}"
                    )
                if was_existing:
                    skipped += 1
                    continue
                latent_key = (sample.image_sha256, sample.height, sample.width)
                if latent_key not in latent_cache:
                    latent_cache[latent_key] = _initial_latent(
                        sample,
                        vae=vae,
                        device=distributed.device,
                        ops=runtime_ops,
                    )
                print(
                    f"[stage2-inference rank={distributed.rank}] "
                    f"generating {sample.sample_key}",
                    flush=True,
                )
                _generate_one_sample(
                    root_guard,
                    sample=sample,
                    initial_latent=latent_cache[latent_key].clone(),
                    prompt_cache=prompt_cache,
                    pipelines=pipelines,
                    vae=vae,
                    checkpoint=checkpoint,
                    inference_config=inference_config,
                    ops=runtime_ops,
                )
                generated += 1
    except Exception as exc:  # noqa: BLE001 - synchronize every rank before raising
        local_exception = exc
    finally:
        if latent_cache is not None:
            latent_cache.clear()
        if pipelines is not None:
            pipelines.clear()
        if prompt_cache is not None:
            prompt_cache.clear()
        latent_cache = None
        pipelines = None
        prompt_cache = None
        vae = None
        generator = None
        loaded = None
        resources.clear()
        gc.collect()
        if distributed.device.type == "cuda":
            torch.cuda.empty_cache()

    generation = _collective_phase_result(
        distributed,
        payload={
            "error": (
                None
                if local_exception is None
                else f"{type(local_exception).__name__}: {local_exception}"
            ),
            "checkpoint": checkpoint,
            "generated": generated,
            "skipped": skipped,
        },
        local_exception=local_exception,
        ops=runtime_ops,
        phase="generation",
    )
    checkpoint_identities = [
        dict(item["checkpoint"])
        for item in generation
        if item.get("checkpoint") is not None
    ]
    if not checkpoint_identities:
        raise RuntimeError("no Stage-2 rank loaded the Generator EMA checkpoint")
    if any(value != checkpoint_identities[0] for value in checkpoint_identities[1:]):
        raise RuntimeError("Stage-2 ranks loaded different Generator checkpoints")
    checkpoint = checkpoint_identities[0]
    runtime_ops.barrier(distributed)

    final_exception: Exception | None = None
    manifest_path: Path | None = None
    index_path: Path | None = None
    if distributed.rank == 0:
        try:
            runtime_ops.assert_runtime_asset_identities(
                runtime_assets,
                include_source_manifest=True,
            )
            final_code_version = dict(runtime_ops.capture_code_version())
            if final_code_version != code_version:
                raise RuntimeError("Git state changed during Stage-2 inference")
            manifest_path, index_path = _finalize_artifacts(
                root_guard,
                samples=samples,
                checkpoint=checkpoint,
                inference_config=inference_config,
                metadata=metadata,
                code_version=code_version,
                ops=runtime_ops,
            )
        except Exception as exc:  # noqa: BLE001 - propagate rank-0 failure
            final_exception = exc
    _collective_phase_result(
        distributed,
        payload={
            "error": (
                None
                if final_exception is None
                else f"{type(final_exception).__name__}: {final_exception}"
            )
        },
        local_exception=final_exception,
        ops=runtime_ops,
        phase="finalization",
    )
    runtime_ops.barrier(distributed)
    return {
        "status": "complete",
        "rank": distributed.rank,
        "world_size": distributed.world_size,
        "local_samples": len(local_samples),
        "local_generated": generated,
        "local_skipped": skipped,
        "total_samples": len(samples),
        "checkpoint": checkpoint,
        "manifest": os.fspath(manifest_path) if manifest_path is not None else None,
        "review_index": os.fspath(index_path) if index_path is not None else None,
    }


__all__ = [
    "Stage2InferenceDistributedContext",
    "Stage2InferenceRuntimeOps",
    "initialize_stage2_inference_distributed",
    "run_stage2_inference",
]
