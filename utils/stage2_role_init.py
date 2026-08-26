"""Strict, sequential, init-only construction for the three Stage-2 DiTs."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import gc
import hashlib
import os
from pathlib import Path
import stat
import sys
from typing import Any, TypeVar

import torch

from model.stage2_dmd import Stage2DMD, Stage2DiTRole
from utils.lora_utils import build_lora_shard_schema, strict_load_lora_state_dict
from utils.stage1_io import canonical_json_sha256, sha256_file
from utils.stage2_fsdp2 import audit_stage2_fsdp2_role, fsdp2_wrap_stage2_role
from utils.stage2_roles import (
    Stage2TargetAudit,
    audit_stage2_frozen_real_score,
    configure_stage2_role_lora,
    stage2_adapter_digest,
)

_T = TypeVar("_T")
StageRunner = Callable[[str, Callable[[], _T]], _T]

_INIT_ONLY_TRIPWIRES = (
    "module_call_impl",
    "direct_module_forward",
    "optimizer",
    "ema",
    "text_encoder",
    "vae",
    "dataloader",
)


@contextmanager
def stage2_init_only_side_effect_guard():
    """Fail immediately if Batch 2 constructs or executes forbidden machinery."""

    # These imports define classes only.  The guard proves none is instantiated;
    # it does not claim their Python modules were absent from the process.
    import torch.utils.data
    import utils.distributed as distributed_utils
    import utils.wan_5b_wrapper as wan_wrapper

    audit: dict[str, Any] = {
        "tripwires_enforced": list(_INIT_ONLY_TRIPWIRES),
        "forward_calls": 0,
        "optimizer_created": False,
        "ema_created": False,
        "text_encoder_created": False,
        "vae_created": False,
        "dataloader_created": False,
    }
    originals: list[tuple[type, str, Any]] = []
    previous_profile = sys.getprofile()

    def install(owner: type, attribute: str, audit_key: str, label: str) -> None:
        original = getattr(owner, attribute)
        originals.append((owner, attribute, original))

        def tripwire(*args, **kwargs):
            del args, kwargs
            if audit_key == "forward_calls":
                audit[audit_key] += 1
            else:
                audit[audit_key] = True
            raise RuntimeError(f"Stage-2 init-only side-effect tripwire: {label}")

        setattr(owner, attribute, tripwire)

    install(torch.nn.Module, "_call_impl", "forward_calls", "module forward")
    install(torch.optim.Optimizer, "__init__", "optimizer_created", "optimizer")
    install(
        distributed_utils.EMA_FSDP,
        "__init__",
        "ema_created",
        "EMA_FSDP",
    )
    install(
        distributed_utils.TrainableShardedEMA,
        "__init__",
        "ema_created",
        "TrainableShardedEMA",
    )
    install(
        wan_wrapper.WanTextEncoder,
        "__init__",
        "text_encoder_created",
        "WanTextEncoder",
    )
    install(
        wan_wrapper.WanVAEWrapper,
        "__init__",
        "vae_created",
        "WanVAEWrapper",
    )
    install(
        torch.utils.data.DataLoader,
        "__init__",
        "dataloader_created",
        "DataLoader",
    )

    def profile_module_forward(frame, event, arg):
        if (
            event == "call"
            and frame.f_code.co_name == "forward"
            and isinstance(frame.f_locals.get("self"), torch.nn.Module)
        ):
            audit["forward_calls"] += 1
            raise RuntimeError(
                "Stage-2 init-only side-effect tripwire: direct module forward"
            )
        if previous_profile is not None:
            previous_profile(frame, event, arg)

    sys.setprofile(profile_module_forward)
    try:
        yield audit
    finally:
        sys.setprofile(previous_profile)
        for owner, attribute, original in reversed(originals):
            setattr(owner, attribute, original)


def _file_identity_from_stat(value: os.stat_result) -> dict[str, int]:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _current_file_identity(path: Path) -> dict[str, int]:
    return _file_identity_from_stat(path.stat())


def _sha256_open_file(handle: Any, *, path: Path) -> str:
    del path
    digest = hashlib.sha256()
    while chunk := handle.read(8 << 20):
        digest.update(chunk)
    return digest.hexdigest()


def _resume_asset_file_attestation(
    entry: Mapping[str, Any],
    *,
    label: str,
    cache: dict[tuple[str, int, str], dict[str, int]],
) -> dict[str, int]:
    """Reauthenticate one persisted asset and return this job's live identity."""

    if not isinstance(entry, Mapping):
        raise TypeError(f"{label} must be an object")
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label}.path must be a non-empty string")
    path = Path(raw_path).expanduser()
    try:
        resolved = path.resolve(strict=True)
        file_stat = path.lstat()
    except OSError as exc:
        raise RuntimeError(
            f"{label} is unavailable during Stage-2 resume: {path}"
        ) from exc
    if path != resolved or stat.S_ISLNK(file_stat.st_mode):
        raise RuntimeError(
            f"{label} must remain a canonical non-symlink path during Stage-2 "
            f"resume: recorded={path}, resolved={resolved}"
        )
    if not stat.S_ISREG(file_stat.st_mode):
        raise RuntimeError(f"{label} is not a regular file: {path}")

    expected_size = entry.get("size")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
    ):
        raise ValueError(f"{label}.size must be a non-negative integer")
    expected_sha = entry.get("sha256")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
    ):
        raise ValueError(f"{label}.sha256 must be a lowercase SHA256")

    cache_key = (str(resolved), expected_size, expected_sha)
    cached_identity = cache.get(cache_key)
    if cached_identity is not None:
        current_identity = _current_file_identity(resolved)
        if resolved.is_symlink() or current_identity != cached_identity:
            raise RuntimeError(
                "Stage-2 checkpoint file identity changed during resume rank-0 "
                f"audit: expected={cached_identity}, actual={current_identity}, "
                f"path={resolved}"
            )
        return dict(cached_identity)

    with resolved.open("rb") as handle:
        before_stat = os.fstat(handle.fileno())
        if not stat.S_ISREG(before_stat.st_mode):
            raise RuntimeError(f"{label} descriptor is not a regular file")
        actual_sha = _sha256_open_file(handle, path=resolved)
        after_stat = os.fstat(handle.fileno())
    before_identity = _file_identity_from_stat(before_stat)
    after_identity = _file_identity_from_stat(after_stat)
    if before_identity != after_identity:
        raise RuntimeError(
            "Stage-2 checkpoint file identity changed during resume rank-0 "
            f"audit: before={before_identity}, after={after_identity}, path={resolved}"
        )
    if before_identity["size"] != expected_size:
        raise RuntimeError(
            "Stage-2 checkpoint file size changed during resume rank-0 audit: "
            f"expected={expected_size}, actual={before_identity['size']}, "
            f"path={resolved}"
        )
    try:
        after_stat = resolved.lstat()
    except OSError as exc:
        raise RuntimeError(
            f"{label} disappeared during Stage-2 resume rank-0 audit: {resolved}"
        ) from exc
    if stat.S_ISLNK(after_stat.st_mode) or not stat.S_ISREG(after_stat.st_mode):
        raise RuntimeError(
            f"{label} changed file type during Stage-2 resume rank-0 audit: {resolved}"
        )
    path_identity = _current_file_identity(resolved)
    if after_identity != path_identity:
        raise RuntimeError(
            "Stage-2 checkpoint path changed during resume rank-0 audit: "
            f"descriptor={after_identity}, path_identity={path_identity}, "
            f"path={resolved}"
        )
    if actual_sha != expected_sha:
        raise RuntimeError(
            "Stage-2 checkpoint content SHA changed during resume rank-0 audit: "
            f"expected={expected_sha}, actual={actual_sha}, path={resolved}"
        )
    cache[cache_key] = after_identity
    return dict(after_identity)


def refresh_stage2_role_asset_identities(
    assets: Mapping[str, Any],
    *,
    architecture_root: str | Path | None = None,
) -> dict[str, Any]:
    """Refresh historical stat identities only after a rank-0 content audit.

    Checkpoint provenance deliberately keeps the immutable path/size/SHA and
    semantic asset contract. Device, inode and timestamps authenticate only
    one runtime mount observation, so a resumed job must not reuse those old
    values as its live TOCTOU bracket. The architecture config may be relocated
    by the launch config, but its recorded size/SHA remain authoritative.
    """

    if not isinstance(assets, Mapping):
        raise TypeError("Stage-2 resume assets must be an object")
    expected_roles = {"generator", "real_score", "fake_score"}
    if set(assets) != expected_roles:
        raise ValueError(
            "Stage-2 resume assets must contain exactly generator/real_score/"
            f"fake_score; actual={sorted(assets)}"
        )
    refreshed = copy.deepcopy(dict(assets))
    cache: dict[tuple[str, int, str], dict[str, int]] = {}
    live_architecture_path = None
    if architecture_root is not None:
        live_architecture_path = (
            Path(architecture_root).expanduser() / "config.json"
        ).resolve(strict=True)
    for role in ("generator", "real_score", "fake_score"):
        asset = refreshed[role]
        if not isinstance(asset, dict):
            raise TypeError(f"Stage-2 resume asset {role} must be an object")
        files = asset.get("checkpoint_files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"Stage-2 resume asset {role} lacks checkpoint_files")
        for index, entry in enumerate(files):
            if not isinstance(entry, dict):
                raise TypeError(
                    f"Stage-2 resume asset {role}.checkpoint_files[{index}] "
                    "must be an object"
                )
            entry["identity"] = _resume_asset_file_attestation(
                entry,
                label=f"Stage-2 {role}.checkpoint_files[{index}]",
                cache=cache,
            )
        checkpoint_path = asset.get("checkpoint_path")
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise ValueError(
                f"Stage-2 resume asset {role}.checkpoint_path must be a path string"
            )
        checkpoint_entries = [
            entry for entry in files if entry.get("path") == checkpoint_path
        ]
        if len(checkpoint_entries) != 1:
            raise RuntimeError(
                f"Stage-2 resume asset {role}.checkpoint_path must identify exactly "
                f"one authenticated checkpoint file: path={checkpoint_path}"
            )
        if asset.get("checkpoint_format") != "wan_native_transformer" and (
            asset.get("checkpoint_sha256") != checkpoint_entries[0].get("sha256")
        ):
            raise RuntimeError(
                f"Stage-2 resume asset {role} checkpoint SHA disagrees with its "
                "authenticated file entry"
            )
        architecture = asset.get("architecture_file")
        if not isinstance(architecture, dict):
            raise ValueError(f"Stage-2 resume asset {role} lacks architecture_file")
        if live_architecture_path is not None:
            architecture["path"] = str(live_architecture_path)
        architecture["identity"] = _resume_asset_file_attestation(
            architecture,
            label=f"Stage-2 {role}.architecture_file",
            cache=cache,
        )
    return refreshed


def _assert_verified_asset_files_unchanged(
    asset: Mapping[str, Any],
    *,
    verify_content_hash: bool = False,
    content_hash_cache: set[tuple[str, str, tuple[tuple[str, int], ...]]] | None = None,
) -> None:
    files = asset.get("checkpoint_files")
    if not isinstance(files, list) or not files:
        raise ValueError("verified Stage-2 asset lacks checkpoint_files")
    architecture_file = asset.get("architecture_file")
    if architecture_file is not None:
        files = [*files, architecture_file]
    for entry in files:
        if not isinstance(entry, Mapping):
            raise TypeError("verified Stage-2 checkpoint file entry must be an object")
        path = Path(entry["path"])
        identity = _current_file_identity(path)
        if identity != entry.get("identity"):
            raise RuntimeError(
                "Stage-2 checkpoint file identity changed after its rank-0 hash "
                f"audit: expected={entry.get('identity')}, actual={identity}, "
                f"path={path}"
            )
        if verify_content_hash:
            cache_key = (
                str(path),
                str(entry.get("sha256")),
                tuple(sorted(identity.items())),
            )
            if content_hash_cache is None or cache_key not in content_hash_cache:
                actual_sha = sha256_file(path)
                if actual_sha != entry.get("sha256"):
                    raise RuntimeError(
                        "Stage-2 checkpoint content SHA changed after load: "
                        f"expected={entry.get('sha256')}, actual={actual_sha}, path={path}"
                    )
                if content_hash_cache is not None:
                    content_hash_cache.add(cache_key)


@dataclass(frozen=True)
class Stage2RoleInitializationAudit:
    role: str
    backbone: str
    base_checkpoint_sha256: str
    base_state_tensor_count: int
    base_state_parameter_count: int
    base_dtype: str
    strict_reload_succeeded: bool
    trainable_policy: str
    activation_checkpointing: bool
    adapter_seed: int | None
    adapter_digest: str | None
    target_audit: Mapping[str, Any] | None
    canonical_adapter_keys: tuple[str, ...]
    pre_fsdp_adapter_tensor_count: int
    pre_fsdp_trainable_parameters: int
    post_fsdp: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Stage2InitializedRoles:
    model: Stage2DMD
    role_audits: dict[str, Stage2RoleInitializationAudit]
    adapter_digests: dict[str, str]
    # Immutable pre-FSDP schemas are required by both DCP optimizer state and
    # selective adapter checkpointing.  They cannot be reconstructed safely
    # from sharded parameters after initialization.
    lora_schemas: dict[str, Mapping[str, Any]]


def stage2_role_seed(training_seed: int, role: str) -> int:
    if isinstance(training_seed, bool) or not isinstance(training_seed, int):
        raise TypeError("Stage-2 training seed must be an integer")
    if training_seed < 0 or role not in {"generator", "fake_score"}:
        raise ValueError("invalid Stage-2 adapter seed input")
    digest = hashlib.sha256(f"stage2-adapter:{training_seed}:{role}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _default_stage_runner(label: str, callback: Callable[[], _T]) -> _T:
    del label
    return callback()


def _build_meta_role(resolved: Any, role: str) -> Stage2DiTRole:
    # Heavy Wan and Accelerate imports remain behind the init-only call.  Merely
    # importing Stage2DMD or asking the CLI for --help never imports T5/VAE code.
    from accelerate import init_empty_weights
    from utils.wan_5b_wrapper import build_wan_model

    is_causal = role == "generator"
    with init_empty_weights():
        transformer = build_wan_model(
            model_name=resolved.model_name,
            is_causal=is_causal,
            architecture_root=resolved.architecture_root,
            init_weights=False,
            # The public W=16 excludes the permanent S=1 sink.  The causal
            # transformer's internal attention span therefore uses the
            # derived physical capacity S+W=17 and protects one sink frame.
            local_attn_size=(resolved.physical_kv_capacity_frames if is_causal else -1),
            sink_size=(resolved.global_sink_frames if is_causal else 0),
            num_frame_per_block=(resolved.chunk_frames if is_causal else 1),
        )
    return Stage2DiTRole(transformer, role=role, is_causal=is_causal)


def _torch_load_weights(path: Path) -> Any:
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=True)
    except RuntimeError as exc:
        message = str(exc).lower()
        if "mmap" not in message or not any(
            token in message for token in ("zip", "serialization", "torch.save")
        ):
            raise
        # Legacy torch.save serialization cannot be mmap'ed.  Keep
        # weights_only=True and preserve every subsequent exact schema check.
        return torch.load(path, map_location="cpu", weights_only=True)


def _select_state_dict(payload: Any, selector: str) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, Mapping):
        raise TypeError("Stage-2 checkpoint payload must be a mapping")
    from utils.stage2_role_manifest import is_stage2_training_state_payload_key

    present_forbidden = sorted(
        (key for key in payload if is_stage2_training_state_payload_key(key)),
        key=repr,
    )
    if present_forbidden:
        raise ValueError(
            f"Stage-2 immutable base checkpoint contains training state: {present_forbidden}"
        )
    state = payload if selector == "root" else payload.get(selector)
    if not isinstance(state, Mapping):
        raise KeyError(
            f"Stage-2 checkpoint is missing manifest-selected state_dict {selector!r}"
        )
    if not state:
        raise ValueError("Stage-2 selected state_dict is empty")
    non_string = [key for key in state if not isinstance(key, str)]
    if non_string:
        raise TypeError("Stage-2 state_dict keys must all be strings")
    non_tensor = [
        key for key, value in state.items() if not isinstance(value, torch.Tensor)
    ]
    if non_tensor:
        raise TypeError(f"Stage-2 state_dict contains non-tensors: {non_tensor[:8]}")
    return state


def _audit_teacher_payload_contract(
    payload: Mapping[str, Any], selector: str, expected: Mapping[str, Any]
) -> None:
    if sorted(payload) != expected.get("top_level_keys"):
        raise ValueError("Stage-2 teacher payload top-level keys drifted")
    if expected.get("state_dict_selector") != selector:
        raise ValueError("Stage-2 teacher payload selector drifted")
    state = payload.get(selector)
    if not isinstance(state, Mapping) or len(state) != expected.get(
        "state_tensor_count"
    ):
        raise ValueError("Stage-2 teacher selected state tensor count drifted")
    if canonical_json_sha256(sorted(state)) != expected.get("state_dict_keys_sha256"):
        raise ValueError("Stage-2 teacher state_dict key schema drifted")
    if expected.get("load_target") not in {"role_wrapper", "bare_transformer"}:
        raise ValueError("Stage-2 teacher load_target contract is invalid")
    metadata = {key: value for key, value in payload.items() if key != selector}
    if metadata != expected.get("metadata"):
        raise ValueError("Stage-2 teacher payload scalar metadata drifted")


def _audit_immutable_state_dict(
    state: Mapping[str, torch.Tensor],
) -> dict[str, int]:
    lora_keys = [key for key in state if "lora_" in key]
    if lora_keys:
        raise ValueError(
            f"Stage-2 immutable full base still contains LoRA keys: {lora_keys[:8]}"
        )
    wrong_dtype = [
        (key, str(value.dtype))
        for key, value in state.items()
        if value.is_floating_point() and value.dtype != torch.bfloat16
    ]
    if wrong_dtype:
        raise TypeError(f"Stage-2 immutable full base must be BF16: {wrong_dtype[:8]}")
    nonfinite = [
        key
        for key, value in state.items()
        if value.is_floating_point() and not bool(torch.isfinite(value).all().item())
    ]
    if nonfinite:
        raise ValueError(
            f"Stage-2 immutable full base contains non-finite tensors: {nonfinite[:8]}"
        )
    return {
        "tensor_count": len(state),
        "parameter_count": sum(value.numel() for value in state.values()),
    }


def _strict_assign_state(
    target: torch.nn.Module,
    state: Mapping[str, torch.Tensor],
) -> None:
    has_meta = any(parameter.is_meta for parameter in target.parameters())
    try:
        incompatible = target.load_state_dict(
            state,
            strict=True,
            assign=has_meta,
        )
    except TypeError:
        if has_meta:
            raise RuntimeError(
                "Stage-2 meta initialization requires load_state_dict(assign=True)"
            )
        incompatible = target.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Stage-2 strict base reload returned incompatible keys: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    remaining_meta = [
        name for name, parameter in target.named_parameters() if parameter.is_meta
    ]
    remaining_meta.extend(
        name for name, buffer in target.named_buffers() if buffer.is_meta
    )
    if remaining_meta:
        raise RuntimeError(
            f"Stage-2 strict base load left meta parameters: {remaining_meta[:8]}"
        )


def strict_load_stage2_role_base(
    wrapper: Stage2DiTRole,
    *,
    asset: Mapping[str, Any],
    verify_content_hash: bool = False,
    content_hash_cache: set[tuple[str, str, tuple[tuple[str, int], ...]]] | None = None,
) -> dict[str, int]:
    """Load exactly the format and selector declared by a verified manifest."""

    path = Path(asset["checkpoint_path"])
    checkpoint_format = asset["checkpoint_format"]
    selector = asset["state_dict_selector"]
    _assert_verified_asset_files_unchanged(asset)
    if checkpoint_format == "wan_native_transformer":
        if wrapper.role == "generator":
            raise ValueError(
                "Stage-2 generator cannot use a native bidirectional teacher"
            )
        from wan_5b.textimage2video import load_wan_checkpoint_in_model

        load_wan_checkpoint_in_model(
            wrapper.model,
            path,
            # The manifest gate independently inspected every safetensors slice
            # as BF16.  Explicit BF16 is still required here because Accelerate
            # otherwise adopts the FP32 dtype of meta parameters.
            torch_dtype=torch.bfloat16,
        )
        state = wrapper.model.state_dict()
        summary = _audit_immutable_state_dict(state)
        del state
        _assert_verified_asset_files_unchanged(
            asset,
            verify_content_hash=verify_content_hash,
            content_hash_cache=content_hash_cache,
        )
        summary["strict_reload_succeeded"] = True
        return summary
    if checkpoint_format not in {
        "longlive_stage1_causal_ema_merged",
        "longlive_wrapper_pt",
    }:
        raise ValueError(
            f"unsupported verified Stage-2 checkpoint format: {checkpoint_format!r}"
        )
    payload = _torch_load_weights(path)
    if wrapper.role == "generator":
        expected_payload = {
            "checkpoint_format": "longlive_stage1_causal_ema_merged",
            "checkpoint_version": 1,
            "model_name": "Wan2.2-TI2V-5B",
            "source_training_step": 3075,
            "source_adapter": "adapter_ema.safetensors",
            "dtype": "bfloat16",
            "source_base_sha256": asset["source"]["base"]["sha256"],
        }
        expected_keys = {"generator", *expected_payload}
        if set(payload) != expected_keys:
            raise ValueError(
                "Stage-2 generator payload top-level schema mismatch: "
                f"expected={sorted(expected_keys)}, actual={sorted(payload)}"
            )
        wrong = {
            key: {"expected": expected, "actual": payload.get(key)}
            for key, expected in expected_payload.items()
            if payload.get(key) != expected
        }
        if wrong:
            raise ValueError(
                f"Stage-2 generator checkpoint payload metadata mismatch: {wrong}"
            )
    else:
        expected_contract = asset.get("payload_contract")
        if not isinstance(expected_contract, Mapping):
            raise ValueError("Stage-2 longlive teacher lacks payload_contract")
        _audit_teacher_payload_contract(payload, selector, expected_contract)
    state = _select_state_dict(payload, selector)
    summary = _audit_immutable_state_dict(state)
    if wrapper.role == "generator":
        if any(not key.startswith("model.") for key in state):
            raise ValueError(
                "Stage-2 merged generator state_dict must target the role wrapper"
            )
        target = wrapper
    else:
        load_target = asset["payload_contract"]["load_target"]
        target = wrapper if load_target == "role_wrapper" else wrapper.model
    _strict_assign_state(target, state)
    del state, payload
    gc.collect()
    _assert_verified_asset_files_unchanged(
        asset,
        verify_content_hash=verify_content_hash,
        content_hash_cache=content_hash_cache,
    )
    summary["strict_reload_succeeded"] = True
    return summary


def _audit_loaded_wan_contract(wrapper: Stage2DiTRole, resolved: Any) -> None:
    model = wrapper.model
    expected_class = "CausalWanModel" if wrapper.is_causal else "WanModel"
    if model.__class__.__name__ != expected_class:
        raise TypeError(
            f"Stage-2 {wrapper.role} expected {expected_class}, got "
            f"{model.__class__.__name__}"
        )
    expected = {
        "model_type": "ti2v",
        "patch_size": (1, 2, 2),
        "text_len": 512,
        "in_dim": 48,
        "dim": 3072,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "text_dim": 4096,
        "out_dim": 48,
        "num_heads": 24,
        "num_layers": 30,
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
    }
    if wrapper.is_causal:
        expected.update(
            {
                "local_attn_size": int(resolved.physical_kv_capacity_frames),
                "sink_size": int(resolved.global_sink_frames),
                "num_frame_per_block": int(resolved.chunk_frames),
                "use_relative_rope": False,
                "rope_method": "linear",
                "rope_temporal_offset": 0.0,
            }
        )
    else:
        expected["window_size"] = (-1, -1)
    wrong = {}
    for key, expected_value in expected.items():
        actual = getattr(model, key, None)
        if key in {"patch_size", "window_size"} and actual is not None:
            actual = tuple(actual)
        if actual != expected_value:
            wrong[key] = {"expected": expected_value, "actual": actual}
    if len(getattr(model, "blocks", ())) != 30:
        wrong["blocks"] = {"expected": 30, "actual": len(getattr(model, "blocks", ()))}
    if wrong:
        raise RuntimeError(f"Stage-2 {wrapper.role} Wan contract mismatch: {wrong}")


def _role_block_class(role: str) -> str:
    return "CausalWanAttentionBlock" if role == "generator" else "WanAttentionBlock"


def initialize_stage2_roles(
    resolved: Any,
    *,
    assets: Mapping[str, Mapping[str, Any]],
    mesh: Any,
    stage_runner: StageRunner = _default_stage_runner,
    role_builder: Callable[[Any, str], Stage2DiTRole] = _build_meta_role,
    is_main_process: bool = False,
    resume_adapter_states: Mapping[str, Mapping[str, torch.Tensor]] | None = None,
) -> Stage2InitializedRoles:
    """Build/load/audit/shard G, real, and fake sequentially.

    The caller must verify manifests before entering this function.  Each local
    construction stage is separated from its collective FSDP stage so a
    distributed caller can enforce WORLD consensus before any rank advances.
    """

    mode = getattr(resolved, "initialization_mode", None)
    if resume_adapter_states is not None:
        if mode not in {"init_from_stage1", "resume_stage2"} or set(
            resume_adapter_states
        ) != {
            "generator",
            "fake_score",
        }:
            raise ValueError(
                "Stage-2 resume requires exact generator/fake_score raw adapters"
            )
    elif mode == "resume_stage2":
        raise ValueError("Stage-2 resume mode requires raw checkpoint adapters")
    elif mode == "init_from_stage1":
        pass
    else:
        raise ValueError(f"unsupported Stage-2 initialization_mode={mode!r}")
    expected_asset_roles = {"generator", "real_score", "fake_score"}
    if set(assets) != expected_asset_roles:
        raise ValueError(f"Stage-2 assets must contain {sorted(expected_asset_roles)}")
    if (
        assets["real_score"]["checkpoint_sha256"]
        != assets["fake_score"]["checkpoint_sha256"]
    ):
        raise ValueError("Stage-2 fake_score must use the exact real_score base SHA")

    wrappers: dict[str, Stage2DiTRole] = {}
    audits: dict[str, Stage2RoleInitializationAudit] = {}
    adapter_digests: dict[str, str] = {}
    lora_schemas: dict[str, Mapping[str, Any]] = {}
    content_hash_cache: set[tuple[str, str, tuple[tuple[str, int], ...]]] = set()
    for role in ("generator", "real_score", "fake_score"):
        asset = assets[role]

        def construct_and_audit() -> tuple[
            Stage2DiTRole,
            dict[str, int],
            Stage2TargetAudit | None,
            Mapping[str, Any] | None,
            str | None,
            int | None,
        ]:
            _assert_verified_asset_files_unchanged(asset)
            wrapper = role_builder(resolved, role)
            if not isinstance(wrapper, Stage2DiTRole) or wrapper.role != role:
                raise TypeError(f"role_builder returned an invalid {role} wrapper")
            base_summary = strict_load_stage2_role_base(
                wrapper,
                asset=asset,
                verify_content_hash=is_main_process,
                content_hash_cache=content_hash_cache,
            )
            _audit_loaded_wan_contract(wrapper, resolved)
            wrapper.requires_grad_(False)
            adapter_audit = None
            lora_schema = None
            adapter_digest = None
            adapter_seed = None
            if role == "real_score":
                wrapper.eval()
                if bool(getattr(wrapper.model, "is_gradient_checkpointing", False)):
                    raise RuntimeError(
                        "Stage-2 real_score activation checkpointing must be disabled"
                    )
                audit_stage2_frozen_real_score(
                    wrapper,
                    expected_base_dtype=torch.bfloat16,
                )
            else:
                if role == "fake_score":
                    wrapper.model.enable_gradient_checkpointing()
                checkpointing = bool(
                    getattr(wrapper.model, "is_gradient_checkpointing", False)
                )
                if checkpointing != (role == "fake_score"):
                    raise RuntimeError(
                        f"Stage-2 {role} activation-checkpointing contract failed: "
                        f"actual={checkpointing}"
                    )
                wrapper.train()
                spec = (
                    resolved.generator_adapter
                    if role == "generator"
                    else resolved.fake_score_adapter
                )
                adapter_seed = stage2_role_seed(resolved.training_seed, role)
                wrapper.model, adapter_audit = configure_stage2_role_lora(
                    wrapper.model,
                    role=role,
                    spec=spec,
                    seed=adapter_seed,
                    is_main_process=is_main_process,
                )
                if resume_adapter_states is not None:
                    strict_load_lora_state_dict(
                        wrapper.model,
                        resume_adapter_states[role],
                        expected_dtype=torch.float32,
                        require_finite=True,
                        verify_tensors=True,
                    )
                lora_schema = build_lora_shard_schema(
                    wrapper.model,
                    expected_dtype=torch.float32,
                )
                adapter_digest = stage2_adapter_digest(wrapper.model)
            return (
                wrapper,
                base_summary,
                adapter_audit,
                lora_schema,
                adapter_digest,
                adapter_seed,
            )

        (
            wrapper,
            base_summary,
            adapter_audit,
            lora_schema,
            adapter_digest,
            adapter_seed,
        ) = stage_runner(f"{role}: build/load/local audit", construct_and_audit)
        if role == "generator":
            expected_tensors = resolved.generator_adapter.expected_adapter_tensors
            expected_numel = resolved.generator_adapter.expected_trainable_parameters
        elif role == "fake_score":
            expected_tensors = resolved.fake_score_adapter.expected_adapter_tensors
            expected_numel = resolved.fake_score_adapter.expected_trainable_parameters
        else:
            expected_tensors = 0
            expected_numel = 0
        wrapper = stage_runner(
            f"{role}: FSDP2 FULL_SHARD",
            lambda: fsdp2_wrap_stage2_role(
                wrapper,
                transformer=wrapper.model,
                mesh=mesh,
                role=role,
                block_class_name=_role_block_class(role),
                expected_trainable_tensors=expected_tensors,
                expected_trainable_parameters=expected_numel,
            ),
        )
        post_fsdp = stage_runner(
            f"{role}: post-FSDP2 audit",
            lambda: audit_stage2_fsdp2_role(
                wrapper,
                role=role,
                expected_schema=lora_schema,
            ),
        )

        def audit_activation_checkpointing() -> bool:
            actual = bool(getattr(wrapper.model, "is_gradient_checkpointing", False))
            if actual != (role == "fake_score"):
                raise RuntimeError(
                    f"Stage-2 {role} post-FSDP activation-checkpointing drifted: "
                    f"actual={actual}"
                )
            return actual

        activation_checkpointing = stage_runner(
            f"{role}: post-FSDP activation-checkpoint audit",
            audit_activation_checkpointing,
        )
        canonical_keys = tuple(lora_schema or ())
        audits[role] = Stage2RoleInitializationAudit(
            role=role,
            backbone=("causal" if role == "generator" else "bidirectional_ti2v"),
            base_checkpoint_sha256=asset["checkpoint_sha256"],
            base_state_tensor_count=base_summary["tensor_count"],
            base_state_parameter_count=base_summary["parameter_count"],
            base_dtype="bfloat16",
            strict_reload_succeeded=bool(
                base_summary.get("strict_reload_succeeded", False)
            ),
            trainable_policy=("frozen" if role == "real_score" else "adapter_only"),
            activation_checkpointing=activation_checkpointing,
            adapter_seed=adapter_seed,
            adapter_digest=adapter_digest,
            target_audit=(adapter_audit.to_dict() if adapter_audit else None),
            canonical_adapter_keys=canonical_keys,
            pre_fsdp_adapter_tensor_count=len(canonical_keys),
            pre_fsdp_trainable_parameters=(
                sum(
                    int(torch.Size(spec.global_shape).numel())
                    for spec in (lora_schema or {}).values()
                )
            ),
            post_fsdp=post_fsdp,
        )
        wrappers[role] = wrapper
        if adapter_digest is not None:
            adapter_digests[role] = adapter_digest
        if lora_schema is not None:
            lora_schemas[role] = lora_schema
        gc.collect()

    model = Stage2DMD(
        generator=wrappers["generator"],
        real_score=wrappers["real_score"],
        fake_score=wrappers["fake_score"],
    )
    return Stage2InitializedRoles(
        model=model,
        role_audits=audits,
        adapter_digests=adapter_digests,
        lora_schemas=lora_schemas,
    )
