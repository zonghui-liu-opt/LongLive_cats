# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""Shared LoRA configuration and adapter checkpoint helpers.

The Stage-1 path deliberately keeps two concepts separate:

* canonical adapter state dictionaries are complete, CPU tensors suitable for
  safetensors files; and
* selective sharded state dictionaries contain only local LoRA shards and are
  never promoted to an FSDP full state dictionary by this module.

Legacy callers without ``target_patterns`` retain the original attention-block
scan.  New Stage-1 callers use exact, full-name regular expressions.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import os
from pathlib import Path
import hashlib
import json
import re
import tempfile
from typing import Any

import peft
from peft import get_peft_model_state_dict
import torch
from torch.distributed.tensor import DTensor, Replicate, Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from utils.parameter_names import map_parameter_names_to_expected

_LORA_PARAMETER_MARKERS = (".lora_A.", ".lora_B.")
_CANONICAL_LORA_KEY_MARKERS = (".lora_A.weight", ".lora_B.weight")
STAGE2_LORA_LOAD_API_VERSION = "longlive_stage2_lora_load/v1"


@dataclass(frozen=True)
class LoraTensorSpec:
    """Pre-FSDP immutable schema for one canonical adapter tensor."""

    raw_parameter_name: str
    global_shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass(frozen=True)
class LocalLoraShard:
    """One rank's local fragment of a canonical LoRA tensor.

    The original seven fields are retained for checkpoint compatibility.  The
    remaining fields describe the FSDP2 per-parameter DTensor layout without
    requiring a state-dict call or a full-tensor materialization.
    """

    tensor: torch.Tensor
    global_shape: tuple[int, ...]
    intra_param_start: int
    shard_rank: int
    shard_world_size: int
    shard_group_ranks: tuple[int, ...]
    fsdp_unit_fingerprint: str
    local_shape: tuple[int, ...] = ()
    mesh_shape: tuple[int, ...] = ()
    mesh_dim_names: tuple[str | None, ...] = ()
    placements: tuple[str, ...] = ()
    mesh_coordinate: tuple[int, ...] = ()
    mesh_ranks: tuple[tuple[int, ...], ...] = ()
    replica_group_ranks: tuple[int, ...] = ()
    authoritative_shard_group_ranks: tuple[int, ...] = ()
    shard_dim: int | None = None
    shard_offset: int = 0
    global_rank: int = 0
    is_dtensor: bool = False

    @property
    def global_numel(self) -> int:
        result = 1
        for size in self.global_shape:
            result *= size
        return result

    @property
    def intra_param_end(self) -> int:
        return self.intra_param_start + self.tensor.numel()

    @property
    def dtype(self) -> torch.dtype:
        """Expose the local tensor dtype without pretending this is full state."""

        return self.tensor.dtype


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _is_lora_parameter_name(name: str) -> bool:
    return any(marker in name for marker in _LORA_PARAMETER_MARKERS)


def _is_canonical_lora_key(name: str) -> bool:
    return any(marker in name for marker in _CANONICAL_LORA_KEY_MARKERS)


def _compile_target_patterns(target_patterns: Any) -> tuple[re.Pattern[str], ...]:
    if isinstance(target_patterns, str):
        target_patterns = [target_patterns]
    if not isinstance(target_patterns, Sequence) or not target_patterns:
        raise ValueError("adapter.target_patterns must be a non-empty sequence")

    compiled = []
    for pattern in target_patterns:
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(
                "every adapter.target_patterns entry must be a non-empty string"
            )
        try:
            compiled.append(re.compile(pattern))
        except re.error as exc:
            raise ValueError(f"invalid LoRA target pattern {pattern!r}: {exc}") from exc
    return tuple(compiled)


def resolve_lora_target_modules(
    transformer: torch.nn.Module,
    model_name: str,
    lora_config: Any,
    *,
    all_causal: bool = False,
) -> tuple[str, ...]:
    """Resolve deterministic full module names targeted by LoRA.

    When ``target_patterns`` is present, each regex is applied with
    :func:`re.fullmatch` to the transformer's full ``named_modules`` name.
    Matching a non-``Linear`` module or matching no modules is an error.  When
    the field is absent, the historical attention-block scan is preserved.
    """

    target_patterns = _config_get(lora_config, "target_patterns", None)
    named_modules = sorted(transformer.named_modules(), key=lambda item: item[0])

    if target_patterns is not None:
        compiled = _compile_target_patterns(target_patterns)
        matched_by_pattern: list[list[str]] = [[] for _ in compiled]
        selected: set[str] = set()
        non_linear: list[str] = []

        for name, module in named_modules:
            matching_indices = [
                index
                for index, pattern in enumerate(compiled)
                if pattern.fullmatch(name)
            ]
            if not matching_indices:
                continue
            for index in matching_indices:
                matched_by_pattern[index].append(name)
            if not isinstance(module, torch.nn.Linear):
                non_linear.append(name or "<root>")
            else:
                selected.add(name)

        unmatched = [
            pattern.pattern
            for pattern, matches in zip(compiled, matched_by_pattern)
            if not matches
        ]
        if unmatched:
            raise ValueError(f"LoRA target patterns matched no modules: {unmatched}")
        if non_linear:
            raise TypeError(
                "LoRA target patterns may only match torch.nn.Linear modules; "
                f"matched non-linear modules: {sorted(set(non_linear))}"
            )
        if not selected:
            raise ValueError("LoRA target patterns resolved to no Linear modules")
        return tuple(sorted(selected))

    # Legacy behavior: scan all Linear descendants of the selected attention
    # block class.  Keep this path intentionally narrow to avoid changing DMD.
    if model_name == "generator":
        adapter_target_modules = {"CausalWanAttentionBlock"}
    elif model_name == "fake_score":
        adapter_target_modules = {
            "CausalWanAttentionBlock" if all_causal else "WanAttentionBlock"
        }
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    target_linear_modules: set[str] = set()
    for name, module in named_modules:
        if module.__class__.__name__ not in adapter_target_modules:
            continue
        for full_name, submodule in module.named_modules(prefix=name):
            if isinstance(submodule, torch.nn.Linear):
                target_linear_modules.add(full_name)
    return tuple(sorted(target_linear_modules))


def assert_lora_b_weights_zero(lora_model: torch.nn.Module) -> tuple[str, ...]:
    """Fail unless every trainable LoRA-B parameter is exactly zero."""

    b_parameters = [
        (name, parameter)
        for name, parameter in lora_model.named_parameters()
        if ".lora_B." in name
    ]
    if not b_parameters:
        raise ValueError("LoRA model contains no lora_B parameters")

    nonzero = []
    for name, parameter in b_parameters:
        if parameter.numel() and torch.count_nonzero(parameter.detach()).item() != 0:
            nonzero.append(name)
    if nonzero:
        raise ValueError(
            f"LoRA-B initialization must be all-zero; nonzero tensors: {nonzero}"
        )
    return tuple(name for name, _ in b_parameters)


def get_canonical_lora_state_dict(
    lora_model: torch.nn.Module,
    *,
    state_dict: Mapping[str, Any] | None = None,
) -> OrderedDict[str, Any]:
    """Return PEFT's canonical adapter keys in deterministic order.

    Passing ``state_dict`` is important for selective FSDP workflows: PEFT then
    filters the already-selective mapping and never asks the model for a full
    state dictionary.
    """

    has_fsdp1 = any(isinstance(module, FSDP) for module in lora_model.modules())
    has_dtensor = any(
        isinstance(parameter, DTensor) for parameter in lora_model.parameters()
    )
    if state_dict is None and (has_fsdp1 or has_dtensor):
        raise RuntimeError(
            "canonical adapter extraction from a sharded model is forbidden; "
            "use get_lora_sharded_state_dict() or a fresh unsharded PeftModel instead"
        )
    adapter_state = get_peft_model_state_dict(lora_model, state_dict=state_dict)
    return OrderedDict((key, adapter_state[key]) for key in sorted(adapter_state))


def audit_lora_model(
    lora_model: torch.nn.Module,
    *,
    target_module_names: Sequence[str] | None = None,
    expected_target_modules: int | None = None,
    expected_trainable_parameters: int | None = None,
    expected_adapter_tensors: int | None = None,
    expected_trainable_dtype: torch.dtype | None = None,
    require_lora_only: bool = True,
    check_b_zero: bool = False,
) -> dict[str, Any]:
    """Audit an unsharded PEFT model before FSDP wrapping."""

    trainable = sorted(
        (name, parameter)
        for name, parameter in lora_model.named_parameters()
        if parameter.requires_grad
    )
    trainable_names = tuple(name for name, _ in trainable)
    non_lora_trainable = [
        name for name in trainable_names if not _is_lora_parameter_name(name)
    ]
    if require_lora_only and non_lora_trainable:
        raise ValueError(f"non-LoRA trainable parameters found: {non_lora_trainable}")
    if expected_trainable_dtype is not None:
        wrong_dtypes = {
            name: parameter.dtype
            for name, parameter in trainable
            if parameter.dtype != expected_trainable_dtype
        }
        if wrong_dtypes:
            raise TypeError(
                "LoRA master parameters have unexpected dtypes: "
                f"expected {expected_trainable_dtype}, got {wrong_dtypes}"
            )

    canonical_state = get_canonical_lora_state_dict(lora_model)
    non_adapter_keys = [
        key for key in canonical_state if not _is_canonical_lora_key(key)
    ]
    if require_lora_only and non_adapter_keys:
        raise ValueError(
            f"non-LoRA tensors found in canonical adapter state: {non_adapter_keys}"
        )

    actual_target_prefixes = {
        key.rsplit(".lora_A.weight", 1)[0]
        for key in canonical_state
        if key.endswith(".lora_A.weight")
    }
    target_module_count = len(actual_target_prefixes)
    selected_target_names = tuple(sorted(target_module_names or ()))
    if selected_target_names and target_module_count != len(selected_target_names):
        raise ValueError(
            "PEFT target count differs from the exact resolved allowlist: "
            f"resolved={len(selected_target_names)}, wrapped={target_module_count}"
        )

    trainable_parameter_count = sum(parameter.numel() for _, parameter in trainable)
    adapter_tensor_count = len(canonical_state)

    expected_values = (
        ("target modules", expected_target_modules, target_module_count),
        (
            "trainable parameters",
            expected_trainable_parameters,
            trainable_parameter_count,
        ),
        ("adapter tensors", expected_adapter_tensors, adapter_tensor_count),
    )
    for label, expected, actual in expected_values:
        if expected is not None and actual != int(expected):
            raise ValueError(
                f"unexpected LoRA {label}: expected {expected}, got {actual}"
            )

    if check_b_zero:
        assert_lora_b_weights_zero(lora_model)

    return {
        "target_module_names": selected_target_names,
        "target_module_count": target_module_count,
        "trainable_parameter_names": trainable_names,
        "trainable_parameter_count": trainable_parameter_count,
        "adapter_tensor_keys": tuple(canonical_state),
        "adapter_tensor_count": adapter_tensor_count,
    }


def configure_lora_for_model(
    transformer,
    model_name,
    lora_config,
    is_main_process=True,
    all_causal=False,
):
    """Configure LoRA for a Wan transformer, preserving the legacy scan path."""

    target_linear_modules = resolve_lora_target_modules(
        transformer,
        model_name,
        lora_config,
        all_causal=all_causal,
    )
    if not target_linear_modules:
        raise ValueError(f"No LoRA target Linear modules found for {model_name}")

    if is_main_process:
        print(
            f"LoRA target modules for {model_name}: "
            f"{len(target_linear_modules)} Linear layers"
        )
        if _config_get(lora_config, "verbose", False):
            for module_name in target_linear_modules:
                print(f"  - {module_name}")

    adapter_type = _config_get(lora_config, "type", "lora")
    if adapter_type != "lora":
        raise NotImplementedError(f"Adapter type {adapter_type} is not implemented")

    rank = int(_config_get(lora_config, "rank", 16))
    alpha = _config_get(lora_config, "alpha", None) or rank
    exact_targeting = _config_get(lora_config, "target_patterns", None) is not None

    # Explicit bias/modules_to_save are part of the Stage-1 exact configuration.
    # The legacy branch historically ignored these fields, so retain that behavior.
    bias = _config_get(lora_config, "bias", "none") if exact_targeting else "none"
    modules_to_save = (
        list(_config_get(lora_config, "modules_to_save", []) or [])
        if exact_targeting
        else None
    )
    peft_config = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=float(_config_get(lora_config, "dropout", 0.0)),
        bias=bias,
        modules_to_save=modules_to_save,
        target_modules=list(target_linear_modules),
    )
    lora_model = peft.get_peft_model(
        transformer,
        peft_config,
        autocast_adapter_dtype=True,
    )

    audit = audit_lora_model(
        lora_model,
        target_module_names=target_linear_modules,
        expected_target_modules=_config_get(
            lora_config, "expected_target_modules", None
        ),
        expected_trainable_parameters=_config_get(
            lora_config, "expected_trainable_parameters", None
        ),
        expected_adapter_tensors=_config_get(
            lora_config, "expected_adapter_tensors", None
        ),
        expected_trainable_dtype=torch.float32 if exact_targeting else None,
        require_lora_only=exact_targeting,
        check_b_zero=True,
    )
    # This metadata is non-persistent and helps later pre-FSDP diagnostics.
    lora_model._longlive_lora_audit = audit

    if is_main_process:
        print("peft_config", peft_config)
        lora_model.print_trainable_parameters()
    return lora_model


def validate_canonical_lora_state_dict(
    lora_model: torch.nn.Module,
    adapter_state_dict: Mapping[str, torch.Tensor],
    *,
    expected_dtype: torch.dtype | None = None,
    require_finite: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Validate exact canonical keys, shapes, dtypes, and finite values."""

    if not isinstance(adapter_state_dict, Mapping):
        raise TypeError("adapter state must be a mapping")
    expected = get_canonical_lora_state_dict(lora_model)
    expected_keys = set(expected)
    actual_keys = set(adapter_state_dict)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing or extra:
        raise ValueError(f"adapter key mismatch: missing={missing}, extra={extra}")

    validated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in sorted(expected):
        tensor = adapter_state_dict[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"adapter tensor {key!r} is not a torch.Tensor")
        expected_tensor = expected[key]
        if tuple(tensor.shape) != tuple(expected_tensor.shape):
            raise ValueError(
                f"adapter shape mismatch for {key}: "
                f"expected {tuple(expected_tensor.shape)}, got {tuple(tensor.shape)}"
            )
        required_dtype = expected_dtype or expected_tensor.dtype
        if tensor.dtype != required_dtype:
            raise TypeError(
                f"adapter dtype mismatch for {key}: expected {required_dtype}, got {tensor.dtype}"
            )
        if require_finite and (tensor.is_floating_point() or tensor.is_complex()):
            if not bool(torch.isfinite(tensor).all().item()):
                raise ValueError(f"adapter tensor {key!r} contains non-finite values")
        validated[key] = tensor
    return validated


def strict_load_lora_state_dict(
    lora_model: torch.nn.Module,
    adapter_state_dict: Mapping[str, torch.Tensor],
    *,
    expected_dtype: torch.dtype | None = None,
    require_finite: bool = True,
    verify_tensors: bool = True,
):
    """Strictly load canonical default-adapter LoRA A/B tensors.

    PEFT's generic loader probes Hugging Face tensor parallelism whenever a
    distributed process group exists, even when this model uses only FSDP2.
    That optional probe makes checkpoint resume depend on a Transformers
    integration which is irrelevant to this project.  Stage-2 has a narrower
    contract: one complete default LoRA adapter is loaded before FSDP wrapping.
    Resolve that exact canonical/runtime bijection here and use PyTorch's
    native partial state load while retaining the existing value audit.
    """

    validated = validate_canonical_lora_state_dict(
        lora_model,
        adapter_state_dict,
        expected_dtype=expected_dtype,
        require_finite=require_finite,
    )

    runtime_by_canonical: dict[str, str] = {}
    for runtime_name, _parameter in lora_model.named_parameters():
        if not _is_lora_parameter_name(runtime_name):
            continue
        canonical_key = _canonical_key_from_parameter_name(runtime_name)
        previous = runtime_by_canonical.get(canonical_key)
        if previous is not None:
            raise ValueError(
                "canonical LoRA key maps to multiple runtime parameters: "
                f"key={canonical_key!r}, parameters={[previous, runtime_name]}"
            )
        runtime_by_canonical[canonical_key] = runtime_name

    expected_keys = set(validated)
    runtime_keys = set(runtime_by_canonical)
    if runtime_keys != expected_keys:
        raise ValueError(
            "canonical/runtime LoRA parameter mapping is incomplete: "
            f"missing={sorted(expected_keys - runtime_keys)}, "
            f"extra={sorted(runtime_keys - expected_keys)}"
        )
    runtime_state = OrderedDict(
        (runtime_by_canonical[key], validated[key]) for key in sorted(validated)
    )
    incompatible = lora_model.load_state_dict(runtime_state, strict=False)
    missing_adapter = sorted(set(incompatible.missing_keys).intersection(runtime_state))
    unexpected = list(getattr(incompatible, "unexpected_keys", []) or [])
    if missing_adapter or unexpected:
        raise ValueError(
            "PyTorch rejected canonical LoRA adapter tensors: "
            f"missing={missing_adapter}, unexpected={unexpected}"
        )

    if verify_tensors:
        loaded = get_canonical_lora_state_dict(lora_model)
        if set(loaded) != set(validated):
            raise RuntimeError(
                "loaded PEFT adapter keys changed after strict validation"
            )
        unequal = []
        for key, source in validated.items():
            actual = loaded[key]
            expected_value = source.detach().to(
                device=actual.device, dtype=actual.dtype
            )
            if not torch.equal(actual.detach(), expected_value):
                unequal.append(key)
        if unequal:
            raise RuntimeError(f"PEFT adapter tensor verification failed: {unequal}")
    return incompatible


def save_lora_safetensors_strict(
    lora_model: torch.nn.Module,
    path: str | os.PathLike[str],
    *,
    adapter_state_dict: Mapping[str, torch.Tensor] | None = None,
    dtype: torch.dtype = torch.float32,
    metadata: Mapping[str, Any] | None = None,
) -> OrderedDict[str, torch.Tensor]:
    """Atomically save a complete canonical adapter as contiguous CPU tensors."""

    from safetensors.torch import save_file

    source = (
        get_canonical_lora_state_dict(lora_model)
        if adapter_state_dict is None
        else adapter_state_dict
    )
    canonical = OrderedDict(
        (
            key,
            tensor.detach().to(device="cpu", dtype=dtype).contiguous(),
        )
        for key, tensor in sorted(source.items())
    )
    validated = validate_canonical_lora_state_dict(
        lora_model,
        canonical,
        expected_dtype=dtype,
        require_finite=True,
    )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    try:
        safe_metadata = (
            {str(key): str(value) for key, value in metadata.items()}
            if metadata
            else None
        )
        save_file(dict(validated), temporary_path, metadata=safe_metadata)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
    return validated


def load_lora_safetensors_strict(
    lora_model: torch.nn.Module,
    path: str | os.PathLike[str],
    *,
    expected_dtype: torch.dtype = torch.float32,
    require_finite: bool = True,
    verify_tensors: bool = True,
) -> OrderedDict[str, torch.Tensor]:
    """Load a safetensors adapter after exact schema and value validation."""

    from safetensors.torch import load_file

    state = OrderedDict(
        sorted(load_file(str(path), device="cpu").items(), key=lambda item: item[0])
    )
    strict_load_lora_state_dict(
        lora_model,
        state,
        expected_dtype=expected_dtype,
        require_finite=require_finite,
        verify_tensors=verify_tensors,
    )
    return state


def _clean_fsdp_parameter_name(name: str) -> str:
    return (
        name.replace("_fsdp_wrapped_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("_orig_mod.", "")
    )


def _canonical_key_from_parameter_name(name: str) -> str:
    cleaned = _clean_fsdp_parameter_name(name)
    cleaned = cleaned.replace(".lora_A.default.", ".lora_A.")
    cleaned = cleaned.replace(".lora_B.default.", ".lora_B.")
    if not _is_canonical_lora_key(cleaned):
        raise ValueError(
            f"trainable non-LoRA parameter is not a default LoRA A/B tensor: {name!r}"
        )
    return cleaned


def build_lora_shard_schema(
    lora_model: torch.nn.Module,
    *,
    expected_dtype: torch.dtype = torch.float32,
) -> OrderedDict[str, LoraTensorSpec]:
    """Capture canonical key/shape/dtype before FSDP wrapping."""

    if any(isinstance(module, FSDP) for module in lora_model.modules()):
        raise ValueError("LoRA shard schema must be captured before FSDP wrapping")
    dtensor_parameters = [
        name
        for name, parameter in lora_model.named_parameters()
        if isinstance(parameter, DTensor)
    ]
    if dtensor_parameters:
        raise ValueError(
            "LoRA shard schema must be captured before FSDP2 fully_shard; "
            f"found DTensor parameters {dtensor_parameters}"
        )
    flattened = [
        name
        for name, parameter in lora_model.named_parameters()
        if bool(getattr(parameter, "_fsdp_flattened", False))
    ]
    if flattened:
        raise ValueError(
            "LoRA shard schema must be captured before FSDP wrapping; "
            f"found already-flattened parameters {flattened}"
        )
    canonical = get_canonical_lora_state_dict(lora_model)
    specs: dict[str, LoraTensorSpec] = {}
    for raw_name, parameter in lora_model.named_parameters():
        if not parameter.requires_grad:
            continue
        canonical_key = _canonical_key_from_parameter_name(raw_name)
        if canonical_key in specs:
            raise ValueError(f"duplicate canonical LoRA parameter key: {canonical_key}")
        if parameter.dtype != expected_dtype:
            raise TypeError(
                f"LoRA master dtype mismatch for {raw_name}: "
                f"expected {expected_dtype}, got {parameter.dtype}"
            )
        specs[canonical_key] = LoraTensorSpec(
            raw_parameter_name=_clean_fsdp_parameter_name(raw_name),
            global_shape=tuple(parameter.shape),
            dtype=parameter.dtype,
        )

    if set(specs) != set(canonical):
        raise ValueError(
            "pre-FSDP LoRA shard schema differs from PEFT canonical state: "
            f"missing={sorted(set(canonical) - set(specs))}, "
            f"extra={sorted(set(specs) - set(canonical))}"
        )
    for key, spec in specs.items():
        if tuple(canonical[key].shape) != spec.global_shape:
            raise ValueError(
                f"pre-FSDP LoRA shape mismatch for {key}: "
                f"parameter={spec.global_shape}, PEFT={tuple(canonical[key].shape)}"
            )
    return OrderedDict((key, specs[key]) for key in sorted(specs))


def _match_expected_adapter_key(
    candidate: str,
    expected_schema: Mapping[str, LoraTensorSpec],
) -> str:
    return map_parameter_names_to_expected(
        (candidate,),
        expected_schema,
        label="FSDP LoRA canonical key",
        require_complete=False,
    )[candidate]


def _chunk_size_and_offset(
    dim_size: int,
    chunks: int,
    coordinate: int,
) -> tuple[int, int]:
    """Return the ``torch.chunk``-compatible size and offset for one shard."""

    if dim_size < 0 or chunks <= 0 or not 0 <= coordinate < chunks:
        raise ValueError(
            f"invalid shard geometry: dim_size={dim_size}, chunks={chunks}, "
            f"coordinate={coordinate}"
        )
    chunk_size = (dim_size + chunks - 1) // chunks
    offset = min(coordinate * chunk_size, dim_size)
    return min(chunk_size, dim_size - offset), offset


def _normalize_mesh_ranks(mesh_tensor: torch.Tensor) -> tuple[tuple[int, ...], ...]:
    if mesh_tensor.ndim == 1:
        return (
            tuple(int(rank) for rank in mesh_tensor.detach().to(device="cpu").tolist()),
        )
    if mesh_tensor.ndim != 2:
        raise ValueError(
            "selective FSDP2 LoRA export requires a one- or two-dimensional "
            f"DeviceMesh, got ndim={mesh_tensor.ndim}"
        )
    return tuple(
        tuple(int(rank) for rank in row)
        for row in mesh_tensor.detach().to(device="cpu").tolist()
    )


def _collect_fsdp2_lora_metadata(
    lora_model: torch.nn.Module,
    *,
    expected_schema: Mapping[str, LoraTensorSpec],
    expected_mesh_shape: Sequence[int],
    expected_mesh_ranks: Sequence[Sequence[int]] | None,
    expected_mesh_dim_names: Sequence[str],
    expected_dtype: torch.dtype,
) -> OrderedDict[str, dict[str, Any]]:
    """Inspect only trainable LoRA DTensors and their already-local shards."""

    if any(isinstance(module, FSDP) for module in lora_model.modules()):
        raise RuntimeError(
            "Stage-1 selective adapter export accepts FSDP2 DTensors only; "
            "FSDP1 FlatParam models are forbidden and have no full-state fallback"
        )
    trainable = [
        (name, parameter)
        for name, parameter in lora_model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("model contains no trainable LoRA parameters")
    non_dtensor = [
        name for name, parameter in trainable if not isinstance(parameter, DTensor)
    ]
    if non_dtensor:
        raise RuntimeError(
            "FSDP2 LoRA topology is ambiguous because some trainable parameters are not "
            f"DTensors: {non_dtensor}"
        )
    if not torch.distributed.is_initialized():
        raise RuntimeError("FSDP2 DTensor audit requires an initialized process group")

    required_mesh_shape = tuple(int(value) for value in expected_mesh_shape)
    required_mesh_dim_names = tuple(str(value) for value in expected_mesh_dim_names)
    if len(required_mesh_shape) not in (1, 2) or len(required_mesh_dim_names) != len(
        required_mesh_shape
    ):
        raise ValueError(
            "selective FSDP2 LoRA export requires matching one- or two-"
            f"dimensional mesh metadata, got shape={required_mesh_shape}, "
            f"names={required_mesh_dim_names}"
        )
    required_mesh_ranks = (
        tuple(tuple(int(rank) for rank in row) for row in expected_mesh_ranks)
        if expected_mesh_ranks is not None
        else None
    )
    metadata: dict[str, dict[str, Any]] = {}
    reference_topology: tuple[Any, ...] | None = None
    for raw_name, parameter in trainable:
        candidate_key = _canonical_key_from_parameter_name(raw_name)
        canonical_key = _match_expected_adapter_key(candidate_key, expected_schema)
        if canonical_key in metadata:
            raise ValueError(f"duplicate local LoRA DTensor key: {canonical_key}")
        spec = expected_schema[canonical_key]
        if parameter.dtype != spec.dtype or parameter.dtype != expected_dtype:
            raise TypeError(
                f"local LoRA master dtype mismatch for {canonical_key}: "
                f"schema={spec.dtype}, required={expected_dtype}, actual={parameter.dtype}"
            )
        global_shape = tuple(int(size) for size in parameter.shape)
        if global_shape != tuple(spec.global_shape):
            raise ValueError(
                f"FSDP2 global shape mismatch for {canonical_key}: "
                f"expected {spec.global_shape}, got {global_shape}"
            )

        mesh = parameter.device_mesh
        mesh_shape = tuple(int(size) for size in mesh.shape)
        if mesh_shape != required_mesh_shape:
            raise ValueError(
                f"wrong selective-LoRA DeviceMesh shape for {canonical_key}: "
                f"expected {required_mesh_shape}, got {mesh_shape}"
            )
        mesh_ranks = _normalize_mesh_ranks(mesh.mesh)
        if required_mesh_ranks is not None and mesh_ranks != required_mesh_ranks:
            raise ValueError(
                f"wrong selective-LoRA DeviceMesh ranks for {canonical_key}: "
                f"expected {required_mesh_ranks}, got {mesh_ranks}"
            )
        coordinate_value = mesh.get_coordinate()
        if coordinate_value is None:
            raise RuntimeError(
                f"current rank is not a member of the LoRA DeviceMesh for {canonical_key}"
            )
        coordinate = tuple(int(value) for value in coordinate_value)
        if len(coordinate) != len(required_mesh_shape):
            raise ValueError(
                f"wrong DeviceMesh coordinate for {canonical_key}: {coordinate}"
            )

        placements = tuple(parameter.placements)
        if len(required_mesh_shape) == 2:
            if not (
                len(placements) == 2
                and isinstance(placements[0], Replicate)
                and isinstance(placements[1], Shard)
                and int(placements[1].dim) == 0
            ):
                raise ValueError(
                    "two-dimensional selective LoRA DTensors require placements "
                    f"(Replicate(), Shard(dim=0)); {canonical_key} has {placements}"
                )
            placement_names = ("replicate", "shard:0")
            replica_coordinate, shard_coordinate = coordinate
            shard_mesh_size = mesh_shape[1]
            current_mesh_rank = mesh_ranks[replica_coordinate][shard_coordinate]
            shard_group_ranks = tuple(mesh_ranks[replica_coordinate])
            replica_group_ranks = tuple(row[shard_coordinate] for row in mesh_ranks)
            authoritative_ranks = tuple(mesh_ranks[0])
        else:
            if not (
                len(placements) == 1
                and isinstance(placements[0], Shard)
                and int(placements[0].dim) == 0
            ):
                raise ValueError(
                    "one-dimensional selective LoRA DTensors require placement "
                    f"Shard(dim=0); {canonical_key} has {placements}"
                )
            placement_names = ("shard:0",)
            (shard_coordinate,) = coordinate
            shard_mesh_size = mesh_shape[0]
            current_mesh_rank = mesh_ranks[0][shard_coordinate]
            shard_group_ranks = tuple(mesh_ranks[0])
            replica_group_ranks = (current_mesh_rank,)
            authoritative_ranks = tuple(mesh_ranks[0])
        mesh_dim_names_value = getattr(mesh, "mesh_dim_names", None)
        mesh_dim_names = (
            tuple(mesh_dim_names_value)
            if mesh_dim_names_value is not None
            else tuple(None for _ in required_mesh_shape)
        )
        if len(mesh_dim_names) != len(required_mesh_shape):
            raise ValueError(
                f"wrong DeviceMesh dim-name metadata for {canonical_key}: {mesh_dim_names}"
            )
        if mesh_dim_names != required_mesh_dim_names:
            raise ValueError(
                f"wrong selective-LoRA DeviceMesh dim names for {canonical_key}: "
                f"expected {required_mesh_dim_names}, got {mesh_dim_names}"
            )
        local_tensor = parameter.to_local()
        if not isinstance(local_tensor, torch.Tensor) or isinstance(
            local_tensor, DTensor
        ):
            raise TypeError(
                f"DTensor.to_local() did not return a local Tensor for {canonical_key}"
            )
        local_shape = tuple(int(size) for size in local_tensor.shape)
        local_dim_size, shard_offset = _chunk_size_and_offset(
            global_shape[0], shard_mesh_size, shard_coordinate
        )
        expected_local_shape = (local_dim_size, *global_shape[1:])
        if local_shape != expected_local_shape:
            raise ValueError(
                f"wrong local DTensor shape for {canonical_key}: "
                f"expected {expected_local_shape}, got {local_shape}"
            )
        global_rank = int(torch.distributed.get_rank())
        if current_mesh_rank != global_rank:
            raise ValueError(
                f"DeviceMesh coordinate/rank mismatch for {canonical_key}: "
                f"coordinate={coordinate}, rank={global_rank}, mesh={mesh_ranks}"
            )
        row_numel = int(torch.Size(global_shape[1:]).numel())
        intra_param_start = shard_offset * row_numel
        fingerprint_payload = {
            "key": canonical_key,
            "global_shape": global_shape,
            "dtype": str(parameter.dtype),
            "mesh_shape": mesh_shape,
            "mesh_ranks": mesh_ranks,
            "placements": placement_names,
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        topology = (mesh_shape, mesh_dim_names, mesh_ranks, placement_names)
        if reference_topology is None:
            reference_topology = topology
        elif topology != reference_topology:
            raise ValueError(
                "LoRA DTensors do not share one unambiguous topology: "
                f"{canonical_key}"
            )
        metadata[canonical_key] = {
            "raw_name": raw_name,
            "parameter": parameter,
            "local_tensor": local_tensor,
            "global_shape": global_shape,
            "intra_param_start": intra_param_start,
            "shard_rank": shard_coordinate,
            "shard_world_size": shard_mesh_size,
            "shard_group_ranks": shard_group_ranks,
            "fsdp_unit_fingerprint": fingerprint,
            "local_shape": local_shape,
            "mesh_shape": mesh_shape,
            "mesh_dim_names": mesh_dim_names,
            "placements": placement_names,
            "mesh_coordinate": coordinate,
            "mesh_ranks": mesh_ranks,
            "replica_group_ranks": replica_group_ranks,
            "authoritative_shard_group_ranks": authoritative_ranks,
            "shard_dim": 0,
            "shard_offset": shard_offset,
            "global_rank": global_rank,
            "is_dtensor": True,
        }
    if set(metadata) != set(expected_schema):
        raise ValueError(
            "FSDP2 trainable LoRA keys differ from the immutable pre-shard schema: "
            f"missing={sorted(set(expected_schema) - set(metadata))}, "
            f"extra={sorted(set(metadata) - set(expected_schema))}"
        )
    return OrderedDict((key, metadata[key]) for key in sorted(metadata))


def audit_fsdp2_lora_dtensor_topology(
    lora_model: torch.nn.Module,
    *,
    expected_schema: Mapping[str, LoraTensorSpec],
    expected_mesh_shape: Sequence[int] = (2, 3),
    expected_mesh_ranks: Sequence[Sequence[int]] | None = ((0, 1, 2), (3, 4, 5)),
    expected_mesh_dim_names: Sequence[str] = ("replicate", "shard"),
    expected_dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    """Audit Stage-1 FSDP2 layout without touching any frozen parameter value."""

    metadata = _collect_fsdp2_lora_metadata(
        lora_model,
        expected_schema=expected_schema,
        expected_mesh_shape=expected_mesh_shape,
        expected_mesh_ranks=expected_mesh_ranks,
        expected_mesh_dim_names=expected_mesh_dim_names,
        expected_dtype=expected_dtype,
    )
    first = next(iter(metadata.values()))
    return {
        "adapter_tensor_count": len(metadata),
        "global_trainable_parameters": sum(
            int(torch.Size(spec.global_shape).numel())
            for spec in expected_schema.values()
        ),
        "canonical_keys": tuple(metadata),
        "mesh_shape": first["mesh_shape"],
        "mesh_dim_names": first["mesh_dim_names"],
        "mesh_ranks": first["mesh_ranks"],
        "placements": first["placements"],
        "mesh_coordinate": first["mesh_coordinate"],
        "shard_group_ranks": first["shard_group_ranks"],
        "replica_group_ranks": first["replica_group_ranks"],
        "authoritative_shard_group_ranks": first["authoritative_shard_group_ranks"],
    }


def get_lora_sharded_state_dict(
    lora_model: torch.nn.Module,
    *,
    cpu_offload: bool = True,
    expected_schema: Mapping[str, LoraTensorSpec] | None = None,
    expected_adapter_tensors: int | None = None,
    expected_keys: Iterable[str] | None = None,
    expected_shapes: Mapping[str, Sequence[int]] | None = None,
    expected_dtype: torch.dtype | None = torch.float32,
    require_finite: bool = True,
    expected_mesh_shape: Sequence[int] = (2, 3),
    expected_mesh_ranks: Sequence[Sequence[int]] | None = ((0, 1, 2), (3, 4, 5)),
    expected_mesh_dim_names: Sequence[str] = ("replicate", "shard"),
) -> OrderedDict[str, LocalLoraShard]:
    """Copy only FSDP2 local LoRA DTensor shards, with no state-dict call.

    FSDP1 is rejected outright.  A completely unsharded model remains supported
    as a narrow unit-test/fresh-model fallback; mixed sharded/unsharded trainable
    parameters are always an error.
    """

    if any(isinstance(module, FSDP) for module in lora_model.modules()):
        raise RuntimeError(
            "Stage-1 selective export does not support FSDP1 and will not fall "
            "back to a state_dict/full gather; use FSDP2 fully_shard"
        )
    named_parameters = list(lora_model.named_parameters())
    flattened = [
        name
        for name, parameter in named_parameters
        if bool(getattr(parameter, "_fsdp_flattened", False))
    ]
    if flattened:
        raise RuntimeError(
            "selective LoRA export received an inner/legacy FSDP shard without "
            "DTensor topology; pass the outer FSDP root (or migrate it to FSDP2): "
            f"{flattened}"
        )
    trainable_dtensor = any(
        parameter.requires_grad and isinstance(parameter, DTensor)
        for _, parameter in named_parameters
    )
    any_dtensor = any(
        isinstance(parameter, DTensor) for _, parameter in named_parameters
    )
    if trainable_dtensor and expected_schema is None:
        raise ValueError(
            "FSDP2 selective LoRA export requires the immutable pre-shard schema "
            "from build_lora_shard_schema()"
        )
    if any_dtensor and not trainable_dtensor:
        raise RuntimeError(
            "sharded frozen parameters with unsharded trainable LoRA are an "
            "ambiguous Stage-1 topology"
        )
    if expected_schema is None:
        expected_schema = build_lora_shard_schema(
            lora_model,
            expected_dtype=expected_dtype or torch.float32,
        )

    fsdp2_metadata: Mapping[str, Mapping[str, Any]] = {}
    if trainable_dtensor:
        fsdp2_metadata = _collect_fsdp2_lora_metadata(
            lora_model,
            expected_schema=expected_schema,
            expected_mesh_shape=expected_mesh_shape,
            expected_mesh_ranks=expected_mesh_ranks,
            expected_mesh_dim_names=expected_mesh_dim_names,
            expected_dtype=expected_dtype or torch.float32,
        )

    records: dict[str, LocalLoraShard] = {}
    for raw_name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        candidate_key = _canonical_key_from_parameter_name(raw_name)
        canonical_key = _match_expected_adapter_key(candidate_key, expected_schema)
        if canonical_key in records:
            raise ValueError(f"duplicate local LoRA shard key: {canonical_key}")
        spec = expected_schema[canonical_key]
        cleaned_raw_name = _clean_fsdp_parameter_name(raw_name)
        map_parameter_names_to_expected(
            (cleaned_raw_name,),
            (spec.raw_parameter_name,),
            label="post-FSDP LoRA raw name",
        )
        if parameter.dtype != spec.dtype:
            raise TypeError(
                f"local LoRA master dtype mismatch for {canonical_key}: "
                f"expected {spec.dtype}, got {parameter.dtype}"
            )
        if expected_dtype is not None and parameter.dtype != expected_dtype:
            raise TypeError(
                f"selective adapter dtype mismatch for {canonical_key}: "
                f"expected {expected_dtype}, got {parameter.dtype}"
            )

        if trainable_dtensor:
            if canonical_key not in fsdp2_metadata:
                raise RuntimeError(
                    f"missing audited FSDP2 metadata for {canonical_key}"
                )
            metadata = fsdp2_metadata[canonical_key]
            global_shape = tuple(metadata["global_shape"])
            source_local_tensor = metadata["local_tensor"]
        else:
            global_shape = tuple(parameter.shape)
            source_local_tensor = parameter
            metadata = {
                "intra_param_start": 0,
                "shard_rank": 0,
                "shard_world_size": 1,
                "shard_group_ranks": (0,),
                "fsdp_unit_fingerprint": "unwrapped",
                "local_shape": global_shape,
                "mesh_shape": (),
                "mesh_dim_names": (),
                "placements": (),
                "mesh_coordinate": (),
                "mesh_ranks": (),
                "replica_group_ranks": (0,),
                "authoritative_shard_group_ranks": (0,),
                "shard_dim": None,
                "shard_offset": 0,
                "global_rank": 0,
                "is_dtensor": False,
            }
        if global_shape != tuple(spec.global_shape):
            raise ValueError(
                f"selective adapter global shape mismatch for {canonical_key}: "
                f"expected {spec.global_shape}, got {global_shape}"
            )

        local_tensor = source_local_tensor.detach().clone().contiguous()
        if cpu_offload:
            local_tensor = local_tensor.to(device="cpu")
        if require_finite and local_tensor.numel():
            if not bool(torch.isfinite(local_tensor).all().item()):
                raise ValueError(
                    f"selective adapter local shard {canonical_key!r} contains non-finite values"
                )
        records[canonical_key] = LocalLoraShard(
            tensor=local_tensor,
            global_shape=global_shape,
            intra_param_start=int(metadata["intra_param_start"]),
            shard_rank=int(metadata["shard_rank"]),
            shard_world_size=int(metadata["shard_world_size"]),
            shard_group_ranks=tuple(metadata["shard_group_ranks"]),
            fsdp_unit_fingerprint=str(metadata["fsdp_unit_fingerprint"]),
            local_shape=tuple(metadata["local_shape"]),
            mesh_shape=tuple(metadata["mesh_shape"]),
            mesh_dim_names=tuple(metadata["mesh_dim_names"]),
            placements=tuple(metadata["placements"]),
            mesh_coordinate=tuple(metadata["mesh_coordinate"]),
            mesh_ranks=tuple(tuple(row) for row in metadata["mesh_ranks"]),
            replica_group_ranks=tuple(metadata["replica_group_ranks"]),
            authoritative_shard_group_ranks=tuple(
                metadata["authoritative_shard_group_ranks"]
            ),
            shard_dim=metadata["shard_dim"],
            shard_offset=int(metadata["shard_offset"]),
            global_rank=int(metadata["global_rank"]),
            is_dtensor=bool(metadata["is_dtensor"]),
        )

    canonical = OrderedDict((key, records[key]) for key in sorted(records))
    if set(canonical) != set(expected_schema):
        raise ValueError(
            "selective adapter key mismatch against pre-FSDP schema: "
            f"missing={sorted(set(expected_schema) - set(canonical))}, "
            f"extra={sorted(set(canonical) - set(expected_schema))}"
        )

    if expected_adapter_tensors is not None and len(canonical) != int(
        expected_adapter_tensors
    ):
        raise ValueError(
            "unexpected selective adapter tensor count: "
            f"expected {expected_adapter_tensors}, got {len(canonical)}"
        )
    if expected_keys is not None:
        expected_key_set = set(expected_keys)
        actual_key_set = set(canonical)
        if expected_key_set != actual_key_set:
            raise ValueError(
                "selective adapter key mismatch: "
                f"missing={sorted(expected_key_set - actual_key_set)}, "
                f"extra={sorted(actual_key_set - expected_key_set)}"
            )

    for key, value in canonical.items():
        shape = value.global_shape
        if expected_shapes is not None:
            if key not in expected_shapes:
                raise ValueError(f"selective adapter shape schema has no key {key!r}")
            expected_shape = tuple(expected_shapes[key])
            if shape != expected_shape:
                raise ValueError(
                    f"selective adapter shape mismatch for {key}: "
                    f"expected {expected_shape}, got {shape}"
                )
        value_dtype = value.tensor.dtype
        if expected_dtype is not None and value_dtype != expected_dtype:
            raise TypeError(
                f"selective adapter dtype mismatch for {key}: "
                f"expected {expected_dtype}, got {value_dtype}"
            )
    if expected_shapes is not None and set(expected_shapes) != set(canonical):
        extra_schema = sorted(set(expected_shapes) - set(canonical))
        raise ValueError(f"unused selective adapter shape schema keys: {extra_schema}")
    return canonical


def consolidate_lora_shards(
    shards_by_rank: Sequence[Mapping[str, LocalLoraShard]],
    *,
    expected_schema: Mapping[str, LoraTensorSpec],
) -> OrderedDict[str, torch.Tensor]:
    """Reconstruct canonical CPU tensors from one authoritative shard group."""

    if not shards_by_rank:
        raise ValueError("no LoRA shard mappings were provided")
    expected_keys = set(expected_schema)
    for index, rank_mapping in enumerate(shards_by_rank):
        if set(rank_mapping) != expected_keys:
            raise ValueError(
                f"LoRA shard mapping {index} key mismatch: "
                f"missing={sorted(expected_keys - set(rank_mapping))}, "
                f"extra={sorted(set(rank_mapping) - expected_keys)}"
            )

    consolidated: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key in sorted(expected_schema):
        spec = expected_schema[key]
        records = [rank_mapping[key] for rank_mapping in shards_by_rank]
        world_sizes = {record.shard_world_size for record in records}
        group_ranks = {record.shard_group_ranks for record in records}
        fingerprints = {record.fsdp_unit_fingerprint for record in records}
        shapes = {record.global_shape for record in records}
        if len(world_sizes) != 1 or len(group_ranks) != 1 or len(fingerprints) != 1:
            raise ValueError(
                f"inconsistent FSDP shard topology for adapter tensor {key}"
            )
        if shapes != {tuple(spec.global_shape)}:
            raise ValueError(
                f"inconsistent global shapes for {key}: expected {spec.global_shape}, got {shapes}"
            )
        shard_world_size = next(iter(world_sizes))
        shard_ranks = [record.shard_rank for record in records]
        if len(records) != shard_world_size or set(shard_ranks) != set(
            range(shard_world_size)
        ):
            raise ValueError(
                f"adapter tensor {key} requires exactly one authoritative record per shard rank; "
                f"world_size={shard_world_size}, ranks={shard_ranks}"
            )

        dtensor_flags = {record.is_dtensor for record in records}
        if len(dtensor_flags) != 1:
            raise ValueError(
                f"mixed DTensor/unsharded records for adapter tensor {key}"
            )
        if next(iter(dtensor_flags)):
            mesh_shapes = {record.mesh_shape for record in records}
            mesh_names = {record.mesh_dim_names for record in records}
            mesh_ranks = {record.mesh_ranks for record in records}
            placements = {record.placements for record in records}
            authoritative_groups = {
                record.authoritative_shard_group_ranks for record in records
            }
            replica_coordinates = {
                record.mesh_coordinate[0]
                for record in records
                if len(record.mesh_coordinate) == 2
            }
            if not all(len(record.mesh_coordinate) == 2 for record in records):
                raise ValueError(f"missing 2D mesh coordinate for adapter tensor {key}")
            if not (
                len(mesh_shapes) == len(mesh_names) == len(mesh_ranks) == 1
                and placements == {("replicate", "shard:0")}
                and len(authoritative_groups) == 1
                and replica_coordinates == {0}
            ):
                raise ValueError(
                    f"records for {key} are not one authoritative FSDP2 shard row"
                )
            authoritative_group = next(iter(authoritative_groups))
            if tuple(authoritative_group) != tuple(next(iter(group_ranks))):
                raise ValueError(
                    f"authoritative group metadata mismatch for adapter tensor {key}"
                )

        global_numel = int(torch.Size(spec.global_shape).numel())
        output = torch.empty(global_numel, dtype=spec.dtype, device="cpu")
        covered = torch.zeros(global_numel, dtype=torch.bool, device="cpu")
        for record in records:
            local = record.tensor.detach().to(device="cpu")
            if local.dtype != spec.dtype:
                raise TypeError(
                    f"local LoRA shard dtype mismatch for {key}: "
                    f"expected {spec.dtype}, got {local.dtype}"
                )
            local_shape = record.local_shape or tuple(local.shape)
            if tuple(local.shape) != tuple(local_shape):
                raise ValueError(
                    f"local shape metadata mismatch for {key}: "
                    f"metadata={local_shape}, tensor={tuple(local.shape)}"
                )
            if record.is_dtensor:
                expected_rows, expected_offset = _chunk_size_and_offset(
                    int(spec.global_shape[0]),
                    record.shard_world_size,
                    record.shard_rank,
                )
                expected_local_shape = (expected_rows, *tuple(spec.global_shape[1:]))
                if tuple(local_shape) != expected_local_shape:
                    raise ValueError(
                        f"wrong DTensor local shape for {key}: "
                        f"expected={expected_local_shape}, got={local_shape}"
                    )
                if record.shard_dim != 0 or record.shard_offset != expected_offset:
                    raise ValueError(
                        f"wrong DTensor shard metadata for {key}: "
                        f"dim={record.shard_dim}, offset={record.shard_offset}"
                    )
                expected_global_rank = record.authoritative_shard_group_ranks[
                    record.shard_rank
                ]
                if record.global_rank != expected_global_rank:
                    raise ValueError(
                        f"non-authoritative rank supplied for {key}: "
                        f"rank={record.global_rank}, expected={expected_global_rank}"
                    )
            start = record.intra_param_start
            end = record.intra_param_end
            if record.is_dtensor:
                row_numel = int(torch.Size(spec.global_shape[1:]).numel())
                expected_start = record.shard_offset * row_numel
                if start != expected_start:
                    raise ValueError(
                        f"wrong flattened shard offset for {key}: "
                        f"expected={expected_start}, got={start}"
                    )
            if start < 0 or end > global_numel:
                raise ValueError(
                    f"local LoRA shard interval is out of range for {key}: [{start}, {end})"
                )
            if local.numel() and bool(covered[start:end].any().item()):
                raise ValueError(f"overlapping local LoRA shard intervals for {key}")
            output[start:end].copy_(local.reshape(-1))
            covered[start:end] = True
        if not bool(covered.all().item()):
            missing_count = int((~covered).sum().item())
            raise ValueError(
                f"local LoRA shards leave {missing_count} uncovered values for {key}"
            )
        if output.numel() and not bool(torch.isfinite(output).all().item()):
            raise ValueError(
                f"consolidated LoRA tensor {key!r} contains non-finite values"
            )
        consolidated[key] = output.reshape(spec.global_shape)
    return consolidated


def validate_lora_replica_shards(
    replica_shards: Sequence[Mapping[str, LocalLoraShard]],
) -> dict[str, Any]:
    """Bitwise-validate the two DP replicas for one FSDP2 shard coordinate."""

    if len(replica_shards) != 2:
        raise ValueError(
            f"Stage-1 DP replica validation requires two mappings, got {len(replica_shards)}"
        )
    left, right = replica_shards
    if set(left) != set(right):
        raise ValueError(
            "DP replica LoRA keys differ: "
            f"left_only={sorted(set(left) - set(right))}, "
            f"right_only={sorted(set(right) - set(left))}"
        )

    def invariant(record: LocalLoraShard) -> tuple:
        return (
            record.global_shape,
            record.local_shape,
            record.tensor.dtype,
            record.shard_rank,
            record.shard_world_size,
            record.mesh_shape,
            record.mesh_dim_names,
            record.placements,
            record.mesh_ranks,
            record.replica_group_ranks,
            record.authoritative_shard_group_ranks,
            record.shard_dim,
            record.shard_offset,
            record.intra_param_start,
            record.fsdp_unit_fingerprint,
        )

    for key in sorted(left):
        records = (left[key], right[key])
        if not all(record.is_dtensor for record in records):
            raise ValueError(f"DP replica validation requires DTensor shards for {key}")
        if invariant(records[0]) != invariant(records[1]):
            raise ValueError(f"DP replica topology differs for adapter tensor {key}")
        coordinates = {record.mesh_coordinate for record in records}
        shard_coordinate = records[0].shard_rank
        if coordinates != {(0, shard_coordinate), (1, shard_coordinate)}:
            raise ValueError(
                f"wrong DP replica coordinates for {key}: {sorted(coordinates)}"
            )
        if {record.global_rank for record in records} != set(
            records[0].replica_group_ranks
        ):
            raise ValueError(f"wrong DP replica global ranks for adapter tensor {key}")
        left_tensor = records[0].tensor.detach().to(device="cpu")
        right_tensor = records[1].tensor.detach().to(device="cpu")
        if not torch.equal(left_tensor, right_tensor):
            raise ValueError(f"DP replica LoRA values differ for adapter tensor {key}")
    first = left[next(iter(sorted(left)))]
    return {
        "adapter_tensor_count": len(left),
        "shard_rank": first.shard_rank,
        "replica_group_ranks": first.replica_group_ranks,
    }


def _local_shard_digest(shards: Mapping[str, LocalLoraShard]) -> str:
    digest = hashlib.sha256()
    for key in sorted(shards):
        record = shards[key]
        payload = {
            "key": key,
            "global_shape": record.global_shape,
            "local_shape": record.local_shape,
            "dtype": str(record.tensor.dtype),
            "shard_rank": record.shard_rank,
            "shard_world_size": record.shard_world_size,
            "mesh_shape": record.mesh_shape,
            "mesh_dim_names": record.mesh_dim_names,
            "placements": record.placements,
            "mesh_ranks": record.mesh_ranks,
            "replica_group_ranks": record.replica_group_ranks,
            "authoritative_shard_group_ranks": record.authoritative_shard_group_ranks,
            "shard_dim": record.shard_dim,
            "shard_offset": record.shard_offset,
            "intra_param_start": record.intra_param_start,
            "fingerprint": record.fsdp_unit_fingerprint,
        }
        digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        tensor_bytes = (
            record.tensor.detach()
            .to(device="cpu")
            .contiguous()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        digest.update(tensor_bytes)
    return digest.hexdigest()


def _collective_flag_device(group: Any = None) -> torch.device:
    backend = str(torch.distributed.get_backend(group)).lower()
    if "nccl" in backend:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL collective requested without CUDA")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _world_consensus(local_success: bool) -> bool:
    flag = torch.tensor(
        1 if local_success else 0,
        dtype=torch.int32,
        device=_collective_flag_device(),
    )
    torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
    return bool(flag.item())


def _validate_local_fsdp2_shards_for_gather(
    local_shards: Mapping[str, LocalLoraShard],
    *,
    expected_schema: Mapping[str, LoraTensorSpec],
    global_rank: int,
) -> None:
    if set(local_shards) != set(expected_schema):
        raise ValueError("local LoRA shard keys differ from the expected schema")
    for key, spec in expected_schema.items():
        record = local_shards[key]
        if not isinstance(record, LocalLoraShard) or not record.is_dtensor:
            raise TypeError(
                f"distributed gather requires FSDP2 LocalLoraShard for {key}"
            )
        if record.global_rank != global_rank:
            raise ValueError(f"local shard rank metadata mismatch for {key}")
        if record.tensor.device.type != "cpu":
            raise ValueError(
                "gather_fsdp2_lora_state_dict requires cpu_offload=True local shards"
            )
        if (
            record.tensor.dtype != spec.dtype
            or record.global_shape != spec.global_shape
        ):
            raise ValueError(f"local shard schema mismatch for {key}")
        if record.mesh_shape != (2, 3) or record.placements != (
            "replicate",
            "shard:0",
        ):
            raise ValueError(f"wrong locked Stage-1 FSDP2 topology for {key}")
        if record.mesh_dim_names != ("replicate", "shard"):
            raise ValueError(f"wrong locked Stage-1 DeviceMesh dim names for {key}")
        if record.authoritative_shard_group_ranks != (0, 1, 2):
            raise ValueError(f"wrong authoritative shard ranks for {key}")
        if record.replica_group_ranks != (
            record.shard_rank,
            record.shard_rank + 3,
        ):
            raise ValueError(f"wrong DP replica group ranks for {key}")


def gather_fsdp2_lora_state_dict(
    local_shards: Mapping[str, LocalLoraShard],
    *,
    authoritative_shard_group: Any,
    replica_group: Any,
    dst_global_rank: int = 0,
    expected_schema: Mapping[str, LoraTensorSpec] | None = None,
) -> OrderedDict[str, torch.Tensor] | None:
    """Validate replicas and gather only ranks 0/1/2's LoRA shards to rank 0.

    All six ranks must enter this function in the same order.  DP pairs exchange
    only SHA-256 digests.  The three authoritative ranks then use a directed
    object gather; ranks 1--5 never materialize complete adapter tensors.
    """

    if not torch.distributed.is_initialized():
        raise RuntimeError("distributed FSDP2 LoRA gather requires torch.distributed")
    if torch.distributed.get_world_size() != 6:
        raise ValueError("Stage-1 FSDP2 LoRA gather requires WORLD_SIZE=6")
    if dst_global_rank != 0:
        raise ValueError("Stage-1 canonical adapter destination must be global rank 0")
    global_rank = int(torch.distributed.get_rank())
    if expected_schema is None:
        expected_schema = OrderedDict(
            (
                key,
                LoraTensorSpec(
                    raw_parameter_name="",
                    global_shape=tuple(record.global_shape),
                    dtype=record.tensor.dtype,
                ),
            )
            for key, record in sorted(local_shards.items())
        )

    local_error: Exception | None = None
    try:
        _validate_local_fsdp2_shards_for_gather(
            local_shards,
            expected_schema=expected_schema,
            global_rank=global_rank,
        )
        actual_replica_ranks = tuple(
            torch.distributed.get_process_group_ranks(replica_group)
        )
        first_record = next(iter(local_shards.values()))
        if actual_replica_ranks != first_record.replica_group_ranks:
            raise ValueError(
                f"replica process group mismatch: expected "
                f"{first_record.replica_group_ranks}, got {actual_replica_ranks}"
            )
    except Exception as exc:  # synchronize failure before any subgroup collective
        local_error = exc
    if not _world_consensus(local_error is None):
        raise RuntimeError(
            "FSDP2 LoRA local topology validation failed on at least one rank"
        ) from local_error

    pair_payloads: list[Any] = [None, None]
    torch.distributed.all_gather_object(
        pair_payloads,
        (global_rank, _local_shard_digest(local_shards)),
        group=replica_group,
    )
    pair_error: Exception | None = None
    expected_pair = next(iter(local_shards.values())).replica_group_ranks
    if tuple(item[0] for item in pair_payloads) != expected_pair:
        pair_error = ValueError(
            f"DP replica collective rank order mismatch: expected {expected_pair}, "
            f"got {tuple(item[0] for item in pair_payloads)}"
        )
    elif pair_payloads[0][1] != pair_payloads[1][1]:
        pair_error = ValueError(
            f"DP replica LoRA shards differ for ranks {expected_pair}"
        )
    if not _world_consensus(pair_error is None):
        raise RuntimeError(
            "FSDP2 LoRA DP replica consistency validation failed"
        ) from pair_error

    authoritative_ranks = (0, 1, 2)
    is_authoritative = global_rank in authoritative_ranks
    authority_error: Exception | None = None
    if is_authoritative:
        try:
            actual_authoritative_ranks = tuple(
                torch.distributed.get_process_group_ranks(authoritative_shard_group)
            )
            if actual_authoritative_ranks != authoritative_ranks:
                raise ValueError(
                    "authoritative shard process group must contain global ranks "
                    f"{authoritative_ranks}, got {actual_authoritative_ranks}"
                )
        except Exception as exc:
            authority_error = exc
    if not _world_consensus(authority_error is None):
        raise RuntimeError(
            "invalid authoritative FSDP2 shard group"
        ) from authority_error

    gathered: list[Any] | None = None
    if is_authoritative:
        if global_rank == dst_global_rank:
            gathered = [None] * len(authoritative_ranks)
        torch.distributed.gather_object(
            local_shards,
            object_gather_list=gathered,
            dst=dst_global_rank,
            group=authoritative_shard_group,
        )

    result: OrderedDict[str, torch.Tensor] | None = None
    result_error: Exception | None = None
    if global_rank == dst_global_rank:
        try:
            assert gathered is not None
            result = consolidate_lora_shards(
                gathered,
                expected_schema=expected_schema,
            )
        except Exception as exc:
            result_error = exc
    status: list[Any] = [
        (
            (result_error is None, None if result_error is None else str(result_error))
            if global_rank == dst_global_rank
            else None
        )
    ]
    torch.distributed.broadcast_object_list(status, src=dst_global_rank)
    if not status[0][0]:
        raise RuntimeError(f"rank-0 LoRA shard consolidation failed: {status[0][1]}")
    return result if global_rank == dst_global_rank else None


def gather_lora_state_dict(lora_model):
    """Legacy full-state adapter gather; never use this in the Stage-1 path."""

    with FSDP.state_dict_type(
        lora_model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
    ):
        full = lora_model.state_dict()
    return get_canonical_lora_state_dict(lora_model, state_dict=full)


def load_lora_checkpoint(
    lora_model,
    lora_state_dict,
    model_name,
    is_main_process=True,
):
    """Strictly load a canonical in-memory LoRA checkpoint."""

    if is_main_process:
        print(
            f"Loading LoRA {model_name} weights: {len(lora_state_dict)} keys in checkpoint"
        )
    result = strict_load_lora_state_dict(lora_model, lora_state_dict)
    if is_main_process:
        print(f"LoRA {model_name} weights loaded successfully")
    return result
