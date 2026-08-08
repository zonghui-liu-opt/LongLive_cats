"""Strict Stage-2 role and LoRA construction primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any

import torch
from torch import nn

from utils.lora_utils import (
    audit_lora_model,
    configure_lora_for_model,
    get_canonical_lora_state_dict,
    resolve_lora_target_modules,
)
from utils.stage1_io import canonical_json_sha256
from utils.stage2_config import Stage2AdapterSpec

STAGE2_NUM_BLOCKS = 30
STAGE2_TARGET_SUFFIXES = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "ffn.0",
    "ffn.2",
)
_ADAPTER_ROLES = {"generator", "fake_score"}


def expected_stage2_target_names(
    *, num_blocks: int = STAGE2_NUM_BLOCKS
) -> tuple[str, ...]:
    return tuple(
        sorted(
            f"blocks.{block_index}.{suffix}"
            for block_index in range(int(num_blocks))
            for suffix in STAGE2_TARGET_SUFFIXES
        )
    )


@dataclass(frozen=True)
class Stage2TargetAudit:
    role: str
    rank: int
    alpha: int
    target_module_names: tuple[str, ...]
    target_module_shapes: tuple[tuple[str, int, int], ...]
    target_module_count: int
    adapter_tensor_count: int
    trainable_parameter_count: int
    target_schema_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_adapter_role(role: str, spec: Stage2AdapterSpec) -> None:
    if role == "real_score":
        raise ValueError("Stage-2 real_score never receives LoRA")
    if role not in _ADAPTER_ROLES:
        raise ValueError(f"unknown Stage-2 adapter role: {role!r}")
    if spec.role != role:
        raise ValueError(
            f"Stage-2 adapter role mismatch: requested={role!r}, spec={spec.role!r}"
        )


def audit_stage2_target_schema(
    transformer: nn.Module,
    spec: Stage2AdapterSpec,
) -> Stage2TargetAudit:
    """Resolve and shape-audit the complete 180-module baseline allowlist."""

    _validate_adapter_role(spec.role, spec)
    selected = resolve_lora_target_modules(
        transformer,
        spec.role,
        {
            "target_patterns": spec.target_patterns,
            "type": "lora",
            "rank": spec.rank,
        },
    )
    expected = expected_stage2_target_names()
    if selected != expected:
        raise ValueError(
            "Stage-2 exact target allowlist drifted: "
            f"missing={sorted(set(expected) - set(selected))}, "
            f"extra={sorted(set(selected) - set(expected))}"
        )

    named_modules = dict(transformer.named_modules())
    shapes: list[tuple[str, int, int]] = []
    trainable_parameters = 0
    for name in selected:
        module = named_modules[name]
        if not isinstance(module, nn.Linear):
            raise TypeError(f"Stage-2 target is not Linear: {name}")
        input_features = int(module.in_features)
        output_features = int(module.out_features)
        shapes.append((name, input_features, output_features))
        trainable_parameters += int(spec.rank) * (input_features + output_features)

    target_count = len(selected)
    adapter_tensor_count = 2 * target_count
    exact = (
        ("target modules", spec.expected_target_modules, target_count),
        ("adapter tensors", spec.expected_adapter_tensors, adapter_tensor_count),
        (
            "trainable parameter count",
            spec.expected_trainable_parameters,
            trainable_parameters,
        ),
    )
    for label, expected_value, actual_value in exact:
        if int(expected_value) != int(actual_value):
            raise ValueError(
                f"Stage-2 {spec.role} {label} mismatch: "
                f"expected={expected_value}, actual={actual_value}"
            )
    schema_payload = {
        "role": spec.role,
        "rank": int(spec.rank),
        "alpha": int(spec.alpha),
        "target_patterns": list(spec.target_patterns),
        "target_module_shapes": [list(item) for item in shapes],
    }
    return Stage2TargetAudit(
        role=spec.role,
        rank=int(spec.rank),
        alpha=int(spec.alpha),
        target_module_names=selected,
        target_module_shapes=tuple(shapes),
        target_module_count=target_count,
        adapter_tensor_count=adapter_tensor_count,
        trainable_parameter_count=trainable_parameters,
        target_schema_sha256=canonical_json_sha256(schema_payload),
    )


def _adapter_config(spec: Stage2AdapterSpec) -> dict[str, Any]:
    return {
        "type": "lora",
        "rank": int(spec.rank),
        "alpha": int(spec.alpha),
        "dropout": float(spec.dropout),
        "bias": spec.bias,
        "modules_to_save": list(spec.modules_to_save),
        "target_patterns": list(spec.target_patterns),
        "expected_target_modules": int(spec.expected_target_modules),
        "expected_trainable_parameters": int(spec.expected_trainable_parameters),
        "expected_adapter_tensors": int(spec.expected_adapter_tensors),
    }


def _canonical_target_name(key: str) -> str:
    marker = "blocks."
    index = key.find(marker)
    if index < 0:
        raise ValueError(f"canonical LoRA key does not contain a block target: {key}")
    value = key[index:]
    for suffix in (".lora_A.weight", ".lora_B.weight"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    raise ValueError(f"canonical LoRA key is not an A/B tensor: {key}")


def configure_stage2_role_lora(
    transformer: nn.Module,
    *,
    role: str,
    spec: Stage2AdapterSpec,
    seed: int,
    is_main_process: bool = False,
) -> tuple[nn.Module, Stage2TargetAudit]:
    """Attach one fresh role-specific adapter without perturbing global RNG."""

    _validate_adapter_role(role, spec)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Stage-2 adapter seed must be a non-negative integer")
    target_audit = audit_stage2_target_schema(transformer, spec)
    transformer.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        # The roles are materialized on CPU.  Seed only the CPU default
        # generator: torch.manual_seed() also seeds every visible CUDA device,
        # which would leak G/F initialization order into later rollout RNG.
        torch.default_generator.manual_seed(seed)
        lora_model = configure_lora_for_model(
            transformer,
            model_name=role,
            lora_config=_adapter_config(spec),
            is_main_process=is_main_process,
        )
    lora_audit = audit_lora_model(
        lora_model,
        target_module_names=target_audit.target_module_names,
        expected_target_modules=spec.expected_target_modules,
        expected_trainable_parameters=spec.expected_trainable_parameters,
        expected_adapter_tensors=spec.expected_adapter_tensors,
        expected_trainable_dtype=torch.float32,
        require_lora_only=True,
        check_b_zero=True,
    )
    canonical = get_canonical_lora_state_dict(lora_model)
    a_targets = {
        _canonical_target_name(key)
        for key in canonical
        if key.endswith(".lora_A.weight")
    }
    b_targets = {
        _canonical_target_name(key)
        for key in canonical
        if key.endswith(".lora_B.weight")
    }
    expected_targets = set(target_audit.target_module_names)
    if a_targets != expected_targets or b_targets != expected_targets:
        raise ValueError(
            "Stage-2 PEFT canonical target set differs from the exact allowlist"
        )

    modules = dict(transformer.named_modules())
    for key, tensor in canonical.items():
        target = _canonical_target_name(key)
        linear = modules[target]
        expected_shape = (
            (spec.rank, linear.in_features)
            if key.endswith(".lora_A.weight")
            else (linear.out_features, spec.rank)
        )
        if tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(
                f"Stage-2 LoRA tensor shape mismatch for {key}: "
                f"expected={expected_shape}, actual={tuple(tensor.shape)}"
            )
    if lora_audit["adapter_tensor_count"] != target_audit.adapter_tensor_count:
        raise AssertionError("Stage-2 LoRA audit tensor count drifted")
    return lora_model, target_audit


def audit_stage2_frozen_real_score(
    model: nn.Module,
    *,
    expected_base_dtype: torch.dtype | None = None,
) -> dict[str, Any]:
    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if trainable:
        raise ValueError(
            f"Stage-2 real_score has trainable parameters: {trainable[:8]}"
        )
    lora_names = [
        name
        for name, _ in model.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    ]
    if lora_names:
        raise ValueError("Stage-2 real_score must not contain LoRA parameters")
    if expected_base_dtype is not None:
        wrong = [
            (name, str(parameter.dtype))
            for name, parameter in model.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != expected_base_dtype
        ]
        if wrong:
            raise TypeError(f"Stage-2 real_score base dtype mismatch: {wrong[:8]}")
    return {
        "trainable_tensor_count": 0,
        "trainable_parameter_count": 0,
        "adapter_tensor_count": 0,
    }


def stage2_adapter_digest(lora_model: nn.Module) -> str:
    """Hash a fresh canonical adapter for cross-rank initialization consensus."""

    digest = hashlib.sha256()
    canonical = get_canonical_lora_state_dict(lora_model)
    for key, tensor in canonical.items():
        value = tensor.detach().to(device="cpu").contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
