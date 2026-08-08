from __future__ import annotations

from dataclasses import replace
import subprocess
import sys

import pytest
import torch
from torch import nn

from model.stage2_dmd import Stage2DMD, Stage2DiTRole
from utils.stage2_config import Stage2AdapterSpec
from utils.stage1_io import canonical_json_sha256, sha256_file
from utils.stage2_roles import (
    STAGE2_TARGET_SUFFIXES,
    audit_stage2_frozen_real_score,
    audit_stage2_target_schema,
    configure_stage2_role_lora,
    expected_stage2_target_names,
)
from utils.stage2_role_init import strict_load_stage2_role_base


def _file_identity(path):
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _verified_teacher_asset(checkpoint, payload, selector):
    state = payload.get(selector)
    state_keys = sorted(state) if isinstance(state, dict) else []
    metadata = {key: value for key, value in payload.items() if key != selector}
    return {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_format": "longlive_wrapper_pt",
        "state_dict_selector": selector,
        "checkpoint_files": [
            {
                "name": checkpoint.name,
                "path": str(checkpoint),
                "size": checkpoint.stat().st_size,
                "sha256": sha256_file(checkpoint),
                "identity": _file_identity(checkpoint),
            }
        ],
        "payload_contract": {
            "top_level_keys": sorted(payload),
            "state_dict_selector": selector,
            "state_tensor_count": len(state_keys),
            "state_dict_keys_sha256": canonical_json_sha256(state_keys),
            "load_target": "role_wrapper",
            "metadata": metadata,
        },
    }


class _SelfAttention(nn.Module):
    def __init__(self, width: int, *, device: str | None = None):
        super().__init__()
        self.q = nn.Linear(width, width, bias=False, device=device)
        self.k = nn.Linear(width, width, bias=False, device=device)
        self.v = nn.Linear(width, width, bias=False, device=device)
        self.o = nn.Linear(width, width, bias=False, device=device)


class _Block(nn.Module):
    def __init__(self, width: int, hidden: int, *, device: str | None = None):
        super().__init__()
        self.self_attn = _SelfAttention(width, device=device)
        self.cross_attn = nn.ModuleDict(
            {"q": nn.Linear(width, width, bias=False, device=device)}
        )
        self.ffn = nn.Sequential(
            nn.Linear(width, hidden, bias=False, device=device),
            nn.GELU(),
            nn.Linear(hidden, width, bias=False, device=device),
        )


class _Transformer(nn.Module):
    def __init__(
        self,
        *,
        blocks: int = 30,
        width: int = 4,
        hidden: int = 8,
        device: str | None = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_Block(width, hidden, device=device) for _ in range(blocks)]
        )
        self.head = nn.Linear(width, width, bias=False, device=device)


def _adapter(role: str, *, rank: int, width: int, hidden: int) -> Stage2AdapterSpec:
    per_block = rank * (8 * width + 2 * (width + hidden))
    return Stage2AdapterSpec(
        role=role,
        rank=rank,
        alpha=rank,
        dropout=0.0,
        bias="none",
        modules_to_save=(),
        target_patterns=(
            r"^blocks\.[0-9]+\.self_attn\.(q|k|v|o)$",
            r"^blocks\.[0-9]+\.ffn\.(0|2)$",
        ),
        expected_target_modules=180,
        expected_trainable_parameters=30 * per_block,
        expected_adapter_tensors=360,
    )


def test_release_target_schema_uses_exact_180_full_names_and_parameter_formula():
    transformer = _Transformer(width=3072, hidden=14336, device="meta")
    generator = _adapter("generator", rank=32, width=3072, hidden=14336)
    fake = _adapter("fake_score", rank=64, width=3072, hidden=14336)

    generator_audit = audit_stage2_target_schema(transformer, generator)
    fake_audit = audit_stage2_target_schema(transformer, fake)

    expected_names = expected_stage2_target_names()
    assert len(expected_names) == 30 * len(STAGE2_TARGET_SUFFIXES) == 180
    assert generator_audit.target_module_names == expected_names
    assert fake_audit.target_module_names == expected_names
    assert generator_audit.adapter_tensor_count == 360
    assert fake_audit.adapter_tensor_count == 360
    assert generator_audit.trainable_parameter_count == 57_016_320
    assert fake_audit.trainable_parameter_count == 114_032_640
    assert all("cross_attn" not in name for name in expected_names)
    assert all("head" not in name for name in expected_names)


@pytest.mark.parametrize(
    ("transformer", "match"),
    [
        (_Transformer(blocks=29, device="meta"), "exact target allowlist"),
        (_Transformer(blocks=31, device="meta"), "exact target allowlist"),
        (_Transformer(width=5, hidden=8, device="meta"), "parameter count"),
    ],
)
def test_target_schema_rejects_block_or_dimension_drift(transformer, match):
    spec = _adapter("generator", rank=2, width=4, hidden=8)
    with pytest.raises(ValueError, match=match):
        audit_stage2_target_schema(transformer, spec)


def test_materialized_peft_adapters_are_exact_fp32_zero_b_and_role_isolated():
    generator_base = _Transformer()
    fake_base = _Transformer()
    real_base = _Transformer()
    generator_base.requires_grad_(False)
    fake_base.requires_grad_(False)
    real_base.requires_grad_(False)

    generator_model, generator_audit = configure_stage2_role_lora(
        generator_base,
        role="generator",
        spec=_adapter("generator", rank=2, width=4, hidden=8),
        seed=101,
    )
    fake_model, fake_audit = configure_stage2_role_lora(
        fake_base,
        role="fake_score",
        spec=_adapter("fake_score", rank=2, width=4, hidden=8),
        seed=202,
    )
    real_audit = audit_stage2_frozen_real_score(real_base)

    assert generator_audit.adapter_tensor_count == 360
    assert fake_audit.adapter_tensor_count == 360
    assert real_audit["trainable_parameter_count"] == 0
    assert all(
        parameter.dtype == torch.float32
        for parameter in generator_model.parameters()
        if parameter.requires_grad
    )
    assert all(
        torch.count_nonzero(parameter).item() == 0
        for name, parameter in generator_model.named_parameters()
        if ".lora_B." in name
    )
    generator_ids = {
        id(parameter)
        for parameter in generator_model.parameters()
        if parameter.requires_grad
    }
    fake_ids = {
        id(parameter)
        for parameter in fake_model.parameters()
        if parameter.requires_grad
    }
    assert generator_ids.isdisjoint(fake_ids)

    roles = Stage2DMD(
        generator=Stage2DiTRole(generator_model, role="generator", is_causal=True),
        real_score=Stage2DiTRole(real_base, role="real_score", is_causal=False),
        fake_score=Stage2DiTRole(fake_model, role="fake_score", is_causal=False),
    )
    roles.audit_independent_role_storage()


def test_role_and_adapter_spec_must_match():
    with pytest.raises(ValueError, match="role mismatch"):
        configure_stage2_role_lora(
            _Transformer().requires_grad_(False),
            role="generator",
            spec=_adapter("fake_score", rank=2, width=4, hidden=8),
            seed=1,
        )


def test_manifest_selected_wrapper_state_load_is_strict_and_storage_independent(
    tmp_path,
):
    source = Stage2DiTRole(
        _Transformer().to(dtype=torch.bfloat16),
        role="real_score",
        is_causal=False,
    ).requires_grad_(False)
    checkpoint = tmp_path / "teacher.pt"
    payload = {"real_score": source.state_dict()}
    torch.save(payload, checkpoint)
    asset = _verified_teacher_asset(checkpoint, payload, "real_score")
    left = Stage2DiTRole(
        _Transformer().to(dtype=torch.bfloat16),
        role="real_score",
        is_causal=False,
    )
    right = Stage2DiTRole(
        _Transformer().to(dtype=torch.bfloat16),
        role="fake_score",
        is_causal=False,
    )
    left_summary = strict_load_stage2_role_base(left, asset=asset)
    right_summary = strict_load_stage2_role_base(right, asset=asset)
    assert left_summary == right_summary
    for left_parameter, right_parameter in zip(left.parameters(), right.parameters()):
        assert torch.equal(left_parameter, right_parameter)
        assert (
            left_parameter.untyped_storage().data_ptr()
            != right_parameter.untyped_storage().data_ptr()
        )


def test_generator_v2_payload_load_is_exact_and_strict(tmp_path):
    source = Stage2DiTRole(
        _Transformer().to(dtype=torch.bfloat16),
        role="generator",
        is_causal=True,
    ).requires_grad_(False)
    base_sha256 = "a" * 64
    payload = {
        "generator": source.state_dict(),
        "checkpoint_format": "longlive_stage1_causal_ema_merged",
        "checkpoint_version": 1,
        "model_name": "Wan2.2-TI2V-5B",
        "source_training_step": 3750,
        "source_adapter": "adapter_ema.safetensors",
        "dtype": "bfloat16",
        "source_base_sha256": base_sha256,
    }
    checkpoint = tmp_path / "generator.pt"
    torch.save(payload, checkpoint)
    asset = {
        **_verified_teacher_asset(checkpoint, payload, "generator"),
        "checkpoint_format": "longlive_stage1_causal_ema_merged",
        "source": {"base": {"sha256": base_sha256}},
    }
    target = Stage2DiTRole(
        _Transformer().to(dtype=torch.bfloat16),
        role="generator",
        is_causal=True,
    )
    summary = strict_load_stage2_role_base(target, asset=asset)
    assert summary["strict_reload_succeeded"] is True

    payload["global_step"] = 3750
    torch.save(payload, checkpoint)
    asset = {
        **_verified_teacher_asset(checkpoint, payload, "generator"),
        "checkpoint_format": "longlive_stage1_causal_ema_merged",
        "source": {"base": {"sha256": base_sha256}},
    }
    with pytest.raises(ValueError, match="top-level schema"):
        strict_load_stage2_role_base(target, asset=asset)


@pytest.mark.parametrize(
    "failure",
    [
        "selector",
        "dtype",
        "lora",
        "nonfinite",
        "optimizer_state_dict",
        "global_step",
        "rng_states",
        "model_ema",
    ],
)
def test_strict_base_loader_rejects_unmanifested_or_unsafe_payload(tmp_path, failure):
    wrapper = Stage2DiTRole(
        _Transformer().to(dtype=torch.bfloat16),
        role="real_score",
        is_causal=False,
    )
    state = dict(wrapper.state_dict())
    payload = {"real_score": state}
    selector = "real_score"
    match = ""
    if failure == "selector":
        selector = "model"
        match = "selected state"
    elif failure == "dtype":
        key = next(iter(state))
        state[key] = state[key].float()
        match = "BF16"
    elif failure == "lora":
        state["model.blocks.0.self_attn.q.lora_A.weight"] = torch.zeros(
            1, dtype=torch.bfloat16
        )
        match = "LoRA keys"
    elif failure == "nonfinite":
        key = next(iter(state))
        state[key] = state[key].clone()
        state[key].view(-1)[0] = torch.nan
        match = "non-finite"
    else:
        payload[failure] = 1 if failure == "global_step" else {}
        match = "training state"
    checkpoint = tmp_path / f"{failure}.pt"
    torch.save(payload, checkpoint)
    with pytest.raises((KeyError, TypeError, ValueError), match=match):
        strict_load_stage2_role_base(
            wrapper,
            asset=_verified_teacher_asset(checkpoint, payload, selector),
        )
    with pytest.raises(ValueError, match="real_score never receives LoRA"):
        configure_stage2_role_lora(
            _Transformer().requires_grad_(False),
            role="real_score",
            spec=replace(
                _adapter("generator", rank=2, width=4, hidden=8), role="real_score"
            ),
            seed=1,
        )


def test_post_load_content_sha_is_recomputed_and_architecture_identity_is_bound(
    tmp_path,
):
    source = Stage2DiTRole(
        _Transformer().to(dtype=torch.bfloat16),
        role="real_score",
        is_causal=False,
    ).requires_grad_(False)
    payload = {"real_score": source.state_dict()}
    checkpoint = tmp_path / "teacher.pt"
    torch.save(payload, checkpoint)
    asset = _verified_teacher_asset(checkpoint, payload, "real_score")
    asset["checkpoint_files"][0]["sha256"] = "0" * 64
    target = Stage2DiTRole(
        _Transformer().to(dtype=torch.bfloat16),
        role="real_score",
        is_causal=False,
    )
    with pytest.raises(RuntimeError, match="content SHA changed after load"):
        strict_load_stage2_role_base(
            target,
            asset=asset,
            verify_content_hash=True,
        )

    architecture = tmp_path / "config.json"
    architecture.write_text("{}", encoding="utf-8")
    asset = _verified_teacher_asset(checkpoint, payload, "real_score")
    asset["architecture_file"] = {
        "name": architecture.name,
        "path": str(architecture),
        "size": architecture.stat().st_size,
        "sha256": sha256_file(architecture),
        "identity": _file_identity(architecture),
    }
    architecture.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity changed"):
        strict_load_stage2_role_base(target, asset=asset)


def test_importing_stage2_role_model_does_not_import_legacy_trainer_t5_or_vae():
    program = """
import sys
from model.stage2_dmd import Stage2DMD
assert Stage2DMD.__name__ == 'Stage2DMD'
for name in (
    'trainer.distillation',
    'wan_5b.modules.t5',
    'wan_5b.modules.vae2_1',
    'wan_5b.modules.vae2_2',
):
    assert name not in sys.modules, name
print('stage2-role-import-ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", program],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "stage2-role-import-ok"
