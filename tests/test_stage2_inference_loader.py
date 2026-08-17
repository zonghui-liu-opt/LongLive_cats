from __future__ import annotations

import json
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from model.stage2_dmd import Stage2DiTRole
from utils.stage1_io import canonical_json_sha256, sha256_file
from utils.stage2_config import resolve_stage2_config
from utils.stage2_inference_loader import (
    Stage2InferenceLoaderOps,
    load_stage2_ema_generator_for_inference,
)

_ROOT = Path(__file__).resolve().parents[1]
_SHA_A = "a" * 64
_SHA_B = "b" * 64
_EMA_KEY = "base_model.model.tiny.lora_A.weight"


class _TinyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(2, 2, dtype=torch.bfloat16), requires_grad=False
        )
        self.is_gradient_checkpointing = False


class _TinyLora(torch.nn.Module):
    def __init__(
        self,
        base: _TinyBackbone,
        *,
        merge_calls: list[bool],
        leave_adapter: bool,
    ) -> None:
        super().__init__()
        self.base = base
        self.lora_A = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self._merge_calls = merge_calls
        self._leave_adapter = leave_adapter

    def merge_and_unload(self, *, safe_merge: bool) -> torch.nn.Module:
        self._merge_calls.append(safe_merge)
        if self._leave_adapter:
            return self
        with torch.no_grad():
            self.base.weight.add_(self.lora_A.item())
        return self.base


def _resolved_training_config():
    return resolve_stage2_config(
        OmegaConf.load(_ROOT / "configs" / "train_i2v_stage2_600cats.yaml")
    )


def _checkpoint_fixture(tmp_path: Path) -> tuple[Path, Any, dict[str, Any]]:
    resolved = _resolved_training_config()
    checkpoint = tmp_path / "checkpoint_stage2_g000040"
    checkpoint.mkdir()
    (checkpoint / "_SUCCESS").touch()
    saved = json.loads(json.dumps(resolved.to_dict(), allow_nan=False))
    (checkpoint / "resolved_config.json").write_text(
        json.dumps(saved, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    architecture = tmp_path / "architecture"
    architecture.mkdir()
    (architecture / "config.json").write_text("{}\n", encoding="utf-8")
    return checkpoint, resolved, saved


def _generator_asset() -> dict[str, Any]:
    return {
        "manifest_path": "/fixture/stage1_step3075.manifest.json",
        "checkpoint_path": "/fixture/stage1_step3075_ema_merged.pt",
        "checkpoint_sha256": _SHA_A,
        "source_step": 3075,
    }


def _file_identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _architecture_file(path: Path) -> dict[str, Any]:
    return {
        "name": "config.json",
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "identity": _file_identity(path),
    }


def _manifest(resolved: Any, *, completed_g: int = 40) -> dict[str, Any]:
    return {
        "manifest_sha256": _SHA_A,
        "completed_generator_updates": completed_g,
        "config": {
            "contract_hash": resolved.contract_hash(),
            "launch_hash": resolved.launch_hash(),
        },
        "files": [
            {
                "name": "generator_ema.safetensors",
                "size": 4,
                "sha256": _SHA_B,
            }
        ],
    }


def _ops(
    *,
    checkpoint: Path,
    resolved: Any,
    saved: dict[str, Any],
    calls: dict[str, Any],
    returned_asset: dict[str, Any] | None = None,
    returned_resolved_config: dict[str, Any] | None = None,
    completed_g: int = 40,
    leave_adapter: bool = False,
    lora_error: Exception | None = None,
    recorded_architecture_file: dict[str, Any] | None = None,
    include_architecture_file: bool = True,
) -> Stage2InferenceLoaderOps:
    asset = _generator_asset()
    if recorded_architecture_file is None:
        recorded_architecture_file = _architecture_file(
            checkpoint.parent / "architecture" / "config.json"
        )
    recorded_asset = deepcopy(asset)
    if include_architecture_file:
        recorded_asset["architecture_file"] = deepcopy(recorded_architecture_file)

    def read_checkpoint(directory: Path, **kwargs: Any) -> Any:
        calls["checkpoint_reader"] = {"directory": directory, **kwargs}
        return SimpleNamespace(
            directory=checkpoint,
            resolved_config=(
                returned_resolved_config
                if returned_resolved_config is not None
                else saved
            ),
            manifest=_manifest(resolved, completed_g=completed_g),
            provenance={"assets": {"generator": recorded_asset}},
            generator_ema=OrderedDict(
                [(_EMA_KEY, torch.tensor([2.0], dtype=torch.float32))]
            ),
        )

    def validate_manifest(
        manifest_path: str,
        *,
        expected_checkpoint_path: str,
        expected_step: int,
    ) -> dict[str, Any]:
        calls["manifest"] = {
            "manifest_path": manifest_path,
            "expected_checkpoint_path": expected_checkpoint_path,
            "expected_step": expected_step,
        }
        return returned_asset if returned_asset is not None else deepcopy(asset)

    def build_generator(_resolved: Any, architecture_root: Path) -> Stage2DiTRole:
        calls["build"] = architecture_root
        return Stage2DiTRole(_TinyBackbone(), role="generator", is_causal=True)

    def load_base(wrapper: Any, **kwargs: Any) -> dict[str, Any]:
        calls["base"] = {"wrapper": wrapper, **kwargs}
        return {
            "tensor_count": 1,
            "parameter_count": 4,
            "strict_reload_succeeded": True,
        }

    def validate_contract(wrapper: Any, actual_resolved: Any) -> None:
        assert wrapper.role == "generator"
        assert wrapper.is_causal is True
        assert actual_resolved is resolved or (
            actual_resolved.to_dict() == resolved.to_dict()
        )
        calls["contract_validations"] = calls.get("contract_validations", 0) + 1

    def configure_lora(
        base: _TinyBackbone,
        **kwargs: Any,
    ) -> tuple[_TinyLora, object]:
        calls["configure"] = kwargs
        return (
            _TinyLora(
                base,
                merge_calls=calls.setdefault("merge", []),
                leave_adapter=leave_adapter,
            ),
            object(),
        )

    def load_lora(module: _TinyLora, state: Any, **kwargs: Any) -> None:
        calls["lora_loader"] = {"state": state, **kwargs}
        if lora_error is not None:
            raise lora_error
        assert list(state) == [_EMA_KEY]
        with torch.no_grad():
            module.lora_A.copy_(state[_EMA_KEY])

    return Stage2InferenceLoaderOps(
        read_generator_ema_checkpoint=read_checkpoint,
        validate_generator_manifest=validate_manifest,
        build_generator=build_generator,
        load_generator_base=load_base,
        validate_generator_contract=validate_contract,
        configure_generator_lora=configure_lora,
        load_generator_lora=load_lora,
    )


def test_loads_only_generator_ema_and_safe_merges_to_frozen_bf16(
    tmp_path: Path,
) -> None:
    checkpoint, resolved, saved = _checkpoint_fixture(tmp_path)
    calls: dict[str, Any] = {}
    loaded = load_stage2_ema_generator_for_inference(
        checkpoint,
        architecture_root=tmp_path / "architecture",
        device="cpu",
        ops=_ops(
            checkpoint=checkpoint,
            resolved=resolved,
            saved=saved,
            calls=calls,
        ),
    )

    reader = calls["checkpoint_reader"]
    assert reader == {
        "directory": checkpoint.resolve(),
        "expected_contract_hash": resolved.contract_hash(),
        "expected_launch_hash": resolved.launch_hash(),
        "expected_resolved_config": saved,
    }
    assert calls["manifest"]["expected_step"] == 3075
    assert calls["manifest"]["manifest_path"].endswith("stage1_step3075.manifest.json")
    assert calls["base"]["verify_content_hash"] is True
    runtime_architecture = calls["base"]["asset"]["architecture_file"]
    architecture_path = tmp_path / "architecture" / "config.json"
    assert runtime_architecture == _architecture_file(architecture_path)
    assert calls["configure"]["role"] == "generator"
    assert calls["configure"]["spec"].rank == 32
    assert calls["lora_loader"]["expected_dtype"] is torch.float32
    assert calls["lora_loader"]["require_finite"] is True
    assert calls["lora_loader"]["verify_tensors"] is True
    assert calls["merge"] == [True]
    assert calls["contract_validations"] == 2
    assert loaded.generator.training is False
    assert all(not value.requires_grad for value in loaded.generator.parameters())
    assert {
        value.dtype
        for value in loaded.generator.parameters()
        if value.is_floating_point()
    } == {torch.bfloat16}
    assert loaded.checkpoint_identity() == {
        "directory": str(checkpoint.resolve()),
        "manifest_sha256": _SHA_A,
        "completed_generator_updates": 40,
        "contract_hash": resolved.contract_hash(),
        "generator_ema_sha256": _SHA_B,
    }


def test_rank0_trusted_generator_attestation_skips_duplicate_base_hash_scan(
    tmp_path: Path,
) -> None:
    checkpoint, resolved, saved = _checkpoint_fixture(tmp_path)
    architecture_file = _architecture_file(tmp_path / "architecture" / "config.json")
    trusted_asset = {
        **_generator_asset(),
        "architecture_file": architecture_file,
    }
    calls: dict[str, Any] = {}

    loaded = load_stage2_ema_generator_for_inference(
        checkpoint,
        architecture_root=tmp_path / "architecture",
        device="cpu",
        trusted_generator_asset=trusted_asset,
        expected_recorded_generator_asset_sha256=canonical_json_sha256(trusted_asset),
        ops=_ops(
            checkpoint=checkpoint,
            resolved=resolved,
            saved=saved,
            calls=calls,
            recorded_architecture_file=architecture_file,
        ),
    )

    assert "manifest" not in calls
    assert calls["base"]["verify_content_hash"] is False
    assert loaded.generator_asset == trusted_asset


def test_trusted_generator_attestation_rejects_checkpoint_provenance_drift(
    tmp_path: Path,
) -> None:
    checkpoint, resolved, saved = _checkpoint_fixture(tmp_path)
    architecture_file = _architecture_file(tmp_path / "architecture" / "config.json")
    trusted_asset = {
        **_generator_asset(),
        "architecture_file": architecture_file,
    }
    calls: dict[str, Any] = {}

    with pytest.raises(RuntimeError, match="recorded Generator asset differs"):
        load_stage2_ema_generator_for_inference(
            checkpoint,
            architecture_root=tmp_path / "architecture",
            device="cpu",
            trusted_generator_asset=trusted_asset,
            expected_recorded_generator_asset_sha256="f" * 64,
            ops=_ops(
                checkpoint=checkpoint,
                resolved=resolved,
                saved=saved,
                calls=calls,
                recorded_architecture_file=architecture_file,
            ),
        )

    assert "build" not in calls


def test_rejects_tampered_saved_derived_config_before_checkpoint_reader(
    tmp_path: Path,
) -> None:
    checkpoint, resolved, saved = _checkpoint_fixture(tmp_path)
    saved["derived"]["generator_stage1_step"] = 1
    (checkpoint / "resolved_config.json").write_text(
        json.dumps(saved), encoding="utf-8"
    )
    calls: dict[str, Any] = {}

    with pytest.raises(RuntimeError, match="cannot be reproduced"):
        load_stage2_ema_generator_for_inference(
            checkpoint,
            architecture_root=tmp_path / "architecture",
            device="cpu",
            ops=_ops(
                checkpoint=checkpoint,
                resolved=resolved,
                saved=saved,
                calls=calls,
            ),
        )
    assert "checkpoint_reader" not in calls


def test_rejects_duplicate_resolved_config_json_keys(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint_stage2_g000040"
    checkpoint.mkdir()
    (checkpoint / "_SUCCESS").touch()
    (checkpoint / "resolved_config.json").write_text(
        '{"config":{},"config":{},"derived":{}}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_stage2_ema_generator_for_inference(
            checkpoint,
            architecture_root=tmp_path / "architecture",
            device="cpu",
        )


def test_rejects_checkpoint_reader_resolved_config_disagreement(
    tmp_path: Path,
) -> None:
    checkpoint, resolved, saved = _checkpoint_fixture(tmp_path)
    calls: dict[str, Any] = {}
    wrong = deepcopy(saved)
    wrong["derived"]["generator_stage1_step"] = 3074

    with pytest.raises(RuntimeError, match="different resolved config"):
        load_stage2_ema_generator_for_inference(
            checkpoint,
            architecture_root=tmp_path / "architecture",
            device="cpu",
            ops=_ops(
                checkpoint=checkpoint,
                resolved=resolved,
                saved=saved,
                calls=calls,
                returned_resolved_config=wrong,
            ),
        )
    assert "build" not in calls


def test_rejects_architecture_content_replaced_after_checkpoint_provenance(
    tmp_path: Path,
) -> None:
    checkpoint, resolved, saved = _checkpoint_fixture(tmp_path)
    architecture_path = tmp_path / "architecture" / "config.json"
    recorded_architecture = _architecture_file(architecture_path)
    # Preserve the file size while changing the bytes at the same canonical path.
    architecture_path.write_text("[]\n", encoding="utf-8")
    calls: dict[str, Any] = {}

    with pytest.raises(RuntimeError, match="architecture SHA256 differs"):
        load_stage2_ema_generator_for_inference(
            checkpoint,
            architecture_root=tmp_path / "architecture",
            device="cpu",
            ops=_ops(
                checkpoint=checkpoint,
                resolved=resolved,
                saved=saved,
                calls=calls,
                recorded_architecture_file=recorded_architecture,
            ),
        )
    assert "build" not in calls


def test_requires_production_generator_architecture_provenance(tmp_path: Path) -> None:
    checkpoint, resolved, saved = _checkpoint_fixture(tmp_path)
    calls: dict[str, Any] = {}

    with pytest.raises(ValueError, match="architecture_file schema mismatch"):
        load_stage2_ema_generator_for_inference(
            checkpoint,
            architecture_root=tmp_path / "architecture",
            device="cpu",
            ops=_ops(
                checkpoint=checkpoint,
                resolved=resolved,
                saved=saved,
                calls=calls,
                include_architecture_file=False,
            ),
        )
    assert "build" not in calls


@pytest.mark.parametrize(
    ("returned_asset", "message"),
    [
        ({**_generator_asset(), "source_step": 3074}, "not Stage-1 step3075"),
        (
            {**_generator_asset(), "checkpoint_sha256": _SHA_B},
            "provenance differs",
        ),
    ],
)
def test_rejects_stage1_lineage_or_live_manifest_drift(
    tmp_path: Path,
    returned_asset: dict[str, Any],
    message: str,
) -> None:
    checkpoint, resolved, saved = _checkpoint_fixture(tmp_path)
    calls: dict[str, Any] = {}

    with pytest.raises(RuntimeError, match=message):
        load_stage2_ema_generator_for_inference(
            checkpoint,
            architecture_root=tmp_path / "architecture",
            device="cpu",
            ops=_ops(
                checkpoint=checkpoint,
                resolved=resolved,
                saved=saved,
                calls=calls,
                returned_asset=returned_asset,
            ),
        )
    assert "build" not in calls


def test_strict_ema_load_failure_never_merges(tmp_path: Path) -> None:
    checkpoint, resolved, saved = _checkpoint_fixture(tmp_path)
    calls: dict[str, Any] = {}

    with pytest.raises(ValueError, match="EMA key mismatch"):
        load_stage2_ema_generator_for_inference(
            checkpoint,
            architecture_root=tmp_path / "architecture",
            device="cpu",
            ops=_ops(
                checkpoint=checkpoint,
                resolved=resolved,
                saved=saved,
                calls=calls,
                lora_error=ValueError("EMA key mismatch"),
            ),
        )
    assert calls["merge"] == []


def test_rejects_adapter_tensors_left_by_merge(tmp_path: Path) -> None:
    checkpoint, resolved, saved = _checkpoint_fixture(tmp_path)
    calls: dict[str, Any] = {}

    with pytest.raises(RuntimeError, match="left adapter tensors"):
        load_stage2_ema_generator_for_inference(
            checkpoint,
            architecture_root=tmp_path / "architecture",
            device="cpu",
            ops=_ops(
                checkpoint=checkpoint,
                resolved=resolved,
                saved=saved,
                calls=calls,
                leave_adapter=True,
            ),
        )
    assert calls["merge"] == [True]


def test_requires_initialized_ema_checkpoint_before_model_build(
    tmp_path: Path,
) -> None:
    checkpoint, resolved, saved = _checkpoint_fixture(tmp_path)
    calls: dict[str, Any] = {}

    with pytest.raises(RuntimeError, match="G>=40"):
        load_stage2_ema_generator_for_inference(
            checkpoint,
            architecture_root=tmp_path / "architecture",
            device="cpu",
            ops=_ops(
                checkpoint=checkpoint,
                resolved=resolved,
                saved=saved,
                calls=calls,
                completed_g=39,
            ),
        )
    assert "build" not in calls


def test_rejects_symlinked_architecture_root_before_model_build(tmp_path: Path) -> None:
    checkpoint, resolved, saved = _checkpoint_fixture(tmp_path)
    linked = tmp_path / "linked-architecture"
    linked.symlink_to(tmp_path / "architecture", target_is_directory=True)
    calls: dict[str, Any] = {}

    with pytest.raises(FileNotFoundError, match="linked-architecture"):
        load_stage2_ema_generator_for_inference(
            checkpoint,
            architecture_root=linked,
            device="cpu",
            ops=_ops(
                checkpoint=checkpoint,
                resolved=resolved,
                saved=saved,
                calls=calls,
            ),
        )
    assert "build" not in calls


def test_loader_source_does_not_import_legacy_inference_pipeline() -> None:
    source = (_ROOT / "utils" / "stage2_inference_loader.py").read_text(
        encoding="utf-8"
    )
    forbidden_module = "pipeline." + "causal_diffusion_inference"
    assert forbidden_module not in source
