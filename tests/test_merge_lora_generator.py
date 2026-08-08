from omegaconf import OmegaConf
import pytest
import torch

from scripts.merge_lora_generator import merge_stage1_ema_checkpoint
from utils.lora_utils import configure_lora_for_model, save_lora_safetensors_strict
from utils.stage1_checkpoint import (
    CHECKPOINT_SCHEMA,
    CHECKPOINT_SCHEMA_VERSION,
    write_checkpoint_manifest,
    write_success_marker,
)
from utils.stage1_io import atomic_write_json, canonical_json_sha256, sha256_file


class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(2, 2, bias=False)


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_Block()])


class _Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Tiny()


def _config():
    return OmegaConf.create(
        {
            "model_kwargs": {"model_name": "tiny"},
            "adapter": {
                "type": "lora",
                "rank": 1,
                "alpha": 1,
                "dropout": 0.0,
                "bias": "none",
                "modules_to_save": [],
                "target_patterns": [r"^blocks\.0\.self_attn\.q$"],
                "expected_target_modules": 1,
                "expected_trainable_parameters": 4,
                "expected_adapter_tensors": 2,
            },
        }
    )


def _make_checkpoint(tmp_path):
    base_model = _Wrapper()
    with torch.no_grad():
        base_model.model.blocks[0].self_attn.q.weight.zero_()
    base = tmp_path / "base.pt"
    torch.save({"generator": base_model.state_dict()}, base)

    directory = tmp_path / "checkpoint_model_000075"
    directory.mkdir()
    config = _config()
    OmegaConf.save(config, directory / "resolved_config.yaml")
    atomic_write_json(
        directory / "base_reference.json",
        {"base_sha256": sha256_file(base)},
    )
    peft_model = configure_lora_for_model(
        _Tiny(), "generator", config.adapter, is_main_process=False
    )
    for key, tensor in peft_model.state_dict().items():
        if ".lora_A." in key:
            tensor.data.copy_(torch.tensor([[1.0, 2.0]]))
        elif ".lora_B." in key:
            tensor.data.copy_(torch.tensor([[3.0], [4.0]]))
    save_lora_safetensors_strict(
        peft_model,
        directory / "adapter_ema.safetensors",
        metadata={
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": "ema",
            "completed_step": 75,
            "tensor_count": 2,
            "global_numel": 4,
            "dtype": "float32",
        },
    )
    # Raw intentionally differs; merge must never select it.
    for name, parameter in peft_model.named_parameters():
        if "lora_" in name:
            parameter.data.fill_(99.0)
    save_lora_safetensors_strict(
        peft_model,
        directory / "adapter_raw.safetensors",
        metadata={
            "schema": CHECKPOINT_SCHEMA,
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": "raw",
            "completed_step": 75,
            "tensor_count": 2,
            "global_numel": 4,
            "dtype": "float32",
        },
    )
    write_checkpoint_manifest(
        directory,
        completed_step=75,
        world_size=6,
        sequence_parallel_size=3,
        data_parallel_size=2,
        resumable=False,
    )
    write_success_marker(directory, resumable=False)
    return base, directory, config


def test_merge_uses_only_ema_and_matches_ba(tmp_path):
    base, directory, config = _make_checkpoint(tmp_path)
    output = tmp_path / "merged.pt"
    manifest = merge_stage1_ema_checkpoint(
        base_checkpoint=base,
        training_checkpoint=directory,
        output_path=output,
        config=config,
        wrapper_builder=_Wrapper,
    )
    checkpoint = torch.load(output, map_location="cpu", weights_only=False)
    weight = checkpoint["generator"]["model.blocks.0.self_attn.q.weight"]
    expected = torch.tensor([[3.0, 6.0], [4.0, 8.0]], dtype=torch.bfloat16)
    assert torch.equal(weight, expected)
    assert all("lora_" not in key for key in checkpoint["generator"])
    assert manifest["training_checkpoint"]["adapter"] == "adapter_ema.safetensors"
    assert manifest["schema_version"] == 2
    assert manifest["training_checkpoint"]["raw_adapter"]["metadata"]["kind"] == "raw"
    assert manifest["training_checkpoint"]["ema_adapter"]["metadata"]["kind"] == "ema"
    assert manifest["adapter"]["target_module_names"] == ["blocks.0.self_attn.q"]
    assert len(manifest["adapter"]["tensor_schema"]) == 2
    assert manifest["manifest_sha256"] == canonical_json_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )


def test_merge_rejects_base_hash_mismatch(tmp_path):
    base, directory, config = _make_checkpoint(tmp_path)
    other = tmp_path / "other.pt"
    other.write_bytes(base.read_bytes() + b"changed")
    with pytest.raises(RuntimeError, match="Base checkpoint hash mismatch"):
        merge_stage1_ema_checkpoint(
            base_checkpoint=other,
            training_checkpoint=directory,
            output_path=tmp_path / "merged.pt",
            config=config,
            wrapper_builder=_Wrapper,
        )


def test_merge_rejects_missing_formal_adapter_metadata(tmp_path):
    base, directory, config = _make_checkpoint(tmp_path)
    from safetensors.torch import load_file, save_file

    ema_path = directory / "adapter_ema.safetensors"
    state = load_file(str(ema_path))
    save_file(state, str(ema_path), metadata={"adapter_kind": "ema"})
    write_checkpoint_manifest(
        directory,
        completed_step=75,
        world_size=6,
        sequence_parallel_size=3,
        data_parallel_size=2,
        resumable=False,
    )
    with pytest.raises(RuntimeError, match="ema adapter metadata mismatch"):
        merge_stage1_ema_checkpoint(
            base_checkpoint=base,
            training_checkpoint=directory,
            output_path=tmp_path / "merged.pt",
            config=config,
            wrapper_builder=_Wrapper,
        )


@pytest.mark.parametrize(
    "source_name",
    [
        "base",
        "adapter_raw.safetensors",
        "adapter_ema.safetensors",
        "resolved_config.yaml",
        "checkpoint_manifest.json",
        "base_reference.json",
        "_SUCCESS",
        "new-file-inside-checkpoint.pt",
    ],
)
@pytest.mark.parametrize("output_kind", ["checkpoint", "manifest"])
def test_merge_outputs_can_never_overlap_stage1_sources_or_tree(
    tmp_path, source_name, output_kind
):
    base, directory, config = _make_checkpoint(tmp_path)
    unsafe = base if source_name == "base" else directory / source_name
    checkpoint_output = tmp_path / "merged.pt"
    manifest_output = tmp_path / "merged.manifest.json"
    if output_kind == "checkpoint":
        checkpoint_output = unsafe
    else:
        manifest_output = unsafe
    with pytest.raises(ValueError, match="outside the immutable Stage-1 checkpoint"):
        merge_stage1_ema_checkpoint(
            base_checkpoint=base,
            training_checkpoint=directory,
            output_path=checkpoint_output,
            output_manifest_path=manifest_output,
            config=config,
            wrapper_builder=_Wrapper,
        )


def test_merge_rejects_existing_output_and_config_drift(tmp_path):
    base, directory, config = _make_checkpoint(tmp_path)
    output = tmp_path / "already-exists.pt"
    output.write_bytes(b"owned-by-user")
    with pytest.raises(FileExistsError, match="immutable"):
        merge_stage1_ema_checkpoint(
            base_checkpoint=base,
            training_checkpoint=directory,
            output_path=output,
            config=config,
            wrapper_builder=_Wrapper,
        )

    drifted = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    drifted.adapter.rank = 2
    with pytest.raises(ValueError, match="config override differs"):
        merge_stage1_ema_checkpoint(
            base_checkpoint=base,
            training_checkpoint=directory,
            output_path=tmp_path / "not-published.pt",
            config=drifted,
            wrapper_builder=_Wrapper,
        )


def test_merge_rejects_source_snapshot_drift_before_publish(tmp_path, monkeypatch):
    import scripts.merge_lora_generator as merge_module

    base, directory, config = _make_checkpoint(tmp_path)
    original = merge_module._source_snapshot
    calls = 0

    def drifting_snapshot(paths):
        nonlocal calls
        calls += 1
        value = original(paths)
        if calls == 2:
            value[0] = {**value[0], "sha256": "0" * 64}
        return value

    monkeypatch.setattr(merge_module, "_source_snapshot", drifting_snapshot)
    output = tmp_path / "drifted.pt"
    with pytest.raises(RuntimeError, match="source artifacts changed"):
        merge_stage1_ema_checkpoint(
            base_checkpoint=base,
            training_checkpoint=directory,
            output_path=output,
            config=config,
            wrapper_builder=_Wrapper,
        )
    assert not output.exists()


def test_merge_manifest_failure_removes_new_checkpoint(tmp_path, monkeypatch):
    import scripts.merge_lora_generator as merge_module

    base, directory, config = _make_checkpoint(tmp_path)

    def fail_manifest(*args, **kwargs):
        del args, kwargs
        raise OSError("injected manifest publication failure")

    monkeypatch.setattr(merge_module, "atomic_write_json", fail_manifest)
    output = tmp_path / "orphan-protected.pt"
    manifest = tmp_path / "orphan-protected.manifest.json"
    with pytest.raises(OSError, match="injected"):
        merge_stage1_ema_checkpoint(
            base_checkpoint=base,
            training_checkpoint=directory,
            output_path=output,
            output_manifest_path=manifest,
            config=config,
            wrapper_builder=_Wrapper,
        )
    assert not output.exists()
    assert not manifest.exists()
