from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from scripts.merge_lora_generator import merge_stage1_ema_checkpoint
from utils.lora_utils import configure_lora_for_model, save_lora_safetensors_strict
from utils.stage1_checkpoint import write_checkpoint_manifest, write_success_marker
from utils.stage1_io import atomic_write_json, sha256_file


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
    state = {}
    for key, tensor in peft_model.state_dict().items():
        if ".lora_A." in key:
            tensor.data.copy_(torch.tensor([[1.0, 2.0]]))
        elif ".lora_B." in key:
            tensor.data.copy_(torch.tensor([[3.0], [4.0]]))
    save_lora_safetensors_strict(
        peft_model,
        directory / "adapter_ema.safetensors",
        metadata={"adapter_kind": "ema", "optimizer_step": 75},
    )
    # Raw intentionally differs; merge must never select it.
    for name, parameter in peft_model.named_parameters():
        if "lora_" in name:
            parameter.data.fill_(99.0)
    save_lora_safetensors_strict(
        peft_model,
        directory / "adapter_raw.safetensors",
        metadata={"adapter_kind": "raw", "optimizer_step": 75},
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
