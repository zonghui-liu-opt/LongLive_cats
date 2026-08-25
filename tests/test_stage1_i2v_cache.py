import json

import pytest
import torch
from omegaconf import OmegaConf

from scripts.precompute_stage1_i2v_cache import (
    _build_stage1_cache_vae,
    _validate_precompute_runtime,
)
from utils.stage1_i2v_data import (
    Stage1I2VCacheDataset,
    load_cache_manifest,
    rgb_uint8_to_vae_input,
    save_cache_artifact,
    stage1_i2v_cache_collate,
    write_cache_manifest,
)
from utils.stage1_i2v_data import Stage1I2VRecord


def _tensors(h=30, w=52):
    return {
        "video_latent": torch.zeros(24, 48, h, w, dtype=torch.bfloat16),
        "initial_latent": torch.ones(1, 48, h, w, dtype=torch.bfloat16),
        "prompt_embeds": torch.zeros(512, 4096, dtype=torch.bfloat16),
        "prompt_mask": torch.ones(512, dtype=torch.bool),
    }


def test_rgb_channel_order_and_range():
    value = torch.tensor([[[0, 127, 255]]], dtype=torch.uint8)
    converted = rgb_uint8_to_vae_input(value)
    assert converted.shape == (1, 1, 3)
    assert converted[0, 0, 0].item() == -1.0
    assert converted[0, 0, 2].item() == 1.0


def test_cache_roundtrip_overwrites_video_latent_zero(tmp_path):
    artifact = save_cache_artifact(
        tmp_path / "sample_000000.safetensors", _tensors(), metadata={"row_id": 0}
    )
    record = Stage1I2VRecord(
        row_id=0,
        video_path=tmp_path / "source.mp4",
        prompt="cat",
        input_image_path=tmp_path / "source.png",
        height=480,
        width=832,
        bucket="landscape",
        canonical_row={"prompt": "cat"},
        row_sha256="rowhash",
    )
    source = {"aggregate_sha256": "sourcehash"}
    write_cache_manifest(tmp_path, records=[record], artifacts=[artifact], source_fingerprint=source)
    dataset = Stage1I2VCacheDataset(tmp_path, expected_num_samples=1, verify_on_read=True)
    item = dataset[0]
    assert torch.equal(item["clean_latent"][0:1], item["initial_latent"])
    batch = stage1_i2v_cache_collate([item])
    assert batch["clean_latent"].shape == (1, 24, 48, 30, 52)
    assert batch["prompt_embeds"].shape == (1, 512, 4096)


def test_cache_hash_and_manifest_hash_fail_fast(tmp_path):
    artifact = save_cache_artifact(
        tmp_path / "sample_000000.safetensors", _tensors(), metadata={"row_id": 0}
    )
    record = Stage1I2VRecord(0, tmp_path / "v", "cat", tmp_path / "i", 480, 832, "landscape", {}, "row")
    path = write_cache_manifest(tmp_path, records=[record], artifacts=[artifact], source_fingerprint={"aggregate_sha256": "x"})
    manifest = json.loads(path.read_text())
    manifest["num_samples"] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="manifest hash mismatch"):
        load_cache_manifest(tmp_path)


def test_cache_rejects_wrong_shape_or_dtype(tmp_path):
    tensors = _tensors()
    tensors["video_latent"] = tensors["video_latent"].float()
    with pytest.raises(ValueError, match="dtype"):
        save_cache_artifact(tmp_path / "bad.safetensors", tensors, metadata={})


def test_cache_precompute_runtime_enforces_world_size_and_cuda():
    config = OmegaConf.create(
        {
            "cache_precompute": {
                "expected_world_size": 4,
                "require_cuda": True,
                "log_every_records": 10,
            }
        }
    )
    with pytest.raises(RuntimeError, match="world-size mismatch"):
        _validate_precompute_runtime(
            config, world_size=1, device=torch.device("cpu")
        )
    with pytest.raises(RuntimeError, match="requires CUDA"):
        _validate_precompute_runtime(
            config, world_size=4, device=torch.device("cpu")
        )


def test_cache_precompute_runtime_keeps_generic_config_compatible():
    config = OmegaConf.create({})
    assert (
        _validate_precompute_runtime(
            config, world_size=1, device=torch.device("cpu")
        )
        == 10
    )


def test_stage1_cache_builder_configures_complete_vae_as_bfloat16(monkeypatch):
    from utils import wan_5b_wrapper

    class TinyVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv3d(3, 3, kernel_size=1)
            self.register_buffer("scale", torch.ones(1, dtype=torch.float32))

    checkpoint = object()
    captured = {}

    def fake_wrapper(*, vae_checkpoint):
        captured["checkpoint"] = vae_checkpoint
        return TinyVAE()

    monkeypatch.setattr(wan_5b_wrapper, "WanVAEWrapper", fake_wrapper)
    vae = _build_stage1_cache_vae(checkpoint, device=torch.device("cpu"))

    assert captured["checkpoint"] is checkpoint
    assert {parameter.dtype for parameter in vae.parameters()} == {torch.bfloat16}
    assert vae.scale.dtype == torch.bfloat16
    assert not vae.training
    assert all(not parameter.requires_grad for parameter in vae.parameters())
