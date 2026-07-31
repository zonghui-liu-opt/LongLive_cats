import json
from pathlib import Path

import pytest
from safetensors.torch import save_file
import torch

from scripts.convert_diffsynth_wan22_to_longlive import (
    CHECKPOINT_FORMAT,
    convert_diffsynth_checkpoint,
)
from utils.stage1_io import sha256_file
from wan_5b.textimage2video import (
    resolve_wan_checkpoint_path,
    wan_checkpoint_source_files,
)


class TinyCausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 2)
        self.register_buffer("revision", torch.tensor([7], dtype=torch.int64))


class TinyGeneratorWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyCausalModel()


def _tiny_state():
    return {
        "proj.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3) / 8,
        "proj.bias": torch.tensor([0.25, -0.5], dtype=torch.float32),
        "revision": torch.tensor([7], dtype=torch.int64),
    }


def _convert(source, output):
    return convert_diffsynth_checkpoint(
        source_checkpoint=source,
        output_path=output,
        model_builder=TinyCausalModel,
        reload_wrapper_builder=TinyGeneratorWrapper,
        model_name="tiny-ti2v",
        conversion_command=["converter", "--tiny"],
    )


def test_direct_safetensors_converts_bf16_and_strictly_reloads(tmp_path):
    source = tmp_path / "merged.safetensors"
    output = tmp_path / "converted_causal_base.pt"
    save_file(_tiny_state(), source)
    source_hash = sha256_file(source)

    manifest = _convert(source, output)

    assert sha256_file(source) == source_hash
    assert Path(resolve_wan_checkpoint_path(source)) == source
    assert manifest["checkpoint_format"] == CHECKPOINT_FORMAT
    assert manifest["source"]["files"] == [
        {
            "path": source.name,
            "size": source.stat().st_size,
            "sha256": source_hash,
        }
    ]
    assert manifest["source_state"]["dtype_counts"] == {
        "float32": 2,
        "int64": 1,
    }
    assert manifest["generator_state"]["dtype_counts"] == {
        "bfloat16": 2,
        "int64": 1,
    }
    assert manifest["conversion_command"] == ["converter", "--tiny"]
    assert manifest["strict_reload"] is True
    assert manifest["output"]["sha256"] == sha256_file(output)

    payload = torch.load(output, map_location="cpu", weights_only=False)
    assert payload["checkpoint_format"] == CHECKPOINT_FORMAT
    assert set(payload["generator"]) == {
        "model.proj.weight",
        "model.proj.bias",
        "model.revision",
    }
    assert payload["generator"]["model.proj.weight"].dtype == torch.bfloat16
    assert payload["generator"]["model.proj.bias"].dtype == torch.bfloat16
    assert payload["generator"]["model.revision"].dtype == torch.int64
    torch.testing.assert_close(
        payload["generator"]["model.proj.weight"],
        _tiny_state()["proj.weight"].to(torch.bfloat16),
    )

    manifest_file = output.with_suffix(".manifest.json")
    assert json.loads(manifest_file.read_text(encoding="utf-8")) == manifest
    assert not list(tmp_path.glob(".*.tmp"))


def test_sharded_index_hashes_index_and_every_referenced_shard(tmp_path):
    source_dir = tmp_path / "flat"
    source_dir.mkdir()
    state = _tiny_state()
    shard1 = source_dir / "diffusion_pytorch_model-00001-of-00002.safetensors"
    shard2 = source_dir / "diffusion_pytorch_model-00002-of-00002.safetensors"
    save_file({"proj.weight": state["proj.weight"]}, shard1)
    save_file(
        {"proj.bias": state["proj.bias"], "revision": state["revision"]}, shard2
    )
    index = source_dir / "diffusion_pytorch_model.safetensors.index.json"
    index.write_text(
        json.dumps(
            {
                "weight_map": {
                    "proj.weight": shard1.name,
                    "proj.bias": shard2.name,
                    "revision": shard2.name,
                }
            }
        ),
        encoding="utf-8",
    )
    before = {path.name: sha256_file(path) for path in (index, shard1, shard2)}

    output = tmp_path / "converted.pt"
    manifest = _convert(source_dir, output)

    assert Path(resolve_wan_checkpoint_path(source_dir)) == index
    assert [Path(path) for path in wan_checkpoint_source_files(source_dir)] == [
        index,
        shard1,
        shard2,
    ]
    assert [entry["path"] for entry in manifest["source"]["files"]] == [
        index.name,
        shard1.name,
        shard2.name,
    ]
    assert {path.name: sha256_file(path) for path in (index, shard1, shard2)} == before
    assert output.is_file()


@pytest.mark.parametrize(
    "bad_state",
    [
        {"proj.weight": torch.zeros(2, 3), "revision": torch.tensor([7])},
        {
            "proj.weight": torch.zeros(4, 3),
            "proj.bias": torch.zeros(2),
            "revision": torch.tensor([7]),
        },
        {**_tiny_state(), "unexpected": torch.ones(1)},
    ],
)
def test_missing_shape_mismatch_and_unexpected_keys_fail_before_output(
    tmp_path, bad_state
):
    source = tmp_path / "bad.safetensors"
    output = tmp_path / "must_not_exist.pt"
    save_file({key: value.contiguous() for key, value in bad_state.items()}, source)

    with pytest.raises((RuntimeError, ValueError)):
        _convert(source, output)

    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()


def test_index_rejects_missing_or_escaping_shards(tmp_path):
    index = tmp_path / "diffusion_pytorch_model.safetensors.index.json"
    for shard_name in ("missing.safetensors", "../escape.safetensors"):
        index.write_text(
            json.dumps({"weight_map": {"proj.weight": shard_name}}),
            encoding="utf-8",
        )
        with pytest.raises((FileNotFoundError, ValueError)):
            wan_checkpoint_source_files(index)


def test_failed_fresh_reload_does_not_replace_existing_output(tmp_path):
    source = tmp_path / "source.safetensors"
    output = tmp_path / "converted.pt"
    save_file(_tiny_state(), source)
    output.write_bytes(b"previous-valid-output")

    class WrongWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Linear(3, 2)

    with pytest.raises(RuntimeError):
        convert_diffsynth_checkpoint(
            source_checkpoint=source,
            output_path=output,
            model_builder=TinyCausalModel,
            reload_wrapper_builder=WrongWrapper,
        )

    assert output.read_bytes() == b"previous-valid-output"
    assert not output.with_suffix(".manifest.json").exists()


def _import_wan_wrapper(monkeypatch):
    # transformers<5 (the declared runtime) exports this unused symbol.  The
    # local developer environment may be newer, so make the legacy causal-model
    # import harmless without changing production behavior.
    import transformers.models.x_clip.modeling_x_clip as xclip_module

    monkeypatch.setattr(xclip_module, "x_clip_loss", lambda *_a, **_k: None, raising=False)
    import utils.wan_5b_wrapper as wrapper_module

    return wrapper_module


def test_real_tiny_bidirectional_and_causal_keys_shapes_are_identical(monkeypatch):
    _import_wan_wrapper(monkeypatch)
    from wan_5b.modules.causal_model import CausalWanModel
    from wan_5b.modules.model import WanModel

    kwargs = {
        "model_type": "ti2v",
        "patch_size": (1, 2, 2),
        "text_len": 8,
        "in_dim": 4,
        "dim": 16,
        "ffn_dim": 32,
        "freq_dim": 8,
        "text_dim": 12,
        "out_dim": 4,
        "num_heads": 2,
        "num_layers": 1,
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
    }
    bidirectional = WanModel(window_size=(-1, -1), **kwargs)
    causal = CausalWanModel(
        local_attn_size=-1, sink_size=0, num_frame_per_block=8, **kwargs
    )

    bidirectional_shapes = {
        key: tuple(value.shape) for key, value in bidirectional.state_dict().items()
    }
    causal_shapes = {key: tuple(value.shape) for key, value in causal.state_dict().items()}
    assert causal_shapes == bidirectional_shapes


def test_architecture_only_wrapper_uses_explicit_config_without_weights(
    tmp_path, monkeypatch
):
    wrapper_module = _import_wan_wrapper(monkeypatch)
    config = {
        "model_type": "ti2v",
        "patch_size": [1, 2, 2],
        "text_len": 8,
        "in_dim": 4,
        "dim": 16,
        "ffn_dim": 32,
        "freq_dim": 8,
        "text_dim": 12,
        "out_dim": 4,
        "num_heads": 2,
        "num_layers": 1,
        "window_size": [-1, -1],
        "qk_norm": True,
        "cross_attn_norm": True,
        "eps": 1e-6,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    model = wrapper_module.build_wan_model(
        model_name="tiny-ti2v",
        is_causal=True,
        architecture_root=tmp_path,
        init_weights=False,
        num_frame_per_block=8,
    )

    assert model.num_layers == 1
    assert model.num_frame_per_block == 8
    assert model.in_dim == 4


def test_text_encoder_paths_and_optional_boolean_mask(tmp_path, monkeypatch):
    wrapper_module = _import_wan_wrapper(monkeypatch)
    captured = {}

    class FakeT5(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()), requires_grad=False)

        def load_state_dict(self, state_dict, *args, **kwargs):
            captured["t5_state"] = state_dict
            return torch.nn.modules.module._IncompatibleKeys([], [])

        def forward(self, ids, mask):
            del mask
            return torch.ones(ids.shape[0], ids.shape[1], 4, device=ids.device)

    class FakeTokenizer:
        def __init__(self, *, name, seq_len, clean):
            captured["tokenizer"] = (name, seq_len, clean)

        def __call__(self, prompts, **kwargs):
            captured["prompts"] = (list(prompts), kwargs)
            return (
                torch.tensor([[1, 2, 0], [3, 0, 0]]),
                torch.tensor([[1, 1, 0], [1, 0, 0]]),
            )

    checkpoint = tmp_path / "t5.pt"
    tokenizer_dir = tmp_path / "tokenizer"
    monkeypatch.setattr(wrapper_module, "umt5_xxl", lambda **_kwargs: FakeT5())
    monkeypatch.setattr(wrapper_module, "HuggingfaceTokenizer", FakeTokenizer)
    monkeypatch.setattr(wrapper_module.torch, "load", lambda path, **_kwargs: {"path": path})

    encoder = wrapper_module.WanTextEncoder(
        t5_checkpoint=checkpoint,
        tokenizer_dir=tokenizer_dir,
        device="cpu",
    )
    result = encoder(["one", "two"], return_mask=True)

    assert captured["t5_state"] == {"path": str(checkpoint.resolve())}
    assert captured["tokenizer"] == (str(tokenizer_dir.resolve()), 512, "whitespace")
    assert result["prompt_mask"].dtype == torch.bool
    assert result["prompt_mask"].tolist() == [[True, True, False], [True, False, False]]
    assert torch.count_nonzero(result["prompt_embeds"][0, 2:]) == 0
    assert torch.count_nonzero(result["prompt_embeds"][1, 1:]) == 0


def test_vae_checkpoint_path_is_explicit(tmp_path, monkeypatch):
    wrapper_module = _import_wan_wrapper(monkeypatch)
    captured = {}

    class FakeVAE(torch.nn.Module):
        pass

    def fake_video_vae(*, pretrained_path):
        captured["path"] = pretrained_path
        return FakeVAE()

    monkeypatch.setattr(wrapper_module, "_video_vae", fake_video_vae)
    checkpoint = tmp_path / "vae.pt"
    wrapper_module.WanVAEWrapper(vae_checkpoint=checkpoint)
    assert captured["path"] == str(checkpoint.resolve())
