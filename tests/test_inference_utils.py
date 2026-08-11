from __future__ import annotations

import pytest
import torch

from utils.inference_utils import (
    ExplicitIndexSampler,
    load_inference_lora_checkpoint,
    resolve_inference_sample_indices,
    resolve_inference_sample_seeds,
)
from utils.lora_utils import (
    configure_lora_for_model,
    get_canonical_lora_state_dict,
    save_lora_safetensors_strict,
)


class TinyTransformer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, value):
        return self.proj(value)


def _build_lora_model():
    model = TinyTransformer()
    model.requires_grad_(False)
    return configure_lora_for_model(
        model,
        "generator",
        {
            "type": "lora",
            "rank": 2,
            "alpha": 2,
            "dropout": 0.0,
            "bias": "none",
            "modules_to_save": [],
            "target_patterns": [r"^proj$"],
            "expected_target_modules": 1,
            "expected_trainable_parameters": 16,
            "expected_adapter_tensors": 2,
        },
        is_main_process=False,
    )


def test_explicit_index_and_seed_contracts_preserve_dataset_indices():
    indices = resolve_inference_sample_indices([0, 2], dataset_size=3)
    seeds = resolve_inference_sample_seeds([101, 102, 103], dataset_size=3)

    assert tuple(ExplicitIndexSampler(indices)) == (0, 2)
    assert seeds == (101, 102, 103)
    assert resolve_inference_sample_indices(None, dataset_size=3) == (0, 1, 2)
    assert resolve_inference_sample_seeds(None, dataset_size=3) is None


@pytest.mark.parametrize("indices", [[0, 0], [-1], [3], []])
def test_explicit_sample_indices_reject_invalid_shards(indices):
    with pytest.raises((TypeError, ValueError)):
        resolve_inference_sample_indices(indices, dataset_size=3)


@pytest.mark.parametrize("seeds", [[1, 2], [1, -1, 3], [1, True, 3]])
def test_sample_seeds_reject_wrong_length_or_invalid_values(seeds):
    with pytest.raises((TypeError, ValueError)):
        resolve_inference_sample_seeds(seeds, dataset_size=3)


def test_inference_loader_strictly_loads_stage1_safetensors(tmp_path):
    source = _build_lora_model()
    generator = torch.Generator().manual_seed(17)
    for parameter in source.parameters():
        if parameter.requires_grad:
            parameter.data.copy_(
                torch.randn(
                    parameter.shape,
                    generator=generator,
                    dtype=parameter.dtype,
                )
            )
    checkpoint = tmp_path / "adapter_ema.safetensors"
    expected = save_lora_safetensors_strict(source, checkpoint)

    target = _build_lora_model()
    report = load_inference_lora_checkpoint(target, checkpoint)
    actual = get_canonical_lora_state_dict(target)

    assert report["format"] == "safetensors"
    assert report["tensor_count"] == 2
    assert tuple(actual) == tuple(expected)
    assert all(torch.equal(actual[key], expected[key]) for key in expected)
