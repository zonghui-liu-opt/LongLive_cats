from __future__ import annotations

import pytest

from utils.parameter_names import map_parameter_names_to_expected


def test_parameter_name_mapping_accepts_exact_and_unique_segment_suffixes():
    expected = (
        "base_model.model.block.lora_A.default.weight",
        "base_model.model.block.lora_B.default.weight",
    )
    assert map_parameter_names_to_expected(
        (
            "model.base_model.model.block.lora_A.default.weight",
            "base_model.model.block.lora_B.default.weight",
        ),
        expected,
        label="Stage-2 Generator LoRA",
    ) == {
        "base_model.model.block.lora_B.default.weight": (
            "base_model.model.block.lora_B.default.weight"
        ),
        "model.base_model.model.block.lora_A.default.weight": (
            "base_model.model.block.lora_A.default.weight"
        ),
    }


@pytest.mark.parametrize(
    ("actual", "expected", "match"),
    [
        (("prefix.weight",), ("weight", "prefix.weight"), "uniquely"),
        (("prefix.other",), ("weight",), "did not map"),
        (("one.weight", "two.weight"), ("weight",), "collision"),
        (("one.weight",), ("one.weight", "two.weight"), "incomplete"),
        (("one.weight", "one.weight"), ("one.weight",), "duplicate"),
        (("one.weight",), ("one.weight", "one.weight"), "duplicate"),
    ],
)
def test_parameter_name_mapping_rejects_ambiguous_missing_collision_and_drift(
    actual, expected, match
):
    with pytest.raises(ValueError, match=match):
        map_parameter_names_to_expected(actual, expected, label="Stage-2 test")


def test_parameter_name_mapping_requires_dot_segment_boundary():
    with pytest.raises(ValueError, match="did not map"):
        map_parameter_names_to_expected(
            ("notweight",),
            ("weight",),
            label="Stage-2 test",
        )
