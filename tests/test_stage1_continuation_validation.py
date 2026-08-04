from __future__ import annotations

from collections import Counter
import copy
import csv
from pathlib import Path

import pytest

from utils.stage1_continuation_validation import (
    CONTINUATION_ACTION_ORDERS,
    CONTINUATION_CATS,
    CONTINUATION_EXPECTED_BLOCKS,
    CONTINUATION_METADATA_COLUMNS,
    load_continuation_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_METADATA = (
    PROJECT_ROOT
    / "testsets"
    / "metadata_8cases_two_actions_continuation_480x832_253frames.csv"
)


def _formal_rows_with_absolute_images() -> list[dict[str, str]]:
    with FORMAL_METADATA.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["input_image"] = str(
            (FORMAL_METADATA.parent / row["input_image"]).resolve()
        )
    return rows


def _write_metadata(
    path: Path,
    rows: list[dict[str, str]],
    *,
    fieldnames=CONTINUATION_METADATA_COLUMNS,
) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def test_formal_continuation_metadata_matches_locked_contract():
    records = load_continuation_metadata(FORMAL_METADATA)

    assert len(records) == 8
    assert [record.row_id for record in records] == list(range(8))
    assert Counter(record.cat_id for record in records) == {
        cat_id: 2 for cat_id in CONTINUATION_CATS
    }
    assert Counter(record.action_order for record in records) == {
        action_order: 4 for action_order in CONTINUATION_ACTION_ORDERS
    }
    assert len({record.case_group for record in records}) == 8
    assert set(Counter(record.input_image_path for record in records).values()) == {2}
    for record in records:
        assert record.case_group == f"{record.cat_id}_{record.action_order}"
        assert record.block_schedule == CONTINUATION_EXPECTED_BLOCKS
        assert record.latent_frame_schedule == (24, 16, 24)
        assert record.total_latent_frames == 64
        assert record.soft_reanchor is True
        assert len(record.image_sha256) == len(record.row_sha256) == 64
        # The legal phrase contains the generic word “动作”; the loader must
        # only reject explicit HOLD action semantics.
        assert "耳朵轻微动作" in record.hold_prompt


@pytest.mark.parametrize(
    ("fieldnames", "message"),
    [
        (
            tuple(
                name for name in CONTINUATION_METADATA_COLUMNS if name != "hold_prompt"
            ),
            "header must exactly match",
        ),
        (
            (*CONTINUATION_METADATA_COLUMNS, "unexpected"),
            "header must exactly match",
        ),
        (
            tuple(reversed(CONTINUATION_METADATA_COLUMNS)),
            "header must exactly match",
        ),
    ],
)
def test_continuation_metadata_rejects_missing_unknown_or_reordered_columns(
    tmp_path,
    fieldnames,
    message,
):
    metadata = tmp_path / "metadata.csv"
    _write_metadata(
        metadata, _formal_rows_with_absolute_images(), fieldnames=fieldnames
    )
    with pytest.raises(ValueError, match=message):
        load_continuation_metadata(metadata)


def test_continuation_metadata_rejects_duplicate_columns(tmp_path):
    metadata = tmp_path / "metadata.csv"
    duplicate_header = list(CONTINUATION_METADATA_COLUMNS)
    duplicate_header[-1] = "hold_blocks"
    with metadata.open("w", encoding="utf-8-sig", newline="") as handle:
        csv.writer(handle).writerow(duplicate_header)

    with pytest.raises(ValueError, match="duplicate columns.*hold_blocks"):
        load_continuation_metadata(metadata)


def test_continuation_metadata_rejects_extra_unheaded_values(tmp_path):
    metadata = tmp_path / "metadata.csv"
    rows = _formal_rows_with_absolute_images()
    with metadata.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CONTINUATION_METADATA_COLUMNS)
        writer.writerow(
            [rows[0][name] for name in CONTINUATION_METADATA_COLUMNS] + ["extra"]
        )

    with pytest.raises(ValueError, match="extra unheaded values"):
        load_continuation_metadata(metadata)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("uppercase_boolean", "canonical lowercase 'true'"),
        ("wrong_action_order", "unsupported action_order"),
        ("wrong_case_group", "case_group.*must equal"),
        ("duplicate_case_group", "duplicate case_group"),
        ("wrong_block_schedule", "block schedule must be"),
        ("hold_forbidden_action", "hold_prompt contains forbidden"),
        ("missing_common_constraint", "is not self-contained"),
        ("mixed_action_prompt", "mixes actions"),
        ("wrong_image_reuse", "four input images, each exactly twice"),
    ],
)
def test_continuation_metadata_rejects_semantic_contract_violations(
    tmp_path,
    mutation,
    message,
):
    rows = copy.deepcopy(_formal_rows_with_absolute_images())
    if mutation == "uppercase_boolean":
        rows[0]["soft_reanchor"] = "TRUE"
    elif mutation == "wrong_action_order":
        rows[0]["action_order"] = "jump_only"
    elif mutation == "wrong_case_group":
        rows[0]["case_group"] = "ragdoll_wrong"
    elif mutation == "duplicate_case_group":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "wrong_block_schedule":
        rows[0]["hold_blocks"] = "1"
    elif mutation == "hold_forbidden_action":
        rows[0]["hold_prompt"] += " 猫咪眨眼并准备跳跃。"
    elif mutation == "missing_common_constraint":
        rows[0]["action_a_prompt"] = rows[0]["action_a_prompt"].replace(
            "纯白背景，", ""
        )
    elif mutation == "mixed_action_prompt":
        rows[0]["action_a_prompt"] += " 逗猫棒引导猫咪扑抓。"
    elif mutation == "wrong_image_reuse":
        rows[4]["input_image"] = rows[0]["input_image"]
    else:  # pragma: no cover
        raise AssertionError(mutation)

    metadata = tmp_path / "metadata.csv"
    _write_metadata(metadata, rows)
    with pytest.raises(ValueError, match=message):
        load_continuation_metadata(metadata)


def test_continuation_metadata_requires_exactly_eight_rows(tmp_path):
    metadata = tmp_path / "metadata.csv"
    _write_metadata(metadata, _formal_rows_with_absolute_images()[:-1])

    with pytest.raises(ValueError, match="exactly 8 rows"):
        load_continuation_metadata(metadata)
