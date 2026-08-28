from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image

from utils.dataset import (
    ImagePromptDataset,
    MetadataImagePromptDataset,
    image_prompt_collate_fn,
)

FIELDNAMES = ("input_image", "prompt", "height", "width", "bucket")


def _write_metadata(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def test_metadata_dataset_resolves_relative_images_and_preserves_csv_order(
    tmp_path, monkeypatch
):
    testset = tmp_path / "testset"
    image_dir = testset / "images_20cases_832x480"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (48, 32), color=(255, 0, 0)).save(image_dir / "000554.png")
    Image.new("RGB", (48, 32), color=(0, 255, 0)).save(image_dir / "000123.png")
    metadata = testset / "metadata_2cases_32x48.csv"
    _write_metadata(
        metadata,
        [
            {
                "input_image": "images_20cases_832x480/000554.png",
                "prompt": "第一只猫抬起前爪。",
                "height": 32,
                "width": 48,
                "bucket": "landscape",
            },
            {
                "input_image": "images_20cases_832x480/000123.png",
                "prompt": "第二只猫跳跃。",
                "height": 32,
                "width": 48,
                "bucket": "landscape",
            },
        ],
    )

    unrelated_cwd = tmp_path / "unrelated-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    dataset = MetadataImagePromptDataset(
        metadata_path=str(metadata),
        image_size=(32, 48),
        num_blocks=3,
    )

    assert dataset._mode == "metadata-image+prompt"
    assert dataset.metadata_path == metadata.resolve()
    assert dataset.metadata_sha256 == hashlib.sha256(metadata.read_bytes()).hexdigest()
    assert [Path(record.input_image).name for record in dataset.records] == [
        "000554.png",
        "000123.png",
    ]
    # Runtime consumes the tensors decoded during preflight rather than
    # reopening a source that might change while the 5B model loads.
    Image.new("RGB", (48, 32), color=(0, 0, 255)).save(image_dir / "000554.png")
    first = dataset[0]
    second = dataset[1]
    assert first["idx"] == 0
    assert first["prompts"] == ["第一只猫抬起前爪。"] * 3
    assert second["prompts"] == ["第二只猫跳跃。"] * 3
    assert first["image"].shape == (3, 32, 48)
    assert first["image"].dtype == torch.float16
    assert torch.equal(
        first["image"][:, 0, 0],
        torch.tensor([1.0, -1.0, -1.0], dtype=torch.float16),
    )

    batch = image_prompt_collate_fn([first, second])
    assert batch["image"].shape == (2, 3, 32, 48)
    assert batch["prompts"] == [first["prompts"], second["prompts"]]
    assert batch["idx"].tolist() == [0, 1]


def test_metadata_dataset_allows_one_image_with_distinct_prompts(tmp_path):
    image_dir = tmp_path / "images_20cases_832x480"
    image_dir.mkdir()
    image_path = image_dir / "000313.png"
    Image.new("RGB", (48, 32), color=(12, 34, 56)).save(image_path)
    metadata = tmp_path / "metadata.csv"
    _write_metadata(
        metadata,
        [
            {
                "input_image": "images_20cases_832x480/000313.png",
                "prompt": "The cat jumps over a toy.",
                "height": 32,
                "width": 48,
                "bucket": "landscape",
            },
            {
                "input_image": "images_20cases_832x480/000313.png",
                "prompt": "The same cat slowly sits down.",
                "height": 32,
                "width": 48,
                "bucket": "landscape",
            },
        ],
    )

    dataset = MetadataImagePromptDataset(
        metadata_path=str(metadata),
        image_size=(32, 48),
        num_blocks=3,
    )

    assert len(dataset) == 2
    assert dataset.records[0].input_image == dataset.records[1].input_image
    assert dataset.records[0].image_sha256 == dataset.records[1].image_sha256
    assert dataset.records[0].row_sha256 != dataset.records[1].row_sha256
    assert dataset[0]["idx"] == 0
    assert dataset[1]["idx"] == 1
    assert dataset[0]["prompts"] == ["The cat jumps over a toy."] * 3
    assert dataset[1]["prompts"] == ["The same cat slowly sits down."] * 3
    assert torch.equal(dataset[0]["image"], dataset[1]["image"])
    batch = image_prompt_collate_fn([dataset[0], dataset[1]])
    assert batch["image"].shape == (2, 3, 32, 48)
    assert batch["idx"].tolist() == [0, 1]
    assert batch["prompts"] == [dataset[0]["prompts"], dataset[1]["prompts"]]


def test_metadata_dataset_rejects_profile_geometry_mismatch(tmp_path):
    Image.new("RGB", (48, 32)).save(tmp_path / "000554.png")
    metadata = tmp_path / "metadata.csv"
    _write_metadata(
        metadata,
        [
            {
                "input_image": "000554.png",
                "prompt": "cat",
                "height": 32,
                "width": 48,
                "bucket": "landscape",
            }
        ],
    )

    with pytest.raises(ValueError, match="configured inference size"):
        MetadataImagePromptDataset(
            metadata_path=str(metadata),
            image_size=(48, 32),
            num_blocks=3,
        )


def test_metadata_dataset_rejects_trailing_dot_in_image_path(tmp_path):
    Image.new("RGB", (48, 32)).save(tmp_path / "000554.png")
    metadata = tmp_path / "metadata.csv"
    _write_metadata(
        metadata,
        [
            {
                "input_image": "000554.png.",
                "prompt": "cat",
                "height": 32,
                "width": 48,
                "bucket": "landscape",
            }
        ],
    )

    with pytest.raises(FileNotFoundError, match="missing input image"):
        MetadataImagePromptDataset(
            metadata_path=str(metadata),
            image_size=(32, 48),
            num_blocks=3,
        )


def test_legacy_image_prompt_directory_mode_is_unchanged(tmp_path):
    image_dir = tmp_path / "images"
    prompt_dir = tmp_path / "prompts"
    image_dir.mkdir()
    prompt_dir.mkdir()
    Image.new("RGB", (48, 32), color=(128, 128, 128)).save(image_dir / "cat.png")
    (prompt_dir / "cat.txt").write_text("legacy prompt\n", encoding="utf-8")

    dataset = ImagePromptDataset(
        data_path=str(tmp_path),
        image_size=(32, 48),
        num_blocks=3,
    )
    item = dataset[0]
    assert dataset._mode == "image+prompt"
    assert item["prompts"] == ["legacy prompt"] * 3
    assert item["image"].shape == (3, 32, 48)
