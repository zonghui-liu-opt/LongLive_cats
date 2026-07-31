import csv
from pathlib import Path

from PIL import Image
import pytest

from utils.stage1_i2v_data import load_stage1_i2v_manifest


def _write_csv(path: Path, rows, extra=False):
    fields = ["video", "prompt", "input_image", "height", "width", "bucket"]
    if extra:
        fields.append("source_note")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_resolves_paths_and_hashes_unknown_columns(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"fixture")
    Image.new("RGB", (832, 480), color=(1, 2, 3)).save(tmp_path / "a.png")
    csv_path = tmp_path / "metadata.csv"
    row = {
        "video": "a.mp4",
        "prompt": "cat",
        "input_image": "a.png",
        "height": "480",
        "width": "832",
        "bucket": "landscape",
        "source_note": "kept in hash",
    }
    _write_csv(csv_path, [row], extra=True)
    records = load_stage1_i2v_manifest(csv_path, expected_num_samples=1)
    assert records[0].video_path == (tmp_path / "a.mp4").resolve()
    assert records[0].canonical_row["source_note"] == "kept in hash"
    before = records[0].row_sha256
    row["source_note"] = "changed"
    _write_csv(csv_path, [row], extra=True)
    after = load_stage1_i2v_manifest(csv_path)[0].row_sha256
    assert before != after


def test_manifest_rejects_duplicate_video_and_wrong_bucket(tmp_path):
    csv_path = tmp_path / "metadata.csv"
    rows = [
        {"video": "a.mp4", "prompt": "a", "input_image": "a.png", "height": "480", "width": "832", "bucket": "landscape"},
        {"video": "a.mp4", "prompt": "b", "input_image": "b.png", "height": "480", "width": "832", "bucket": "landscape"},
    ]
    _write_csv(csv_path, rows)
    with pytest.raises(ValueError, match="duplicate video"):
        load_stage1_i2v_manifest(csv_path, require_files=False)
    rows[1]["video"] = "b.mp4"
    rows[0]["bucket"] = "portrait"
    _write_csv(csv_path, rows)
    with pytest.raises(ValueError, match="does not match"):
        load_stage1_i2v_manifest(csv_path, require_files=False)


def test_manifest_rejects_non_rgb_and_exif_rotation(tmp_path):
    (tmp_path / "a.mp4").write_bytes(b"fixture")
    Image.new("RGBA", (832, 480)).save(tmp_path / "a.png")
    csv_path = tmp_path / "metadata.csv"
    _write_csv(csv_path, [{"video": "a.mp4", "prompt": "cat", "input_image": "a.png", "height": "480", "width": "832", "bucket": "landscape"}])
    with pytest.raises(ValueError, match="must be RGB"):
        load_stage1_i2v_manifest(csv_path)


def test_local_one_row_fixture_cannot_claim_full_dataset():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="exactly 600"):
        load_stage1_i2v_manifest(
            root / "training_sets" / "metadata_600clips_480x832_buckets.csv",
            expected_num_samples=600,
            require_files=False,
        )
