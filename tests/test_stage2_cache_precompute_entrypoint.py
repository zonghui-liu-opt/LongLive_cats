from __future__ import annotations

import csv
from pathlib import Path
import subprocess

from PIL import Image
import pytest

from scripts import validate_stage2_i2v_cache_inputs as validator
from utils.stage1_i2v_data import load_stage1_i2v_manifest

PROJECT_ROOT = Path(__file__).parents[1]
ENTRYPOINT = PROJECT_ROOT / "precompute_stage2_i2v_cache_h100_8gpu.sh"
ACTIONS = ("head_tilt_and_wink", "jump", "toy_play")


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _inputs(tmp_path: Path, *, metadata_has_action: bool = False):
    videos = tmp_path / "videos"
    images = tmp_path / "images"
    videos.mkdir()
    images.mkdir()
    metadata_rows = []
    action_rows = []
    for row_id in range(6):
        video = videos / f"{row_id}.mp4"
        image = images / f"{row_id}.png"
        video.write_bytes(f"video-{row_id}".encode())
        Image.new("RGB", (832, 480), (row_id, 0, 0)).save(image)
        metadata_row = {
            "video": f"videos/{row_id}.mp4",
            "prompt": f"cat action {row_id}",
            "input_image": f"images/{row_id}.png",
            "height": "480",
            "width": "832",
            "bucket": "landscape",
        }
        if metadata_has_action:
            metadata_row["action_id"] = ACTIONS[row_id % 3]
        metadata_rows.append(metadata_row)
        action_rows.append(
            {
                "video": f"videos/{row_id}.mp4",
                "action_id": ACTIONS[row_id % 3],
            }
        )
    metadata = tmp_path / "metadata.csv"
    action_sidecar = tmp_path / "actions.csv"
    metadata_fields = [
        "video",
        "prompt",
        "input_image",
        "height",
        "width",
        "bucket",
    ]
    if metadata_has_action:
        metadata_fields.append("action_id")
    _write_csv(metadata, metadata_fields, metadata_rows)
    _write_csv(action_sidecar, ["video", "action_id"], action_rows)

    records = load_stage1_i2v_manifest(
        metadata, expected_num_samples=6, require_files=True, validate_images=True
    )
    source = {
        "manifest_sha256": "a" * 64,
        "records": [
            {
                "row_id": record.row_id,
                "row_sha256": record.row_sha256,
                "height": record.height,
                "width": record.width,
                "bucket": record.bucket,
            }
            for record in records
        ],
        "source_fingerprint": {
            "records": [
                {"row_id": record.row_id, "row_sha256": record.row_sha256}
                for record in records
            ]
        },
    }
    source_path = tmp_path / "cache_manifest.json"
    source_path.write_text("{}\n", encoding="utf-8")
    return metadata, action_sidecar, source_path, source


def test_input_validator_accepts_six_column_metadata_and_exact_sidecar(
    tmp_path, monkeypatch
):
    metadata, action_sidecar, source_path, source = _inputs(tmp_path)
    monkeypatch.setattr(
        validator,
        "load_f25_preparation_input_manifest",
        lambda *_args, **_kwargs: source,
    )

    report = validator.validate_stage2_i2v_cache_inputs(
        metadata_path=metadata,
        action_labels_path=action_sidecar,
        source_cache_manifest_path=source_path,
        expected_num_samples=6,
        expected_num_actions=3,
        expected_samples_per_action=2,
    )

    assert report["status"] == "ok"
    assert report["action_ids"] == sorted(ACTIONS)
    assert report["action_counts"] == {action: 2 for action in sorted(ACTIONS)}
    assert report["target_latent_policy"] == "F25_sink_plus_24_future"


def test_input_validator_rejects_action_column_in_cache_bound_metadata(
    tmp_path, monkeypatch
):
    metadata, action_sidecar, source_path, source = _inputs(
        tmp_path, metadata_has_action=True
    )
    monkeypatch.setattr(
        validator,
        "load_f25_preparation_input_manifest",
        lambda *_args, **_kwargs: source,
    )

    with pytest.raises(ValueError, match="metadata header must be exactly"):
        validator.validate_stage2_i2v_cache_inputs(
            metadata_path=metadata,
            action_labels_path=action_sidecar,
            source_cache_manifest_path=source_path,
            expected_num_samples=6,
            expected_num_actions=3,
            expected_samples_per_action=2,
        )


def test_input_validator_rejects_absolute_sidecar_video_for_relative_metadata(
    tmp_path, monkeypatch
):
    metadata, action_sidecar, source_path, source = _inputs(tmp_path)
    with action_sidecar.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["video"] = str((tmp_path / rows[0]["video"]).resolve())
    _write_csv(action_sidecar, ["video", "action_id"], rows)
    monkeypatch.setattr(
        validator,
        "load_f25_preparation_input_manifest",
        lambda *_args, **_kwargs: source,
    )

    with pytest.raises(ValueError, match="must exactly match a metadata video string"):
        validator.validate_stage2_i2v_cache_inputs(
            metadata_path=metadata,
            action_labels_path=action_sidecar,
            source_cache_manifest_path=source_path,
            expected_num_samples=6,
            expected_num_actions=3,
            expected_samples_per_action=2,
        )


def test_input_validator_rejects_unbalanced_actions(tmp_path, monkeypatch):
    metadata, action_sidecar, source_path, source = _inputs(tmp_path)
    with action_sidecar.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["action_id"] = ACTIONS[1]
    _write_csv(action_sidecar, ["video", "action_id"], rows)
    monkeypatch.setattr(
        validator,
        "load_f25_preparation_input_manifest",
        lambda *_args, **_kwargs: source,
    )

    with pytest.raises(ValueError, match="exactly 3 actions with 2 samples each"):
        validator.validate_stage2_i2v_cache_inputs(
            metadata_path=metadata,
            action_labels_path=action_sidecar,
            source_cache_manifest_path=source_path,
            expected_num_samples=6,
            expected_num_actions=3,
            expected_samples_per_action=2,
        )


def test_input_validator_rejects_metadata_that_differs_from_stage1_manifest(
    tmp_path, monkeypatch
):
    metadata, action_sidecar, source_path, source = _inputs(tmp_path)
    source["records"][2]["row_sha256"] = "b" * 64
    monkeypatch.setattr(
        validator,
        "load_f25_preparation_input_manifest",
        lambda *_args, **_kwargs: source,
    )

    with pytest.raises(RuntimeError, match="metadata differs.*row 2"):
        validator.validate_stage2_i2v_cache_inputs(
            metadata_path=metadata,
            action_labels_path=action_sidecar,
            source_cache_manifest_path=source_path,
            expected_num_samples=6,
            expected_num_actions=3,
            expected_samples_per_action=2,
        )


def test_h100_entrypoint_runs_only_the_stage2_cache_chain():
    text = ENTRYPOINT.read_text(encoding="utf-8")

    for command in (
        "scripts/validate_stage2_i2v_cache_inputs.py",
        "scripts/prepare_stage2_i2v_f25_cache.py",
        "upgrade-source-manifest",
        "prepare-negative",
        "scripts/audit_stage2_i2v_cache.py audit",
    ):
        assert command in text
    for marker in (
        "CHECK_INPUTS_PASS",
        "CHECK_H100_PASS",
        "CHECK_F25_PASS",
        "CHECK_NEGATIVE_PASS",
        "CHECK_AUDIT_PASS",
        "STAGE2_F25_CACHE_PASS",
    ):
        assert text.count(marker) == 1
    assert "--nproc-per-node=8" in text
    assert "declare -A SEEN_GPU_IDS" in text
    assert 'CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}"' in text
    assert "cache_manifest.attested.json" in text
    assert "negative_conditioning_manifest.json" in text
    for name in (
        "LONG_LIVE_STAGE2_ARCHITECTURE_ROOT",
        "LONG_LIVE_STAGE2_GENERATOR_BASE",
        "LONG_LIVE_STAGE2_GENERATOR_MANIFEST",
        "LONG_LIVE_STAGE2_REAL_SCORE_BASE",
        "LONG_LIVE_STAGE2_REAL_SCORE_MANIFEST",
        "LONG_LIVE_STAGE2_METADATA_PATH",
        "LONG_LIVE_STAGE2_SOURCE_MANIFEST",
        "LONG_LIVE_STAGE2_ACTION_LABELS_PATH",
        "LONG_LIVE_STAGE2_CACHE_DIR",
        "LONG_LIVE_STAGE2_NEGATIVE_MANIFEST",
    ):
        assert f'export {name}="' in text
    assert "video_latent[1:25]" not in text
    assert "train.py" not in text
    assert "preflight_stage2_roles.py" not in text
    assert "merge_lora_generator.py" not in text

    completed = subprocess.run(
        ["bash", "-n", str(ENTRYPOINT)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
