#!/usr/bin/env python3
"""Validate Stage-2 F25 metadata/action/source inputs before CUDA startup."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.stage1_i2v_schema import STAGE1_REQUIRED_COLUMNS  # noqa: E402
from utils.stage2_action_contract import STAGE2_EXPECTED_ACTION_COUNTS  # noqa: E402
from utils.stage2_i2v_data import (  # noqa: E402
    load_f25_preparation_input_manifest,
)


def _read_csv(
    path: Path, *, expected_header: list[str], label: str
) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_header:
            raise ValueError(
                f"{label} header must be exactly {','.join(expected_header)}; "
                f"got {reader.fieldnames}."
            )
        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(
                    f"{label} row {row_number} contains extra unheaded values."
                )
            rows.append({name: str(row[name]) for name in expected_header})
    return rows


def validate_stage2_i2v_cache_inputs(
    *,
    metadata_path: str | Path,
    action_labels_path: str | Path,
    source_cache_manifest_path: str | Path,
    expected_num_samples: int = 600,
    expected_num_actions: int = 3,
    expected_samples_per_action: int | None = None,
    expected_action_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Return a deterministic report or fail before any GPU process starts."""

    from utils.stage1_i2v_data import load_stage1_i2v_manifest

    if expected_samples_per_action is not None and expected_action_counts is not None:
        raise ValueError(
            "Specify expected_samples_per_action or expected_action_counts, not both."
        )
    uniform_expected_count: int | None = None
    if expected_action_counts is None:
        if expected_samples_per_action is None:
            expected_action_counts = dict(STAGE2_EXPECTED_ACTION_COUNTS)
        else:
            if (
                isinstance(expected_samples_per_action, bool)
                or not isinstance(expected_samples_per_action, int)
                or expected_samples_per_action <= 0
            ):
                raise ValueError("expected_samples_per_action must be positive.")
            if (
                expected_num_samples
                != expected_num_actions * expected_samples_per_action
            ):
                raise ValueError(
                    "expected_num_samples must equal expected_num_actions * "
                    "expected_samples_per_action."
                )
            uniform_expected_count = expected_samples_per_action
    if expected_action_counts is not None:
        if len(expected_action_counts) != expected_num_actions:
            raise ValueError(
                "expected_action_counts must contain expected_num_actions entries."
            )
        for action_id, count in expected_action_counts.items():
            if (
                not isinstance(action_id, str)
                or not action_id
                or action_id != action_id.strip()
            ):
                raise ValueError(
                    "expected_action_counts keys must be clean action ids."
                )
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError(
                    "expected_action_counts values must be positive integers."
                )
        if sum(expected_action_counts.values()) != expected_num_samples:
            raise ValueError("expected_action_counts must sum to expected_num_samples.")
    metadata_path = Path(metadata_path).expanduser().resolve()
    action_labels_path = Path(action_labels_path).expanduser().resolve()
    source_cache_manifest_path = Path(source_cache_manifest_path).expanduser().resolve()
    for path in (metadata_path, action_labels_path, source_cache_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata_rows = _read_csv(
        metadata_path,
        expected_header=list(STAGE1_REQUIRED_COLUMNS),
        label="metadata",
    )
    if len(metadata_rows) != expected_num_samples:
        raise ValueError(
            f"metadata must contain exactly {expected_num_samples} rows; "
            f"got {len(metadata_rows)}."
        )
    records = load_stage1_i2v_manifest(
        metadata_path,
        expected_num_samples=expected_num_samples,
        require_files=True,
        validate_images=True,
    )

    source = load_f25_preparation_input_manifest(
        source_cache_manifest_path,
        expected_num_samples=expected_num_samples,
    )
    source_records = source["records"]
    fingerprint_records = source["source_fingerprint"]["records"]
    for record, source_record, fingerprint_record in zip(
        records, source_records, fingerprint_records
    ):
        expected = {
            "row_id": record.row_id,
            "row_sha256": record.row_sha256,
            "height": record.height,
            "width": record.width,
            "bucket": record.bucket,
        }
        for key, value in expected.items():
            if source_record.get(key) != value:
                raise RuntimeError(
                    f"metadata differs from Stage-1 cache at row {record.row_id}: "
                    f"{key}={value!r}, manifest={source_record.get(key)!r}."
                )
        if (
            fingerprint_record.get("row_id") != record.row_id
            or fingerprint_record.get("row_sha256") != record.row_sha256
        ):
            raise RuntimeError(
                f"metadata differs from Stage-1 source fingerprint at row "
                f"{record.row_id}."
            )

    action_rows = _read_csv(
        action_labels_path,
        expected_header=["video", "action_id"],
        label="action sidecar",
    )
    if len(action_rows) != expected_num_samples:
        raise ValueError(
            f"action sidecar must contain exactly {expected_num_samples} rows; "
            f"got {len(action_rows)}."
        )
    metadata_videos = [row["video"] for row in metadata_rows]
    expected_videos = set(metadata_videos)
    if len(expected_videos) != expected_num_samples:
        raise ValueError("metadata video strings must be unique.")

    actions_by_video: dict[str, str] = {}
    for row_number, row in enumerate(action_rows, start=2):
        video = row["video"]
        action_id = row["action_id"]
        if not video or video != video.strip():
            raise ValueError(
                f"action sidecar row {row_number} has an invalid video value."
            )
        if video not in expected_videos:
            raise ValueError(
                f"action sidecar row {row_number} video must exactly match a "
                f"metadata video string; got {video!r}."
            )
        if video in actions_by_video:
            raise ValueError(f"action sidecar contains duplicate video {video!r}.")
        if not action_id or action_id != action_id.strip():
            raise ValueError(
                f"action sidecar row {row_number} has an invalid action_id."
            )
        actions_by_video[video] = action_id

    missing = [video for video in metadata_videos if video not in actions_by_video]
    if missing:
        raise ValueError(
            f"action sidecar is missing {len(missing)} metadata videos; "
            f"first={missing[0]!r}."
        )
    counts = Counter(actions_by_video[video] for video in metadata_videos)
    if uniform_expected_count is not None:
        if len(counts) != expected_num_actions or set(counts.values()) != {
            uniform_expected_count
        }:
            raise ValueError(
                f"action sidecar must contain exactly {expected_num_actions} actions "
                f"with {uniform_expected_count} samples each; got "
                f"{dict(sorted(counts.items()))}."
            )
    elif dict(counts) != expected_action_counts:
        raise ValueError(
            "action sidecar counts must be exactly "
            f"{expected_action_counts}; got "
            f"{dict(sorted(counts.items()))}."
        )

    return {
        "status": "ok",
        "metadata_path": str(metadata_path),
        "action_labels_path": str(action_labels_path),
        "source_cache_manifest_path": str(source_cache_manifest_path),
        "num_samples": expected_num_samples,
        "action_counts": dict(sorted(counts.items())),
        "action_ids": sorted(counts),
        "source_manifest_sha256": source["manifest_sha256"],
        "source_latent_policy": "F24_or_proven_F25_migration_input",
        "target_latent_policy": "F25_sink_plus_24_future",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-path", required=True)
    parser.add_argument("--action-labels-path", required=True)
    parser.add_argument("--source-cache-manifest", required=True)
    parser.add_argument("--expected-num-samples", type=int, default=600)
    parser.add_argument("--expected-num-actions", type=int, default=3)
    parser.add_argument("--expected-samples-per-action", type=int)
    parser.add_argument(
        "--action-ids-only",
        action="store_true",
        help="Print one validated action id per line for shell orchestration.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = validate_stage2_i2v_cache_inputs(
        metadata_path=args.metadata_path,
        action_labels_path=args.action_labels_path,
        source_cache_manifest_path=args.source_cache_manifest,
        expected_num_samples=args.expected_num_samples,
        expected_num_actions=args.expected_num_actions,
        expected_samples_per_action=args.expected_samples_per_action,
    )
    if args.action_ids_only:
        print("\n".join(report["action_ids"]))
    else:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
