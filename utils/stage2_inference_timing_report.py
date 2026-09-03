"""Validation and pure-data reports for optional Stage-2 inference timings."""

from __future__ import annotations

import csv
import io
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from utils.stage2_inference_batch import Stage2InferenceSample

STAGE2_INFERENCE_TIMING_SCHEMA = "longlive_stage2_inference_timing/v1"
STAGE2_INFERENCE_TIMING_SUMMARY_SCHEMA = "longlive_stage2_inference_timing_summary/v1"
_SECONDS = (
    "dit_seconds",
    "vae_decode_seconds",
    "video_postprocess_seconds",
    "total_seconds",
    "other_seconds",
)
_CALLS = ("dit_calls", "vae_decode_calls", "video_postprocess_calls")
_TIMING_KEYS = {
    "schema",
    "method",
    "device",
    "device_name",
    "rank",
    "rank_sample_index",
    "cold_start",
    "process_run_id",
    *_SECONDS,
    *_CALLS,
}
_GROUP_FIELDS = ("profile", "dataset", "height", "width", "device_name", "method")
_CSV_FIELDS = (
    "sample_key",
    "profile",
    "dataset",
    "row_id",
    "height",
    "width",
    "seed",
    "rank",
    "rank_sample_index",
    "process_run_id",
    "cold_start",
    "device",
    "device_name",
    "method",
    "output_frames",
    *_SECONDS,
    *_CALLS,
    "trace_sha256",
)


def _plain_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a plain nonnegative integer")
    return value


def validate_stage2_timing(value: Any) -> dict[str, Any]:
    """Return a validated copy, accepting zero-call injected runtime results."""

    if not isinstance(value, Mapping) or set(value) != _TIMING_KEYS:
        raise ValueError("Stage-2 inference timing schema keys mismatch")
    timing = dict(value)
    if timing["schema"] != STAGE2_INFERENCE_TIMING_SCHEMA:
        raise ValueError("Stage-2 inference timing schema mismatch")
    device = timing["device"]
    if device == "cpu":
        expected_method = "cpu_wall"
    elif isinstance(device, str) and re.fullmatch(r"cuda:[0-9]+", device):
        expected_method = "cuda_synchronized_wall"
    else:
        raise ValueError("timing.device must be cpu or cuda:N")
    if timing["method"] != expected_method:
        raise ValueError("timing.method does not match timing.device")
    if not isinstance(timing["device_name"], str) or not timing["device_name"].strip():
        raise ValueError("timing.device_name must be a nonempty string")
    if not isinstance(timing["process_run_id"], str) or not re.fullmatch(
        r"[0-9a-fA-F]{32}", timing["process_run_id"]
    ):
        raise ValueError("timing.process_run_id must be a 32-character hex string")
    for key in (*_CALLS, "rank", "rank_sample_index"):
        _plain_nonnegative_int(timing[key], f"timing.{key}")
    if type(timing["cold_start"]) is not bool or timing["cold_start"] != (
        timing["rank_sample_index"] == 0
    ):
        raise ValueError("timing.cold_start must equal (rank_sample_index == 0)")
    for key in _SECONDS:
        seconds = timing[key]
        if (
            type(seconds) not in (int, float)
            or not math.isfinite(seconds)
            or seconds < 0
        ):
            raise ValueError(f"timing.{key} must be finite nonnegative seconds")
        timing[key] = float(seconds)
    accounted = (
        timing["dit_seconds"]
        + timing["vae_decode_seconds"]
        + timing["video_postprocess_seconds"]
        + timing["other_seconds"]
    )
    if not math.isclose(accounted, timing["total_seconds"], rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            "timing stage seconds plus other_seconds must equal total_seconds"
        )
    return timing


def _statistics(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.9
    lower = math.floor(position)
    upper = math.ceil(position)
    p90 = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p90": p90,
        "min": ordered[0],
        "max": ordered[-1],
    }


def _subset(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "total_output_frames": sum(row["output_frames"] for row in rows),
        "metrics": {key: _statistics([row[key] for row in rows]) for key in _SECONDS},
    }


def build_stage2_timing_report(
    samples: Sequence[Stage2InferenceSample],
    traces: Mapping[str, Mapping[str, Any]],
    *,
    checkpoint: Mapping[str, Any],
    code_version: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    """Build summary JSON data and CSV bytes without reading or writing files.

    Traces are already artifact-validated by the caller. Historical traces may
    omit timing; they remain explicitly missing rather than zero measurements.
    """

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    trace_hashes: dict[str, str] = {}
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for sample in sorted(samples, key=lambda sample: sample.sample_key):
        trace = traces.get(sample.sample_key)
        if trace is not None and "trace_sha256" in trace:
            trace_hashes[sample.sample_key] = trace["trace_sha256"]
        generation = trace.get("generation", {}) if trace is not None else {}
        if "timing" not in generation:
            missing.append(sample.sample_key)
            continue
        timing = validate_stage2_timing(generation["timing"])
        output = trace.get("output", {})
        frames = _plain_nonnegative_int(
            (
                output["frame_count"]
                if "frame_count" in output
                else generation["output_pixel_frames"]
            ),
            "output.frame_count",
        )
        row = {
            "sample_key": sample.sample_key,
            "profile": sample.profile,
            "dataset": sample.dataset,
            "row_id": sample.row_id,
            "height": sample.height,
            "width": sample.width,
            "seed": sample.seed,
            "output_frames": frames,
            "trace_sha256": trace.get("trace_sha256", ""),
            **{key: value for key, value in timing.items() if key != "schema"},
        }
        rows.append(row)
        group_key = tuple(row[key] for key in _GROUP_FIELDS)
        groups.setdefault(group_key, []).append(row)

    summary = {
        "schema": STAGE2_INFERENCE_TIMING_SUMMARY_SCHEMA,
        "source": {
            "checkpoint": deepcopy(dict(checkpoint)),
            "code_version": deepcopy(dict(code_version)),
            "trace_sha256_by_sample": trace_hashes,
        },
        "sample_count": len(samples),
        "measured_sample_count": len(rows),
        "missing_timing_sample_count": len(missing),
        "missing_timing_sample_keys": missing,
        "group_by": list(_GROUP_FIELDS),
        "statistics": {
            "unit": "seconds",
            "total_scope": (
                "Generation start through atomic MP4 commit, including intermediate "
                "output guards, ffprobe and file hash validation. Excludes model loading, "
                "sample trace construction/writes and report/manifest finalization."
            ),
            "weighting": "Each measured sample has equal weight; video durations are not normalized.",
            "p90": "Linear interpolation at 0.9 * (sample_count - 1) in sorted values.",
            "all_samples": "All samples with recorded timing, including cold starts.",
            "excluding_first_per_rank_run": (
                "Exclude cold_start=true (rank_sample_index=0) for every rank and "
                "process_run_id, including separate runs represented by resumed traces. "
                "No additional warmup samples are generated."
            ),
            "missing_timing": "Excluded from CSV rows and all timing statistics; never treated as zero.",
        },
        "groups": [
            {
                **dict(zip(_GROUP_FIELDS, key)),
                "all_samples": _subset(group_rows),
                "excluding_first_per_rank_run": _subset(
                    [row for row in group_rows if not row["cold_start"]]
                ),
            }
            for key, group_rows in sorted(groups.items())
        ],
    }
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=_CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({**row, "cold_start": str(row["cold_start"]).lower()})
    return summary, output.getvalue().encode("utf-8")
