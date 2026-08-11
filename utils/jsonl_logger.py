"""Append-only JSONL metrics and lineage helpers for training.

The defaults remain the Stage-1 protocol.  Stage-2 supplies its own schema,
record types, and logical-step key through the same small writer so both
trainers share the crash-tolerant append/flush/fsync implementation without
mixing their metric contracts.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from typing import Any
import uuid

JSONL_SCHEMA = "longlive_stage1_metrics"
JSONL_SCHEMA_VERSION = 1


def _json_line(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )


def _scan_jsonl_bytes(
    path: Path, raw: bytes
) -> tuple[list[dict[str, Any]], int | None]:
    """Return parsed records and the byte offset of one disposable tail."""

    raw_lines = raw.splitlines(keepends=True)
    nonempty_indices = [
        index for index, line in enumerate(raw_lines) if line.rstrip(b"\r\n").strip()
    ]
    last_nonempty = nonempty_indices[-1] if nonempty_indices else -1
    records: list[dict[str, Any]] = []
    prefix_bytes = 0
    for index, line in enumerate(raw_lines):
        content = line.rstrip(b"\r\n")
        if not content.strip():
            prefix_bytes += len(line)
            continue
        try:
            value = json.loads(content.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            if index == last_nonempty:
                return records, prefix_bytes
            raise ValueError(
                f"Invalid JSONL record at {path}:{index + 1}; only a truncated "
                "final line may be ignored."
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL record at {path}:{index + 1} is not an object.")
        records.append(value)
        prefix_bytes += len(line)
    return records, None


def read_jsonl_tolerant(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Read JSONL while allowing only the final non-empty line to be truncated."""
    path = Path(path)
    if not path.exists():
        return []
    records, _truncate_before = _scan_jsonl_bytes(path, path.read_bytes())
    return records


def _repair_jsonl_tail_for_append(path: Path) -> None:
    """Discard one crash-truncated final record before appending a new run.

    Read-time tolerance alone is insufficient: appending after an invalid tail
    would turn that tail into middle-of-file corruption.  This function first
    validates that every earlier non-empty line is valid JSON, then either
    truncates the sole invalid final line or adds a missing newline after a
    complete final record.  The repair is flushed before the writer opens.
    """

    if not path.exists() or path.stat().st_size == 0:
        return
    raw = path.read_bytes()
    _records, truncate_before = _scan_jsonl_bytes(path, raw)

    if truncate_before is not None:
        with path.open("r+b") as handle:
            handle.truncate(truncate_before)
            handle.flush()
            os.fsync(handle.fileno())
        return
    if not raw.endswith(b"\n"):
        with path.open("ab") as handle:
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())


def run_parent_map(records: Sequence[Mapping[str, Any]]) -> dict[str, str | None]:
    parents: dict[str, str | None] = {}
    for record in records:
        if record.get("record_type") != "run_start":
            continue
        run_id = str(record.get("run_id", ""))
        if not run_id:
            raise ValueError("run_start record is missing run_id.")
        parent = record.get("parent_run_id")
        parents[run_id] = str(parent) if parent not in (None, "") else None
    return parents


def latest_run_id(records: Sequence[Mapping[str, Any]]) -> str | None:
    for record in reversed(records):
        if record.get("record_type") == "run_start" and record.get("run_id"):
            return str(record["run_id"])
    return None


def lineage_run_ids(
    records: Sequence[Mapping[str, Any]], run_id: str | None = None
) -> list[str]:
    parents = run_parent_map(records)
    current = run_id or latest_run_id(records)
    if current is None:
        return []
    lineage: list[str] = []
    seen: set[str] = set()
    while current is not None:
        if current in seen:
            raise ValueError(f"Run lineage contains a cycle at {current}.")
        if current not in parents:
            raise ValueError(f"Run lineage references missing parent/run {current}.")
        seen.add(current)
        lineage.append(current)
        current = parents[current]
    lineage.reverse()
    return lineage


def records_for_latest_lineage(
    records: Sequence[Mapping[str, Any]],
    *,
    record_type: str = "train_step",
    run_id: str | None = None,
    step_key: str | None = None,
) -> list[dict[str, Any]]:
    """Return latest-lineage records, with child runs overriding parent steps."""
    lineage = lineage_run_ids(records, run_id=run_id)
    if not lineage:
        return []
    by_run: dict[str, list[tuple[int, Mapping[str, Any]]]] = {
        item: [] for item in lineage
    }
    for sequence, record in enumerate(records):
        owner = str(record.get("run_id", ""))
        if owner in by_run and record.get("record_type") == record_type:
            by_run[owner].append((sequence, record))

    if record_type != "train_step":
        return [dict(record) for owner in lineage for _, record in by_run[owner]]

    if step_key is None:
        train_records = [record for owner in lineage for _, record in by_run[owner]]
        has_logical = ["logical_substep_id" in record for record in train_records]
        if any(has_logical) and not all(has_logical):
            raise ValueError(
                "train_step lineage mixes optimizer_step and logical_substep_id records."
            )
        step_key = (
            "logical_substep_id"
            if has_logical and all(has_logical)
            else "optimizer_step"
        )

    # Iterate root -> child so overlapping logical/optimizer steps are replaced
    # by the child while unrelated historical steps remain visible.
    selected: dict[int, tuple[int, int, dict[str, Any]]] = {}
    for depth, owner in enumerate(lineage):
        for sequence, record in by_run[owner]:
            if step_key not in record:
                raise ValueError(f"train_step record in run {owner} has no {step_key}.")
            step = int(record[step_key])
            selected[step] = (depth, sequence, dict(record))
    return [selected[step][2] for step in sorted(selected)]


def next_attempt_index(
    records: Sequence[Mapping[str, Any]],
    *,
    run_id: str | None = None,
    checkpoint_next_attempt_index: int = 0,
) -> int:
    lineage = set(lineage_run_ids(records, run_id=run_id))
    maximum = -1
    for record in records:
        if str(record.get("run_id", "")) not in lineage:
            continue
        if "attempt_index" in record:
            maximum = max(maximum, int(record["attempt_index"]))
    checkpoint_value = int(checkpoint_next_attempt_index)
    if checkpoint_value < 0:
        raise ValueError("checkpoint_next_attempt_index must be non-negative.")
    return max(checkpoint_value, maximum + 1)


@dataclass(frozen=True)
class LogicalWorkload:
    global_samples: int
    logical_source_frames: int
    dit_tokens: int
    supervised_tokens: int
    patch_tokens_per_latent_frame: int


def logical_workload(
    *,
    latent_height: int,
    latent_width: int,
    global_samples: int,
    latent_frames: int = 24,
    supervised_frames: int = 23,
    source_frames: int = 93,
    patch_size: tuple[int, int, int] = (1, 2, 2),
) -> LogicalWorkload:
    temporal_patch, height_patch, width_patch = map(int, patch_size)
    values = {
        "latent_height": latent_height,
        "latent_width": latent_width,
        "global_samples": global_samples,
        "latent_frames": latent_frames,
        "supervised_frames": supervised_frames,
        "source_frames": source_frames,
        "temporal_patch": temporal_patch,
        "height_patch": height_patch,
        "width_patch": width_patch,
    }
    if any(
        isinstance(value, bool) or int(value) != value or int(value) <= 0
        for value in values.values()
    ):
        raise ValueError(
            f"Logical workload dimensions must be positive integers: {values}"
        )
    if latent_height % height_patch or latent_width % width_patch:
        raise ValueError("Latent H/W must be divisible by the spatial patch size.")
    if latent_frames % temporal_patch:
        raise ValueError("latent_frames must be divisible by the temporal patch size.")
    patch_tokens = (latent_height // height_patch) * (latent_width // width_patch)
    # The locked DiT logical-work definition accounts for both noised and
    # teacher-forcing clean streams.  SP is intentionally absent here.
    dit_per_sample = 2 * (latent_frames // temporal_patch) * patch_tokens
    supervised_per_sample = supervised_frames * patch_tokens
    return LogicalWorkload(
        global_samples=global_samples,
        logical_source_frames=global_samples * source_frames,
        dit_tokens=global_samples * dit_per_sample,
        supervised_tokens=global_samples * supervised_per_sample,
        patch_tokens_per_latent_frame=patch_tokens,
    )


def throughput_fields(
    workload: LogicalWorkload, elapsed_seconds: float
) -> dict[str, float]:
    elapsed = float(elapsed_seconds)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError(
            f"elapsed_seconds must be positive and finite, got {elapsed_seconds}."
        )
    return {
        "logical_dit_tokens_per_second": workload.dit_tokens / elapsed,
        "supervised_tokens_per_second": workload.supervised_tokens / elapsed,
        "samples_per_second": workload.global_samples / elapsed,
        "logical_pixel_frames_per_second": workload.logical_source_frames / elapsed,
    }


def elapsed_summary(elapsed_by_rank: Iterable[float]) -> dict[str, float]:
    values = [float(value) for value in elapsed_by_rank]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("elapsed_by_rank must contain positive finite values.")
    maximum = max(values)
    mean = sum(values) / len(values)
    return {
        "step_seconds_max": maximum,
        "step_seconds_mean": mean,
        "straggler_ratio": maximum / mean,
    }


class JsonlLogger:
    """Rank-zero append-only writer with resume-safe attempt numbering."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        experiment_id: str,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        resume_from_step: int = 0,
        checkpoint_next_attempt_index: int = 0,
        fsync_every_steps: int = 10,
        enabled: bool = True,
        run_metadata: Mapping[str, Any] | None = None,
        schema: str = JSONL_SCHEMA,
        schema_version: int = JSONL_SCHEMA_VERSION,
        supported_record_types: Iterable[str] = ("train_step", "nonfinite_attempt"),
    ) -> None:
        self.path = Path(path)
        self.experiment_id = str(experiment_id)
        self.run_id = run_id or uuid.uuid4().hex
        self.parent_run_id = parent_run_id
        self.resume_from_step = int(resume_from_step)
        self.fsync_every_steps = int(fsync_every_steps)
        self.enabled = bool(enabled)
        self.schema = str(schema)
        self.schema_version = int(schema_version)
        self.supported_record_types = frozenset(
            str(value) for value in supported_record_types
        )
        self._writes_since_fsync = 0
        self._handle = None

        if self.fsync_every_steps <= 0:
            raise ValueError("fsync_every_steps must be positive.")
        if not self.schema or self.schema_version < 1:
            raise ValueError(
                "schema must be non-empty and schema_version must be positive."
            )
        if (
            not self.supported_record_types
            or "run_start" in self.supported_record_types
        ):
            raise ValueError(
                "supported_record_types must be non-empty and exclude run_start."
            )
        if self.enabled and self.path.exists():
            _repair_jsonl_tail_for_append(self.path)
        existing = read_jsonl_tolerant(self.path) if self.path.exists() else []
        # Before the new run_start exists, compute attempts from the requested
        # parent lineage.  A cold run with no parent starts at checkpoint value.
        if parent_run_id is not None:
            self.next_attempt_index = next_attempt_index(
                existing,
                run_id=parent_run_id,
                checkpoint_next_attempt_index=checkpoint_next_attempt_index,
            )
        else:
            maximum = max(
                (
                    int(record["attempt_index"])
                    for record in existing
                    if "attempt_index" in record
                ),
                default=-1,
            )
            self.next_attempt_index = max(
                int(checkpoint_next_attempt_index), maximum + 1
            )

        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self.path.open("a", encoding="utf-8", buffering=1)
            start = {
                "schema": self.schema,
                "schema_version": self.schema_version,
                "record_type": "run_start",
                "experiment_id": self.experiment_id,
                "run_id": self.run_id,
                "parent_run_id": self.parent_run_id,
                "resume_from_step": self.resume_from_step,
                "next_attempt_index": self.next_attempt_index,
                "wall_time_unix": time.time(),
            }
            if run_metadata:
                overlap = set(start).intersection(run_metadata)
                if overlap:
                    raise ValueError(
                        f"run_metadata cannot replace reserved fields: {sorted(overlap)}"
                    )
                start.update(run_metadata)
            self._write(start, force_fsync=True)

    def _write(self, record: Mapping[str, Any], *, force_fsync: bool = False) -> None:
        if not self.enabled:
            return
        assert self._handle is not None
        self._handle.write(_json_line(record))
        self._handle.flush()
        self._writes_since_fsync += 1
        if force_fsync or self._writes_since_fsync >= self.fsync_every_steps:
            os.fsync(self._handle.fileno())
            self._writes_since_fsync = 0

    def append_attempt(self, record_type: str, fields: Mapping[str, Any]) -> int:
        if record_type not in self.supported_record_types:
            raise ValueError(f"Unsupported attempt record_type {record_type!r}.")
        reserved = {
            "schema",
            "schema_version",
            "record_type",
            "experiment_id",
            "run_id",
            "parent_run_id",
            "resume_from_step",
            "attempt_index",
        }
        overlap = reserved.intersection(fields)
        if overlap:
            raise ValueError(
                f"Attempt fields cannot replace reserved fields: {sorted(overlap)}"
            )
        attempt_index = self.next_attempt_index
        record = {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "record_type": record_type,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "resume_from_step": self.resume_from_step,
            "attempt_index": attempt_index,
            **dict(fields),
        }
        # Encode before advancing the counter so non-finite/unsupported values
        # cannot create an unlogged attempt gap.
        _json_line(record)
        self._write(record)
        self.next_attempt_index += 1
        return attempt_index

    # The Stage-2 protocol contains successful substeps, cycle summaries, and
    # checkpoint events.  They are all append-only attempt-sequenced records;
    # this spelling makes call sites read naturally while preserving the
    # Stage-1 ``append_attempt`` API.
    append_record = append_attempt

    def close(self) -> None:
        if self._handle is not None:
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


# Preferred spelling for new code; keep the original class name importable.
JSONLLogger = JsonlLogger
