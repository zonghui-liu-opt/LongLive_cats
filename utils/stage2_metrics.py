"""Typed Stage-2 JSONL metrics built on the shared crash-safe writer."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from collections.abc import Mapping, Sequence

from utils.jsonl_logger import (
    JsonlLogger,
    lineage_run_ids,
    read_jsonl_tolerant,
)

STAGE2_METRICS_SCHEMA = "longlive_stage2_metrics/v1"
STAGE2_METRICS_SCHEMA_VERSION = 1
STAGE2_METRIC_RECORD_TYPES = frozenset(
    {
        "train_step",
        "cycle_summary",
        "nonfinite_attempt",
        "checkpoint_event",
        "run_end",
    }
)

# These names are the producer/plotter contract.  Keeping the contract here
# prevents the two sides from silently drifting (for example ``pre_clip`` vs
# ``preclip`` or bytes vs GiB).
STAGE2_TIMING_FIELDS = (
    "data_seconds_max",
    "h2d_seconds_max",
    "rollout_seconds_max",
    "fake_score_seconds_max",
    "real_cond_seconds_max",
    "real_uncond_seconds_max",
    "loss_build_seconds_max",
    "backward_seconds_max",
    "clip_optimizer_seconds_max",
    "ema_seconds_max",
)
STAGE2_MEMORY_FIELDS = (
    "gpu_memory_allocated_gib_max",
    "gpu_memory_reserved_gib_max",
    "gpu_memory_free_gib_min",
)
STAGE2_BASE_THROUGHPUT_FIELDS = (
    "samples_per_second",
    "generated_latents_per_second",
    "generator_forward_calls",
    "generator_logical_query_tokens",
    "generator_tokens_per_second",
)
STAGE2_ROLE_THROUGHPUT_FIELDS = {
    "fake_score": (
        "fake_score_forward_calls",
        "fake_score_logical_query_tokens",
        "fake_score_tokens_per_second",
    ),
    "generator": (
        "fake_score_forward_calls",
        "fake_score_logical_query_tokens",
        "fake_score_tokens_per_second",
        "real_score_forward_calls",
        "real_score_logical_query_tokens",
        "real_score_tokens_per_second",
    ),
}


def _plain_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}.")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite, got {value!r}.")
    return result


def _nonnegative_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result < 0.0:
        raise ValueError(f"{label} must be non-negative, got {value!r}.")
    return result


def _positive_number(value: Any, label: str) -> float:
    result = _finite_number(value, label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive, got {value!r}.")
    return result


def _required(fields: Mapping[str, Any], key: str, *, role: str | None = None) -> Any:
    if key not in fields:
        prefix = f"{role} train_step" if role is not None else "Stage-2 metric"
        raise ValueError(f"{prefix} is missing required field {key!r}.")
    return fields[key]


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} must be a non-empty string, got {value!r}.")
    return value


def _validate_histogram(value: Any, label: str) -> int:
    if not isinstance(value, Mapping) or not value:
        raise TypeError(f"{label} must be a non-empty object.")
    total = 0
    for key, count in value.items():
        _nonempty_string(key, f"{label} key")
        total += _plain_nonnegative_int(count, f"{label}[{key!r}]")
    if total <= 0:
        raise ValueError(f"{label} must contain at least one observation.")
    return total


def _validate_active_workload(
    fields: Mapping[str, Any], *, prefix: str, role: str
) -> None:
    calls = _plain_nonnegative_int(
        _required(fields, f"{prefix}_forward_calls", role=role),
        f"{prefix}_forward_calls",
    )
    tokens = _plain_nonnegative_int(
        _required(fields, f"{prefix}_logical_query_tokens", role=role),
        f"{prefix}_logical_query_tokens",
    )
    rate = _nonnegative_number(
        _required(fields, f"{prefix}_tokens_per_second", role=role),
        f"{prefix}_tokens_per_second",
    )
    if calls == 0 or tokens == 0 or rate == 0.0:
        raise ValueError(
            f"{role} train_step requires positive {prefix} calls, logical tokens, "
            "and tokens/s."
        )
    wall_seconds = _positive_number(fields.get("step_seconds_max"), "step_seconds_max")
    expected_rate = tokens / wall_seconds
    if not math.isclose(rate, expected_rate, rel_tol=2.0e-6, abs_tol=1.0e-9):
        raise ValueError(
            f"{prefix}_tokens_per_second must equal logical query tokens divided "
            f"by step_seconds_max: {rate} != {expected_rate}."
        )


def _validate_timing(fields: Mapping[str, Any], *, role: str) -> None:
    maximum = _positive_number(
        _required(fields, "step_seconds_max", role=role), "step_seconds_max"
    )
    mean = _positive_number(
        _required(fields, "step_seconds_mean", role=role), "step_seconds_mean"
    )
    if maximum + 1.0e-12 < mean:
        raise ValueError("step_seconds_max must be >= step_seconds_mean.")
    ratio = _positive_number(
        _required(fields, "straggler_ratio", role=role), "straggler_ratio"
    )
    expected_ratio = maximum / mean
    if not math.isclose(ratio, expected_ratio, rel_tol=2.0e-6, abs_tol=1.0e-9):
        raise ValueError(
            "straggler_ratio must equal step_seconds_max/step_seconds_mean: "
            f"{ratio} != {expected_ratio}."
        )
    parts = [
        _nonnegative_number(_required(fields, key, role=role), key)
        for key in STAGE2_TIMING_FIELDS
    ]
    closure = _finite_number(
        _required(fields, "timing_closure_error_seconds", role=role),
        "timing_closure_error_seconds",
    )
    if not math.isclose(
        sum(parts) + closure,
        maximum,
        rel_tol=2.0e-6,
        abs_tol=max(1.0e-8, maximum * 2.0e-6),
    ):
        raise ValueError(
            "detailed Stage-2 timing plus timing_closure_error_seconds must "
            "close to step_seconds_max."
        )


def _validate_memory(fields: Mapping[str, Any], *, role: str) -> None:
    allocated, reserved, _free = (
        _nonnegative_number(_required(fields, key, role=role), key)
        for key in STAGE2_MEMORY_FIELDS
    )
    if reserved + 1.0e-9 < allocated:
        raise ValueError(
            "gpu_memory_reserved_gib_max must be >= " "gpu_memory_allocated_gib_max."
        )


def _validate_throughput(fields: Mapping[str, Any], *, role: str) -> None:
    _positive_number(
        _required(fields, "samples_per_second", role=role), "samples_per_second"
    )
    _positive_number(
        _required(fields, "generated_latents_per_second", role=role),
        "generated_latents_per_second",
    )
    _validate_active_workload(fields, prefix="generator", role=role)
    _validate_active_workload(fields, prefix="fake_score", role=role)
    if role == "generator":
        _validate_active_workload(fields, prefix="real_score", role=role)


def validate_stage2_clock(
    fields: Mapping[str, Any], *, committed: bool, record_type: str
) -> None:
    """Validate the two optimizer clocks and the strict 5F -> 1G cursor."""

    fake = _plain_nonnegative_int(
        fields.get("completed_fake_updates"), "completed_fake_updates"
    )
    generator = _plain_nonnegative_int(
        fields.get("completed_generator_updates"),
        "completed_generator_updates",
    )
    cycles = _plain_nonnegative_int(fields.get("completed_cycles"), "completed_cycles")
    substep = fields.get("cycle_substep")
    if substep not in {"F1", "F2", "F3", "F4", "F5", "G"}:
        raise ValueError("cycle_substep must be one of F1..F5 or G.")
    if cycles != generator:
        raise ValueError("completed_cycles must equal completed_generator_updates.")
    if committed:
        expected_fake = (
            5 * generator + int(substep[1:])
            if str(substep).startswith("F")
            else 5 * generator
        )
    else:
        expected_fake = (
            5 * generator + int(substep[1:]) - 1
            if str(substep).startswith("F")
            else 5 * generator + 5
        )
    if fake != expected_fake:
        raise ValueError(
            "Stage-2 clock violates the strict 5F->1G invariant: "
            f"record_type={record_type}, F={fake}, G={generator}, "
            f"cycle_substep={substep}."
        )
    logical = _plain_nonnegative_int(
        fields.get("logical_substep_id"), "logical_substep_id"
    )
    expected = fake + generator - (1 if committed else 0)
    if logical != expected:
        raise ValueError(
            f"logical_substep_id mismatch: expected={expected}, actual={logical}."
        )


def _validate_train_step(fields: Mapping[str, Any]) -> None:
    validate_stage2_clock(fields, committed=True, record_type="train_step")
    role = fields.get("role")
    substep = str(fields["cycle_substep"])
    if role == "fake_score":
        if not substep.startswith("F"):
            raise ValueError(
                "a successful fake_score step must use cycle_substep F1..F5."
            )
    elif role == "generator":
        if substep != "G":
            raise ValueError("a successful generator step must use cycle_substep G.")
    else:
        raise ValueError("Stage-2 train_step role must be fake_score or generator.")
    numerator = _finite_number(
        _required(fields, "loss_numerator", role=role), "loss_numerator"
    )
    count = _plain_nonnegative_int(
        _required(fields, "loss_count", role=role), "loss_count"
    )
    if count < 1:
        raise ValueError("loss_count must be positive.")
    loss = _finite_number(_required(fields, "loss", role=role), "loss")
    expected_loss = numerator / count
    if not math.isclose(loss, expected_loss, rel_tol=2.0e-6, abs_tol=1.0e-12):
        raise ValueError(
            f"loss must equal global numerator/count: {loss} != {expected_loss}."
        )
    _nonempty_string(_required(fields, "phase", role=role), "phase")
    _nonnegative_number(
        _required(fields, "preclip_grad_norm", role=role), "preclip_grad_norm"
    )
    _positive_number(_required(fields, "learning_rate", role=role), "learning_rate")
    timestep_min = _finite_number(
        _required(fields, "score_timestep_min", role=role), "score_timestep_min"
    )
    timestep_mean = _finite_number(
        _required(fields, "score_timestep_mean", role=role), "score_timestep_mean"
    )
    timestep_max = _finite_number(
        _required(fields, "score_timestep_max", role=role), "score_timestep_max"
    )
    if not timestep_min <= timestep_mean <= timestep_max:
        raise ValueError("score timestep summary must satisfy min <= mean <= max.")
    _validate_histogram(
        _required(fields, "score_timestep_histogram", role=role),
        "score_timestep_histogram",
    )
    edge_mass = _nonnegative_number(
        _required(fields, "score_timestep_edge_mass", role=role),
        "score_timestep_edge_mass",
    )
    if edge_mass > 1.0:
        raise ValueError("score_timestep_edge_mass must be in [0, 1].")
    _nonnegative_number(
        _required(fields, "exit_step_mean", role=role), "exit_step_mean"
    )
    _validate_histogram(
        _required(fields, "exit_histogram", role=role), "exit_histogram"
    )
    _validate_timing(fields, role=role)
    _validate_memory(fields, role=role)
    _validate_throughput(fields, role=role)

    branch = _nonempty_string(_required(fields, "branch", role=role), "branch")
    probability = _nonnegative_number(
        _required(fields, "dfd_probability", role=role), "dfd_probability"
    )
    if probability > 1.0:
        raise ValueError("dfd_probability must be in [0, 1].")
    branch_is_dfd = _plain_nonnegative_int(
        _required(fields, "branch_is_dfd", role=role), "branch_is_dfd"
    )
    if branch_is_dfd not in (0, 1):
        raise ValueError("branch_is_dfd must be 0 or 1.")
    if role == "fake_score":
        if branch != "flow_dsm" or probability != 0.0 or branch_is_dfd != 0:
            raise ValueError(
                "fake_score train_step must use flow_dsm with zero DFD selection."
            )
        for key in ("target_flow_l2", "prediction_l2"):
            _nonnegative_number(_required(fields, key, role=role), key)
    else:
        if branch not in {"dmd", "dfd"}:
            raise ValueError("generator branch must be dmd or dfd.")
        if branch_is_dfd != int(branch == "dfd"):
            raise ValueError("branch_is_dfd must agree with generator branch.")
        # The reducer clamps the denominator for numerical stability, but the
        # diagnostic intentionally records its raw (pre-clamp) value.  A raw
        # zero is mathematically valid and must not make logging fail after an
        # optimizer/EMA commit.
        denominator_min = _nonnegative_number(
            _required(fields, "denominator_min", role=role), "denominator_min"
        )
        denominator_mean = _nonnegative_number(
            _required(fields, "denominator_mean", role=role), "denominator_mean"
        )
        denominator_max = _nonnegative_number(
            _required(fields, "denominator_max", role=role), "denominator_max"
        )
        if not denominator_min <= denominator_mean <= denominator_max:
            raise ValueError("denominator summary must satisfy min <= mean <= max.")
        for key in ("raw_score_difference_l2", "fake_x0_l2", "real_x0_l2"):
            _nonnegative_number(_required(fields, key, role=role), key)
        _nonempty_string(_required(fields, "ema_action", role=role), "ema_action")
    if not isinstance(_required(fields, "dry_run", role=role), bool):
        raise TypeError("dry_run must be bool.")


def _validate_cycle_summary(fields: Mapping[str, Any]) -> None:
    validate_stage2_clock(fields, committed=True, record_type="cycle_summary")
    if fields["cycle_substep"] != "G":
        raise ValueError("cycle_summary is valid only after G commits a cycle.")
    cycle = _positive_number(fields.get("cycle_seconds"), "cycle_seconds")
    fake = _positive_number(fields.get("fake_score_seconds"), "fake_score_seconds")
    generator = _positive_number(fields.get("generator_seconds"), "generator_seconds")
    if not math.isclose(cycle, fake + generator, rel_tol=2.0e-6, abs_tol=1.0e-8):
        raise ValueError(
            "cycle_seconds must equal fake_score_seconds + generator_seconds."
        )
    fake_fraction = _nonnegative_number(
        fields.get("fake_score_time_fraction"), "fake_score_time_fraction"
    )
    generator_fraction = _nonnegative_number(
        fields.get("generator_time_fraction"), "generator_time_fraction"
    )
    if not math.isclose(fake_fraction + generator_fraction, 1.0, abs_tol=2.0e-6):
        raise ValueError("cycle role time fractions must sum to one.")
    for key in (
        "cycle_samples_per_second",
        "fake_score_samples_per_second",
        "generator_samples_per_second",
    ):
        _positive_number(fields.get(key), key)
    if (
        _plain_nonnegative_int(fields.get("successful_substeps"), "successful_substeps")
        != 6
    ):
        raise ValueError("cycle_summary.successful_substeps must equal 6.")
    _plain_nonnegative_int(fields.get("nonfinite_attempts"), "nonfinite_attempts")
    if not isinstance(fields.get("dry_run"), bool):
        raise TypeError("cycle_summary.dry_run must be bool.")


def validate_stage2_metric_record(record_type: str, fields: Mapping[str, Any]) -> None:
    if record_type not in STAGE2_METRIC_RECORD_TYPES:
        raise ValueError(f"Unsupported Stage-2 metric record_type {record_type!r}.")
    if record_type == "train_step":
        _validate_train_step(fields)
    elif record_type == "nonfinite_attempt":
        validate_stage2_clock(fields, committed=False, record_type="nonfinite_attempt")
        attempt = _plain_nonnegative_int(
            fields.get("attempt_number_for_substep"),
            "attempt_number_for_substep",
        )
        if attempt not in (1, 2):
            raise ValueError("attempt_number_for_substep must be 1 or 2.")
        if fields.get("skipped") is not True or fields.get("nonfinite") is not True:
            raise ValueError("nonfinite_attempt must be marked skipped and nonfinite.")
        role = fields.get("role")
        substep = str(fields["cycle_substep"])
        if role not in {"fake_score", "generator"}:
            raise ValueError("nonfinite_attempt.role must be fake_score or generator.")
        if (role == "fake_score") != substep.startswith("F"):
            raise ValueError("nonfinite_attempt role must agree with cycle_substep.")
        _nonempty_string(fields.get("batch_identity"), "batch_identity")
        _nonempty_string(fields.get("reason"), "reason")
        _nonnegative_number(fields.get("elapsed_seconds"), "elapsed_seconds")
        _validate_memory(fields, role=str(role))
        if not isinstance(fields.get("dry_run"), bool):
            raise TypeError("nonfinite_attempt.dry_run must be bool.")
    elif record_type == "cycle_summary":
        _validate_cycle_summary(fields)
    elif record_type == "checkpoint_event":
        validate_stage2_clock(fields, committed=True, record_type="checkpoint_event")
        if fields["cycle_substep"] != "G":
            raise ValueError("checkpoint_event is valid only after G commits a cycle.")
        _nonempty_string(fields.get("checkpoint_path"), "checkpoint_path")
        _nonempty_string(fields.get("checkpoint_sha256"), "checkpoint_sha256")
        _plain_nonnegative_int(fields.get("checkpoint_bytes"), "checkpoint_bytes")
        _nonnegative_number(fields.get("elapsed_seconds"), "elapsed_seconds")
        if fields.get("success_marker") is not True:
            raise ValueError("checkpoint_event.success_marker must be true.")
        if not isinstance(fields.get("dry_run"), bool):
            raise TypeError("checkpoint_event.dry_run must be bool.")
    elif record_type == "run_end":
        validate_stage2_clock(fields, committed=True, record_type="run_end")
        if fields["cycle_substep"] != "G":
            raise ValueError("run_end is valid only after G commits a cycle.")
        status = fields.get("status")
        if status not in {"complete", "smoke_complete", "interrupted", "failed"}:
            raise ValueError("run_end.status is invalid.")
        if not isinstance(fields.get("dry_run"), bool):
            raise TypeError("run_end.dry_run must be bool.")
        if status == "complete" and fields["dry_run"]:
            raise ValueError("a dry-run cannot claim run_end.status=complete.")
        if status == "smoke_complete" and not fields["dry_run"]:
            raise ValueError("run_end.status=smoke_complete requires dry_run=true.")


class Stage2MetricsLogger:
    """Rank-zero Stage-2 writer with validated dual-clock records."""

    def __init__(
        self,
        path: str | Path,
        *,
        experiment_id: str,
        run_id: str | None = None,
        parent_run_id: str | None = None,
        resume_from_logical_substep: int = 0,
        checkpoint_next_attempt_index: int = 0,
        fsync_every_steps: int = 1,
        enabled: bool = True,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._writer = JsonlLogger(
            path,
            experiment_id=experiment_id,
            run_id=run_id,
            parent_run_id=parent_run_id,
            resume_from_step=resume_from_logical_substep,
            checkpoint_next_attempt_index=checkpoint_next_attempt_index,
            fsync_every_steps=fsync_every_steps,
            enabled=enabled,
            run_metadata=run_metadata,
            schema=STAGE2_METRICS_SCHEMA,
            schema_version=STAGE2_METRICS_SCHEMA_VERSION,
            supported_record_types=STAGE2_METRIC_RECORD_TYPES,
        )

    @property
    def run_id(self) -> str:
        return self._writer.run_id

    @property
    def next_attempt_index(self) -> int:
        return self._writer.next_attempt_index

    def append(self, record_type: str, fields: Mapping[str, Any]) -> int:
        validate_stage2_metric_record(record_type, fields)
        return self._writer.append_record(record_type, fields)

    def close(self) -> None:
        self._writer.close()

    def __enter__(self) -> "Stage2MetricsLogger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def stage2_records_for_latest_lineage(
    records: Sequence[Mapping[str, Any]],
    *,
    record_type: str = "train_step",
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    ordered_lineage = lineage_run_ids(records, run_id=run_id)
    lineage = set(ordered_lineage)
    for record in records:
        if str(record.get("run_id", "")) not in lineage:
            continue
        if (
            record.get("schema") != STAGE2_METRICS_SCHEMA
            or int(record.get("schema_version", -1)) != STAGE2_METRICS_SCHEMA_VERSION
        ):
            raise ValueError("latest lineage contains a non-Stage-2 metric schema.")
    # At each child run_start, everything in the ancestor at or after the
    # child's next logical substep is an uncommitted stale suffix.  Remove that
    # suffix even when the child has no record of the same optional type (for
    # example, a parent nonfinite marker followed by a clean child replay).
    run_starts: dict[str, Mapping[str, Any]] = {}
    for record in records:
        owner = str(record.get("run_id", ""))
        if owner in lineage and record.get("record_type") == "run_start":
            if owner in run_starts:
                raise ValueError(f"Stage-2 run {owner} has multiple run_start records.")
            run_starts[owner] = record

    # ``attempt_index`` is intentionally not the identity: it remains globally
    # monotonic while ``logical_substep_id`` names the resumable logical point.
    selected: dict[int | tuple[int, int], tuple[int, int, dict[str, Any]]] = {}
    for depth, owner in enumerate(ordered_lineage):
        if owner not in run_starts:
            raise ValueError(f"Stage-2 lineage run {owner} has no run_start record.")
        if depth:
            boundary = _plain_nonnegative_int(
                run_starts[owner].get("resume_from_step"),
                f"run_start[{owner}].resume_from_step",
            )
            selected = {
                identity: value
                for identity, value in selected.items()
                if (identity[0] if isinstance(identity, tuple) else identity) < boundary
            }
        for sequence, record in enumerate(records):
            if (
                str(record.get("run_id", "")) == owner
                and record.get("record_type") == record_type
            ):
                if "logical_substep_id" not in record:
                    raise ValueError(
                        f"{record_type} record in run {owner} has no logical_substep_id."
                    )
                logical = int(record["logical_substep_id"])
                # Two failed attempts at the same logical substep are distinct
                # audit records.  A resumed child still replaces a stale
                # parent attempt with the same logical-id/attempt-number pair.
                identity: int | tuple[int, int]
                if record_type == "nonfinite_attempt":
                    identity = (logical, int(record["attempt_number_for_substep"]))
                else:
                    identity = logical
                selected[identity] = (depth, sequence, dict(record))
    return [selected[key][2] for key in sorted(selected)]


def load_stage2_metrics(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl_tolerant(path)


__all__ = [
    "STAGE2_BASE_THROUGHPUT_FIELDS",
    "STAGE2_MEMORY_FIELDS",
    "STAGE2_METRICS_SCHEMA",
    "STAGE2_METRICS_SCHEMA_VERSION",
    "STAGE2_ROLE_THROUGHPUT_FIELDS",
    "STAGE2_TIMING_FIELDS",
    "Stage2MetricsLogger",
    "load_stage2_metrics",
    "stage2_records_for_latest_lineage",
    "validate_stage2_clock",
    "validate_stage2_metric_record",
]
