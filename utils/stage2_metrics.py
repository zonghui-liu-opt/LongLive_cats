"""Typed Stage-2 JSONL metrics built on the shared crash-safe writer."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from utils.jsonl_logger import (
    JsonlLogger,
    JsonlSnapshot,
    lineage_run_ids,
    read_jsonl_snapshot,
    read_jsonl_tolerant,
)

STAGE2_METRICS_SCHEMA = "longlive_stage2_metrics/v1"
STAGE2_METRICS_SCHEMA_VERSION = 1
STAGE2_TIMING_CLOSURE_ABS_SECONDS = 0.1
STAGE2_TIMING_CLOSURE_REL_FRACTION = 0.05
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
    "orchestration_seconds_max",
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


def _canonical_sha256(value: Any) -> str:
    def semantic_json_value(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {key: semantic_json_value(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [semantic_json_value(child) for child in item]
        if isinstance(item, float) and item.is_integer():
            return int(item)
        return item

    payload = json.dumps(
        semantic_json_value(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object.")
    return value


def _sha256_string(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest.")
    return value


def _derived_integer(derived: Mapping[str, Any], key: str) -> int:
    if key not in derived:
        raise ValueError(f"resolved_config.derived is missing {key!r}.")
    return _plain_nonnegative_int(derived[key], f"resolved_config.derived.{key}")


def validate_stage2_run_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate redundant run metadata against authoritative resolved config.

    The resolved launch payload is the source of truth.  Human-friendly
    terminal, phase, topology, and workload summaries are retained in JSONL,
    but they must be exact projections of that payload rather than a second
    independently editable contract.
    """

    metadata = _mapping(metadata, "Stage-2 run metadata")
    resolved = _mapping(metadata.get("resolved_config"), "resolved_config")
    _mapping(resolved.get("config"), "resolved_config.config")
    derived = _mapping(resolved.get("derived"), "resolved_config.derived")

    if derived.get("metrics_schema") != STAGE2_METRICS_SCHEMA:
        raise ValueError(
            "resolved_config.derived.metrics_schema must equal "
            f"{STAGE2_METRICS_SCHEMA!r}."
        )

    expected_world_size = _derived_integer(derived, "expected_world_size")
    fake_per_generator = _derived_integer(derived, "fake_updates_per_generator_update")
    phase_a_generator = _derived_integer(derived, "phase_a_generator_updates")
    phase_a_fake = _derived_integer(derived, "phase_a_fake_updates")
    phase_b_generator = _derived_integer(derived, "phase_b_generator_updates")
    phase_b_fake = _derived_integer(derived, "phase_b_fake_updates")
    total_fake = _derived_integer(derived, "total_fake_updates")
    total_generator = _derived_integer(derived, "total_generator_updates")
    total_cycles = _derived_integer(derived, "total_cycles")
    if fake_per_generator != 5:
        raise ValueError(
            "resolved_config must encode exactly five fake-score updates per "
            "generator update."
        )
    if (
        phase_a_fake != 5 * phase_a_generator
        or phase_b_fake != 5 * phase_b_generator
        or total_generator != phase_a_generator + phase_b_generator
        or total_fake != phase_a_fake + phase_b_fake
        or total_cycles != total_generator
    ):
        raise ValueError(
            "resolved_config Stage-2 phase/terminal arithmetic is invalid."
        )

    world_size = _plain_nonnegative_int(metadata.get("world_size"), "world_size")
    if world_size != expected_world_size:
        raise ValueError("world_size disagrees with resolved_config.derived.")

    topology = _mapping(metadata.get("topology"), "topology")
    topology_projection = {
        key: derived.get(key)
        for key in (
            "fsdp_backend",
            "sharding_strategy",
            "microbatch_size_per_device",
            "gradient_accumulation_steps",
            "global_batch_size",
        )
    }
    if dict(topology) != topology_projection:
        raise ValueError("topology disagrees with resolved_config.derived.")

    role_hashes = _mapping(metadata.get("role_hashes"), "role_hashes")
    expected_roles = {"generator", "real_score", "fake_score"}
    if set(role_hashes) != expected_roles:
        raise ValueError(
            "role_hashes must contain exactly generator, real_score, and fake_score."
        )
    for role, digest in role_hashes.items():
        _sha256_string(digest, f"role_hashes.{role}")

    terminal = _mapping(metadata.get("terminal_counts"), "terminal_counts")
    expected_terminal = {
        "fake_score_updates": total_fake,
        "generator_updates": total_generator,
        "cycles": total_cycles,
    }
    if dict(terminal) != expected_terminal:
        raise ValueError("terminal_counts disagree with resolved_config.derived.")

    boundaries = metadata.get("phase_boundaries")
    if not isinstance(boundaries, Sequence) or isinstance(boundaries, (str, bytes)):
        raise TypeError("phase_boundaries must be a list.")
    expected_boundaries = [
        {"label": "Phase A end", "generator_update": phase_a_generator}
    ]
    if phase_b_generator:
        expected_boundaries.append(
            {"label": "Phase B end", "generator_update": total_generator}
        )
    if list(boundaries) != expected_boundaries:
        raise ValueError("phase_boundaries disagree with resolved_config.derived.")

    workload = _mapping(metadata.get("workload"), "workload")
    patch_tokens = _derived_integer(derived, "patch_tokens_per_frame")
    score_frames = _derived_integer(derived, "score_input_frames")
    future_frames = _derived_integer(derived, "future_latent_frames")
    sink_frames = _derived_integer(derived, "global_sink_frames")
    generated_frames = _derived_integer(derived, "generated_episode_frames")
    expected_workload = {
        "patch_tokens_per_frame": patch_tokens,
        "score_input_frames": score_frames,
        "loss_future_frames": future_frames,
        "sink_in_score_compute": True,
        "sink_in_loss": False,
    }
    if dict(workload) != expected_workload:
        raise ValueError("workload disagrees with resolved_config.derived.")
    if (
        score_frames != sink_frames + generated_frames
        or future_frames != generated_frames
    ):
        raise ValueError("resolved_config workload frame accounting is inconsistent.")

    if not isinstance(metadata.get("dry_run"), bool):
        raise TypeError("run metadata dry_run must be bool.")
    smoke_mode = metadata.get("smoke_mode")
    if smoke_mode not in {None, "C0", "C1", "C2"}:
        raise ValueError("run metadata smoke_mode must be null or C0/C1/C2.")

    _sha256_string(metadata.get("config_contract_sha256"), "config_contract_sha256")
    launch_hash = _sha256_string(
        metadata.get("config_launch_sha256"), "config_launch_sha256"
    )
    expected_launch_hash = _canonical_sha256(resolved)
    if launch_hash != expected_launch_hash:
        raise ValueError(
            "config_launch_sha256 does not fingerprint resolved_config exactly."
        )
    return dict(derived)


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
    closure_limit = max(
        STAGE2_TIMING_CLOSURE_ABS_SECONDS,
        STAGE2_TIMING_CLOSURE_REL_FRACTION * maximum,
    )
    if abs(closure) > closure_limit + 1.0e-12:
        raise ValueError(
            "timing closure error exceeds max(0.1 seconds, "
            "0.05 * step_seconds_max): "
            f"abs({closure}) > {closure_limit}."
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


def validate_stage2_run_start(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one persisted Stage-2 ``run_start`` and return its derived data."""

    if record.get("record_type") != "run_start":
        raise ValueError("Stage-2 run metadata must come from a run_start record.")
    if record.get("schema") != STAGE2_METRICS_SCHEMA:
        raise ValueError("run_start does not use the Stage-2 metric schema.")
    if record.get("schema_version") != STAGE2_METRICS_SCHEMA_VERSION:
        raise ValueError("run_start uses an unsupported Stage-2 schema_version.")
    _nonempty_string(record.get("run_id"), "run_start.run_id")
    _plain_nonnegative_int(record.get("resume_from_step"), "run_start.resume_from_step")
    return validate_stage2_run_metadata(record)


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
        resume_boundary = _plain_nonnegative_int(
            resume_from_logical_substep, "resume_from_logical_substep"
        )
        if parent_run_id is None:
            if resume_boundary != 0:
                raise ValueError(
                    "a cold Stage-2 run must start at logical_substep_id 0."
                )
        elif resume_boundary % 6:
            raise ValueError(
                "a resumed Stage-2 child must start at a complete 5F -> 1G "
                "cycle boundary (a multiple of six logical substeps)."
            )
        metadata = _mapping(run_metadata, "Stage-2 run metadata")
        derived = validate_stage2_run_metadata(metadata)
        self._resume_boundary = resume_boundary
        self._next_train_logical_substep = resume_boundary
        self._dry_run = bool(metadata["dry_run"])
        self._terminal_counts = {
            "completed_fake_updates": int(derived["total_fake_updates"]),
            "completed_generator_updates": int(derived["total_generator_updates"]),
            "completed_cycles": int(derived["total_cycles"]),
        }
        self._terminal_logical_substeps = (
            self._terminal_counts["completed_fake_updates"]
            + self._terminal_counts["completed_generator_updates"]
        )
        self._writer = JsonlLogger(
            path,
            experiment_id=experiment_id,
            run_id=run_id,
            parent_run_id=parent_run_id,
            resume_from_step=resume_boundary,
            checkpoint_next_attempt_index=checkpoint_next_attempt_index,
            fsync_every_steps=fsync_every_steps,
            enabled=enabled,
            run_metadata=metadata,
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
        logical = _plain_nonnegative_int(
            fields.get("logical_substep_id"), "logical_substep_id"
        )
        terminal_no_work_finalize = (
            record_type == "run_end"
            and fields.get("status") == "complete"
            and fields.get("dry_run") is False
            and self._resume_boundary == self._terminal_logical_substeps
            and self._next_train_logical_substep == self._resume_boundary
            and logical == self._resume_boundary - 1
            and all(
                fields.get(key) == expected
                for key, expected in self._terminal_counts.items()
            )
        )
        if logical < self._resume_boundary and not terminal_no_work_finalize:
            raise ValueError(
                f"record logical_substep_id={logical} is before this run's resume "
                f"boundary {self._resume_boundary}."
            )
        if fields.get("dry_run") is not self._dry_run:
            raise ValueError("record dry_run disagrees with run_start metadata.")
        if record_type == "train_step" and logical != self._next_train_logical_substep:
            raise ValueError(
                "train_step must use the next logical_substep_id in the contiguous "
                "F1..F5 -> G sequence: "
                f"expected={self._next_train_logical_substep}, actual={logical}."
            )
        if (
            record_type == "nonfinite_attempt"
            and logical != self._next_train_logical_substep
        ):
            raise ValueError(
                "nonfinite_attempt must target the next logical_substep_id: "
                f"expected={self._next_train_logical_substep}, actual={logical}."
            )
        attempt_index = self._writer.append_record(record_type, fields)
        if record_type == "train_step":
            self._next_train_logical_substep += 1
        return attempt_index

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

    semantic_projection: dict[str, Any] | None = None
    for depth, owner in enumerate(ordered_lineage):
        if owner not in run_starts:
            raise ValueError(f"Stage-2 lineage run {owner} has no run_start record.")
        start = run_starts[owner]
        derived = validate_stage2_run_start(start)
        boundary = _plain_nonnegative_int(
            start.get("resume_from_step"), f"run_start[{owner}].resume_from_step"
        )
        if depth == 0:
            if start.get("parent_run_id") not in (None, ""):
                raise ValueError("Stage-2 lineage root unexpectedly declares a parent.")
            if boundary != 0:
                raise ValueError(
                    "Stage-2 lineage root must begin at logical substep 0."
                )
        elif boundary % 6:
            raise ValueError(
                "a resumed Stage-2 child must start at a complete 5F -> 1G "
                "cycle boundary."
            )

        projection = {
            "config_contract_sha256": start.get("config_contract_sha256"),
            "world_size": start.get("world_size"),
            "topology": start.get("topology"),
            "role_hashes": start.get("role_hashes"),
            "terminal_counts": start.get("terminal_counts"),
            "phase_boundaries": start.get("phase_boundaries"),
            "workload": start.get("workload"),
            "metrics_schema": derived.get("metrics_schema"),
        }
        if semantic_projection is None:
            semantic_projection = projection
        elif projection != semantic_projection:
            raise ValueError(
                "Stage-2 child run changes the parent lineage's resolved training "
                "contract."
            )

        parent_id = start.get("parent_run_id")
        expected_parent = None if depth == 0 else ordered_lineage[depth - 1]
        normalized_parent = None if parent_id in (None, "") else str(parent_id)
        if normalized_parent != expected_parent:
            raise ValueError(f"Stage-2 run {owner} declares the wrong parent_run_id.")

        terminal_counts = {
            "completed_fake_updates": int(derived["total_fake_updates"]),
            "completed_generator_updates": int(derived["total_generator_updates"]),
            "completed_cycles": int(derived["total_cycles"]),
        }
        terminal_logical_substeps = (
            terminal_counts["completed_fake_updates"]
            + terminal_counts["completed_generator_updates"]
        )
        owner_has_train_steps_at_or_after_boundary = any(
            str(candidate.get("run_id", "")) == owner
            and candidate.get("record_type") == "train_step"
            and isinstance(candidate.get("logical_substep_id"), int)
            and not isinstance(candidate.get("logical_substep_id"), bool)
            and int(candidate["logical_substep_id"]) >= boundary
            for candidate in records
        )

        for record in records:
            if str(record.get("run_id", "")) != owner:
                continue
            current_type = record.get("record_type")
            if current_type == "run_start":
                continue
            if current_type not in STAGE2_METRIC_RECORD_TYPES:
                raise ValueError(
                    f"Stage-2 lineage contains unsupported record_type {current_type!r}."
                )
            if (
                record.get("schema") != STAGE2_METRICS_SCHEMA
                or record.get("schema_version") != STAGE2_METRICS_SCHEMA_VERSION
            ):
                raise ValueError("latest lineage contains a non-Stage-2 metric schema.")
            if record.get("parent_run_id") != start.get("parent_run_id"):
                raise ValueError(
                    f"Stage-2 record in run {owner} has wrong parent_run_id."
                )
            if record.get("resume_from_step") != boundary:
                raise ValueError(
                    f"Stage-2 record in run {owner} has wrong resume_from_step."
                )
            logical = _plain_nonnegative_int(
                record.get("logical_substep_id"),
                f"{current_type}.logical_substep_id",
            )
            terminal_no_work_finalize = (
                current_type == "run_end"
                and record.get("status") == "complete"
                and record.get("dry_run") is False
                and start.get("dry_run") is False
                and boundary == terminal_logical_substeps
                and logical == boundary - 1
                and not owner_has_train_steps_at_or_after_boundary
                and all(
                    record.get(key) == expected
                    for key, expected in terminal_counts.items()
                )
            )
            if logical < boundary and not terminal_no_work_finalize:
                raise ValueError(
                    f"Stage-2 child record in run {owner} appears before its resume "
                    f"boundary {boundary}."
                )
            if record.get("dry_run") is not start.get("dry_run"):
                raise ValueError(
                    f"Stage-2 record in run {owner} disagrees with run_start dry_run."
                )
            validate_stage2_metric_record(str(current_type), record)

    # ``attempt_index`` is intentionally not the identity: it remains globally
    # monotonic while ``logical_substep_id`` names the resumable logical point.
    selected: dict[int | tuple[int, int], tuple[int, int, dict[str, Any]]] = {}
    for depth, owner in enumerate(ordered_lineage):
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
        owner_identities: set[int | tuple[int, int]] = set()
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
                if identity in owner_identities:
                    raise ValueError(
                        f"Stage-2 run {owner} repeats {record_type} identity "
                        f"{identity!r}."
                    )
                owner_identities.add(identity)
                selected[identity] = (depth, sequence, dict(record))
    result = [selected[key][2] for key in sorted(selected)]
    if record_type == "train_step":
        logical_ids = [int(record["logical_substep_id"]) for record in result]
        if logical_ids != list(range(len(logical_ids))):
            raise ValueError(
                "Stage-2 train_step lineage must be one contiguous "
                "F1..F5 -> G sequence starting at logical substep 0."
            )
    return result


def load_stage2_metrics(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl_tolerant(path)


def load_stage2_metrics_snapshot(path: str | Path) -> JsonlSnapshot:
    """Load records and SHA-256 from one Stage-2 JSONL byte snapshot."""

    return read_jsonl_snapshot(path)


__all__ = [
    "STAGE2_BASE_THROUGHPUT_FIELDS",
    "STAGE2_MEMORY_FIELDS",
    "STAGE2_METRICS_SCHEMA",
    "STAGE2_METRICS_SCHEMA_VERSION",
    "STAGE2_ROLE_THROUGHPUT_FIELDS",
    "STAGE2_TIMING_FIELDS",
    "Stage2MetricsLogger",
    "load_stage2_metrics",
    "load_stage2_metrics_snapshot",
    "stage2_records_for_latest_lineage",
    "validate_stage2_clock",
    "validate_stage2_metric_record",
    "validate_stage2_run_metadata",
    "validate_stage2_run_start",
]
