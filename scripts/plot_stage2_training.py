#!/usr/bin/env python3
"""Render authoritative Stage-2 JSONL metrics to nine PNG/SVG figures."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import html
import json
import math
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.jsonl_logger import latest_run_id  # noqa: E402
from utils.stage2_metrics import (  # noqa: E402
    STAGE2_METRICS_SCHEMA,
    STAGE2_TIMING_FIELDS,
    load_stage2_metrics,
    stage2_records_for_latest_lineage,
    validate_stage2_metric_record,
)

_BRANCH_COLORS = {"dmd": "#1f77b4", "dfd": "#d95f02"}
_ROLE_COLORS = {"generator": "#3b5bdb", "fake_score": "#2b8a3e"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(record: Mapping[str, Any], key: str) -> float | None:
    if key not in record:
        return None
    try:
        value = float(record[key])
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _rolling(values: Sequence[float | None], window: int) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(values)):
        candidates = [
            value
            for value in values[max(0, index - window + 1) : index + 1]
            if value is not None
        ]
        result.append(sum(candidates) / len(candidates) if candidates else None)
    return result


def _latest_run_start(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    run_id = latest_run_id(records)
    if run_id is None:
        raise ValueError("Stage-2 metrics contain no run_start record.")
    for record in reversed(records):
        if record.get("record_type") == "run_start" and record.get("run_id") == run_id:
            if record.get("schema") != STAGE2_METRICS_SCHEMA:
                raise ValueError(
                    "latest run_start does not use the Stage-2 metric schema."
                )
            return dict(record)
    raise AssertionError("latest run_start disappeared")


def _latest_run_end(
    records: Sequence[Mapping[str, Any]], run_start: Mapping[str, Any]
) -> dict[str, Any] | None:
    run_id = run_start["run_id"]
    candidates = [
        dict(record)
        for record in records
        if record.get("record_type") == "run_end" and record.get("run_id") == run_id
    ]
    if not candidates:
        return None
    selected = max(candidates, key=lambda record: int(record["attempt_index"]))
    validate_stage2_metric_record("run_end", selected)
    return selected


def _phase_markers(run_start: Mapping[str, Any]) -> tuple[tuple[str, int], ...]:
    raw = run_start.get("phase_boundaries", ())
    if raw is None:
        return ()
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("run_start.phase_boundaries must be a list.")
    markers: list[tuple[str, int]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"phase_boundaries[{index}] must be an object.")
        label = str(item.get("label", "")).strip()
        update = item.get("generator_update")
        if (
            not label
            or isinstance(update, bool)
            or not isinstance(update, int)
            or update < 0
        ):
            raise ValueError(f"phase_boundaries[{index}] is invalid.")
        markers.append((label, update))
    return tuple(markers)


def _add_phase_markers(
    axis: Any, markers: Sequence[tuple[str, int]], *, scale: int
) -> None:
    for label, generator_update in markers:
        position = generator_update * scale
        axis.axvline(position, color="black", linestyle="--", alpha=0.28, linewidth=0.8)
        axis.annotate(
            label,
            (position, 1.0),
            xycoords=("data", "axes fraction"),
            fontsize=7,
            ha="right",
        )


def _style_axis(axis: Any, *, xlabel: str | None = None) -> None:
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    axis.grid(alpha=0.2)


def _series(
    axis: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    x_key: str,
    field: str,
    label: str,
    rolling_window: int,
    color: str | None = None,
    raw: bool = True,
    linestyle: str = "-",
) -> bool:
    x_values = [int(record[x_key]) for record in records]
    values = [_number(record, field) for record in records]
    points = [(x, value) for x, value in zip(x_values, values) if value is not None]
    if not points:
        return False
    x, y = zip(*points)
    if raw:
        line = axis.plot(
            x,
            y,
            alpha=0.22,
            linewidth=0.8,
            color=color,
            linestyle=linestyle,
            label=f"{label} raw",
        )[0]
        color = line.get_color()
    smooth = [
        (step, value)
        for step, value in zip(x_values, _rolling(values, rolling_window))
        if value is not None
    ]
    if smooth:
        sx, sy = zip(*smooth)
        axis.plot(
            sx,
            sy,
            linewidth=1.6,
            color=color,
            linestyle=linestyle,
            label=label,
        )
    return True


def _legend(axis: Any, *other_axes: Any) -> None:
    handles, labels = axis.get_legend_handles_labels()
    for other in other_axes:
        other_handles, other_labels = other.get_legend_handles_labels()
        handles.extend(other_handles)
        labels.extend(other_labels)
    if handles:
        axis.legend(handles, labels, loc="best", fontsize=7)


def _role(records: Sequence[dict[str, Any]], role: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get("role") == role]


def _terminal_counts(run_start: Mapping[str, Any]) -> dict[str, int]:
    raw = run_start.get("terminal_counts")
    if not isinstance(raw, Mapping):
        raise ValueError("run_start.terminal_counts is required for Stage-2 plots.")
    expected_keys = {"fake_score_updates", "generator_updates", "cycles"}
    if set(raw) != expected_keys:
        raise ValueError(
            "run_start.terminal_counts keys must be " + ", ".join(sorted(expected_keys))
        )
    result = {}
    for key in sorted(expected_keys):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"terminal_counts.{key} must be non-negative integer.")
        result[key] = value
    if result["fake_score_updates"] != 5 * result["generator_updates"]:
        raise ValueError(
            "terminal_counts must encode exactly five F updates per G update."
        )
    if result["cycles"] != result["generator_updates"]:
        raise ValueError("terminal_counts.cycles must equal generator_updates.")
    return result


def _validate_plot_records(
    generator: Sequence[dict[str, Any]],
    fake: Sequence[dict[str, Any]],
    cycles: Sequence[dict[str, Any]],
) -> None:
    for record in [*generator, *fake]:
        validate_stage2_metric_record("train_step", record)
    for record in cycles:
        validate_stage2_metric_record("cycle_summary", record)
    for role, selected, key in (
        ("generator", generator, "completed_generator_updates"),
        ("fake_score", fake, "completed_fake_updates"),
    ):
        values = [int(record[key]) for record in selected]
        if values != sorted(set(values)):
            raise ValueError(f"{role} update axis must be strictly increasing.")


def _validate_complete_structure(
    *,
    generator: Sequence[dict[str, Any]],
    fake: Sequence[dict[str, Any]],
    cycles: Sequence[dict[str, Any]],
    terminal: Mapping[str, int],
) -> None:
    expected_g = list(range(1, terminal["generator_updates"] + 1))
    expected_f = list(range(1, terminal["fake_score_updates"] + 1))
    expected_cycles = list(range(1, terminal["cycles"] + 1))
    if [
        int(record["completed_generator_updates"]) for record in generator
    ] != expected_g:
        raise ValueError(
            "complete Stage-2 lineage does not contain every G update exactly once."
        )
    if [int(record["completed_fake_updates"]) for record in fake] != expected_f:
        raise ValueError(
            "complete Stage-2 lineage does not contain every F update exactly once."
        )
    if [int(record["completed_cycles"]) for record in cycles] != expected_cycles:
        raise ValueError(
            "complete Stage-2 lineage does not contain every cycle exactly once."
        )


def _write_html_index(
    *,
    path: Path,
    jsonl_path: Path,
    run_start: Mapping[str, Any],
    run_end: Mapping[str, Any] | None,
    figures: Sequence[Path],
    counts: Mapping[str, int],
    status: str,
) -> None:
    figure_rows = []
    thumbnails = []
    for figure in figures:
        figure_rows.append(
            '<tr><td><a href="{}">{}</a></td><td><code>{}</code></td></tr>'.format(
                html.escape(figure.name), html.escape(figure.name), _sha256(figure)
            )
        )
        if figure.suffix == ".png":
            thumbnails.append(
                '<figure><a href="{}"><img src="{}" alt="{}"></a>'
                "<figcaption>{}</figcaption></figure>".format(
                    *(html.escape(figure.name),) * 4
                )
            )
    summary = {
        "status": status,
        "run_end_status": None if run_end is None else run_end.get("status"),
        "run_id": run_start.get("run_id"),
        "parent_run_id": run_start.get("parent_run_id"),
        "completed_fake_updates": counts["fake_score_updates"],
        "completed_generator_updates": counts["generator_updates"],
        "completed_cycles": counts["cycles"],
        "config_contract_sha256": run_start.get("config_contract_sha256"),
        "config_launch_sha256": run_start.get("config_launch_sha256"),
        "metrics_jsonl_sha256": _sha256(jsonl_path),
    }
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Stage-2 training metrics</title>
<style>body{{font:14px system-ui;max-width:1180px;margin:2rem auto;padding:0 1rem}}code{{font-size:12px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:.45rem;text-align:left}}.partial{{color:#9a6700}}.complete{{color:#087f23}}figure{{margin:2rem 0}}img{{width:100%;height:auto;border:1px solid #ddd}}figcaption{{font-weight:600}}</style></head>
<body><h1>Stage-2 training metrics</h1><p class="{html.escape(status)}">Status: <strong>{html.escape(status)}</strong></p>
<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))}</pre>
<h2>Figures</h2>{''.join(thumbnails)}
<h2>Artifacts and SHA-256</h2><table><thead><tr><th>File</th><th>SHA-256</th></tr></thead><tbody>{''.join(figure_rows)}</tbody></table>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def _plot_generator_loss(
    plt: Any,
    records: Sequence[dict[str, Any]],
    *,
    rolling_window: int,
    markers: Sequence[tuple[str, int]],
) -> Any:
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    if records:
        for branch, label in (("dmd", "DMD surrogate"), ("dfd", "DFD surrogate")):
            selected = [record for record in records if record["branch"] == branch]
            _series(
                axes[0],
                selected,
                x_key="completed_generator_updates",
                field="loss",
                label=label,
                rolling_window=rolling_window,
                color=_BRANCH_COLORS[branch],
            )
        _legend(axes[0])
        for field, label, color in (
            ("denominator_mean", "Holistic denominator", "#6f42c1"),
            ("raw_score_difference_l2", "Raw score-difference L2", "#c2255c"),
        ):
            _series(
                axes[1],
                records,
                x_key="completed_generator_updates",
                field=field,
                label=label,
                rolling_window=rolling_window,
                color=color,
            )
        _legend(axes[1])
    else:
        axes[0].text(0.5, 0.5, "Generator has not completed an update", ha="center")
    axes[0].set_title("Branch-coloured Generator surrogate loss")
    axes[1].set_title("Generator denominator and score difference")
    for axis in axes:
        _add_phase_markers(axis, markers, scale=1)
        _style_axis(axis, xlabel="completed generator updates")
    return figure


def _plot_fake_loss(
    plt: Any,
    records: Sequence[dict[str, Any]],
    *,
    rolling_window: int,
    markers: Sequence[tuple[str, int]],
) -> Any:
    figure, axis = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    for field, label, color in (
        ("loss", "Fake-score raw-flow MSE", "#2b8a3e"),
        ("target_flow_l2", "Target flow L2", "#e67700"),
        ("prediction_l2", "Prediction L2", "#1971c2"),
    ):
        _series(
            axis,
            records,
            x_key="completed_fake_updates",
            field=field,
            label=label,
            rolling_window=rolling_window,
            color=color,
        )
    _legend(axis)
    _add_phase_markers(axis, markers, scale=5)
    axis.set_title("Fake-score flow DSM loss")
    _style_axis(axis, xlabel="completed fake-score updates")
    return figure


def _plot_optimization(
    plt: Any,
    generator: Sequence[dict[str, Any]],
    fake: Sequence[dict[str, Any]],
    *,
    rolling_window: int,
    markers: Sequence[tuple[str, int]],
) -> Any:
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for axis, selected, role, x_key, scale in (
        (axes[0], generator, "Generator", "completed_generator_updates", 1),
        (axes[1], fake, "Fake-score", "completed_fake_updates", 5),
    ):
        learning_axis = axis.twinx()
        _series(
            axis,
            selected,
            x_key=x_key,
            field="preclip_grad_norm",
            label=f"{role} preclip grad norm",
            rolling_window=rolling_window,
            color=_ROLE_COLORS[role.lower().replace("-", "_")],
        )
        _series(
            learning_axis,
            selected,
            x_key=x_key,
            field="learning_rate",
            label=f"{role} learning rate",
            rolling_window=rolling_window,
            color="#c92a2a",
        )
        axis.set_title(f"{role} optimization (independent update axis)")
        axis.set_ylabel("gradient norm")
        learning_axis.set_ylabel("learning rate")
        _legend(axis, learning_axis)
        _add_phase_markers(axis, markers, scale=scale)
        _style_axis(axis, xlabel=x_key.replace("_", " "))
    return figure


def _plot_role_throughput(
    plt: Any,
    records: Sequence[dict[str, Any]],
    *,
    role: str,
    rolling_window: int,
    markers: Sequence[tuple[str, int]],
) -> Any:
    x_key = (
        "completed_generator_updates"
        if role == "generator"
        else "completed_fake_updates"
    )
    scale = 1 if role == "generator" else 5
    title = "Generator" if role == "generator" else "Fake-score"
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for field, label, color in (
        ("step_seconds_max", f"{title} update seconds", "#495057"),
        ("samples_per_second", "Samples/s", "#2b8a3e"),
        ("generated_latents_per_second", "Generated latents/s", "#e67700"),
    ):
        _series(
            axes[0],
            records,
            x_key=x_key,
            field=field,
            label=label,
            rolling_window=rolling_window,
            color=color,
        )
    token_specs = [
        ("generator_tokens_per_second", "Generator logical query tokens/s", "#3b5bdb"),
        (
            "fake_score_tokens_per_second",
            "Fake-score logical query tokens/s",
            "#2b8a3e",
        ),
    ]
    if role == "generator":
        token_specs.append(
            (
                "real_score_tokens_per_second",
                "Real-score logical query tokens/s",
                "#c92a2a",
            )
        )
    for field, label, color in token_specs:
        _series(
            axes[1],
            records,
            x_key=x_key,
            field=field,
            label=label,
            rolling_window=rolling_window,
            color=color,
        )
    axes[0].set_title(f"{title} role wall time and sample/latent throughput")
    axes[1].set_title(f"{title} role model-specific logical query throughput")
    for axis in axes:
        _legend(axis)
        _add_phase_markers(axis, markers, scale=scale)
        _style_axis(axis, xlabel=x_key.replace("_", " "))
    return figure


def _plot_cycle_throughput(
    plt: Any,
    records: Sequence[dict[str, Any]],
    *,
    rolling_window: int,
    markers: Sequence[tuple[str, int]],
) -> Any:
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for field, label, color in (
        ("cycle_seconds", "5F -> 1G cycle seconds", "#495057"),
        ("cycle_samples_per_second", "Cycle role-samples/s", "#6741d9"),
        ("fake_score_samples_per_second", "F samples/s", "#2b8a3e"),
        ("generator_samples_per_second", "G samples/s", "#3b5bdb"),
    ):
        _series(
            axes[0],
            records,
            x_key="completed_cycles",
            field=field,
            label=label,
            rolling_window=rolling_window,
            color=color,
        )
    for field, label, color in (
        ("fake_score_time_fraction", "F role time fraction", "#2b8a3e"),
        ("generator_time_fraction", "G role time fraction", "#3b5bdb"),
    ):
        _series(
            axes[1],
            records,
            x_key="completed_cycles",
            field=field,
            label=label,
            rolling_window=rolling_window,
            color=color,
        )
    axes[0].set_title("Complete-cycle wall time and role throughput")
    axes[1].set_title("Complete-cycle role time share")
    for axis in axes:
        _legend(axis)
        _add_phase_markers(axis, markers, scale=1)
        _style_axis(axis, xlabel="completed cycles")
    return figure


def _plot_time_breakdown(
    plt: Any,
    records: Sequence[dict[str, Any]],
    *,
    rolling_window: int,
    markers: Sequence[tuple[str, int]],
) -> Any:
    figure, axes = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)
    labels = {
        "data_seconds_max": "Data wait",
        "h2d_seconds_max": "H2D",
        "rollout_seconds_max": "Rollout",
        "fake_score_seconds_max": "Fake score",
        "real_cond_seconds_max": "Real cond",
        "real_uncond_seconds_max": "Real uncond",
        "loss_build_seconds_max": "Loss build",
        "backward_seconds_max": "Backward",
        "clip_optimizer_seconds_max": "Clip + optimizer",
        "ema_seconds_max": "EMA",
    }
    for field in STAGE2_TIMING_FIELDS:
        _series(
            axes[0],
            records,
            x_key="logical_substep_id",
            field=field,
            label=labels[field],
            rolling_window=rolling_window,
            raw=False,
        )
    enriched = []
    for record in records:
        item = dict(record)
        item["timing_parts_sum_seconds"] = sum(
            float(record[key]) for key in STAGE2_TIMING_FIELDS
        )
        enriched.append(item)
    for field, label, color in (
        ("step_seconds_max", "Update wall seconds", "#212529"),
        ("timing_parts_sum_seconds", "Detailed parts sum", "#6741d9"),
        ("timing_closure_error_seconds", "Wall-time closure error", "#c92a2a"),
    ):
        _series(
            axes[1],
            enriched,
            x_key="logical_substep_id",
            field=field,
            label=label,
            rolling_window=rolling_window,
            color=color,
        )
    axes[0].set_title("Detailed Stage-2 update time breakdown")
    axes[1].set_title("Detailed timing closure against update wall time")
    for axis in axes:
        _legend(axis)
        _add_phase_markers(axis, markers, scale=6)
        _style_axis(axis, xlabel="logical substep id")
    return figure


def _plot_memory_straggler(
    plt: Any,
    records: Sequence[dict[str, Any]],
    *,
    rolling_window: int,
    markers: Sequence[tuple[str, int]],
) -> Any:
    figure, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    for field, label, color in (
        ("gpu_memory_allocated_gib_max", "Max allocated GiB", "#3b5bdb"),
        ("gpu_memory_reserved_gib_max", "Max reserved GiB", "#6741d9"),
        ("gpu_memory_free_gib_min", "Min free GiB", "#2b8a3e"),
    ):
        _series(
            axes[0],
            records,
            x_key="logical_substep_id",
            field=field,
            label=label,
            rolling_window=rolling_window,
            color=color,
        )
    _series(
        axes[1],
        records,
        x_key="logical_substep_id",
        field="straggler_ratio",
        label="Cross-rank max/mean step-time ratio",
        rolling_window=rolling_window,
        color="#c92a2a",
    )
    axes[0].set_title("CUDA allocated, reserved, and free memory")
    axes[1].set_title("Cross-rank straggler")
    for axis in axes:
        _legend(axis)
        _add_phase_markers(axis, markers, scale=6)
        _style_axis(axis, xlabel="logical substep id")
    return figure


def _aggregate_histogram(
    records: Sequence[Mapping[str, Any]], field: str
) -> dict[str, int]:
    result: dict[str, int] = {}
    for record in records:
        for key, value in record[field].items():
            result[str(key)] = result.get(str(key), 0) + int(value)
    return result


def _plot_timestep_exit_phase(
    plt: Any,
    records: Sequence[dict[str, Any]],
    generator: Sequence[dict[str, Any]],
    nonfinite: Sequence[dict[str, Any]],
    *,
    markers: Sequence[tuple[str, int]],
) -> Any:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    timestep = _aggregate_histogram(records, "score_timestep_histogram")
    exits = _aggregate_histogram(records, "exit_histogram")
    axes[0, 0].bar(list(timestep), list(timestep.values()), color="#1971c2")
    axes[0, 0].tick_params(axis="x", labelrotation=45)
    axes[0, 0].set_title("Score timestep histogram")
    axes[0, 0].set_ylabel("observations")
    axes[0, 1].bar(list(exits), list(exits.values()), color="#2b8a3e")
    axes[0, 1].set_title("Random-exit coverage histogram")
    axes[0, 1].set_xlabel("exit step")

    dmd_count = 0
    dfd_count = 0
    x_values = []
    dmd_values = []
    dfd_values = []
    for record in generator:
        dmd_count += int(record["branch"] == "dmd")
        dfd_count += int(record["branch"] == "dfd")
        x_values.append(int(record["completed_generator_updates"]))
        dmd_values.append(dmd_count)
        dfd_values.append(dfd_count)
    axes[1, 0].step(
        x_values,
        dmd_values,
        where="post",
        label="Cumulative DMD",
        color=_BRANCH_COLORS["dmd"],
    )
    axes[1, 0].step(
        x_values,
        dfd_values,
        where="post",
        label="Cumulative DFD",
        color=_BRANCH_COLORS["dfd"],
    )
    axes[1, 0].set_title("Cumulative DMD / DFD branch counts")
    _legend(axes[1, 0])
    _add_phase_markers(axes[1, 0], markers, scale=1)
    _style_axis(axes[1, 0], xlabel="completed generator updates")

    _series(
        axes[1, 1],
        generator,
        x_key="logical_substep_id",
        field="dfd_probability",
        label="Scheduled DFD probability",
        rolling_window=1,
        color="#6741d9",
    )
    _series(
        axes[1, 1],
        generator,
        x_key="logical_substep_id",
        field="branch_is_dfd",
        label="Selected DFD branch",
        rolling_window=1,
        color=_BRANCH_COLORS["dfd"],
    )
    for index, record in enumerate(nonfinite):
        label = "Nonfinite attempt" if index == 0 else None
        axes[1, 1].axvline(
            int(record["logical_substep_id"]),
            color="#c92a2a",
            linestyle=":",
            linewidth=1.2,
            alpha=0.8,
            label=label,
        )
    if not nonfinite:
        axes[1, 1].text(
            0.98,
            0.04,
            "No nonfinite attempts",
            ha="right",
            va="bottom",
            transform=axes[1, 1].transAxes,
            fontsize=8,
        )
    axes[1, 1].set_title("DFD schedule/selection and nonfinite markers")
    _legend(axes[1, 1])
    _add_phase_markers(axes[1, 1], markers, scale=6)
    _style_axis(axes[1, 1], xlabel="logical substep id")
    for axis in (axes[0, 0], axes[0, 1]):
        axis.grid(axis="y", alpha=0.2)
    return figure


def plot_stage2_training(
    jsonl_path: str | Path,
    output_dir: str | Path,
    *,
    rolling_window: int = 20,
    formats: Iterable[str] = ("png", "svg"),
    include_dry_run: bool = False,
    require_complete: bool = False,
) -> list[Path]:
    """Write nine figures plus HTML from the latest authoritative lineage."""

    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive.")
    formats = tuple(str(value).lower() for value in formats)
    if not formats or any(value not in {"png", "svg"} for value in formats):
        raise ValueError("formats must contain png and/or svg.")
    jsonl_path = Path(jsonl_path).expanduser().resolve()
    records = load_stage2_metrics(jsonl_path)
    run_start = _latest_run_start(records)
    train_steps = stage2_records_for_latest_lineage(records, record_type="train_step")
    cycles = stage2_records_for_latest_lineage(records, record_type="cycle_summary")
    nonfinite = stage2_records_for_latest_lineage(
        records, record_type="nonfinite_attempt"
    )
    if not include_dry_run:
        train_steps = [
            record for record in train_steps if not record.get("dry_run", False)
        ]
        cycles = [record for record in cycles if not record.get("dry_run", False)]
        nonfinite = [record for record in nonfinite if not record.get("dry_run", False)]
    if not train_steps:
        raise ValueError("No non-dry-run Stage-2 train_step records in latest lineage.")
    generator = _role(train_steps, "generator")
    fake = _role(train_steps, "fake_score")
    _validate_plot_records(generator, fake, cycles)
    for record in nonfinite:
        validate_stage2_metric_record("nonfinite_attempt", record)

    terminal = _terminal_counts(run_start)
    actual = {
        "fake_score_updates": max(
            (int(record["completed_fake_updates"]) for record in train_steps), default=0
        ),
        "generator_updates": max(
            (int(record["completed_generator_updates"]) for record in train_steps),
            default=0,
        ),
        "cycles": max(
            (int(record["completed_cycles"]) for record in cycles), default=0
        ),
    }
    run_end = _latest_run_end(records, run_start)
    terminal_reached = actual == terminal
    claims_complete = run_end is not None and run_end.get("status") == "complete"
    if claims_complete and not terminal_reached:
        raise ValueError(
            "latest run_end.status=complete but resolved terminal counts were not "
            f"reached: actual={actual}, terminal={terminal}."
        )
    complete = bool(claims_complete and terminal_reached)
    if claims_complete:
        _validate_complete_structure(
            generator=generator, fake=fake, cycles=cycles, terminal=terminal
        )
    if require_complete and not complete:
        run_end_status = None if run_end is None else run_end.get("status")
        raise ValueError(
            "Stage-2 log is incomplete: "
            f"actual={actual}, terminal={terminal}, run_end.status={run_end_status!r}."
        )
    status = "complete" if complete else "partial"
    markers = _phase_markers(run_start)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plots: list[tuple[str, Callable[[], Any]]] = [
        (
            "generator_loss",
            lambda: _plot_generator_loss(
                plt, generator, rolling_window=rolling_window, markers=markers
            ),
        ),
        (
            "fake_score_loss",
            lambda: _plot_fake_loss(
                plt, fake, rolling_window=rolling_window, markers=markers
            ),
        ),
        (
            "optimization",
            lambda: _plot_optimization(
                plt,
                generator,
                fake,
                rolling_window=rolling_window,
                markers=markers,
            ),
        ),
        (
            "generator_throughput",
            lambda: _plot_role_throughput(
                plt,
                generator,
                role="generator",
                rolling_window=rolling_window,
                markers=markers,
            ),
        ),
        (
            "fake_score_throughput",
            lambda: _plot_role_throughput(
                plt,
                fake,
                role="fake_score",
                rolling_window=rolling_window,
                markers=markers,
            ),
        ),
        (
            "cycle_throughput",
            lambda: _plot_cycle_throughput(
                plt, cycles, rolling_window=rolling_window, markers=markers
            ),
        ),
        (
            "time_breakdown",
            lambda: _plot_time_breakdown(
                plt, train_steps, rolling_window=rolling_window, markers=markers
            ),
        ),
        (
            "memory_straggler",
            lambda: _plot_memory_straggler(
                plt, train_steps, rolling_window=rolling_window, markers=markers
            ),
        ),
        (
            "timestep_exit_phase",
            lambda: _plot_timestep_exit_phase(
                plt, train_steps, generator, nonfinite, markers=markers
            ),
        ),
    ]

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for stem, build in plots:
        figure = build()
        figure.suptitle(f"Stage-2 {stem.replace('_', ' ')} ({status})", fontsize=13)
        for extension in formats:
            path = output_dir / f"{stem}.{extension}"
            figure.savefig(path, dpi=160 if extension == "png" else None)
            written.append(path)
        plt.close(figure)
    index = output_dir / "index.html"
    _write_html_index(
        path=index,
        jsonl_path=jsonl_path,
        run_start=run_start,
        run_end=run_end,
        figures=written,
        counts=actual,
        status=status,
    )
    written.append(index)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument(
        "--formats", nargs="+", default=("png", "svg"), choices=("png", "svg")
    )
    parser.add_argument("--include-dry-run", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = plot_stage2_training(
        args.jsonl,
        args.output_dir,
        rolling_window=args.rolling_window,
        formats=args.formats,
        include_dry_run=args.include_dry_run,
        require_complete=args.require_complete,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
