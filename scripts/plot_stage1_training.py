#!/usr/bin/env python3
"""Render Stage-1 JSONL training metrics to manual PNG/SVG plots."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.jsonl_logger import read_jsonl_tolerant, records_for_latest_lineage


def _number(record: dict[str, Any], *keys: str) -> float | None:
    value: Any = record
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rolling(values: list[float | None], window: int) -> list[float | None]:
    result: list[float | None] = []
    for index in range(len(values)):
        valid = [value for value in values[max(0, index - window + 1): index + 1] if value is not None]
        result.append(sum(valid) / len(valid) if valid else None)
    return result


def _plot_series(ax, steps, records, specs, rolling_window):
    plotted = False
    for label, getter in specs:
        values = [getter(record) for record in records]
        points = [(step, value) for step, value in zip(steps, values) if value is not None]
        if not points:
            continue
        plotted = True
        x, y = zip(*points)
        line = ax.plot(x, y, alpha=0.22, linewidth=0.8, label=f"{label} raw")[0]
        rolling = _rolling(values, rolling_window)
        smooth = [(step, value) for step, value in zip(steps, rolling) if value is not None]
        if smooth:
            sx, sy = zip(*smooth)
            ax.plot(sx, sy, linewidth=1.6, color=line.get_color(), label=label)
    if not plotted:
        ax.text(0.5, 0.5, "No matching metrics", ha="center", va="center", transform=ax.transAxes)
    else:
        ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.2)
    for marker in (300, 480):
        ax.axvline(marker, color="black", linestyle="--", alpha=0.25, linewidth=0.8)


def plot_training_metrics(
    jsonl_path: str | Path,
    output_dir: str | Path,
    *,
    rolling_window: int = 20,
    formats: Iterable[str] = ("png", "svg"),
) -> list[Path]:
    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive.")
    formats = tuple(str(value).lower() for value in formats)
    if not formats or any(value not in {"png", "svg"} for value in formats):
        raise ValueError("formats must contain png and/or svg.")

    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    records = records_for_latest_lineage(read_jsonl_tolerant(jsonl_path))
    if not records:
        raise ValueError(f"No train_step records found in latest lineage: {jsonl_path}")
    steps = [int(record["optimizer_step"]) for record in records]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures: list[tuple[str, str, list[tuple[str, Callable[[dict[str, Any]], float | None]]]]] = [
        (
            "loss",
            "Flow-matching loss",
            [
                ("total", lambda r: _number(r, "loss_total")),
                ("block0", lambda r: _number(r, "blocks", "block0", "loss") or _number(r, "block0_loss")),
                ("block1", lambda r: _number(r, "blocks", "block1", "loss") or _number(r, "block1_loss")),
                ("block2", lambda r: _number(r, "blocks", "block2", "loss") or _number(r, "block2_loss")),
            ],
        ),
        (
            "throughput",
            "Logical throughput",
            [
                ("DiT tokens/s", lambda r: _number(r, "logical_dit_tokens_per_second")),
                ("supervised tokens/s", lambda r: _number(r, "supervised_tokens_per_second")),
            ],
        ),
        (
            "step_time_samples",
            "Step time and sample throughput",
            [
                ("step seconds max", lambda r: _number(r, "step_seconds_max")),
                ("samples/s", lambda r: _number(r, "samples_per_second")),
            ],
        ),
        (
            "lr_grad_norm",
            "Optimization",
            [
                ("learning rate", lambda r: _number(r, "lr")),
                ("pre-clip grad norm", lambda r: _number(r, "pre_clip_grad_norm")),
            ],
        ),
        (
            "error_recycling",
            "Error recycling",
            [
                ("scheduled active", lambda r: _number(r, "er", "scheduled_active_probability") or _number(r, "scheduled_active_probability")),
                ("actual applied rate", lambda r: _number(r, "er", "actual_applied_rate") or _number(r, "actual_applied_rate")),
                ("buffer mean", lambda r: _number(r, "er", "buffer_entries_mean") or _number(r, "buffer_entries_mean")),
            ],
        ),
        (
            "memory_straggler",
            "GPU memory and straggler ratio",
            [
                ("allocated GiB", lambda r: _number(r, "gpu_memory_allocated_gib_max")),
                ("reserved GiB", lambda r: _number(r, "gpu_memory_reserved_gib_max")),
                ("straggler", lambda r: _number(r, "straggler_ratio")),
            ],
        ),
    ]

    written: list[Path] = []
    for stem, title, specs in figures:
        figure, axis = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
        _plot_series(axis, steps, records, specs, rolling_window)
        axis.set_title(title)
        axis.set_xlabel("Completed optimizer steps")
        for extension in formats:
            path = output_dir / f"{stem}.{extension}"
            figure.savefig(path, dpi=160 if extension == "png" else None)
            written.append(path)
        plt.close(figure)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True, help="Path to train_metrics.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--formats", nargs="+", default=("png", "svg"), choices=("png", "svg"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = plot_training_metrics(
        args.jsonl,
        args.output_dir,
        rolling_window=args.rolling_window,
        formats=args.formats,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
