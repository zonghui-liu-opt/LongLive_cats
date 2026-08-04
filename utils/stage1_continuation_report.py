"""Human-review HTML for the fixed Stage-1 continuation matrix."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import html
import os
from pathlib import Path
from typing import Any
from urllib.parse import quote

from utils.stage1_continuation_validation import ContinuationMetadataRecord
from utils.stage1_io import atomic_write_bytes

EXPECTED_CASE_COUNT = 8
EXPECTED_SINKS = (0, 1)
LATENT_FRAME_RANGES = (
    ("A", 0, 23),
    ("HOLD", 24, 39),
    ("B", 40, 63),
)
PIXEL_FRAME_RANGES = (
    ("A", 0, 92),
    ("HOLD", 93, 156),
    ("B", 157, 252),
)


@dataclass(frozen=True)
class _ReportSample:
    case_group: str
    sink_size: int
    seed: int
    relative_video_url: str


def _require_fixed_records(
    records: Sequence[ContinuationMetadataRecord],
) -> list[ContinuationMetadataRecord]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise TypeError(
            "records must be a sequence of ContinuationMetadataRecord values"
        )
    if len(records) != EXPECTED_CASE_COUNT:
        raise ValueError(
            f"continuation report requires exactly {EXPECTED_CASE_COUNT} records"
        )
    if any(not isinstance(record, ContinuationMetadataRecord) for record in records):
        raise TypeError("every report record must be a ContinuationMetadataRecord")

    row_ids = [record.row_id for record in records]
    if len(set(row_ids)) != EXPECTED_CASE_COUNT or set(row_ids) != set(
        range(EXPECTED_CASE_COUNT)
    ):
        raise ValueError(
            "continuation report records must have unique row_id values 0..7"
        )
    case_groups = [record.case_group for record in records]
    if len(set(case_groups)) != EXPECTED_CASE_COUNT:
        raise ValueError(
            "continuation report records must have unique case_group values"
        )
    return sorted(records, key=lambda record: record.row_id)


def _relative_video_url(output_video: Any, *, work_dir: Path) -> str:
    try:
        raw_path = Path(os.fspath(output_video)).expanduser()
    except TypeError as error:
        raise TypeError("sample output_video must be path-like") from error
    video_path = (
        raw_path.resolve()
        if raw_path.is_absolute()
        else (work_dir / raw_path).resolve()
    )
    try:
        relative_path = video_path.relative_to(work_dir)
    except ValueError as error:
        raise ValueError(
            f"sample output_video must be contained in work_dir: {video_path}"
        ) from error
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    return quote(relative_path.as_posix(), safe="/._-")


def _index_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    records: Sequence[ContinuationMetadataRecord],
    work_dir: Path,
) -> dict[tuple[str, int], _ReportSample]:
    expected_count = EXPECTED_CASE_COUNT * len(EXPECTED_SINKS)
    if isinstance(samples, (str, bytes)) or not isinstance(samples, Sequence):
        raise TypeError("samples must be a sequence of mappings")
    if len(samples) != expected_count:
        raise ValueError(
            f"continuation report requires exactly {expected_count} samples"
        )

    known_groups = {record.case_group for record in records}
    indexed: dict[tuple[str, int], _ReportSample] = {}
    for sample_index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TypeError(f"sample {sample_index} must be a mapping")
        missing_fields = [
            field
            for field in ("case_group", "sink_size", "seed", "output_video")
            if field not in sample
        ]
        if missing_fields:
            raise ValueError(
                f"sample {sample_index} is missing required fields {missing_fields}"
            )

        case_group = sample["case_group"]
        if not isinstance(case_group, str) or not case_group:
            raise TypeError(
                f"sample {sample_index} case_group must be a non-empty string"
            )
        if case_group not in known_groups:
            raise ValueError(
                f"sample {sample_index} has unknown case_group {case_group!r}"
            )

        sink_size = sample["sink_size"]
        if type(sink_size) is not int or sink_size not in EXPECTED_SINKS:
            raise ValueError(f"sample {sample_index} sink_size must be exactly 0 or 1")
        seed = sample["seed"]
        if type(seed) is not int:
            raise TypeError(f"sample {sample_index} seed must be an integer")

        key = (case_group, sink_size)
        if key in indexed:
            raise ValueError(
                "duplicate continuation sample for "
                f"case_group={case_group!r}, sink_size={sink_size}"
            )
        indexed[key] = _ReportSample(
            case_group=case_group,
            sink_size=sink_size,
            seed=seed,
            relative_video_url=_relative_video_url(
                sample["output_video"], work_dir=work_dir
            ),
        )

    expected_keys = {
        (record.case_group, sink_size)
        for record in records
        for sink_size in EXPECTED_SINKS
    }
    missing_keys = sorted(expected_keys - indexed.keys())
    if missing_keys:
        raise ValueError(
            f"continuation report is missing sample combinations {missing_keys}"
        )
    for record in records:
        seeds = {indexed[(record.case_group, sink)].seed for sink in EXPECTED_SINKS}
        if len(seeds) != 1:
            raise ValueError(
                f"sink variants for {record.case_group!r} must use the same seed"
            )
    return indexed


def _ranges_html(ranges: Sequence[tuple[str, int, int]]) -> str:
    return " · ".join(
        f"{html.escape(label)} {start}–{end}" for label, start, end in ranges
    )


def build_continuation_comparison_html(
    records: Sequence[ContinuationMetadataRecord],
    samples: Sequence[Mapping[str, Any]],
    *,
    work_dir: str | os.PathLike[str],
) -> str:
    """Build a portable 8-case by 2-sink continuation review page."""

    work_dir_path = Path(work_dir).expanduser().resolve()
    if not work_dir_path.is_dir():
        raise NotADirectoryError(work_dir_path)
    ordered_records = _require_fixed_records(records)
    samples_by_key = _index_samples(
        samples, records=ordered_records, work_dir=work_dir_path
    )

    style = """
body{font-family:system-ui,sans-serif;margin:24px;color:#202124;background:#fafafa}
table{border-collapse:collapse;background:#fff;width:100%}th,td{border:1px solid #ddd;padding:10px;vertical-align:top}
th{position:sticky;top:0;background:#f1f3f4;z-index:1}.case{min-width:300px;max-width:520px}
video{display:block;width:100%;max-width:440px;max-height:440px;background:#111}.prompt{white-space:pre-wrap;font-size:12px}
.facts{font-size:12px;color:#4b4b4b}.sync-controls{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:8px 0}
.sync-controls input{width:180px}
"""
    lines = [
        "<!doctype html>",
        '<html lang="zh-CN"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>Stage-1 continuation sink 对比</title><style>{style}</style></head><body>",
        "<h1>Stage-1 continuation sink 对比</h1>",
        "<p>同一行使用相同输入、seed 和噪声计划；请人工检查动作顺序、HOLD 稳定性与两处边界。</p>",
        '<table id="continuation-comparison"><thead><tr><th>case</th><th>sink=0</th><th>sink=1</th></tr></thead><tbody>',
    ]
    latent_ranges = _ranges_html(LATENT_FRAME_RANGES)
    pixel_ranges = _ranges_html(PIXEL_FRAME_RANGES)
    for record in ordered_records:
        case_group = html.escape(record.case_group, quote=True)
        row_dom_id = f"continuation-row-{record.row_id}"
        lines.extend(
            [
                f'<tr id="{row_dom_id}" data-sync-row data-case-group="{case_group}">',
                '<td class="case">',
                f"<strong>{html.escape(record.cat_id)} · {html.escape(record.action_order)}</strong>",
                f'<div class="facts">case_group: {case_group}<br>soft re-anchor: {str(record.soft_reanchor).lower()}<br>',
                f"latent frames: {latent_ranges}<br>pixel frames: {pixel_ranges}</div>",
                '<div class="sync-controls" aria-label="row synchronized video controls">',
                '<button type="button" data-sync-action="play">Play</button>',
                '<button type="button" data-sync-action="pause">Pause</button>',
                '<label>Seek <input type="range" min="0" max="1" step="0.001" value="0" data-sync-action="seek"></label>',
                '<button type="button" data-sync-action="reset">Reset</button>',
                "</div>",
                f'<p class="prompt"><strong>A</strong> {html.escape(record.action_a_prompt)}</p>',
                f'<p class="prompt"><strong>HOLD</strong> {html.escape(record.hold_prompt)}</p>',
                f'<p class="prompt"><strong>B</strong> {html.escape(record.action_b_prompt)}</p>',
                "</td>",
            ]
        )
        for sink_size in EXPECTED_SINKS:
            sample = samples_by_key[(record.case_group, sink_size)]
            source = html.escape(sample.relative_video_url, quote=True)
            lines.append(
                f'<td class="sink-cell" data-sink="{sink_size}">'
                f'<div class="facts">seed: {sample.seed} · sink: {sink_size}</div>'
                f'<video controls preload="metadata" src="{source}" '
                f'data-case-group="{case_group}" data-sink="{sink_size}"></video></td>'
            )
        lines.append("</tr>")
    lines.extend(
        [
            "</tbody></table>",
            """<script>
(() => {
  document.querySelectorAll('[data-sync-row]').forEach((row) => {
    const videos = Array.from(row.querySelectorAll('video'));
    const seek = row.querySelector('[data-sync-action="seek"]');
    row.querySelector('[data-sync-action="play"]').addEventListener('click', () => {
      videos.forEach((video) => { void video.play(); });
    });
    row.querySelector('[data-sync-action="pause"]').addEventListener('click', () => {
      videos.forEach((video) => video.pause());
    });
    seek.addEventListener('input', () => {
      const ratio = Number(seek.value);
      videos.forEach((video) => {
        if (Number.isFinite(video.duration)) video.currentTime = ratio * video.duration;
      });
    });
    row.querySelector('[data-sync-action="reset"]').addEventListener('click', () => {
      videos.forEach((video) => { video.pause(); video.currentTime = 0; });
      seek.value = '0';
    });
    videos[0].addEventListener('timeupdate', () => {
      if (Number.isFinite(videos[0].duration) && videos[0].duration > 0) {
        seek.value = String(videos[0].currentTime / videos[0].duration);
      }
    });
  });
})();
</script>""",
            "</body></html>",
        ]
    )
    return "".join(lines)


def write_continuation_comparison_html(
    destination: str | os.PathLike[str],
    records: Sequence[ContinuationMetadataRecord],
    samples: Sequence[Mapping[str, Any]],
    *,
    work_dir: str | os.PathLike[str],
) -> Path:
    """Atomically write :func:`build_continuation_comparison_html`."""

    destination_path = Path(destination).expanduser().resolve()
    payload = build_continuation_comparison_html(
        records, samples, work_dir=work_dir
    ).encode("utf-8")
    atomic_write_bytes(destination_path, payload)
    return destination_path
