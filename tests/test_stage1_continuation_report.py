from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

import pytest

from utils.stage1_continuation_report import build_continuation_comparison_html
from utils.stage1_continuation_validation import load_continuation_metadata

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_METADATA = (
    PROJECT_ROOT
    / "testsets"
    / "metadata_8cases_two_actions_continuation_480x832_253frames.csv"
)


class _ReportDOM(HTMLParser):
    def __init__(self):
        super().__init__()
        self.rows: list[dict[str, str]] = []
        self.videos: list[dict[str, str]] = []
        self.actions: list[dict[str, str]] = []

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == "tr" and "data-sync-row" in attributes:
            self.rows.append(attributes)
        elif tag == "video":
            self.videos.append(attributes)
        elif attributes.get("data-sync-action"):
            self.actions.append(attributes)


@pytest.fixture
def report_inputs(tmp_path):
    records = load_continuation_metadata(FORMAL_METADATA, validate_images=False)
    samples = []
    for record in records:
        for sink_size in (0, 1):
            video_path = (
                tmp_path
                / "checkpoint_model_003750"
                / "continuation"
                / record.case_group
                / f"sink{sink_size}.mp4"
            )
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(f"{record.case_group}-{sink_size}".encode())
            samples.append(
                {
                    "case_group": record.case_group,
                    "sink_size": sink_size,
                    "seed": 1,
                    "output_video": str(video_path),
                }
            )
    return records, samples, tmp_path


def _parse(document: str) -> _ReportDOM:
    parser = _ReportDOM()
    parser.feed(document)
    return parser


def test_report_uses_explicit_case_sink_mapping_and_relative_urls(report_inputs):
    records, samples, work_dir = report_inputs
    document = build_continuation_comparison_html(
        list(reversed(records)), list(reversed(samples)), work_dir=work_dir
    )
    dom = _parse(document)

    assert len(dom.rows) == 8
    assert len(dom.videos) == 16
    videos_by_key = {
        (video["data-case-group"], int(video["data-sink"])): video
        for video in dom.videos
    }
    for record in records:
        for sink_size in (0, 1):
            video = videos_by_key[(record.case_group, sink_size)]
            assert video["controls"] is None
            assert not video["src"].startswith("/")
            assert video["src"].endswith(
                f"continuation/{record.case_group}/sink{sink_size}.mp4"
            )
            assert video["src"] not in {
                str(Path(sample["output_video"]).resolve()) for sample in samples
            }


def test_report_contains_two_native_videos_and_sync_dom_per_row(report_inputs):
    records, samples, work_dir = report_inputs
    document = build_continuation_comparison_html(records, samples, work_dir=work_dir)
    dom = _parse(document)

    assert len(dom.actions) == 8 * 4
    assert {action["data-sync-action"] for action in dom.actions} == {
        "play",
        "pause",
        "seek",
        "reset",
    }
    for row in dom.rows:
        group = row["data-case-group"]
        row_videos = [v for v in dom.videos if v["data-case-group"] == group]
        assert [int(video["data-sink"]) for video in row_videos] == [0, 1]

    assert "querySelectorAll('[data-sync-row]')" in document
    assert "void video.play()" in document
    assert "video.pause()" in document
    assert "video.currentTime = ratio * video.duration" in document
    assert "video.currentTime = 0" in document


def test_report_displays_prompts_seed_anchor_and_fixed_boundaries(report_inputs):
    records, samples, work_dir = report_inputs
    document = build_continuation_comparison_html(records, samples, work_dir=work_dir)

    for record in records:
        assert record.cat_id in document
        assert record.action_order in document
        assert record.action_a_prompt in document
        assert record.hold_prompt in document
        assert record.action_b_prompt in document
    assert document.count("seed: 1") == 16
    assert document.count("soft re-anchor: true") == 8
    dash = chr(0x2013)
    assert f"A 0{dash}23 · HOLD 24{dash}39 · B 40{dash}63" in document
    assert f"A 0{dash}92 · HOLD 93{dash}156 · B 157{dash}252" in document
    lowered = document.lower()
    for forbidden in ("score", "winner", "recommendation", "排名"):
        assert forbidden not in lowered


@pytest.mark.parametrize("bad_sink", [-1, 2, True, "1"])
def test_report_rejects_invalid_sink(report_inputs, bad_sink):
    records, samples, work_dir = report_inputs
    samples[0] = {**samples[0], "sink_size": bad_sink}
    with pytest.raises(ValueError, match="sink_size must be exactly 0 or 1"):
        build_continuation_comparison_html(records, samples, work_dir=work_dir)


def test_report_rejects_duplicate_and_missing_case_sink_mapping(report_inputs):
    records, samples, work_dir = report_inputs
    samples[-1] = dict(samples[0])
    with pytest.raises(ValueError, match="duplicate continuation sample"):
        build_continuation_comparison_html(records, samples, work_dir=work_dir)

    with pytest.raises(ValueError, match="exactly 16 samples"):
        build_continuation_comparison_html(records, samples[:-1], work_dir=work_dir)


def test_report_rejects_unknown_case_group(report_inputs):
    records, samples, work_dir = report_inputs
    samples[0] = {**samples[0], "case_group": "unknown_case"}
    with pytest.raises(ValueError, match="unknown case_group"):
        build_continuation_comparison_html(records, samples, work_dir=work_dir)


def test_report_rejects_output_video_outside_work_dir(report_inputs, tmp_path_factory):
    records, samples, work_dir = report_inputs
    outside = tmp_path_factory.mktemp("outside") / "sink0.mp4"
    outside.write_bytes(b"video")
    samples[0] = {**samples[0], "output_video": outside}

    with pytest.raises(ValueError, match="must be contained in work_dir"):
        build_continuation_comparison_html(records, samples, work_dir=work_dir)


def test_report_rejects_sink_variants_with_different_seeds(report_inputs):
    records, samples, work_dir = report_inputs
    samples[1] = {**samples[1], "seed": 2}
    with pytest.raises(ValueError, match="must use the same seed"):
        build_continuation_comparison_html(records, samples, work_dir=work_dir)
