import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
import pytest

import scripts.run_stage1_training_checkpoints_validation as checkpoint_runner
from scripts.run_stage1_training_checkpoints_validation import (
    _load_prompt_style_review_records,
    _parse_num_latent_frames,
    _parse_steps,
    _prompt_style_comparison_html,
    build_parser,
    run_validation,
)
from utils.config import DEFAULT_NEGATIVE_PROMPT
from utils.stage1_causal_validation import discover_stage1_training_checkpoints
from utils.stage1_checkpoint import write_checkpoint_manifest, write_success_marker
from utils.stage1_io import atomic_write_json


def _write_artifact_checkpoint(root: Path, step: int, base_sha256: str) -> Path:
    directory = root / f"checkpoint_model_{step:06d}"
    directory.mkdir()
    (directory / "adapter_raw.safetensors").write_bytes(b"raw")
    (directory / "adapter_ema.safetensors").write_bytes(b"ema")
    (directory / "resolved_config.yaml").write_text("model_kwargs: {}\n", encoding="utf-8")
    atomic_write_json(directory / "base_reference.json", {"base_sha256": base_sha256})
    write_checkpoint_manifest(
        directory,
        completed_step=step,
        world_size=6,
        sequence_parallel_size=3,
        data_parallel_size=2,
        resumable=False,
    )
    write_success_marker(directory, resumable=False)
    return directory


def _write_testset(root: Path) -> Path:
    image_path = root / "cat.png"
    Image.new("RGB", (832, 480), color=(220, 220, 220)).save(image_path)
    metadata = root / "metadata.csv"
    with metadata.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_image", "prompt", "height", "width", "bucket"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "input_image": image_path.name,
                "prompt": "a cat moves",
                "height": 480,
                "width": 832,
                "bucket": "landscape",
            }
        )
    return metadata


PROMPT_STYLE_FIELDNAMES = [
    "input_image",
    "prompt",
    "height",
    "width",
    "bucket",
    "case_group",
    "prompt_style",
    "cat_id",
    "action_order",
]


def _prompt_style_rows() -> list[dict[str, object]]:
    common = {
        "input_image": "cat.png",
        "height": 480,
        "width": 832,
        "bucket": "landscape",
        "case_group": "ragdoll_jump_then_toy",
        "cat_id": "ragdoll",
        "action_order": "jump_then_toy",
    }
    return [
        {
            **common,
            "prompt": "绝对 <时间轴> 提示，完整保留。",
            "prompt_style": "absolute_timeline",
        },
        {
            **common,
            "prompt": "顺序提示 & 完整保留。",
            "prompt_style": "sequential",
        },
        {
            **common,
            "prompt": "阶段相对时间提示，第二阶段重新归零。",
            "prompt_style": "phase_relative",
        },
    ]


def _write_prompt_style_testset(
    root: Path,
    *,
    rows: list[dict[str, object]] | None = None,
    fieldnames: list[str] | None = None,
    name: str = "prompt_style_metadata.csv",
) -> Path:
    Image.new("RGB", (832, 480), color=(220, 220, 220)).save(root / "cat.png")
    Image.new("RGB", (832, 480), color=(200, 200, 200)).save(root / "other.png")
    rows = rows or _prompt_style_rows()
    fieldnames = fieldnames or PROMPT_STYLE_FIELDNAMES
    metadata = root / name
    with metadata.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return metadata


def _runner_args(
    root: Path,
    *,
    metadata: Path,
    checkpoints: list[Path],
    work_dir: Path,
    **overrides,
) -> SimpleNamespace:
    values = {
        "metadata": str(metadata),
        "training_root": None,
        "training_checkpoint": [str(checkpoint) for checkpoint in checkpoints],
        "steps": None,
        "work_dir": str(work_dir),
        "base_checkpoint": str(root / "base.pt"),
        "base_manifest": str(root / "base.manifest.json"),
        "source_checkpoint": None,
        "architecture_root": str(root / "architecture"),
        "t5_checkpoint": str(root / "t5.pt"),
        "tokenizer_dir": str(root / "tokenizer"),
        "vae_checkpoint": str(root / "vae.pt"),
        "num_latent_frames": 24,
        "allow_repeated_input_images": False,
        "comparison_mode": "checkpoint",
        "sampling_steps": 50,
        "guidance_scale": 5.0,
        "seed": 1,
        "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
        "merge_device": "cpu",
        "minimum_first_frame_psnr_db": 12.0,
        "minimum_frame_std": 5.0,
        "minimum_temporal_abs_diff": 0.05,
        "keep_merged": False,
        "continue_on_error": False,
        "skip_base_finite_check": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_checkpoint_discovery_is_sorted_selectable_and_rejects_incomplete(tmp_path):
    base_sha256 = "a" * 64
    _write_artifact_checkpoint(tmp_path, 300, base_sha256)
    selected = _write_artifact_checkpoint(tmp_path, 75, base_sha256)

    assert discover_stage1_training_checkpoints(
        tmp_path, expected_base_sha256=base_sha256
    ) == [selected.resolve(), (tmp_path / "checkpoint_model_000300").resolve()]
    assert discover_stage1_training_checkpoints(
        tmp_path, steps=[300], expected_base_sha256=base_sha256
    ) == [(tmp_path / "checkpoint_model_000300").resolve()]
    with pytest.raises(RuntimeError, match="missing"):
        discover_stage1_training_checkpoints(
            tmp_path, steps=[150], expected_base_sha256=base_sha256
        )

    (tmp_path / "checkpoint_model_000450").mkdir()
    with pytest.raises(RuntimeError, match="incomplete"):
        discover_stage1_training_checkpoints(tmp_path, steps=[75])


def test_parse_steps_is_strict():
    assert _parse_steps("75,300,750") == [75, 300, 750]
    with pytest.raises(argparse.ArgumentTypeError, match="duplicates"):
        _parse_steps("75,75")


def test_batch_runner_uses_ema_merged_checkpoint_and_writes_comparison(tmp_path):
    base_sha256 = "b" * 64
    checkpoint = _write_artifact_checkpoint(tmp_path, 75, base_sha256)
    metadata = _write_testset(tmp_path)
    work_dir = tmp_path / "validation"
    command_calls = []

    def fake_base_validator(*_args, **_kwargs):
        return {"status": "pass", "output_sha256": base_sha256}

    def fake_merge(**kwargs):
        output_path = Path(kwargs["output_path"])
        output_path.write_bytes(b"merged-ema")
        atomic_write_json(kwargs["output_manifest_path"], {"kind": "ema"})
        assert Path(kwargs["training_checkpoint"]) == checkpoint
        return {"output": {"sha256": "c" * 64}}

    output_video_holder = {}

    def fake_prepare(**kwargs):
        merged_path = Path(kwargs["base_checkpoint"])
        assert merged_path.name == "stage1_causal_ema_merged.pt"
        assert merged_path.read_bytes() == b"merged-ema"
        assert kwargs["negative_prompt"] == DEFAULT_NEGATIVE_PROMPT
        assert kwargs["num_latent_frames"] == 24
        assert kwargs["allow_repeated_input_images"] is False
        prepared_root = Path(kwargs["output_root"])
        config_path = prepared_root / "configs" / "landscape.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("fixture: true\n", encoding="utf-8")
        output_dir = prepared_root / "videos" / "landscape_480x832"
        output_dir.mkdir(parents=True)
        output_video = output_dir / "rank0-0-0_regular.mp4"
        output_video.write_bytes(b"video")
        output_video_holder["path"] = output_video
        atomic_write_json(prepared_root / "prepared_manifest.json", {"fixture": True})
        return {
            "buckets": [
                {
                    "bucket_id": "landscape_480x832",
                    "config_path": str(config_path),
                }
            ]
        }

    def fake_output_validator(*_args, **_kwargs):
        return {
            "status": "pass",
            "sample_count": 1,
            "samples": [
                {
                    "row_id": 0,
                    "output_video": str(output_video_holder["path"]),
                    "metrics": {
                        "first_frame_psnr_db": 31.0,
                        "mean_frame_std": 22.0,
                        "mean_temporal_abs_diff": 1.5,
                    },
                }
            ],
        }

    args = _runner_args(
        tmp_path,
        metadata=metadata,
        checkpoints=[checkpoint],
        work_dir=work_dir,
    )
    report = run_validation(
        args,
        base_validator=fake_base_validator,
        merge_fn=fake_merge,
        prepare_fn=fake_prepare,
        output_validator=fake_output_validator,
        command_runner=lambda *call_args, **call_kwargs: command_calls.append(
            (call_args, call_kwargs)
        ),
    )

    assert report["status"] == "pass"
    assert report["passed_checkpoint_count"] == 1
    assert report["checkpoints"][0]["merged_checkpoint_retained"] is False
    assert not (work_dir / "checkpoint_model_000075" / "stage1_causal_ema_merged.pt").exists()
    assert len(command_calls) == 1
    comparison = (work_dir / "comparison.html").read_text(encoding="utf-8")
    assert "step 000075" in comparison
    assert "rank0-0-0_regular.mp4" in comparison
    persisted = json.loads((work_dir / "validation_report.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "pass"


def test_num_latent_frames_cli_defaults_and_validation(tmp_path):
    required = [
        "--work-dir",
        str(tmp_path / "work"),
        "--base-checkpoint",
        str(tmp_path / "base.pt"),
        "--base-manifest",
        str(tmp_path / "base.json"),
        "--architecture-root",
        str(tmp_path / "architecture"),
        "--t5-checkpoint",
        str(tmp_path / "t5.pt"),
        "--tokenizer-dir",
        str(tmp_path / "tokenizer"),
        "--vae-checkpoint",
        str(tmp_path / "vae.pt"),
    ]
    parser = build_parser()
    defaults = parser.parse_args(required)
    assert defaults.num_latent_frames == 24
    assert defaults.allow_repeated_input_images is False
    assert defaults.comparison_mode == "checkpoint"

    selected = parser.parse_args(
        required
        + [
            "--num-latent-frames",
            "64",
            "--allow-repeated-input-images",
            "--comparison-mode",
            "prompt-style",
        ]
    )
    assert selected.num_latent_frames == 64
    assert selected.allow_repeated_input_images is True
    assert selected.comparison_mode == "prompt-style"
    assert _parse_num_latent_frames("64") == 64

    for invalid in ("0", "7", "25", "not-an-integer"):
        with pytest.raises(SystemExit):
            parser.parse_args(required + ["--num-latent-frames", invalid])


def test_prompt_style_review_metadata_parser_accepts_generic_valid_groups(tmp_path):
    rows = _prompt_style_rows()
    second_group = [
        {
            **row,
            "case_group": "tabby_toy_then_jump",
            "cat_id": "tabby",
            "action_order": "toy_then_jump",
        }
        for row in _prompt_style_rows()
    ]
    metadata = _write_prompt_style_testset(tmp_path, rows=rows + second_group)

    records = _load_prompt_style_review_records(metadata)

    assert len(records) == 6
    assert [record.row_id for record in records] == [0, 1, 2, 3, 4, 5]
    assert {record.case_group for record in records} == {
        "ragdoll_jump_then_toy",
        "tabby_toy_then_jump",
    }


def test_prompt_style_review_metadata_rejects_missing_review_column(tmp_path):
    fieldnames = [name for name in PROMPT_STYLE_FIELDNAMES if name != "cat_id"]
    metadata = _write_prompt_style_testset(tmp_path, fieldnames=fieldnames)

    with pytest.raises(ValueError, match="missing required columns.*cat_id"):
        _load_prompt_style_review_records(metadata)


@pytest.mark.parametrize(
    "styles",
    [
        ["absolute_timeline", "sequential", "sequential"],
        ["absolute_timeline", "sequential"],
    ],
)
def test_prompt_style_review_metadata_rejects_duplicate_or_missing_style(
    tmp_path, styles
):
    rows = _prompt_style_rows()[: len(styles)]
    for row, prompt_style in zip(rows, styles, strict=True):
        row["prompt_style"] = prompt_style
    metadata = _write_prompt_style_testset(tmp_path, rows=rows)

    with pytest.raises(ValueError, match="exactly one row for each prompt_style"):
        _load_prompt_style_review_records(metadata)


def test_prompt_style_review_metadata_rejects_wrong_group_name(tmp_path):
    rows = _prompt_style_rows()
    rows[1]["case_group"] = "ragdoll_wrong_order"
    metadata = _write_prompt_style_testset(tmp_path, rows=rows)

    with pytest.raises(ValueError, match="case_group.*must equal"):
        _load_prompt_style_review_records(metadata)


def test_prompt_style_review_metadata_rejects_mixed_group_image_or_geometry(tmp_path):
    rows = _prompt_style_rows()
    rows[1]["input_image"] = "other.png"
    metadata = _write_prompt_style_testset(tmp_path, rows=rows)

    with pytest.raises(ValueError, match="must use one input image, geometry"):
        _load_prompt_style_review_records(metadata)


def test_prompt_style_html_uses_exact_row_ids_and_relative_urls(tmp_path):
    metadata = _write_prompt_style_testset(tmp_path)
    review_records = _load_prompt_style_review_records(metadata)
    work_dir = tmp_path / "validation"
    output_dir = work_dir / "checkpoint_model_003750" / "prepared" / "videos"
    output_dir.mkdir(parents=True)
    absolute_video = output_dir / "absolute clip.mp4"
    sequential_video = output_dir / "sequential clip.mp4"
    phase_relative_video = output_dir / "phase relative clip.mp4"
    absolute_video.write_bytes(b"absolute")
    sequential_video.write_bytes(b"sequential")
    phase_relative_video.write_bytes(b"phase-relative")
    report = {
        "checkpoints": [
            {
                "optimizer_step": 3750,
                "status": "pass",
                # Deliberately reverse the samples: the HTML must map by row_id.
                "samples": [
                    {"row_id": 2, "output_video": str(phase_relative_video)},
                    {"row_id": 1, "output_video": str(sequential_video)},
                    {"row_id": 0, "output_video": str(absolute_video)},
                ],
            }
        ]
    }

    comparison = _prompt_style_comparison_html(
        report,
        review_records=review_records,
        work_dir=work_dir,
    )

    absolute_url = (
        "checkpoint_model_003750/prepared/videos/absolute%20clip.mp4"
    )
    sequential_url = (
        "checkpoint_model_003750/prepared/videos/sequential%20clip.mp4"
    )
    phase_relative_url = (
        "checkpoint_model_003750/prepared/videos/phase%20relative%20clip.mp4"
    )
    assert absolute_url in comparison
    assert sequential_url in comparison
    assert phase_relative_url in comparison
    assert (
        comparison.index(absolute_url)
        < comparison.index(sequential_url)
        < comparison.index(phase_relative_url)
    )
    assert str(work_dir.resolve()) not in comparison
    assert "ragdoll_jump_then_toy" in comparison
    assert "cat: ragdoll" in comparison
    assert "action order: jump_then_toy" in comparison
    assert "绝对 &lt;时间轴&gt; 提示，完整保留。" in comparison
    assert "顺序提示 &amp; 完整保留。" in comparison
    assert "阶段相对时间提示，第二阶段重新归零。" in comparison
    assert "absolute_timeline</strong> · row 0" in comparison
    assert "sequential</strong> · row 1" in comparison
    assert "phase_relative</strong> · row 2" in comparison

    duplicate = {
        "checkpoints": [
            {
                "optimizer_step": 3750,
                "status": "pass",
                "samples": [
                    {"row_id": 0, "output_video": str(absolute_video)},
                    {"row_id": 0, "output_video": str(sequential_video)},
                ],
            }
        ]
    }
    with pytest.raises(RuntimeError, match="duplicate output row_id"):
        _prompt_style_comparison_html(
            duplicate,
            review_records=review_records,
            work_dir=work_dir,
        )

    missing = {
        "checkpoints": [
            {
                "optimizer_step": 3750,
                "status": "pass",
                "samples": [{"row_id": 0, "output_video": str(absolute_video)}],
            }
        ]
    }
    with pytest.raises(RuntimeError, match="output row_id set mismatch"):
        _prompt_style_comparison_html(
            missing,
            review_records=review_records,
            work_dir=work_dir,
        )


def test_runner_threads_64_latents_and_repeated_image_opt_in(tmp_path, monkeypatch):
    base_sha256 = "d" * 64
    checkpoint = _write_artifact_checkpoint(tmp_path, 3750, base_sha256)
    metadata = _write_testset(tmp_path)
    loader_calls = []
    prepare_calls = []
    real_loader = checkpoint_runner.load_causal_testset_records

    def tracking_loader(metadata_path, *, allow_repeated_input_images=False):
        loader_calls.append(allow_repeated_input_images)
        return real_loader(
            metadata_path,
            allow_repeated_input_images=allow_repeated_input_images,
        )

    monkeypatch.setattr(
        checkpoint_runner,
        "load_causal_testset_records",
        tracking_loader,
    )

    def fake_merge(**kwargs):
        Path(kwargs["output_path"]).write_bytes(b"merged-ema")
        atomic_write_json(kwargs["output_manifest_path"], {"kind": "ema"})
        return {"output": {"sha256": "e" * 64}}

    class PreparationReached(RuntimeError):
        pass

    def stop_after_prepare_call(**kwargs):
        prepare_calls.append(kwargs)
        raise PreparationReached("captured preparation arguments")

    args = _runner_args(
        tmp_path,
        metadata=metadata,
        checkpoints=[checkpoint],
        work_dir=tmp_path / "validation_64",
        num_latent_frames=64,
        allow_repeated_input_images=True,
    )
    with pytest.raises(PreparationReached, match="captured preparation arguments"):
        run_validation(
            args,
            base_validator=lambda *_args, **_kwargs: {
                "status": "pass",
                "output_sha256": base_sha256,
            },
            merge_fn=fake_merge,
            prepare_fn=stop_after_prepare_call,
        )

    assert loader_calls == [True]
    assert len(prepare_calls) == 1
    assert prepare_calls[0]["num_latent_frames"] == 64
    assert prepare_calls[0]["num_frame_per_block"] == 8
    assert prepare_calls[0]["minimum_source_frames"] == 97
    assert prepare_calls[0]["allow_repeated_input_images"] is True


def test_prompt_style_rejects_multiple_checkpoints_before_merge(tmp_path):
    base_sha256 = "f" * 64
    checkpoints = [
        _write_artifact_checkpoint(tmp_path, 3000, base_sha256),
        _write_artifact_checkpoint(tmp_path, 3750, base_sha256),
    ]
    metadata = _write_testset(tmp_path)
    merge_calls = []
    args = _runner_args(
        tmp_path,
        metadata=metadata,
        checkpoints=checkpoints,
        work_dir=tmp_path / "multi_checkpoint_validation",
        comparison_mode="prompt-style",
        allow_repeated_input_images=True,
    )

    with pytest.raises(ValueError, match="requires exactly one checkpoint"):
        run_validation(
            args,
            base_validator=lambda *_args, **_kwargs: {
                "status": "pass",
                "output_sha256": base_sha256,
            },
            merge_fn=lambda **kwargs: merge_calls.append(kwargs),
        )

    assert merge_calls == []


def test_two_action_metadata_matches_locked_experiment_contract(tmp_path):
    project_root = Path(__file__).resolve().parents[1]
    metadata = (
        project_root
        / "testsets"
        / "metadata_12cases_two_actions_480x832_253frames.csv"
    )
    expected_fields = [
        "input_image",
        "prompt",
        "height",
        "width",
        "bucket",
        "case_group",
        "prompt_style",
        "cat_id",
        "action_order",
    ]
    with metadata.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == expected_fields
        rows = list(reader)

    assert len(rows) == 12
    assert Counter(row["cat_id"] for row in rows) == {
        "ragdoll": 3,
        "russian_forest": 3,
        "siamese": 3,
        "tabby": 3,
    }
    assert Counter(row["action_order"] for row in rows) == {
        "jump_then_toy": 12,
    }
    expected_images = {
        "ragdoll": (
            "images_480x832/ragdoll_cat_832x480_no_distortion.png",
            "832",
            "480",
            "portrait",
        ),
        "russian_forest": (
            "images_480x832/russian_forest_cat_480x832_no_distortion.png",
            "480",
            "832",
            "landscape",
        ),
        "siamese": (
            "images_480x832/siamese_cat_832x480_no_distortion.png",
            "832",
            "480",
            "portrait",
        ),
        "tabby": (
            "images_480x832/tabby_cat_832x480_no_distortion.png",
            "832",
            "480",
            "portrait",
        ),
    }
    cat_identity = {
        "ragdoll": "布偶猫",
        "russian_forest": "俄罗斯森林长毛猫",
        "siamese": "暹罗猫",
        "tabby": "狸花猫",
    }
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        assert row["case_group"] == f"{row['cat_id']}_{row['action_order']}"
        assert (
            row["input_image"],
            row["height"],
            row["width"],
            row["bucket"],
        ) == expected_images[row["cat_id"]]
        assert cat_identity[row["cat_id"]] in row["prompt"]
        groups.setdefault(row["case_group"], []).append(row)
    assert len(groups) == 4

    absolute_anchors = (
        "0-1秒：",
        "1-3秒：",
        "3-4秒：",
        "4-5秒：",
        "5-7秒：",
        "7-8秒：",
        "8-10.54秒：",
    )
    sequential_anchors = (
        "开始时",
        "先",
        "恢复坐姿",
        "短暂停顿",
        "随后",
        "恢复最终坐姿",
        "剩余视频保持静止",
    )
    phase_relative_anchors = (
        "第一阶段（相对于该阶段起点）",
        "短暂停顿后进入第二阶段（重新以该阶段起点为0秒）",
        "剩余视频保持静止",
    )
    jump = (
        "猫咪被前上方目标吸引，身体轻微前倾，后腿发力，完成一次自然小幅跳跃，"
        "前爪向前伸出，身体保持协调。"
    )
    toy = (
        "主人在镜头前方轻轻晃动逗猫棒，猫咪目光自然跟随，先抬一只前爪试探，"
        "再用两只前爪交替轻快扑抓，身体主要保持坐姿。"
    )
    final_still = (
        "完成第二个动作并恢复坐姿后，猫咪继续注视镜头前方，身体、四肢和尾巴"
        "保持最终稳定状态直到视频结束。"
    )
    common_invariants = (
        "全程摄像机静止",
        "纯白背景",
        "全景构图",
        "猫咪始终居中",
        "始终100%留在画面内",
        jump,
        toy,
        final_still,
    )
    for case_group, trio in groups.items():
        assert len(trio) == 3
        assert {row["prompt_style"] for row in trio} == set(
            checkpoint_runner.PROMPT_STYLE_ORDER
        )
        assert len(
            {
                (
                    row["input_image"],
                    row["height"],
                    row["width"],
                    row["bucket"],
                    row["cat_id"],
                    row["action_order"],
                )
                for row in trio
            }
        ) == 1
        by_style = {row["prompt_style"]: row for row in trio}
        absolute = by_style["absolute_timeline"]["prompt"]
        sequential = by_style["sequential"]["prompt"]
        phase_relative = by_style["phase_relative"]["prompt"]
        assert all(absolute.count(anchor) == 1 for anchor in absolute_anchors)
        absolute_positions = [absolute.index(anchor) for anchor in absolute_anchors]
        assert absolute_positions == sorted(absolute_positions)
        assert "秒" not in sequential
        sequential_positions = [
            sequential.index(anchor) for anchor in sequential_anchors
        ]
        assert sequential_positions == sorted(sequential_positions)
        phase_relative_positions = [
            phase_relative.index(anchor) for anchor in phase_relative_anchors
        ]
        assert phase_relative_positions == sorted(phase_relative_positions)
        for anchor in ("0-1秒：", "1-3秒：", "3-4秒："):
            assert phase_relative.count(anchor) == 2
        for global_anchor in ("4-5秒：", "5-7秒：", "7-8秒：", "8-10.54秒："):
            assert global_anchor not in phase_relative
        for prompt in (absolute, sequential, phase_relative):
            assert all(invariant in prompt for invariant in common_invariants)
            assert prompt.index(jump) < prompt.index(toy), case_group

    with pytest.raises(ValueError, match="duplicate input image"):
        checkpoint_runner.load_causal_testset_records(metadata)
    causal_records = checkpoint_runner.load_causal_testset_records(
        metadata,
        allow_repeated_input_images=True,
    )
    review_records = _load_prompt_style_review_records(metadata)
    assert len(causal_records) == len(review_records) == 12
    assert [record.row_id for record in causal_records] == list(range(12))

    work_dir = tmp_path / "validation"
    samples = [
        {
            "row_id": record.row_id,
            "output_video": str(
                work_dir
                / "checkpoint_model_003750"
                / "prepared"
                / "videos"
                / f"row_{record.row_id:04d}.mp4"
            ),
        }
        for record in reversed(review_records)
    ]
    comparison = _prompt_style_comparison_html(
        {
            "checkpoints": [
                {
                    "optimizer_step": 3750,
                    "status": "pass",
                    "samples": samples,
                }
            ]
        },
        review_records=review_records,
        work_dir=work_dir,
    )
    assert comparison.count('<tr><td class="case">') == 4
    assert comparison.count("<video ") == 12
    assert comparison.count("absolute_timeline</strong>") == 4
    assert comparison.count("sequential</strong>") == 4
    assert comparison.count("phase_relative</strong>") == 4
    assert str(work_dir.resolve()) not in comparison
    for case_group in groups:
        assert f"<strong>{case_group}</strong>" in comparison
    for row_id in range(12):
        assert f"row_{row_id:04d}.mp4" in comparison

    legacy_metadata = project_root / "testsets" / "metadata_6cases_480x832.csv"
    assert hashlib.sha256(legacy_metadata.read_bytes()).hexdigest() == (
        "e59a14deb87ee3dc34236ddce30ad9478f0082d5a5cf35cc89b15cedd16bd94e"
    )
