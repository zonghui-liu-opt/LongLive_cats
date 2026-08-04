#!/usr/bin/env python3
"""Prepare the cat action-sequence fixture for LongLive BF16 TI2V inference.

The source metadata contains a Russian forest cat example for the sequence
"forward jump -> teaser wand".  This script preserves that first-frame identity,
adds a physically causal yarn-ball action, and writes the image/prompt pair
expected by ``ImagePromptDataset``.

The generated PNG remains ordinary 8-bit RGB data.  ``inference.py`` converts
the decoded image and its VAE latent to BF16 at runtime.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = (
    REPO_ROOT
    / "testsets"
    / "metadata_12cases_two_actions_480x832_253frames.csv"
)
DEFAULT_OUTPUT = REPO_ROOT / "testsets" / "processed_bf16_ti2v"
OUTPUT_STEM = "russian_forest_jump_wand_yarn"
TARGET_SIZE = (1280, 704)  # width, height supported by Wan2.2-TI2V-5B
LATENT_FRAMES = 64
LATENT_CHANNELS = 48
LATENT_HEIGHT = 44
LATENT_WIDTH = 80
NUM_FRAMES_PER_BLOCK = 8
TEMPORAL_COMPRESSION_RATIO = 4
FPS = 24


# Keep this as one physical line.  ImagePromptDataset treats every non-empty
# line as a new shot and would otherwise insert a scene-transition prefix.
ACTION_PROMPT = (
    "首帧是一只居中蹲坐的俄罗斯森林长毛猫，棕褐色虎斑长毛蓬松，浅绿色眼睛明亮。"
    "摄像机全程静止，纯白水平地面和背景，全景构图，外观与光照连续，不切镜。"
    "0-0.8秒：猫咪保持直立蹲坐。"
    "0.8-3秒：猫咪前倾、后腿屈曲蓄力，后爪蹬地，沿短小抛物线向前跳；"
    "受重力下降，前爪先着地、后爪随后落下，四肢屈曲缓冲，尾巴摆动维持平衡。"
    "3-3.6秒：猫咪在落点调整重心并坐稳，不滑动或瞬移。"
    "3.6-6.3秒：逗猫棒从右侧平滑伸入并左右摆动，猫咪目光跟随，先单爪试探，"
    "再双爪交替扑抓；玩具只因主人牵引或猫爪接触改变方向，随后沿原路退出。"
    "6.3-6.8秒：红色毛线球紧接着从同一右侧受轻推滚入，可见旋转并因地面摩擦减速，"
    "不能凭空出现。"
    "6.8-10.2秒：猫咪前移重心并轻拍毛线球；球受碰撞改向，依靠惯性继续滚动并减速，"
    "短线头贴地随滚动逐步松开。猫咪小步跟随再拍一次，球不能穿过猫爪或无故变向。"
    "10.2-10.54秒：猫咪用双爪夹停毛线球并稳定坐好。"
    "猫咪、尾巴和出现后的玩具始终完整留在画面内。"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    """Prefer repository-relative paths while supporting temporary outputs."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _select_source(metadata_path: Path) -> tuple[dict[str, str], Path]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        matches = [
            row
            for row in rows
            if row.get("cat_id") == "russian_forest"
            and row.get("action_order") == "jump_then_toy"
            and row.get("prompt_style") == "absolute_timeline"
        ]

    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one russian_forest/jump_then_toy/absolute_timeline "
            f"row in {metadata_path}, found {len(matches)}."
        )

    row = matches[0]
    source_image = metadata_path.parent / row["input_image"]
    if not source_image.is_file():
        raise FileNotFoundError(f"Metadata image does not exist: {source_image}")
    return row, source_image


def _prepare_image(source: Path, destination: Path) -> dict[str, object]:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        source_size = image.size
        resized = ImageOps.contain(
            image,
            TARGET_SIZE,
            method=Image.Resampling.LANCZOS,
        )
        canvas = Image.new("RGB", TARGET_SIZE, color=(255, 255, 255))
        offset = (
            (TARGET_SIZE[0] - resized.width) // 2,
            (TARGET_SIZE[1] - resized.height) // 2,
        )
        canvas.paste(resized, offset)
        canvas.save(destination, format="PNG", optimize=True)

    return {
        "source_size_wh": list(source_size),
        "resized_size_wh": [resized.width, resized.height],
        "target_size_wh": list(TARGET_SIZE),
        "padding_left_top": list(offset),
        "resize_mode": "aspect-preserving contain with centered white padding",
    }


def prepare(metadata_path: Path, output_dir: Path, force: bool = False) -> dict[str, object]:
    metadata_path = metadata_path.resolve()
    output_dir = output_dir.resolve()
    row, source_image = _select_source(metadata_path)

    images_dir = output_dir / "images"
    prompts_dir = output_dir / "prompts"
    image_path = images_dir / f"{OUTPUT_STEM}.png"
    prompt_path = prompts_dir / f"{OUTPUT_STEM}.txt"
    manifest_path = output_dir / "manifest.json"
    readme_path = output_dir / "README.md"

    existing = [
        path
        for path in (image_path, prompt_path, manifest_path, readme_path)
        if path.exists()
    ]
    if existing and not force:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing outputs: {formatted}. Use --force.")

    images_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    image_info = _prepare_image(source_image, image_path)
    prompt_path.write_text(ACTION_PROMPT + "\n", encoding="utf-8")

    decoded_frames = (LATENT_FRAMES - 1) * TEMPORAL_COMPRESSION_RATIO + 1
    manifest: dict[str, object] = {
        "format": "longlive-image-prompt-v1",
        "task": "BF16 TI2V cat action sequence",
        "actions": ["forward_jump", "teaser_wand", "yarn_ball"],
        "source": {
            "metadata": _display_path(metadata_path),
            "case_group": row["case_group"],
            "prompt_style": row["prompt_style"],
            "image": _display_path(source_image),
            "image_sha256": _sha256(source_image),
        },
        "prepared": {
            "image": _display_path(image_path),
            "image_sha256": _sha256(image_path),
            "prompt": _display_path(prompt_path),
            "prompt_non_empty_lines": 1,
            "prompt_strategy": "single-line absolute timeline to preserve one continuous shot",
            **image_info,
        },
        "inference_shape": [
            1,
            LATENT_FRAMES,
            LATENT_CHANNELS,
            LATENT_HEIGHT,
            LATENT_WIDTH,
        ],
        "num_frame_per_block": NUM_FRAMES_PER_BLOCK,
        "num_blocks": LATENT_FRAMES // NUM_FRAMES_PER_BLOCK,
        "temporal_compression_ratio": TEMPORAL_COMPRESSION_RATIO,
        "decoded_frames": decoded_frames,
        "fps": FPS,
        "duration_seconds": decoded_frames / FPS,
        "precision_note": "PNG is uint8 RGB; inference.py casts image and VAE latent to torch.bfloat16.",
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    readme_path.write_text(
        "# BF16 TI2V cat action sequence\n\n"
        "Generated by `scripts/prepare_cat_ti2v_testset.py`. The PNG and TXT "
        "share a stem so `ImagePromptDataset` can load them directly.\n\n"
        "The prompt is intentionally one non-empty line: splitting the three actions "
        "across lines would make the current loader mark them as separate scene cuts.\n\n"
        "Use `configs/inference_i2v_cat_sequence_bf16.yaml` after setting a merged "
        "I2V BF16 checkpoint compatible with the configured sampling steps.\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite this fixture's generated files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare(args.metadata, args.output_dir, force=args.force)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
