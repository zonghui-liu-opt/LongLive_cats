import csv
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image
import pytest
from safetensors.torch import save_file
import torch

from scripts.convert_diffsynth_wan22_to_longlive import convert_diffsynth_checkpoint
from scripts.prepare_stage1_causal_testsets import build_parser as build_prepare_parser
from utils.stage1_io import atomic_write_json, canonical_json_sha256
from utils.stage1_causal_validation import (
    load_causal_testset_records,
    prepare_causal_testsets,
    validate_causal_testset_outputs,
    validate_converted_causal_base,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(3, 2)


class TinyWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyModel()


def _write_metadata(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["input_image", "prompt", "height", "width", "bucket"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _model_paths(root: Path):
    architecture_root = root / "architecture"
    tokenizer_dir = root / "tokenizer"
    architecture_root.mkdir()
    tokenizer_dir.mkdir()
    t5_checkpoint = root / "t5.pt"
    vae_checkpoint = root / "vae.pt"
    base_checkpoint = root / "base.pt"
    for path in (t5_checkpoint, vae_checkpoint, base_checkpoint):
        path.write_bytes(b"fixture")
    return {
        "base_checkpoint": base_checkpoint,
        "architecture_root": architecture_root,
        "t5_checkpoint": t5_checkpoint,
        "tokenizer_dir": tokenizer_dir,
        "vae_checkpoint": vae_checkpoint,
    }


def _fake_carrier_writer(_image, output, *, frame_count, fps):
    Path(output).write_bytes(f"frames={frame_count},fps={fps}".encode())


def _fake_carrier_probe(path):
    name = str(path)
    portrait = "portrait_832x480" in name
    payload = Path(path).read_text(encoding="utf-8")
    values = dict(field.split("=", 1) for field in payload.split(","))
    return {
        "width": 480 if portrait else 832,
        "height": 832 if portrait else 480,
        "frame_count": int(values["frames"]),
        "fps": float(values["fps"]),
    }


def test_testsets_are_strictly_parsed_and_split_by_geometry(tmp_path):
    Image.new("RGB", (832, 480), color=(240, 240, 240)).save(tmp_path / "land.png")
    Image.new("RGB", (480, 832), color=(220, 220, 220)).save(tmp_path / "port.png")
    metadata = tmp_path / "metadata.csv"
    _write_metadata(
        metadata,
        [
            {
                "input_image": "land.png",
                "prompt": "landscape cat",
                "height": 480,
                "width": 832,
                "bucket": "landscape",
            },
            {
                "input_image": "port.png",
                "prompt": "portrait cat",
                "height": 832,
                "width": 480,
                "bucket": "portrait",
            },
        ],
    )
    records = load_causal_testset_records(metadata)
    assert [record.row_id for record in records] == [0, 1]
    assert [record.bucket_id for record in records] == [
        "landscape_480x832",
        "portrait_832x480",
    ]

    manifest = prepare_causal_testsets(
        metadata_path=metadata,
        output_root=tmp_path / "prepared",
        **_model_paths(tmp_path),
        carrier_writer=_fake_carrier_writer,
        carrier_probe=_fake_carrier_probe,
    )

    assert manifest["metadata"]["record_count"] == 2
    assert manifest["frame_policy"]["num_latent_frames"] == 24
    assert manifest["frame_policy"]["expected_pixel_frames"] == 93
    assert manifest["frame_policy"]["carrier_frames"] == 97
    assert [bucket["bucket_id"] for bucket in manifest["buckets"]] == [
        "landscape_480x832",
        "portrait_832x480",
    ]
    configs = [OmegaConf.load(bucket["config_path"]) for bucket in manifest["buckets"]]
    assert list(configs[0].data.image_or_video_shape) == [1, 24, 48, 30, 52]
    assert list(configs[1].data.image_or_video_shape) == [1, 24, 48, 52, 30]
    assert configs[0].model_kwargs.init_weights is False
    assert configs[0].model_kwargs.local_attn_size == -1
    assert configs[0].model_kwargs.sink_size == 0
    assert configs[0].inference.local_attn_size == -1
    assert configs[0].inference.sink_size == 0
    assert configs[0].model_paths.vae_checkpoint.endswith("vae.pt")
    assert configs[0].inference.negative_prompt == manifest["sampling"]["negative_prompt"]
    assert manifest["sampling"]["solver"] == "unipc"
    assert "adapter" not in configs[0]


def test_repeated_input_images_require_explicit_opt_in_across_preparation(tmp_path):
    Image.new("RGB", (832, 480), color=(240, 240, 240)).save(tmp_path / "cat.png")
    metadata = tmp_path / "metadata.csv"
    _write_metadata(
        metadata,
        [
            {
                "input_image": "cat.png",
                "prompt": "cat jumps and then sits",
                "height": 480,
                "width": 832,
                "bucket": "landscape",
            },
            {
                "input_image": "cat.png",
                "prompt": "cat plays and then sits",
                "height": 480,
                "width": 832,
                "bucket": "landscape",
            },
        ],
    )

    with pytest.raises(ValueError, match="duplicate input image"):
        load_causal_testset_records(metadata)

    records = load_causal_testset_records(
        metadata,
        allow_repeated_input_images=True,
    )
    assert len(records) == 2
    assert records[0].input_image == records[1].input_image
    assert records[0].image_sha256 == records[1].image_sha256
    assert records[0].row_sha256 != records[1].row_sha256

    model_paths = _model_paths(tmp_path)
    with pytest.raises(ValueError, match="duplicate input image"):
        prepare_causal_testsets(
            metadata_path=metadata,
            output_root=tmp_path / "prepared-default",
            **model_paths,
            carrier_writer=_fake_carrier_writer,
            carrier_probe=_fake_carrier_probe,
        )

    manifest = prepare_causal_testsets(
        metadata_path=metadata,
        output_root=tmp_path / "prepared-opt-in",
        **model_paths,
        allow_repeated_input_images=True,
        carrier_writer=_fake_carrier_writer,
        carrier_probe=_fake_carrier_probe,
    )
    prepared_records = manifest["buckets"][0]["records"]
    assert manifest["metadata"]["record_count"] == 2
    assert [record["row_id"] for record in prepared_records] == [0, 1]
    assert prepared_records[0]["image_sha256"] == prepared_records[1]["image_sha256"]
    assert prepared_records[0]["row_sha256"] != prepared_records[1]["row_sha256"]


@pytest.mark.parametrize(
    ("gate", "error_type", "message"),
    [
        ("missing", FileNotFoundError, "missing input image"),
        ("non_rgb", ValueError, "input image must be RGB"),
        ("size", ValueError, "image size"),
        ("exif", ValueError, "EXIF orientation"),
        ("bucket", ValueError, "does not match"),
        ("prompt", ValueError, "prompt must be non-empty"),
    ],
)
def test_repeated_image_opt_in_preserves_other_validation_gates(
    tmp_path,
    gate,
    error_type,
    message,
):
    image_path = tmp_path / "cat.png"
    if gate == "non_rgb":
        Image.new("L", (832, 480)).save(image_path)
    elif gate == "size":
        Image.new("RGB", (816, 480)).save(image_path)
    elif gate == "exif":
        exif = Image.Exif()
        exif[274] = 6
        Image.new("RGB", (832, 480)).save(image_path, exif=exif)
    elif gate != "missing":
        Image.new("RGB", (832, 480)).save(image_path)

    metadata = tmp_path / "metadata.csv"
    _write_metadata(
        metadata,
        [
            {
                "input_image": "cat.png",
                "prompt": "" if gate == "prompt" else "cat",
                "height": 480,
                "width": 832,
                "bucket": "portrait" if gate == "bucket" else "landscape",
            }
        ],
    )
    with pytest.raises(error_type, match=message):
        load_causal_testset_records(
            metadata,
            allow_repeated_input_images=True,
        )


def test_64_latent_preparation_records_253_frames_and_causal_attention(tmp_path):
    Image.new("RGB", (832, 480), color=(240, 240, 240)).save(tmp_path / "cat.png")
    metadata = tmp_path / "metadata.csv"
    _write_metadata(
        metadata,
        [
            {
                "input_image": "cat.png",
                "prompt": "cat performs two actions",
                "height": 480,
                "width": 832,
                "bucket": "landscape",
            }
        ],
    )

    manifest = prepare_causal_testsets(
        metadata_path=metadata,
        output_root=tmp_path / "prepared",
        **_model_paths(tmp_path),
        num_latent_frames=64,
        carrier_writer=_fake_carrier_writer,
        carrier_probe=_fake_carrier_probe,
    )

    assert manifest["frame_policy"] == {
        "num_latent_frames": 64,
        "num_frame_per_block": 8,
        "temporal_compression_ratio": 4,
        "expected_pixel_frames": 253,
        "carrier_frames": 253,
        "fps": 24,
    }
    config = OmegaConf.load(manifest["buckets"][0]["config_path"])
    assert list(config.data.image_or_video_shape) == [1, 64, 48, 30, 52]
    assert config.num_output_frames == 64
    assert config.model_kwargs.local_attn_size == -1
    assert config.model_kwargs.sink_size == 0
    assert config.inference.local_attn_size == -1
    assert config.inference.sink_size == 0
    assert config.inference.streaming_vae is False


def test_prepare_cli_exposes_repeated_image_opt_in():
    required_args = [
        "--output-root",
        "prepared",
        "--base-checkpoint",
        "base.pt",
        "--architecture-root",
        "architecture",
        "--t5-checkpoint",
        "t5.pt",
        "--tokenizer-dir",
        "tokenizer",
        "--vae-checkpoint",
        "vae.pt",
    ]
    parser = build_prepare_parser()
    assert parser.parse_args(required_args).allow_repeated_input_images is False
    assert (
        parser.parse_args(required_args + ["--allow-repeated-input-images"])
        .allow_repeated_input_images
        is True
    )


def test_testset_parser_rejects_bucket_or_geometry_mismatch(tmp_path):
    Image.new("RGB", (832, 480)).save(tmp_path / "cat.png")
    metadata = tmp_path / "metadata.csv"
    _write_metadata(
        metadata,
        [{
            "input_image": "cat.png",
            "prompt": "cat",
            "height": 480,
            "width": 832,
            "bucket": "portrait",
        }],
    )
    with pytest.raises(ValueError, match="does not match"):
        load_causal_testset_records(metadata)


def test_converted_base_audit_checks_hash_schema_bf16_and_source(tmp_path):
    source = tmp_path / "source.safetensors"
    save_file(
        {
            "proj.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
            "proj.bias": torch.tensor([0.25, -0.5]),
        },
        source,
    )
    output = tmp_path / "converted.pt"
    manifest = convert_diffsynth_checkpoint(
        source_checkpoint=source,
        output_path=output,
        model_builder=TinyModel,
        reload_wrapper_builder=TinyWrapper,
        causal_config={
            "local_attn_size": -1,
            "sink_size": 0,
            "num_frame_per_block": 8,
        },
        use_meta_init=False,
    )
    manifest_path = output.with_suffix(".manifest.json")

    report = validate_converted_causal_base(
        output,
        manifest_path,
        source_checkpoint=source,
    )

    assert report["status"] == "pass"
    assert report["strict_reload"] is True
    assert report["source"]["checked"] is True
    assert report["dtype_counts"] == {"bfloat16": 2}
    assert manifest["coverage"]["key_percent"] == 100.0


@pytest.mark.parametrize("output_model_type", ["regular", "lora"])
def test_output_gate_maps_every_bucket_index_and_rejects_stale_files(
    tmp_path,
    output_model_type,
):
    Image.new("RGB", (832, 480), color=(240, 240, 240)).save(tmp_path / "cat.png")
    metadata = tmp_path / "metadata.csv"
    _write_metadata(
        metadata,
        [{
            "input_image": "cat.png",
            "prompt": "cat moves",
            "height": 480,
            "width": 832,
            "bucket": "landscape",
        }],
    )
    prepared_root = tmp_path / "prepared"
    manifest = prepare_causal_testsets(
        metadata_path=metadata,
        output_root=prepared_root,
        **_model_paths(tmp_path),
        carrier_writer=_fake_carrier_writer,
        carrier_probe=_fake_carrier_probe,
    )
    output_dir = Path(manifest["buckets"][0]["output_dir"])
    manifest["buckets"][0]["output_model_type"] = output_model_type
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    atomic_write_json(prepared_root / "prepared_manifest.json", manifest)
    output_video = output_dir / f"rank0-0-0_{output_model_type}.mp4"
    output_video.write_bytes(b"rendered-video")

    report = validate_causal_testset_outputs(
        prepared_root / "prepared_manifest.json",
        probe=lambda _path: {
            "width": 832,
            "height": 480,
            "frame_count": 93,
            "fps": 24.0,
        },
        pixel_metrics=lambda _video, _image: {
            "decoded_frame_count": 93.0,
            "first_frame_psnr_db": 30.0,
            "mean_frame_std": 25.0,
            "mean_temporal_abs_diff": 2.0,
        },
    )
    assert report["status"] == "pass"
    assert report["sample_count"] == 1

    (output_dir / "stale.mp4").write_bytes(b"stale")
    with pytest.raises(RuntimeError, match="output set mismatch"):
        validate_causal_testset_outputs(
            prepared_root / "prepared_manifest.json",
            probe=lambda _path: {},
            pixel_metrics=lambda _video, _image: {},
        )
