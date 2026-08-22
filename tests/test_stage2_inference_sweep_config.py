from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from utils.stage1_io import canonical_json_sha256, sha256_file
from utils.stage2_inference_sweep_config import (
    STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA,
    load_stage2_inference_sweep_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = PROJECT_ROOT / "configs" / "infer_i2v_stage2_baseline.yaml"


def _write_base_config(tmp_path: Path, *, profiles=None) -> Path:
    raw = OmegaConf.to_container(OmegaConf.load(BASELINE_CONFIG), resolve=True)
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text("{}\n", encoding="utf-8")
    raw.update(
        {
            "stage2_checkpoint": str(tmp_path / "checkpoint"),
            "source_cache_manifest": str(source_manifest),
            "architecture_root": str(tmp_path / "architecture"),
            "t5_checkpoint": str(tmp_path / "t5.pth"),
            "tokenizer_dir": str(tmp_path / "tokenizer"),
            "vae_checkpoint": str(tmp_path / "vae.pth"),
            "single_metadata": str(
                PROJECT_ROOT / "testsets" / "metadata_6cases_480x832.csv"
            ),
            "two_action_metadata": str(
                PROJECT_ROOT
                / "testsets"
                / "metadata_8cases_two_actions_continuation_480x832_253frames.csv"
            ),
            "output_root": str(tmp_path / "unused-base-output"),
            "profiles": profiles or ["baseline_c8w16k4s1"],
        }
    )
    path = tmp_path / "base.yaml"
    OmegaConf.save(config=OmegaConf.create(raw), f=path)
    return path


def _raw_sweep() -> dict:
    return {
        "schema": STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA,
        "base_config": "base.yaml",
        "output_root": "sweep-output",
        "profile_set": {
            "named": ["stress_c8w24k4s1"],
            "grids": [
                {
                    "chunk_frames": [8, 4],
                    "local_window_frames": [16, 8],
                    "num_denoising_steps": [4, 2],
                }
            ],
            "cases": [],
        },
        "evaluation": {
            "mode": "formal",
            "seeds": [4, 3, 2, 1],
            "single_row_ids": [5, 4, 3, 2, 1, 0],
            "two_action_row_ids": [7, 6, 5, 4, 3, 2, 1, 0],
        },
    }


def _write_sweep(tmp_path: Path, raw: dict, *, name: str = "sweep.yaml") -> Path:
    path = tmp_path / name
    OmegaConf.save(config=OmegaConf.create(raw), f=path)
    return path


def test_sweep_inherits_strict_baseline_and_canonicalizes_profiles(tmp_path):
    base = _write_base_config(tmp_path)
    path = _write_sweep(tmp_path, _raw_sweep())

    resolved = load_stage2_inference_sweep_config(path)

    assert resolved.schema == STAGE2_INFERENCE_SWEEP_CONFIG_SCHEMA
    assert resolved.base_config_path == str(base.resolve())
    assert resolved.base_config_sha256 == sha256_file(base)
    assert resolved.stage2_checkpoint == str((tmp_path / "checkpoint").resolve())
    assert resolved.source_cache_manifest == str(
        (tmp_path / "source-manifest.json").resolve()
    )
    assert resolved.output_root == str((PROJECT_ROOT / "sweep-output").resolve())
    assert resolved.seeds == (1, 2, 3, 4)
    assert resolved.single_row_ids == tuple(range(6))
    assert resolved.two_action_row_ids == tuple(range(8))
    assert resolved.evaluation_mode == "formal"
    assert resolved.profiles == tuple(spec.name for spec in resolved.rollout_specs)
    topology_order = [
        (
            spec.chunk_frames,
            spec.local_window_frames,
            spec.global_sink_frames,
            spec.num_denoising_steps,
        )
        for spec in resolved.rollout_specs
    ]
    assert topology_order == sorted(topology_order)
    assert len(topology_order) == 9
    assert len(topology_order) == len(set(topology_order))
    assert all(spec.global_sink_frames == 1 for spec in resolved.rollout_specs)
    assert resolved.profile_set_sha256 == canonical_json_sha256(
        [spec.to_dict() for spec in resolved.rollout_specs]
    )
    json.dumps(resolved.to_dict(), allow_nan=False, sort_keys=True)


def test_sweep_hashes_are_order_invariant_and_launch_only_binds_output(tmp_path):
    _write_base_config(tmp_path)
    left = _raw_sweep()
    left["profile_set"]["named"] = ["stress_c8w24k4s1", "c4w12k4s1"]
    left["evaluation"] = {
        "mode": "quick",
        "seeds": [19, 7],
        "single_row_ids": [5, 1],
        "two_action_row_ids": [7, 0, 3],
    }
    right = copy.deepcopy(left)
    right["profile_set"]["named"].reverse()
    grid = right["profile_set"]["grids"][0]
    for value in grid.values():
        value.reverse()
    right["evaluation"]["seeds"].reverse()
    right["evaluation"]["single_row_ids"].reverse()
    right["evaluation"]["two_action_row_ids"].reverse()
    right["output_root"] = "another-output"

    resolved_left = load_stage2_inference_sweep_config(
        _write_sweep(tmp_path, left, name="left.yaml")
    )
    resolved_right = load_stage2_inference_sweep_config(
        _write_sweep(tmp_path, right, name="right.yaml")
    )

    assert resolved_left.profiles == resolved_right.profiles
    assert resolved_left.profile_set_sha256 == resolved_right.profile_set_sha256
    assert resolved_left.seeds == resolved_right.seeds == (7, 19)
    assert resolved_left.single_row_ids == resolved_right.single_row_ids == (1, 5)
    assert (
        resolved_left.two_action_row_ids
        == resolved_right.two_action_row_ids
        == (
            0,
            3,
            7,
        )
    )
    assert resolved_left.contract_hash() == resolved_right.contract_hash()
    assert resolved_left.launch_hash() != resolved_right.launch_hash()


def test_sweep_rejects_nonbaseline_base_and_non_s1_named_profile(tmp_path):
    _write_base_config(tmp_path, profiles=["c4w8k2s1"])
    path = _write_sweep(tmp_path, _raw_sweep())
    with pytest.raises(ValueError, match="only the formal baseline"):
        load_stage2_inference_sweep_config(path)

    _write_base_config(tmp_path)
    raw = _raw_sweep()
    raw["profile_set"]["named"] = ["c8w16k4s4"]
    with pytest.raises(ValueError, match="existing S1 named profile"):
        load_stage2_inference_sweep_config(_write_sweep(tmp_path, raw))


def test_sweep_rejects_duplicate_topology_across_named_and_dynamic(tmp_path):
    _write_base_config(tmp_path)
    raw = _raw_sweep()
    raw["profile_set"] = {
        "named": ["baseline_c8w16k4s1"],
        "grids": [],
        "cases": [
            {
                "chunk_frames": 8,
                "local_window_frames": 16,
                "num_denoising_steps": 4,
            }
        ],
    }
    with pytest.raises(ValueError, match="duplicate rollout topologies"):
        load_stage2_inference_sweep_config(_write_sweep(tmp_path, raw))


def test_sweep_rejects_expansion_beyond_32_before_building_profiles(tmp_path):
    _write_base_config(tmp_path)
    raw = _raw_sweep()
    raw["profile_set"] = {
        "named": [],
        "grids": [
            {
                "chunk_frames": [2],
                "local_window_frames": list(range(2, 25, 2)),
                "num_denoising_steps": [2, 3, 4],
            }
        ],
        "cases": [],
    }
    with pytest.raises(ValueError, match="maximum of 32"):
        load_stage2_inference_sweep_config(_write_sweep(tmp_path, raw))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda raw: raw.update({"extra": 1}), "config root schema mismatch"),
        (
            lambda raw: raw["profile_set"].pop("cases"),
            "profile_set schema mismatch",
        ),
        (
            lambda raw: raw["profile_set"]["grids"][0].update({"sink": [1]}),
            r"grids\[0\] schema mismatch",
        ),
        (
            lambda raw: raw["profile_set"].update(
                {"named": [], "grids": [], "cases": []}
            ),
            "at least one rollout profile",
        ),
        (
            lambda raw: raw["evaluation"].update({"extra": True}),
            "evaluation schema mismatch",
        ),
    ],
)
def test_sweep_rejects_schema_drift(tmp_path, mutate, message):
    _write_base_config(tmp_path)
    raw = _raw_sweep()
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        load_stage2_inference_sweep_config(_write_sweep(tmp_path, raw))


@pytest.mark.parametrize(
    ("evaluation", "message"),
    [
        (
            {
                "mode": "formal",
                "seeds": [1],
                "single_row_ids": list(range(6)),
                "two_action_row_ids": list(range(8)),
            },
            "formal evaluation requires",
        ),
        (
            {
                "mode": "quick",
                "seeds": [],
                "single_row_ids": [0],
                "two_action_row_ids": [0],
            },
            "seeds must be non-empty",
        ),
        (
            {
                "mode": "quick",
                "seeds": [7, 7],
                "single_row_ids": [0],
                "two_action_row_ids": [0],
            },
            "seeds must contain unique",
        ),
        (
            {
                "mode": "quick",
                "seeds": [7],
                "single_row_ids": [6],
                "two_action_row_ids": [0],
            },
            r"single_row_ids\[0\].*\[0, 5\]",
        ),
        (
            {
                "mode": "preview",
                "seeds": [7],
                "single_row_ids": [0],
                "two_action_row_ids": [0],
            },
            "mode must be 'formal' or 'quick'",
        ),
    ],
)
def test_sweep_evaluation_modes_are_strict(tmp_path, evaluation, message):
    _write_base_config(tmp_path)
    raw = _raw_sweep()
    raw["evaluation"] = evaluation
    with pytest.raises(ValueError, match=message):
        load_stage2_inference_sweep_config(_write_sweep(tmp_path, raw))


def test_sweep_loader_rejects_symlink(tmp_path):
    _write_base_config(tmp_path)
    target = _write_sweep(tmp_path, _raw_sweep())
    link = tmp_path / "linked-sweep.yaml"
    link.symlink_to(target)
    with pytest.raises(FileNotFoundError):
        load_stage2_inference_sweep_config(link)
