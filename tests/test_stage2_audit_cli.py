import inspect
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import audit_stage2_i2v_cache as audit_cli


def test_direct_cli_refuses_nonisolated_startup_before_argparse_shadow(tmp_path):
    shadow_root = tmp_path / "shadow"
    shadow_root.mkdir()
    marker = tmp_path / "argparse-imported.txt"
    (shadow_root / "argparse.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        "raise RuntimeError('argparse shadow executed')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(shadow_root)
    completed = subprocess.run(
        [sys.executable, str(audit_cli.__file__), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "python -I -B scripts/audit_stage2_i2v_cache.py" in completed.stderr
    assert not marker.exists()


def test_isolated_direct_cli_does_not_depend_on_git_state(tmp_path):
    shadow_root = tmp_path / "shadow"
    shadow_root.mkdir()
    marker = tmp_path / "project-imported.txt"
    (shadow_root / "omegaconf.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        "raise RuntimeError('isolated startup imported shadow module')\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["GIT_DIR"] = str(tmp_path / "decoy.git")
    environment["PYTHONPATH"] = str(shadow_root)
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(audit_cli.__file__), "--help"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()


def test_formal_audit_cli_has_no_dirty_or_sample_count_bypass():
    parser = audit_cli._parser()
    subparsers_action = next(
        action
        for action in parser._actions
        if getattr(action, "choices", None) is not None
    )
    audit_help = subparsers_action.choices["audit"].format_help()
    assert "--allow-dirty-code" not in audit_help
    assert "--code-version" not in audit_help
    assert "--expected-num-samples" not in audit_help

    upgrade_help = subparsers_action.choices["upgrade-source-manifest"].format_help()
    assert "--expected-source-manifest-sha256" in upgrade_help
    assert "--operator-id" in upgrade_help
    assert "--operator-attestation" in upgrade_help
    assert "--legacy-source-cache-manifest" in upgrade_help
    assert (
        "code_version"
        not in inspect.signature(audit_cli.audit_stage2_i2v_cache).parameters
    )
    assert (
        "code_version"
        not in inspect.signature(
            audit_cli.upgrade_legacy_source_cache_manifest_text_encoding
        ).parameters
    )


@pytest.mark.parametrize(
    ("argument", "label"),
    [
        ("metadata_path", "--metadata-path"),
        ("cache_dir", "--cache-dir"),
        ("negative_conditioning_manifest", "--negative-conditioning-manifest"),
        ("action_labels_path", "--action-labels-path"),
        ("output_manifest", "--output-manifest"),
    ],
)
def test_formal_audit_cli_rejects_paths_that_drift_from_resolved_config(
    tmp_path, monkeypatch, argument, label
):
    cache_dir = tmp_path / "cache"
    resolved = SimpleNamespace(
        cache_dir=cache_dir,
        metadata_path=tmp_path / "metadata.csv",
        negative_conditioning_manifest=tmp_path / "negative.json",
        action_labels_path=tmp_path / "actions.csv",
    )
    monkeypatch.setattr(audit_cli.OmegaConf, "load", lambda _path: object())
    monkeypatch.setattr(audit_cli, "resolve_stage2_config", lambda _config: resolved)
    values = {
        "config_path": str(tmp_path / "stage2.yaml"),
        "cache_dir": None,
        "metadata_path": None,
        "negative_conditioning_manifest": None,
        "action_labels_path": None,
        "output_manifest": None,
        "source_cache_manifest": str(tmp_path / "attested-source.json"),
        "action_id": ["head_tilt", "jump", "toy_play"],
    }
    values[argument] = str(tmp_path / f"drift-{argument}")
    with pytest.raises(RuntimeError, match=label):
        audit_cli._audit(SimpleNamespace(**values))


@pytest.mark.parametrize("padding_side", ["right", "left"])
def test_wan_text_contract_validator_executes_special_token_and_padding_probe(
    monkeypatch, padding_side
):
    from utils import wan_5b_wrapper

    class _FakeTokenizer:
        def __init__(self, *, name, seq_len, clean):
            assert name.endswith("tokenizer")
            self.seq_len = seq_len
            self.clean = clean
            self.tokenizer = SimpleNamespace(padding_side=padding_side)

        def __call__(self, sequence, *, return_mask, add_special_tokens):
            assert return_mask is True
            batch = len(sequence)
            valid = torch.tensor([4, 3]) if add_special_tokens else torch.tensor([3, 2])
            mask = torch.arange(self.seq_len).view(1, -1) < valid.view(batch, 1)
            return torch.zeros(batch, self.seq_len, dtype=torch.long), mask.long()

    monkeypatch.setattr(wan_5b_wrapper, "HuggingfaceTokenizer", _FakeTokenizer)
    if padding_side == "left":
        with pytest.raises(RuntimeError, match="right padding"):
            wan_5b_wrapper.audit_wan_text_encoding_tokenizer_contract(
                "/fixture/tokenizer"
            )
    else:
        result = wan_5b_wrapper.audit_wan_text_encoding_tokenizer_contract(
            "/fixture/tokenizer"
        )
        assert result["sequence_length"] == 512
        assert result["cleaning"] == "whitespace"
        assert result["add_special_tokens"] is True
        assert result["padding_side"] == "right"
        assert result["embedding_padding_value"] == 0.0
