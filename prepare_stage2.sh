#!/usr/bin/env bash

set -euo pipefail

# Teacher manifest creation is CPU-only. Keep the later Stage-2 baseline
# preflight environment aligned with its locked 8-rank topology.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export STAGE2_PYTHON=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python
export STAGE2_TORCHRUN="$(dirname "$STAGE2_PYTHON")/torchrun"
export STAGE1_BASE=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/checkpoints/stage1/converted_causal_base.pt
export STAGE1_CKPT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/results/stage1_600cats_phaseA10epochs_phaseB20epochs/checkpoint_model_003075
export G_MERGED=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/checkpoints/stage2/stage1_step3075_ema_merged.pt
export G_MANIFEST=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/checkpoints/stage2/stage1_step3075_ema_merged.manifest.json
export TEACHER_MANIFEST=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/checkpoints/stage2/real_score_teacher.manifest.json
export TEACHER_CKPT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/ckpts
export TEACHER_PROVENANCE_RECORD=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/merge_manifest.json
export ARCH_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B

# These are semantic operator attestations, not properties recoverable from a
# safetensors file. Set both to 1 only after checking the DiffSynth SFT config:
# full-sequence bidirectional TI2V, flow prediction, and one global timestep per
# video (not causal/per-block timesteps).
if [[ "${ATTEST_CAT_DOMAIN_BIDIRECTIONAL_TI2V:-0}" != 1 ]]; then
  echo "Set ATTEST_CAT_DOMAIN_BIDIRECTIONAL_TI2V=1 after verifying the SFT contract" >&2
  exit 1
fi
if [[ "${ATTEST_VIDEO_GLOBAL_FLOW:-0}" != 1 ]]; then
  echo "Set ATTEST_VIDEO_GLOBAL_FLOW=1 after verifying the SFT timestep/objective" >&2
  exit 1
fi

# Bind provenance to the complete DiffSynth merge record. Its file SHA covers
# the baseline paths, LoRA path/SHA, alpha, dtype, merged-state checksum, save
# verification, reload verification, and deterministic mode together. The
# Stage-2 manifest builder separately hashes the actual teacher safetensors.
provenance_fields="$("$STAGE2_PYTHON" -I -B - \
  "$TEACHER_PROVENANCE_RECORD" "$ARCH_ROOT" "$TEACHER_CKPT" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import re
import sys

from safetensors import safe_open

manifest_path = Path(sys.argv[1]).expanduser().resolve()
architecture_root = Path(sys.argv[2]).expanduser().resolve()
teacher_root = Path(sys.argv[3]).expanduser().resolve()
if not manifest_path.is_file():
    raise FileNotFoundError(manifest_path)

value = json.loads(manifest_path.read_text(encoding="utf-8"))
expected = {
    "format": "DiffSynth-Studio Wan2.2-TI2V-5B merged LoRA",
    "merge_dtype": "torch.bfloat16",
    "merged_dit_files": ["diffusion_pytorch_model.safetensors"],
    "save_checksum_verified": True,
    "reload_verified": True,
    "deterministic": "strict",
}
wrong = {
    key: {"expected": expected_value, "actual": value.get(key)}
    for key, expected_value in expected.items()
    if value.get(key) != expected_value
}
if wrong:
    raise RuntimeError(f"DiffSynth merge manifest contract mismatch: {wrong}")

recorded_base = Path(value.get("baseline_model_root", "")).expanduser().resolve()
if recorded_base != architecture_root:
    raise RuntimeError(
        "DiffSynth baseline_model_root differs from Stage-2 ARCH_ROOT: "
        f"{recorded_base} != {architecture_root}"
    )

expected_checkpoint = teacher_root / "diffusion_pytorch_model.safetensors"
if not expected_checkpoint.is_file():
    raise FileNotFoundError(expected_checkpoint)

tensor_bytes = 0
with safe_open(str(expected_checkpoint), framework="pt", device="cpu") as handle:
    for key in handle.keys():
        tensor_slice = handle.get_slice(key)
        if tensor_slice.get_dtype() != "BF16":
            raise TypeError(
                f"Merged teacher tensor is not BF16: {key}={tensor_slice.get_dtype()}"
            )
        tensor_bytes += math.prod(tensor_slice.get_shape()) * 2
if tensor_bytes != value.get("merged_total_size_bytes"):
    raise RuntimeError(
        "Actual safetensors payload size differs from DiffSynth merge manifest: "
        f"{tensor_bytes} != {value.get('merged_total_size_bytes')}"
    )

merged_state_sha256 = value.get("merged_state_sha256")
if not isinstance(merged_state_sha256, str) or re.fullmatch(
    r"[0-9a-f]{64}", merged_state_sha256
) is None:
    raise ValueError("DiffSynth merged_state_sha256 must be 64 lowercase hex digits")

lora_sha256 = value.get("lora_sha256")
if not isinstance(lora_sha256, str) or re.fullmatch(
    r"[0-9a-f]{64}", lora_sha256
) is None:
    raise ValueError("DiffSynth lora_sha256 must be 64 lowercase hex digits")
if not isinstance(value.get("lora_path"), str) or not value["lora_path"]:
    raise ValueError("DiffSynth merge manifest lacks lora_path")
lora_path = Path(value["lora_path"]).expanduser().resolve()
if not lora_path.is_file():
    raise FileNotFoundError(lora_path)
digest = hashlib.sha256()
with lora_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
if digest.hexdigest() != lora_sha256:
    raise RuntimeError("LoRA file SHA256 differs from DiffSynth merge manifest")

source_kind = "diffsynth_sft_lora_merge"
source_identifier = f"{lora_path.parent.name}:{lora_path.stem}"
source_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
print(
    "\t".join((source_kind, source_identifier, source_sha256)),
    end="",
)
PY
)"
IFS=$'\t' read -r TEACHER_SOURCE_KIND TEACHER_SOURCE_ID TEACHER_SOURCE_SHA256 \
  <<< "$provenance_fields"
export TEACHER_SOURCE_KIND TEACHER_SOURCE_ID TEACHER_SOURCE_SHA256

test ! -e "$TEACHER_MANIFEST"

"$STAGE2_PYTHON" -I -B scripts/create_stage2_teacher_manifest.py \
  --checkpoint "$TEACHER_CKPT" \
  --architecture-root "$ARCH_ROOT" \
  --output "$TEACHER_MANIFEST" \
  --checkpoint-format wan_native_transformer \
  --state-dict-selector root \
  --source-kind "$TEACHER_SOURCE_KIND" \
  --source-identifier "$TEACHER_SOURCE_ID" \
  --source-sha256 "$TEACHER_SOURCE_SHA256" \
  --conversion-command none \
  --attest-cat-domain-bidirectional-ti2v \
  --attest-video-global-flow

test -s "$TEACHER_MANIFEST"
