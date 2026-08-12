#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "STAGE2_PRETRAIN_CHECK_FAILED (line ${LINENO})" >&2' ERR

SOURCE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE2_WORK_ROOT="${STAGE2_WORK_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_pretrain_check}"

# 正式数据工具要求执行仓库绝对干净。自动创建只含已提交代码的副本，
# checkpoints/results/logs 全部放在副本外面。
if [[ "${STAGE2_PRETRAIN_INNER:-0}" != "1" ]]; then
  SOURCE_COMMIT="$(git -C "$SOURCE_REPO" rev-parse HEAD)"
  SOURCE_BRANCH="$(git -C "$SOURCE_REPO" branch --show-current)"
  [[ -n "$SOURCE_BRANCH" ]] || { echo "请先切到 stage-2 分支。" >&2; exit 1; }
  CLEAN_REPO="$STAGE2_WORK_ROOT/code-${SOURCE_COMMIT:0:12}"
  mkdir -p "$STAGE2_WORK_ROOT"
  if [[ ! -e "$CLEAN_REPO" ]]; then
    git clone --quiet --no-local --branch "$SOURCE_BRANCH" "$SOURCE_REPO" "$CLEAN_REPO"
  fi
  [[ "$(git -C "$CLEAN_REPO" rev-parse HEAD)" == "$SOURCE_COMMIT" ]] || {
    echo "干净代码副本版本不一致，请换一个 STAGE2_WORK_ROOT。" >&2
    exit 1
  }
  export STAGE2_PRETRAIN_INNER=1
  export STAGE2_SOURCE_COMMIT="$SOURCE_COMMIT"
  exec bash "$CLEAN_REPO/prepare_stage2.sh"
fi

REPO_ROOT="$SOURCE_REPO"
cd "$REPO_ROOT"
[[ "$(git rev-parse HEAD)" == "${STAGE2_SOURCE_COMMIT:?}" ]]
[[ -z "$(git status --porcelain=v1 --untracked-files=all)" ]]
[[ -z "$(git ls-files --others --ignored --exclude-standard)" ]]

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$STAGE2_WORK_ROOT/pycache"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

STAGE2_PYTHON="${STAGE2_PYTHON:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python}"
STAGE2_TORCHRUN="${STAGE2_TORCHRUN:-$(dirname "$STAGE2_PYTHON")/torchrun}"
STAGE2_CONFIG="configs/train_i2v_stage2_600cats.yaml"

ARCH_ROOT="${ARCH_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B}"
TEACHER_CKPT="${TEACHER_CKPT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/ckpts}"
TEACHER_PROVENANCE_RECORD="${TEACHER_PROVENANCE_RECORD:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/merge_manifest.json}"
STAGE1_BASE="${STAGE1_BASE:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/checkpoints/stage1/converted_causal_base.pt}"
STAGE1_CKPT="${STAGE1_CKPT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/results/stage1_600cats_3750steps}"
METADATA_600="${METADATA_600:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/metadata_600clips_480x832_buckets.csv}"
STAGE1_CACHE_MANIFEST="${STAGE1_CACHE_MANIFEST:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/cache_480x832_buckets/ar_stage1_i2v_600cats/cache_manifest.json}"
ACTION_SIDECAR_600="${ACTION_SIDECAR_600:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/action_labels_600.csv}"
OPERATOR_ID="${OPERATOR_ID:-l00832862}"

ASSET_DIR="$STAGE2_WORK_ROOT/assets"
LOG_DIR="$STAGE2_WORK_ROOT/logs"
RUN_DIR="$STAGE2_WORK_ROOT/run"
TEACHER_MANIFEST="${TEACHER_MANIFEST:-$ASSET_DIR/real_score_teacher.manifest.json}"
G_MERGED="${G_MERGED:-$ASSET_DIR/stage1_step3750_ema_merged.pt}"
G_MANIFEST="${G_MANIFEST:-${G_MERGED%.pt}.manifest.json}"
STAGE2_CACHE_DIR="${STAGE2_CACHE_DIR:-$STAGE2_WORK_ROOT/f25_600_v1}"
F25_BASE="$STAGE2_CACHE_DIR/cache_manifest.json"
F25_SUCCESS="$STAGE2_CACHE_DIR/_F25_SUCCESS.json"
F25_ATTESTED="${F25_ATTESTED:-$STAGE2_CACHE_DIR/cache_manifest.attested.json}"
NEGATIVE_DIR="${NEGATIVE_DIR:-$STAGE2_WORK_ROOT/negative_v1}"
NEGATIVE_MANIFEST="$NEGATIVE_DIR/negative_conditioning_manifest.json"
NEGATIVE_TENSORS="$NEGATIVE_DIR/negative_conditioning.safetensors"
FINAL_CACHE_MANIFEST="$STAGE2_CACHE_DIR/stage2_i2v_manifest.json"
ROLE_INIT_DIR="${ROLE_INIT_DIR:-$STAGE2_WORK_ROOT/role_init}"
VAE_CKPT="$ARCH_ROOT/Wan2.2_VAE.pth"
T5_CKPT="$ARCH_ROOT/models_t5_umt5-xxl-enc-bf16.pth"
TOKENIZER_DIR="$ARCH_ROOT/google/umt5-xxl"

mkdir -p "$ASSET_DIR" "$LOG_DIR" "$RUN_DIR"

[[ -x "$STAGE2_PYTHON" ]] || { echo "不可执行：$STAGE2_PYTHON" >&2; exit 1; }
[[ -x "$STAGE2_TORCHRUN" ]] || { echo "不可执行：$STAGE2_TORCHRUN" >&2; exit 1; }
for path in \
  "$ARCH_ROOT" "$TEACHER_CKPT" \
  "$TEACHER_PROVENANCE_RECORD" "$STAGE1_BASE" "$STAGE1_CKPT" \
  "$METADATA_600" "$STAGE1_CACHE_MANIFEST" "$ACTION_SIDECAR_600" \
  "$VAE_CKPT" "$T5_CKPT" "$TOKENIZER_DIR"; do
  [[ -e "$path" ]] || { echo "缺少：$path" >&2; exit 1; }
done

[[ "${ATTEST_STAGE2_TEACHER:-0}" == "1" ]] || {
  echo "确认该双向合并权重就是 Stage-2 teacher 后，执行：export ATTEST_STAGE2_TEACHER=1" >&2
  exit 1
}

"$STAGE2_PYTHON" -B - <<'PY'
import torch

assert torch.cuda.device_count() == 8, f"需要 8 张 GPU，实际 {torch.cuda.device_count()}"
names = [torch.cuda.get_device_name(i) for i in range(8)]
assert all("H100" in name for name in names), names
assert all(torch.cuda.get_device_properties(i).total_memory >= 79 * 1024**3 for i in range(8))
assert torch.cuda.is_bf16_supported()
print("8xH100 BF16 environment: OK")
PY

# 1. real-score teacher manifest
provenance_fields="$("$STAGE2_PYTHON" -I -B - \
  "$TEACHER_PROVENANCE_RECORD" "$ARCH_ROOT" "$TEACHER_CKPT" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from safetensors import safe_open

record_path = Path(sys.argv[1]).expanduser().resolve()
architecture_root = Path(sys.argv[2]).expanduser().resolve()
teacher_root = Path(sys.argv[3]).expanduser().resolve()
value = json.loads(record_path.read_text(encoding="utf-8"))
expected = {
    "format": "DiffSynth-Studio Wan2.2-TI2V-5B merged LoRA",
    "merge_dtype": "torch.bfloat16",
    "merged_dit_files": ["diffusion_pytorch_model.safetensors"],
    "save_checksum_verified": True,
    "reload_verified": True,
    "deterministic": "strict",
}
assert all(value.get(key) == expected_value for key, expected_value in expected.items())
assert Path(value["baseline_model_root"]).expanduser().resolve() == architecture_root
teacher_file = teacher_root / "diffusion_pytorch_model.safetensors"
tensor_bytes = 0
with safe_open(str(teacher_file), framework="pt", device="cpu") as handle:
    for key in handle.keys():
        tensor_slice = handle.get_slice(key)
        assert tensor_slice.get_dtype() == "BF16", (key, tensor_slice.get_dtype())
        tensor_bytes += math.prod(tensor_slice.get_shape()) * 2
assert tensor_bytes == value["merged_total_size_bytes"]
assert re.fullmatch(r"[0-9a-f]{64}", value["merged_state_sha256"])
lora_path = Path(value["lora_path"]).expanduser().resolve()
lora_digest = hashlib.sha256()
with lora_path.open("rb") as handle:
    for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
        lora_digest.update(chunk)
lora_sha = lora_digest.hexdigest()
assert lora_sha == value["lora_sha256"]
source_sha = hashlib.sha256(record_path.read_bytes()).hexdigest()
print("\t".join(("diffsynth_sft_lora_merge", f"{lora_path.parent.name}:{lora_path.stem}", source_sha)), end="")
PY
)"
IFS=$'\t' read -r TEACHER_SOURCE_KIND TEACHER_SOURCE_ID TEACHER_SOURCE_SHA256 <<< "$provenance_fields"

if [[ ! -e "$TEACHER_MANIFEST" ]]; then
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
fi

"$STAGE2_PYTHON" -B - \
  "$TEACHER_MANIFEST" "$TEACHER_CKPT" "$ARCH_ROOT" "$TEACHER_SOURCE_SHA256" <<'PY'
import sys
from pathlib import Path
from utils.stage2_role_manifest import validate_stage2_teacher_manifest

result = validate_stage2_teacher_manifest(
    Path(sys.argv[1]),
    expected_checkpoint_path=Path(sys.argv[2]),
    expected_architecture_root=Path(sys.argv[3]),
)
assert result["provenance"]["source_sha256"] == sys.argv[4]
assert result["provenance"]["conversion_command"] == ["none"]
PY
echo "CHECK_1_TEACHER_PASS"

# 2. Stage-1 step3750 EMA Generator
if [[ ! -e "$G_MERGED" && ! -e "$G_MANIFEST" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$STAGE2_PYTHON" -I -B scripts/merge_lora_generator.py \
    --base-checkpoint "$STAGE1_BASE" \
    --training-checkpoint "$STAGE1_CKPT" \
    --output-path "$G_MERGED" \
    --device cuda:0 \
    2>&1 | tee "$LOG_DIR/merge_generator.log"
elif [[ ! -f "$G_MERGED" || ! -f "$G_MANIFEST" ]]; then
  echo "Generator 合并结果不完整；请换一个 STAGE2_WORK_ROOT。" >&2
  exit 1
fi

"$STAGE2_PYTHON" -B - "$G_MANIFEST" "$G_MERGED" <<'PY'
import sys
from pathlib import Path
from utils.stage2_role_manifest import validate_stage2_generator_manifest
validate_stage2_generator_manifest(
    Path(sys.argv[1]), expected_checkpoint_path=Path(sys.argv[2]), expected_step=3750
)
PY
echo "CHECK_2_GENERATOR_PASS"

# 3. 配置路径绑定
export LONG_LIVE_STAGE2_ARCHITECTURE_ROOT="$ARCH_ROOT"
export LONG_LIVE_STAGE2_GENERATOR_BASE="$G_MERGED"
export LONG_LIVE_STAGE2_GENERATOR_MANIFEST="$G_MANIFEST"
export LONG_LIVE_STAGE2_REAL_SCORE_BASE="$TEACHER_CKPT"
export LONG_LIVE_STAGE2_REAL_SCORE_MANIFEST="$TEACHER_MANIFEST"
export LONG_LIVE_STAGE2_METADATA_PATH="$METADATA_600"
export LONG_LIVE_STAGE2_SOURCE_MANIFEST="$F25_ATTESTED"
export LONG_LIVE_STAGE2_ACTION_LABELS_PATH="$ACTION_SIDECAR_600"
export LONG_LIVE_STAGE2_CACHE_DIR="$STAGE2_CACHE_DIR"
export LONG_LIVE_STAGE2_NEGATIVE_MANIFEST="$NEGATIVE_MANIFEST"

config_fields="$("$STAGE2_PYTHON" -B - "$STAGE2_CONFIG" "$RUN_DIR/resolved_config.json" <<'PY'
import json
import sys
from pathlib import Path
from omegaconf import OmegaConf
from utils.stage2_config import resolve_stage2_config

resolved = resolve_stage2_config(OmegaConf.load(sys.argv[1]))
payload = resolved.to_dict()
payload["contract_sha256"] = resolved.contract_hash()
payload["launch_sha256"] = resolved.launch_hash()
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("\t".join((resolved.contract_hash(), resolved.launch_hash())), end="")
PY
)"
IFS=$'\t' read -r CONTRACT_HASH LAUNCH_HASH <<< "$config_fields"

"$STAGE2_PYTHON" -B - "$STAGE2_CONFIG" "$CONTRACT_HASH" "$LAUNCH_HASH" <<'PY'
import os
import sys
from omegaconf import OmegaConf
from utils.stage2_config import resolve_stage2_config

r = resolve_stage2_config(OmegaConf.load(sys.argv[1]))
assert r.contract_hash() == sys.argv[2]
assert r.launch_hash() == sys.argv[3]
assert r.generator_stage1_step == 3750
assert r.init_generator_checkpoint == os.environ["LONG_LIVE_STAGE2_GENERATOR_BASE"]
assert r.init_generator_manifest == os.environ["LONG_LIVE_STAGE2_GENERATOR_MANIFEST"]
assert r.init_real_score_checkpoint == os.environ["LONG_LIVE_STAGE2_REAL_SCORE_BASE"]
assert r.init_real_score_manifest == os.environ["LONG_LIVE_STAGE2_REAL_SCORE_MANIFEST"]
assert r.source_cache_manifest == os.environ["LONG_LIVE_STAGE2_SOURCE_MANIFEST"]
assert r.negative_conditioning_manifest == os.environ["LONG_LIVE_STAGE2_NEGATIVE_MANIFEST"]
PY
echo "CHECK_3_CONFIG_PASS"

# 4. F25 + 文本语义证明 + negative + 600 条正式审计
if [[ ! -f "$F25_BASE" || ! -f "$F25_SUCCESS" ]]; then
  "$STAGE2_TORCHRUN" \
    --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
    --no-python "$STAGE2_PYTHON" -I -B \
    scripts/prepare_stage2_i2v_f25_cache.py \
    --config-path "$STAGE2_CONFIG" \
    --source-cache-manifest "$STAGE1_CACHE_MANIFEST" \
    --vae-checkpoint "$VAE_CKPT" \
    2>&1 | tee "$LOG_DIR/f25.log"
fi

"$STAGE2_PYTHON" -B - "$F25_BASE" <<'PY'
import sys
from pathlib import Path
from utils.stage2_i2v_data import load_source_cache_manifest
load_source_cache_manifest(Path(sys.argv[1]), expected_num_samples=600)
PY

F25_BASE_SELF_SHA="$("$STAGE2_PYTHON" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["manifest_sha256"])' \
  "$F25_BASE")"
TEXT_ATTESTATION='I attest that this legacy positive cache was encoded with the locked Wan seq512/whitespace/add-special-tokens/right-padding/exact-zero-padding contract.'
if [[ ! -e "$F25_ATTESTED" ]]; then
  "$STAGE2_PYTHON" -I -B scripts/audit_stage2_i2v_cache.py \
    upgrade-source-manifest \
    --base-source-cache-manifest "$F25_BASE" \
    --output-manifest "$F25_ATTESTED" \
    --expected-source-manifest-sha256 "$F25_BASE_SELF_SHA" \
    --t5-checkpoint "$T5_CKPT" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --operator-id "$OPERATOR_ID" \
    --operator-attestation "$TEXT_ATTESTATION" \
    --expected-num-samples 600 \
    2>&1 | tee "$LOG_DIR/source_attestation.log"
fi

"$STAGE2_PYTHON" -B - "$F25_ATTESTED" "$(git rev-parse HEAD)" <<'PY'
import sys
from pathlib import Path
from utils.stage2_i2v_data import load_source_cache_manifest
load_source_cache_manifest(
    Path(sys.argv[1]),
    expected_num_samples=600,
    require_text_encoding_upgrade=True,
    expected_upgrade_code_version=f"git:{sys.argv[2]}",
)
PY

if [[ ! -e "$NEGATIVE_DIR" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$STAGE2_PYTHON" -I -B scripts/audit_stage2_i2v_cache.py \
    prepare-negative \
    --source-cache-manifest "$F25_ATTESTED" \
    --t5-checkpoint "$T5_CKPT" \
    --tokenizer-dir "$TOKENIZER_DIR" \
    --output-dir "$NEGATIVE_DIR" \
    --expected-num-samples 600 \
    --device cuda:0 \
    2>&1 | tee "$LOG_DIR/negative.log"
elif [[ ! -f "$NEGATIVE_MANIFEST" || ! -f "$NEGATIVE_TENSORS" ]]; then
  echo "negative 结果不完整；请换一个 STAGE2_WORK_ROOT。" >&2
  exit 1
fi

"$STAGE2_PYTHON" -B - "$F25_ATTESTED" "$NEGATIVE_MANIFEST" <<'PY'
import sys
from pathlib import Path
from utils.stage2_i2v_data import load_negative_conditioning, load_source_cache_manifest
source = load_source_cache_manifest(
    Path(sys.argv[1]), expected_num_samples=600, require_text_encoding_upgrade=True
)
load_negative_conditioning(Path(sys.argv[2]), source_cache_manifest=source, load_tensors=False)
PY

mapfile -t ACTION_IDS < <("$STAGE2_PYTHON" -B - "$ACTION_SIDECAR_600" <<'PY'
import csv
import sys
from collections import Counter

with open(sys.argv[1], "r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    assert reader.fieldnames == ["video", "action_id"]
    rows = list(reader)
assert len(rows) == 600, len(rows)
counts = Counter(row["action_id"].strip() for row in rows)
assert len(counts) == 3 and sorted(counts.values()) == [200, 200, 200], counts
for action_id in sorted(counts):
    print(action_id)
PY
)
[[ "${#ACTION_IDS[@]}" == "3" ]]

if [[ ! -e "$FINAL_CACHE_MANIFEST" ]]; then
  "$STAGE2_PYTHON" -I -B scripts/audit_stage2_i2v_cache.py audit \
    --config-path "$STAGE2_CONFIG" \
    --source-cache-manifest "$F25_ATTESTED" \
    --action-id "${ACTION_IDS[0]}" \
    --action-id "${ACTION_IDS[1]}" \
    --action-id "${ACTION_IDS[2]}" \
    2>&1 | tee "$LOG_DIR/cache_audit.log"
fi

"$STAGE2_PYTHON" -B - \
  "$FINAL_CACHE_MANIFEST" "$METADATA_600" "$F25_ATTESTED" "$NEGATIVE_MANIFEST" \
  "$CONTRACT_HASH" "$LAUNCH_HASH" <<'PY'
import sys
from pathlib import Path
from utils.stage2_i2v_data import load_stage2_i2v_manifest, validate_stage2_i2v_runtime_bindings

manifest = load_stage2_i2v_manifest(Path(sys.argv[1]), expected_num_samples=600)
validate_stage2_i2v_runtime_bindings(
    manifest,
    metadata_path=Path(sys.argv[2]),
    source_cache_manifest_path=Path(sys.argv[3]),
    negative_conditioning_manifest_path=Path(sys.argv[4]),
    config_contract_sha256=sys.argv[5],
    config_launch_sha256=sys.argv[6],
    expected_num_samples=600,
)
assert sorted(manifest["actions"]["counts"].values()) == [200, 200, 200]
PY
echo "CHECK_4_DATA_PASS"

# 5. 8xH100 角色初始化；只初始化，不训练
if [[ ! -e "$ROLE_INIT_DIR" ]]; then
  "$STAGE2_TORCHRUN" \
    --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
    --no-python "$STAGE2_PYTHON" -I -B \
    scripts/preflight_stage2_roles.py \
    --config "$STAGE2_CONFIG" \
    --output-dir "$ROLE_INIT_DIR" \
    --expected-git-commit "$(git rev-parse HEAD)" \
    2>&1 | tee "$LOG_DIR/role_init.log"
elif [[ ! -f "$ROLE_INIT_DIR/role_init_manifest.json" || ! -f "$ROLE_INIT_DIR/ROLE_INIT_COMPLETE" ]]; then
  echo "角色初始化结果不完整；请换一个 STAGE2_WORK_ROOT。" >&2
  exit 1
fi

"$STAGE2_PYTHON" -B - \
  "$ROLE_INIT_DIR/role_init_manifest.json" "$CONTRACT_HASH" "$LAUNCH_HASH" "$(git rev-parse HEAD)" <<'PY'
import json
import sys
from pathlib import Path
from utils.stage1_io import canonical_json_sha256

path = Path(sys.argv[1])
record = json.loads(path.read_text(encoding="utf-8"))
body = dict(record)
recorded_hash = body.pop("manifest_sha256")
assert canonical_json_sha256(body) == recorded_hash
assert record["schema"] == "longlive_stage2_role_init_manifest"
assert record["artifact_kind"] == "init_only_audit_not_training_checkpoint"
assert record["initialization_mode"] == "init_from_stage1"
assert record["config"]["contract_hash"] == sys.argv[2]
assert record["config"]["launch_hash"] == sys.argv[3]
assert record["code"]["git_commit"] == sys.argv[4]
assert set(record["roles"]) == {"generator", "real_score", "fake_score"}
assert record["side_effects"] == {
    "tripwires_enforced": [
        "module_call_impl",
        "direct_module_forward",
        "optimizer",
        "ema",
        "text_encoder",
        "vae",
        "dataloader",
    ],
    "forward_calls": 0,
    "optimizer_created": False,
    "ema_created": False,
    "text_encoder_created": False,
    "vae_created": False,
    "dataloader_created": False,
}
assert record["fsdp"]["world_size"] == 8
assert record["fsdp"]["sequence_parallel_size"] == 1
assert record["fsdp"]["data_parallel_size"] == 8
assert record["fsdp"]["mesh_shape"] == [8]
assert record["fsdp"]["mesh_dim_names"] == ["shard"]
assert record["fsdp"]["sharding_strategy"] == "FULL_SHARD"
assert record["fsdp"]["all_roles_independently_wrapped"] is True
assert (path.parent / "ROLE_INIT_COMPLETE").is_file()
assert (path.parent / "ROLE_INIT_COMPLETE").stat().st_size == 0
assert not (path.parent / "_SUCCESS").exists()
PY
echo "CHECK_5_ROLE_INIT_PASS"
echo "STAGE2_PRETRAIN_PASS"
