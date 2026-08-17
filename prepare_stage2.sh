#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "STAGE2_PRETRAIN_CHECK_FAILED (line ${LINENO})" >&2' ERR

SOURCE_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SOURCE_REPO
STAGE2_WORK_ROOT="${STAGE2_WORK_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_stage2_new}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$STAGE2_WORK_ROOT/pycache"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

STAGE2_PYTHON="${STAGE2_PYTHON:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python}"
STAGE2_TORCHRUN="${STAGE2_TORCHRUN:-$(dirname "$STAGE2_PYTHON")/torchrun}"
STAGE2_CONFIG="${STAGE2_CONFIG:-$SOURCE_REPO/configs/train_i2v_stage2_600cats.yaml}"

ARCH_ROOT="${ARCH_ROOT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B}"
TEACHER_CKPT="${TEACHER_CKPT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/ckpts}"
TEACHER_PROVENANCE_RECORD="${TEACHER_PROVENANCE_RECORD:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/ckpts/merge_manifest.json}"
STAGE1_BASE="${STAGE1_BASE:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/checkpoints/stage1/converted_causal_base.pt}"
STAGE1_CKPT="${STAGE1_CKPT:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/results/stage1_600cats_phaseA10epochs_phaseB20epochs/checkpoint_model_003075}"
METADATA_600="${METADATA_600:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/metadata_600clips_480x832_buckets.csv}"
STAGE1_CACHE_MANIFEST="${STAGE1_CACHE_MANIFEST:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/cache_480x832_buckets/ar_stage1_i2v_600cats/cache_manifest.json}"
ACTION_SIDECAR_600="${ACTION_SIDECAR_600:-/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/action_labels_600cats.csv}"
OPERATOR_ID="${OPERATOR_ID:-l00832862}"

ASSET_DIR="$STAGE2_WORK_ROOT/assets"
LOG_DIR="$STAGE2_WORK_ROOT/logs"
RUN_DIR="$STAGE2_WORK_ROOT/run"
TEACHER_MANIFEST="${TEACHER_MANIFEST:-$ASSET_DIR/real_score_teacher.manifest.json}"
G_MERGED="${G_MERGED:-$ASSET_DIR/stage1_step3075_ema_merged.pt}"
G_MANIFEST="${G_MANIFEST:-${G_MERGED%.pt}.manifest.json}"
STAGE2_CACHE_DIR="${STAGE2_CACHE_DIR:-$STAGE2_WORK_ROOT/stage2_600cats_f25_v1}"
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

[[ -x "$STAGE2_PYTHON" ]] || { echo "不可执行：$STAGE2_PYTHON" >&2; exit 1; }
[[ -x "$STAGE2_TORCHRUN" ]] || { echo "不可执行：$STAGE2_TORCHRUN" >&2; exit 1; }
STAGE2_WORK_ROOT_REAL="$("$STAGE2_PYTHON" -I -B -c \
  'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
  "$STAGE2_WORK_ROOT")"
mkdir -p "$ASSET_DIR" "$LOG_DIR" "$RUN_DIR"
cd -- "$SOURCE_REPO"

[[ "${ATTEST_STAGE2_TEACHER:-0}" == "1" ]] || {
  echo "确认该双向合并权重就是 Stage-2 teacher 后，执行：export ATTEST_STAGE2_TEACHER=1" >&2
  exit 1
}

for path in \
  "$ARCH_ROOT" "$TEACHER_CKPT" \
  "$TEACHER_PROVENANCE_RECORD" "$STAGE1_BASE" "$STAGE1_CKPT" \
  "$METADATA_600" "$STAGE1_CACHE_MANIFEST" "$ACTION_SIDECAR_600" \
  "$VAE_CKPT" "$T5_CKPT" "$TOKENIZER_DIR"; do
  [[ -e "$path" ]] || { echo "缺少：$path" >&2; exit 1; }
done

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


# 2. Stage-1 step3075 EMA Generator
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
    Path(sys.argv[1]), expected_checkpoint_path=Path(sys.argv[2]), expected_step=3075
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

"$STAGE2_PYTHON" -B - "$STAGE2_CONFIG" "$RUN_DIR/resolved_config.json" <<'PY'
import json
import os
import sys
from pathlib import Path
from omegaconf import OmegaConf
from utils.stage2_config import resolve_stage2_config

resolved = resolve_stage2_config(OmegaConf.load(sys.argv[1]))

# ---- 先做全部校验，全部通过后才落盘 ----
assert resolved.generator_stage1_step == 3075, \
    f"generator_stage1_step mismatch: expected 3075, got {resolved.generator_stage1_step}"

assert resolved.init_generator_checkpoint == os.environ["LONG_LIVE_STAGE2_GENERATOR_BASE"], \
    f"init_generator_checkpoint mismatch: resolved={resolved.init_generator_checkpoint!r}, env={os.environ['LONG_LIVE_STAGE2_GENERATOR_BASE']!r}"

assert resolved.init_generator_manifest == os.environ["LONG_LIVE_STAGE2_GENERATOR_MANIFEST"], \
    f"init_generator_manifest mismatch: resolved={resolved.init_generator_manifest!r}, env={os.environ['LONG_LIVE_STAGE2_GENERATOR_MANIFEST']!r}"

assert resolved.init_real_score_checkpoint == os.environ["LONG_LIVE_STAGE2_REAL_SCORE_BASE"], \
    f"init_real_score_checkpoint mismatch: resolved={resolved.init_real_score_checkpoint!r}, env={os.environ['LONG_LIVE_STAGE2_REAL_SCORE_BASE']!r}"

assert resolved.init_real_score_manifest == os.environ["LONG_LIVE_STAGE2_REAL_SCORE_MANIFEST"], \
    f"init_real_score_manifest mismatch: resolved={resolved.init_real_score_manifest!r}, env={os.environ['LONG_LIVE_STAGE2_REAL_SCORE_MANIFEST']!r}"

assert resolved.source_cache_manifest == os.environ["LONG_LIVE_STAGE2_SOURCE_MANIFEST"], \
    f"source_cache_manifest mismatch: resolved={resolved.source_cache_manifest!r}, env={os.environ['LONG_LIVE_STAGE2_SOURCE_MANIFEST']!r}"

assert resolved.negative_conditioning_manifest == os.environ["LONG_LIVE_STAGE2_NEGATIVE_MANIFEST"], \
    f"negative_conditioning_manifest mismatch: resolved={resolved.negative_conditioning_manifest!r}, env={os.environ['LONG_LIVE_STAGE2_NEGATIVE_MANIFEST']!r}"

# ---- 校验全部通过，写入 resolved_config.json ----
payload = resolved.to_dict()
payload["contract_sha256"] = resolved.contract_hash()
payload["launch_sha256"] = resolved.launch_hash()
Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
echo "CHECK_3_CONFIG_PASS"

# 3.5 清理与本次 launch_hash 不一致的过期产物，让下游守卫自然触发重跑
#     只处理 launch 级绑定的两类产物：
#       a) $FINAL_CACHE_MANIFEST（仅该文件；同目录 F25 缓存不含 launch 绑定，保留复用）
#       b) $ROLE_INIT_DIR（整目录；manifest 缺失/损坏/hash 不一致/COMPLETE 缺失均视为过期）
CURRENT_LAUNCH_HASH="$("$STAGE2_PYTHON" -B -c \
  'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["launch_sha256"])' \
  "$RUN_DIR/resolved_config.json")"
[[ "$CURRENT_LAUNCH_HASH" =~ ^[0-9a-f]{64}$ ]] || {
  echo "launch_sha256 非法：$CURRENT_LAUNCH_HASH" >&2
  exit 1
}
echo "[prune] mode=archive current launch_hash=$CURRENT_LAUNCH_HASH"

prune_stale() {
  local target="$1" reason="$2"
  local target_real
  target_real="$("$STAGE2_PYTHON" -I -B -c \
    'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' \
    "$target")"
  if [[ "$target_real" == "$STAGE2_WORK_ROOT_REAL" ]]; then
    echo "拒绝归档整个 STAGE2_WORK_ROOT：$target_real" >&2
    exit 1
  fi
  case "$target_real/" in
    "$STAGE2_WORK_ROOT_REAL/"*) ;;
    *)
      echo "拒绝归档工作根目录外的过期产物：$target_real" >&2
      exit 1
      ;;
  esac
  local backup="${target_real}.stale_$(date +%Y%m%d_%H%M%S)_$$"
  echo "[prune] 归档过期产物 ${target_real} -> ${backup}（${reason}）"
  mv -- "$target_real" "$backup"
}

if [[ -e "$FINAL_CACHE_MANIFEST" ]]; then
  FINAL_CACHE_HASH="$("$STAGE2_PYTHON" -B -c '
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8"))["provenance"]["config_launch_sha256"])
except Exception:
    print("MISSING")
' "$FINAL_CACHE_MANIFEST")"
  if [[ "$FINAL_CACHE_HASH" != "$CURRENT_LAUNCH_HASH" ]]; then
    prune_stale "$FINAL_CACHE_MANIFEST" \
      "manifest launch_hash=$FINAL_CACHE_HASH != current"
  fi
fi

if [[ -e "$ROLE_INIT_DIR" ]]; then
  ROLE_INIT_HASH="$("$STAGE2_PYTHON" -B -c '
import json, sys
try:
    print(json.load(open(sys.argv[1], encoding="utf-8"))["config"]["launch_hash"])
except Exception:
    print("MISSING")
' "$ROLE_INIT_DIR/role_init_manifest.json")"
  if [[ "$ROLE_INIT_HASH" != "$CURRENT_LAUNCH_HASH" ]]; then
    prune_stale "$ROLE_INIT_DIR" "manifest launch_hash=$ROLE_INIT_HASH != current"
  elif [[ ! -f "$ROLE_INIT_DIR/ROLE_INIT_COMPLETE" ]]; then
    prune_stale "$ROLE_INIT_DIR" "缺少 ROLE_INIT_COMPLETE，视为半成品"
  fi
fi
echo "CHECK_3P5_PRUNE_PASS"

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

"$STAGE2_PYTHON" -B - "$F25_ATTESTED" <<'PY'
import sys
from pathlib import Path
from utils.stage2_i2v_data import load_source_cache_manifest
load_source_cache_manifest(
    Path(sys.argv[1]),
    expected_num_samples=600,
    require_text_encoding_upgrade=True,
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
from utils.stage2_action_contract import STAGE2_EXPECTED_ACTION_COUNTS

with open(sys.argv[1], "r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    assert reader.fieldnames == ["video", "action_id"], reader.fieldnames
    rows = list(reader)
assert len(rows) == 600, len(rows)
counts = Counter(row["action_id"].strip() for row in rows)
assert dict(counts) == STAGE2_EXPECTED_ACTION_COUNTS, counts
for action_id in sorted(counts):
    print(action_id)
PY
)
if [[ "${#ACTION_IDS[@]}" != "3" ]]; then
  echo "ACTION_IDS 数量异常：期望 3 个，实际 ${#ACTION_IDS[@]} 个" >&2
  exit 1
fi

if [[ ! -e "$FINAL_CACHE_MANIFEST" ]]; then
  "$STAGE2_PYTHON" -I -B scripts/audit_stage2_i2v_cache.py audit \
    --config-path "$STAGE2_CONFIG" \
    --source-cache-manifest "$F25_ATTESTED" \
    --action-labels-path "$ACTION_SIDECAR_600" \
    --action-id "${ACTION_IDS[0]}" \
    --action-id "${ACTION_IDS[1]}" \
    --action-id "${ACTION_IDS[2]}" \
    2>&1 | tee "$LOG_DIR/cache_audit.log"
fi

"$STAGE2_PYTHON" -B - \
  "$FINAL_CACHE_MANIFEST" "$METADATA_600" "$F25_ATTESTED" "$NEGATIVE_MANIFEST" \
  "$STAGE2_CONFIG" <<'PY'
import sys
from pathlib import Path
from omegaconf import OmegaConf
from utils.stage2_action_contract import STAGE2_EXPECTED_ACTION_COUNTS
from utils.stage2_config import resolve_stage2_config
from utils.stage2_i2v_data import load_stage2_i2v_manifest, validate_stage2_i2v_runtime_bindings

resolved = resolve_stage2_config(OmegaConf.load(sys.argv[5]))
manifest = load_stage2_i2v_manifest(Path(sys.argv[1]), expected_num_samples=600)
validate_stage2_i2v_runtime_bindings(
    manifest,
    metadata_path=Path(sys.argv[2]),
    source_cache_manifest_path=Path(sys.argv[3]),
    negative_conditioning_manifest_path=Path(sys.argv[4]),
    config_contract_sha256=resolved.contract_hash(),
    config_launch_sha256=resolved.launch_hash(),
    expected_num_samples=600,
)
assert manifest["actions"]["counts"] == STAGE2_EXPECTED_ACTION_COUNTS, \
    f"action counts mismatch: manifest={manifest['actions']['counts']}, expected={STAGE2_EXPECTED_ACTION_COUNTS}"
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
    2>&1 | tee "$LOG_DIR/role_init.log"
elif [[ ! -f "$ROLE_INIT_DIR/role_init_manifest.json" || ! -f "$ROLE_INIT_DIR/ROLE_INIT_COMPLETE" ]]; then
  echo "角色初始化结果不完整；请换一个 STAGE2_WORK_ROOT。" >&2
  exit 1
fi

"$STAGE2_PYTHON" -B - \
  "$ROLE_INIT_DIR/role_init_manifest.json" "$STAGE2_CONFIG" <<'PY'
import json
import sys
from pathlib import Path
from omegaconf import OmegaConf
from utils.stage1_io import canonical_json_sha256
from utils.stage2_config import resolve_stage2_config

resolved = resolve_stage2_config(OmegaConf.load(sys.argv[2]))

path = Path(sys.argv[1])
record = json.loads(path.read_text(encoding="utf-8"))
body = dict(record)
recorded_hash = body.pop("manifest_sha256")
assert canonical_json_sha256(body) == recorded_hash
assert record["schema"] == "longlive_stage2_role_init_manifest"
assert record["artifact_kind"] == "init_only_audit_not_training_checkpoint"
assert record["initialization_mode"] == "init_from_stage1"
assert record["config"]["contract_hash"] == resolved.contract_hash()
assert record["config"]["launch_hash"] == resolved.launch_hash()
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

# 6. real FSDP2 gradient-accumulation parity；只跑 tiny 参数门禁，不启动训练
"$STAGE2_TORCHRUN" \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B \
  tests/stage2_fsdp2_accumulation_gate.py --require-h100 \
  2>&1 | tee "$LOG_DIR/fsdp2_accumulation_gate.log"
echo "CHECK_6_FSDP2_ACCUMULATION_PASS"
echo "STAGE2_PRETRAIN_PASS"
