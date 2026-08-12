# Stage-2 8×H100：real-score manifest 后完整部署与训练

这份手册从 `prepare_stage2.sh` 已输出 `REAL_SCORE_TEACHER_MANIFEST_PASS` 开始，一次走完：

1. 校验 real-score teacher manifest；
2. 合并并校验 Stage-1 step3750 EMA Generator；
3. 生成 F25 cache、positive text attestation 和 negative conditioning；
4. 生成正式 600 条 cache audit manifest；
5. 做三角色 init-only；
6. 依次运行 C0、C1、C2 H100 smoke；
7. 使用全新目录启动正式训练、断点恢复并验收 checkpoint、JSONL、九图和 HTML。

任何非零退出、Traceback、OOM、NCCL 卡死、hash/bitwise/config binding 不一致都立即停止。不要改 JSON、删失败现场、复制 latent 或降低验收条件来“继续跑”。

当前代码把 Generator 来源明确锁定为 **Stage-1 step3750 EMA**。`prepare_stage2.sh`只生成双向 teacher 的 manifest，不生成 Generator，也不把 step3075 变成合法的 Stage-2 起点。

## 0. 一次填写全部绝对路径

执行代码必须来自独立 clean clone；模型、cache、日志和训练 checkpoint 必须在该 clone 外。仓库会同时拒绝 tracked dirty、untracked 和 ignored 文件，因而不能把 `.pt`、`.log` 或训练输出写进执行 clone。

```bash
set -euo pipefail

export STAGE2_PYTHON=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python
export STAGE2_TORCHRUN="$(dirname "$STAGE2_PYTHON")/torchrun"
export STAGE2_GIT_SOURCE=https://github.com/zonghui-liu-opt/LongLive_cats.git

# 新建的代码执行 clone；必须不存在。
export STAGE2_REPO=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0-stage2-clean

# 以下目录均在 STAGE2_REPO 外。
export STAGE2_ASSET_DIR=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/checkpoints/stage2
export STAGE2_RUN_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs
export STAGE2_CACHE_DIR=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_cache/f25_600_v1
export NEGATIVE_DIR=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_negative/v1
export ROLE_INIT_DIR="$STAGE2_RUN_ROOT/role_init"
export SMOKE_DIR="$STAGE2_RUN_ROOT/smoke_c0_c1_c2"
export FORMAL_DIR="$STAGE2_RUN_ROOT/formal_600cats"

# 已生成的双向 teacher 与来源记录。
export ARCH_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B
export TEACHER_CKPT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/ckpts
export TEACHER_PROVENANCE_RECORD=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/merge_manifest.json
export TEACHER_MANIFEST="$STAGE2_ASSET_DIR/real_score_teacher.manifest.json"

# Generator 必须来自正式 Stage-1 step3750 checkpoint 的 EMA adapter。
export STAGE1_BASE=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/checkpoints/stage1/converted_causal_base.pt
export STAGE1_CKPT=/绝对路径/checkpoint_model_003750
export G_MERGED="$STAGE2_ASSET_DIR/stage1_step3750_ema_merged.pt"
export G_MANIFEST="$STAGE2_ASSET_DIR/stage1_step3750_ema_merged.manifest.json"

# 600 条数据及同源编码资产。
export STAGE1_CACHE_MANIFEST=/绝对路径/Stage1原始cache/cache_manifest.json
export METADATA_600=/绝对路径/metadata_600.csv
export ACTION_SIDECAR_600=/绝对路径/action_labels_600.csv
export VAE_CKPT="$ARCH_ROOT/Wan2.2_VAE.pth"
export T5_CKPT="$ARCH_ROOT/models_t5_umt5-xxl-enc-bf16.pth"
export TOKENIZER_DIR="$ARCH_ROOT/google/umt5-xxl"
export ACTION_ID_1=替换为真实动作枚举1
export ACTION_ID_2=替换为真实动作枚举2
export ACTION_ID_3=替换为真实动作枚举3
export OPERATOR_ID=替换为真实执行人或任务ID

test -x "$STAGE2_PYTHON"
test -x "$STAGE2_TORCHRUN"
test "$STAGE2_REPO" != "$STAGE2_ASSET_DIR"
test "$STAGE2_REPO" != "$STAGE2_RUN_ROOT"
test "$STAGE2_REPO" != "$STAGE2_CACHE_DIR"
```

`ACTION_SIDECAR_600` 的精确表头为 `video,action_id`，共600行，三个action各200行，video与metadata一一对应。禁止从prompt、文件名或行号推断action。

## 1. 校验刚生成的 real-score teacher manifest

这里有三个不同的 SHA：

- `manifest_sha256`：manifest内容的自哈希；
- `sha256sum real_score_teacher.manifest.json`：JSON文件字节哈希；
- `checkpoint.source_files_sha256`：teacher权重文件列表的聚合哈希。

`provenance.source_sha256`必须等于完整 `merge_manifest.json` 的文件SHA，而不是其中的 `merged_state_sha256`、`lora_sha256` 或命令行最后打印的manifest自哈希。

```bash
test -s "$TEACHER_PROVENANCE_RECORD"
test -s "$TEACHER_MANIFEST"

"$STAGE2_PYTHON" -I -B - \
  "$TEACHER_MANIFEST" "$TEACHER_PROVENANCE_RECORD" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest_path = Path(sys.argv[1]).resolve()
provenance_path = Path(sys.argv[2]).resolve()
value = json.loads(manifest_path.read_text(encoding="utf-8"))
body = dict(value)
recorded_self_hash = body.pop("manifest_sha256")
canonical = json.dumps(
    body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
).encode("utf-8")
assert hashlib.sha256(canonical).hexdigest() == recorded_self_hash
assert value["schema"] == "longlive_stage2_teacher_manifest"
assert value["role"] == "real_score"
assert value["checkpoint"]["format"] == "wan_native_transformer"
assert value["checkpoint"]["dtype"] == "bfloat16"
assert value["provenance"]["conversion_command"] == ["none"]
assert value["provenance"]["source_sha256"] == sha(provenance_path)
assert value["operator_attestation"] == {
    "cat_domain_bidirectional_ti2v": True,
    "video_global_flow": True,
}
print(
    "REAL_SCORE_MANIFEST_VERIFY_PASS",
    "self_sha=" + recorded_self_hash,
    "file_sha=" + sha(manifest_path),
    "merge_record_file_sha=" + sha(provenance_path),
    "teacher_files_sha=" + value["checkpoint"]["source_files_sha256"],
)
PY
```

旧文件若失败，不要原地改JSON。给旧文件改名留档，设置一个不存在的新 `TEACHER_MANIFEST`，使用当前 `prepare_stage2.sh`重新生成。

## 2. 获取 clean stage-2 clone 并验证环境

```bash
test ! -e "$STAGE2_REPO"
git clone --branch stage-2 --single-branch "$STAGE2_GIT_SOURCE" "$STAGE2_REPO"
cd "$STAGE2_REPO"
git fetch origin stage-2
test "$(git rev-parse HEAD)" = "$(git rev-parse FETCH_HEAD)"
export STAGE2_COMMIT="$(git rev-parse HEAD)"

mkdir -p "$STAGE2_ASSET_DIR" "$STAGE2_RUN_ROOT" \
  "$(dirname "$STAGE2_CACHE_DIR")" "$(dirname "$NEGATIVE_DIR")"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$STAGE2_RUN_ROOT/pycache"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(git ls-files --others --ignored --exclude-standard)"

nvidia-smi -L | tee "$STAGE2_RUN_ROOT/gpus.txt"
"$STAGE2_PYTHON" -I -B - <<'PY'
import torch

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(8)]
assert all("H100" in name for name in names), names
assert all(torch.cuda.get_device_properties(i).total_memory >= 79 * 1024**3 for i in range(8))
assert torch.cuda.is_bf16_supported()
print("GPU_ENV_PASS", torch.__version__, torch.version.cuda, names)
PY

PYTHONPATH="$PWD" "$STAGE2_PYTHON" -B -m pytest -q -p no:cacheprovider \
  tests/test_stage2_*.py \
  2>&1 | tee "$STAGE2_RUN_ROOT/stage2_tests.log"

test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(git ls-files --others --ignored --exclude-standard)"
```

不要把通过标准写死为历史测试数量；当前分支可能增加测试。唯一标准是本次命令退出码0且 `0 failed`。

## 3. 合并并校验 Stage-1 step3750 EMA Generator

`prepare_stage2.sh`没有执行这一步。先验证输入checkpoint自报step3750，再合并 `adapter_ema.safetensors`，禁止使用raw adapter。

```bash
test -s "$STAGE1_BASE"
test -d "$STAGE1_CKPT"
test ! -e "$G_MERGED"
test ! -e "$G_MANIFEST"

"$STAGE2_PYTHON" -I -B - "$STAGE1_CKPT/checkpoint_manifest.json" <<'PY'
import json
import sys

m = json.load(open(sys.argv[1], encoding="utf-8"))
assert m["schema"] == "longlive_stage1_lora_checkpoint"
assert m["completed_step"] == 3750
assert m["next_update_index"] == 3750
assert m["resumable"] is False
print("STAGE1_STEP3750_INPUT_PASS", m["manifest_sha256"])
PY

CUDA_VISIBLE_DEVICES=0 "$STAGE2_PYTHON" -I -B scripts/merge_lora_generator.py \
  --base-checkpoint "$STAGE1_BASE" \
  --training-checkpoint "$STAGE1_CKPT" \
  --output-path "$G_MERGED" \
  --device cuda:0 \
  2>&1 | tee "$STAGE2_RUN_ROOT/generator_merge.log"

test -s "$G_MERGED"
test -s "$G_MANIFEST"

PYTHONPATH="$PWD" "$STAGE2_PYTHON" -B - \
  "$G_MANIFEST" "$G_MERGED" <<'PY'
import json
import sys
from utils.stage2_role_manifest import validate_stage2_generator_manifest

result = validate_stage2_generator_manifest(
    sys.argv[1], expected_checkpoint_path=sys.argv[2], expected_step=3750
)
print("GENERATOR_STEP3750_MANIFEST_PASS", json.dumps(result, sort_keys=True))
PY
```

## 4. 先绑定最终路径，再解析一次正式配置

以下环境变量现在由 `configs/train_i2v_stage2_600cats.yaml`真实读取。即使F25 attestation和negative文件尚未产生，也先写入它们的最终路径；后面不再换路径。

```bash
export STAGE2_CONFIG="$PWD/configs/train_i2v_stage2_600cats.yaml"
export F25_BASE="$STAGE2_CACHE_DIR/cache_manifest.json"
export F25_SUCCESS="$STAGE2_CACHE_DIR/_F25_SUCCESS.json"
export F25_ATTESTED="$STAGE2_CACHE_DIR/cache_manifest.attested.json"
export NEGATIVE_MANIFEST="$NEGATIVE_DIR/negative_conditioning_manifest.json"
export FINAL_CACHE_MANIFEST="$STAGE2_CACHE_DIR/stage2_i2v_manifest.json"

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

export CONTRACT_HASH="$(
  PYTHONPATH="$PWD" "$STAGE2_PYTHON" -B -m utils.stage2_config \
    --config "$STAGE2_CONFIG" --contract-hash-only
)"
test "$CONTRACT_HASH" = aa4d7be1e05c846df14cee5417a298afe668429f41faa671f3021754a5616c00

export LAUNCH_HASH="$(
  PYTHONPATH="$PWD" "$STAGE2_PYTHON" -B -m utils.stage2_config \
    --config "$STAGE2_CONFIG" --hash-only
)"
PYTHONPATH="$PWD" "$STAGE2_PYTHON" -B -m utils.stage2_config \
  --config "$STAGE2_CONFIG" > "$STAGE2_RUN_ROOT/resolved_config.json"

PYTHONPATH="$PWD" "$STAGE2_PYTHON" -B - \
  "$STAGE2_CONFIG" "$LAUNCH_HASH" <<'PY'
import os
import sys
from omegaconf import OmegaConf
from utils.stage2_config import resolve_stage2_config

r = resolve_stage2_config(OmegaConf.load(sys.argv[1]))
assert r.launch_hash() == sys.argv[2]
assert r.generator_stage1_step == 3750
assert r.init_generator_checkpoint == os.environ["LONG_LIVE_STAGE2_GENERATOR_BASE"]
assert r.init_generator_manifest == os.environ["LONG_LIVE_STAGE2_GENERATOR_MANIFEST"]
assert r.init_real_score_checkpoint == os.environ["LONG_LIVE_STAGE2_REAL_SCORE_BASE"]
assert r.init_real_score_manifest == os.environ["LONG_LIVE_STAGE2_REAL_SCORE_MANIFEST"]
assert r.source_cache_manifest == os.environ["LONG_LIVE_STAGE2_SOURCE_MANIFEST"]
assert r.negative_conditioning_manifest == os.environ["LONG_LIVE_STAGE2_NEGATIVE_MANIFEST"]
print("CONFIG_BINDING_PASS", r.contract_hash(), r.launch_hash())
PY
```

后续每个阶段都继承同一shell环境。重新登录后必须从本节完整重导；不能只补其中几个变量。

## 5. 生成 F25、文本证明、negative 和正式600条audit

Stage-2需要 `sink1 + future24 = F25`。合格F25原字节复用；F24必须从同一原视频0–96帧与同一VAE重提。禁止padding、复制latent、截断或改成23个future。

```bash
test ! -e "$STAGE2_CACHE_DIR"
test ! -e "$NEGATIVE_DIR"

"$STAGE2_TORCHRUN" \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B \
  scripts/prepare_stage2_i2v_f25_cache.py \
  --config-path "$STAGE2_CONFIG" \
  --source-cache-manifest "$STAGE1_CACHE_MANIFEST" \
  --vae-checkpoint "$VAE_CKPT" \
  2>&1 | tee "$STAGE2_RUN_ROOT/f25_prepare.log"

test -s "$F25_BASE"
test -s "$F25_SUCCESS"

export F25_BASE_SELF_SHA="$(
  "$STAGE2_PYTHON" -I -B -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["manifest_sha256"])' \
    "$F25_BASE"
)"
export TEXT_ATTESTATION='I attest that this legacy positive cache was encoded with the locked Wan seq512/whitespace/add-special-tokens/right-padding/exact-zero-padding contract.'

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
  2>&1 | tee "$STAGE2_RUN_ROOT/source_attestation.log"

CUDA_VISIBLE_DEVICES=0 "$STAGE2_PYTHON" -I -B \
  scripts/audit_stage2_i2v_cache.py prepare-negative \
  --source-cache-manifest "$F25_ATTESTED" \
  --t5-checkpoint "$T5_CKPT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --output-dir "$NEGATIVE_DIR" \
  --expected-num-samples 600 \
  --device cuda:0 \
  2>&1 | tee "$STAGE2_RUN_ROOT/negative_prepare.log"

"$STAGE2_PYTHON" -I -B scripts/audit_stage2_i2v_cache.py audit \
  --config-path "$STAGE2_CONFIG" \
  --source-cache-manifest "$F25_ATTESTED" \
  --action-id "$ACTION_ID_1" \
  --action-id "$ACTION_ID_2" \
  --action-id "$ACTION_ID_3" \
  2>&1 | tee "$STAGE2_RUN_ROOT/cache_audit.log"

test -s "$NEGATIVE_DIR/negative_conditioning.safetensors"
test -s "$NEGATIVE_MANIFEST"
test -s "$FINAL_CACHE_MANIFEST"

"$STAGE2_PYTHON" -I -B - \
  "$FINAL_CACHE_MANIFEST" "$CONTRACT_HASH" "$LAUNCH_HASH" <<'PY'
import json
import sys

m = json.load(open(sys.argv[1], encoding="utf-8"))
assert m["schema"] == "longlive_stage2_i2v_cache"
assert m["num_samples"] == 600
assert sorted(m["actions"]["counts"].values()) == [200, 200, 200]
assert sum(m["orientation_counts"].values()) == 600
assert m["provenance"]["config_contract_sha256"] == sys.argv[2]
assert m["provenance"]["config_launch_sha256"] == sys.argv[3]
print("FORMAL_CACHE_AUDIT_PASS", m["actions"]["counts"], m["manifest_sha256"])
PY
```

## 6. 用最终配置做8卡三角色 init-only

这一步完整加载 G、frozen real-score 和独立 fake-score底座，建立 r32/r64 LoRA并做1D FULL_SHARD审计，但不执行forward、optimizer或训练。

```bash
test ! -e "$ROLE_INIT_DIR"

"$STAGE2_TORCHRUN" \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B \
  scripts/preflight_stage2_roles.py \
  --config "$STAGE2_CONFIG" \
  --output-dir "$ROLE_INIT_DIR" \
  --expected-git-commit "$STAGE2_COMMIT" \
  2>&1 | tee "$STAGE2_RUN_ROOT/role_init.log"

test -s "$ROLE_INIT_DIR/role_init_manifest.json"
test -f "$ROLE_INIT_DIR/ROLE_INIT_COMPLETE"
test ! -e "$ROLE_INIT_DIR/_SUCCESS"
echo ROLE_INIT_ONLY_PASS
```

`ROLE_INIT_COMPLETE`只证明模型资产、角色隔离和FSDP初始化；它不是训练checkpoint的 `_SUCCESS`。

## 7. 顺序运行 C0 → C1 → C2 smoke

三个命令必须使用同一个全新 `SMOKE_DIR`：C0 cold+save，C1只允许resume C0并强制纯DMD+save，C2只允许resume C1并强制DFD+discard。C2不得产生第三个checkpoint。

```bash
test ! -e "$SMOKE_DIR"

"$STAGE2_TORCHRUN" \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B train.py \
  --config_path "$STAGE2_CONFIG" \
  --logdir "$SMOKE_DIR" \
  --stage2-smoke C0 \
  2>&1 | tee "$STAGE2_RUN_ROOT/smoke_C0.log"

test -f "$SMOKE_DIR/checkpoint_stage2_g000001/_SUCCESS"

"$STAGE2_TORCHRUN" \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B train.py \
  --config_path "$STAGE2_CONFIG" \
  --logdir "$SMOKE_DIR" \
  --stage2-smoke C1 \
  2>&1 | tee "$STAGE2_RUN_ROOT/smoke_C1.log"

test -f "$SMOKE_DIR/checkpoint_stage2_g000002/_SUCCESS"

"$STAGE2_TORCHRUN" \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B train.py \
  --config_path "$STAGE2_CONFIG" \
  --logdir "$SMOKE_DIR" \
  --stage2-smoke C2 \
  2>&1 | tee "$STAGE2_RUN_ROOT/smoke_C2.log"

test ! -e "$SMOKE_DIR/checkpoint_stage2_g000003"
export SMOKE_JSONL="$SMOKE_DIR/metrics/stage2_train_metrics.jsonl"
test -s "$SMOKE_JSONL"

"$STAGE2_PYTHON" -I -B - "$SMOKE_JSONL" <<'PY'
import json
import sys

records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
cycles = [r for r in records if r.get("record_type") == "cycle_summary"][-3:]
ends = [r for r in records if r.get("record_type") == "run_end"][-3:]
checkpoints = [r for r in records if r.get("record_type") == "checkpoint_event"][-2:]
assert [r["smoke_mode"] for r in cycles] == ["C0", "C1", "C2"]
assert all(r["smoke_acceptance"]["status"] == "PASS" for r in cycles)
assert [r["smoke_mode"] for r in ends] == ["C0", "C1", "C2"]
assert all(r["status"] == "smoke_complete" and r["dry_run"] is True for r in ends)
assert [r["smoke_mode"] for r in checkpoints] == ["C0", "C1"]
print("STAGE2_C0_C1_C2_SMOKE_PASS")
PY

"$STAGE2_PYTHON" -I -B scripts/plot_stage2_training.py \
  --jsonl "$SMOKE_JSONL" \
  --output-dir "$SMOKE_DIR/plots_verified" \
  --include-dry-run

test -s "$SMOKE_DIR/plots_verified/index.html"
test "$(find "$SMOKE_DIR/plots_verified" -maxdepth 1 -type f | wc -l)" -eq 19
```

smoke内部已检查allocated/reserved/free显存、NVML余量、跨rank straggler以及C0→C1→C2 live allocation增长。任一 `smoke_acceptance.status` 不是 `PASS`，都不能开始formal。

## 8. 使用全新目录启动正式训练与resume

正式训练禁止resume smoke checkpoint，所以必须使用另一个从未存在的 `FORMAL_DIR`。首次启动显式关闭auto-resume，证明是cold start；不要传 `--stage2-smoke`、`--no-save` 或 `--no-visualize`。

```bash
test ! -e "$FORMAL_DIR"
test "$(PYTHONPATH="$PWD" "$STAGE2_PYTHON" -B -m utils.stage2_config \
  --config "$STAGE2_CONFIG" --hash-only)" = "$LAUNCH_HASH"

"$STAGE2_TORCHRUN" \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B train.py \
  --config_path "$STAGE2_CONFIG" \
  --logdir "$FORMAL_DIR" \
  --no-auto-resume \
  2>&1 | tee "$STAGE2_RUN_ROOT/formal.log"
```

若作业中断：

- 没有任何 `checkpoint_stage2_g*/_SUCCESS`：不要复用半成品目录，换一个全新 `FORMAL_DIR` cold start；
- 已有完整 `_SUCCESS`：保持同一代码commit、配置、环境变量、数据和 `FORMAL_DIR`，去掉 `--no-auto-resume` 后重启；
- 存在损坏、缺 `_SUCCESS` 或更晚的不完整checkpoint：保存现场并停止，trainer会fail-closed，禁止手删后假装resume。

正式resume命令：

```bash
test -n "$(find "$FORMAL_DIR" -maxdepth 2 -name _SUCCESS -print -quit)"

"$STAGE2_TORCHRUN" \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B train.py \
  --config_path "$STAGE2_CONFIG" \
  --logdir "$FORMAL_DIR" \
  2>&1 | tee -a "$STAGE2_RUN_ROOT/formal_resume.log"
```

baseline终点固定为 G=280、F=1400、cycle=280；最终checkpoint应为 `checkpoint_stage2_g000280`。G在第40次成功更新后初始化EMA，因此最终checkpoint必须包含 `generator_raw.safetensors`、`generator_ema.safetensors` 和 `fake_score_raw.safetensors`。

```bash
export FORMAL_JSONL="$FORMAL_DIR/metrics/stage2_train_metrics.jsonl"
export FINAL_CHECKPOINT="$FORMAL_DIR/checkpoint_stage2_g000280"

test -f "$FINAL_CHECKPOINT/_SUCCESS"
test -s "$FINAL_CHECKPOINT/checkpoint_manifest.json"
test -s "$FINAL_CHECKPOINT/generator_raw.safetensors"
test -s "$FINAL_CHECKPOINT/generator_ema.safetensors"
test -s "$FINAL_CHECKPOINT/fake_score_raw.safetensors"
test -s "$FORMAL_JSONL"

"$STAGE2_PYTHON" -I -B scripts/plot_stage2_training.py \
  --jsonl "$FORMAL_JSONL" \
  --output-dir "$FORMAL_DIR/plots_verified" \
  --require-complete

test -s "$FORMAL_DIR/plots_verified/index.html"
test "$(find "$FORMAL_DIR/plots_verified" -maxdepth 1 -type f | wc -l)" -eq 19

"$STAGE2_PYTHON" -I -B - "$FORMAL_JSONL" <<'PY'
import json
import sys

records = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8") if line.strip()]
run_end = [r for r in records if r.get("record_type") == "run_end"][-1]
assert run_end["status"] == "complete"
assert run_end["dry_run"] is False
assert run_end["smoke_mode"] is None
assert run_end["completed_generator_updates"] == 280
assert run_end["completed_fake_updates"] == 1400
assert run_end["completed_cycles"] == 280
print("STAGE2_FORMAL_TRAINING_PASS", run_end["run_id"])
PY

test "$(git rev-parse HEAD)" = "$STAGE2_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(git ls-files --others --ignored --exclude-standard)"
```

`plots_verified`应有9组PNG、9组SVG和1个 `index.html`：generator/fake loss、optimization、G/F/cycle吞吐、time breakdown、memory/straggler、timestep/exit/phase。

## 9. 回传的最小证据

```text
git commit：
GPU_ENV_PASS：
pytest 最后一行：
REAL_SCORE_MANIFEST_VERIFY_PASS：
GENERATOR_STEP3750_MANIFEST_PASS：
CONFIG_BINDING_PASS：
FORMAL_CACHE_AUDIT_PASS：
ROLE_INIT_ONLY_PASS：
STAGE2_C0_C1_C2_SMOKE_PASS：
C0/C1/C2 三份日志路径：
formal / resume 日志路径：
最终 checkpoint manifest_sha256：
STAGE2_FORMAL_TRAINING_PASS：
plots_verified/index.html 路径：
```

## 10. 常见失败：看到就停

| 输出或现象 | 含义与处理 |
|---|---|
| teacher `provenance.source_sha256` 等于 `merged_state_sha256` | 使用了旧sidecar；重新运行当前 `prepare_stage2.sh`，不要手改JSON。 |
| `generator_stage1_step` 不是3750 | 当前baseline resolver明确拒绝；不要用3075文件冒充3750。 |
| dirty checkout / ignored files | 模型、log或输出写进了执行clone；换clean clone并把所有产物移到仓库外，禁止直接 `git clean -fdx`。 |
| config binding或launch hash变化 | shell变量不完整或中途换路径；停止并从第4节完整重导，受影响的formal audit必须重做。 |
| `F25[:24] differs bitwise` | 原cache、视频、VAE或预处理provenance不一致；停止。 |
| action不是200/200/200 | 修正人工确认的sidecar；禁止从文本或顺序猜标签。 |
| C1找不到checkpoint | C0未完整保存，或使用了不同 `SMOKE_DIR`；不能跳到C1。 |
| C2产生g000003 checkpoint | smoke CLI/代码契约漂移；停止，不得进入formal。 |
| formal拒绝smoke checkpoint | 这是预期保护；formal必须使用全新目录cold start。 |
| incomplete checkpoint / missing `_SUCCESS` | 原子提交未完成；保存现场，禁止手补marker或删除损坏目录后继续。 |
| OOM、non-finite、NCCL hang、NVML/free/straggler失败 | 保存日志和 `nvidia-smi`；不要降低batch、门禁或研究配置后继续同一lineage。 |
