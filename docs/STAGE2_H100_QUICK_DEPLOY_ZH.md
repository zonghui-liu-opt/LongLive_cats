# Stage-2 检查点 A：内网 8×H100 快速部署

这份手册只完成 **Stage-2 训练前准备**：代码回归、模型资产、三角色初始化、F25 cache、文本条件、negative conditioning 和 600 条数据审计。

全部通过后，看到 `CHECKPOINT_A_H100_PREP_PASS` 就停下，把结果发回检查。**不要启动 Stage-2 训练。** 任一步非零退出、Traceback、OOM、NCCL 卡死、hash/bitwise 不一致，都立即停止；不要手改 manifest 或 tensor 绕过门禁。

## 0. 只改这一块：填写真实绝对路径

为什么：这些路径会进入本次 launch hash。第一次解析配置后，整条流程中不能换路径、换权重或换数据。

```bash
set -euo pipefail

# 使用与正式训练完全相同的 Python 环境。
export STAGE2_PYTHON=/绝对路径/conda_env/bin/python
export STAGE2_TORCHRUN="$(dirname "$STAGE2_PYTHON")/torchrun"
export STAGE2_GIT_SOURCE=https://github.com/zonghui-liu-opt/LongLive_cats.git

# fresh clone、日志和新产物；全部必须在仓库外使用独立目录。
export STAGE2_REPO=/local_nvme/longlive_stage2_checkpoint_a
export STAGE2_RUN_DIR=/local_nvme/longlive_stage2_runs/checkpoint_a_001
export STAGE2_ASSET_DIR=/local_nvme/longlive_stage2_assets/checkpoint_a_001
export STAGE2_CACHE_DIR=/local_nvme/longlive_stage2_cache/f25_600_v1
export NEGATIVE_DIR=/local_nvme/longlive_stage2_negative/v1
export ROLE_INIT_DIR="$STAGE2_RUN_DIR/role_init"

# Stage-1 Generator：正式 step3750 checkpoint 及其不可变 base。
export STAGE1_BASE=/绝对路径/stage1_immutable_base.pt
export STAGE1_CKPT=/绝对路径/checkpoint_model_003750

# real-score/fake-score 共用的初始化 teacher 与 Wan 架构目录。
# 注意：这里不是上面的 Stage-1 causal Generator，而是 Stage-1 训练前那份
# “猫域 SFT LoRA 已 merge、仍保持双向 attention”的 Wan teacher。
export ARCH_ROOT=/绝对路径/Wan2.2-TI2V-5B
export TEACHER_CKPT=/绝对路径/merged_bi-direct_Wan2.2-5B-cats/ckpts

# 上述双向 teacher 的原始 merge 记录。先运行：
# sha256sum "$TEACHER_PROVENANCE_RECORD"
# 再把输出第一列复制到 TEACHER_SOURCE_SHA256；不要填 LoRA 自身的 SHA。
export TEACHER_PROVENANCE_RECORD=/绝对路径/merge_manifest.json
export TEACHER_SOURCE_KIND=diffsynth_sft_lora_merge
export TEACHER_SOURCE_ID='Wan2.2-TI2V-5B_cats_LoRA_rank64_600clips_5e-5_ga4steps:epoch-79'
export TEACHER_SOURCE_SHA256=替换为merge_manifest.json的64位小写SHA256

# 原始 Stage-1 cache、原视频 metadata 和同源模型。
export STAGE1_CACHE_MANIFEST=/绝对路径/原始Stage1缓存/cache_manifest.json
export VAE_CKPT=/绝对路径/Wan2.2_VAE.pth
export T5_CKPT=/绝对路径/T5
export TOKENIZER_DIR=/绝对路径/tokenizer
export METADATA_600=/绝对路径/metadata_600.csv

# 官方 Stage-1 manifest 没有 action_id，因此本次必须提供人工确认的 sidecar。
# 精确表头：video,action_id；600 行；3 个 action 各 200 行；video 与 metadata 一一对应。
export ACTION_SIDECAR_600=/绝对路径/action_labels_600.csv
export ACTION_ID_1=替换为真实动作枚举1
export ACTION_ID_2=替换为真实动作枚举2
export ACTION_ID_3=替换为真实动作枚举3
export OPERATOR_ID=替换为真实执行人或任务ID
```

不要从 prompt、文件名或行号猜 action。不要把 `LONG_LIVE_STAGE2_ACTION_LABELS_PATH` 设为空字符串或字符串 `null`。

内网不能访问 GitHub 时：先在联网机器的最新 `stage-2` clone 中执行 `git bundle create LongLive-stage2.bundle stage-2`，把 bundle 拷入内网，再把 `STAGE2_GIT_SOURCE` 改成该 bundle 的绝对路径。必须通过 Git 克隆并保留 `.git`，不能只复制源码目录。

## 1. 获取 clean `stage-2`，确认 8 张 H100

做什么：fresh clone 已上传的 `stage-2`，禁止 Python bytecode 写进仓库，并核对 GPU。

为什么：正式 cache 工具会拒绝任何 tracked、untracked **或 ignored** 文件；8 卡拓扑也是锁定契约。

```bash
test -x "$STAGE2_PYTHON"
test -x "$STAGE2_TORCHRUN"
test ! -e "$STAGE2_REPO"
test ! -e "$STAGE2_RUN_DIR"
test ! -e "$STAGE2_ASSET_DIR"
test ! -e "$STAGE2_CACHE_DIR"
test ! -e "$NEGATIVE_DIR"

git clone --branch stage-2 --single-branch \
  "$STAGE2_GIT_SOURCE" "$STAGE2_REPO"
cd "$STAGE2_REPO"
git fetch origin stage-2
test "$(git rev-parse HEAD)" = "$(git rev-parse FETCH_HEAD)"
export STAGE2_COMMIT="$(git rev-parse HEAD)"

mkdir -p "$STAGE2_RUN_DIR" "$STAGE2_ASSET_DIR" \
  "$(dirname "$STAGE2_CACHE_DIR")" "$(dirname "$NEGATIVE_DIR")"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$STAGE2_RUN_DIR/pycache"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(git ls-files --others --ignored --exclude-standard)"
echo CLEAN_CHECKOUT_PASS

nvidia-smi -L | tee "$STAGE2_RUN_DIR/gpus.txt"
"$STAGE2_PYTHON" -I -B - <<'PY'
import torch

assert torch.cuda.device_count() == 8, torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(8)]
assert all("H100" in name for name in names), names
assert all(
    torch.cuda.get_device_properties(i).total_memory >= 79 * 1024**3
    for i in range(8)
)
assert torch.cuda.is_bf16_supported()
print("GPU_ENV_PASS", torch.__version__, torch.version.cuda, names)
PY
```

成功输出：

- `CLEAN_CHECKOUT_PASS`；两个 Git 检查都没有内容。
- `GPU_ENV_PASS`，后面列出 8 张 H100；每张显存至少约 79 GiB，BF16 可用。

## 2. 跑完整 Stage-2 CPU/接口回归

做什么：在加载 5B 权重前验证配置、F25 数据链、sampler、真实 tiny Wan 接口、rollout 和 DMD/DFD loss。

为什么：先排除代码和依赖问题，避免浪费 H100 时间。

```bash
PYTHONPATH="$PWD" \
"$STAGE2_PYTHON" -B -m pytest -q -p no:cacheprovider \
  tests/test_stage2_*.py \
  2>&1 | tee "$STAGE2_RUN_DIR/stage2_tests.log"

test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(git ls-files --others --ignored --exclude-standard)"
```

成功输出：`329 passed`、`0 failed`。14 条现有 `torch.jit.script_method` 弃用 warning 不影响通过；出现其他失败就停。

## 3. 准备 Generator 和 real-score teacher

先区分两类权重，后面不要交叉使用：

- **Generator（G）**：Stage-1 causal base + step3750 EMA LoRA；就是第 3.1 节合并和你已经对比推理的模型。
- **real-score / fake-score 的共同底座**：Stage-1 训练前的猫域双向 Wan teacher。real-score 加载后完全冻结；fake-score 从同一权重独立初始化，再挂新的 r64 LoRA。

因此，Stage-1 causal merged 模型即使推理效果正常，也不能填到 `TEACHER_CKPT`。

### 3.1 合并 Stage-1 step3750 EMA Generator

做什么：只选择正式 checkpoint 中的 `adapter_ema.safetensors`，严格合并回不可变 causal base。

为什么：Stage-2 的 G 必须从 Stage-1 step3750 EMA 起步，不能误用 raw adapter 或其他 step。

```bash
export G_MERGED="$STAGE2_ASSET_DIR/stage1_step3750_ema_merged.pt"
export G_MANIFEST="$STAGE2_ASSET_DIR/stage1_step3750_ema_merged.manifest.json"
test ! -e "$G_MERGED"
test ! -e "$G_MANIFEST"

CUDA_VISIBLE_DEVICES=0 \
"$STAGE2_PYTHON" -I -B scripts/merge_lora_generator.py \
  --base-checkpoint "$STAGE1_BASE" \
  --training-checkpoint "$STAGE1_CKPT" \
  --output-path "$G_MERGED" \
  --device cuda:0 \
  2>&1 | tee "$STAGE2_RUN_DIR/generator_merge.log"

test -s "$G_MERGED"
test -s "$G_MANIFEST"
```

成功输出：

```text
Merged EMA adapter to ...
SHA256: <64位sha256>
```

`_SUCCESS`、step、EMA metadata、target 数量或 strict reload 任一不匹配，脚本会失败；不要手补旧 checkpoint。

### 3.2 给双向 real-score teacher 生成“身份证” manifest

这一步**不会生成或修改模型权重**，只读取已经合并好的双向 teacher，检查权重文件、BF16 tensor 和 Wan 架构，然后写出一份 JSON sidecar。下面命令适用于“已经是原生 Wan safetensors 格式”的 teacher。

三个需要人工填写的 `source-*` 字段只是记录 teacher 从哪里来：

| 字段 | 人话解释 | 本项目建议值 |
|---|---|---|
| `source-kind` | teacher 的产生方式 | `diffsynth_sft_lora_merge` |
| `source-identifier` | 稳定、可读的训练/合并任务名 | `Wan2.2-TI2V-5B_cats_LoRA_rank64_600clips_5e-5_ga4steps:epoch-79` |
| `source-sha256` | 上述来源记录的防篡改指纹 | 对配套的 `merge_manifest.json` 运行 `sha256sum` 后的第一列 |

不要把 Stage-1 step3750 LoRA/merged Generator 的 SHA 填到这里，也不要填 `merge_manifest.json` 里的 `lora_sha256`：它只代表 LoRA 文件，不能绑定完整的 base + LoRA merge 记录。当前 teacher 的实际 safetensors 文件 hash 会由下面的脚本自动逐个计算，不需要手工填写。

代码可以自动检查权重结构和 hash，但“它确实是猫域、双向 TI2V、video-global flow teacher”仍需由掌握训练来源的人确认；这就是两个 `--attest-*` 开关的含义。

```bash
export TEACHER_MANIFEST="$STAGE2_ASSET_DIR/real_score_teacher.manifest.json"
test ! -e "$TEACHER_MANIFEST"
test -s "$TEACHER_PROVENANCE_RECORD"
test "$(sha256sum "$TEACHER_PROVENANCE_RECORD" | awk '{print $1}')" = \
  "$TEACHER_SOURCE_SHA256"

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
  --attest-video-global-flow \
  2>&1 | tee "$STAGE2_RUN_DIR/teacher_manifest.log"

test -s "$TEACHER_MANIFEST"
```

成功输出：

```text
Wrote ...
Manifest SHA256: <64位sha256>
```

这里的 `--conversion-command none` 表示：`TEACHER_CKPT` 已经是最终使用的原生 Wan teacher，在生成本 sidecar 前没有再做额外格式转换；此前的 base + SFT LoRA merge 历史由 `TEACHER_PROVENANCE_RECORD` 绑定。如果 teacher 后来又转成 LongLive wrapper 或经过其他格式转换，就不能继续写 `none`，必须改成真实格式、selector 和完整转换 argv。来源语义无法确认就停。

## 4. 锁定同一份运行配置，做 8 卡三角色 init-only

做什么：一次性导出正式路径，确认 contract hash，然后初始化 G、frozen real-score 和 F 三个独立 5B 角色。

为什么：先验证权重、LoRA、对象隔离和 FSDP2，避免先花时间重提 cache 才发现模型资产不可用。

```bash
export NEGATIVE_MANIFEST="$NEGATIVE_DIR/negative_conditioning_manifest.json"
export STAGE2_CONFIG="$PWD/configs/train_i2v_stage2.yaml"

export LONG_LIVE_STAGE2_ARCHITECTURE_ROOT="$ARCH_ROOT"
export LONG_LIVE_STAGE2_GENERATOR_BASE="$G_MERGED"
export LONG_LIVE_STAGE2_GENERATOR_MANIFEST="$G_MANIFEST"
export LONG_LIVE_STAGE2_REAL_SCORE_BASE="$TEACHER_CKPT"
export LONG_LIVE_STAGE2_REAL_SCORE_MANIFEST="$TEACHER_MANIFEST"
export LONG_LIVE_STAGE2_METADATA_PATH="$METADATA_600"
export LONG_LIVE_STAGE2_ACTION_LABELS_PATH="$ACTION_SIDECAR_600"
export LONG_LIVE_STAGE2_CACHE_DIR="$STAGE2_CACHE_DIR"
export LONG_LIVE_STAGE2_NEGATIVE_MANIFEST="$NEGATIVE_MANIFEST"

CONTRACT_HASH="$(
  PYTHONPATH="$PWD" "$STAGE2_PYTHON" -B -m utils.stage2_config \
    --config "$STAGE2_CONFIG" --contract-hash-only
)"
test "$CONTRACT_HASH" = \
  aa4d7be1e05c846df14cee5417a298afe668429f41faa671f3021754a5616c00

PYTHONPATH="$PWD" "$STAGE2_PYTHON" -B -m utils.stage2_config \
  --config "$STAGE2_CONFIG" > "$STAGE2_RUN_DIR/resolved_config.json"
echo "CONFIG_BINDING_PASS contract=$CONTRACT_HASH"

test ! -e "$ROLE_INIT_DIR"
"$STAGE2_TORCHRUN" \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B \
  scripts/preflight_stage2_roles.py \
  --config "$STAGE2_CONFIG" \
  --output-dir "$ROLE_INIT_DIR" \
  --expected-git-commit "$STAGE2_COMMIT" \
  2>&1 | tee "$STAGE2_RUN_DIR/role_init.log"

test -s "$ROLE_INIT_DIR/role_init_manifest.json"
test -f "$ROLE_INIT_DIR/ROLE_INIT_COMPLETE"
test ! -e "$ROLE_INIT_DIR/_SUCCESS"
```

成功输出：rank 0 最后一条 JSON 含：

```json
{"status":"passed","manifest_sha256":"...","rank_consensus_sha256":"..."}
```

这一步同时严格检查：单机 world size 8、1D `FULL_SHARD`、G 的 r32 LoRA、real-score 全冻结、F 的 r64 LoRA、三角色参数/存储隔离，以及 forward/optimizer/EMA/T5/VAE/DataLoader 全部未创建或未调用。`ROLE_INIT_COMPLETE` 只是初始化审计，不是训练 `_SUCCESS`。

## 5. 检查真实 tensor，并生成 native F25 cache

做什么：逐条读取原 cache 的实际 safetensors，而不是相信 manifest 声明；按实际 latent 帧数处理 600 条数据。

为什么：Stage-2 固定需要 1 个 sink + 24 个真实 future latent，共 F25。方案不变：F25 复用，F24 必须从原 pixel 视频的 0–96 帧重新提取。

输入要求：不能只拷 `cache_manifest.json`。原 cache 的全部 safetensors，以及 `METADATA_600` 引用的 600 个原视频和 input image，都必须保持原路径可读且 hash 不变；即使已有 proven F25，脚本也会逐条复核这些来源。

```bash
test ! -e "$STAGE2_CACHE_DIR"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(git ls-files --others --ignored --exclude-standard)"

"$STAGE2_TORCHRUN" \
  --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B \
  scripts/prepare_stage2_i2v_f25_cache.py \
  --config-path "$STAGE2_CONFIG" \
  --source-cache-manifest "$STAGE1_CACHE_MANIFEST" \
  --vae-checkpoint "$VAE_CKPT" \
  2>&1 | tee "$STAGE2_RUN_DIR/f25_prepare.log"

export F25_BASE="$STAGE2_CACHE_DIR/cache_manifest.json"
export F25_SUCCESS="$STAGE2_CACHE_DIR/_F25_SUCCESS.json"
test -s "$F25_BASE"
test -s "$F25_SUCCESS"

"$STAGE2_PYTHON" -I -B - "$F25_BASE" <<'PY'
import json
import sys

m = json.load(open(sys.argv[1], encoding="utf-8"))
s = m["preparation"]["summary"]
assert m["schema"] == "longlive_stage2_i2v_f25_source_cache"
assert set(s) == {"reused_f25", "reverified_f25", "reencoded_f24"}
assert sum(s.values()) == 600
print("F25_CACHE_PASS", s, m["manifest_sha256"])
PY
```

成功输出：各 rank 有 `source_scan=...`、`assigned=...` 和 `completed=...` 进度；rank 0 有 `final_verify=...`；最后 JSON 含 `"status":"ok"`，并且 `F25_CACHE_PASS` 的三项总和为 600。

- `reused_f25`：proven F25，原 artifact 字节直接复用，不加载 VAE。
- `reverified_f25`：已有 F25 缺少旧的 97 帧声明；用同一 VAE 对原视频 0–96 帧完整复验，逐 bit 一致后仍发布原 artifact 字节。
- `reencoded_f24`：F24 用同一 VAE 对原视频 0–96 帧一次性重提 F25，并强制新 F25 的前 24 帧与旧 F24 逐 bit 一致；initial/prompt/mask 原样保留。

禁止 padding、复制 latent、截断或把目标改成 23 个新 latent。原视频不足 97 帧、VAE 不同、F24 前缀不一致，都必须失败且不能产生成功 marker。

## 6. 补齐 positive text provenance，并生成 negative conditioning

做什么：先证明 positive cache 使用锁定的 T5/tokenizer 文本设置，再用完全相同的 T5/tokenizer 编码统一 negative prompt。

为什么：positive/negative 文本空间不同会让 CFG 和 DMD/DFD 训练失真。

```bash
export F25_ATTESTED="$STAGE2_CACHE_DIR/cache_manifest.attested.json"
export F25_BASE_SELF_SHA="$(
  "$STAGE2_PYTHON" -I -B -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["manifest_sha256"])' \
    "$F25_BASE"
)"
export TEXT_ATTESTATION='I attest that this legacy positive cache was encoded with the locked Wan seq512/whitespace/add-special-tokens/right-padding/exact-zero-padding contract.'

test ! -e "$F25_ATTESTED"
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
  2>&1 | tee "$STAGE2_RUN_DIR/source_attestation.log"

test -s "$F25_ATTESTED"
test ! -e "$NEGATIVE_DIR"
CUDA_VISIBLE_DEVICES=0 \
"$STAGE2_PYTHON" -I -B scripts/audit_stage2_i2v_cache.py \
  prepare-negative \
  --source-cache-manifest "$F25_ATTESTED" \
  --t5-checkpoint "$T5_CKPT" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --output-dir "$NEGATIVE_DIR" \
  --expected-num-samples 600 \
  --device cuda:0 \
  2>&1 | tee "$STAGE2_RUN_DIR/negative_prepare.log"

test -s "$NEGATIVE_DIR/negative_conditioning.safetensors"
test -s "$NEGATIVE_MANIFEST"
```

成功输出：

- attestation 的最终 JSON 含 `"status":"ok"`、`manifest_sha256`、`text_encoding_contract_sha256` 和 `"code_version":"git:<当前commit>"`。
- negative 的最终 JSON 含 `"status":"ok"`、64 位 `artifact_sha256`，且 `prompt_valid_tokens > 0`。

`F25_BASE_SELF_SHA` 是 manifest 内部的 self-hash，不是 `sha256sum` 的文件 hash。正常首次部署不要使用 `--force`。negative 必须放在 F25 cache 目录外。

## 7. 正式扫描 600 条 cache

做什么：用同一份 env-resolved YAML 全扫 600 个 F25 artifact，绑定 metadata、action、negative、配置和代码版本。

为什么：这是训练数据唯一正式入口；没有这个 manifest，后续 trainer 必须拒绝启动。

```bash
"$STAGE2_PYTHON" -I -B scripts/audit_stage2_i2v_cache.py \
  audit \
  --config-path "$STAGE2_CONFIG" \
  --source-cache-manifest "$F25_ATTESTED" \
  --action-id "$ACTION_ID_1" \
  --action-id "$ACTION_ID_2" \
  --action-id "$ACTION_ID_3" \
  2>&1 | tee "$STAGE2_RUN_DIR/cache_audit.log"

export FINAL_CACHE_MANIFEST="$STAGE2_CACHE_DIR/stage2_i2v_manifest.json"
test -s "$FINAL_CACHE_MANIFEST"

"$STAGE2_PYTHON" -I -B - "$FINAL_CACHE_MANIFEST" <<'PY'
import json
import sys

m = json.load(open(sys.argv[1], encoding="utf-8"))
assert m["schema"] == "longlive_stage2_i2v_cache"
assert m["num_samples"] == 600
assert sorted(m["actions"]["counts"].values()) == [200, 200, 200]
assert sum(m["orientation_counts"].values()) == 600
print("FORMAL_CACHE_AUDIT_PASS", m["actions"]["counts"], m["manifest_sha256"])
PY
```

成功输出：CLI 最终 JSON 含 `"status":"ok"`、`"num_samples":600`，三个 `action_counts` 各 200；随后打印 `FORMAL_CACHE_AUDIT_PASS`。

不要用 `--metadata-path`、`--cache-dir`、`--negative-conditioning-manifest`、`--action-labels-path` 或 `--output-manifest` 临时换路径；正式路径只能来自第 4 节锁定的环境变量。

## 8. 最终检查、回传，然后停止

做什么：只验证检查点 A 必需产物和禁止项，不运行任何训练命令。

```bash
test "$(git rev-parse HEAD)" = "$STAGE2_COMMIT"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test -z "$(git ls-files --others --ignored --exclude-standard)"
test -s "$G_MERGED"
test -s "$G_MANIFEST"
test -s "$TEACHER_MANIFEST"
test -s "$ROLE_INIT_DIR/role_init_manifest.json"
test -f "$ROLE_INIT_DIR/ROLE_INIT_COMPLETE"
test ! -e "$ROLE_INIT_DIR/_SUCCESS"
test -s "$F25_BASE"
test -s "$F25_SUCCESS"
test -s "$F25_ATTESTED"
test -s "$NEGATIVE_MANIFEST"
test -s "$FINAL_CACHE_MANIFEST"
echo CHECKPOINT_A_H100_PREP_PASS
```

成功输出：`CHECKPOINT_A_H100_PREP_PASS`。到这里立即停止，不运行 `train.py`，也不创建 optimizer、训练 checkpoint、训练日志或可视化。

请回传下面这些内容：

```text
git commit：
GPU_ENV_PASS 那一行：
pytest 最后一行：
Generator SHA256：
Teacher manifest SHA256：
role-init 最后一条 JSON：
F25_CACHE_PASS 那一行：
attestation 最后一条 JSON：
negative 最后一条 JSON：
formal audit 最后一条 JSON：
FORMAL_CACHE_AUDIT_PASS 那一行：
最终标记：CHECKPOINT_A_H100_PREP_PASS
```

## 常见失败：看到就停

| 输出或现象 | 含义 / 正确处理 |
|---|---|
| `Refusing non-isolated...` | 命令缺少 `-I -B`；按本文原命令重跑。 |
| dirty checkout / ignored files | 仓库里出现改动、`.pytest_cache`、`__pycache__` 等；换 fresh clone 或确认来源后移走，**不要直接 `git clean -fdx`**。 |
| world-size mismatch | 必须单机 8 进程、8 张 H100。 |
| VAE/T5/tokenizer hash mismatch | 使用的不是原 positive cache 的同源资产；不要强行继续。 |
| `F25[:24] differs bitwise` | 原 cache、视频、VAE 或预处理 provenance 不一致；立即停止。 |
| action 不是 200/200/200 | 修正人工确认的 action 标签来源；禁止从文本或行号推断。 |
| unexpected files in F25 output | 日志、negative 或其他文件放错目录；用全新的独立输出目录重跑。 |
| config/launch hash changed | 中途换了环境变量或路径；不要复用旧输出目录。 |
| NCCL hang、OOM、non-finite | 保存日志与 `nvidia-smi` 现场并停止，不得降低门禁或改研究方案。 |
