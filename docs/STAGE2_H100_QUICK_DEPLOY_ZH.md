# Stage‑2 内网 H100 快速部署与实验记录

本文是 Stage‑2 的滚动运行手册。每次内网部署都新增一条记录，写清代码版本、命令、通过标准、真实结果和产物路径；失败不得改写为通过。

约定：

- 所有命令都在仓库根目录、正式 LongLive Python 环境中执行。
- 每次实验使用新的 `run_id` 和输出目录，不覆盖旧目录。
- 只提交文本日志、manifest 和结论；模型、cache、视频等大文件留在内网。
- `待执行 / 待确认 / 失败 / 通过` 必须对应真实状态。当前门禁未通过前，不进入下一批代码。

## 运行总览

| run_id | 日期 | 代码基线 | 内容 | 状态 |
|---|---|---|---|---|
| `H100-001` | 2026‑08‑08 | `639ba658a3a7f40bb792fbca4100877d2f7554df` | Batch 1 配置契约 | 通过 |
| `H100-002` | 待执行 | 运行时 `stage-2` clean HEAD | Batch 2 三角色 init-only | 待执行 |

## 通用准备

首次获取分支：

```bash
git fetch longlive-cats stage-2
git switch --track -c stage-2 longlive-cats/stage-2
```

已有本地分支：

```bash
git switch stage-2
git pull --ff-only longlive-cats stage-2
test "$(git rev-parse --abbrev-ref HEAD)" = stage-2
test "$(git rev-parse HEAD)" = "$(git rev-parse longlive-cats/stage-2)"
```

每次执行前记录环境，`STAGE2_RUN_DIR` 必须是本次新目录：

```bash
export STAGE2_RUN_ID=H100-001
export STAGE2_RUN_DIR=/local_nvme/longlive_stage2_runs/$STAGE2_RUN_ID
test ! -e "$STAGE2_RUN_DIR"
mkdir -p "$STAGE2_RUN_DIR"

git rev-parse HEAD | tee "$STAGE2_RUN_DIR/git_commit.txt"
python --version 2>&1 | tee "$STAGE2_RUN_DIR/python_version.txt"
python -m pip show torch diffusers transformers peft omegaconf \
  > "$STAGE2_RUN_DIR/python_packages.txt"
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader > "$STAGE2_RUN_DIR/gpu_inventory.csv"
```

## H100‑001：Batch 1 配置契约门禁

### 目标与边界

验证内网依赖栈能够解析 Stage‑2 raw YAML、得到固定派生值，并保持 Stage‑1/DMD 回归。

本次不会加载 CUDA 模型、权重或 600 条 cache，不会启动训练，也不代表 C0/C1/C2 通过。UniPC 只验证仓库 scheduler 的只读 timetable characterization。

### 1. 确认代码版本

```bash
export STAGE2_CODE_BASE=639ba658a3a7f40bb792fbca4100877d2f7554df
git merge-base --is-ancestor "$STAGE2_CODE_BASE" HEAD
git diff --exit-code "$STAGE2_CODE_BASE" HEAD -- \
  configs/train_i2v_stage2.yaml \
  utils/stage2_config.py \
  tests/test_stage2_config.py
git status --short --branch
```

通过标准：当前分支包含该代码基线，且其后的文档提交没有改变本批三个实现文件。

### 2. 运行 Stage‑2 契约测试

```bash
set -o pipefail
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider tests/test_stage2_config.py \
  2>&1 | tee "$STAGE2_RUN_DIR/stage2_config_tests.log"
```

通过标准：`103 passed`，并验证 UniPC K4/shift5 timetable 为 `[999, 937, 833, 624]`、terminal sigma 为 0。

### 3. 运行相关回归

```bash
set -o pipefail
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider \
  tests/test_stage1_config.py \
  tests/test_stage1_lazy_imports.py \
  tests/test_dmd_i2v_conditioning.py \
  tests/test_i2v_sequence_parallel_config.py \
  tests/test_i2v_teacher_forcing_context.py \
  tests/test_stage1_schedule.py \
  tests/test_stage1_loss_metrics.py \
  tests/test_trainable_ema.py \
  tests/test_lora_utils.py \
  tests/test_error_buffer.py \
  tests/test_stage1_i2v_cache.py \
  tests/test_stage1_sampler.py \
  tests/test_distributed_sampler_seed.py \
  --deselect tests/test_stage1_config.py::test_release_stage1_config_has_one_locked_source_of_truth \
  2>&1 | tee "$STAGE2_RUN_DIR/related_regression_tests.log"
```

通过标准：`63 passed`。显式排除的测试只锁定公开仓库 Stage‑1 release YAML；真实内网 Stage‑1 使用不同训练参数，且 Batch 1 未修改 Stage‑1 YAML 或其解析路径。其余共享基础设施回归必须全部通过。

### 4. 导出配置契约

```bash
set -o pipefail
PYTHONPATH="$PWD" python -m utils.stage2_config \
  --config configs/train_i2v_stage2.yaml \
  --contract-hash-only \
  | tee "$STAGE2_RUN_DIR/contract_hash.txt"

PYTHONPATH="$PWD" python -m utils.stage2_config \
  --config configs/train_i2v_stage2.yaml \
  > "$STAGE2_RUN_DIR/resolved_config.json"
```

通过标准：

- contract hash 为 `aa4d7be1e05c846df14cee5417a298afe668429f41faa671f3021754a5616c00`；
- `capacity=17`、`score_seq_len=9750`、`G=280`、`F=1400`；
- Generator EMA target 为 `generator_adapter`；
- launch hash 可因内网绝对路径不同而变化，不要求与本地一致。

### 5. 真实结果

用户已回报：

```text
执行人：未提供
执行时间：未提供
节点/调度任务号：未提供
当前HEAD：未提供
代码基线：639ba658a3a7f40bb792fbca4100877d2f7554df
Stage‑2 tests：103 passed
相关回归：63 passed
显式排除：test_release_stage1_config_has_one_locked_source_of_truth
排除原因：公开Stage‑1 release YAML与实际内网Stage‑1训练参数不一致；本批未改Stage‑1配置路径
配置契约：全部通过
产物目录：未提供
结论：Batch 1门禁通过，允许开始Batch 2
失败摘要：无
```

## H100‑002：Batch 2 三角色 init-only 门禁

### 目标与边界

只验证以下闭环：Stage‑1 step3750 EMA 合并资产、独立 real-score teacher、G/real/F 三个独立 DiT、G/F 独立 LoRA，以及单机 8×H100 的一维 FSDP2 `FULL_SHARD`。

本次不执行任何 forward/backward，不创建 optimizer、EMA、T5、VAE 或 DataLoader，也不验证显存候选、C0/C1/C2、cache、score、rollout、loss、checkpoint/resume 或训练。

### 1. 固定 clean HEAD 与新目录

本节是 H100‑002 的完整准备流程；不要先重复执行上面的通用运行目录创建块。

```bash
set -euo pipefail
git switch stage-2
git pull --ff-only longlive-cats stage-2
test "$(git rev-parse --abbrev-ref HEAD)" = stage-2
test "$(git rev-parse HEAD)" = "$(git rev-parse longlive-cats/stage-2)"

export STAGE2_RUN_ID=H100-002
export STAGE2_RUN_DIR=/local_nvme/longlive_stage2_runs/$STAGE2_RUN_ID
export STAGE2_INIT_DIR=$STAGE2_RUN_DIR/role_init
test ! -e "$STAGE2_RUN_DIR"
mkdir -p "$STAGE2_RUN_DIR"

export STAGE2_CODE_BASE=$(git rev-parse HEAD)
git status --porcelain=v1 --untracked-files=all \
  > "$STAGE2_RUN_DIR/git_status_before.txt"
test ! -s "$STAGE2_RUN_DIR/git_status_before.txt"
printf '%s\n' "$STAGE2_CODE_BASE" > "$STAGE2_RUN_DIR/git_commit.txt"
python --version 2>&1 | tee "$STAGE2_RUN_DIR/python_version.txt"
python -m pip show torch accelerate diffusers transformers peft safetensors omegaconf \
  > "$STAGE2_RUN_DIR/python_packages.txt"
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
  --format=csv,noheader > "$STAGE2_RUN_DIR/gpu_inventory.csv"
```

日志、权重和输出必须放在仓库外。preflight 会拒绝 dirty worktree，也会拒绝已经存在的 `STAGE2_INIT_DIR`。

### 2. 生成严格输入资产

#### 2.1 重生 Stage‑1 step3750 EMA merge v2

```bash
set -euo pipefail
export STAGE1_BASE=/path/to/stage1_immutable_base.pt
export STAGE1_CKPT=/path/to/checkpoint_model_003750
export STAGE2_ASSET_DIR=/local_nvme/longlive_stage2_assets
mkdir -p "$STAGE2_ASSET_DIR"

export G_MERGED=$STAGE2_ASSET_DIR/stage1_step3750_ema_merged.pt
export G_MANIFEST=${G_MERGED%.pt}.manifest.json
test ! -e "$G_MERGED"
test ! -e "$G_MANIFEST"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
python scripts/merge_lora_generator.py \
  --base-checkpoint "$STAGE1_BASE" \
  --training-checkpoint "$STAGE1_CKPT" \
  --output-path "$G_MERGED" \
  --device cuda:0 \
  2>&1 | tee "$STAGE2_RUN_DIR/generator_merge.log"

test -s "$G_MERGED"
test -s "$G_MANIFEST"
```

输出必须位于 Stage‑1 checkpoint 目录之外。脚本要求正式 `_SUCCESS`、6/3/2 拓扑、step3750、raw/EMA 两份 safetensors 的正式 metadata、180 targets/360 tensors，并在发布前后重验全部来源。若历史 checkpoint 缺字段，立即停止并保留原件，不得手工补 metadata 或绕过校验。

#### 2.2 为独立 real-score teacher 建立 sidecar

先由操作者确认它确实是 cat-domain、bidirectional TI2V、video-global flow teacher；这两个语义不能从权重键名自动推断。

```bash
set -euo pipefail
export ARCH_ROOT=/path/to/Wan2.2-TI2V-5B
export TEACHER_CKPT=/path/to/cat_domain_bidirectional_teacher
export TEACHER_MANIFEST=$STAGE2_ASSET_DIR/real_score_teacher.manifest.json
export TEACHER_SOURCE_KIND=direct_internal_training_checkpoint
export TEACHER_SOURCE_ID=REPLACE_WITH_TRUSTED_JOB_OR_CONVERSION_ID
export TEACHER_SOURCE_SHA256=REPLACE_WITH_TRUSTED_64_HEX_SHA256

# 原生 Wan safetensors：TEACHER_CKPT 可为单文件、index.json 或目录。
export TEACHER_FORMAT=wan_native_transformer
export TEACHER_SELECTOR=root

# 若是 LongLive .pt 包装，可改为：
# export TEACHER_FORMAT=longlive_wrapper_pt
# export TEACHER_SELECTOR=real_score  # 优先使用real_score或model
# legacy payload只有在可信bidirectional teacher确实保存到generator键时才选generator；
# 绝不能把Stage-1 causal Generator冒充real-score teacher。

test ! -e "$TEACHER_MANIFEST"

PYTHONPATH="$PWD" python scripts/create_stage2_teacher_manifest.py \
  --checkpoint "$TEACHER_CKPT" \
  --architecture-root "$ARCH_ROOT" \
  --output "$TEACHER_MANIFEST" \
  --checkpoint-format "$TEACHER_FORMAT" \
  --state-dict-selector "$TEACHER_SELECTOR" \
  --source-kind "$TEACHER_SOURCE_KIND" \
  --source-identifier "$TEACHER_SOURCE_ID" \
  --source-sha256 "$TEACHER_SOURCE_SHA256" \
  --conversion-command none \
  --attest-cat-domain-bidirectional-ti2v \
  --attest-video-global-flow \
  2>&1 | tee "$STAGE2_RUN_DIR/teacher_manifest.log"

test -s "$TEACHER_MANIFEST"
```

原生路径只接受 BF16 safetensors，并绑定 index 与全部 shards；LongLive包装内的浮点tensor也必须全为BF16且不得含LoRA。`.bin`、FP16/FP32、裸 `.pt` state dict 或未知格式会 fail-closed。需要转换时，先在独立目录生成规范资产，再用多个 `--conversion-command <argv项>` 记录真实转换命令；不得写 `none` 冒充未转换。

### 3. 绑定 Stage‑2 raw YAML 的运行时路径

```bash
set -euo pipefail
export LONG_LIVE_STAGE2_ARCHITECTURE_ROOT="$ARCH_ROOT"
export LONG_LIVE_STAGE2_GENERATOR_BASE="$G_MERGED"
export LONG_LIVE_STAGE2_GENERATOR_MANIFEST="$G_MANIFEST"
export LONG_LIVE_STAGE2_REAL_SCORE_BASE="$TEACHER_CKPT"
export LONG_LIVE_STAGE2_REAL_SCORE_MANIFEST="$TEACHER_MANIFEST"

# 本批不读取下面两项，但仍应绑定未来正式资产，确保launch hash可追溯。
export LONG_LIVE_STAGE2_CACHE_DIR=/path/to/stage2_i2v_600_bf16
export LONG_LIVE_STAGE2_NEGATIVE_MANIFEST=/path/to/stage2_negative_conditioning_manifest.json
```

### 4. 运行 Batch 2 本地契约门禁

```bash
set -euo pipefail
PYTHONPATH="$PWD" PYTHONDONTWRITEBYTECODE=1 \
python -m pytest -q -p no:cacheprovider \
  tests/test_stage2_config.py \
  tests/test_stage2_role_initialization.py \
  tests/test_stage2_role_manifest.py \
  tests/test_stage2_init_only.py \
  tests/test_lora_utils.py \
  tests/test_merge_lora_generator.py \
  tests/test_diffsynth_causal_converter.py \
  tests/test_stage1_lazy_imports.py \
  tests/test_stage1_fsdp2.py \
  2>&1 | tee "$STAGE2_RUN_DIR/batch2_tests.log"
```

通过标准：`212 passed`，无失败。警告必须逐条确认仅为已知的 `torch.jit` deprecation。

### 5. 启动 8×H100 init-only preflight

```bash
set -euo pipefail
PYTHONPATH="$PWD" torchrun --standalone --nnodes=1 --nproc-per-node=8 \
  scripts/preflight_stage2_roles.py \
  --config configs/train_i2v_stage2.yaml \
  --output-dir "$STAGE2_INIT_DIR" \
  --expected-git-commit "$STAGE2_CODE_BASE" \
  2>&1 | tee "$STAGE2_RUN_DIR/role_init_torchrun.log"
```

任何 hash/provenance/shape/dtype/LoRA/FSDP/8-rank consensus 错误、OOM、NCCL hang、tripwire、fallback 或非零退出码都判失败，不得继续 Batch 3。

### 6. 验收原子产物

```bash
set -euo pipefail
test -s "$STAGE2_INIT_DIR/role_init_manifest.json"
test -f "$STAGE2_INIT_DIR/ROLE_INIT_COMPLETE"
test ! -e "$STAGE2_INIT_DIR/_SUCCESS"

PYTHONPATH="$PWD" python - "$STAGE2_INIT_DIR/role_init_manifest.json" <<'PY'
import json, os, sys
from utils.stage1_io import canonical_json_sha256

p = sys.argv[1]
m = json.load(open(p, encoding="utf-8"))
body = {k: v for k, v in m.items() if k != "manifest_sha256"}
assert m["manifest_sha256"] == canonical_json_sha256(body)
assert m["artifact_kind"] == "init_only_audit_not_training_checkpoint"
assert m["code"]["git_commit"] == os.environ["STAGE2_CODE_BASE"]
assert m["role_isolation"]["parameter_objects_disjoint"] is True
assert m["role_isolation"]["parameter_storage_disjoint"] is True
assert m["assets"]["real_score"]["checkpoint_sha256"] == m["assets"]["fake_score_base_sha256"]

expected = {
    "generator": (32, 180, 360, 57016320, False),
    "real_score": (None, 0, 0, 0, False),
    "fake_score": (64, 180, 360, 114032640, True),
}
for role, (rank, targets, tensors, params, checkpointing) in expected.items():
    item = m["roles"][role]
    assert item["strict_reload_succeeded"] is True
    assert item["activation_checkpointing"] is checkpointing
    assert item["pre_fsdp_adapter_tensor_count"] == tensors
    assert item["pre_fsdp_trainable_parameters"] == params
    if rank is not None:
        assert item["target_audit"]["rank"] == rank
        assert item["target_audit"]["target_module_count"] == targets
    else:
        assert item["target_audit"] is None
    post = item["post_fsdp"]
    assert post["all_parameters_are_dtensor"] is True
    assert post["fsdp_module_count"] == 31
    assert post["mesh_shape"] == [8]
    assert post["mesh_dim_names"] == ["shard"]
    assert post["placements"] == ["shard:0"]

assert m["fsdp"]["world_size"] == 8
assert m["fsdp"]["sharding_strategy"] == "FULL_SHARD"
assert m["fsdp"]["all_roles_independently_wrapped"] is True
assert m["side_effects"]["forward_calls"] == 0
for key in ("optimizer_created", "ema_created", "text_encoder_created", "vae_created", "dataloader_created"):
    assert m["side_effects"][key] is False
print("H100-002 manifest audit: PASS")
PY
```

通过只代表资产、三角色初始化、独立 LoRA 和 8 卡 FSDP2 拓扑闭环；它不是训练 checkpoint，也不放行 C0/C1/C2。

### 7. 真实结果

```text
状态：待执行
执行人：
执行时间：
节点/调度任务号：
当前HEAD：
Batch 2 tests：
generator merge manifest SHA256：
teacher manifest SHA256：
role-init manifest SHA256：
产物目录：
失败摘要：
结论：通过后才允许开始Batch 3
```

## 后续运行记录模板

复制本节并追加为新的 `H100-NNN`，同时更新“运行总览”。

````markdown
## H100-NNN：<批次/实验名>

### 元信息

- 日期：
- 执行人：
- branch / commit：
- 节点 / GPU：
- run_id / 输出目录：

### 目标与明确不做

- 本次验证：
- 本次不验证：

### 输入与资产

- config / contract hash：
- base / adapter / cache manifest hash：
- checkpoint / resume来源：

### 执行命令

```bash
# 只填写实际执行过的命令
```

### 通过标准

- [ ] 测试与进程退出码为0
- [ ] shape、dtype、role、hash、计数与本批契约一致
- [ ] 无OOM、non-finite、NCCL hang或未解释的fallback
- [ ] 必需manifest、日志、checkpoint或视频完整

### 真实结果

- 状态：待执行 / 失败 / 通过
- 关键指标：
- 产物路径：
- 失败现场：
- 结论与下一门禁：
````
