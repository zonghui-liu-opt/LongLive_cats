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
| `H100-001` | 2026‑08‑08 | `639ba658a3a7f40bb792fbca4100877d2f7554df` | Batch 1 配置契约 | 待确认：回归删除1项 |

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
  2>&1 | tee "$STAGE2_RUN_DIR/related_regression_tests.log"
```

通过标准：`64 passed`。

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
相关回归：63 passed；用户删除了1个测试
配置契约：全部通过
产物目录：未提供
结论：配置契约通过；整体门禁等待确认被删除测试的名称与原因
失败摘要：无测试失败；存在1项未执行的原基线测试
```

在确认被删除测试不属于关键回归后，才能把总览状态改为“通过”并开始 Batch 2。若该测试因真实失败而被删除，应恢复测试、保留日志并修复原因，不能用删除测试替代通过。

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
