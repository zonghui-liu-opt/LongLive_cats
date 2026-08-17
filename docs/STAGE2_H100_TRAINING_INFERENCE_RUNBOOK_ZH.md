# Stage-2：8×H100 预检、正式训练、绘图与 EMA 推理

本文是 Stage-2 的完整内网执行手册。短文档
`STAGE2_H100_QUICK_DEPLOY_ZH.md` 只负责训练前 6 项检查；本文从同一组已审计资产继续执行
C0/C1/C2、正式训练、断点恢复、训练曲线和 baseline 推理。

正常执行不需要逐段复制本文。直接运行 `bash run_stage2_h100.sh help`，再按脚本显示的
`prepare → smoke → train → control → plot → infer` 六步操作；本文只保留完整原理和故障排查命令。

所有命令均在同一个 shell 中执行。该流程不读取版本控制元数据，也不要求代码目录处于提交态。
建议仍把工作产物放在独立目录，便于容量管理、归档和故障恢复。

## 1. 固定代码目录、解释器和输出目录

```bash
set -euo pipefail

export STAGE2_PROJECT_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0
export STAGE2_PYTHON=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python
export STAGE2_TORCHRUN="$(dirname "$STAGE2_PYTHON")/torchrun"
export STAGE2_WORK_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_stage2_new
export STAGE2_TRAIN_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_training

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$STAGE2_WORK_ROOT/pycache"
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd "$STAGE2_PROJECT_ROOT"
[[ -x "$STAGE2_PYTHON" ]]
[[ -x "$STAGE2_TORCHRUN" ]]
mkdir -p "$STAGE2_WORK_ROOT/logs" "$STAGE2_TRAIN_ROOT"
```

若旧 prepare 曾生成 `checkpoints/`、`results/`、cache 或临时 YAML，先由操作者确认后归档；
不要用未经检查的批量删除命令。

## 2. 绑定内网正式资产并完成 6 项训练前门禁

以下路径与当前内网环境一致；若实际资产位置不同，只改路径，不改训练契约。动作 sidecar 必须是
`head_tilt_and_wink=198`、`jump=202`、`play_with_a_cat_wand=200`，总计 600 条。

```bash
export ARCH_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B
export TEACHER_CKPT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/ckpts
export TEACHER_PROVENANCE_RECORD="$TEACHER_CKPT/merge_manifest.json"
export STAGE1_BASE=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/checkpoints/stage1/converted_causal_base.pt
export STAGE1_CKPT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/results/stage1_600cats_phaseA10epochs_phaseB20epochs/checkpoint_model_003075
export METADATA_600=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/metadata_600clips_480x832_buckets.csv
export STAGE1_CACHE_MANIFEST=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/cache_480x832_buckets/ar_stage1_i2v_600cats/cache_manifest.json
export ACTION_SIDECAR_600=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/datasets_project/cats/action_labels_600cats.csv
export ATTEST_STAGE2_TEACHER=1
export STAGE2_CONFIG="$STAGE2_PROJECT_ROOT/configs/train_i2v_stage2_600cats.yaml"

bash prepare_stage2.sh 2>&1 | tee "$STAGE2_WORK_ROOT/logs/prepare_release.log"
```

必须同时出现且命令退出码为 0：

```text
CHECK_1_TEACHER_PASS
CHECK_2_GENERATOR_PASS
CHECK_3_CONFIG_PASS
CHECK_4_DATA_PASS
CHECK_5_ROLE_INIT_PASS
CHECK_6_FSDP2_ACCUMULATION_PASS
STAGE2_PRETRAIN_PASS
```

第 6 项是新增的真实 8×H100 FSDP2 release gate；以前只通过旧版 prepare 不能替代它。

prepare 的 `export` 不会反向修改父 shell。训练前在当前 shell 明确重绑同一批产物：

```bash
export LONG_LIVE_STAGE2_ARCHITECTURE_ROOT="$ARCH_ROOT"
export LONG_LIVE_STAGE2_GENERATOR_BASE="$STAGE2_WORK_ROOT/assets/stage1_step3075_ema_merged.pt"
export LONG_LIVE_STAGE2_GENERATOR_MANIFEST="$STAGE2_WORK_ROOT/assets/stage1_step3075_ema_merged.manifest.json"
export LONG_LIVE_STAGE2_REAL_SCORE_BASE="$TEACHER_CKPT"
export LONG_LIVE_STAGE2_REAL_SCORE_MANIFEST="$STAGE2_WORK_ROOT/assets/real_score_teacher.manifest.json"
export LONG_LIVE_STAGE2_METADATA_PATH="$METADATA_600"
export LONG_LIVE_STAGE2_SOURCE_MANIFEST="$STAGE2_WORK_ROOT/stage2_600cats_f25_v1/cache_manifest.attested.json"
export LONG_LIVE_STAGE2_ACTION_LABELS_PATH="$ACTION_SIDECAR_600"
export LONG_LIVE_STAGE2_CACHE_DIR="$STAGE2_WORK_ROOT/stage2_600cats_f25_v1"
export LONG_LIVE_STAGE2_NEGATIVE_MANIFEST="$STAGE2_WORK_ROOT/negative_v1/negative_conditioning_manifest.json"
export ACTIVE_CONFIG="$STAGE2_CONFIG"
```

## 3. micro2×acc4 的 C0/C1/C2

选择一个从未用于正式训练的临时目录。C0 冷启动并保存，C1 只从 C0 恢复、跑纯 DMD
并保存，C2 只从 C1 恢复、强制 DFD 且不保存。

```bash
export STAGE2_SMOKE_DIR="$STAGE2_TRAIN_ROOT/smoke_micro2_acc4"
mkdir -p "$STAGE2_SMOKE_DIR"

"$STAGE2_TORCHRUN" --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B train.py \
  --config_path "$ACTIVE_CONFIG" --logdir "$STAGE2_SMOKE_DIR" \
  --stage2-smoke C0 --no-visualize \
  2>&1 | tee "$STAGE2_SMOKE_DIR/C0.log"

"$STAGE2_TORCHRUN" --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B train.py \
  --config_path "$ACTIVE_CONFIG" --logdir "$STAGE2_SMOKE_DIR" \
  --stage2-smoke C1 --no-visualize \
  2>&1 | tee "$STAGE2_SMOKE_DIR/C1.log"

"$STAGE2_TORCHRUN" --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B train.py \
  --config_path "$ACTIVE_CONFIG" --logdir "$STAGE2_SMOKE_DIR" \
  --stage2-smoke C2 --no-visualize \
  2>&1 | tee "$STAGE2_SMOKE_DIR/C2.log"
```

每个命令都必须退出 0。trainer 会逐 cycle 强制检查：无 nonfinite/角色污染/cache 错误，
`allocated≤85%`、`reserved≤90%`、空闲显存 `≥max(8 GiB,10%)`、
`max/mean≤1.15`，以及相邻 cycle live allocated 增长 `≤1 GiB`。任一检查不通过都不允许
继续正式训练。C0/C1 checkpoint 还会保存一份不消费运行态的“下一 F1”探针；C1/C2 在首个
F1 前按同一生产顺序精确重放并核对 sampler batch、exit、loader RNG、rollout noise、score
timestep/noise 与全部计数器，完全一致后才消费一次。任一本周期发生过 nonfinite（即使随后
精确重放成功）也不得标记 smoke PASS。

## 4. 失败时唯一首选回退：micro1×acc8

只有 micro2×acc4 未通过上述门槛时才执行本节。在仓库外复制配置，只改两个字段；然后用该
配置重新运行 prepare 绑定新的 launch hash，并在全新的 smoke 目录重跑全部 C0/C1/C2。

```bash
export MICRO1_CONFIG="$STAGE2_WORK_ROOT/configs/train_i2v_stage2_600cats_micro1_acc8.yaml"
mkdir -p "$(dirname "$MICRO1_CONFIG")"
"$STAGE2_PYTHON" -B - \
  "$STAGE2_PROJECT_ROOT/configs/train_i2v_stage2_600cats.yaml" "$MICRO1_CONFIG" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf

source, destination = map(Path, sys.argv[1:])
config = OmegaConf.load(source)
config.training.microbatch_size_per_device = 1
config.training.gradient_accumulation_steps = 8
OmegaConf.save(config, destination)
PY

export STAGE2_CONFIG="$MICRO1_CONFIG"
export ACTIVE_CONFIG="$MICRO1_CONFIG"
bash prepare_stage2.sh 2>&1 | tee "$STAGE2_WORK_ROOT/logs/prepare_micro1_acc8.log"
export STAGE2_SMOKE_DIR="$STAGE2_TRAIN_ROOT/smoke_micro1_acc8"
mkdir -p "$STAGE2_SMOKE_DIR"
```

随后把第 3 节三条命令原样各执行一次。micro1 仍失败时停止并保存日志；不要降低 global
batch、不要改 C/W/K/CFG，也不要开启不安全的 Generator activation checkpoint。只有任务书
允许的 Generator grad-exit saved-tensor CPU offload 可作为下一次单独 profile 候选。

## 5. 正式冷启动与自动断点恢复

正式目录必须与 smoke 目录不同且首次启动时为空。正式训练绝不能带 `--stage2-smoke`，也不能
恢复 C0/C1 产物。下面命令从 Stage-1 step 3075 初始化，完成 G=280、F=1400；选择通过门禁的
`ACTIVE_CONFIG`。

```bash
export STAGE2_FORMAL_DIR="$STAGE2_TRAIN_ROOT/formal_baseline"
mkdir -p "$STAGE2_FORMAL_DIR"

"$STAGE2_TORCHRUN" --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B train.py \
  --config_path "$ACTIVE_CONFIG" --logdir "$STAGE2_FORMAL_DIR" \
  --no-visualize \
  2>&1 | tee "$STAGE2_FORMAL_DIR/formal.log"
```

进程被正常中断或机器重启后，重新导出第 1、2 节变量，并在同一 `STAGE2_FORMAL_DIR` 运行
完全相同的命令；默认 auto-resume 只选择最新带 `_SUCCESS` 的完整 checkpoint。不要传
`--no-auto-resume`。如果 checkpoint/数据/config/phase arm 不一致，正确行为是 fail closed，而
不是跳过校验。checkpoint 会把 Stage-2 源码 SHA-256 写入 code provenance 供审计；恢复前可
直接核对该摘要，不需要版本控制元数据。

训练完整结束后必须存在：

```bash
test -f "$STAGE2_FORMAL_DIR/checkpoint_stage2_g000280/_SUCCESS"
test -f "$STAGE2_FORMAL_DIR/metrics/stage2_train_metrics.jsonl"
```

## 6. 从同一 A24/G240 分叉 B0 matched control

baseline 正式任务是 B1（Phase B 使用 DMD/DFD）。它完成后保留的 G240 是 A24 共同父点；B0
必须从这个 checkpoint 继续跑 4 个纯 DMD epoch，不能拿 A24 终点直接与 B1 的 A24+B4 比较。
复制本轮实际通过门禁的 `ACTIVE_CONFIG`，只做以下四项字段变换：

```bash
export STAGE2_A24_ANCHOR="$STAGE2_FORMAL_DIR/checkpoint_stage2_g000240"
export STAGE2_B0_CONFIG="$STAGE2_WORK_ROOT/configs/train_i2v_stage2_600cats_b0.yaml"
export STAGE2_B0_DIR="$STAGE2_TRAIN_ROOT/formal_matched_b0"
test -f "$STAGE2_A24_ANCHOR/_SUCCESS"
mkdir -p "$(dirname "$STAGE2_B0_CONFIG")" "$STAGE2_B0_DIR"

"$STAGE2_PYTHON" -B - \
  "$ACTIVE_CONFIG" "$STAGE2_B0_CONFIG" "$STAGE2_A24_ANCHOR" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf

source, destination, anchor = map(Path, sys.argv[1:])
config = OmegaConf.load(source)
config.checkpoints.init_from_stage1 = None
config.checkpoints.resume_stage2 = str(anchor.resolve(strict=True))
config.training.phase_b_mode = "dmd_only"
config.training.phase_b_dfd_probability_max = 0.0
OmegaConf.save(config, destination)
PY

"$STAGE2_TORCHRUN" --standalone --nnodes=1 --nproc-per-node=8 --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B train.py \
  --config_path "$STAGE2_B0_CONFIG" --logdir "$STAGE2_B0_DIR" \
  --no-visualize \
  2>&1 | tee "$STAGE2_B0_DIR/formal.log"

test -f "$STAGE2_B0_DIR/checkpoint_stage2_g000280/_SUCCESS"
```

首次启动时，trainer 会从 checkpoint 内已认证的 `metrics_lineage.jsonl` 导入 A24 前缀，因此
B0 目录必须是新的且不能预放一份别的 metrics 文件。中断后重新执行同一条 B0 torchrun 命令：
显式 G240 仍作为 immutable ancestry anchor，但 auto-resume 会选择 B0 目录内最新完整 child，并逐边
校验 canonical parent path 与 manifest SHA‑256；不会反复从 G240 开始，也不会接受另一条本地
lineage。

## 7. 从两条正式 JSONL 生成训练图

```bash
"$STAGE2_PYTHON" -B scripts/plot_stage2_training.py \
  --jsonl "$STAGE2_FORMAL_DIR/metrics/stage2_train_metrics.jsonl" \
  --output-dir "$STAGE2_FORMAL_DIR/plots" \
  --formats png svg --require-complete

"$STAGE2_PYTHON" -B scripts/plot_stage2_training.py \
  --jsonl "$STAGE2_B0_DIR/metrics/stage2_train_metrics.jsonl" \
  --output-dir "$STAGE2_B0_DIR/plots" \
  --formats png svg --require-complete
```

`--require-complete` 会重新校验完整 `F1…F5→G` lineage、phase/terminal、配置 hash 与时间闭合；
闭合条件固定为
`abs(error_seconds) <= max(0.1, 0.05 * step_seconds_max)`，不满足时不会生成“看似成功”的图。

## 8. B1 G280 Generator-EMA baseline 推理

推理不读取版本控制状态。runner 会校验各 rank 的 Stage-2 源码 SHA-256 一致，并在结束前再次
确认源码快照未变化；每个 rank 只加载一次 T5、Generator EMA 和 VAE，使用 seeds 1–4；同一 `(样本, seed)` 的 A/B noise 来自一条连续
48-frame RNG 流：A 取前 24、B 取后 24，不在 B 前重置 seed。

```bash
cd "$STAGE2_PROJECT_ROOT"

export LONG_LIVE_STAGE2_INFERENCE_CHECKPOINT="$STAGE2_FORMAL_DIR/checkpoint_stage2_g000280"
export LONG_LIVE_STAGE2_ARCHITECTURE_ROOT="$ARCH_ROOT"
export LONG_LIVE_STAGE2_T5_CHECKPOINT="$ARCH_ROOT/models_t5_umt5-xxl-enc-bf16.pth"
export LONG_LIVE_STAGE2_TOKENIZER_DIR="$ARCH_ROOT/google/umt5-xxl"
export LONG_LIVE_STAGE2_VAE_CHECKPOINT="$ARCH_ROOT/Wan2.2_VAE.pth"
export LONG_LIVE_STAGE2_INFERENCE_OUTPUT="$STAGE2_TRAIN_ROOT/inference_g280_baseline"

bash infer_stage2_baseline.sh \
  2>&1 | tee "$STAGE2_TRAIN_ROOT/inference_g280_baseline.log"
```

baseline 矩阵应得到 56 个 MP4 和 56 个 JSON trace：单动作 `6×4=24` 个 96 帧视频，
双动作 `8×4=32` 个 192 帧视频；最后才发布 `manifest.json` 和 `index.html`。执行最终复验：

```bash
test "$(find "$LONG_LIVE_STAGE2_INFERENCE_OUTPUT/videos" -type f -name '*.mp4' | wc -l)" -eq 56
test "$(find "$LONG_LIVE_STAGE2_INFERENCE_OUTPUT/traces" -type f -name '*.json' | wc -l)" -eq 56
test -f "$LONG_LIVE_STAGE2_INFERENCE_OUTPUT/manifest.json"
test -f "$LONG_LIVE_STAGE2_INFERENCE_OUTPUT/index.html"
"$STAGE2_PYTHON" -B - "$LONG_LIVE_STAGE2_INFERENCE_OUTPUT/manifest.json" <<'PY'
import json
from pathlib import Path
import sys
from utils.stage2_inference_artifacts import validate_stage2_inference_manifest

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
validated = validate_stage2_inference_manifest(manifest)
assert validated["status"] == "complete"
assert validated["expected_sample_count"] == 56
print("STAGE2_BASELINE_INFERENCE_ARTIFACTS=PASS")
PY
```

正常异常会在各 rank 间同步并返回非零；但 `SIGKILL`、断电等硬中断若恰好发生在 MP4 与 trace
分别提交的夹缝，可能留下孤立文件或原子临时文件。runner 会 fail closed，不会把它当作可续跑
pair。此时先由操作者保留现场，归档整个 inference 输出目录，再换一个全新的仓库外
`LONG_LIVE_STAGE2_INFERENCE_OUTPUT` 重跑；不要手工拼接或删除单个产物后宣称同一批次 complete。

技术 trace/manifest 通过只证明帧数、seed/noise、profile、scheduler/cache、checkpoint、配置和文件
hash 正确，不代表视觉质量自动合格。启动时rank0会依据checkpoint绑定的source manifest只做一次
T5、tokenizer全树、VAE、Generator base与architecture config内容认证；全rank在实际loader前后
复核文件身份，四类resolved/runtime contract/launch hash会进入每份trace和最终manifest。同路径
替换模型资产或output root会在写新样本前fail closed。最后打开 `index.html`，由人工逐项检查猫身份、动作强度、
A→B 连续性、首尾稳定性和伪影；C4/K2/S4/S8 等 profile 仍只是 inference-only 压力测试，不能
写成已经完成部署适配训练。
