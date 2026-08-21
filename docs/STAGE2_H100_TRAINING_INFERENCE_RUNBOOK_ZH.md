# Stage-2：4/8×H100 预检、正式训练、绘图与 EMA 推理

本文是 Stage-2 的完整内网执行手册。短文档
`STAGE2_H100_QUICK_DEPLOY_ZH.md` 只负责训练前 6 项检查；本文从同一组已审计资产继续执行
C0/C1/C2、正式训练、断点恢复、训练曲线和 baseline 推理。

正常执行不需要逐段复制本文。直接运行 `bash run_stage2_h100.sh help`，再按脚本显示的
`prepare → smoke → train → control → plot → infer` 六步操作；本文只保留完整原理和故障排查命令。
入口默认使用仓库内`configs/train_i2v_stage2_600cats_micro1_acc8.yaml`，即
`8卡×micro1×acc8`或`4卡×micro1×acc16`，两者均为`global batch 64`。该文件是严格的
`h100_micro1_acc8_longrun`合同：Phase A=360 epoch、Phase B=40 epoch、Generator/Fake-score
LR=`1e-5/2e-6`，派生为A=`3600G/18000F`、B=`400G/2000F`、总计=`4000G/20000F`。
启动前显式设置`STAGE2_CONFIG`仍可覆盖默认值，wrapper会从实际resolved config动态派生终点。

这是一个全新research contract，不能从旧micro1 A24/B4的G240/G280做exact resume。必须使用本文新的
`STAGE2_WORK_ROOT`和`STAGE2_TRAIN_ROOT`，从Stage-1 step3075冷启动；一旦开始，不要在同一lineage
中途再改epochs、LR、milestones或路径。中断恢复只允许继续使用完全相同的config/hash和输出目录。
4卡与8卡属于不同topology合同，不能跨拓扑exact resume。

所有命令均在同一个 shell 中执行。该流程不读取版本控制元数据，也不要求代码目录处于提交态。
建议仍把工作产物放在独立目录，便于容量管理、归档和故障恢复。

## 1. 固定代码目录、解释器和输出目录

```bash
set -euo pipefail

export STAGE2_PROJECT_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0
export STAGE2_PYTHON=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python
export STAGE2_TORCHRUN="$(dirname "$STAGE2_PYTHON")/torchrun"
export STAGE2_GPUS="${STAGE2_GPUS:-8}"  # 4卡时先执行：export STAGE2_GPUS=4
if [[ "$STAGE2_GPUS" == 4 ]]; then STAGE2_SUFFIX=_4gpus; CUDA_VISIBLE_DEVICES=0,1,2,3; else STAGE2_SUFFIX=; CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7; fi
export CUDA_VISIBLE_DEVICES
export LONG_LIVE_STAGE2_GRADIENT_ACCUMULATION_STEPS="$((64 / STAGE2_GPUS))"
export LONG_LIVE_STAGE2_PREFLIGHT_MICRO2_ACCUMULATION_STEPS="$((32 / STAGE2_GPUS))"
export STAGE2_WORK_ROOT="/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_stage2_h100_micro1_acc8_longrun${STAGE2_SUFFIX}"
export STAGE2_TRAIN_ROOT="/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_training_h100_micro1_acc8_longrun${STAGE2_SUFFIX}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1
export PYTHONPYCACHEPREFIX="$STAGE2_WORK_ROOT/pycache"
export OMP_NUM_THREADS=1
cd "$STAGE2_PROJECT_ROOT"
[[ -x "$STAGE2_PYTHON" ]]
[[ -x "$STAGE2_TORCHRUN" ]]
mkdir -p "$STAGE2_WORK_ROOT/logs" "$STAGE2_TRAIN_ROOT"
```

若代码通过手工拷贝部署，且出现 `Stage-2 DMD runtime API mismatch`、
`timing closure error exceeds`、任意 `parameter names mismatch` 或
`No module named 'transformers.integrations.tensor_parallel'`，先只同步单文件
`scripts/apply_stage2_innernet_hotfix.py`，再执行：

```bash
"$STAGE2_PYTHON" scripts/apply_stage2_innernet_hotfix.py --project-root "$STAGE2_PROJECT_ROOT"
```

该脚本不读取 Git：它累计修复 model callback API、trainer/metrics timing orchestration，以及
EMA、optimizer、FSDP2、LoRA gather/load、checkpoint/resume 的统一参数命名契约，识别旧版、当前版及
混合版；全部源码先通过 compile/AST，全部备份完成后才事务替换，失败自动回滚，并用隔离
Python 重新执行 model/timing、parameter-name 和 distributed-safe LoRA load runtime API audit。必须同时看到
`STAGE2_DMD_RUNTIME_API=PASS`、`STAGE2_PARAMETER_NAMES_API=PASS`、
`STAGE2_LORA_LOAD_API=PASS` 和
`STAGE2_INNERNET_HOTFIX=PATCHED`（重复执行则为
`ALREADY_APPLIED`）；若报告 `FAIL`，目标文件不会被猜测性改写，应保留错误和备份路径排查。

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
export STAGE2_CONFIG="$STAGE2_PROJECT_ROOT/configs/train_i2v_stage2_600cats_micro1_acc8.yaml"

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

第 6 项是对应当前4/8×H100 topology的真实FSDP2 release gate；以前只通过旧版 prepare 不能替代它。

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

## 3. 默认8卡micro1×acc8 / 4卡micro1×acc16的 C0/C1/C2

选择一个从未用于正式训练的临时目录。C0 冷启动并保存，C1 只从 C0 恢复、跑纯 DMD
并保存，C2 只从 C1 恢复、强制 DFD 且不保存。

```bash
export STAGE2_SMOKE_DIR="$STAGE2_TRAIN_ROOT/smoke_longrun"
mkdir -p "$STAGE2_SMOKE_DIR"

"$STAGE2_TORCHRUN" --standalone --nnodes=1 --nproc-per-node="$STAGE2_GPUS" --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B train.py \
  --config_path "$ACTIVE_CONFIG" --logdir "$STAGE2_SMOKE_DIR" \
  --stage2-smoke C0 --no-visualize \
  2>&1 | tee "$STAGE2_SMOKE_DIR/C0.log"

"$STAGE2_TORCHRUN" --standalone --nnodes=1 --nproc-per-node="$STAGE2_GPUS" --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B train.py \
  --config_path "$ACTIVE_CONFIG" --logdir "$STAGE2_SMOKE_DIR" \
  --stage2-smoke C1 --no-visualize \
  2>&1 | tee "$STAGE2_SMOKE_DIR/C1.log"

"$STAGE2_TORCHRUN" --standalone --nnodes=1 --nproc-per-node="$STAGE2_GPUS" --max-restarts=0 \
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

### 3.1 为什么默认不再使用micro2×acc4，以及OOM时如何读峰值

Stage-2 每个逻辑子步开始前会重置CUDA峰值，只有该子步成功返回后才把
`max_memory_allocated`、`max_memory_reserved`和最小free写进JSONL。CUDA OOM在rollout内部直接
抛出时，失败子步的峰值来不及写入；最后一条22.9 GiB之类的记录通常只代表此前成功的
fake-score/no-grad子步。`nvidia-smi`同样只是采样瞬间，可能看不到Generator autograd和FSDP2
all-gather叠加形成的短时峰值。

典型真实失败如下：单卡总79.19 GiB，当前rank进程已占78.91 GiB，其中PyTorch live allocated
69.91 GiB、reserved-but-unallocated 7.11 GiB；下一层FSDP2 unshard还要连续申请318 MiB，但
全卡只剩273 MiB。这里同时存在两个问题：69.91 GiB真实工作集已经过大，7.11 GiB缓存池碎片
又使连续buffer更难取得。即使allocator调优让这次申请侥幸成功，69.91 GiB仍超过85%门禁约
67.31 GiB，约77.02 GiB reserved也超过90%门禁约71.27 GiB，micro2不能进入正式训练。

FSDP2不会把4/8张卡拼成一块共享显存。三套5B角色的参数会分片，但每个rank仍需独立容纳
rollout activation、self/cross-KV和临时通信buffer；每个transformer block forward前还会在本卡
all-gather完整block。Generator因cache反向正确性明确关闭activation checkpoint，micro2会近似
翻倍主导激活。若OOM报告中“当前进程占用”已经接近整卡总量，则外部进程不是主因；若两者
差距很大，再用以下命令核对同卡其他PID：

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name --format=csv,noheader
```

## 4. 旧shell迁移或显式覆盖时锁定micro1×global64

当前`run_stage2_h100.sh`已经默认使用仓库内micro1×acc8配置，不再需要从canonical配置现场复制。
如果当前shell曾显式设置旧`STAGE2_CONFIG`，请在`prepare`前重新绑定，并为该launch hash使用
全新的smoke目录：

```bash
export MICRO1_CONFIG="$STAGE2_PROJECT_ROOT/configs/train_i2v_stage2_600cats_micro1_acc8.yaml"
export STAGE2_CONFIG="$MICRO1_CONFIG"
export ACTIVE_CONFIG="$MICRO1_CONFIG"
export STAGE2_SMOKE_DIR="$STAGE2_TRAIN_ROOT/smoke_longrun"

# 必须在新的torchrun进程启动前设置，只缓解碎片，不能替代micro1。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 绑定micro1 launch hash；prepare会安全复用已认证的昂贵资产。
bash run_stage2_h100.sh prepare

# 使用全新目录从C0开始，依次完成C0/C1/C2及显存门禁。
bash run_stage2_h100.sh smoke
```

必须看到`STAGE2_GUIDE_PREPARE=PASS`和`STAGE2_GUIDE_SMOKE=PASS`。micro1把相对micro2的单次
主导激活近似减半，而acc8保持`1×8×8=64`的global batch；不要复用已经OOM的micro2目录或
任何partial smoke lineage。后续正式训练也必须在同一shell中继续使用这个
`STAGE2_CONFIG`/`ACTIVE_CONFIG`。

### 4.1 micro1仍OOM时的唯一二级显存候选

先完整保留micro1日志。不要降低global batch、不要改C/W/K/CFG，也不要开启不安全的Generator
activation checkpoint。只允许在micro1基础上单独profile Generator grad-exit saved-tensor CPU
offload；它会增加host pinned memory和PCIe开销：

```bash
export MICRO1_OFFLOAD_CONFIG="$STAGE2_WORK_ROOT/configs/train_i2v_stage2_600cats_micro1_acc8_offload.yaml"
"$STAGE2_PYTHON" -B - "$MICRO1_CONFIG" "$MICRO1_OFFLOAD_CONFIG" <<'PY'
from pathlib import Path
import sys
from omegaconf import OmegaConf

source, destination = map(Path, sys.argv[1:])
config = OmegaConf.load(source)
config.infra.saved_tensor_cpu_offload = True
OmegaConf.save(config, destination)
PY

export STAGE2_CONFIG="$MICRO1_OFFLOAD_CONFIG"
export ACTIVE_CONFIG="$MICRO1_OFFLOAD_CONFIG"
export STAGE2_SMOKE_DIR="$STAGE2_TRAIN_ROOT/smoke_micro1_acc8_offload_v1"
bash run_stage2_h100.sh prepare
bash run_stage2_h100.sh smoke
```

该候选也必须完整通过C0/C1/C2和原显存余量门禁才能用于正式训练；不得只以“不再抛OOM”作为
放行依据。

## 5. 正式冷启动与自动断点恢复

正式目录必须与 smoke 目录不同且首次启动时为空。正式训练绝不能带 `--stage2-smoke`，也不能
恢复 C0/C1 产物。下面命令从 Stage-1 step 3075 初始化，完成 G=4000、F=20000；选择通过门禁的
`ACTIVE_CONFIG`。

```bash
export STAGE2_FORMAL_DIR="$STAGE2_TRAIN_ROOT/formal_b1_longrun"
mkdir -p "$STAGE2_FORMAL_DIR"

"$STAGE2_TORCHRUN" --standalone --nnodes=1 --nproc-per-node="$STAGE2_GPUS" --max-restarts=0 \
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

长程profile每40G做一次常规checkpoint，并额外发布配置中的显式milestone；retention永久保留这些
milestone和最近两个可恢复点。因此G3600共同父点与G4000终点不会被Phase B清理。不要在训练运行时
手工移动formal目录内的checkpoint；若要做只读快照，先确认`_SUCCESS`再复制到独立root。

训练完整结束后必须存在：

```bash
test -f "$STAGE2_FORMAL_DIR/checkpoint_stage2_g004000/_SUCCESS"
test -f "$STAGE2_FORMAL_DIR/metrics/stage2_train_metrics.jsonl"
```

## 6. 从同一 A360/G3600 分叉 B0 matched control

正式任务是 B1（Phase B 使用 DMD/DFD）。它完成后永久保留的 G3600 是 A360 共同父点；B0
必须从这个 checkpoint 继续跑 40 个纯 DMD epoch，不能拿 A360 终点直接与 B1 的 A360+B40 比较。
复制本轮实际通过门禁的 `ACTIVE_CONFIG`，只做以下四项字段变换：

```bash
export STAGE2_A360_ANCHOR="$STAGE2_FORMAL_DIR/checkpoint_stage2_g003600"
export STAGE2_B0_CONFIG="$STAGE2_WORK_ROOT/configs/train_i2v_stage2_600cats_b0.yaml"
export STAGE2_B0_DIR="$STAGE2_TRAIN_ROOT/formal_matched_b0"
test -f "$STAGE2_A360_ANCHOR/_SUCCESS"
mkdir -p "$(dirname "$STAGE2_B0_CONFIG")" "$STAGE2_B0_DIR"

"$STAGE2_PYTHON" -B - \
  "$ACTIVE_CONFIG" "$STAGE2_B0_CONFIG" "$STAGE2_A360_ANCHOR" <<'PY'
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

"$STAGE2_TORCHRUN" --standalone --nnodes=1 --nproc-per-node="$STAGE2_GPUS" --max-restarts=0 \
  --no-python "$STAGE2_PYTHON" -I -B train.py \
  --config_path "$STAGE2_B0_CONFIG" --logdir "$STAGE2_B0_DIR" \
  --no-visualize \
  2>&1 | tee "$STAGE2_B0_DIR/formal.log"

test -f "$STAGE2_B0_DIR/checkpoint_stage2_g004000/_SUCCESS"
```

首次启动时，trainer 会从 checkpoint 内已认证的 `metrics_lineage.jsonl` 导入 A360 前缀，因此
B0 目录必须是新的且不能预放一份别的 metrics 文件。中断后重新执行同一条 B0 torchrun 命令：
显式 G3600 仍作为 immutable ancestry anchor，但 auto-resume 会选择 B0 目录内最新完整 child，并逐边
校验 canonical parent path 与 manifest SHA‑256；不会反复从 G3600 开始，也不会接受另一条本地
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
`orchestration_seconds_max`单列梯度/参数finite审计、optimizer state审计、分布式状态一致性和
控制流开销；这些真实耗时不再误算为closure error，原5%门禁没有放宽。

## 8. B1 多权重 Generator-EMA baseline 批量推理

推理不读取版本控制状态。runner 会校验各 rank 的 Stage-2 源码 SHA-256 一致，并在结束前再次
确认源码快照未变化；每个 checkpoint 的每个 rank 只加载一次 T5、Generator EMA 和 VAE，使用 seeds 1–4；同一 `(样本, seed)` 的 A/B noise 来自一条连续
48-frame RNG 流：A 取前 24、B 取后 24，不在 B 前重置 seed。

```bash
cd "$STAGE2_PROJECT_ROOT"

# 推理formal_b1_longrun下所有带_SUCCESS且G>=40的权重：
bash run_stage2_h100.sh infer all

# 或者只推理指定step；可任意顺序输入，脚本会数字排序并去重：
bash run_stage2_h100.sh infer 40 80 120 160 200 240 400 800 1200 1600 2400 3200 3600 4000

# 若其他完整long-run权重已保全到独立快照根，可单独扫描该根：
STAGE2_INFERENCE_CHECKPOINT_ROOT="$STAGE2_TRAIN_ROOT/early_checkpoint_snapshots" \
  bash run_stage2_h100.sh infer all

# 无参数时推理当前resolved config的最终权重（默认G4000）：
bash run_stage2_h100.sh infer
```

`infer all`只扫描`$STAGE2_FORMAL_DIR/checkpoint_stage2_gXXXXXX/_SUCCESS`，并对每个
checkpoint的contract、phase arm、step和`generator_ema.safetensors`做严格验证。批量结果与日志分别位于：

```text
$STAGE2_INFERENCE_ROOT/inference_gXXXXXX_baseline/
$STAGE2_INFERENCE_ROOT/logs/inference_gXXXXXX_baseline.log
```

每个权重都必须得到56个MP4和56个JSON trace：单动作`6×4=24`个96帧视频，
双动作`8×4=32`个192帧视频；最后才发布`manifest.json`和`index.html`。只有看到
`STAGE2_GUIDE_INFER=PASS checkpoints=N samples_per_checkpoint=56 total_samples=56*N`才表示整批完成。

重复执行同一命令会严格复验已完成的video+trace pair并跳过，然后继续剩余样本或权重。
`SIGKILL`、断电或旧版root identity误报后不要删除输出目录；直接重跑。若留下已通过ffprobe的
video-only单边产物，runner会先保留到`.stage2-incomplete/`，再确定性重生video+trace。坏视频、
symlink、trace-only或已有最终manifest的损坏批次仍会fail closed，此时按报错保留现场，不要手工拼接产物。

技术 trace/manifest 通过只证明帧数、seed/noise、profile、scheduler/cache、checkpoint、配置和文件
hash 正确，不代表视觉质量自动合格。启动时rank0会依据checkpoint绑定的source manifest只做一次
T5、tokenizer全树、VAE、Generator base与architecture config内容认证；全rank在实际loader前后
复核文件身份，四类resolved/runtime contract/launch hash会进入每份trace和最终manifest。同路径
替换模型资产或output root会在写新样本前fail closed。最后打开 `index.html`，由人工逐项检查猫身份、动作强度、
A→B 连续性、首尾稳定性和伪影；C4/K2/S4/S8 等 profile 仍只是 inference-only 压力测试，不能
写成已经完成部署适配训练。
