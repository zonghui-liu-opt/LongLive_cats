# Stage-2 训练中快速推理 G60/G70

适用于 Phase A 尚未结束、训练不能停止，但需要查看已保存 checkpoint 效果的场景。

## 结论

- G60/G70 已满足 `G>=40`，可以加载 Generator EMA 推理。
- 推理只读带 `_SUCCESS` 的完整 checkpoint，不会影响后续训练。
- 不要在训练占用的 8 张 H100 上同时推理；应使用另一台节点或其他空闲 H100。
- 不要运行 `bash run_stage2_h100.sh infer`，它只接受最终 G280。

## 1. 立即保全 G60/G70

训练默认只保留最近两个普通 checkpoint。请在训练节点另开终端执行：

```bash
set -Eeuo pipefail

export STAGE2_TRAIN_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_training
export STAGE2_FORMAL_DIR="$STAGE2_TRAIN_ROOT/formal_baseline"
export SNAPSHOT_ROOT="$STAGE2_TRAIN_ROOT/early_checkpoint_snapshots"

mkdir -p "$SNAPSHOT_ROOT"

for step in 000060 000070; do
    src="$STAGE2_FORMAL_DIR/checkpoint_stage2_g${step}"
    dst="$SNAPSHOT_ROOT/checkpoint_stage2_g${step}"
    test -f "$src/_SUCCESS"
    test ! -e "$dst"
    cp -al "$src" "$dst"
done
```

`cp -al` 在同一文件系统创建硬链接快照，几乎不增加空间占用。不要修改 checkpoint 内的文件。

## 2. 在空闲 H100 节点推理

该节点必须能访问与训练一致的代码、checkpoint、Stage-1 Generator base、T5、tokenizer、VAE、source manifest 和测试图片。
先同步远程 `stage-2` 最新代码，再直接运行仓库内的一键脚本：

```bash
cd /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0
git pull --ff-only longlive-cats stage-2

# 默认使用8卡推理G70。
bash infer_stage2_tmp.sh

# 只跑G60。
bash infer_stage2_tmp.sh 60

# 顺序比较G60、G70。
bash infer_stage2_tmp.sh 60 70
```

脚本会立即打印预检结果，模型认证期间每30秒输出心跳；失败会显示日志路径，成功后验收
56个视频、56个trace、`manifest.json`和`index.html`。它只读快照权重，不会停止训练。

启动输出必须包含类似：

```text
STAGE2_FFPROBE=PASS (/usr/bin/ffprobe)
```

脚本会跳过PATH中存在但无法运行的ffprobe并尝试后续系统候选。如需指定特殊安装位置：

```bash
export LONG_LIVE_FFPROBE=/实际可运行的路径/ffprobe
bash infer_stage2_tmp.sh 70
```

最新版还修复了跨节点挂载时 `device/inode/mtime` 不同导致的
`Generator provenance differs from its live manifest` 误报；权重SHA256、大小、manifest和血缘仍严格校验。

如果视频已经生成后出现：

```text
Stage-2 output root identity changed
```

这是旧版把共享存储目录的`device/inode/mode`当作永久身份导致的误报。同步最新`stage-2`后，直接对原命令续跑，
不要删除输出目录；完整的video+trace会重新校验后跳过。新版会在输出根创建
`.stage2-output-root-anchor.json`作为持久身份，不要修改或删除它；真实目录替换和symlink仍会被拒绝。

如果旧错误发生在MP4提交后、trace写入前，新版会先再次严格验收这个单边视频，将原文件保留到
`<输出目录>/.stage2-incomplete/`，再用同一sample/seed重生成完整video+trace。坏视频、symlink、
trace-only或已有最终manifest的损坏结果不会自动处理。隐藏隔离区不计入56个正式视频；整批验收通过后可人工归档。

### 2.1 使用另一台 8×H100 加速

现有入口使用样本级数据并行。8 张卡会把 56 个样本平均分片，每张卡生成 7 个视频：

```text
56 个视频 ÷ 8 张卡 = 每张卡 7 个视频
```

视频总数仍为 56。每张卡都会加载完整 Generator，这不是把一个模型拆到 8 张卡上的模型并行。
实际加速比还会受到模型加载、VAE 解码和共享存储读取速度影响。

如果只想尽快查看最新效果，仅用全部8卡执行：

```bash
bash infer_stage2_tmp.sh 70
```

如果需要比较 G60/G70，推荐让两个 checkpoint 依次各用全部 8 卡。也可以在两个终端中分别使用
GPU 0–3 跑 G60、GPU 4–7 跑 G70；两种方式的理想总耗时接近，但顺序使用 8 卡更简单，且共享
存储的瞬时读取压力更小。不要让两个 8 卡任务同时占用同一组 GPU。

### 2.2 C/W/K 推理消融：一次同步，内网直接切 quick/formal

消融入口复用同一个 Generator、T5、VAE、metadata、断点续跑和产物验收链；不会为每组超参重复加载一套进程。
默认矩阵在 `configs/infer_i2v_stage2_sweep.yaml`：包含冻结的
`baseline_c8w16k4s1`、本轮训练匹配的 `c4w16k4s1` 和 7 个 deployment-only 动态点。动态点固定
`S=1`，因此不会误入训练。

需要增删超参时，只在外网同步代码前编辑一次该 YAML 的 `profile_set.cases`：

```yaml
profile_set:
  named: [baseline_c8w16k4s1]
  grids: []
  cases:
    - {chunk_frames: 6, local_window_frames: 12, num_denoising_steps: 3}
    - {chunk_frames: 8, local_window_frames: 24, num_denoising_steps: 6}
```

约束是：`C>=2` 且整除 24；`W>=C`、`W%C=0`、`W<=24`；`K=1..8`。
也可以在 `grids` 中给三个字段各写一个列表，自动展开笛卡尔积；重复拓扑、非法组合和超过 32 个 profile
都会在加载模型前直接拒绝。动态 profile id 包含完整拓扑、shift=5 timetable 和 sigma FP32 bits 的 SHA-256，
不同实验不会覆盖到同一路径。

代码同步到内网后，先做不启动 CUDA/torchrun 的完整计划预检。该步骤复用真实 metadata/sample
planner，会读取并校验所选行与首帧，不再只按配置长度估算：

```bash
cd /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0

STAGE2_INFERENCE_CONFIG=configs/infer_i2v_stage2_sweep.yaml \
STAGE2_INFERENCE_PLAN_ONLY=1 \
STAGE2_INFERENCE_NPROC=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash infer_stage2_tmp.sh 70
```

确认输出中的 `expected_sample_count`、每组 C/W/K、DiT forward 数和 self-KV GiB 后，去掉
`STAGE2_INFERENCE_PLAN_ONLY=1` 即可跑默认 quick 集合。默认 quick 是 4 个基础案例 × 1 seed × 9 profiles = 36 个视频；
4 张卡能让同一案例跨 profile 落到同一 rank，最大化 prompt/首帧 latent 复用：

```bash
STAGE2_INFERENCE_CONFIG=configs/infer_i2v_stage2_sweep.yaml \
STAGE2_INFERENCE_NPROC=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash infer_stage2_tmp.sh 70
```

quick 通过后，不改 YAML、不改代码，直接把同一 profile 集晋级为 56 个正式案例/seed 组合：

```bash
STAGE2_INFERENCE_CONFIG=configs/infer_i2v_stage2_sweep.yaml \
STAGE2_SWEEP_EVALUATION=formal \
STAGE2_INFERENCE_NPROC=7 \
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
bash infer_stage2_tmp.sh 70
```

脚本会从 resolved plan 动态计算视频/trace 数，不再写死 56。sweep 默认输出目录包含 resolved contract
hash，quick、formal、不同 profile 集和 baseline 不会混写；同一命令中断后原样重跑即可续跑，不要向既有
输出目录追加、删除或重排 profile。每个样本 trace 还绑定当前推理源码闭包，旧代码生成的样本不能被新代码
静默混入同一 manifest。

同一输出目录同时只能有一个任务，脚本会创建相邻的 `*.stage2-run.lock` 并在正常结束、中断或报错时清理。
若机器异常断电留下 stale lock，先确认对应任务和 PID 已不存在，再只删除报错中给出的那个 lock 目录；不要在
任务仍运行时手工删除。

## 3. 为什么每个 checkpoint 生成 56 个视频

默认 baseline 测试矩阵为：

```text
单动作：6 个案例 × 4 个 seed = 24
双动作：8 个案例 × 4 个 seed = 32
合计：24 + 32 = 56
```

因此同时测试 G60 和 G70 会得到 112 个视频。结果分别位于：

```text
$STAGE2_TRAIN_ROOT/inference_early_g000060/index.html
$STAGE2_TRAIN_ROOT/inference_early_g000070/index.html
```

G60/G70 都是 Phase A 早期结果，只用于观察收敛趋势，不能代替 G240/G280 的最终验收。
