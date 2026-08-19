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
