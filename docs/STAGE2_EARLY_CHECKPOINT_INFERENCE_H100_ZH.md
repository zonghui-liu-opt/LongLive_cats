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

```bash
set -Eeuo pipefail

export STAGE2_PROJECT_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0
export STAGE2_PYTHON=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python
export STAGE2_TORCHRUN="$(dirname "$STAGE2_PYTHON")/torchrun"
export STAGE2_WORK_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_stage2_new
export STAGE2_TRAIN_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_training
export SNAPSHOT_ROOT="$STAGE2_TRAIN_ROOT/early_checkpoint_snapshots"

export ARCH_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B
export LONG_LIVE_STAGE2_SOURCE_MANIFEST="$STAGE2_WORK_ROOT/stage2_600cats_f25_v1/cache_manifest.attested.json"
export LONG_LIVE_STAGE2_ARCHITECTURE_ROOT="$ARCH_ROOT"
export LONG_LIVE_STAGE2_T5_CHECKPOINT="$ARCH_ROOT/models_t5_umt5-xxl-enc-bf16.pth"
export LONG_LIVE_STAGE2_TOKENIZER_DIR="$ARCH_ROOT/google/umt5-xxl"
export LONG_LIVE_STAGE2_VAE_CHECKPOINT="$ARCH_ROOT/Wan2.2_VAE.pth"

# 一张空闲 H100；多张卡时同时修改这两个变量。
export CUDA_VISIBLE_DEVICES=0
export INFER_NPROC=1

cd "$STAGE2_PROJECT_ROOT"

infer_one() {
    local step="$1"
    export LONG_LIVE_STAGE2_INFERENCE_CHECKPOINT="$SNAPSHOT_ROOT/checkpoint_stage2_g${step}"
    export LONG_LIVE_STAGE2_INFERENCE_OUTPUT="$STAGE2_TRAIN_ROOT/inference_early_g${step}"

    test -f "$LONG_LIVE_STAGE2_INFERENCE_CHECKPOINT/_SUCCESS"
    test ! -e "$LONG_LIVE_STAGE2_INFERENCE_OUTPUT"

    "$STAGE2_TORCHRUN" \
      --standalone --nnodes=1 --nproc-per-node="$INFER_NPROC" --max-restarts=0 \
      --no-python "$STAGE2_PYTHON" -I -B \
      scripts/run_stage2_inference.py \
      --config configs/infer_i2v_stage2_baseline.yaml \
      2>&1 | tee "$STAGE2_TRAIN_ROOT/inference_early_g${step}.log"
}

# 只看最新效果时仅执行 infer_one 000070。
infer_one 000060
infer_one 000070
```

一张卡可以运行；多张空闲卡只负责分摊样本，每张卡仍会加载完整 Generator。例如 4 卡：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export INFER_NPROC=4
```

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
