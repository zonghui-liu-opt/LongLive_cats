# Stage‑1：单机 6×H100 80G 部署与训练

本路径只用于 `configs/train_i2v_ar.yaml` 的 causal I2V Stage‑1 LoRA SFT。
它要求 PyTorch FSDP2，并使用固定的 2D HSDP DeviceMesh：
`[[0,1,2],[3,4,5]]`，维度名为 `("replicate", "shard")`。
legacy FSDP1 训练入口不受影响。

## 1. 环境与硬件前提

- 单机恰好 6 张可见的 NVIDIA H100 80G；不要用 `CUDA_VISIBLE_DEVICES`
  重排成跨机或非连续 rank。
- NVIDIA 驱动、CUDA 和 NCCL 必须与 PyTorch 2.8 wheel 匹配。
- Python 环境必须提供 `torch>=2.8,<2.9`、`torchao==0.13.0`、PEFT、
  safetensors、Accelerate 以及 `requirements.txt` 中的项目依赖。
- 数据、模型和输出目录应位于本机高速盘；训练启动后不得修改 CSV、原视频、
  输入图片、VAE/T5/tokenizer、converted base 或 cache artifact。

安装项目依赖前，先按内网 CUDA 镜像源安装对应的 PyTorch 2.8 wheel，再执行：

```bash
python -m pip install -r requirements.txt
```

检查环境；输出必须为 6 张 H100、每张显存至少 75 GiB，并且 BF16/NCCL 可用：

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("nccl:", torch.cuda.nccl.version())
print("visible GPUs:", torch.cuda.device_count())
print("bf16:", torch.cuda.is_bf16_supported())
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
print("FSDP2 public API: OK")
PY
```

训练进程还会在构建 DeviceMesh 前逐 rank 重做这些检查；任何不一致都会直接停止。

## 2. 设置任务专用路径

只设置本任务使用的环境变量，不要覆盖 `HOME`、`CODEX_HOME` 等系统变量：

```bash
export LONG_LIVE_STAGE1_ARCHITECTURE_ROOT=/data/models/Wan2.2-TI2V-5B
export LONG_LIVE_STAGE1_T5_CHECKPOINT=/data/models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
export LONG_LIVE_STAGE1_TOKENIZER_DIR=/data/models/Wan2.2-TI2V-5B/google/umt5-xxl
export LONG_LIVE_STAGE1_VAE_CHECKPOINT=/data/models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
export LONG_LIVE_STAGE1_BASE_CHECKPOINT=/data/longlive/converted_causal_base.pt
export LONG_LIVE_STAGE1_BASE_MANIFEST=/data/longlive/converted_causal_base.manifest.json
export LONG_LIVE_STAGE1_CACHE_DIR=/data/longlive/stage1_i2v_cache
export LONG_LIVE_STAGE1_DRY_RUN_DIR=/data/longlive/runs/stage1_dry_run
export LONG_LIVE_STAGE1_TRAIN_DIR=/data/longlive/runs/stage1_formal
```

配置中的 `data.metadata_path` 默认是
`training_sets/metadata_600clips_480x832_buckets.csv`。若内网文件不在该位置，复制配置并只改
`data.metadata_path`，并设置
`export LONG_LIVE_STAGE1_CONFIG_PATH=/absolute/path/to/copied_config.yaml`；不要修改 600、
SP3×DP2、accumulation=2、LoRA target 或训练步数。

## 3. 一次性转换 causal base

转换器只读 DiffSynth 源目录，输出 LongLive native BF16 checkpoint 和 manifest：

```bash
python scripts/convert_diffsynth_wan22_to_longlive.py \
  --source-checkpoint /data/models/diffsynth_wan22_ti2v_5b \
  --architecture-root "$LONG_LIVE_STAGE1_ARCHITECTURE_ROOT" \
  --output-path "$LONG_LIVE_STAGE1_BASE_CHECKPOINT" \
  --manifest-path "$LONG_LIVE_STAGE1_BASE_MANIFEST" \
  --num-frame-per-block 8
```

只有 converter 报告 100% key/shape coverage、fresh causal wrapper `strict=True` reload，
且源文件转换前后 hash 不变，才能继续。

## 4. 一次性构建 600 条离线 cache

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=6 \
  scripts/precompute_stage1_i2v_cache.py \
  --config-path configs/train_i2v_ar.yaml \
  --cache-dir "$LONG_LIVE_STAGE1_CACHE_DIR"
```

完成条件：manifest 恰好 600 条、row id 连续、视频无重复、每条均为 24 latent
frames、输入图为 1 latent frame、tensor dtype/shape/hash 全部通过。正式训练启动时 rank0
会再次流式审计源数据、模型依赖和全部 cache，结果再广播给其他 ranks。

## 5. 本地测试与 6 rank 分布式门禁

先运行 CPU/纯函数测试：

```bash
PYTHONPATH="$PWD" python -m pytest -q tests
```

然后运行项目提供的 6-rank FSDP2/DTensor parity 测试入口。以下条件任一失败都不得启动
正式训练：

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  tests/test_stage1_fsdp2.py \
  tests/test_stage1_fsdp2_hsdp_parity.py \
  tests/test_lora_utils.py \
  tests/test_trainable_ema.py \
  tests/test_stage1_lora_checkpoint.py

PYTHONPATH="$PWD" torchrun --standalone --nnodes=1 --nproc_per_node=6 \
  tests/stage1_checkpoint_fsdp2_h100_smoke.py \
  --output /local_nvme/longlive_checkpoint_smoke
```

第二条命令必须在真实 6×H100 上运行，且 `--output` 下不得已有同名 smoke checkpoint；
它覆盖 NCCL、2D HSDP、selective raw/EMA gather、DCP optimizer roundtrip、rank-local
resume state 与 marker 最终发布。本地 CPU/Gloo 测试通过不能替代这条门禁。

- LoRA 必须恰好 180 个 target Linear、360 个 adapter tensor、57,016,320 个参数；
- frozen base 必须为 BF16，LoRA master/AdamW moments/EMA 必须为 FP32；
- selective adapter gather 只能读取 trainable DTensor local shard，不得 materialize 冻结 5B；
- 2D mesh、SP3×DP2 梯度与 non-SP reference 更新必须在测试容差内一致；
- canonical raw/EMA adapter 必须能 strict roundtrip。

## 6. 隔离 dry-run 后启动正式训练

确保 dry-run 与正式目录不同，且正式首次启动的目录为空。冷启动时，启动器会创建两个完全
独立的 `torchrun` 进程：第一个只执行 2 个 micro-step + 1 次 optimizer update，不保存
resumable checkpoint；成功后第二个进程才从 step 0 正式训练。若正式目录已含 resumable
marker，启动器会跳过旧 dry-run 目录并只启动正式恢复进程；checkpoint 的完整性仍由 Trainer
再次严格验证。

```bash
bash train_stage1_i2v.sh
```

dry-run 必须确认：同一 SP group 的 sample id 相同、两个 DP sample 不同但 orientation
相同、loss/grad finite、一次 update 的 logical DiT/supervised tokens 分别为
74,880/35,880、无 OOM/NCCL hang，并完成隔离的 EMA swap/gather/restore smoke。

正式训练共 5 epochs / 750 次成功 optimizer updates；每 75 steps 保存一次 adapter-only
checkpoint。JSONL 是唯一持久 metric source：
`$LONG_LIVE_STAGE1_TRAIN_DIR/metrics/train_metrics.jsonl`。

手动画图：

```bash
python scripts/plot_stage1_training.py \
  --jsonl "$LONG_LIVE_STAGE1_TRAIN_DIR/metrics/train_metrics.jsonl" \
  --output-dir "$LONG_LIVE_STAGE1_TRAIN_DIR/metrics/plots" \
  --rolling-window 20 \
  --formats png svg
```

## 7. 恢复、产物与最终 merge

同一命令重启时只会自动选择同时具有 `_SUCCESS`、`_RESUMABLE_SUCCESS` 且 manifest/
base/topology/hash 完整的最新 checkpoint。恢复后下一次 `update_index` 等于目录中的 completed
step；例如 `checkpoint_model_000300` 从 `update_index=300` 开始。不要手工删除或改写
checkpoint 内文件。

用户完成外部视觉验证并选定某个 checkpoint 后，再运行 merge；训练器不会自动选择 best：

```bash
python scripts/merge_lora_generator.py \
  --base-checkpoint "$LONG_LIVE_STAGE1_BASE_CHECKPOINT" \
  --training-checkpoint "$LONG_LIVE_STAGE1_TRAIN_DIR/checkpoint_model_XXXXXX" \
  --output-path /data/longlive/stage1_causal_ema_merged.pt
```

merge 只有在 base/checkpoint hash、360 个 EMA LoRA tensor、safe merge、BF16 输出以及 fresh
causal loader `strict=True` reload 全部通过后才算完成。视觉质量与 checkpoint 选择仍由用户负责。
