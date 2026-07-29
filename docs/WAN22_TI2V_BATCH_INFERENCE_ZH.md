# Wan2.2-TI2V-5B 原始双向批量推理

本文用于在单机 H100 上，对 CSV 中的首帧和 prompt 执行原始
Wan2.2-TI2V-5B 全序列双向推理。入口是：

```text
scripts/infer_wan22_ti2v_batch.py
```

它不会加载 LongLive 因果模型权重。batch 内会共享 DiT 前向，并自动按
`(height, width)` 分桶，横屏和竖屏不会混入同一个 tensor batch。

## 1. 准备环境和权重

建议使用项目文档中的 Python 3.10、PyTorch 2.8/CUDA 12.8 环境：

```bash
conda create -n wan22-ti2v python=3.10 -y
conda activate wan22-ti2v
pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

内网机器不能访问公网时，应提前把 Conda 环境或 wheelhouse，以及完整的
`Wan2.2-TI2V-5B` 权重目录复制进去。权重目录至少应包含：

```text
Wan2.2-TI2V-5B/
├── config.json
├── diffusion_pytorch_model*.safetensors
├── Wan2.2_VAE.pth
├── models_t5_umt5-xxl-enc-bf16.pth
└── google/umt5-xxl/          # 完整 tokenizer 文件
```

先在仓库根目录做不加载模型的检查：

```bash
python scripts/infer_wan22_ti2v_batch.py \
  --metadata testsets/metadata_6cases_480x832.csv \
  --checkpoint-dir /data/models/Wan2.2-TI2V-5B \
  --batch-size 2 \
  --dry-run
```

## 2. 单卡 H100

先用 batch 1 验证环境和权重，再逐步增大到 2 或 3：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_wan22_ti2v_batch.py \
  --metadata testsets/metadata_6cases_480x832.csv \
  --checkpoint-dir /data/models/Wan2.2-TI2V-5B \
  --output-dir outputs/wan22_ti2v_81f \
  --batch-size 2 \
  --frame-num 81 \
  --sampling-steps 50 \
  --shift 3.0 \
  --guide-scale 5.0 \
  --seed 0
```

若显存不足：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_wan22_ti2v_batch.py \
  --metadata testsets/metadata_6cases_480x832.csv \
  --checkpoint-dir /data/models/Wan2.2-TI2V-5B \
  --output-dir outputs/wan22_ti2v_81f \
  --batch-size 1 \
  --offload-model \
  --t5-cpu
```

## 3. 多卡 H100

每张卡能够独立容纳模型时，优先用数据并行。不同 GPU 会处理不同 CSV
batch，适合当前 6 条测试集：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/infer_wan22_ti2v_batch.py \
  --metadata testsets/metadata_6cases_480x832.csv \
  --checkpoint-dir /data/models/Wan2.2-TI2V-5B \
  --output-dir outputs/wan22_ti2v_81f \
  --parallel-mode data \
  --batch-size 2
```

如果单卡放不下目标 batch，可让多张卡共同执行同一个全序列 batch。该模式
复用仓库已有的 Ulysses Sequence Parallel；GPU 数必须能整除 24 个注意力头，
常用值为 2、3、4、6、8：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/infer_wan22_ti2v_batch.py \
  --metadata testsets/metadata_6cases_480x832.csv \
  --checkpoint-dir /data/models/Wan2.2-TI2V-5B \
  --output-dir outputs/wan22_ti2v_81f \
  --parallel-mode ulysses \
  --batch-size 2
```

## 4. 输出和恢复

每条样本输出一个 MP4 和一个同名 JSON，例如：

```text
outputs/wan22_ti2v_81f/
├── 0000_calico_cat_480x832_no_distortion.mp4
└── 0000_calico_cat_480x832_no_distortion.json
```

JSON 记录 prompt、输入图、seed 和采样参数。默认启用 `--resume`，已有的有效
MP4 会被跳过；需要覆盖重跑时传 `--no-resume`。

当前 CSV 的 `480×832` 和 `832×480` 会保持原始尺寸，81 帧对应约 3.38 秒
的 24 FPS 视频。每条样本使用 `--seed + CSV行号`，所以改变 batch 大小或
数据并行 GPU 数不会改变该样本的初始噪声。

常见问题：

- CUDA OOM：先把 `--batch-size` 降为 1，再启用 `--offload-model --t5-cpu`。
- 找不到 FlashAttention：确认它针对内网机器上的当前 PyTorch/CUDA 版本编译。
- 找不到 tokenizer：检查权重目录中的 `google/umt5-xxl/` 是否完整，避免运行时访问公网。
- MP4 写入失败：确认系统或 `imageio-ffmpeg` 提供的 FFmpeg 支持 `libx264`。
