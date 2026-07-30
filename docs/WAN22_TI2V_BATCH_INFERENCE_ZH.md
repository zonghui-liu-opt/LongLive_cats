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

内网机器不能访问公网时，应提前把 Conda 环境或 wheelhouse，以及模型权重
复制进去。本入口同时支持两种 DiT 权重布局：

- 含 `config.json` 的 Diffusers 目录；
- `merge_ti2v5b_lora.py` 生成的 DiffSynth 扁平合并目录。此布局不生成
  `config.json`，代码会使用仓库内置的原始 TI2V-5B 架构严格加载权重。

推荐合并时设置 `AUX_FILES_MODE=copy`，得到可独立搬运的目录：

```bash
MODEL_ROOT=/data/models/Wan2.2-TI2V-5B \
LORA_PATH=/data/lora/epoch-26.safetensors \
MERGED_MODEL_ROOT=/data/models/merged_bi-direct_Wan2.2-5B-cats \
AUX_FILES_MODE=copy \
bash merge_ti2v5b_lora.sh
```

完整合并目录至少包含：

```text
merged_bi-direct_Wan2.2-5B-cats/
├── diffusion_pytorch_model*.safetensors
├── diffusion_pytorch_model.safetensors.index.json  # 多 shard 时
├── Wan2.2_VAE.pth
├── models_t5_umt5-xxl-enc-bf16.pth
├── google/umt5-xxl/          # 完整 tokenizer 文件
└── merge_manifest.json       # 推荐保留，非加载必需
```

若合并时使用 `AUX_FILES_MODE=none`，只搬运合并后的 DiT，并在推理时用
`--auxiliary-dir` 指向原始 Wan2.2-TI2V-5B 根目录。使用默认的 `symlink`
时要确认软链接在内网目标机仍有效；直接复制目录时更推荐 `copy`。

先在仓库根目录做不加载模型和 CUDA 的完整路径检查：

```bash
python scripts/infer_wan22_ti2v_batch.py \
  --metadata testsets/metadata_6cases_480x832.csv \
  --checkpoint-dir /data/models/merged_bi-direct_Wan2.2-5B-cats \
  --batch-size 2 \
  --dry-run
```

## 2. 单卡 H100

### 关键采样参数

如需复现 DiffSynth `WanVideoPipeline` 的一阶 Flow-Matching Euler 推理，指定：

```bash
--solver euler \
--sampling-steps 50 \
--frame-num 97 \
--t-shift 5.0 \
--negative-prompt "模糊，低质量，畸形，多余肢体"
```

其中 `--t-shift`、`--t_shift` 和 `--shift` 完全等价，对应 DiffSynth 的
`sigma_shift`。`97` 满足 Wan2.2 VAE 的 `4n+1` 帧数约束。若省略
`--negative-prompt`，使用 Wan 内置默认负面词；若要与原 DiffSynth 脚本的空
negative prompt 完全一致，请显式传 `--negative-prompt ''`。

先用 batch 1 验证环境和权重，再逐步增大到 2 或 3：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/infer_wan22_ti2v_batch.py \
  --metadata testsets/metadata_6cases_480x832.csv \
  --checkpoint-dir /data/models/merged_bi-direct_Wan2.2-5B-cats \
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
  --checkpoint-dir /data/models/merged_bi-direct_Wan2.2-5B-cats \
  --output-dir outputs/wan22_ti2v_81f \
  --batch-size 1 \
  --offload-model \
  --t5-cpu
```

若合并目录只有 DiT 权重，命令增加：

```bash
  --auxiliary-dir /data/models/Wan2.2-TI2V-5B
```

## 3. 多卡 H100

每张卡能够独立容纳模型时，优先用数据并行。不同 GPU 会处理不同 CSV
batch，适合当前 6 条测试集：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  scripts/infer_wan22_ti2v_batch.py \
  --metadata testsets/metadata_6cases_480x832.csv \
  --checkpoint-dir /data/models/merged_bi-direct_Wan2.2-5B-cats \
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
  --checkpoint-dir /data/models/merged_bi-direct_Wan2.2-5B-cats \
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

- 报缺少 `config.json`：更新到本实现后，DiffSynth 合并目录不需要该文件；
  不要手工伪造配置。先用上面的 `--dry-run` 检查实际路径。
- T5/VAE/tokenizer 缺失：若合并目录只含 DiT，传入原始模型的
  `--auxiliary-dir`；若是失效软链接，重新以 `AUX_FILES_MODE=copy` 合并或复制。
- CUDA OOM：先把 `--batch-size` 降为 1，再启用 `--offload-model --t5-cpu`。
- 找不到 FlashAttention：确认它针对内网机器上的当前 PyTorch/CUDA 版本编译。
- 找不到 tokenizer：检查权重目录中的 `google/umt5-xxl/` 是否完整，避免运行时访问公网。
- MP4 写入失败：确认系统或 `imageio-ffmpeg` 提供的 FFmpeg 支持 `libx264`。
