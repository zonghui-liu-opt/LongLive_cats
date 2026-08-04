# H100 内网 BF16 TI2V 快速推理

本指南使用仓库内已准备好的猫咪首帧和动作提示词，生成“向前跳跃 → 玩逗猫棒 → 玩毛线球”的约 10.54 秒、24 FPS 视频。

> 注意：公开的 `model_bf16.pt` 是 AR generator 与 4-step DMD LoRA 的 BF16 合并权重，但其发布训练没有使用 I2V 数据。它可以运行当前 I2V 条件链路，首帧一致性与动作质量属于实验性结果，并非官方 I2V checkpoint 的质量保证。

## 1. 准备代码与模型

在服务器上进入本仓库根目录。若服务器可以访问 GitHub，可直接获取本分支：

```bash
git clone --single-branch --branch LongLive2.0-0804 \
  https://github.com/zonghui-liu-opt/LongLive_cats.git
cd LongLive_cats
```

LongLive 合并权重已经配置为以下固定路径：

```text
/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/LongLive-2.0-5B/model_bf16.pt
```

另外必须把完整的 `Wan-AI/Wan2.2-TI2V-5B` 基础组件放在仓库相对路径：

```text
wan_models/Wan2.2-TI2V-5B/
```

该目录至少需要包含 DiT safetensors 分片及索引、`config.json`、`Wan2.2_VAE.pth`、`models_t5_umt5-xxl-enc-bf16.pth` 和 `google/umt5-xxl/` tokenizer。内网环境建议提前传入整个 Hugging Face 仓库目录。

快速检查：

```bash
test -f /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/LongLive-2.0-5B/model_bf16.pt
test -f wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
test -f wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth
test -f wan_models/Wan2.2-TI2V-5B/diffusion_pytorch_model.safetensors.index.json
test -f wan_models/Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer.json
```

可选：验证 LongLive 权重完整性，预期 SHA256 为 `ec9063a44ea3c91e8ff55edcdd58dba3f1bcf6ac9091249629cb57fcebe35fd8`。

```bash
sha256sum /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/LongLive-2.0-5B/model_bf16.pt
```

## 2. 创建运行环境

推荐环境为 Python 3.10、PyTorch 2.8.0 + CUDA 12.8。内网无公网时，将下列安装源替换成内网 PyPI/制品镜像或预下载 wheel；注意 `requirements.txt` 还包含一个 GitHub CLIP 依赖，也需要提前缓存。

```bash
conda create -n longlive2 python=3.10 -y
conda activate longlive2

pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```

确认 H100 与 BF16：

```bash
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())'
```

最后一个值应为 `True`。

## 3. 运行猫咪 TI2V 推理

测试数据已在 `testsets/processed_bf16_ti2v/` 中，无需再次预处理。配置已指向上述 LongLive 权重，并且没有设置 `lora_ckpt` 或 `adapter`，避免对合并权重重复加载 LoRA。该 fixture 只有一个样本，请使用下面的单卡命令。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
python inference.py \
  --config_path configs/inference_i2v_cat_sequence_bf16.yaml
```

输出位于：

```text
videos/cat_jump_wand_yarn_bf16/rank0-0-0_regular.mp4
videos/cat_jump_wand_yarn_bf16/rank0-0-0_regular_prompts.txt
```

配置生成 64 个 latent frames，解码为 253 帧，即约 10.54 秒。首次运行需要加载约 10 GB 的 LongLive generator，以及 Wan 的 T5、VAE 和 DiT 基础组件，启动时间较长属于正常现象。

## 4. 常见问题

- 报 `No such file or directory: wan_models/...`：基础 Wan2.2 组件没有放到仓库根目录下的固定相对路径。
- 报 checkpoint key 不匹配：确认使用的是合并版 `model_bf16.pt`，且配置中没有 `lora_ckpt` 和 `adapter`。
- 显存不足：先关闭同卡其他进程；仍不足时可将配置中的 `save_latents_only` 设为 `true`，先确认去噪链路，再单独解码 latent。
- 首帧约束或动作质量不理想：这是公开权重没有使用 I2V 数据训练的已知限制；正式质量验证需要 I2V AR → I2V DMD/LoRA 专训权重。
