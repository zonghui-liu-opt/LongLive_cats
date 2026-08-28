# Stage‑1 Teacher‑Forcing LoRA Self‑Rollout（H100）

本入口直接加载 Stage‑1 causal base + `adapter_ema.safetensors`，用于评估经过
Phase A teacher forcing 与 Phase B error recycling 的 LoRA 在真实自回归部署状态下的表现。

## 推理语义

默认 profile 是 `C8 / total-W17 / S1 / H8`：

```text
永久首帧 global sink 1 + 上一块模型生成 KV 8 + 当前 noisy chunk 8 = 总窗口 17
```

每个 current chunk 都来自预先采样的连续 noise plan。每个 noisy DiT forward
只读已有 clean KV、不会提交 provisional KV；最后一步用 UniPC exact sigma 与
CFG 后的 raw flow 直接得到 x0，再以 `t=0` clean forward 提交给下一块。

默认生成 24 个 future latent，最终一次性 decode `S1 + future24 = 25 latent`，
输出包含 conditioning frame 的 97 帧视频。默认 K50；K4 使用同一语义，只压缩
denoising 预算。

| 阶段 | global cursor（latent） | local cursor（latent） |
|---|---:|---:|
| sink preload | 1 | 1 |
| chunk 0 clean commit | 9 | 9 |
| chunk 1 clean commit | 17 | 17 |
| chunk 2 clean commit | 25 | 17 |

## 权重要求

Checkpoint 必须是正式 Stage‑1 bundle，至少包含：

```text
_SUCCESS
adapter_ema.safetensors
adapter_raw.safetensors
base_reference.json
checkpoint_manifest.json
resolved_config.yaml
```

入口会在加载 CUDA 模型前核验全部 manifest 文件的 size/SHA256、base lineage、
EMA 文件名、rank32/360 tensors adapter 合同及 expected completed step。

审计到的 A10+B20 run 每 epoch 是 150 updates：Phase A 在 step1500 结束，完整
Phase B 在 step4500 结束。step3750 已完成 A 并进入 B，但仍是 Phase B 的第 15/20
epoch；若运行该消融，必须显式把 expected step 改成3750，trace 会标记
`phase_b_error_recycling_in_progress`。

## 环境变量

```bash
export CUDA_VISIBLE_DEVICES=0
export PYTHON_BIN=/path/to/conda/env/bin/python

export LONG_LIVE_STAGE1_ARCHITECTURE_ROOT=/path/to/Wan2.2-TI2V-5B
export LONG_LIVE_STAGE1_T5_CHECKPOINT=/path/to/models_t5_umt5-xxl-enc-bf16.pth
export LONG_LIVE_STAGE1_TOKENIZER_DIR=/path/to/google/umt5-xxl
export LONG_LIVE_STAGE1_VAE_CHECKPOINT=/path/to/Wan2.2_VAE.pth
export LONG_LIVE_STAGE1_BASE_CHECKPOINT=/path/to/converted_causal_base.pt
export LONG_LIVE_STAGE1_CHECKPOINT_DIR=/path/to/checkpoint_model_004500

# CSV metadata模式；input_image相对路径以CSV所在目录为基准解析。
export LONG_LIVE_STAGE1_ROLLOUT_INPUT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/testset
export LONG_LIVE_STAGE1_ROLLOUT_METADATA=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/testset/metadata_20cases_480x832.csv
export LONG_LIVE_STAGE1_ROLLOUT_OUTPUT=/path/to/stage1_rollout_outputs
```

`metadata_20cases_480x832.csv`可替换成实际文件名。当前示例profile固定输出
480×832横屏，因此CSV应为：

```csv
input_image,prompt,height,width,bucket
images_20cases_832x480/000313.png,第一条提示词,480,832,landscape
images_20cases_832x480/000313.png,同一首帧的第二条提示词,480,832,landscape
```

这里的`images_20cases_832x480/000313.png`会解析成
`/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/testset/images_20cases_832x480/000313.png`。
CSV的一行就是一个独立推理样本，因此同一张首帧可以出现在多行并搭配不同prompt；
这些行会分别生成输出，并由CSV行号和trace区分。
正式YAML固定`save_with_index: true`，上例两行会分别使用
`rank0-0-0_lora.*`和`rank0-1-0_lora.*`作为产物前缀；同图的`image_sha256`
相同，但`row_sha256`和`prompt_sha256`各自独立。每一行的prompt会作为该条rollout
全程使用的global prompt，而不是在同一个episode内切换prompt。
不要在`.png`后增加句点。入口会在加载5B模型前逐行检查必需列、prompt、文件存在、
RGB、EXIF、真实尺寸、bucket和图片hash；CSV中的全部记录按行顺序推理，
`inference.num_samples: 1`表示每条记录生成一个样本，不是只处理CSV第一行。

正式脚本`infer_stage1_teacher_forcing_rollout.sh`要求显式提供metadata CSV，避免漏设变量后
加载5B模型才发现输入模式错误。底层`inference.py`仍保留旧的
`images/<id>.png + prompts/<id>.txt`目录兼容；如需使用旧模式，可取消metadata变量并直接
调用配置：

```bash
unset LONG_LIVE_STAGE1_ROLLOUT_METADATA
export LONG_LIVE_STAGE1_ROLLOUT_INPUT=/path/to/i2v_inputs
"${PYTHON_BIN}" inference.py \
  --config_path configs/infer_i2v_stage1_teacher_forcing_rollout.yaml
```

## 执行

默认 C8/W17/S1/K50/shift5/CFG5：

```bash
./infer_stage1_teacher_forcing_rollout.sh
```

同一权重做 K4：

```bash
STAGE1_ROLLOUT_SAMPLING_STEPS=4 \
./infer_stage1_teacher_forcing_rollout.sh
```

指定 profile，例如 C4、保留12个历史latent，因此总窗口是 `1+12+4=17`：

```bash
STAGE1_ROLLOUT_CHUNK_SIZE=4 \
STAGE1_ROLLOUT_WINDOW_SIZE=17 \
STAGE1_ROLLOUT_SAMPLING_STEPS=4 \
STAGE1_ROLLOUT_TIMESTEP_SHIFT=5.0 \
STAGE1_ROLLOUT_GUIDANCE_SCALE=5.0 \
./infer_stage1_teacher_forcing_rollout.sh
```

允许的 chunk 是2、4、8；`history = total_window - 1 - chunk`必须至少一个完整
chunk并按chunk对齐，local window不得超过24。C8是训练时的block size，C8/W17
只是本入口的默认部署profile；trace会分别记录这两个事实。由于训练阶段没有
generated-history KV rollout，所有这些self-rollout profile（包括默认项）都是
deployment OOD评估。

若确实要评估已知的step3750中间权重：

```bash
export LONG_LIVE_STAGE1_CHECKPOINT_DIR=/path/to/checkpoint_model_003750
STAGE1_ROLLOUT_EXPECTED_STEP=3750 \
STAGE1_ROLLOUT_SAMPLING_STEPS=4 \
./infer_stage1_teacher_forcing_rollout.sh
```

也可以直接调用 `inference.py`；对应参数是
`--stage1_rollout_{chunk_size,window_size,sampling_steps,timestep_shift,guidance_scale,expected_step}`。

## 产物与边界

每个 MP4（或 latent PT）旁会写 `*_stage1_rollout_trace.json`，记录 profile/hash、
固定 negative prompt/hash、base/EMA/manifest hash、训练phase、输入noise identity、
CSV/row/image hash、每个chunk的schedule/sigma bit pattern、cache cursor、commit policy、
输出hash和帧数。

Phase A/B 训练始终是 teacher forcing；Phase B 注入预测残差，但不是KV self-rollout。
因此本入口是在测量真实 train/infer gap，而不是声称权重曾用同一rollout拓扑训练。
global sink 和小于C8的chunk也属于纯部署策略。
