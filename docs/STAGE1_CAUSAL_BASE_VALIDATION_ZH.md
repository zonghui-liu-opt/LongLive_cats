# Stage‑1 causal base 转换与 testsets 视频验证

本文验证两个彼此独立的结论：

1. DiffSynth 权重已完整、可追溯地转换为 LongLive native BF16 causal base；
2. converted base 能通过真实 causal I2V 路径读取首帧与 prompt，并为仓库内 6 条
   `testsets` 生成技术上有效的 MP4。

第二项是运行链路 smoke，不是 Stage‑1 画质验收。converted base 尚未经过 causal LoRA
适配，因此不能用其动作质量否定一次严格通过的权重转换；最终语义和动作效果应在 Stage‑1
EMA LoRA merge 后，以相同 testsets/seed 再做人工对比。

## 1. 验证内容

### 1.1 转换 artifact 门禁

`scripts/validate_stage1_causal_base.py` 会重新读取 checkpoint、manifest 和可选的 DiffSynth
源目录，并验证：

- checkpoint format/version 与 manifest 一致；
- 输出文件 size/SHA256 与 manifest 一致；
- 当前源 index/shard 列表、逐文件 SHA256 与 aggregate SHA256 未变化；
- generator key 集合、每个 tensor 的 shape/dtype/numel 与 manifest 完全一致；
- 所有浮点 tensor 均为 BF16，默认逐 tensor 检查 finite；
- converter 记录了 fresh causal wrapper `strict=True` reload；
- 新版 converter 明确记录 100% key/shape coverage 和
  `num_frame_per_block=8` causal config。

随后真实 `inference.py` 会再次对同一个 generator 执行 `strict=True` load。任何 missing、
unexpected 或 shape mismatch 都会在生成前失败。

### 1.2 testsets 格式预处理

原始文件 `testsets/metadata_6cases_480x832.csv` 只有 `input_image + prompt`，而 causal
I2V 入口使用 `MultiVideoConcatDataset` 的目录格式。预处理器会为每张输入图编码一个只作
首帧载体的 97-frame/24fps MP4，并生成同名 caption JSON：

```text
prepared/
├── datasets/
│   ├── landscape_480x832/
│   │   ├── video/0000_row0000/000.mp4
│   │   └── caption/0000_row0000/000.json
│   └── portrait_832x480/
├── configs/
│   ├── landscape_480x832.yaml
│   └── portrait_832x480.yaml
├── videos/
│   ├── landscape_480x832/
│   └── portrait_832x480/
└── prepared_manifest.json
```

横屏 `[1,24,48,30,52]` 与竖屏 `[1,24,48,52,30]` 必须分两次推理；预处理不做
resize/crop/stretch。每个 carrier 的首帧保持原图尺寸，稳定 row id、源图片/载体 hash、
推理 config 和输出映射均写入 manifest。

### 1.3 MP4 自动门禁

`scripts/validate_stage1_causal_outputs.py` 对 6 个期望输出逐一验证：

- 输出集合准确，无缺失或旧文件混入；
- 480×832 与 832×480 方向/尺寸保持不变；
- 24 latent frames 解码为恰好 `1+(24-1)×4=93` 帧；
- 帧率为 24fps，OpenCV 可完整解码；
- 首帧相对原输入 PSNR 默认不低于 12dB；
- 全片平均像素标准差默认不低于 5，排除黑/白/纯色坏片；
- 相邻帧平均绝对差默认不低于 0.05，排除完全冻结的视频。

这些阈值只判断技术有效性。猫咪身份、动作是否符合 prompt、肢体完整性和 8-latent block
边界是否存在肉眼跳变，仍需查看生成的 6 个 MP4。

## 2. 运行前准备

以下命令都在仓库根目录执行。先确认当前代码是 `stage-1`，并检查 H100、BF16 和 FFmpeg：

```bash
cd /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0
git branch --show-current
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
ffmpeg -version | head -1

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("visible GPUs:", torch.cuda.device_count())
print("bf16:", torch.cuda.is_bf16_supported())
PY
```

`git branch --show-current` 必须输出 `stage-1`。如需更新代码，先确认指向
`zonghui-liu-opt/LongLive_cats` 的 remote 名称，再对该 remote 执行
`git pull --ff-only <remote> stage-1`，不要误从上游 NVlabs remote 拉取同名分支。

设置模型依赖与验证目录。下面路径与 `infer_batch.sh` 使用同一套 Wan2.2 5B 资源：

```bash
export LONG_LIVE_STAGE1_PROJECT_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0
export LONG_LIVE_STAGE1_AUXILIARY_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B
export LONG_LIVE_STAGE1_SOURCE_CHECKPOINT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/ckpts
export LONG_LIVE_STAGE1_ARCHITECTURE_ROOT="$LONG_LIVE_STAGE1_AUXILIARY_ROOT"
export LONG_LIVE_STAGE1_T5_CHECKPOINT="$LONG_LIVE_STAGE1_AUXILIARY_ROOT/models_t5_umt5-xxl-enc-bf16.pth"
export LONG_LIVE_STAGE1_TOKENIZER_DIR="$LONG_LIVE_STAGE1_AUXILIARY_ROOT/google/umt5-xxl"
export LONG_LIVE_STAGE1_VAE_CHECKPOINT="$LONG_LIVE_STAGE1_AUXILIARY_ROOT/Wan2.2_VAE.pth"
export LONG_LIVE_STAGE1_BASE_CHECKPOINT="$LONG_LIVE_STAGE1_PROJECT_ROOT/checkpoints/stage1/converted_causal_base.pt"
export LONG_LIVE_STAGE1_BASE_MANIFEST="$LONG_LIVE_STAGE1_PROJECT_ROOT/checkpoints/stage1/converted_causal_base.manifest.json"

export LONG_LIVE_STAGE1_VALIDATION_DIR=/local_nvme/stage1_causal_base_validation_seed1
export LONG_LIVE_STAGE1_PREPARED_ROOT="$LONG_LIVE_STAGE1_VALIDATION_DIR/prepared"
```

验证目录必须是全新的空目录。预处理器和一键脚本会主动拒绝非空目录，以免把旧 MP4 混入
本次结果。失败后不要覆盖原目录；保留现场，换一个带新后缀的目录重跑。

## 3. 参数与 `infer_batch.sh` 的对应关系

推荐先按 baseline 参数验证：`t_shift=5.0`、CFG `5.0`、50 denoise steps、seed `1`，负面
prompt 也与 `infer_batch.sh` 相同。但当前 causal 推理入口与原始 bidirectional batch 入口的
solver 能力并不完全相同：

| 参数 | causal config 字段 | 本次值 | 说明 |
|---|---|---:|---|
| t_shift | `model_kwargs.timestep_shift` | 5.0 | 对应 `infer_batch.sh --t-shift 5.0` |
| CFG | `inference.guidance_scale` | 5.0 | `1.0` 会关闭 unconditional CFG 分支；`5.0` 计算量约为非 CFG 的两倍 |
| denoise steps | `inference.sampling_steps` | 50 | converted base 来源是原始 Wan2.2 merged 权重，不使用蒸馏模型的 4-step 预期 |
| solver | 当前代码固定为 `unipc` | UniPC | causal pipeline 尚未从 YAML/CLI 读取 solver |
| seed | `logging.seed` | 1 | 固定后便于复现和横向比较 |
| negative prompt | `inference.negative_prompt` | 见下文 | 与 baseline 保持一致 |

`pipeline/causal_diffusion_inference.py` 当前将 `sample_solver` 固定为 `unipc`。内部虽然还有
`dpm++` 分支，但配置尚未接通；Euler 只在 `scripts/infer_wan22_ti2v_batch.py` 的原始
bidirectional 推理路径中实现。因而：

- 不要把 `infer_batch.sh` 的 `--solver euler` 直接追加给 `inference.py`，它不接受该参数；
- 不要只在 YAML 中添加 `solver: euler`，当前 causal pipeline 会忽略它；
- 本文命令实际使用 UniPC。若必须对齐 Euler 或选择 DPM++，需要先修改 causal pipeline、
  接通配置并补充回归测试，再做结果比较。

帧数也需要区分：预处理生成 97 帧静态 carrier，只用于满足输入视频读取契约；causal 模型输出
固定为 24 latent frames，经时间压缩比 4 的 VAE 解码后是
`1 + (24 - 1) × 4 = 93` 个像素帧。不要为了得到 97 帧输出而把 latent frames 改成 25；
Stage‑1 的契约是 24 latent frames，即 3 个、每个 8 latent frames 的 causal block。

## 4. 分步运行：审计、预处理、推理、验片

### 4.1 审计 `converted_causal_base.pt`

先单独验证转换产物。默认会扫描全部 BF16 tensor 是否 finite，耗时和主机内存占用都较高，
但正式门禁不建议跳过：

```bash
mkdir -p "$LONG_LIVE_STAGE1_VALIDATION_DIR"

python scripts/validate_stage1_causal_base.py \
  --base-checkpoint "$LONG_LIVE_STAGE1_BASE_CHECKPOINT" \
  --base-manifest "$LONG_LIVE_STAGE1_BASE_MANIFEST" \
  --source-checkpoint "$LONG_LIVE_STAGE1_SOURCE_CHECKPOINT" \
  --num-frame-per-block 8 \
  --report-path "$LONG_LIVE_STAGE1_VALIDATION_DIR/base_validation_report.json"
```

输出必须包含 `"status": "pass"`、`"strict_reload": true`，并确认 source
`"checked": true`。如只做快速排障，可临时追加 `--skip-finite-check`，但该结果不能替代最终
完整审计。

### 4.2 把 testsets 转成 causal I2V 格式

运行预处理器。它读取 6-case CSV，不 resize/crop/stretch 原图，按 480×832 横屏和
832×480 竖屏拆成两组，并生成 97-frame/24fps 静态 carrier、caption JSON、两份 inference
YAML 和可追溯 manifest：

```bash
python scripts/prepare_stage1_causal_testsets.py \
  --metadata testsets/metadata_6cases_480x832.csv \
  --output-root "$LONG_LIVE_STAGE1_PREPARED_ROOT" \
  --base-checkpoint "$LONG_LIVE_STAGE1_BASE_CHECKPOINT" \
  --architecture-root "$LONG_LIVE_STAGE1_ARCHITECTURE_ROOT" \
  --t5-checkpoint "$LONG_LIVE_STAGE1_T5_CHECKPOINT" \
  --tokenizer-dir "$LONG_LIVE_STAGE1_TOKENIZER_DIR" \
  --vae-checkpoint "$LONG_LIVE_STAGE1_VAE_CHECKPOINT" \
  --num-latent-frames 24 \
  --num-frame-per-block 8 \
  --minimum-source-frames 97 \
  --sampling-steps 50 \
  --guidance-scale 5.0 \
  --seed 1
```

预处理完成后应存在：

```text
$LONG_LIVE_STAGE1_PREPARED_ROOT/
├── prepared_manifest.json
├── configs/
│   ├── landscape_480x832.yaml
│   └── portrait_832x480.yaml
├── datasets/
│   ├── landscape_480x832/
│   └── portrait_832x480/
└── videos/
    ├── landscape_480x832/   # 推理前为空
    └── portrait_832x480/    # 推理前为空
```

可用下面命令快速检查 manifest 中确实有 6 条、两个 bucket：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["LONG_LIVE_STAGE1_PREPARED_ROOT"])
manifest = json.loads((root / "prepared_manifest.json").read_text())
print("records:", manifest["metadata"]["record_count"])
print("buckets:", [(b["bucket_id"], len(b["records"])) for b in manifest["buckets"]])
print("frame_policy:", manifest["frame_policy"])
PY
```

期望 `records: 6`，两个 bucket 各 3 条，`num_latent_frames=24`、
`num_frame_per_block=8`、`expected_pixel_frames=93`、`carrier_frames=97`。

### 4.3 复核或覆盖 t_shift、CFG、steps、negative prompt 和 seed

预处理器已经写入 t_shift `5.0`、CFG `5.0`、50 steps、seed `1`，以及与
`infer_batch.sh` 一致的负面 prompt。若需要显式复核或统一覆盖两份 YAML，可运行：

```bash
export LONG_LIVE_STAGE1_T_SHIFT=5.0
export LONG_LIVE_STAGE1_CFG=5.0
export LONG_LIVE_STAGE1_DENOISE_STEPS=50
export LONG_LIVE_STAGE1_SEED=1
export LONG_LIVE_STAGE1_SOLVER=unipc
export LONG_LIVE_STAGE1_NEGATIVE_PROMPT='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

python - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf

solver = os.environ["LONG_LIVE_STAGE1_SOLVER"].lower()
if solver != "unipc":
    raise ValueError("当前 causal inference 只接通 UniPC；不能在 YAML 中选择 Euler/DPM++")

config_dir = Path(os.environ["LONG_LIVE_STAGE1_PREPARED_ROOT"]) / "configs"
for path in sorted(config_dir.glob("*.yaml")):
    cfg = OmegaConf.load(path)
    cfg.model_kwargs.timestep_shift = float(os.environ["LONG_LIVE_STAGE1_T_SHIFT"])
    cfg.inference.sampling_steps = int(os.environ["LONG_LIVE_STAGE1_DENOISE_STEPS"])
    cfg.inference.guidance_scale = float(os.environ["LONG_LIVE_STAGE1_CFG"])
    cfg.inference.negative_prompt = os.environ["LONG_LIVE_STAGE1_NEGATIVE_PROMPT"]
    cfg.logging.seed = int(os.environ["LONG_LIVE_STAGE1_SEED"])
    OmegaConf.save(cfg, path)
    print("updated:", path)
PY
```

确认两份 YAML 的模型路径和参数：

```bash
rg -n 'generator_ckpt|timestep_shift|sampling_steps|guidance_scale|negative_prompt|seed|output_folder' \
  "$LONG_LIVE_STAGE1_PREPARED_ROOT/configs"
```

### 4.4 加载 converted base 批量推理

最稳妥的方式是单卡顺序运行两个 bucket；每个进程都以 batch size 1 顺序生成该 bucket 的
3 条视频，并通过 `checkpoints.generator_ckpt` 对 converted base 做真实 `strict=True` load：

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES=6 python inference.py \
  --config_path "$LONG_LIVE_STAGE1_PREPARED_ROOT/configs/landscape_480x832.yaml" \
  2>&1 | tee "$LONG_LIVE_STAGE1_VALIDATION_DIR/infer_landscape.log"

CUDA_VISIBLE_DEVICES=6 python inference.py \
  --config_path "$LONG_LIVE_STAGE1_PREPARED_ROOT/configs/portrait_832x480.yaml" \
  2>&1 | tee "$LONG_LIVE_STAGE1_VALIDATION_DIR/infer_portrait.log"
```

若有两张空闲 H100，可让两个几何 bucket 各占一张卡并行；这仍然是两个独立的单卡进程：

```bash
CUDA_VISIBLE_DEVICES=6 python inference.py \
  --config_path "$LONG_LIVE_STAGE1_PREPARED_ROOT/configs/landscape_480x832.yaml" \
  > "$LONG_LIVE_STAGE1_VALIDATION_DIR/infer_landscape.log" 2>&1 &
landscape_pid=$!

CUDA_VISIBLE_DEVICES=7 python inference.py \
  --config_path "$LONG_LIVE_STAGE1_PREPARED_ROOT/configs/portrait_832x480.yaml" \
  > "$LONG_LIVE_STAGE1_VALIDATION_DIR/infer_portrait.log" 2>&1 &
portrait_pid=$!

wait "$landscape_pid"
wait "$portrait_pid"
```

不要对同一份 3-sample config 使用
`torchrun --nproc_per_node=2 inference.py`。当前分布式入口使用
`DistributedSampler(..., drop_last=True)`，3 条样本无法被 2 个 rank 整除时可能丢掉 1 条，
最终输出集合门禁会失败。也不要给 causal `inference.py` 传
`--parallel-mode`、`--batch-size`、`--solver` 或 `--t-shift`；这些是另一条 batch 脚本的 CLI。

### 4.5 自动验证 6 个 MP4

两次推理均以 code 0 结束后运行输出门禁：

```bash
python scripts/validate_stage1_causal_outputs.py \
  --prepared-manifest "$LONG_LIVE_STAGE1_PREPARED_ROOT/prepared_manifest.json" \
  --minimum-first-frame-psnr-db 12.0 \
  --minimum-frame-std 5.0 \
  --minimum-temporal-abs-diff 0.05 \
  --report-path "$LONG_LIVE_STAGE1_VALIDATION_DIR/output_validation_report.json"
```

终端必须打印 `"status": "pass"` 和 `"sample_count": 6`。报告逐条记录输出 SHA256、尺寸、
帧数、fps、首帧 PSNR、画面标准差和相邻帧平均差。对应视频位于：

```text
$LONG_LIVE_STAGE1_PREPARED_ROOT/videos/landscape_480x832/   # 3 个 H×W=480×832 MP4
$LONG_LIVE_STAGE1_PREPARED_ROOT/videos/portrait_832x480/    # 3 个 H×W=832×480 MP4
```

### 4.6 人工验片

自动门禁只证明视频容器和基础像素/时序指标有效，还要逐个查看 6 个 MP4：

1. 首帧是否保持输入猫咪身份、构图和横竖方向；
2. 主体是否持续可见，是否出现全黑、全白、严重闪烁或突然换主体；
3. 动作是否基本响应 prompt，猫的脸、四肢和尾巴是否明显畸形；
4. 第 8/16/24 latent frame 附近是否出现 causal block 边界跳变；
5. 与相同 seed/CFG/steps/t_shift 的 bidirectional baseline 对比时，明确记录 solver 不同：
   baseline 可为 Euler，而当前 causal 结果是 UniPC，不能把差异全部归因于权重转换。

## 5. 一键验证入口

不需要自定义 negative prompt、也不需要拆开观察每一步时，可用一张 H100 顺序完成 base
审计、testsets 预处理、两组推理和 MP4 门禁：

一键流程与第 4 节的分步流程二选一；两者不能复用同一个非空验证目录。

```bash
export LONG_LIVE_STAGE1_VALIDATION_DIR=/local_nvme/stage1_causal_base_validation_seed1

CUDA_VISIBLE_DEVICES=0 python scripts/run_stage1_causal_testsets_validation.py \
  --metadata testsets/metadata_6cases_480x832.csv \
  --work-dir "$LONG_LIVE_STAGE1_VALIDATION_DIR" \
  --sampling-steps 50 \
  --guidance-scale 5.0 \
  --seed 1
```

该入口当前固定使用 causal UniPC、t_shift `5.0`，生成配置中的 negative prompt 默认为空。
如果需要与 `infer_batch.sh` 的负面 prompt 严格对齐，应使用第 4 节的分步流程修改 YAML 后
推理。若只检查路径、hash、CSV、carrier 和生成配置而暂不占用 GPU：

```bash
python scripts/run_stage1_causal_testsets_validation.py \
  --metadata testsets/metadata_6cases_480x832.csv \
  --work-dir /local_nvme/stage1_causal_base_prepare_only \
  --sampling-steps 50 \
  --guidance-scale 5.0 \
  --seed 1 \
  --prepare-only
```

## 6. 最终通过标准与失败处理

分步流程必须同时满足：

1. `base_validation_report.json` 为 `status=pass`，source hash 已检查，checkpoint/manifest
   一致，tensor 为 BF16 且 finite；
2. 两次 `inference.py` 均以 code 0 退出，日志中没有 missing/unexpected key、shape mismatch、
   NaN 或 CUDA OOM；
3. `output_validation_report.json` 为 `status=pass` 且 `sample_count=6`；
4. 横屏、竖屏目录各有 3 个 MP4，每个分别为对应尺寸、93 帧、24fps；
5. 人工查看 6 个 MP4，确认首帧、主体、方向和基本时序无明显异常。

常见失败的判断顺序：

- base 审计失败：先检查 converted base、manifest 和 source shard 是否来自同一次转换，禁止
  继续推理或训练；
- strict load 失败：说明 key/shape 或 causal wrapper 配置不一致，不能用 `strict=False` 绕过；
- 缺少第 3/第 6 条视频：检查是否误用了多 rank `torchrun`，改为两个独立单卡进程并在全新
  目录重跑；
- 帧数为 97 的预期错误：97 是输入 carrier，正确输出是 93 帧；
- 全黑、纯色、冻结、方向错误或首帧严重不一致：转换或真实 causal 运行链路未通过；
- 仅动作质量弱于原始 bidirectional baseline：若所有技术门禁都通过，这不单独证明转换错误。
  converted base 尚未经过 causal LoRA 适配，应继续 Stage‑1，并在 EMA LoRA merge 后以相同
  testsets、seed 和采样参数做最终人工对比。
