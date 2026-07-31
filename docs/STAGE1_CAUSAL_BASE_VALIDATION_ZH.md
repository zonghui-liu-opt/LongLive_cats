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

## 2. H100 一键验证

先按 `STAGE1_H100_RUNBOOK_ZH.md` 第 2 节导出以下变量：

```text
LONG_LIVE_STAGE1_SOURCE_CHECKPOINT
LONG_LIVE_STAGE1_ARCHITECTURE_ROOT
LONG_LIVE_STAGE1_T5_CHECKPOINT
LONG_LIVE_STAGE1_TOKENIZER_DIR
LONG_LIVE_STAGE1_VAE_CHECKPOINT
LONG_LIVE_STAGE1_BASE_CHECKPOINT
LONG_LIVE_STAGE1_BASE_MANIFEST
```

选择一个全新的空目录。单卡 H100 顺序运行横屏和竖屏两组，每组只加载一个 5B causal
模型，不使用 adapter、量化或随机初始化权重：

```bash
export LONG_LIVE_STAGE1_VALIDATION_DIR=/local_nvme/stage1_causal_base_validation_seed1

CUDA_VISIBLE_DEVICES=0 python scripts/run_stage1_causal_testsets_validation.py \
  --metadata testsets/metadata_6cases_480x832.csv \
  --work-dir "$LONG_LIVE_STAGE1_VALIDATION_DIR" \
  --sampling-steps 50 \
  --guidance-scale 5.0 \
  --seed 1
```

这里保留 50 sampling steps，因为输入是原始 Wan2.2 merged bidirectional 权重，不应使用
针对已蒸馏 LongLive checkpoint 的 4-step 质量预期。若只检查路径、hash、CSV 和生成配置而
暂不占用 GPU：

```bash
python scripts/run_stage1_causal_testsets_validation.py \
  --metadata testsets/metadata_6cases_480x832.csv \
  --work-dir /local_nvme/stage1_causal_base_prepare_only \
  --prepare-only
```

工作目录必须为空，脚本拒绝混入上次运行的视频。不要在失败目录上覆盖重跑；换一个新目录，
保留失败现场和 `validation_report.json` 供排查。

## 3. 通过标准与产物

完整成功时终端最终打印 `"status": "pass"`，且：

```text
$LONG_LIVE_STAGE1_VALIDATION_DIR/
├── validation_report.json
└── prepared/
    ├── prepared_manifest.json
    ├── configs/
    └── videos/
        ├── landscape_480x832/   # 3 个 MP4
        └── portrait_832x480/    # 3 个 MP4
```

`validation_report.json` 包含 base/source/output SHA256、tensor 统计以及每条视频的尺寸、帧数、
fps、首帧 PSNR、画面标准差和时序差异。必须同时满足：

1. base audit status 为 `pass`；
2. 两次 `inference.py` 均以 code 0 退出；
3. output report status 为 `pass` 且 `sample_count=6`；
4. 人工查看 6 个 MP4，确认首帧、主体、方向和基本时序无明显异常。

如 converted base 的画质弱于原始 bidirectional baseline，但以上技术门禁全部通过，应继续
Stage‑1 训练，并在 EMA merge 后再评价效果；若出现 strict-load、NaN、全黑、错方向、缺帧或
首帧完全不一致，则不得启动正式训练。
