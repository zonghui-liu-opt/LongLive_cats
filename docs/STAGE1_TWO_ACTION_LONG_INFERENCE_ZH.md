# Stage-1 3750 EMA：10.54 秒双动作推理

本入口只验证 `checkpoint_model_003750` 的 EMA adapter，生成 4 只猫、2 种动作顺序、2 种
prompt 风格组成的 16 条视频。每条视频为 64 latent 帧，经 VAE 解码为 253 pixel 帧，24fps，
约 10.54 秒。

本地代码测试不能替代真实 H100 推理。本流程的自动检查只验证文件、帧数、尺寸、FPS、首帧、
非纯色和非全程冻结；动作是否正确、顺序是否自然、身份与肢体是否稳定，必须人工查看 HTML。

## 1. 设置环境

在仓库根目录使用现有 Stage-1 模型路径。下面路径与当前内网 Stage-1 环境一致；如挂载位置不同，
只替换绝对路径：

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
export LONG_LIVE_STAGE1_TRAIN_DIR="$LONG_LIVE_STAGE1_PROJECT_ROOT/results/stage1_600cats"
export LONG_LIVE_STAGE1_PYTHON=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python
```

Python 环境需包含项目 `requirements.txt` 的依赖；系统还需提供 `ffmpeg`、`ffprobe`、可用的
CUDA/PyTorch 和一张 H100。脚本在未设置 `CUDA_VISIBLE_DEVICES` 时默认使用 GPU 0；如需指定
另一张单卡，在启动前设置，例如 `export CUDA_VISIBLE_DEVICES=2`。

确认目标 checkpoint 至少包含已发布 marker、manifest、base reference、resolved config 和
`adapter_ema.safetensors`：

```bash
ls "$LONG_LIVE_STAGE1_TRAIN_DIR/checkpoint_model_003750"
```

## 2. 启动一次全新实验

为每次运行使用一个不存在或完全为空的新目录。失败现场会保留，重跑时请换一个新目录，不要删除
或覆盖旧目录：

```bash
cd "$LONG_LIVE_STAGE1_PROJECT_ROOT"
./infer_stage1_two_actions_10s.sh \
  /local_nvme/stage1_two_actions_3750_seed1_run01
```

也可以用环境变量传入 work dir：

```bash
export LONG_LIVE_STAGE1_TWO_ACTION_WORK_DIR=/local_nvme/stage1_two_actions_3750_seed1_run02
./infer_stage1_two_actions_10s.sh
```

入口固定显式传入：

- `checkpoint_model_003750`，只使用 EMA merge；
- 64 latent frames、8-frame blocks、253 pixel/carrier frames；
- UniPC、50 sampling steps、CFG 5.0、seed 1；
- prompt-style HTML 和重复首帧显式 opt-in。

脚本不传 `--keep-merged`：输出技术验证成功后会删除可重建的完整 merged checkpoint，保留 merge
manifest、报告和视频；若 merge、preparation、推理或输出技术验证失败，现有流程会保留当时的
merged checkpoint 与 work dir 现场。入口不修改 sink、attention、solver 或 VAE streaming
设置，尾段 8–10.54 秒完全由模型生成，不做复制帧补时。

## 3. 预期产物

成功后目录结构为：

```text
stage1_two_actions_3750_seed1_run01/
├── validation_report.json
├── comparison.html
└── checkpoint_model_003750/
    ├── merge_manifest.json
    ├── output_validation_report.json
    └── prepared/
        ├── prepared_manifest.json
        ├── configs/
        ├── datasets/
        └── videos/
```

快速检查顶层和输出报告：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/local_nvme/stage1_two_actions_3750_seed1_run01")
top = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
output = json.loads(
    (root / "checkpoint_model_003750/output_validation_report.json").read_text(encoding="utf-8")
)
prepared = json.loads(
    (root / "checkpoint_model_003750/prepared/prepared_manifest.json").read_text(encoding="utf-8")
)
print("top:", top["status"], top["checkpoint_count"], top["checkpoints"][0]["optimizer_step"])
print("output:", output["status"], output["sample_count"])
print("frames:", prepared["frame_policy"])
print("sampling:", prepared["sampling"])
PY
```

通过时应看到：

- 顶层 `status=pass`、checkpoint count 为 1、optimizer step 为 3750；
- 输出报告 `status=pass`、sample count 为 16，且没有缺失或额外 MP4；
- frame policy 为 64 latent、253 expected pixel frames、253 carrier frames、24fps、8-frame blocks；
- sampling 为 UniPC、50 steps、CFG 5.0、seed 1；
- 每条视频恰好 253 帧，分辨率与对应首帧一致，并通过首帧、非纯色、非冻结技术门禁。

## 4. 打开并人工比较 HTML

`comparison.html` 使用相对链接，可以连同整个 work dir 一起复制到有浏览器的机器。直接打开：

```bash
xdg-open /local_nvme/stage1_two_actions_3750_seed1_run01/comparison.html
```

页面每行是一个 `case_group`，两列分别为 `absolute_timeline` 和 `sequential`，共 8 行、16 条
视频，并显示猫、动作顺序和完整 prompt。请人工重点判断：

- 跳跃与逗猫棒动作是否按指定顺序各完成一次；
- 两个动作之间的恢复和短暂停顿是否自然；
- 猫咪身份、毛色、肢体和全身入镜是否稳定；
- 摄像机与白色背景是否稳定；
- 后半段在 24-latent 滚动 KV cache 下是否漂移；
- 第二个动作完成后，8–10.54 秒的模型生成尾段是否保持稳定坐姿。

这些视觉结论不会由脚本自动判定。若出现 OOM 或视频退化，请保留本次 work dir 并记录错误；
不要自行开启 streaming VAE、量化、CPU offload、改变 attention/sink，或通过后处理补帧。
