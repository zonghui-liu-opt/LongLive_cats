# Stage-1 Continuation：内网 H100 快速运行

这是 step 3750、10 秒双动作推理当前唯一保留和支持的工程入口。旧的
uniform-prompt 双动作脚本、prompt-style 对比元数据及专用报告链路已经移除。

本入口使用 `checkpoint_model_003750` 的 EMA，在单张 H100 上生成固定矩阵：

- 4 只猫 × 2 个动作顺序 × `sink=0/1`，共 16 条视频；
- 每条按 `A 24 latent → HOLD 16 → B 24` 连续生成；
- B 的首 latent 复用 HOLD 尾 latent，但 KV、全局 cursor 与 RoPE 不重置；
- 64 latent 最后只做一次 VAE decode，输出 253 帧、24fps。

自动报告只做技术门禁，不评价画面优劣，也不会推荐 sink。

## 1. 准备代码与环境

将本任务对应的完整仓库版本同步到 H100 节点；不要只复制 shell，因为 `inference.py`、pipeline、runner 和校验模块需保持同一版本。使用已有 LongLive Python 环境，并确认系统可用 `ffmpeg`、`ffprobe`、CUDA 和一张 H100。

```bash
cd /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0

export LONG_LIVE_STAGE1_PROJECT_ROOT="$PWD"
export LONG_LIVE_STAGE1_AUXILIARY_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B
export LONG_LIVE_STAGE1_ARCHITECTURE_ROOT="$LONG_LIVE_STAGE1_AUXILIARY_ROOT"
export LONG_LIVE_STAGE1_T5_CHECKPOINT="$LONG_LIVE_STAGE1_AUXILIARY_ROOT/models_t5_umt5-xxl-enc-bf16.pth"
export LONG_LIVE_STAGE1_TOKENIZER_DIR="$LONG_LIVE_STAGE1_AUXILIARY_ROOT/google/umt5-xxl"
export LONG_LIVE_STAGE1_VAE_CHECKPOINT="$LONG_LIVE_STAGE1_AUXILIARY_ROOT/Wan2.2_VAE.pth"

export LONG_LIVE_STAGE1_SOURCE_CHECKPOINT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/ckpts
export LONG_LIVE_STAGE1_BASE_CHECKPOINT="$LONG_LIVE_STAGE1_PROJECT_ROOT/checkpoints/stage1/converted_causal_base.pt"
export LONG_LIVE_STAGE1_BASE_MANIFEST="$LONG_LIVE_STAGE1_PROJECT_ROOT/checkpoints/stage1/converted_causal_base.manifest.json"
export LONG_LIVE_STAGE1_TRAIN_DIR="$LONG_LIVE_STAGE1_PROJECT_ROOT/results/stage1_600cats"
export LONG_LIVE_STAGE1_PYTHON=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python
```

快速确认关键输入：

```bash
test -d "$LONG_LIVE_STAGE1_TRAIN_DIR/checkpoint_model_003750"
test -f "$LONG_LIVE_STAGE1_BASE_CHECKPOINT"
test -f "$LONG_LIVE_STAGE1_BASE_MANIFEST"
bash -n infer_stage1_two_actions_continuation_10s.sh
```

## 2. 启动实验

每次必须使用不存在或完全为空的新目录。脚本默认只暴露 GPU 0；指定其他单卡时可先设置 `CUDA_VISIBLE_DEVICES=2`，不要填写多个 GPU。

```bash
cd "$LONG_LIVE_STAGE1_PROJECT_ROOT"
./infer_stage1_two_actions_continuation_10s.sh \
  /local_nvme/stage1_continuation_3750_seed1_run01
```

也可用固定环境变量：

```bash
export LONG_LIVE_STAGE1_CONTINUATION_WORK_DIR=/local_nvme/stage1_continuation_3750_seed1_run02
./infer_stage1_two_actions_continuation_10s.sh
```

runner 会先审计 converted base，再复用现有逻辑合并 step 3750 EMA；随后按横屏、竖屏两个 geometry 依次启动 `inference.py`。固定参数为 UniPC、50 steps、CFG 5.0、seed 1、非量化 KV、非 streaming VAE。

## 3. 检查技术结果

成功目录应包含：

```text
<work-dir>/
├── validation_report.json
├── comparison.html
└── checkpoint_model_003750/
    ├── merge_manifest.json
    ├── output_validation_report.json
    ├── prepared_continuation/
    └── continuation/
        └── <case_group>/
            ├── sink0.mp4
            ├── sink0.session.json
            ├── sink1.mp4
            └── sink1.session.json
```

```bash
WORK=/local_nvme/stage1_continuation_3750_seed1_run01
find "$WORK/checkpoint_model_003750/continuation" -name '*.mp4' | wc -l
find "$WORK/checkpoint_model_003750/continuation" -name '*.session.json' | wc -l

"$LONG_LIVE_STAGE1_PYTHON" - <<'PY'
import json
from pathlib import Path

root = Path("/local_nvme/stage1_continuation_3750_seed1_run01")
top = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
out = json.loads(
    (root / "checkpoint_model_003750/output_validation_report.json")
    .read_text(encoding="utf-8")
)
print("top:", top["status"], "samples:", top.get("sample_count"))
print("output:", out["status"], out["sample_count"])
print("noise pairs:", len({(x["case_group"], x["noise_identity_sha256"]) for x in out["samples"]}))
PY
```

通过时应为 16 个 MP4、16 份 session trace、顶层和输出报告均为 `pass`。每条视频须通过 253 帧、24fps、原分辨率、首帧 PSNR、非纯色和非冻结门禁；同一 case 的 sink0/1 noise identity 必须相同。成功后默认删除可重建的 merged BF16 checkpoint；仅在直接调用 Python runner 调试时可加 `--keep-merged`，专用 shell 不接受该选项。

## 4. 人工查看

把整个 work dir 一起复制到可用浏览器的机器，直接打开 `comparison.html`。页面为 8 行 × 2 列，每行可同步播放、暂停、拖动和归零。

重点人工观察：动作各出现一次且顺序正确、HOLD 只有轻微自然运动、B 在 HOLD 后开始、两个边界是否自然，以及猫的身份、毛色、肢体、白背景和机位是否稳定。报告不会产生视觉分数或 sink 胜者。

## 5. 失败处理

- 不要复用或清空失败 work dir；换一个新目录重跑。
- 若 merged checkpoint 已经生成，runner 会在失败时保留它；顶层失败报告始终保留，已开始样本还会留下原子 partial session trace。
- OOM 时保留现场并反馈；不要自行开启 streaming VAE、量化、CPU offload、相对 RoPE、多 shot sink 或分段 decode。
- 若 trace 报 prompt 达到 512-token 边界、cache/cursor、dtype/device 或 noise slice 不一致，先修输入/代码契约，禁止跳过门禁继续生成。
