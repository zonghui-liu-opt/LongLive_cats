# Stage‑1 训练 checkpoints 的 testsets 推理验证

本流程用于训练结束后横向比较多个 `checkpoint_model_XXXXXX`。它对每个 checkpoint 固定选择
`adapter_ema.safetensors`，严格合并到训练使用的 immutable causal base，再以仓库中的 6 条
`testsets`、相同 seed 和相同采样参数生成视频。

自动门禁只验证文件集合、尺寸、93 帧/24fps、首帧重建、非纯色视频和非完全冻结；猫咪身份、
动作是否符合 prompt、肢体完整性和 8-latent block 边界仍由人工查看 `comparison.html` 判断。
脚本不会自动宣称某个 checkpoint 最优。

## 1. 前置条件

在仓库根目录执行，并沿用训练 runbook 中的模型路径：

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
export LONG_LIVE_STAGE1_TRAIN_DIR="$LONG_LIVE_STAGE1_PROJECT_ROOT/logs/stage1_formal"
```

要求：

- training root 下的 checkpoint 目录名为 `checkpoint_model_XXXXXX`；
- 每个待比较目录至少包含 `_SUCCESS`、raw/EMA adapters、resolved config、base reference 和
  checkpoint manifest；artifact-only checkpoint 可以验证，不要求 `_RESUMABLE_SUCCESS`；
- base 必须仍与训练 checkpoint 中记录的 SHA256 一致；
- FFmpeg/ffprobe、OpenCV、PEFT、safetensors、模型 auxiliary files 均可用；
- validation work dir 必须不存在或为空，旧失败现场不要覆盖，换新目录重跑。

## 2. 推荐命令

验证 training root 下全部已提交 checkpoints：

```bash
export LONG_LIVE_STAGE1_CHECKPOINT_VALIDATION_DIR=/local_nvme/stage1_checkpoint_testsets_seed1

CUDA_VISIBLE_DEVICES=0 python scripts/run_stage1_training_checkpoints_validation.py \
  --training-root "$LONG_LIVE_STAGE1_TRAIN_DIR" \
  --metadata testsets/metadata_6cases_480x832.csv \
  --work-dir "$LONG_LIVE_STAGE1_CHECKPOINT_VALIDATION_DIR" \
  --sampling-steps 50 \
  --guidance-scale 5.0 \
  --seed 1
```

默认参数与 causal baseline 一致：UniPC、50 denoise steps、timestep shift 5.0、CFG 5.0、
seed 1 和仓库统一 negative prompt。当前 causal pipeline 的 solver 固定为 UniPC；不要把
bidirectional baseline 的 `--solver euler` 追加到此命令。

只比较部分 step：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage1_training_checkpoints_validation.py \
  --training-root "$LONG_LIVE_STAGE1_TRAIN_DIR" \
  --steps 75,150,300,450,600,750 \
  --work-dir /local_nvme/stage1_checkpoint_subset_seed1
```

也可以显式选择任意 checkpoint；该参数可重复：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_stage1_training_checkpoints_validation.py \
  --training-checkpoint "$LONG_LIVE_STAGE1_TRAIN_DIR/checkpoint_model_000300" \
  --training-checkpoint "$LONG_LIVE_STAGE1_TRAIN_DIR/checkpoint_model_000750" \
  --work-dir /local_nvme/stage1_checkpoint_300_750_seed1
```

发现模式会校验 training root 中每一个 `checkpoint_model_XXXXXX`。若存在未提交或损坏的同名
目录，脚本会停止，不会静默漏掉它。确实希望其余 step 在某个 checkpoint 失败后继续运行时，
显式增加 `--continue-on-error`；最终总报告仍为 `status=failed` 并返回非零退出码。

## 3. 执行顺序与磁盘策略

脚本严格按 optimizer step 递增，逐 checkpoint 执行：

1. 完整审计 converted base、manifest 与可选 DiffSynth source hashes；
2. 校验训练 checkpoint manifest、全部列出文件的 hash 和 base reference；
3. 只读取 EMA adapter，执行 PEFT `safe_merge`，检查 360 tensors/180 modules/57,016,320
   参数、finite、BF16 和 fresh causal strict reload；
4. 为横屏、竖屏分别准备 carrier/config，顺序调用现有 `inference.py`；
5. 验证 6 个 MP4 并写 step 独立报告；
6. 成功后删除可重建的约 10 GiB 临时 merged checkpoint，保留 merge manifest/hash、视频和
   报告，然后继续下一个 step。

若需要保留每个 full merged checkpoint，增加 `--keep-merged`。比较 10 个 step 时可能额外占用
约 100 GiB，请先确认本地盘容量。某个 step 失败时，它已产生的 merged checkpoint不会删除，
便于排障。

`--skip-base-finite-check` 只可用于快速排障；它不会放宽 adapter/merge 的严格校验，也不能替代
最终完整 base audit。merge 默认在 CPU 上完成，可用 `--merge-device cuda:0` 覆盖，但必须为
模型构造、合并和后续推理预留足够显存。

## 4. 输出与人工选择

```text
validation_dir/
├── validation_report.json
├── comparison.html
├── checkpoint_model_000075/
│   ├── merge_manifest.json
│   ├── output_validation_report.json
│   └── prepared/
│       ├── prepared_manifest.json
│       ├── configs/
│       └── videos/{landscape_480x832,portrait_832x480}/
└── checkpoint_model_XXXXXX/
```

完成条件：

- 顶层 `validation_report.json` 为 `status=pass`；
- `passed_checkpoint_count` 等于选择的 checkpoint 数，`failed_checkpoint_count=0`；
- 每个 step 的 `output_validation_report.json` 为 `status=pass`、`sample_count=6`；
- 打开 `comparison.html`，逐行在相同 test case 下比较各 EMA step。

建议人工重点看：首帧身份保持、目标动作、身体是否完整留在画面内、落地/回正是否自然、背景与
镜头是否稳定，以及第 8/16 latent frame 附近是否跳变。选定 step 后，再用
`scripts/merge_lora_generator.py` 生成需要长期保留的最终 BF16 checkpoint；不要把临时技术指标
当作自动 best-checkpoint 分数。
