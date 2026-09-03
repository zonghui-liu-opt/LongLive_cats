# Stage-2 C4 / W16 / S1 / K4 单动作推理计时

此入口固定 `c4w16k4s1`，只生成单动作视频，分别记录 DiT、VAE 解码、视频后处理的耗时。加载指定 Stage-2 checkpoint 的 Generator EMA 并合并 LoRA，以 BF16、CFG=1、每卡 batch=1 推理。`G400` 表示训练 checkpoint 的 generator update 400；`K4` 表示每个 chunk 的 4 步 denoising，两者相互独立。

固定配置为 4 个当前 latent、12 个历史 latent、1 个全局首图 sink。W16 包含当前 chunk，不包含独立的 sink，实际 self-KV 容量为 17 latent frames。单动作生成 24 个 future latent，共 6 个 chunk：4 步 DiT denoising，加上首次 sink preload 和每 chunk 的 clean recache，共 31 次 DiT 调用。

## 运行

在 H100 服务器的项目根目录中执行。使用这次 G400 checkpoint 进行单动作计时：

```bash
STAGE2_EARLY_CHECKPOINT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/results/stage2_c4w16s1/training/formal_b1_c4w16s1_all_epochs/checkpoint_stage2_g000400 \
  bash infer_stage2_c4w16k4s1_timing.sh formal
```

默认 GPU 0，单进程。`formal` 运行全部 6 个单动作样本及 seeds 1、2、3、4，共 24 个视频。`quick` 只运行 row 0、3，seed 1，共 2 个单动作视频。两个模式都不生成双动作。

使用默认 checkpoint 根目录时可直接指定训练步数。可选的 `plan-quick` / `plan-formal` 只检查资源路径及 CPU 样本计划，不加载模型或初始化 CUDA：

```bash
bash infer_stage2_c4w16k4s1_timing.sh quick G400
bash infer_stage2_c4w16k4s1_timing.sh formal G400
```

每个视频为 96 帧、4 秒、24 FPS。分辨率遵循每行 metadata 的原始横竖屏尺寸：横屏 H×W=480×832，竖屏 H×W=832×480，计时按尺寸分组。quick 中 row 0、3 为横屏，formal 含横竖屏。首图 latent 与 24 个 future latent 合并，VAE 解码 25 latent 为 97 帧后丢弃首帧。CPU 计划会显示 `single_action_sample_count` 为 2 或 24，`two_action_sample_count` 为 0。

结果先看输出目录下的 `timing_samples.csv`：每行对应一个单动作视频，`dit_seconds`、`vae_decode_seconds`、`video_postprocess_seconds` 就是三个组件的耗时，单位秒。`timing_summary.json` 提供汇总。旧结果中 `groups: []`、`measured_sample_count: 0` 表示旧 trace 没有计时；此入口的新样本生成后才会得到实际测量值。

## checkpoint 与数据路径

默认路径与 `run_stage2_c4w16s1_h100.sh` 的 8 卡训练目录一致：

```text
STAGE2_WORK_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_stage2_h100_c4w16s1_micro1_acc8_longrun
STAGE2_TRAIN_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_training_h100_c4w16s1_micro1_acc8_longrun
SNAPSHOT_ROOT=$STAGE2_TRAIN_ROOT/formal_b1_c4w16s1_all_epochs
```

checkpoint 路径为 `$SNAPSHOT_ROOT/checkpoint_stage2_g000400`，需完整保留 `_SUCCESS`、manifest、provenance、EMA 和其余认证文件。C4 训练 G10/G20/G30 尚无 Generator EMA，应选用 G40 或之后已完成的 checkpoint。4 卡训练目录带 `_4gpus` 后缀，请相应覆盖 `STAGE2_WORK_ROOT`、`STAGE2_TRAIN_ROOT`；也可直接覆盖 `SNAPSHOT_ROOT`：

```bash
SNAPSHOT_ROOT=/actual/c4_training/formal_b1_c4w16s1_all_epochs \
  bash infer_stage2_c4w16k4s1_timing.sh quick G400
```

指定单个 checkpoint 和独立输出目录时：

```bash
STAGE2_EARLY_CHECKPOINT=/actual/checkpoint_stage2_g000400 \
STAGE2_EARLY_OUTPUT=/actual/inference_timing_g000400_run1 \
  bash infer_stage2_c4w16k4s1_timing.sh quick
```

只有目录名为 `checkpoint_stage2_gNNNNNN` 时才自动提取训练步数；自定义目录名还需传入 `G400`。没有指定 step 或可提取的 checkpoint 路径时，入口会报错。

模型和数据位于其他目录时，按需覆盖以下变量：

```bash
export STAGE2_PYTHON=/actual/longlive/bin/python
export STAGE2_TORCHRUN=/actual/longlive/bin/torchrun
export STAGE2_WORK_ROOT=/actual/c4_stage2_assets
export LONG_LIVE_STAGE2_SOURCE_MANIFEST=/actual/cache_manifest.attested.json
export ARCH_ROOT=/actual/Wan2.2-TI2V-5B
export LONG_LIVE_STAGE2_ARCHITECTURE_ROOT="$ARCH_ROOT"
export LONG_LIVE_STAGE2_T5_CHECKPOINT="$ARCH_ROOT/models_t5_umt5-xxl-enc-bf16.pth"
export LONG_LIVE_STAGE2_TOKENIZER_DIR="$ARCH_ROOT/google/umt5-xxl"
export LONG_LIVE_STAGE2_VAE_CHECKPOINT="$ARCH_ROOT/Wan2.2_VAE.pth"
export SNAPSHOT_ROOT=/actual/c4_training/formal_b1_c4w16s1_all_epochs
bash infer_stage2_c4w16k4s1_timing.sh quick G400
```

多卡通过独立样本做数据并行，不会把一个视频分布到多卡。覆盖可见卡时会自动推导进程数；也可同时明确两者：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 STAGE2_INFERENCE_NPROC=4 \
  bash infer_stage2_c4w16k4s1_timing.sh formal G400
```

## 计时口径与产物

推理自动计时，各阶段使用同步后的时钟，单位为秒；同步本身会引入测量开销。每个样本的 trace 在 `generation.timing` 下记录：

| 字段 | 范围 |
| --- | --- |
| `dit_seconds` / `dit_calls` | 实际 Generator forward，含 denoising、sink preload、clean recache |
| `vae_decode_seconds` / `vae_decode_calls` | `vae.decode_to_pixel()`，每个视频 1 次 |
| `video_postprocess_seconds` / `video_postprocess_calls` | 丢弃首帧、归一化和 clamp，以及视频导出中的 CPU 传输、uint8 转换、MP4 编码与写入；每个视频 2 段计时 |
| `total_seconds` / `other_seconds` | 断点预检结束至 MP4 提交的总耗时，以及未归入上述三类的开销 |
| `device` / `method` / `schema` | 执行设备、计时方法和记录版本 |

总耗时包括该范围内的调度器、hash/cache 检查、CPU 编排、文件路径检查及提交前视频验证/ffprobe；不包括模型/checkpoint 加载、T5 文本编码、首图 VAE 编码、最终 trace/manifest 验证，因此不能作为端到端服务延迟。记录还包含 `rank`、`rank_sample_index`、`cold_start`、`process_run_id` 和 `device_name`。统计时请保持 GPU 型号、可见卡分配和并发负载一致。

输出沿用现有推理产物布局，并增加 `timing_summary.json` 和 `timing_samples.csv`。每个新生成样本会打印一行计时日志；trace 保留逐样本记录。此入口的汇总只包含 `c4w16k4s1` 单动作样本，按分辨率、设备名称和计时方法分别统计；同型号 GPU 的不同本地卡号可合并统计。

每次进程启动时，每个 rank 的首个实际生成样本标为 `cold_start`；续跑跳过的已有样本不计入序号。没有额外 warmup，汇总提供 `all_samples` 和 `excluding_first_per_rank_run` 两组统计。后者仅排除每次进程运行的首样本，其他形状仍可能在后续样本首次初始化，不能据此声称已充分预热。多卡 quick 可能每个 rank 只有一个样本，此时第二组为空；较充分的性能测量可用默认单卡 formal。

默认输出目录为：

```text
$STAGE2_TRAIN_ROOT/inference_early_g000400_infer_i2v_stage2_c4w16k4s1_timing_quick_<contract-hash>
```

入口打印最终完整路径。目录中还包括 `videos/`、`traces/`、`manifest.json`、`index.html`；日志位于输出目录对应的相邻 `.log` 文件。单动作配置有独立的 contract hash，默认输出不会混入之前 C8 或单双动作混合结果。已完整的视频与 trace 会校验后跳过，沿用历史计时，不能把续跑当作一次新的性能测量。重新测量时通过 `STAGE2_EARLY_OUTPUT` 指定新的空目录。

## 更新代码后继续运行

Stage-2 推理不再计算或校验源码版本，不比较当前代码与历史 trace、manifest、checkpoint 中的 `code_version`，也不要求各 rank 的源码 hash 一致。旧版本字段只保留为历史记录；无需手动编辑 trace、删除视频、更换输出目录或设置跳过校验的环境变量。

更新代码后直接重跑原命令即可继续。完整视频与 trace 保持原样，只生成尚未完成的样本；已有 manifest 仍按真实输入、权重、推理配置和产物检查。旧 HTML 和计时汇总会根据已验证的 trace 自动刷新，避免因展示文件格式或历史代码版本变化再次中断。以前未记录计时的样本在汇总中标为缺失，不补造耗时；新生成样本继续记录三段计时。
