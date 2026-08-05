# LongLive‑2.0：Wan2.2‑TI2V‑5B 猫咪 Causal I2V AR Stage‑1 LoRA SFT

> 状态：设计已确认，等待实现  
> 目标执行者：Codex / GPT‑5.6‑sol  
> 最后确认日期：2026‑07‑30  
> 本文件是本任务的唯一实现规格；实现中不得用未经记录的新假设替换已确认决策。

## 0. 执行纪律

- [ ] 开始前完整阅读本文件、`git status --short`、涉及的现有实现和相关测试；不要只按文件名猜测接口。
- [ ] 保留用户当前工作区：不得恢复、覆盖或顺手修改下列既有变更：
  - `fouroversix/src/fouroversix/model/modules/linear_ori.py`（已删除）
  - `wan_5b/utils/prompt_extend.py`（已删除）
  - `wan_5b/utils/qwen_vl_utils.py`（已删除）
  - `wan_5b/utils/system_prompt.py`（已删除）
  - `training_sets/`（未跟踪，属于用户数据）
- [ ] 先复用现有代码，再做最小扩展。禁止另起一套训练器、LoRA 系统、I2V conditioning、FSDP 包装、VAE/T5 编码、推理框架或 merge 逻辑。
- [ ] 每完成一个小步骤，先运行该步骤的最小测试并记录证据，再勾选；前一 Gate 未通过，不进入后一阶段。
- [ ] 本次新增或修改的 Stage‑1 causal LoRA 路径中，所有 silent fallback 改为 fail-fast：shape、hash、key、步数、world size、缓存版本或 checkpoint 不一致时必须停止；未触及的 DMD/T2V legacy 行为不得借此扩大重构。
- [ ] 不提交、不推送、不删除用户文件，除非用户另行明确授权。
- [ ] 本轮实现不替用户做视频推理效果判断；代码质量验收与视觉质量选择严格分离。

## 1. 最终目标与完成定义

训练得到一个保留三类猫咪动作能力的 **causal I2V AR Stage‑1 checkpoint**：

1. 起点是用户已经合并好的 bidirectional Wan2.2‑TI2V‑5B 猫咪模型。
2. 先将 DiffSynth flat/sharded 权重一次性严格转换为 LongLive native causal base。
3. 冻结 causal base，仅训练指定 q/k/v/o 与两个 FFN Linear 上的 rank‑32 LoRA。
4. 使用 600 条猫咪视频做 5 epochs、750 optimizer updates 的 clean-context Teacher Forcing + Error Recycling 训练。
5. 训练期间仅保存 raw/EMA adapter 与可恢复训练状态，不重复保存冻结的 5B base。
6. 用户自行根据外部推理效果选定某个 checkpoint 后，运行 merge CLI，一次性生成完整 BF16 `stage1_causal_ema_merged.pt`。

工程完成需同时满足：

- [ ] CPU/unit tests 全部通过。
- [ ] 分布式梯度等价性测试通过。
- [ ] 内网完整 600 条缓存预计算与审计通过。
- [ ] 6×H100 dry-run 完成两次 micro-step 和一次 optimizer update，无 OOM/NCCL hang/非有限值。
- [ ] 正式训练可恢复到正确 epoch、sample cursor、phase、LR、optimizer、EMA、RNG，以及下文明确的 canonical-per-SP ER buffer 状态；除 ER canonicalization 这一已接受取舍外，不静默丢状态。
- [ ] JSONL 每个 optimizer update 记录正确的全局 loss 与 logical throughput，手动绘图脚本可生成 PNG/SVG。
- [ ] 用户选定 checkpoint 后，EMA LoRA merge 产物能由现有 LongLive causal loader `strict=True` 重新加载。

视觉质量、动作优劣与 best checkpoint 选择不属于代码验收条件，由用户负责。

## 2. 明确不做

- 不做 Stage‑2 DMD。
- 不决定或调试 `local_attn_size`；Stage‑1 训练不依赖该推理窗口策略。
- 不自动读取或运行 `testsets/`，不生成验证视频，不做动作分类、排序或质量阈值。
- 不给 CSV 增加 `action` 列，不做 action-aware sampler。
- 不划分训练/验证集；600 条全部训练，用户另有测试数据。
- 不启用随机时间裁剪、随机起点、跨视频拼接、倒放、变速、flip、crop、resize、color jitter 或尾帧 padding。
- 不启用 prompt dropout、input-image dropout、noise augmentation、Min-SNR、block reweight、temporal decay 或 NVFP4。
- 不使用 W&B 或 TensorBoard；不在 checkpoint 时自动画图。
- 不将旧 `MultiVideoConcatDataset` 强行改造成 CSV 数据集，不破坏现有 DMD/T2V/legacy LoRA 路径。

## 3. 已锁定的设计

### 3.1 数据与 conditioning

| 项目 | 锁定值 |
|---|---|
| manifest | `training_sets/metadata_600clips_480x832_buckets.csv` |
| schema | `video,prompt,input_image,height,width,bucket`，不需要 `action` |
| 样本数 | 内网完整数据严格 600；本地文件目前仅是 header+1 row fixture，不能冒充完整集 |
| clip 规则 | 每行是独立视频；源视频至少 97 帧，只取 presentation-order frame `0..92` |
| pixel frames | 固定 93；标准97帧源丢弃 index93..96，若源更长也忽略 index92 之后全部内容；不 padding、不重采样、不 resize |
| VAE 输出 | 93 pixel frames → 24 latent frames |
| 空间 latent | 480×832 → 30×52；832×480 → 52×30 |
| AR block | `num_frame_per_block=8`，共 3 blocks |
| 首 latent | 必须由 CSV 的显式 `input_image` 单独编码；保持 clean、`t=0`、不计 loss |
| 有效监督 | block0=7、block1=8、block2=8，总计 23 latent frames |
| prompt | `prompt_mode=repeat_global`；完整 timeline prompt 对 3 blocks 相同 |
| augmentation | 全部关闭；预处理严格确定性 |

frame accounting 复用现有 `temporal_compression_ratio=4`：`1+(93-1)//4=24`。三个 8-latent blocks 对应 raw window 计数为首块 29 帧、后两块各 32 帧，即 `29+32+32=93`；这只是计数校验，不允许分别重编码或产生边界 padding。

缓存中只存一份 `[512,4096]` 的 BF16 prompt embedding，不物理复制三份。训练时把它作为 single global segment 传入；复用 `SequenceParallelHelper.partition_training_inputs()` 中 `num_segments==1` 的现有 passthrough/reuse 语义。这不是 collective broadcast：同一 SP group 的三个 ranks 必须由 sampler/sample-id 断言保证读取同一 cache record，必要时再显式从 SP root 广播。该表示与 `repeat_global` 数学等价，可避免约 4.7 GiB 的冗余文本缓存。

缓存估算（600 条、单份 prompt）：video latent 约 2.0–2.1 GiB，input-image latent 约 0.08 GiB，T5 embedding 约 2.34 GiB，总计约 4.5 GiB，另加 manifest/小量 metadata。

### 3.2 并行、精度与有效 batch

| 项目 | 锁定值 |
|---|---|
| 硬件 | 单机 6×H100 80G；正式入口严格断言 `WORLD_SIZE=6` |
| SP/DP | SP=3 × DP=2 |
| rank 语义 | `[0,1,2]` 是 DP replica 0 的一个样本；`[3,4,5]` 是 replica 1 的另一个样本 |
| local batch | 每个 DP replica `batch_size=1`；不是每张 SP GPU 各一条独立数据 |
| accumulation | 2 个顺序 micro-steps；每次 optimizer update 的有效全局 batch=4 |
| FSDP | `hybrid_full`，显式 `(shard_process_group=SP3, replicate_process_group=DP2)` |
| base storage | BF16 frozen base |
| LoRA master | FP32；PEFT 保持 `autocast_adapter_dtype=True` |
| forward/reduce | BF16 forward/activation，FP32 gradient reduction/buffer |
| optimizer/EMA | AdamW moments FP32；LoRA-only EMA FP32 CPU |
| 其他 | `use_orig_params=True`、gradient checkpointing on、CPU offload false、NVFP4 off |

说明：若把 LoRA master 参数本身存为 BF16，标准 AdamW 会创建 BF16 moments，无法满足 FP32 optimizer state。不得为此另造 master optimizer；采用 FP32 LoRA master、FSDP BF16 forward 是最小且稳定的实现。

每个 SP rank 初始拥有一个 8-latent block；现有 All-to-All 在 attention 内交换 sequence/head 维。Gradient accumulation 是两个顺序 micro-step，不会把两份 activation 同时留在显存。

### 3.3 LoRA 范围

```yaml
adapter:
  type: lora
  rank: 32
  alpha: 32
  dropout: 0.0
  bias: none
  modules_to_save: []
  target_patterns:
    - '^blocks\.[0-9]+\.self_attn\.(q|k|v|o)$'
    - '^blocks\.[0-9]+\.ffn\.(0|2)$'
  expected_target_modules: 180
  expected_trainable_parameters: 57016320
  expected_adapter_tensors: 360
```

- 30 blocks ×（self-attn q/k/v/o 4 个 + FFN 0/2 两个）=180 Linear。
- trainable params 必须精确为 57,016,320；adapter A/B tensor 必须精确为 360 个。
- A 用 PEFT 标准 Kaiming 初始化，B 全零；挂载后逐个检查 B 为零。
- cross-attn、norm、q/k norm、modulation、patch/time embedding、head、T5、VAE 必须全部冻结。
- FSDP 前检查完整 target names、180 modules、57,016,320 trainable params 与 360 adapter tensors。FSDP 后各 rank 只检查 trainable names、`requires_grad`、optimizer 参数身份与 shard topology；不能对单 rank 的 local shard `numel()` 断言全局参数量。rank0 selective gather 后再由 360 keys/shapes 复算全局 57,016,320；任一偏差立即失败。

### 3.4 训练目标

- 名称使用 **clean-context Teacher Forcing**，不是 Diffusion Forcing。
- `num_train_timesteps=1000`，`timestep_shift=5.0`。
- 每个 8-latent block 独立采样一个 timestep，block 内共享；复用 `BaseModel._get_timestep()`。
- 显式 image latent 始终 clean、`t=0`，并在所有 ER 注入之后再次覆盖。
- Flow Matching MSE；23 个有效 latent frame 等权。
- SP 下按全局有效 frame count 归一化；block0/1/2 单独保留 numerator/count 供日志。
- `prompt_dropout_prob=0`、`input_image_dropout_prob=0`、`noise_augmentation_max_timestep=0`。

### 3.5 训练预算、phase 与精确步号

600 samples、DP2、local batch1、accum2：

```text
每个 rank-group 每 epoch 的 micro-batches = 600 / 2 = 300
每 epoch optimizer updates              = 300 / 2 = 150
Phase A: 2 epochs                        = 300 updates
Phase B: 3 epochs                        = 450 updates
总计: 5 epochs                           = 750 updates
checkpoint: 每 0.5 epoch                 = 75 updates
```

严格区分：

- `update_index`：即将执行的 0-based update，范围 0..749。
- `optimizer_step`：已经完成的 update 数，日志/目录范围 1..750。
- checkpoint `000300` 表示 Phase A 的 300 次 update 已完成；恢复后下一次 `update_index=300`。

| update_index | phase | LR | ER |
|---|---|---|---|
| 0..299 | A / Teacher Forcing | 常数 `1e-5` | collect only，绝不 inject |
| 300..479 | B / ramp | smoothstep `1e-5 → 5e-6` | smoothstep ramp |
| 480..749 | B / plateau | `5e-6` | 目标概率保持 |

Phase B ramp 共 180 updates：

```text
u = clip((update_index - 300) / (180 - 1), 0, 1)
s = 3*u^2 - 2*u^3
lr = 1e-5 + (5e-6 - 1e-5) * s
p_active = 0.5 * s
p_context_given_active = 0.30 / 0.50 = 0.60
p_latent_given_active  = 0.10 / 0.50 = 0.20
p_noise_given_active   = 0
```

因此有效 context/latent/noise 概率严格为 `0.30*s / 0.10*s / 0`；plateau 时 no-injection 约 `0.5 + 0.5×0.4×0.8 = 0.66`，任意注入约 0.34。若 ramp updates 为 1，resolver 必须有显式特判；本配置不是该情况。

同一 accumulation cycle 的两个 micro-step 必须共享同一个 `update_index`、LR 和 ER schedule。Phase 切换不重建 optimizer，`reset_optimizer_between_phases=false`。

schedule 是由 resolved phase config 与 completed step 计算的 stateless 纯函数，不维护第二份可漂移的 scheduler 状态；checkpoint 保存 resolved config/hash 和 completed step即可精确重建。每个 update 开始、读取两个 micro-step 之前，先把该 update 的 LR 写入现有 optimizer param group。

每 0.5 epoch 只产出 checkpoint，供用户在外部手动推理验证；训练器自身不启动 validation。

### 3.6 Error Recycling

只保留一份 latent-error buffer：

```yaml
error_recycling:
  enabled: true
  enable_position_bucketing: true
  num_buckets: 24
  buffer_size_per_bucket: 32
  buffer_warmup_steps: 75
  replacement_strategy: random
  modulate_factor: 0.3
  clean_buffer_update_prob: 0.1
```

- buffer 存 `x0_pred - clean_latent`，`detach()` 后 BF16 CPU；不保留计算图。
- 24 是 timestep bins，不是 24 latent frames；SP3 后每个 SP rank 拥有 8 个 timestep bins 与 1 个 local block position。
- 上限按 `(local_position,timestep_bucket)` 的 32 entries 计算；混合横竖屏仍共享这个容量上限，entry 带 spatial shape，采样时严格筛选相同 shape。不得因两个 orientation 把上限静默翻倍。
- 单进程 buffer 上限约 292 MiB。
- Phase A 始终收集但不注入。前 75 个 `update_index`（严格 `<75`，不是 `<=75`）可在同 SP position 的 DP group 之间 gather 加速 warmup。
- Phase A 的所有 eligible residual 不受 injection gate 抑制，提交到该 rank 所属的 position/timestep/shape bucket（非本 rank timestep shard 仍按既有 ownership 过滤）；Phase B 中 master inactive 的 clean branch 仅以 `clean_buffer_update_prob=0.1` 更新，active branch 正常更新，避免 clean residual 淹没回放分布。
- Phase B 的 logical gate 必须由每个 SP group root 按样本生成并广播；同一视频的三个 SP ranks 不能各自随机决定是否 active/context/latent。
- 每个 SP rank 仍从自己的 local buffer 取对应 block residual。
- helper 返回实际应用的 block 数；gate 命中但 bucket 为空时，realized injection 必须记为 false/0。
- context 与 latent injection 共用此 buffer；`noise_error_buffer=None`，不分配、不计算 `noise_err`、不 gather、不保存。
- 显式 I2V 首 latent 在任何注入后重新用 `input_image` latent 覆盖。

### 3.7 Optimizer 与 LoRA-only EMA

```text
AdamW(LoRA only)
lr: phase schedule
betas: (0.0, 0.999)
eps: 1e-8
weight_decay: 0
max_grad_norm: 10
```

- 记录 clip 前的 global grad norm；clip 后再 step。
- gradient 或 loss 非有限时，该 attempt 不执行 optimizer/EMA、不提交 ER buffer side effects，也不提交 sample cursor/completed step；保留同一对 cached micro-batches，用新 diffusion RNG 重试同一个 `update_index`。记录独立 `nonfinite_attempt` diagnostic；同一 update 连续 2 次非有限则 fail-fast。这样既满足“跳过坏 update”，也不改变 5 epochs/750 个成功 updates 的口径。
- EMA 只跟踪 LoRA local shards：decay=0.99、start at completed optimizer step 75、FP32 CPU。
- step75 optimizer update 完成后初始化 EMA 为当时 raw LoRA；从 step76 起做 decay update。
- 现有 `EMA_FSDP` 保留给 legacy full-finetune；不得拿它 summon/copy 全 5B。

### 3.8 全局 loss 与 backward 缩放

这是 P0 正确性门禁。设每个 SP rank 的本地有效 loss numerator 为 `n_local`，单个样本的 SP 全局 valid count 为 23。

```text
local normalized contribution = n_local / 23
loss_for_backward = local normalized contribution * SP_size / grad_accum
                  = n_local / 23 * 3 / 2
```

原因：FSDP/HYBRID 会在 SP×DP ranks 上平均参数梯度，而目标是 SP contribution 求和、DP samples 求平均。缺少 `×SP_size` 会让真实梯度小 3 倍。必须用 tiny model 做 SP3×DP2 与非 SP reference 的梯度/参数更新 parity test，不能凭日志看起来正常就放行。

日志聚合不得直接 all-reduce rank0 的 scalar，也不得只用最后一个 micro-step：

```text
loss_total = sum(numerator over DP × SP × 2 microsteps)
             / sum(valid_count over DP × SP × 2 microsteps)
```

每个 block 同理。固定布局下，每个 optimizer update 的有效 frame counts 应为：block0=28、block1=32、block2=32、total=92。`loss_total` 是 23 帧加权平均，不能简单平均三个 block mean。

首个 micro-step 可用 FSDP `no_sync()` 优化通信，但 correctness 优先；若当前 PyTorch/FSDP 组合不兼容，可保留两次同步，必须记录原因与性能影响。

### 3.9 JSONL 与 throughput 口径

JSONL 是唯一持久 metric source：rank0 每个 optimizer update append 一行，console 每 10 steps；不自动绘图。

token 数运行时从实际 latent H/W 和 patch `(1,2,2)` 推导，不能只硬编码 390：

```text
patch tokens / latent frame = (H_latent/2) * (W_latent/2)
480×832 or 832×480          = 15 * 26 = 390
DiT logical tokens/sample   = 2 * 24 * 390 = 18,720
supervised tokens/sample    = 23 * 390     = 8,970
global batch4 / update      = 74,880 DiT; 35,880 supervised
logical source frames/update= 4 * 93 = 372
```

SP 只是切分一条 sequence，吞吐量绝不能再乘 3。

计时范围：第一次 `next(dataloader)` **之前** reset GPU peak、CUDA synchronize + `perf_counter()`，因此包含 cache data read、两个 micro-steps、forward/backward、clip、optimizer 与 EMA；该 update 的第二个 micro-step 后，在 clip/optimizer/EMA 完成时 synchronize 并停止。随后跨 6 ranks 聚合 elapsed，分母取 `MAX`。不包含 JSONL、console、GC、checkpoint、plot 或 inference。

每个 attempt 独立计时且拥有同样的 attempted-token numerator。失败 attempt 只写 `nonfinite_attempt` 的 duration/attempted throughput；成功 attempt 写一条 `train_step`，其 throughput 只对应该成功 attempt，不把先前失败 attempt 的时间或 tokens 混入，避免分子分母错配。nonfinite 事件另行保留，整体运行效率可由所有 attempt 事件审计。

每条 `train_step` 至少记录：

- schema/version、record type、experiment/run/parent run id、`resume_from_step`、全局单调 `attempt_index`；
- `optimizer_step`、`update_index`、epoch index/progress、phase、ramp `u/s`、实际 LR；
- global total loss、block0/1/2 numerator/count/loss；
- pre-clip grad norm、nonfinite/skip 状态；
- scheduled active/context/latent/noise probabilities；
- logical gate counts/rates、actual applied block counts/rates；
- error buffer entry/fill 按 SP rank、resolution 的 min/mean/max；
- logical DiT tokens/s、supervised tokens/s、samples/s、logical pixel frames/s；
- step seconds max/mean、straggler=`max/mean`；
- max allocated/reserved GPU memory（跨 rank max）。

每次进程启动先 append `run_start`。resume 创建新的 `run_id`，记录 `parent_run_id` 和 `resume_from_step`；不截断旧日志。启动时扫描该 lineage 中所有合法 JSON 行的最大 `attempt_index`，下一个值取 `max(checkpoint_next_attempt_index, jsonl_max+1)`，保证 stale suffix 存在时仍全局单调。plotter 沿最新 lineage 重建曲线，child 对重叠 optimizer step 覆盖 parent，同时保留原始 JSONL 审计记录；允许忽略最后一行被中断写坏的 JSON。

## 4. 推荐配置形态

实现后 `configs/train_i2v_ar.yaml` 应以配置为 source of truth，不再用独立 `max_iters` 手工复制总步数。下面字段名可在实现时做极小调整，但语义、数值和层级必须保持；调整需同步 schema tests 与本文。

```yaml
infra:
  expected_world_size: 6
  sequence_parallel_size: 3
  data_parallel_size: 2
  sharding_strategy: hybrid_full
  explicit_fsdp_process_groups: true
  mixed_precision: true
  gradient_checkpointing: true
  generator_fsdp_wrap_strategy: size
  cpu_offload: false
  model_quant: false

model_kwargs:
  model_name: Wan2.2-TI2V-5B
  timestep_shift: 5.0
  num_frame_per_block: 8

checkpoints:
  generator_ckpt: /absolute/path/to/converted_causal_base.pt
  base_manifest: /absolute/path/to/converted_causal_base.manifest.json

model_paths:
  architecture_root: /absolute/path/to/Wan2.2-TI2V-5B
  t5_checkpoint: /absolute/path/to/models_t5_umt5-xxl-enc-bf16.pth
  tokenizer_dir: /absolute/path/to/google/umt5-xxl
  vae_checkpoint: /absolute/path/to/Wan2.2_VAE.pth

algorithm:
  trainer: diffusion
  i2v: true
  causal: true
  teacher_forcing: true
  independent_first_frame: true
  num_train_timestep: 1000
  denoising_loss_type: flow
  noise_augmentation_max_timestep: 0
  prompt_dropout_prob: 0.0
  input_image_dropout_prob: 0.0

adapter:
  type: lora
  rank: 32
  alpha: 32
  dropout: 0.0
  bias: none
  modules_to_save: []
  target_patterns:
    - '^blocks\.[0-9]+\.self_attn\.(q|k|v|o)$'
    - '^blocks\.[0-9]+\.ffn\.(0|2)$'
  expected_target_modules: 180
  expected_trainable_parameters: 57016320
  expected_adapter_tensors: 360

preprocessing:
  min_source_frames: 97
  selected_frame_start: 0
  selected_frame_count: 93
  expected_fps: 24
  allow_resize: false
  allow_padding: false
  dtype: bfloat16
  prompt_mode: repeat_global

data:
  backend: stage1_i2v_cache
  metadata_path: training_sets/metadata_600clips_480x832_buckets.csv
  cache_dir: /absolute/path/to/stage1_i2v_cache
  expected_num_samples: 600
  image_or_video_shape: [1, 24, 48, 30, 52]  # canonical temporal/landscape shape
  allowed_latent_spatial_shapes: [[30, 52], [52, 30]]
  batch_size: 1
  num_workers: 2
  deterministic: true
  prompt_mode: repeat_global

training:
  seed: 42  # 可覆盖的可复现实装默认值；resolved config 必须记录实际值
  gradient_accumulation_steps: 2
  reset_optimizer_between_phases: false
  optimizer:
    type: adamw
    beta1: 0.0
    beta2: 0.999
    eps: 1.0e-8
    weight_decay: 0.0
    max_grad_norm: 10.0
  ema:
    enabled: true
    decay: 0.99
    start_step: 75
    dtype: float32
    device: cpu
    trainable_only: true
  phases:
    - name: phase_a_teacher_forcing
      epochs: 2
      lr: {schedule: constant, start: 1.0e-5, end: 1.0e-5}
      error_recycling: {mode: collect_only}
    - name: phase_b_error_recycling
      epochs: 3
      transition: {schedule: smoothstep, ramp_fraction: 0.4}
      lr: {start: 1.0e-5, end: 5.0e-6}
      error_recycling:
        mode: collect_and_inject
        max_active_prob: 0.5
        effective_context_prob: 0.30
        effective_latent_prob: 0.10
        effective_noise_prob: 0.0
  error_recycling:
    enabled: true
    enable_position_bucketing: true
    num_buckets: 24
    buffer_size_per_bucket: 32
    buffer_warmup_steps: 75
    replacement_strategy: random
    modulate_factor: 0.3
    clean_buffer_update_prob: 0.1
  nonfinite_max_attempts_per_update: 2

checkpointing:
  every_epochs: 0.5
  keep_last_resumable: 2
  keep_resumable_steps: [300]
  keep_all_adapters: true
  atomic_success_marker: true

evaluation:
  interval: 0

logging:
  backend: jsonl
  jsonl_path: metrics/train_metrics.jsonl
  console_every_steps: 10
  jsonl_every_steps: 1
  fsync_every_steps: 10
  disable_wandb: true
```

`image_or_video_shape` 只为静态 temporal/SP/error-buffer 校验提供 canonical 值；每个实际 step 的 H/W 必须来自 cache batch，portrait 不能被改成 landscape。

`seed=42` 是实现层的可复现默认值，不是模型效果层面的用户锁定超参；内网运行可覆盖，但实际 seed 必须写入 resolved config、checkpoint 和 JSONL。`model_paths` 必须完全显式：cache-only 正式训练不加载 T5/VAE，但 converter、precompute 与 causal architecture 构造分别使用相应路径；converted base 是正式 generator 权重的唯一来源，不允许先静默加载另一份默认 DiT 再覆盖。

## 5. 必须复用的现有能力

| 能力 | 复用位置 | 最小修改 |
|---|---|---|
| 配置兼容 | `utils/config.py::normalize_config/section_get` | 只扩展新 section 读取，不另建配置框架 |
| timestep per block | `model/base.py::_get_timestep` | 保留 independent block sampling |
| I2V 首帧覆盖 | `utils/i2v_conditioning.py` | 继续调用现有 overwrite/zero-timestep helper |
| Flow loss | `model/diffusion.py::generator_loss` | 在现有 tensor 上暴露 detached numerator/count，不复制 loss |
| SP 切分 | `wan_5b/distributed/sp_training.py::SequenceParallelHelper` | 继续用 partition、loss mask、initial-latent ownership |
| SP config 校验 | `validate_sequence_parallel_training_config` | 扩展 24/SP3/block8 用例 |
| FSDP precision | `utils/distributed.py::fsdp_wrap` | 增加显式 SP/DP process groups；保留 BF16/FP32/use_orig_params |
| sampler seed | `utils/sampler.py::build_training_sampler` | 扩展 resolution-aware sampler，不复制 seed 逻辑 |
| VAE | `utils/wan_5b_wrapper.py::WanVAEWrapper.encode_to_latent` | 构造路径参数化，编码算法不重写 |
| T5 | `utils/wan_5b_wrapper.py::WanTextEncoder` | 构造路径参数化，tokenizer/T5 不重写 |
| prompt shape | `utils/prompt_conditioning.py` | 用于 online equivalence test；cache 不复制三份 embedding |
| ER | `utils/error_buffer.py::ErrorBuffer` | 只加 spatial-shape 安全过滤与新 schema |
| LoRA | `utils/lora_utils.py`、`trainer/distillation.py` | 抽取并复用 exact-target/load/save，不在 diffusion trainer 内复制 PEFT 逻辑 |
| optimizer resume | 现有 `FSDP.optim_state_dict*` | optimizer 只含 LoRA |
| DiffSynth flat load | `wan_5b/textimage2video.py::_flat_checkpoint_path` + Accelerate strict loader | 提取共享 helper 或直接复用，禁止复制 shard parser |
| merge | `scripts/merge_lora_generator.py`、`utils/inference_utils.py` | 扩展为 EMA checkpoint-dir 输入，不新写第二套 merge |
| native state export | `utils/nvfp4_checkpoint.py::cpu_state_dict` | BF16 merge 继续复用，与 NVFP4 训练无关 |

不要改造 `MultiVideoConcatDataset`：它会随机起点、跨视频拼接、fps resample/resize、短视频尾帧 padding，并且 `return_image` 取的是视频首帧，均与本任务冲突。新增窄职责的 CSV/cache dataset，保留 legacy dataset 原样。

## 6. 产物与 checkpoint schema

### 6.1 一次性 causal base

```text
converted_causal_base.pt
converted_causal_base.manifest.json
```

`.pt` 保持现有 LongLive native 结构：

```python
{
    "generator": {"model.<wan_key>": bf16_tensor, ...},
    "checkpoint_format": "longlive_causal_base_init",
    "checkpoint_version": 1,
    ...
}
```

manifest 包含源 shard 列表/逐文件 SHA256/aggregate SHA256、输出 SHA256、key/shape/dtype 摘要、转换命令和版本。原 DiffSynth 目录只读，绝不原地修改。

### 6.2 训练 checkpoint

```text
checkpoint_model_000075/
├── adapter_raw.safetensors
├── adapter_ema.safetensors
├── trainer_state.pt
├── ema_local_rank00000.pt ... ema_local_rank00005.pt
├── rng_state_rank00000.pt ... rng_state_rank00005.pt
├── error_buffer_sp0.pt ... error_buffer_sp2.pt
├── resolved_config.yaml
├── base_reference.json
├── checkpoint_manifest.json
├── _SUCCESS                  # adapter/artifact 完整
└── _RESUMABLE_SUCCESS        # heavy resume state 也完整
```

- `adapter_raw/ema` 都是 canonical FP32、完整 360 tensors；训练恢复使用 raw，不使用 EMA。最终 merge 完成后才把完整 generator 输出统一为 BF16。
- `trainer_state.pt` 保存 LoRA-only optimizer、completed step、next update index、epoch、committed microbatch cursor、phase derivation信息、sampler/DataLoader generator state、`next_attempt_index`、nonfinite计数与 schema version。
- 每 rank 保存 Python/NumPy/Torch CPU/CUDA RNG 和本地 EMA shard。
- 每个 SP position 仅第一个 DP replica 保存 canonical error buffer；恢复后广播给对应 DP replica。这是已确认的容量/文件数取舍：warmup 后两个 DP replicas 的 buffer 会分叉，因此恢复后的 ER 轨迹不承诺与不中断 run bitwise 相同；保存 checkpoint 本身不得修改仍在运行的另一 replica buffer。
- `_SUCCESS` 表示 adapter/manifest artifact 完整；`_RESUMABLE_SUCCESS` 在 optimizer/EMA-local/RNG/buffer 等 heavy state 全部完成后最后写入。auto-resume 只认同时含 `_RESUMABLE_SUCCESS` 与所需文件的目录。
- 所有 75-step raw/EMA adapters 保留，便于用户选 checkpoint。
- resume-heavy state 只保留最近 2 个以及永久 step300；清理 heavy state 前先原子移除 `_RESUMABLE_SUCCESS`/更新 manifest，再删除 heavy files；不得删除 `_SUCCESS`、adapter、manifest、resolved config。
- base hash 由 rank0 计算一次再广播，避免 6 ranks 重复扫描约 10 GiB 文件。
- raw adapter gather、EMA local swap→gather→`finally` restore、`FSDP.optim_state_dict` 都是 collective workflow，必须由全部 6 ranks 以完全相同顺序进入；不能只让 rank0 调用。仅 rank0 materialize 最终 canonical tensors/写文件。各阶段前后 barrier，并对写入/校验成功做 WORLD 共识；只有全体成功后 rank0 才写 success marker。

### 6.3 最终 merge

用户选定 checkpoint 后才运行：

```bash
python scripts/merge_lora_generator.py \
  --base-checkpoint /path/converted_causal_base.pt \
  --training-checkpoint /path/checkpoint_model_XXXXXX \
  --output-path /path/stage1_causal_ema_merged.pt
```

CLI 只选 `adapter_ema.safetensors`，校验 `_SUCCESS`、base SHA256、schema、step、360 keys、shape、finite values、180 modules 与 57,016,320 参数，复用 PEFT `merge_and_unload(safe_merge=True)`；合并后不得残留 `PeftModel` 或 `lora_*` key。输出 BF16 `{"generator": ...}` 与 sidecar manifest，释放模型后 fresh causal wrapper `strict=True` reload。只做结构/load smoke，不运行 `testsets`。

PEFT adapter 的“strict”不能直接把 `set_peft_model_state_dict()` 返回的大量 base-layer `missing_keys` 当失败：先从 fresh、同配置的 PeftModel 用 `get_peft_model_state_dict()` 派生 canonical expected adapter key/shape（canonical key 可能去掉 `.default`），与 safetensors 做 exact set/shape/dtype/finite 比较；加载后再次 extract 并逐 tensor 核验。

## 7. 按顺序实现

### Phase 0 — 基线与保护

- [ ] 记录 `git status --short` 和当前 branch；确认只触碰本任务文件。
- [ ] 运行现有相关 tests，记录 baseline pass/fail，不先改测试掩盖问题。
- [ ] 确认 PyTorch 2.8 环境可取得 FSDP `SHARDED_STATE_DICT`/`get_model_state_dict(... full_state_dict=False, ignore_frozen_params=True)` 及其 shard metadata；不得用 `full_state_dict=True` 后再过滤，因为那会先物化完整 5B。
- [ ] `requirements.txt` 直接声明 `safetensors`（当前仅可能是传递依赖）；Matplotlib 已有，不新增 plotting 框架。

**Gate 0：** 工作区边界清楚；可在不触发 frozen full-state collective 的前提下取得 LoRA-only shards。若做不到，停止并报告，禁止 fallback 到 full 5B gather。

### Phase 1 — 纯函数 config/schedule

- [ ] 新增窄模块 `utils/stage1_schedule.py`，从真实 `len(dataloader)`、DP、batch、accum、phase epochs 推导所有步数。
- [ ] 校验 micro-batches/accum、ramp fraction、checkpoint interval 都能得到整数；否则 fail-fast。
- [ ] 实现 stateless `values_at(update_index)`，精确覆盖 index 299/300/479/480/749。
- [ ] 在 `utils/config.py` 保持 legacy flatten 行为；新 section 优先通过 `section_get` 读取，避免把 `adapter.rank` 等污染到 root。
- [ ] 禁止 `max_iters` 与 phase-derived total 同时成为两个 source of truth；若 legacy config 仍用 `max_iters`，只走 legacy path。
- [ ] 新增 schedule/config CPU tests。

**Gate 1：** 完整数据解析出 150 updates/epoch、300 A、180 ramp、450 B、750 total、75 checkpoint interval；所有边界断言通过。

### Phase 2 — DiffSynth → native causal converter

- [ ] 新增 `scripts/convert_diffsynth_wan22_to_longlive.py`。
- [ ] 复用/抽取 `_flat_checkpoint_path` 与 Accelerate `load_checkpoint_in_model(... strict=True)`；支持单 safetensors、多 shard index 和现有 flat Diffusers directory。
- [ ] 仅加载 DiT，不加载 T5/VAE/tokenizer。
- [ ] 用 TI2V‑5B 参数构造 `CausalWanModel`；bidirectional/causal 的参数 key/shape 必须 100% 相同，因果差异只在 forward/mask/KV。
- [ ] 对 missing、unexpected、shape mismatch、残留 meta tensor 逐项 fail-fast。
- [ ] 输出 BF16 native wrapper state 与完整 hash manifest，采用临时文件 + atomic replace。
- [ ] 释放源模型，fresh causal wrapper 通过现有 `load_generator_checkpoint(... strict=True)` 重载输出。
- [ ] 新增 tiny single/multi-shard converter tests；内网再跑 full 5B conversion smoke。

**Gate 2：** full 5B 转换报告 100% key/shape coverage，输出 strict reload；原目录 hash 未变化。

### Phase 3 — 严格 CSV 解析与离线 cache

- [ ] 新增 `utils/stage1_i2v_data.py`，包含 `Stage1I2VRecord`、manifest loader、cache dataset/collate；不要继续膨胀 legacy dataset。
- [ ] CSV 用 `utf-8-sig`、稳定 row index、相对路径相对 CSV；六个已确认列必须存在，未知附加列可保留进 canonical row/hash但不参与训练语义；prompt 非空，video/input_image 存在且 video 不重复。
- [ ] 验证 image 实际尺寸与 CSV 一致，bucket 与 orientation 一致；不 stretch/crop。
- [ ] 同时验证解码后每个 video frame 的 H/W 与 CSV、input image、bucket 一致；video rotation metadata 必须为0/无，image EXIF orientation 必须为1/无。遇到 alpha、非标准 image mode 或旋转 metadata 时 fail-fast，不 silent composite/autorotate。
- [ ] 用 PyAV 按 presentation order 顺序 decode；decoded frames `>=97`、24fps（允许极小 metadata 浮点容差），只取 index0..92，多余帧忽略。禁止 fps filter、随机 seek 和尾帧 padding。
- [ ] video 以 RGB presentation order 解码，image 显式 `PIL.convert("RGB")`；两者均做 `uint8 → float32/255 → (x-0.5)/0.5` 到 `[−1,1]`，不改变 H/W。分别断言 `pixel [B,C,T,H,W] --WanVAEWrapper.encode_to_latent()--> latent [B,T,C,H,W]`；单条 safetensors 再去掉 batch 维保存 `[T,C,H,W]`。禁止 BGR、自动旋转、隐式 resize/crop；用纯色 synthetic fixture 验证 channel order/range/axis order。
- [ ] 参数化 `WanTextEncoder`/`WanVAEWrapper` 的 auxiliary model paths，同时保留 legacy 默认路径；编码仍调用现有 wrapper。给 `WanTextEncoder` 做最小的可选 `return_mask=True` 扩展，复用同一次 tokenizer call 返回 bool padding mask；mask 只用于 cache 审计/sequence-length 验证，当前 causal DiT 仍只消费 `prompt_embeds`。
- [ ] 新增 `scripts/precompute_stage1_i2v_cache.py`：每条保存原子 safetensors，支持 torchrun 按 stable row index 分片、断点续算；rank0 仅在全部 artifacts 完成后原子写 cache manifest。
- [ ] 每条缓存 `video_latent[24,48,Hl,Wl]`、`initial_latent[1,48,Hl,Wl]`、单份 `prompt_embeds[512,4096]` 为 BF16，`prompt_mask[512]` 为 bool；不复制三份 prompt。
- [ ] dataset 组装训练 `clean_latent` 时令 `clean_latent[0]=initial_latent[0]`，并保留 `initial_latent` 单独输入，确保显式图片是真正首 latent。
- [ ] manifest hash 覆盖 canonical CSV row、video/image 内容、VAE/T5/tokenizer tree、`min97 + fixed indices0..92` frame policy、resolution、dtype、prompt mode、tool/schema version。
- [ ] cache manifest 对每个 stable row id 记录 safetensors 相对路径、artifact SHA256、tensor keys/shapes/dtypes；训练启动逐项验证 cache artifact hash/schema，不能只验证源文件 hash。
- [ ] 每次训练进程启动由 rank0 仅一次流式重算并核对 raw video/image、canonical prompt row、VAE/T5/tokenizer、preprocess config 及全部 cache artifacts 的 hashes/schema，然后向其余 ranks 广播结果；其他 ranks 不重复扫描，epoch 内也不重复。默认不提供“源文件离线仍信任 cache”的 silent bypass，任一源/模型/config/cache变化都使缓存失效。
- [ ] preflight 打印每个 `(height,width)` bucket 的样本数及奇偶性，并验证能构造“warmup 区间同 shape、全 epoch 无重无漏”的 schedule；不把“每个 bucket 必须为偶数”当未经确认的协议。
- [ ] 每个 orientation 至少抽一条做 cached BF16 vs online BF16 `assert_close`，并验证 single-global embed 与三份相同 block prompt 输出等价。
- [ ] 本地 fixture 缺真实媒体，只跑 synthetic tests；完整 600 条验证必须标注为内网 Gate。

**Gate 3：** 内网 manifest 恰好 600 条，无重、无漏、无 padding/resize；每条 video=24 latent、image=1 latent，cache hashes 全部有效，总容量与估算同量级。

### Phase 4 — resolution-aware sampler、真实 epoch 与 resume cursor

- [ ] 扩展 `utils/sampler.py`：在每个 `(height,width)` bucket 内按共享 seed+epoch 无放回 shuffle，再按 `DP×batch=2` 组成 global micro-batch，切给两个 DP replicas；每 epoch 全部 600 条必须无重无漏、不 drop/oversample。
- [ ] 仅在 ER cross-DP warmup 的前 75 updates/150 micro-batches 强制两个 DP replicas 同 orientation 且样本不同。若两个 bucket 都有奇数尾项，把唯一 cross-orientation pair 确定性排到 warmup 之后；warmup 后 FSDP 允许两个 DP samples 不同 orientation，buffer shape filter 保证后续 replay 安全。
- [ ] 同一 SP group 的三个 ranks 使用相同 DP-rank sampler stream；两个 DP groups 使用不同 index。
- [ ] 在 `trainer/diffusion.py` 的 Stage‑1 cache path 保存 `train_dataloader/train_sampler`，不再包 `cycle()`；不要删除共享 `utils.dataset.cycle`，DMD 仍需要它。
- [ ] 实现显式 5-epoch loop，每 epoch 调 `sampler.set_epoch(global_epoch)`。
- [ ] DataLoader 使用专用 `torch.Generator` 与确定性 worker seed，和模型 diffusion/ER RNG 隔离；保存/恢复 loader generator state。
- [ ] checkpoint 只在 accumulation boundary 保存；保存 `global_epoch`、committed `microbatch_cursor_in_epoch`。resume 先重建 iterator并准确跳过已消费 micro-batches，再在下一次 model forward 前恢复每 rank Python/NumPy/Torch CPU/CUDA RNG，避免 iterator 构造/skip 扰动扩散噪声。
- [ ] cache path 的实际 `image_or_video_shape` 从 batch tensor 推导 H/W；canonical config 只校验 T/C。
- [ ] 启动时断言所有 allowed spatial shapes 的 patch tokens/frame 相同；本任务 `(30/2)×(52/2)=(52/2)×(30/2)=390`，因此现有 SP teacher-forcing block-mask cache 可安全复用。未来允许不同面积 bucket 时，必须把 mask cache key 扩为 `(batch,F,frame_seqlen)`。

**Gate 4：** 每 epoch 600 samples 恰好一次，epoch0/1 顺序不同且可复现；warmup micro-batches 的两个 DP samples 同 shape，奇数尾项在 warmup 后仍无重无漏；step75/300 resume 后下一条 sample/RNG 序列与约定的 canonical-buffer resume 规则一致。

### Phase 5 — Stage‑1 LoRA + 显式 FSDP 拓扑

- [ ] 扩展 `utils/lora_utils.py::configure_lora_for_model`：有 `target_patterns` 时用排序后的 full-name `re.fullmatch` exact allowlist；无 patterns 时保留 legacy 扫描，避免破坏 DMD。
- [ ] 将 safetensors adapter save/load/key validation 收敛到 `utils/lora_utils.py` 公共 helper，删除新代码中的重复 loader 分支。
- [ ] 按以下固定顺序重构 diffusion Stage‑1 初始化：找 resume metadata → strict load immutable converted base → base BF16/freeze → FSDP 前挂 LoRA → cold-start zero check 或 strict load raw LoRA → bind/verify SP forward → FSDP wrap → LoRA-only optimizer → EMA/optimizer/state resume → 按 Phase4 创建 dataloader iterator并skip committed cursor → 紧邻下一次 model forward 前恢复各 rank RNG。RNG 恢复后不得再执行会消费模型 Python/NumPy/Torch/CUDA RNG 的初始化。
- [ ] `generator_ckpt` 永远表示 base；adapter-only checkpoint 绝不能落入 raw generator load 分支。
- [ ] 参数化 `WanDiffusionWrapper/CausalWanModel` 的 architecture root/init path；训练时从 architecture config/meta 构造模型并 strict load converted base，不得静默从硬编码 `wan_models/...` 载入另一份 DiT 权重再覆盖。
- [ ] 扩展 `fsdp_wrap` 接收已经创建的 SP shard group 与 DP replicate group；`hybrid_full` 不依赖单节点自动分组。
- [ ] cache-only 且 evaluation=0 时不构造、不 FSDP-wrap T5/VAE；unconditional dict 在该 loss 中未使用，传空值而不是编码 negative prompt。
- [ ] FSDP 前断言 180 targets、57,016,320 trainable params、360 adapter tensors；FSDP 后审计 trainable names、optimizer 参数身份、无 cross-attn/非 LoRA trainable；selective gather 后在 rank0 由 key/shape 复算全局数量。
- [ ] 先做 2-GPU tiny FSDP selective adapter gather test，再做 SP3×DP2 tiny gradient parity test。

**Gate 5：** selective gather/consolidation 只 materialize 360 adapter tensors，trace/内存审计确认没有 frozen full-state collective/物化；保存后的 canonical adapter 能 strict roundtrip 到 fresh PeftModel 并逐 tensor 相等；SP3×DP2 一步参数更新与 non-SP reference 在容差内一致。

### Phase 6 — loss、phase-aware Error Recycling 与正确聚合

- [ ] 在现有 `generator_loss` 上产生 local total/block numerator+count；不要复制 Flow Matching loss。
- [ ] backward 使用 `local_numerator/global_valid_count × SP/accum`，并写 gradient parity test。
- [ ] 两个 micro-step 累积 detached numerator/count，optimizer boundary 后一次性做正确 WORLD/SP/DP reduction。
- [ ] 复用现有 first-latent mask/overwrite helper，显式测试 image latent 与 video latent0 不同时仍正确。
- [ ] `ErrorBuffer` 仍是一份 latent buffer；entry 采样增加 expected spatial shape 过滤，总容量仍按每基础 bucket 32。
- [ ] 删除 Stage‑1 noise buffer 的构造、noise residual 计算、gather、save/load 和日志；legacy config 如需 noise buffer 必须保持兼容路径。
- [ ] phase schedule 每 update 注入 model/trainer；A collect-only，B smoothstep。两个 accum micro-steps共享 schedule。
- [ ] logical gate 由 SP root 生成并广播；注入 helper 返回实际 applied blocks。
- [ ] ER residual add 先 staged；6 ranks 对 loss/grad finite flag 做 WORLD all-reduce，只有全体 finite 且 optimizer update 成功后才共同 commit buffer 与 sample cursor。任一 rank nonfinite 时全体 zero-grad/rollback。
- [ ] 修复 warmup `<75` off-by-one，首 latent 在 ER 后再覆盖。
- [ ] 对 mixed orientation、empty bucket、gate≠actual apply、save/load/resume 写 tests。

**Gate 6：** total/block counts 为 92/28/32/32；A 零注入、B 概率边界正确、noise buffer 完全不存在、横竖 residual 永不互相注入。

### Phase 7 — optimizer、LoRA-only EMA 与轻量 checkpoint

- [ ] 新增 `TrainableShardedEMA`（可放 `utils/distributed.py`），保留现有 `EMA_FSDP` 不变；只枚举 post-FSDP `requires_grad=True` 的 local LoRA shards。
- [ ] EMA CPU FP32 shadow 每个 optimizer update 后更新，不调用 `summon_full_params`。
- [ ] 保存完整 EMA adapter 时：暂存 raw local shards → copy EMA local shards → selective adapter gather → `finally` 精确恢复 raw；任何异常都不能留下 EMA 权重在训练模型中。
- [ ] raw gather、EMA swap/gather/restore、optimizer-state gather 均要求全部6 ranks以同序参与；所有 ranks在 `finally` 恢复并 barrier，只有 WORLD 成功共识后 rank0写 artifact/marker，任一 rank失败则目录不可被识别为完整 checkpoint。
- [ ] 每 rank 保存 local EMA state，固定要求相同 SP3×DP2 topology 才能 resume。
- [ ] 先用 `get_model_state_dict(... full_state_dict=False, cpu_offload=True, ignore_frozen_params=True)`/`SHARDED_STATE_DICT` 取得 LoRA-only shards，再只对 360 个 adapter tensors 做定向 consolidation。HSDP 下沿一个 authoritative SP shard group（例如 ranks0/1/2）聚合，先校验对应 DP replicas 的 adapter/EMA shard 同步，再去除 ranks3/4/5 的重复；禁止对 WORLD 朴素 all-gather。
- [ ] 若从外层 FSDP root 得到的 adapter shard keys 带 `model.` wrapper prefix，只允许按测试证明唯一的 exact prefix strip；随后用 PEFT canonical filter 并 fresh PeftModel roundtrip。禁止猜测 prefix、禁止 `full_state_dict=True` 后过滤、禁止 fallback full 5B。
- [ ] optimizer 继续复用 `FSDP.optim_state_dict/optim_state_dict_to_load`，且断言 param groups 只含 LoRA、moments FP32。
- [ ] 实现上节 checkpoint schema、`_SUCCESS`/`_RESUMABLE_SUCCESS`、atomic writes、base hash、latest resumable discovery 与 retention。
- [ ] nonfinite attempt 不 optimizer/EMA、不提交 buffer/cursor/step；缓存同一两个 micro-batches并用新 diffusion RNG 最多重试一次，第二次仍失败则终止。成功时仍只写一条 `train_step`；失败 attempt 另写 `nonfinite_attempt`，两者都带全局单调 `attempt_index`，避免相同 optimizer step 重号；任何 nonfinite residual 不得进入 buffer。

**Gate 7：** step75 raw/EMA 定义正确；save 前后 raw 参数 bitwise 恢复；checkpoint 不含 frozen 5B；step75/300 可恢复 raw LoRA/optimizer/EMA/RNG/sampler，ER 按 canonical-per-SP 规则恢复且该非-bitwise取舍在 manifest 中显式记录。

### Phase 8 — JSONL-only logger 与手动画图

- [ ] `train.py`、`trainer/__init__.py`（如其 eager import 会间接加载 W&B）与 diffusion trainer 将 W&B 改为 legacy lazy/optional import；Stage‑1 不安装或不可连接 W&B 时仍可启动，且绝不 `init/log/finish`。
- [ ] `config.disable_wandb` 与 CLI 用 OR/tri-state 合并，不能让未传 CLI flag 的默认 `False` 覆盖 YAML `true`；launcher 仍必须显式传 `--disable-wandb`。
- [ ] 新增窄模块 `utils/jsonl_logger.py`，rank0 append、flush every step、按配置 fsync；实现 run lineage 与 schema version。
- [ ] 计时和 GPU peak reset 严格放在 compute window，checkpoint/GC/logging 不得进入分母。
- [ ] 从实际 batch 计算 logical tokens，跨两个 micro-steps/6 ranks去重 SP 语义；验证 74,880/35,880/4/372。
- [ ] 新增 `scripts/plot_stage1_training.py`，Matplotlib `Agg` backend，手动读取 JSONL，容忍 truncated tail，沿 latest resume lineage 去重 step。
- [ ] 输出 PNG/SVG：total+block loss、token throughput、step time+samples、LR+grad norm、ER scheduled/realized+buffer、GPU memory+straggler；raw 低透明度 + 可配 rolling mean；标 step300 与 step480。
- [ ] 训练/checkpoint 路径中不得调用 plotter。

建议手动命令：

```bash
python scripts/plot_stage1_training.py \
  --jsonl logs/train_i2v_ar/metrics/train_metrics.jsonl \
  --output-dir logs/train_i2v_ar/metrics/plots \
  --rolling-window 20 \
  --formats png svg
```

**Gate 8：** synthetic resume/stale-suffix JSONL 能生成全部图；无 GPU、无 W&B、无 TensorBoard 依赖。

### Phase 9 — launcher、正式配置与 6-GPU dry-run

- [ ] 将 `configs/train_i2v_ar.yaml` 改为第 4 节语义，删除当前 SP4/96 latents/accum1/固定600 steps/每步save+eval/0.9 ER/大 noise buffer 设置。
- [ ] 修改 `train_stage1_i2v.sh` 为 `nproc_per_node=6`，明确 JSONL-only；路径允许通过任务专用环境变量覆盖，不复用 `$HOME` 等系统变量。
- [ ] launcher 先启动独立 dry-run 进程：加载真实 cache/base，执行 2 micro-steps + 1 optimizer update，检查 sample ids/collectives/loss/grad/EMA/memory/metrics，然后退出且不写 resumable checkpoint。
- [ ] 因正式 step1 尚未到 EMA start=75，dry-run 模式需额外强制执行一次隔离的 LoRA EMA init/update/swap-gather-restore smoke；只验证机制，不改变正式 schedule，进程退出后全部丢弃。
- [ ] dry-run 成功后由第二个全新 `torchrun` 进程正式从 step0 启动；不能在同一 trainer 内“手工清零”继续。
- [ ] dry-run 使用与正式训练完全不同的 logdir/output dir、`--no-auto-resume`、`--no-save`；正式首次运行目标目录必须干净，dry-run 不得留下任何能被正式 auto-resume 发现的标记或 checkpoint。
- [ ] dry-run 断言：同 SP group sample id相同、两个 DP sample不同但 orientation相同；loss/grad有限；只写一条 dry-run metric；tokens=74,880/35,880；无 NCCL hang/OOM。正式 schedule 中 EMA 应仍为空；另行强制的 EMA smoke 必须标记为 dry-run-only component check。
- [ ] 正式路径 evaluation interval=0，checkpoint 每75 completed steps。

launcher 的等价调用应清晰分成两个进程；具体参数名可以遵循现有 CLI，但语义必须等价：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=6 train.py \
  --config_path configs/train_i2v_ar.yaml \
  --logdir /dedicated/dry_run_dir \
  --stage1-dry-run-one-update --no-save --no-visualize \
  --disable-wandb --no-auto-resume

torchrun --standalone --nnodes=1 --nproc_per_node=6 train.py \
  --config_path configs/train_i2v_ar.yaml \
  --logdir /formal/train_i2v_ar \
  --no-visualize --disable-wandb
```

**Gate 9：** 6×H100 dry-run 全通过，进程退出后正式 run 第一条记录是 `update_index=0/optimizer_step=1`，没有 dry-run 状态污染。

### Phase 10 — EMA merge 与文档收尾

- [ ] 扩展现有 `scripts/merge_lora_generator.py` 接受 training checkpoint directory 与 safetensors EMA；不要另写重复 merge engine。
- [ ] 复用 `load_generator_checkpoint(strict=True)`、exact LoRA config、PEFT safe merge、`cpu_state_dict`。
- [ ] 实现 base/checkpoint hash 与 adapter strict validation、BF16 output、manifest、fresh strict reload。
- [ ] 写 merge 单测：验证 `W'=W+(alpha/r)BA`、只选 EMA、base mismatch/NaN/missing/extra key 失败、merge 后无 LoRA key。
- [ ] 更新 README 或新增简短 Stage‑1 runbook，只写真实命令、输入/输出、resume、plot、merge；不宣称视觉效果。
- [ ] 再次运行全量相关 tests，核对 `git diff --check` 与 `git status --short`，确认未碰用户既有删除/数据。

**Gate 10：** 用户选定的 EMA adapter 可生成 `stage1_causal_ema_merged.pt`，现有 causal loader strict reload；没有自动运行 `testsets`。

## 8. 测试矩阵与建议命令

### 8.1 复用/扩展现有 tests

- `tests/test_i2v_dataset_frame_accounting.py`
  - 93 raw →24 latent→3×8；禁止 padding。
- `tests/test_i2v_sequence_parallel_config.py`
  - 24/SP3/block8 通过；错误 world/SP/block 失败。
- `tests/test_distributed_sampler_seed.py`
  - bucket 内无放回、warmup 两 DP 同 orientation、奇数尾项延后并跨 orientation 配对、epoch 切换、全量无重无漏。
- `tests/test_i2v_teacher_forcing_context.py`
  - explicit image != video latent0；首 latent clean/t0/masked；SP valid=7/8/8。
- `tests/test_inference_prompt_batching.py`
  - single global cached embed 与 repeat three blocks 等价。

### 8.2 新增 tests

- `tests/test_stage1_schedule.py`
- `tests/test_stage1_i2v_manifest.py`
- `tests/test_stage1_i2v_cache.py`
- `tests/test_stage1_i2v_cached_sp.py`
- `tests/test_stage1_epoch_resume.py`
- `tests/test_error_buffer_resolution.py`
- `tests/test_stage1_loss_metrics.py`
- `tests/test_lora_utils.py`
- `tests/test_trainable_ema.py`
- `tests/test_stage1_lora_checkpoint.py`
- `tests/test_diffsynth_causal_converter.py`
- `tests/test_merge_lora_generator.py`
- `tests/test_jsonl_training_plot.py`

按阶段运行，不等全部实现后一次排错：

```bash
pytest -q tests/test_stage1_schedule.py tests/test_i2v_dataset_frame_accounting.py tests/test_i2v_sequence_parallel_config.py
pytest -q tests/test_stage1_i2v_manifest.py tests/test_stage1_i2v_cache.py tests/test_distributed_sampler_seed.py
pytest -q tests/test_i2v_teacher_forcing_context.py tests/test_error_buffer_resolution.py tests/test_stage1_loss_metrics.py
pytest -q tests/test_lora_utils.py tests/test_trainable_ema.py tests/test_stage1_lora_checkpoint.py
pytest -q tests/test_diffsynth_causal_converter.py tests/test_merge_lora_generator.py tests/test_jsonl_training_plot.py
```

分布式 tests 必须另有明确入口：

1. CPU/Gloo numerator-count reduction test。
2. 2-GPU FSDP selective adapter gather/save/resume。
3. 3-rank SP gradient parity（最小拓扑）。
4. 6-rank SP3×DP2 gradient parity 与真实 5B dry-run。

## 9. 风险登记与停止条件

| 优先级 | 风险 | 必须的控制 |
|---|---|---|
| P0 | FSDP 自动 group 与逻辑 SP/DP 不一致 | 显式 shard/replicate process groups + parity test |
| P0 | SP contribution 被 FSDP 多平均一次，梯度小3倍 | backward `×SP` + reference parity |
| P0 | mixed orientation 在 ER DP gather 中静默 H/W 错读 | resolution-aware global micro-batch + buffer shape filter |
| P0 | adapter checkpoint 先 gather 完整5B | selective state API；无可用 API 时停止，不 fallback |
| P0 | full-model EMA 每步 summon 5B | trainable local-shard EMA；保留 legacy EMA 不调用 |
| P0 | `cycle()` 不 set_epoch / resume 重复样本 | 显式 epoch、sampler epoch、microbatch cursor |
| P1 | phase/ramp/warmup off-by-one | 0-based update index tests，warmup严格 `<75` |
| P1 | SP ranks 对同一视频使用不同 ER gate | SP-root sample gate broadcast |
| P1 | 日志只记录最后 micro-step/local rank | numerator/count across accum×SP×DP |
| P1 | artifact-only checkpoint 被误当 resumable | `_SUCCESS` 与 `_RESUMABLE_SUCCESS` 分离，latest 只认后者及完整文件 |
| P1 | JSONL 比 checkpoint 多出 stale suffix | append-only lineage，plotter child override |
| P1 | BF16 LoRA 导致 Adam moments BF16 | FP32 adapter master + BF16 FSDP forward |
| P2 | `no_sync()` 与当前 FSDP 版本不兼容 | correctness-first，可保留同步并记录吞吐影响 |
| P2 | 本地 CSV 只有1条且无媒体 | synthetic local tests；600/5B/6GPU 只在内网宣称通过 |

立即停止并报告，不自行绕过的条件：

- converted base key/shape/hash 不一致；
- 缓存不是完整 600 条或任意 source/model/config hash 失效；
- 无法构造前150个 micro-batches 同 shape、同时全 epoch 600条无重无漏的 DP2 schedule；
- LoRA 不是 180 modules / 57,016,320 params / 360 tensors；
- selective gather 仍 materialize frozen base；
- SP/DP/FSDP gradient parity 失败；
- 6-GPU dry-run OOM、NCCL hang、非有限 loss/grad、token count 不符；
- resume 后下一 sample/schedule 与连续 run 不一致；
- merge 后仍有 PEFT/LoRA key或 strict reload 失败。

## 10. 最终交付清单

- [ ] 严格 DiffSynth→LongLive causal base converter + manifest。
- [ ] 严格 CSV parser、deterministic precompute CLI、BF16 cache + manifest。
- [ ] resolution-aware DP sampler、真实 epoch、精确 mid-epoch resume。
- [ ] exact rank32 LoRA、显式 SP3×DP2 FSDP、正确梯度缩放。
- [ ] 两阶段 5-epoch/750-update schedule 与 30%/10%/0% ER。
- [ ] LoRA-only FP32 EMA、adapter-only checkpoint、atomic resume/retention。
- [ ] JSONL global loss/token throughput logger。
- [ ] 手动 PNG/SVG plotting CLI。
- [ ] 6×H100 isolated dry-run launcher + 正式训练 launcher。
- [ ] 用户选择后运行的 EMA merge CLI，输出 native BF16 causal checkpoint。
- [ ] 全套 CPU/distributed tests 与简短 runbook。
- [ ] 内网H100 * 6卡环境快速部署、测试、启动训练的指导文档(中文简洁)
- [ ] 明确声明：未替用户运行视觉验证，未自动使用 `testsets/`。
