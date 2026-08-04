# Stage‑1 双动作 Continuation Inference 实现任务

> 状态：本地实现、单元测试与静态验收已完成；等待用户在内网 H100 执行 Step 10
>
> 确认日期：2026‑08‑04
>
> 真实推理环境：内网单张 H100；本地只完成代码、单元测试和静态验证
>
> 本文是本任务的实现规格。实现者不得用未记录的新假设替换已确认决策。

## 0. 执行纪律

- [x] 开始前完整阅读本文、`git status --short`、现有 `infer_stage1_two_actions_10s.sh`、`inference.py`、`pipeline/causal_diffusion_inference.py`、`wan_5b/modules/causal_model.py`、Stage‑1 checkpoint validation/preparation helpers 及相关测试。
- [x] 保留用户当前工作区中的所有既有修改、删除和未跟踪文件；不要恢复、覆盖或顺手整理无关内容。
- [x] 保留现有 `infer_stage1_two_actions_10s.sh` 的 uniform-prompt 语义，新增 continuation 专用入口；不得把旧实验静默改成新实验。
- [x] 复用现有 checkpoint 3750 EMA merge、模型加载、技术门禁和视频保存逻辑；不要复制第二套大体相同的 merge/checkpoint 系统。
- [x] 普通 `pipeline.inference()` 必须继续表示“独立新视频”，默认重置状态；只有显式 continuation session 可以跨调用保留状态，防止不同样本串缓存。
- [x] 所有 continuation 不变量 fail-fast：batch、shape、dtype、device、block 对齐、cursor、cache index、sink、CFG、prompt 数量或 session 生命周期不一致时必须报错，禁止静默清 KV 后继续。
- [x] 不提交、不推送、不删除用户文件，不在无 H100/checkpoint 的本地环境伪造视觉推理成功。
- [x] 每完成一个最小步骤，先运行该步骤的验证并在本文勾选；若代码事实与本文冲突且会改变方案，停止并恢复一问一答。

## 1. 目标与完成结果

把当前一次性 64-latent uniform-prompt 实验改造成与真实部署一致的单 GPU、单样本、有状态 continuation inference：

```text
动作 A：24 latent（3 blocks）
HOLD：  16 latent（2 blocks；API 支持任意 N×8）
动作 B：24 latent（3 blocks；首 latent 为 carry anchor）
总计：  64 latent → 连续 VAE 解码 → 253 pixel frames @ 24fps
```

同一 session 中：

- positive/negative self-attention KV Cache 不清空并按现有 24-latent window 滚动；
- global latent cursor 与 temporal RoPE 单调递增；
- A、HOLD、B 分别加载新的文本 conditioning，B 的自然语言事件时钟重新从 0 秒开始；
- B 复用 HOLD 最后一个 clean latent 作为 B 的第一个 `t=0` latent，再生成其余 23 latent；
- 所有 64 latent 累积完成后只做一次连续 VAE decode，不分段解码再拼接。

最终运行固定产生：

- 4 只猫 × 2 个动作顺序 × 2 个 sink 变体 = 16 条视频；
- `sink_size=0` 与 `sink_size=1` 使用完全相同的 seed/noise；
- 一个 8 行 × 2 列的人工对比 HTML，支持同一行同步播放、暂停和拖动；
- 可审计的 session trace/report，但不自动评分、不推荐 sink 胜者。

## 2. 已确认决策

| 项目 | 锁定值 |
|---|---|
| checkpoint | `checkpoint_model_003750` 的 EMA |
| GPU / batch | 单 GPU；`batch_size=1` |
| solver / sampling / CFG | UniPC / 50 steps / 5.0 |
| seed | 仅 `1` |
| negative prompt | 原有默认值，不修改 |
| AR block | 8 latent frames |
| self-KV window | 保持现有 24 latent rolling cache |
| local-attention 配置 | 保持现有 `local_attn_size=-1`；运行时必须审计其当前实际行为仍是 24-latent cache/window，不把它改成新的消融变量 |
| session schedule | `A×3 → HOLD×2 → B×3` |
| 部署 HOLD 长度 | 任意 `N×8`；本次 `N=2` |
| B 起点 | HOLD 最后一个 clean latent 作为 B 第一个 `t=0` anchor；B 总长度仍为 24 |
| 时钟 | 只重置 B prompt 的文本事件时钟；不重置 self-KV、global cursor 或 RoPE |
| sink variants | `sink_size=0`、`sink_size=1` 都运行；不自动选优 |
| 动作顺序 | `jump_then_toy`、`toy_then_jump` |
| 猫咪 | ragdoll、russian_forest、siamese、tabby |
| prompt | A/HOLD/B 都自包含身份、场景、机位、构图约束 |
| HOLD 语义 | 标准稳定坐姿；仅自然轻微呼吸和轻微耳动；不主动写眨眼或任一动作 |
| decode | 分段生成 latent，整段统一 decode |
| 输出 | 64 latent / 253 pixel frames / 24fps |
| 人工判断 | 用户查看 HTML；代码不评分、不排名、不替用户选择 sink |

## 3. 已核实的代码与训练事实

### 3.1 当前脚本没有模拟部署式 prompt 切换

当前 `infer_stage1_two_actions_10s.sh` 生成 64 latent，但 preparation 写入 `uniform_prompt=true`，最终八个 block 使用同一篇完整双动作 prompt：

```text
[AB, AB, AB, AB, AB, AB, AB, AB]
```

目标实验必须实际执行：

```text
[A, A, A, HOLD, HOLD, B, B, B]
```

### 3.2 当前多次调用并不会 continuation

`CausalDiffusionInferencePipeline.inference()` 在第二次顶层调用时会把 self-KV 的 `global_end_index`/`local_end_index` 归零。对象仍持有 tensor 不代表缓存语义被延续。`start_frame_index` 也不能单独修复这一点；非零 cursor 配空 cache 可能导致索引错误。

因此必须增加显式 session/state API，不能简单连续调用三次现有 `inference()`。

### 3.3 训练窗口与 prompt 分布

- Stage‑1 固定训练 24 latent、每 block 8 latent；三个 block 共用一份 `[512,4096]` prompt embedding。
- event 秒数只是 T5 自然语言，没有结构化 event tensor 或 local clock。
- 第一个 latent 是显式 input-image latent、`t=0` 且不计 loss；有效生成 target 是后续 23 latent。
- 训练使用整段 teacher-forcing + block-causal mask，不使用 rolling KV cache。
- 配置没有显式 `sink_size`，模型默认是 0；但训练前向没有 rolling cache，因此 sink/eviction 机制实际未参与训练。`sink=0/1` 均为推理消融。
- 当前推理的 `local_attn_size=-1` 很反直觉：pipeline 在该分支分配 `3 × num_frame_per_block = 24` latent 的 cache，`CausalWanSelfAttention` 构造时也把 `-1` 映射为有效 local size 24。新实现必须用运行时断言验证 cache capacity=`24×frame_seq_length` 和各 attention block 的有效 rolling 行为；不要仅凭配置字面值把它误判成无限历史，也不要未经消融授权改写原配置。

### 3.4 16-latent HOLD 的 cache 含义

默认 cache window `W=24`、当前 block `B=8`。要让动作 B 的首 block attention 完全看不到动作 A，最短对齐等待长度是：

```text
W - B = 24 - 8 = 16 latent
```

在 B 首 block 写入时，活动上下文应为 HOLD 16 + 当前 B block 8；A 已完全淘汰。`sink=1` 时另外保留最初输入图 latent，并相应使用最近 15 个 HOLD latent；仍不得残留 A 动作 latent。

### 3.5 连续帧账

64 latent 连续解码为：

```text
1 + (64 - 1) × 4 = 253 pixel frames
```

边界用于报告和人工检查：

| 段 | latent | pixel frames | 说明 |
|---|---:|---:|---|
| A | 0..23 | 0..92（93 帧） | 首段含整个视频的唯一初始 pixel frame |
| HOLD | 24..39 | 93..156（64 帧） | 约 2.667 秒 |
| B | 40..63 | 157..252（96 帧） | B 的文本局部 0 秒从 frame 157 开始 |

不能把 24/16/24 分别 decode 后拼接；这会改变总帧数并清空 VAE temporal context，产生边界伪影。

## 4. 新 metadata 契约

文件已随本文创建：

`testsets/metadata_8cases_two_actions_continuation_480x832_253frames.csv`

列：

```text
input_image,action_a_prompt,hold_prompt,action_b_prompt,height,width,bucket,
case_group,cat_id,action_order,action_a_blocks,hold_blocks,action_b_blocks,soft_reanchor
```

严格约束：

- header 必须与上述 14 列完全一致；拒绝未知列、重复列、缺列和多余未命名字段。
- 恰好 8 行、4 个 cat、每只猫两个 action order；`case_group=<cat_id>_<action_order>` 且唯一。
- 每张 input image 必须恰好复用两次（对应两个 action order）。该规则只属于新的 continuation loader，不能放宽旧 loader 默认拒绝重复图片的门禁。
- `action_a_blocks=3`、`hold_blocks=2`、`action_b_blocks=3`，总和必须为 8。
- `soft_reanchor=true` 使用 canonical lowercase boolean；不得宽松解析任意字符串。
- 每个 prompt 非空且独立自包含身份、白背景、静止机位、全景居中、完整身体约束。
- A/B 只含各自一个动作，并使用局部 `0-1 / 1-3 / 3-4秒`。
- HOLD 不出现 `眨眼`，也不包含跳跃、逗猫棒、扑抓等动作语义；只描述稳定坐姿、轻微呼吸和轻微耳动。禁词校验只能匹配这些明确动作词，不能粗暴禁止合法短语“耳朵轻微动作”中的通用词“动作”。
- sink 不是 metadata 维度；runner 为每行运行 0/1 两个 sink，不能复制成 16 行。
- loader 保持图片存在、RGB、EXIF、尺寸、orientation/bucket 等现有严格校验。

在 H100 实际 tokenizer 上对 A/HOLD/B 分别检查 token count；必须先使用同一个 tokenizer、同一文本清洗和 special-token 设置，以 `truncation=False`（或等价未截断方式）取得真实长度，再进行正式的 512-token 编码。不能根据已经截断/填充后的 attention mask 反推“未截断”。任何 prompt 达到 512-token truncation 边界必须 fail-fast，禁止静默截断。

## 5. Continuation session 行为契约

具体类名可按现有代码结构选择，但外部语义必须等价于：

```python
# runner 针对一个 semantic case 只生成一次；sink0/1 各使用 clone/view。
noise_plan = make_noise(seed=1, shape=[1, 64, 48, latent_h, latent_w])

session = pipeline.begin_session(
    initial_latent=initial_latent.clone(),
    sink_size=sink_size,
)

session.generate_segment(
    action_a_prompt,
    noise=noise_plan[:, 0:24].clone(),
)
session.generate_segment(
    hold_prompt,
    noise=noise_plan[:, 24:40].clone(),
)
session.generate_segment(
    action_b_prompt,
    noise=noise_plan[:, 40:64].clone(),
    carry_last_latent_as_anchor=True,
)

result = session.finish()  # concatenate session latent, decode once
assert result.latents.shape[1] == 64  # 本实验 runner 的责任
```

### 5.1 Session 必须持有或拥有的状态

- positive/negative self-attention KV caches；
- text conditioning 状态：A/HOLD/B 正向文本各编码一次并在各自 3/2/3 blocks 复用；全局 negative prompt 编码一次并保持不变；
- global latent cursor 和与 cache index 一致的 block cursor；
- session-locked batch=1、resolution、latent channels、dtype、device、CFG、negative prompt、block size、sink 和 RoPE 配置；
- deterministic RNG/noise plan 与已经消费的全局 slice；正式 runner 不允许 segment 内部临时重新采样；
- 已生成 latent 列表；
- lifecycle：active / finished / failed（失败后不可继续复用）。

当前 cross-attention 实现明确 bypass cache、每次重算 text K/V；本任务不得顺手重新启用该缓存。prompt 切换测试应验证实际传入的 prompt embedding/conditioning 已从 A 变为 HOLD、再变为 B，而不是断言 `crossattn_cache.is_init` 或缓存 K/V 发生变化。无论文本路径是否缓存，self-KV 都不能因 prompt 切换重置。

普通 inference 与 session state 必须隔离。最安全的实现是 session 自己拥有 cache/state，或 pipeline 明确禁止并发 active session；不能让下一条独立视频继承上一条 session 的缓存。session finish/fail 并清理后，必须能够从干净状态开启下一 session 或运行普通 inference。

### 5.2 Block 与 cache 更新

- 每次 continuation segment 的总长度必须按 8 对齐；本任务的 B 是唯一例外语义：长度 24 仍拆为三个 8-frame blocks，只是第一个 block 的第一个位置被 carry anchor clamp。
- 每个 block 完成 denoising 后必须用 clean latent、`t=0` 正常 recache，positive/negative caches 同步更新。
- 每个 block 都必须创建独立的 50-step UniPC scheduler，保持现有行为；不能把一个 scheduler 的 multistep state 跨 block 或跨 segment 复用。
- 预期 global latent cursor：A 后 24、HOLD 后 40、B 后 64。
- 设每 latent frame 的 patch-token 数为 `S=frame_seq_length`。raw self-KV index 使用 token 单位，因此每 block 的 `global_end_index` 必须是 `S×[8,16,24,32,40,48,56,64]`，不能直接与 frame cursor 混用。
- raw 24-window `local_end_index` 必须是 `S×[8,16,24,24,24,24,24,24]`。session trace 可以额外记录除以 `S` 后的 frame-normalized 值，但必须同时保留或校验 raw token index。
- 任何 continuation 调用开始前，必须断言 session cursor 与所有 transformer block 的 cache indices 一致。

### 5.3 动作 A 的初始 anchor

- global latent 0 必须等于 session 的 explicit `initial_latent`，并作为 A 输出 24 latent 中的第一个位置；不能把它作为输出之外的额外 context，否则会产生 65 latent 或错误 cursor。
- A 首 block 的第一个位置在所有 denoising steps 中保持 clean、timestep=0；scheduler step 后再次 overwrite，最终用 clean block recache。
- A 总长度仍为 24：initial anchor 1 + 新生成 23。noise plan 的 global index 0 必须存在并保持两个 sink 变体的索引/RNG协议一致，但其内容会被 clamp 覆盖。

### 5.4 Soft re-anchor 精确定义

- HOLD 最后一个 clean latent（global latent 39）作为 B 输出位置 40 的内容；两个位置内容可相同，但位置与 RoPE 仍分别是 39、40。
- B 首 block 的第一个 latent 在每个 sampling step 的模型调用前都必须 overwrite 为 anchor 并把 timestep 置 0；scheduler step 后再次 overwrite，最后用 clean block recache。该 block 其余 7 个 latent 正常 denoise，后两个 blocks 各生成 8 个，共生成 23 个新 latent。
- 不清 self-KV，不把 global cursor/RoPE 重置为 0，不把 B 当成新的独立图片视频。
- B prompt 文本写“本阶段从当前画面重新以 0 秒计时”；这只影响 T5 conditioning。

### 5.5 Sink variants

- 每个 metadata case 分别开启全新的 `sink=0` 和 `sink=1` session；input image 只 encode 一次，两个变体分别使用其 clone。
- `sink=1` 必须通过 legacy leading sink 永久保留最初输入图 latent，而不是 A/HOLD 边界 latent；`multi_shot_sink=false`、`global_sink_size=0`，所有 block 的 `pinned_start` 必须始终为 `-1`。
- runner 用独立 `torch.Generator(...).manual_seed(1)` 为每个 semantic case 一次生成 contiguous `[1,64,48,latent_h,latent_w]` noise plan。两个变体必须使用该 tensor 的 clone/view，并严格切为 `[0:24]`、`[24:40]`、`[40:64]`；initial/soft-anchor 对应的 noise 槽位仍存在但被 clamp。
- noise identity hash 必须包含 contiguous CPU tensor 的 dtype、shape 和原始 bytes；不能只记录 seed 字符串。
- 不启用 relative RoPE、scene-cut prefix、multi-shot sink 或其他未确认消融。

### 5.6 生命周期与失败恢复

- batch>1、非 8 对齐 continuation、空 session 上 soft-anchor、中途改变 shape/dtype/device/sink/CFG、finish 后继续写入均 fail-fast。
- block 生成中发生异常时，session 标记 failed/poisoned 并释放或隔离缓存；禁止在可能半写入的 KV 上继续。runner 必须原子写出 partial failure trace，不能只依赖保留 merged checkpoint。
- 通用 session 的 `finish()` 只验证非空、block 对齐、cache/cursor 一致并统一 decode，不能硬编码总长 64；本实验 runner 负责断言总长恰好 64。`finish()` 返回 result，不直接决定 MP4/JSON 路径；runner 负责原子保存产物并关闭/清理 session。

## 6. Runner、shell 与输出

新增独立入口，建议命名：

`infer_stage1_two_actions_continuation_10s.sh`

它应继续沿用现有 `LONG_LIVE_STAGE1_*` 路径变量和空 work-dir 防覆盖契约；新 work-dir 环境变量固定命名为 `LONG_LIVE_STAGE1_CONTINUATION_WORK_DIR`，同时允许唯一的位置参数覆盖，并固定：

```text
checkpoint_model_003750 EMA
metadata_8cases_two_actions_continuation_480x832_253frames.csv
seed=1, UniPC=50, CFG=5.0
sink variants=0,1
CUDA_VISIBLE_DEVICES 默认 0
```

模型 bootstrap 的集成方式也锁定：保留 `inference.py` 的普通路径，在其现有 config/model/checkpoint/device/dtype 初始化完成后，通过显式 continuation config/manifest 分支调用一个小型 continuation orchestration module。新 runner 仍启动该入口；如需可测试性，可把现有 bootstrap 抽成 `utils/inference_utils.py` 中由两个路径共同调用的 helper。禁止把 `inference.py` 约 500 行的模型加载、LoRA/量化或 device placement 复制进新 runner。

Python runner 应复用现有 checkpoint validation/merge helpers，不得复制 checkpoint merge 实现。建议产物至少包含：

```text
validation_report.json
comparison.html
checkpoint_model_003750/
├── merge_manifest.json
├── output_validation_report.json
└── continuation/
    ├── <case_group>/sink0.mp4
    ├── <case_group>/sink0.session.json
    ├── <case_group>/sink1.mp4
    └── <case_group>/sink1.session.json
```

每份 session trace 至少记录：metadata row/case、prompt SHA256 与未截断 token count、block schedule、sink、seed/noise identity、soft-anchor source/destination、每 block raw token-unit global/local cache end、换算后的 latent-frame cursor、latent/pixel frame边界、solver/steps/CFG/negative prompt hash、输出路径和技术门禁结果。

技术输出门禁继续复用旧 Stage‑1 validation 的首帧 PSNR、mean frame std、temporal absolute difference、帧数/fps/分辨率和缺失/额外 MP4 检查及原阈值；它们只做技术有效性判断，不能转化成视觉评分。若旧 validator 的目录协议不适配新矩阵，应抽取/复用指标函数，而不是降低或删除门禁。

成功后可沿用旧 runner 删除可重建 merged BF16 checkpoint；失败时保留现场。不得删除旧 work dir 或覆盖已有视频。

## 7. 人工对比 HTML

- 8 行：每个 `<cat_id>_<action_order>` 一行。
- 2 列：`sink=0`、`sink=1`，严格用 report/row id 映射，不能按 glob 顺序猜。
- 同一行支持同步播放、暂停、拖动和重新归零；单个视频仍保留原生 controls。
- 显示 cat、动作顺序、A/HOLD/B prompt、seed、sink、soft re-anchor、A/HOLD/B frame 边界。
- 链接相对 work dir，目录整体拷贝后仍能打开。
- 不计算视觉分数，不显示 winner/recommendation，不自动选择 sink。

人工查看重点仅供展示，不作为自动判定：两个动作是否各出现一次、顺序是否正确、B 是否在 HOLD 后开始、身份/毛色/肢体是否稳定、HOLD 是否只有低幅自然运动、两个边界是否自然。

## 8. 验收标准

### 8.1 本地代码验收

- [x] 旧 shell、旧 metadata 与旧 uniform-prompt 结果语义不变。
- [x] 新 metadata 严格解析为 8 行、4 cats × 2 orders，schedule 为 3/2/3，soft re-anchor 为 true。
- [x] 普通 `pipeline.inference()` 仍隔离样本并重置；continuation session 才保留 cache。
- [x] mock/tiny pipeline 证明三次 API 调用后的 cursor 是 24/40/64，cache global/local end 完全符合 5.2。
- [x] prompt 从 A→HOLD→B 切换时 self-KV tensor/index 不归零；实际传入的 positive prompt embedding 确实切换，negative embedding 保持同一份；不要求当前被 bypass 的 cross-attn cache K/V 变化。
- [x] A latent 0 等于 initial latent、全程 t=0 clamp，A 总计仅生成 23 个新 latent且 cursor 仍为24。
- [x] B latent 40 等于 HOLD latent 39，B 只有首位置 t=0 clamp，其余 23 正常生成。
- [x] sink0/1 使用 bitwise identical input noise 和同一 initial latent 的 clones；sink1 在滚动后仍保留最初输入图 cache K/V，且 `pinned_start=-1`。
- [x] 累计 latent shape 为 `[1,64,48,height//16,width//16]`，统一 decode 期望 253 帧。
- [x] session 生命周期与所有 mismatch/错误路径 fail-fast；失败 session 不可继续。
- [x] session finish/fail 清理后可以从干净状态开启下一 session 或普通 inference；失败 runner 原子留下 partial trace。
- [x] comparison HTML 为 8×2 且同步 controls 可由 DOM 单测验证。
- [x] 原有 inference、prompt batching、Stage‑1 checkpoint validation 测试全部通过。
- [x] `bash -n`、CLI `--help`、`git diff --check` 通过。

### 8.2 内网 H100 技术验收

- [ ] checkpoint 只使用 step 3750 EMA；16 条视频全部存在，无额外/缺失输出。
- [ ] 每条视频恰好 253 帧、24fps、分辨率与输入图片一致。
- [ ] 每份 session trace 证明 global cursor 0→64，self-KV 未在 A/HOLD/B 边界清空，sink 与 soft-anchor 契约正确。
- [ ] prompt 均未触发 512-token truncation；无 NaN/Inf。
- [ ] sink0/1 同 case 的 noise identity 相同。
- [ ] HTML 可直接打开、两列映射正确、同步播放和拖动可用。
- [ ] 视觉效果由用户人工查看；技术报告不得宣称某个 sink 更优。

## 9. 最小可验证执行计划

- [x] **Step 1 — 回归与 characterization 基线。** 结果：先用测试锁住普通 `inference()` 的签名/调用方式、连续两次普通 inference 均从 cursor/cache 0 开始、首图 clamp、block prompt 映射、clean recache 调用序列，以及每 block 独立初始化 50-step UniPC scheduler；再记录现有 inference/checkpoint validation/prompt batching 测试状态。未建立基线前不要开始重构。
- [x] **Step 2 — 严格 metadata loader。** 结果：解析新 8-row schema，验证图片、case matrix、3/2/3 blocks 和 canonical boolean。区域：优先扩展现有 validation helper或新增小型 continuation metadata module。验证：正例 + 缺列、重复 case、错误 order、错误 block sum、非法 boolean、HOLD 禁止词等反例。
- [x] **Step 3 — 抽取可复用 block-generation 内核。** 结果：现有一次性 inference 与 session 共用同一 denoise/clean-recache 逻辑；A/HOLD/B 各编码一次并按 3/2/3 blocks 复用 embedding，negative 只编码一次。区域：`pipeline/causal_diffusion_inference.py`。验证：Step 1 characterization 保持不变；segment block 数与 embedding 复用映射精确。
- [x] **Step 4 — 实现 session/state 生命周期。** 结果：begin/continue/finish 保留 pos/neg self-KV 与 cursor，普通 inference 仍隔离。区域：pipeline 或一个紧邻的新 session module。验证：24→16→24 三调用 cache trace；batch/shape/lifecycle 错误测试。
- [x] **Step 5 — 实现 B soft re-anchor。** 结果：carry latent 作为 B 首位置 t=0，旧 KV 与 global RoPE 连续。验证：anchor equality、timestep mask、23 个新 latent、cursor/cache index 单测。
- [x] **Step 6 — 实现 sink/noise 矩阵。** 结果：每 case 两个独立 session，sink0/1 同 noise，sink1 只 pin 原始首图。验证：mock KV 内容和 noise hash。
- [x] **Step 7 — 统一 decode、trace 和技术门禁。** 结果：64 latent 一次 decode，报告包含边界/cache/prompt审计。验证：shape/frame policy、失败保留现场、无分段 decode 调用。
- [x] **Step 8 — 新 runner、shell 与 8×2 HTML。** 结果：一个命令生成16视频和同步对比页，不改旧入口。验证：runner orchestration mock、DOM/HTML断言、`bash -n`、CLI help。
- [x] **Step 9 — 整体本地验证。** 至少运行 continuation 新测试、相关原有回归、`git diff --check`；不得在本地标记 H100 视觉验收完成。
- [ ] **Step 10 — 用户内网 H100 运行。** 用户使用新的空 work dir 执行；按 8.2 检查技术产物，然后人工查看 HTML。

## 10. 非目标与延期项

- 不修改或继续训练 Stage‑1 权重，不改 LoRA/EMA/merge 语义。
- 不支持 sequence parallel、Ulysses 或多 GPU continuation。
- 不支持 batch>1、多并发 session 或 server 调度。
- 不实现在线 streaming VAE；本轮必须最后统一解码。
- 不测试 seed 2/3，不自动重试/挑 seed。
- 不新增 relative RoPE、scene-cut、多-shot sink、KV quant 或其他消融。
- 不把 no-reanchor、清 KV + 新 I2V、复制 HOLD 尾帧或插帧加入正式矩阵。
- 不自动评分、动作识别或决定 sink0/1 胜者。

## 11. 已知风险

- 权重只见过 24 latent、单全局 prompt 和 teacher-forcing；prompt switch、rolling eviction、HOLD、soft re-anchor 都是推理分布外策略。
- `sink=1` 可能改善身份，也可能抑制动作或把姿态拉回首图；本任务只并排呈现。
- 文本里的局部 0 秒不是结构化时钟；动作 B 的时间遵循程度必须人工观察。
- soft re-anchor 会在相邻 global positions 39/40 放置内容相同的 latent，可能出现极短停顿；这是已接受的训练分布对齐权衡。
- 64-latent full VAE decode 比 24 latent 占用更多显存；OOM 时保留现场并报告，未经确认不要自动切 streaming/offload/量化。
