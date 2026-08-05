# Stage-1 3750-step EMA：10.54 秒双动作推理对比任务

> 状态：访谈已确认，等待 Codex / GPT-5.6-sol 实现  
> 确认日期：2026-08-03  
> 目标环境：本地完成代码与单元测试；用户随后在内网单机 H100 上运行真实推理  
> 本文是本任务的实现规格。实现者不得用未记录的新假设替换已确认决策。

## 0. 执行纪律

- [ ] 开始前完整阅读本文、`git status --short`、现有 `infer_stage1.sh`、`scripts/run_stage1_training_checkpoints_validation.py`、`utils/stage1_causal_validation.py` 及对应测试。
- [ ] 保留用户当前工作区，不恢复、覆盖或顺手修改既有变更。特别不要覆盖根目录现有未跟踪 `TASK.md`，也不要修改 `testsets/metadata_6cases_480x832.csv`。
- [ ] 复用现有 EMA merge、testset preparation、`inference.py`、输出技术校验和 HTML 生成链路；不要复制出第二套大体相同的 checkpoint merge/inference runner。
- [ ] 只做支持本任务所需的最小通用扩展，所有现有 24-latent/93-frame checkpoint 验证默认行为必须保持不变。
- [ ] 新的重复首帧能力必须显式 opt-in；默认仍拒绝 metadata 中重复 `input_image`，避免放宽旧验证门禁。
- [ ] 不提交、不推送、不删除用户文件，不在本地伪造 H100 推理成功。真正的视频效果只由用户在内网 H100 上判断。
- [ ] 每完成一个最小步骤先运行对应验证，再勾选。发现当前实现与本文事实冲突且会改变结论时，停止并恢复一问一答，不得静默猜测。

## 1. 目标与成功结果

为人工筛选出的最佳 Stage-1 EMA checkpoint `checkpoint_model_003750` 增加一套可重复的长时双动作推理对比流程：

1. 使用 4 个已有猫咪首帧，分别生成“跳跃→玩逗猫棒”和“玩逗猫棒→跳跃”。
2. 每个动作顺序同时测试绝对时间轴 prompt 与纯顺序语义 prompt。
3. 每条视频真实生成 64 latent 帧，经 VAE 解码为 253 pixel 帧，24fps，约 10.54 秒。
4. 只使用 seed=1，并保持现有 24-latent 滚动 KV 窗口、`sink=0`，不增加推理消融。
5. 复用现有严格 EMA merge 和技术门禁，最终输出 16 条视频及一个按 prompt 风格并排展示的 HTML，供人类判断生成效果。
6. 任务完成并测试通过后, 给我一份简要的中文文档指导我在内网H100上快速部署和实验

完成后，内网用户应能通过一个独立 shell 入口运行任务，并在新的空 work dir 中得到：

```text
validation_report.json
comparison.html
checkpoint_model_003750/
├── merge_manifest.json
├── output_validation_report.json
└── prepared/
    ├── prepared_manifest.json
    ├── configs/
    └── videos/
```

## 2. 已确认决策

| 项目 | 锁定值 |
|---|---|
| checkpoint | 只测 `checkpoint_model_003750/adapter_ema.safetensors` |
| checkpoint 选择方式 | 使用显式 `--training-checkpoint`；不得扫描或批量运行其他 step |
| solver / steps / CFG | UniPC / 50 / 5.0 |
| seed | 仅 `1` |
| negative prompt | 沿用现有 Stage-1 checkpoint 验证默认值 |
| latent / pixel frames | 64 latent / 253 pixel；`1 + (64 - 1) * 4 = 253` |
| FPS / 近似时长 | 24fps / 约 10.54 秒 |
| AR block | `num_frame_per_block=8`，共 8 blocks |
| attention / KV | 保持当前配置 `local_attn_size=-1` 的既有实际行为：24-latent 滚动 cache |
| sink | `sink_size=0`；首帧不会永久 pin |
| 尾段 | 由模型继续生成静止姿态；严禁复制尾帧或后处理补帧 |
| 首帧 | 布偶猫、俄罗斯森林长毛猫、暹罗猫、狸花猫 |
| 动作顺序 | `jump_then_toy`、`toy_then_jump` |
| prompt 风格 | `absolute_timeline`、`sequential` |
| 样本数 | 4 cats × 2 orders × 2 styles = 16 |
| 判断方式 | 自动只做技术门禁；视觉与动作效果由人类查看 HTML |
| 原 4s testset | 保持原文件不变，本任务不重新生成、不混入新 metadata |

用户确认的原单动作成功映射是：布偶猫、俄罗斯森林长毛猫对应逗猫棒；暹罗猫、狸花猫对应跳跃。该口径优先于当前 6-case CSV 中不一致的动作文字。新双动作 prompt 不得根据旧 CSV 的动作标签反推或更改这项决定。

## 3. 范围与非目标

### 3.1 本次范围

- 新建专用 16-case metadata，不修改原 6-case 文件。
- 让现有 checkpoint 验证 runner 可通过 CLI 选择 64 latent 帧，同时保留默认 24。
- 让 testset loader/preparer 在显式开关下允许同一首帧用于多个 prompt。
- 增加 prompt-style 对比 HTML 模式：8 个 case group 为行，两种 prompt 风格为列。
- 新增独立 H100 shell 入口、简短运行说明和必要测试。

### 3.2 明确不做

- 不训练或微调权重，不改 LoRA、EMA、merge 语义。
- 不测其他 checkpoint，不测 seed=2/3，不测 15 秒三动作。
- 不新增 10 秒单动作对照，不重新运行原 6 条 4 秒基线。
- 不测试 `sink_size=1`、64-frame 全历史 attention、更长 local window、prompt 分 block 切换或其他消融。
- 不做自动动作识别、自动 best 判定、评分 CSV、人工打分系统或数据采集建议生成器。
- 不为尾段重复最后一帧，不用视频后处理掩盖模型漂移。
- 不修改训练配置、训练 cache、训练数据或正式 checkpoint。

## 4. 现有实现事实与最小设计

### 4.1 必须基于的代码事实

- `scripts/run_stage1_training_checkpoints_validation.py` 当前把 `num_latent_frames=24`、`num_frame_per_block=8`、`minimum_source_frames=97` 写死后调用 `prepare_causal_testsets()`。
- `prepare_causal_testsets()` 已能根据任意合法 `num_latent_frames` 计算 pixel/carrier frames；64 latent 会得到 253 pixel frames，且 carrier 也应为 253 帧。
- `load_causal_testset_records()` 当前拒绝重复图片路径；新 metadata 每只猫出现 4 次，因此必须增加默认关闭的 opt-in。
- `validate_causal_testset_outputs()` 已从 prepared manifest 读取期望帧数，无需另写一套 253-frame validator。
- 当前生成配置使用 `local_attn_size=-1`、`sink_size=0`。在现有 pipeline 中 `-1` 会分配 `3 * num_frame_per_block = 24` latent 的 KV cache；本任务保持该行为，不改 pipeline。
- 当前 HTML 以 checkpoint 为列；本任务只有一个 checkpoint，需要新增可选的 prompt-style 布局，同时保留原布局为默认。

### 4.2 推荐的最小通用扩展

在 `scripts/run_stage1_training_checkpoints_validation.py` 中增加：

- `--num-latent-frames`：正整数，默认 `24`；必须能被 8 整除。本任务传 `64`。
- `--allow-repeated-input-images`：默认关闭；本任务显式开启，并向初次 metadata load 和 preparation 全链路传递。
- `--comparison-mode {checkpoint,prompt-style}`：默认 `checkpoint`；本任务使用 `prompt-style`。

在 `utils/stage1_causal_validation.py` 中：

- 给 `load_causal_testset_records()` 与 `prepare_causal_testsets()` 增加默认关闭的 `allow_repeated_input_images` 参数。
- 开关关闭时保持现有重复图片报错；开启时允许同一已验证图片被多行复用，但每行仍独立校验 prompt、geometry、bucket、row hash 和图片 hash。
- 不降低缺文件、非 RGB、尺寸、EXIF、bucket、空 prompt 等现有门禁。

prompt-style HTML 可在现有 runner 中读取 metadata 的可选审阅列，不必改变 prepared/output report 的 row-id 映射协议。该模式必须：

- 要求只选择一个 checkpoint。
- 要求每个 `case_group` 恰有 `absolute_timeline` 和 `sequential` 两行。
- 以 8 个 `case_group` 为行、两种 prompt 风格为列嵌入视频，并显示猫、动作顺序和完整 prompt。
- 使用输出报告的 `row_id` 精确映射视频；不得按 glob 顺序猜测。
- 使用相对 work-dir 的 URL，确保把整个 work dir 拷走后 HTML 仍可打开。
- checkpoint 模式的现有 HTML 内容与测试保持不变。

不要新建一份复制现有 merge loop 的专用 Python runner。独立性由 metadata、shell 入口和 CLI 参数实现。

## 5. Metadata 契约与 prompt 设计

### 5.1 文件与 schema

新建：

`testsets/metadata_16cases_two_actions_480x832_253frames.csv`

列顺序：

```text
input_image,prompt,height,width,bucket,case_group,prompt_style,cat_id,action_order
```

约束：

- 16 行；UTF-8；不得使用 Excel 公式或多余索引列。
- 8 个唯一 `case_group`，格式 `<cat_id>_<action_order>`；每组恰好两行。
- `prompt_style` 只允许 `absolute_timeline`、`sequential`。
- `action_order` 只允许 `jump_then_toy`、`toy_then_jump`。
- 每只猫恰好 4 行；同一 case group 的两行必须使用相同图片、geometry、cat 和 action order。
- 图片继续相对 metadata 文件解析：

| cat_id | input_image | height | width | bucket |
|---|---|---:|---:|---|
| `ragdoll` | `images_480x832/ragdoll_cat_832x480_no_distortion.png` | 832 | 480 | portrait |
| `russian_forest` | `images_480x832/russian_forest_cat_480x832_no_distortion.png` | 480 | 832 | landscape |
| `siamese` | `images_480x832/siamese_cat_832x480_no_distortion.png` | 832 | 480 | portrait |
| `tabby` | `images_480x832/tabby_cat_832x480_no_distortion.png` | 832 | 480 | portrait |

### 5.2 Prompt 不变量

每条 prompt 必须包含：对应猫咪身份与外观、静止摄像机、纯白背景、全景、猫咪居中、身体始终 100% 留在画面内。动作原语语义固定如下，两个 prompt 风格只改变时间表达，不得改变动作强度或引入第三个动作：

- **跳跃**：被前上方目标吸引；身体轻微前倾；后腿发力；完成一次自然小幅跳跃；前爪向前伸出；身体协调；落地后收回前爪并恢复标准直立蹲坐。
- **玩逗猫棒**：主人在镜头前方轻轻晃动逗猫棒；猫咪目光自然跟随；先抬一只前爪试探，再用两只前爪交替轻快扑抓；身体主要保持坐姿；结束后收回前爪并恢复标准直立蹲坐。
- **最终静止**：完成第二个动作并恢复坐姿后，继续注视镜头前方，身体、四肢和尾巴保持最终稳定状态直到视频结束；这是模型生成要求，不是后处理指令。

### 5.3 绝对时间轴版

必须使用以下时间结构，并按 action order 替换动作一/二：

```text
0-1秒：初始标准直立蹲坐。
1-3秒：完成动作一。
3-4秒：平滑恢复标准直立蹲坐。
4-5秒：保持稳定坐姿，准备下一次互动。
5-7秒：完成动作二。
7-8秒：平滑恢复最终标准直立蹲坐。
8-10.54秒：保持最终静止姿态直到视频结束。
```

### 5.4 顺序语义版

- 不出现 `5-7秒`、`8-10.54秒` 或任何绝对时间段。
- 使用“开始时→先→恢复坐姿→短暂停顿→随后→恢复最终坐姿→剩余视频保持静止”的顺序描述。
- 身份、动作细节、动作顺序、恢复要求、最终静止和画面约束必须与对应绝对时间轴版语义一致。

## 6. Shell 入口与内网运行契约

新建根目录入口：`infer_stage1_two_actions_10s.sh`。

要求：

- 沿用 `infer_stage1.sh` 的模型、architecture、T5、tokenizer、VAE、base 和 Python 环境变量约定。
- 默认 checkpoint 为 `$LONG_LIVE_STAGE1_TRAIN_DIR/checkpoint_model_003750`，通过 `--training-checkpoint` 显式传递。
- metadata 指向新的 16-case 文件。
- 显式传 `--num-latent-frames 64 --allow-repeated-input-images --comparison-mode prompt-style --sampling-steps 50 --guidance-scale 5.0 --seed 1`。
- 只用一张 GPU，保持 `CUDA_VISIBLE_DEVICES=0` 的现有默认；允许用户在运行前覆盖。
- work dir 使用独立、必须为空的路径；不得复用或覆盖已有失败现场。
- 不添加 `--keep-merged`：成功后删除可重建的完整 merged checkpoint；失败时沿用现有保留现场行为。
- 不添加任何 sink、local-attention、solver 或 VAE streaming 覆盖。

新增简短运行文档 `docs/STAGE1_TWO_ACTION_LONG_INFERENCE_ZH.md`，包含环境变量、命令、预期产物、253-frame 技术门禁、如何打开 HTML，以及“视觉效果必须人工判断”的说明。

## 7. 验收标准

### 7.1 本地代码验收

- [ ] 原 `testsets/metadata_6cases_480x832.csv` 内容不变。
- [ ] 新 metadata 严格为 16 行、4 cats、8 case groups、每组两种 prompt 风格。
- [ ] 默认 parser/preparer 仍拒绝重复首帧；只有显式 opt-in 才接受新 metadata。
- [ ] 24-latent 默认路径仍生成 93 pixel 帧，原 checkpoint 对比 HTML 不变。
- [ ] 64-latent preparation 生成 `[1,64,48,H/16,W/16]` config，manifest 记录 253 expected/carrier frames、24fps、8-frame blocks。
- [ ] 生成 config 中 `model_kwargs.local_attn_size=-1`、`inference.local_attn_size=-1`、`sink_size=0`；未出现其他 attention/sink override。
- [ ] runner 只选择显式 3750 checkpoint 的 EMA merge 路径。
- [ ] prompt-style 模式拒绝缺列、重复/缺失 style、错误 group 或多个 checkpoint。
- [ ] prompt-style HTML 为 8 行 × 2 prompt-style 列，16 条视频均通过 row ID 唯一映射且链接可用。
- [ ] shell 语法通过，CLI help 可解析，相关单元测试通过。

### 7.2 内网 H100 技术验收

- [ ] 顶层 `validation_report.json` 为 `status=pass`，checkpoint count=1，step=3750，sample count=16。
- [ ] `output_validation_report.json` 为 `status=pass`，无缺失或额外 MP4。
- [ ] 每条视频恰好 253 帧、24fps，分辨率与输入图片一致，首帧门禁、非纯色视频和非全程冻结门禁通过。
- [ ] `prepared_manifest.json` 记录 64 latent、253 pixel/carrier frames、seed=1、UniPC/50/CFG5。
- [ ] `comparison.html` 可直接打开并按 case group 并排播放两种 prompt 风格。
- [ ] 尾段来自模型真实生成；输出流程未执行复制帧补时。

视觉动作是否正确、顺序是否遵循、动作切换是否自然、猫咪身份/肢体是否保持、尾段是否稳定均不属于自动代码验收，由用户人工查看 HTML 判断。

## 8. 按最小可验证步骤执行

- [ ] **Step 1 — 建立回归基线。** 结果：确认现有 24-latent 流程和相关测试在修改前的状态。区域：上述 runner、validation helper、两份测试。验证：运行目标 pytest；如基线已失败，先记录且不得把无关失败算作本任务造成。
- [ ] **Step 2 — 增加重复首帧 opt-in。** 结果：默认严格、显式允许复用图片。区域：`utils/stage1_causal_validation.py`、`scripts/prepare_stage1_causal_testsets.py`（如需暴露同名 CLI）、`tests/test_stage1_causal_validation.py`。验证：一正一反测试，确认其余图片门禁不变。
- [ ] **Step 3 — 参数化长时 preparation。** 结果：runner 默认 24，本任务可传 64；64 必须生成 253-frame manifest/config。区域：`scripts/run_stage1_training_checkpoints_validation.py` 及测试。验证：断言参数透传、64%8 门禁、shape/frame policy、默认回归。
- [ ] **Step 4 — 实现 prompt-style HTML。** 结果：单 checkpoint 下按 8 groups × 2 styles 并排展示，默认 checkpoint HTML 无变化。区域：runner HTML helper 与 `tests/test_stage1_checkpoint_inference_validation.py`。验证：正确布局、row-id 映射、相对链接、错误 schema fail-fast。
- [ ] **Step 5 — 写入 16-case metadata。** 结果：所有矩阵组合和 prompt 契约准确。区域：新 CSV。验证：使用正式 loader 在 opt-in 模式解析，检查 16/4/8×2 计数、geometry、图片存在、prompt 风格约束。
- [ ] **Step 6 — 增加 shell 与运行文档。** 结果：用户可在内网通过一个命令运行 3750 EMA。区域：`infer_stage1_two_actions_10s.sh`、新 runbook。验证：`bash -n`、CLI `--help`、命令参数静态检查。
- [ ] **Step 7 — 整体本地验证。** 结果：实现可交给 H100。验证：至少运行：

  ```bash
  pytest -q tests/test_stage1_causal_validation.py tests/test_stage1_checkpoint_inference_validation.py
  bash -n infer_stage1_two_actions_10s.sh
  python scripts/run_stage1_training_checkpoints_validation.py --help
  git diff --check
  ```

- [ ] **Step 8 — 用户内网 H100 运行。** 结果：得到 16 条 253-frame 视频、pass 报告和 HTML。验证：按 7.2 全部技术门禁检查；人工打开 HTML 判断效果。该步骤由用户执行，不得在无 checkpoint/H100 的本地环境标记完成。

## 9. 风险、失败恢复与延期项

- **长时分布外推：** 权重只见过 24 latent/约 4 秒训练样本；64 latent 第 4–8 block 的退化是本测试要观察的结果，不应通过改 prompt、复制帧或自动重试掩盖。
- **prompt 时间分布外推：** 绝对时间轴含 4 秒后的时间表达；顺序语义版用于区分该因素，但不自动计算优劣。
- **滚动 cache 遗忘：** 24-latent cache 且 sink=0 会淘汰早期内容，后半段身份漂移同时反映当前推理策略与权重能力；本轮不修复。
- **显存或 VAE OOM：** 253-frame 解码比 93-frame 更重。失败时保留 work dir 和 merged checkpoint 现场并报告；不得未经确认自动开启 streaming VAE、量化、CPU offload 或改变 attention。
- **重复图片门禁：** 只对新入口显式放宽；若旧流程开始接受重复图片，视为回归失败。
- **metadata 动作映射冲突：** 以本文记录的用户确认映射和新 prompt 为准，不修改旧 CSV。
- **失败重跑：** 现有 runner 要求 work dir 不存在或为空。失败后使用新目录，或由用户明确处理旧现场；脚本不得自动删除。
- **明确延期：** 15 秒三动作、多个 seed、其他 checkpoint、sink/local-window 消融、自动评分与数据采集方案均留待本轮人工结果之后决定。
