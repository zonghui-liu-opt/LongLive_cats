# 发现与决策

## 需求
- 2026-08-04 continuation 新任务：实现显式单样本有状态 session，按 `A×3 → HOLD×2 → B×3` 生成 64 latent；A/HOLD/B 切换文本 conditioning，但 positive/negative self-KV、global cursor 与 temporal RoPE 连续。
- B 复用 HOLD 的 global latent 39 作为 global latent 40 的 soft anchor；A 的 global latent 0 使用 initial image latent。两处 anchor 均在每个采样 step 模型前和 scheduler 后 clamp，且各段长度仍分别为 24/16/24。
- 正式矩阵为新 8-row metadata × sink0/1，共 16 视频；同 case 两个 sink 必须共享 bitwise-identical contiguous 64-latent noise plan 与 initial latent clone，并只在最后对累计 64 latent 做一次 VAE decode。
- 普通 `pipeline.inference()` 必须保持独立样本、默认重置；只有显式 session 可续写。所有 shape/dtype/device/block/cache/cursor/lifecycle 不变量都必须 fail-fast，失败 session 不可复用。
- 新入口必须复用 checkpoint 3750 EMA merge、模型 bootstrap、视频保存与技术门禁；不得复制 `inference.py` 或 checkpoint merge 系统。最终需新增 shell、8×2 同步 HTML、可审计 session trace 和中文内网 H100 快速部署文档。
- 2026-08-04 更新：实验矩阵改为 4 cats × 仅 `jump_then_toy` × 3 prompt styles，共 12 条。
- 三种风格为：全局绝对时间轴 `absolute_timeline`、纯顺序语义 `sequential`、两个阶段分别复用 0–4 秒局部时间轴的 `phase_relative`。
- 新 metadata、prompt-style HTML、shell、runbook 和测试必须同步改为 4 组 × 3 列；不能只改 CSV 导致 runner 拒绝。
- 目标：为 `checkpoint_model_003750/adapter_ema.safetensors` 增加可复现的 10.54 秒双动作推理对比流程；本地只完成代码与单测，真实 H100 推理由用户在内网执行。
- 固定实验矩阵：4 只猫 × 2 个动作顺序 × 2 种 prompt 风格，共 16 条；只用 seed=1、UniPC、50 steps、CFG 5.0。
- 固定时长策略：64 latent 帧，按 `1 + (64 - 1) * 4` 解码为 253 pixel 帧，24fps；`num_frame_per_block=8`，继续使用现有 24-latent 滚动 KV cache、`local_attn_size=-1`、`sink_size=0`。
- 必须复用现有 EMA merge、testset preparation、`inference.py`、输出验证和 HTML 链路；不得复制 runner，也不得伪造 H100 成功。
- 最小代码扩展：runner 新增 `--num-latent-frames`（默认 24 且必须可被 8 整除）、`--allow-repeated-input-images`（默认关闭）、`--comparison-mode {checkpoint,prompt-style}`（默认 checkpoint）。
- loader/preparer 仅在显式 opt-in 时允许重复图片；所有其他图片、prompt、geometry、bucket、hash 门禁保持严格。
- prompt-style HTML：只允许单 checkpoint；每个 `case_group` 必须恰有两种 style；8 行 × 2 列；按输出报告 `row_id` 映射视频；链接相对 work dir；默认 checkpoint HTML 必须无变化。
- 新增严格的 16-case CSV、新 shell 入口 `infer_stage1_two_actions_10s.sh`、中文运行文档 `docs/STAGE1_TWO_ACTION_LONG_INFERENCE_ZH.md` 和必要测试。
- shell 必须显式选择 3750 checkpoint 和所有锁定参数；work dir 必须不存在或为空；成功时不保留 merged checkpoint，失败时保留现场。
- 本地至少验证两份目标 pytest、`bash -n`、runner `--help` 与 `git diff --check`；H100 技术验收明确留给用户。

## 研究发现
- The shared `_denoise_and_recache_block()` now owns scheduler-per-block, optional anchor pre/post clamp, CFG calls and final clean recache. Ordinary `_inference_inner()` calls this kernel and its characterization remains unchanged, providing compatibility evidence before session integration.
- `ContinuationSession` uses separately allocated pos/neg self-KV and cross-attn bundles, an exclusive pipeline lock/runtime context, per-block all-layer cursor audits, prompt token audits, locked CFG/RoPE/shape/dtype/device state, failure poisoning and one final VAE decode. Tiny 24/16/24 tests prove cursor 24/40/64 and local window 8/16/24/24…, with initial anchor at 0 and soft anchor 39→40.
- Session cache audit explicitly rejects quantized KV, verifies all positive/negative layers (not just layer 0), enforces capacity `24×S`, token-unit indices, `pinned_start=-1`, effective attention local size 24 and selected legacy sink. This protects the model wrapper’s layer-0 global metadata shortcut from hidden layer drift.
- Step 2 implemented in `utils/stage1_continuation_validation.py` as a dedicated strict loader that structurally reuses Stage-1 image validation/path/dimension helpers without changing the old causal loader. It validates exact header order, duplicate/extra/missing values, fixed case/image/block matrix, canonical lowercase boolean, common prompt constraints and explicit action/HOLD semantics.
- The formal 8-row CSV was corrected from uppercase `TRUE` to canonical `true`; all other row content was preserved. New metadata tests are 16 passed, and combined with old Stage-1 causal validation are 29 passed.
- Step 1 baseline completed on `/Users/zonghuiliu/anaconda3/bin/python`: existing prompt batching + Stage-1 causal/checkpoint validation were 30 passed; with 2 new direct pipeline characterization tests, the targeted baseline is 32 passed. Independent full-suite baseline before production changes was 159 passed plus 2 subtests, with only existing torch.jit deprecation warnings.
- Characterization now exercises the real `_initialize_sample_scheduler()` seam with a patched UniPC class: each block creates a unique scheduler, requests `set_timesteps(50, shift=5.0)`, and ordinary inference twice resets both positive/negative raw indices to zero. It also locks current/cache start pairs, exact final clean recache calls, block prompt conditioning and first-anchor clamping.
- Architecture audit identified a formal safety restriction: quantized KV rolling works in 8-frame blocks while legacy `sink_size=1` protects only one frame, so continuation formal mode must fail-fast when `kv_quant=True`. Fixed experiment configs already use unquantized KV.
- `pipeline/__init__.py` lazily imports the causal pipeline, so the new characterization/session tests can import the concrete module and instantiate via `__new__` with tiny stubs, avoiding checkpoint/model construction. Existing scheduler-mocking tests elsewhere confirm this project accepts CPU fake schedulers as the standard unit-test approach.
- Current system Python has torch 2.5.0/pytest 9.0.3 but lacks OmegaConf, diffusers and easydict; the prior `/tmp` isolated dependency directory is no longer present. Baseline/new tests must recreate a task-local temporary dependency target and set `PYTHONPATH=$PWD:<target>`, without modifying project/system environments.
- The supplied continuation CSV has the correct 14-column shape and 8 semantic rows, but every `soft_reanchor` value is uppercase `TRUE`. This violates the locked canonical lowercase `true` contract; the strict loader must reject the current file until the data is corrected, and the formal metadata test must lock lowercase spelling.
- Repository tests do not currently exercise `CausalDiffusionInferencePipeline.inference()` directly. Step 1 therefore requires a new mock-based characterization test module before refactoring, covering two-call cache reset, block prompt mapping, scheduler-per-block, initial-anchor pre/post-step clamp and clean recache order.
- The only currently modified training config exposes `num_frame_per_block: 8` but continuation must not rely on or rewrite this user-modified config; the new prepared inference configs should lock all required values independently.
- Exact prompt-length audit is implementable without a second tokenizer: take `pipeline.text_encoder.tokenizer`, apply its `_clean()` (`whitespace`), then call the underlying Hugging Face tokenizer with `add_special_tokens=True`, `padding=False`, `truncation=False`; reject count `>= 512` before calling normal `WanTextEncoder.forward()`. This satisfies same tokenizer/cleaning/special-token semantics and avoids already-truncated attention masks.
- The checkpoint test suite injects fake base/merge/prepare/output/command functions into runner functions, giving a strong reusable pattern for continuation orchestration tests without CUDA/checkpoints. It also explicitly checks merged deletion only after pass and report persistence; the new runner must retain those behaviors and add atomic partial session traces.
- Existing `scripts/run_stage1_causal_testsets_validation.py` is another simple example of empty-work-dir enforcement and `inference.py` subprocess orchestration, but its uniform dataset preparation is not suitable for the continuation schema. Reuse its safety/report principles, not its dataset transformation.
- `scripts/run_stage1_training_checkpoints_validation.py` already has the exact reusable checkpoint flow: atomic top-level report, converted-base audit, explicit checkpoint validation, `_load_merge_config()`, `merge_stage1_ema_checkpoint()`, merged artifact retention on failure and deletion on success. A continuation runner can import/factor these helpers while replacing only preparation/inference/output mapping/HTML; it must not fork merge logic.
- Existing checkpoint runner writes a config per geometry and invokes `inference.py` as a subprocess. The least invasive bootstrap integration is for continuation preparation to write an explicit `continuation` config/manifest section; `inference.py` detects it after the same pipeline bootstrap/device/VAE setup and delegates to a small orchestration module, skipping its ordinary dataset loop.
- Current Stage-1 tests already cover loader image gates, repeated-image opt-in isolation, 64→253 frame policy, config locking, exact output filename sets and metric thresholds. New continuation tests should reuse their fixture patterns but keep the new 14-column parser separate and prove the old loader still rejects repeated images by default.
- Existing checkpoint HTML maps report `row_id` rather than glob order and enforces work-dir-relative links in prompt-style mode. The new 8×2 page can reuse its escaping/URL principles but needs its own case_group×sink renderer and DOM sync-control tests; no score/winner fields should exist.
- `WanDiffusionWrapper._call_model()` reads the first block cache indices in eager Python and publishes them to model-global metadata before compiled forward; session must first assert every block agrees, otherwise the compiled path would silently trust block 0. The existing generator/model application remains the only cache update path to preserve.
- `WanTextEncoder.forward()` uses `HuggingfaceTokenizer(seq_len=512, clean='whitespace')` and only exposes already fixed-length ids/mask. The continuation runner needs a separate pre-encoding audit method using that exact tokenizer/backend, same cleaning and special-token settings with truncation disabled; token count cannot be inferred from the padded/truncated forward mask.
- Existing prompt batching helper encodes the flattened block prompts once and creates per-block condition dict views. continuation can reduce redundant encoding further by explicitly encoding A/HOLD/B once at session start (batch=1) and reusing each dict across the segment’s blocks; the existing ordinary prompt batching tests must remain unchanged.
- `WanVAEWrapper.decode_to_pixel()` decodes the entire `[B,T,C,H,W]` latent tensor in one call per sample and returns the required temporal sequence. The continuation `finish()` should call this exactly once and must reject configured streaming VAE rather than falling into chunked/cached decode helpers.
- `CausalWanModel._apply_cache_updates()` mutates each transformer block cache only after all block forwards finish, then fills raw token-unit global/local indices. Reusing this model call at t=0 is sufficient for clean recache; session invariant checks should assert all pos/neg block indices agree before and after every block rather than duplicating cache mutation logic.
- Model `_forward_inference()` publishes frame/token metadata once, propagates the same `current_start`/`cache_start` into every attention block, collects each block update and applies them together. This is the correct seam for continuation: preserve monotonically increasing latent cursor at pipeline level and continue invoking the existing generator rather than adding a second KV algorithm.
- `utils/stage1_causal_validation.py` already provides strict image RGB/EXIF/geometry/bucket validation, atomic IO helpers, ffprobe, pixel metrics and output thresholds. The new 14-column metadata contract should live in a small continuation module that reuses the shared image validation primitives (or factors them once) without relaxing `load_causal_testset_records()` duplicate-image default.
- Existing `validate_causal_testset_outputs()` assumes prepared bucket directories and `rank0-...` filenames. Continuation output layout differs, so its probe/pixel metric logic and exact original thresholds should be reused in a new exact-output-set validator, not the old directory mapping or a weakened check.
- `CausalWanSelfAttention.__init__()` 明确把配置 `local_attn_size=-1` 映射为运行时 `self.local_attn_size=24`；因此 rolling 条件会启用。pipeline 分配的 cache 也正好是 24 latent。attention 的 `max_attention_size` 虽会被 pipeline override 为全局常量，但可供 attention 的 `window_k/v` 最多只有滚动后的 24-latent cache；必须用运行时断言同时锁定 capacity、attention 有效 local size与 raw indices。
- legacy `sink_size=1` 在无 multi-shot pinned/global sink 时通过 `effective_sink=max(global_sink_tokens,sink_tokens)` 永久保护 cache 缓冲区最前面的 1 latent，并在 rolling 时只移动其后的局部内容；这正是正式 sink1 需要复用的原始首图 leading sink。只要 `multi_shot_sink=false`，无需也不应调用 `_pin_current_chunk()`，`pinned_start` 应始终为 -1。
- self-attention 的 `current_start/current_end` 与 cache `global_end_index/local_end_index` 全部使用 patch-token 单位；RoPE 的 temporal start 由 `current_start // frame_seqlen` 得到。session 必须以 latent-frame cursor 为公共 API，但每次模型调用严格乘 `frame_seq_length`，并在 trace 中保留 raw token index。
- `MultiShotT2VCrossAttention` 当前无条件 bypass `crossattn_cache`，每次从传入 context 重算 K/V；continuation 测试必须观察各段真实 positive conditioning 入参变化以及 negative conditioning 对象复用，不能把 cross-attn cache `is_init` 当作 prompt switch 证据。
- `CausalDiffusionInferencePipeline.inference()` 每次都会编码正向 block prompts 与一次 negative prompt；已有 cache 时显式把 pos/neg self-KV 的 `global_end_index`/`local_end_index` 重置为 0，并清 pinned 状态，所以普通 inference 当前确实隔离样本。session 不可通过多次调用该方法实现。
- `_inference_inner()` 已包含可复用的关键 block 算法：每个 block 单独创建 scheduler；每个采样 step 前对首 I2V block覆盖 initial latent 并将其 timestep 置 0，scheduler 后再次覆盖；随后用 clean latent、t=0 对 positive/negative self-KV recache，再递增 frame cursor。它也会在每 block 强制 cross-attn cache `is_init=false`，与当前 cross-attn bypass 语义兼容。
- 当前 pipeline 把 KV cache、cross-attn cache、cursor 局部变量和临时 model runtime overrides 混在实例字段/单次调用中。最小安全重构需要把“运行时 model override + shared block generation”封装成可复用内部路径，并让显式 session 独占 pipeline（或拥有 cache），同时普通 inference 继续走 reset 初始化。
- `_initialize_kv_cache()` 在有效 `local_attn_size==-1` 时分配 `3 × num_frame_per_block × frame_seq_length`，即 24-latent cache；raw indices 均是 token 单位。`_set_all_modules_max_attention_size(-1)` 另将 attention max 设置为全局默认，真实 rolling 行为仍需在 model attention 更新逻辑中核实，不能仅看该 helper。
- 非 streaming 且 `return_latents=false` 时 `_inference_inner()` 只在所有 block 完成后调用一次 `vae.decode_to_pixel(output)`；continuation 正式路径应强制这条统一 decode 语义并禁用 streaming VAE。
- `inference.py` 当前是 670 行的脚本式入口：模块导入后立即解析 config、初始化分布式/device、构建并加载 pipeline/LoRA/量化/VAE、创建 dataset，然后在逐样本循环中采样整段 noise、编码首图并调用一次 `pipeline.inference()`。continuation 必须复用这条 bootstrap，不能在外部 runner 复制模型初始化。
- 当前 `inference.py` 已从 dataset 取得 per-block prompt 列表并传入 pipeline，且统一由 pipeline 返回 latent 或解码视频；新分支宜在模型/VAE/device 初始化完成后、旧 dataset 循环前由显式 config/manifest 触发小型 orchestration module，普通分支保持原样。
- 旧 `infer_stage1_two_actions_10s.sh` 固定调用 checkpoint validation runner、12-case prompt-style metadata 和 64-latent uniform-prompt preparation；新 continuation shell 必须使用独立 work-dir 变量/metadata/runner，不能改变该文件。
- 当前工作树仍包含用户既有 config 修改、删除文件及多项未跟踪内容；新任务自带的 continuation CSV 也为未跟踪文件。实现必须只触碰 continuation 明确范围，不恢复或整理其他内容。
- 新任务规格共 351 行，锁定了 10 个本地/内网执行步骤；本地只能完成 Step 1–9 和部署文档，H100 真实技术/视觉验收不得伪造。
- 2026-08-04 用户将正式实验矩阵收敛为 4 只猫 × 固定 `jump_then_toy` × 3 种 prompt 风格，共 12 条；第三种为 `phase_relative`，两个动作阶段都只使用训练域内的 `0-1秒`、`1-3秒`、`3-4秒` 相对时间锚点。
- 当前 16-case CSV、专用 shell、runbook、runner 的 `PROMPT_STYLE_ORDER`/HTML 表头及正式 metadata 契约测试均仍锁定旧的 2 顺序 × 2 风格矩阵，需要同步更新，避免数据、页面与文档计数漂移。
- 修改前目标回归再次通过：27 passed、2 条第三方 SWIG deprecation warnings；可作为本轮 12-case 改动的干净基线。
- 12-case 实现后的两份目标回归仍为 27 passed、2 warnings；三风格 parser、HTML row-id 映射及正式 metadata 契约均已纳入测试。
- 正式 12-case preparation 集成通过：`landscape_480x832=3`、`portrait_832x480=9`，64 latent 对应 253 pixel/carrier frames；UniPC/50/CFG5/seed1、attention/sink 与 streaming VAE 锁定值均保持不变。
- 最终静态验收通过：shell syntax、runner help、Python syntax、tracked/untracked whitespace、旧 6-case SHA256 和默认 `_comparison_html` 字节级源码兼容性均无回归。
- 任务规格共 252 行，包含 8 个最小验证步骤；应严格按 Step 1→7 本地执行，Step 8 只能作为用户后续操作。
- 仓库已包含规格要求复用的 runner、validation helper、preparer、两份目标测试、旧 shell 与旧运行文档。
- 4 张目标首帧图片均已存在于 `testsets/images_480x832/`。
- 当前工作树不是干净状态：`configs/train_i2v_ar.yaml` 已修改，3 个文件已删除，`TASK-stage1.md`、`infer_batch.sh`、`infer_stage1.sh`、`training_sets/` 等为未跟踪内容；这些均视为用户现有改动，不覆盖、不恢复、不顺手修改。
- 原 `testsets/metadata_6cases_480x832.csv` 当前未显示为改动，任务全过程必须保持内容不变。
- 现有 `infer_stage1.sh` 是用户未跟踪文件，硬编码了内网路径、GPU 0、旧 6-case metadata 和多 step `--steps` 运行；新入口可沿用其环境变量约定，但必须新建文件且不改写它。
- 独立 preparer 已有 `--num-latent-frames` CLI，并向 helper 透传；当前没有重复首帧 opt-in。为保持接口一致，应在 preparer 增加同名布尔开关并透传。
- runner 当前固定 `num_latent_frames=24`、`num_frame_per_block=8`、`minimum_source_frames=97`，且 metadata 初次 load 与 preparation 内部 load 各执行一次，因此重复图片开关必须同时传到两处。
- runner 已支持显式 `--training-checkpoint`，允许重复该 flag 比较多 checkpoint；prompt-style 模式需在运行昂贵 base/merge 前尽早拒绝多个 checkpoint（可在 checkpoint 解析后、metadata 审阅校验阶段 fail-fast）。
- 现有 `_comparison_html` 按 record 行、checkpoint 列，视频通过报告中 `row_id` 映射并优先生成相对 work-dir URL；默认模式应原样保留其输出结构/文字。
- `CausalTestsetRecord` 只保存核心必需列，不保存额外审阅列。prompt-style 布局可在 runner 额外严格读取 CSV 审阅列，并用 CSV 行号与正式 loader 的 `row_id` 对齐，无需更改 prepared/output 协议。
- helper 中 64 latent 已自然产生 shape `[1,64,48,H/16,W/16]`、253 pixel/carrier frames（因 max(97,253)），并保持 config 的 model/inference `local_attn_size=-1` 与 `sink_size=0`。所需核心变化仅是重复图片参数透传。
- 输出 validator 已从 manifest 读取 expected frames/fps 并严格按每个 prepared record 的 bucket index 校验文件集合、geometry、首帧、非纯色和非冻结，无需修改 253-frame validator。
- 项目文档规定本地测试使用 `PYTHONPATH="$PWD" python -m pytest`；初次直接调用 `pytest` 同时暴露了路径用法错误和当前系统 Python 缺少 `omegaconf`。
- `requirements.txt` 明确包含 `omegaconf`；其余两份目标测试的核心依赖（torch/Pillow/safetensors）在当前 Python 已存在。为不污染仓库或系统环境，可把缺失的小依赖安装到 `/tmp` 隔离目录并通过 `PYTHONPATH` 注入。
- 目标测试导入链为 `test_stage1_causal_validation` → converter → `wan_5b.textimage2video`，收集阶段直接需要 `diffusers`；当前系统已有 transformers、tokenizers、accelerate、einops 等，另缺 easydict/cv2/av，但后两者在目标测试路径中是延迟使用或被测试替身绕过。
- 下一次基线尝试一次性在隔离目录补入锁定版本 `diffusers==0.31.0` 与轻量 `easydict`，继续沿用现有系统 torch；避免安装整个 GPU requirements 集合。
- 原 6-case CSV 的基线 SHA256 为 `e59a14deb87ee3dc34236ddce30ad9478f0082d5a5cf35cc89b15cedd16bd94e`；最终验证必须再次比对。
- 原 CSV 的动作文字确实与规格锁定映射冲突（例如 ragdoll 写歪头、russian_forest 写跳跃、siamese/tabby 写逗猫棒）；新 prompt 必须完全按本任务固定的双动作原语构造，不能据旧标签推断。
- 仓库上层只发现两个 demo 子目录的 `AGENTS.md`，均不作用于当前 LongLive 根目录；当前适用约束为用户消息中的全局偏好和任务规格。
- 现有 checkpoint runbook 强调 work dir 为空、默认成功后删除 merged、失败时保留，以及显式 checkpoint 可重复；新 shell 只选一个 3750 checkpoint并继续沿用这些安全语义。
- 仓库现有推理入口没有统一 Python 变量，只在用户的 `infer_stage1.sh` 中写了内网解释器绝对路径；新入口将采用 `LONG_LIVE_STAGE1_PYTHON`（未设置时使用当前 `python`）并在新 runbook 明确设置方式，既兼容该内网路径也避免把个人绝对路径固化为唯一选择。
- 新 shell 将要求通过位置参数或 `LONG_LIVE_STAGE1_TWO_ACTION_WORK_DIR` 显式给出独立 work dir，主动拒绝非空目录且绝不清理；runner 仍做第二层空目录门禁。
- 两条实现主审确认：重复图片开关只绕过重复路径检查；图片存在/RGB/尺寸/EXIF/bucket/prompt/hash 仍逐行执行。runner 的初次 load 与 preparation 均收到同一开关和 latent 参数。
- prompt-style review parser 独立验证必需列、空/多余字段、两种 style、固定 action order、`case_group=<cat_id>_<action_order>` 和组内图像/geometry/bucket一致；core records 与 review records 再按 row_id/字段逐行对齐。
- prompt-style HTML 要求单 checkpoint、唯一且精确相等的 output row-id 集合，并强制视频位于 work dir 内；链接使用 URL 编码的相对路径。原 `_comparison_html` 函数未改动，默认 checkpoint 分支继续直接调用它。
- 正式新 CSV 已通过独立静态契约审计：精确 9 列、16 行、每猫4行、8组×2风格、动作顺序/geometry/prompt 时间锚点/顺序锚点/画面与最终静止不变量全部通过。
- 正式 loader 对新 CSV 的行为已实测：默认因重复图拒绝，显式 opt-in 成功返回 row_id 0–15；旧 6-case CSV SHA256 仍与基线一致。
- 使用正式 16-case CSV 和 fake carrier 的 preparation 集成检查通过：landscape 4 条、portrait 12 条；两份 config 均为 64 latent，pixel/carrier 253，UniPC/50/CFG5/seed1，model/inference attention=-1、sink=0、streaming VAE=false。
- 从 Git HEAD 与当前文件按 AST 精确提取 `_comparison_html` 后，源码逐字节一致；默认 checkpoint HTML 的生成函数未发生变化。
- 独立 prompt 语义审计结果为 `rows=16 groups=8 issues=0`；两风格每组仅改变时间/顺序表达，没有动作强度、恢复、最终静止或画面约束漂移。
- 独立 shell/runbook 审计发现并修复唯一明确问题：移除了可绕过 3750 的隐藏 checkpoint override；同时将 merged 保留措辞精确限定为 merge/preparation/inference/output-validation 失败阶段。
- 最终本地验收：27 tests passed；shell syntax、CLI help、Python syntax、tracked/untracked whitespace、shell exact flags/禁用 overrides、空 work-dir 防护、旧 CSV hash 均通过。
- 修复后独立 code review 未发现 P0–P2 问题；确认 24-frame 默认、repeat opt-in、64→253、prompt-style row-id/相对 URL、单 checkpoint 与 EMA/失败现场约束均符合规格。

- continuation session 现已锁定并逐段复核 UniPC/50/CFG5/shift5、negative prompt、shape/dtype/device、sink、KV quant、streaming、multi-shot 和有效 RoPE/t-scale 状态；任何 drift 在新 prompt tokenization/T5 编码前即 poison session。
- `frame_seq_length=3` 的 mutation-sensitive 测试锁住 raw token cursor `S×[8..64]`、local `S×[8,16,24,…]`、capacity `24S` 和全局 `current_start=cache_start`；另一个真实 `CausalWanSelfAttention` seam 验证 legacy sink1 滚动后只保留最初一帧的完整 S 个 K/V tokens，pinned 始终为 -1。
- anchor 测试现在精确观察两处 anchor 各 5 次 overwrite（2 次 step 前、2 次 scheduler 后、1 次 final clean recache），并断言 sampling 时仅 A0/B40 的首位置 timestep 为 0，其余 7 个位置非 0。
- continuation preparation 只生成两个 geometry config，并通过显式 `continuation.manifest_path/bucket_id` 在 `inference.py` 完整模型 bootstrap 后、旧 dataset loop 前分支；普通 config 不含该 opt-in 时原路径不变。
- 每个 semantic case 只执行一次 VAE image encode 和一次独立 generator 的 contiguous 64-latent noise sampling；sink0/1 分别开启全新 session，传入相同 initial/noise clone，并严格切片 0:24、24:40、40:64。
- 旧技术门禁已抽为单视频共享函数，continuation validator 对 16 MP4/16 session JSON 做 exact-set 映射、trace/cache/prompt/anchor/decode 审计和 sink pair identity 比对；只有全部 16 条通过后才原子写回 technical pass，避免部分成功伪装整轮成功。
- 新 runner 直接复用 converted-base audit、checkpoint validator、`_load_merge_config()` 与 `merge_stage1_ema_checkpoint()`；固定 step3750，两个 geometry subprocess 复用 `inference.py`，成功最后删除 merged，任一失败保留 merged 与原子报告/partial trace。
- continuation 新增目标测试最终为 111 passed；全仓正式 `tests/` 范围回归为 270 passed、2 subtests passed。直接从仓库根跑无路径限制的 pytest 会额外收集 `fouroversix/` 的可选 Modal 测试并造成 `scripts` 包遮蔽，因此项目回归应明确运行 `pytest -q tests`。
- 最终生产审计推动把 prepared manifest 的 video/trace containment、canonical naming、唯一性与完整 16 映射前移到任何编码/推理/写出之前；即使攻击者重算 manifest 自哈希，路径逃逸仍会 fail-fast。
- 普通 `pipeline.inference()`、continuation begin/generate/finish/abort 与 `clear_cache()` 现在复用同一可重入锁；普通推理整个调用周期持锁，消除了 active-session 检查与 session 激活之间的 TOCTOU。
- 最终 trace 门禁同时验证 3/2/3 schedule、A/HOLD/B 与 negative 的同 tokenizer cleaning/special-token/未截断审计、formal runtime 锁定值、24-latent attention window、raw cursor 和 trace 自身路径；两路独立复核未发现残留 P0/P1/P2。

## 技术决策
| 决策 | 理由 |
|------|------|
| 实现限定在现有 validation/helper/runner 的通用参数扩展 | 符合规格的最小设计并避免复制第二套执行链路 |
| 任何真实视频效果结论留给内网 H100 | 本地环境不具备正式 checkpoint/H100，且规格明确禁止伪造成功 |
| 先记录当前脏工作树并只触碰任务明确范围 | 保护用户既有变更 |
| prompt-style 审阅 schema 与核心 testset record 分离读取 | 不扩大 prepared/output row-id 协议，同时能严格校验专用布局字段 |
| 复用现有 `_comparison_html` 作为 checkpoint 模式，新增独立 prompt-style renderer/dispatcher | 最大限度保证默认 HTML 字节级行为不受影响 |
| prompt-style renderer 对 group 数量保持通用，正式 CSV/测试断言恰好 8 组 | 规格同时要求“最小通用扩展”和本实验 8 组；将实验矩阵约束放在数据验收层更合理 |
| prompt-style metadata 使用独立严格 parser，以 CSV 行号对齐 core records | 可拒绝缺列、style 缺失/重复、错误 group 和组内 geometry/image 不一致，同时不改变 manifest schema |
| runner 在解析 checkpoint 后、merge 前拒绝 prompt-style 多 checkpoint | 满足 fail-fast 且仍复用现有 base hash/checkpoint 严格验证 |
| 将正式 CSV 的语义契约固化为单元测试，而不只保留一次性校验脚本 | 防止后续 prompt、行顺序或图片映射被无意改坏 |
| 专用 shell 不提供 checkpoint override | 锁定实验只能选择 step 3750，避免入口与 runbook/产物路径不一致 |
| 使用新文件名 `metadata_12cases_two_actions_480x832_253frames.csv` | 避免沿用 `16cases` 文件名造成样本数误导，并让 shell/runbook 静态可审计 |
| `phase_relative` 明确写出“相对于该阶段起点”并在第二阶段重新归零 | 既测试训练域内 0–4 秒时间 token 的复用，又避免模型或人工审阅者把第二组 `0-1秒` 误解为全局回退 |
| continuation 使用 pipeline 独占 session 与 session-owned cache bundle | 明确隔离普通独立视频，失败可整体 poison/release，不会把半写 KV 传给下一样本 |
| 同一 semantic case 一次采样完整 noise，再给两个 sink clone | hash 能证明 dtype/shape/raw bytes 一致，anchor 覆盖的位置仍保留同一 RNG 索引协议 |
| portrait/landscape 各写一个 continuation config 并分别复用 `inference.py` bootstrap | `frame_seq_length`/模型 runtime 与 geometry 固定绑定，避免在同一 pipeline 中动态改变 token 网格 |
| 技术验证全部通过后才批量回写 session trace 的 pass 状态 | 第 16 条失败时前 15 条也不会留下误导性的整轮成功标记 |

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
| 修改前回归基线无法收集测试：`ModuleNotFoundError: omegaconf`，同次还报 `scripts` | 当前 shell 的 `pytest`/Python 环境不完整；先定位仓库可用虚拟环境或用正确解释器运行 |
| 隔离补入 OmegaConf 后，转换器导入链继续缺少 `diffusers` | 先静态审计导入链与当前模块可用性，再一次性安装最小依赖集合，避免逐包盲试 |
| 12-case preparation 一次性检查误用 sampling 键 `steps`，触发 `KeyError` | 这是检查脚本字段名错误，不是产品实现失败；读取真实 manifest schema 后使用 `sampling_steps` 重跑 |
| 直接运行仓库根 `pytest -q` 收集到 `fouroversix/` 可选 Modal 测试，并让其 `scripts` 包遮蔽项目根 `scripts` | 按仓库正式测试范围运行 `PYTHONPATH=$PWD ... -m pytest -q tests`，最终 270 passed、2 subtests passed；未安装/修改可选 Modal 环境 |

## 资源
- 任务文档：`stage1模型长视频推理实验.md`
- 必查入口：`infer_stage1.sh`
- 核心 runner：`scripts/run_stage1_training_checkpoints_validation.py`
- validation helper：`utils/stage1_causal_validation.py`
- 目标测试：`tests/test_stage1_causal_validation.py`、`tests/test_stage1_checkpoint_inference_validation.py`
- 独立需求审计与仓库审计均确认：无需修改 inference pipeline、EMA merge 或输出 validator。

## 视觉/浏览器发现
- 暂无。

---
*重要发现应在形成后及时更新。*
