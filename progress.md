# 进度日志

## 会话：2026-08-04

### Phase 7：Stage-1 有状态 Continuation Inference
- **状态：** complete（本地；H100 Step 10 待用户执行）
- 执行的操作：
  - 完整读取 `planning-with-files-zh` 技能并恢复既有 task plan/findings/progress。
  - 完整读取 351 行 `TASK-stage1-continuation-inference.md` 与当前 `git status --short --branch`。
  - 将新目标、10 个实现/验证子阶段和关键兼容性决策写入持久化计划。
  - 完整读取旧 `infer_stage1_two_actions_10s.sh` 与 670 行 `inference.py`，确认旧入口固定 uniform-prompt runner，模型 bootstrap 与逐样本执行均集中在 `inference.py`。
  - 完整读取 1048 行 `pipeline/causal_diffusion_inference.py`，定位普通 inference cache reset、per-block scheduler、I2V clamp、clean recache、cross-attn reset与统一 decode 的现有实现。
  - 阅读 `wan_5b/modules/causal_model.py` 前 1300 行，核实 `local_attn_size=-1` 的真实 24-latent rolling、legacy leading sink、token-unit indices 与 cross-attn cache bypass。
  - 完整读完 1881 行 causal model，确认 cache 更新由现有模型在所有 transformer block forward 后统一提交，session 只需连续传入严格 cursor 并验证索引。
  - 完整读取 807 行 Stage-1 causal validation helper，确认可复用图片/视频探测、原技术阈值和原子 IO，但新 continuation 输出布局需独立严格映射。
  - 完整读取 testset preparer、prompt batching 测试与 716 行 Wan wrapper，确认 eager cache metadata publication、T5 tokenizer 512 固定编码和整段 VAE decode 的复用入口。
  - 完整读取 747 行 checkpoint validation runner 与 417 行 causal validation 测试，确认 EMA merge/report/失败保留可直接复用，continuation 只需替换数据准备、推理映射与 HTML。
  - 完整读取 802 行 checkpoint runner 测试、旧 causal runner 与 tokenizer 实现，确定未截断 token audit 的精确调用方式和无 CUDA runner 测试 seam。
  - 审计新 8-row continuation CSV 与全仓 inference 测试覆盖，发现 `soft_reanchor` 当前为非法大写 `TRUE`，且普通 pipeline inference 尚无直接 characterization 测试。
  - 检查本机测试解释器：torch/pytest 可用，但 OmegaConf、diffusers、easydict 缺失，且旧 `/tmp` 隔离依赖已消失；后续会重建隔离测试依赖。
  - 审阅 pipeline lazy import、prompt-conditioning helper 与既有 fake-scheduler 测试模式，确定可在 CPU 上构建无模型 tiny pipeline characterization。
  - 新增 `tests/test_causal_diffusion_inference_characterization.py`，先不改 production；精确锁定普通 inference 签名、pos/neg cache reset、global/cache start、prompt mapping、initial clamp、clean recache 与每 block 独立 50-step UniPC 初始化。
  - 新 characterization 单独 2 passed；与既有 prompt batching/Stage-1 validation 联合为 32 passed。独立审计记录生产修改前全测试基线 159 passed、2 subtests。
  - 新增严格 continuation metadata dataclass/loader，复用现有 Stage-1 图片、路径、尺寸和哈希 helper；覆盖 exact header、matrix、prompt/HOLD/action 语义与所有主要反例。
  - 将正式 CSV 的 8 个 `soft_reanchor=TRUE` 修正为 canonical lowercase `true`；新测试 16 passed，新旧 loader 联合回归 29 passed。
  - 从普通 `_inference_inner()` 抽取共享 block denoise/anchor/clean-recache 内核；characterization 与 prompt batching 5 passed，旧调用序列不变。
  - 新增独占式 `ContinuationSession`：session-owned pos/neg KV、全层 cursor/capacity/pinned 审计、A/B anchors、prompt token audit、noise hash、失败 poisoning、统一 finish decode 和普通 inference 隔离。
  - 新增 16 个 session/token/lifecycle/sink 测试；continuation 核心联合测试 37 passed。
  - 两路只读核心审查发现并推动修复普通 I2V initial dtype/device 兼容、session sampling/sink/RoPE 锁定、调用入口 cache 审计和全局 cache_start；S=3 与真实 CausalWan attention 测试锁住 token-unit cursor、anchor step 前后 clamp 和 legacy sink1 K/V。
  - 抽取共享单视频技术门禁并保持旧 13 个 causal validation 测试通过；新增 exact 16 MP4/16 trace validator，只在整轮成功后原子回写 technical pass。
  - 新增 continuation preparation manifest、两 geometry config 和 `inference.py` 显式分支；每 case 一次 image encode/一次 64-latent noise sampling，sink0/1 分别使用 clone 并按 24/16/24 切片。
  - 新增 step3750 专用 runner，直接复用 converted-base 审计、checkpoint validator、merge config 和 EMA merge；成功删除 merged，command/output/HTML 失败均保留 merged 与原子失败报告。
  - 新增严格 8×2 同步 HTML、专用 shell 和中文 H100 快速部署/实验文档；旧 `infer_stage1_two_actions_10s.sh` 无 diff。
  - continuation 九份测试模块最终合计 111 passed；`PYTHONPATH=$PWD ... -m pytest -q tests` 全仓正式范围最终回归 270 passed、2 subtests passed，仅有 14 条既有 torch.jit deprecation warning。
  - 两路最终只读审计发现并推动修复 prepared manifest 写前路径逃逸、ordinary/session 互斥 TOCTOU 与 trace 字段门禁不足；新增重签名逃逸、并发锁、正负 prompt audit、locked runtime 等失败测试。
  - 修复后独立生产与规格复核均未发现残留 P0/P1/P2；runner/shell/runbook 的 step3750、失败保留与 H100 操作边界完成复核。
- 当前工作：
  - 本地 Step 1–9 已完成；下一步由用户按中文 runbook 在内网 H100 执行 Step 10 与 8.2 技术/人工验收。
- 创建/修改的文件：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `TASK-stage1-continuation-inference.md`
  - `tests/test_causal_diffusion_inference_characterization.py`
  - `utils/stage1_continuation_validation.py`
  - `tests/test_stage1_continuation_validation.py`
  - `testsets/metadata_8cases_two_actions_continuation_480x832_253frames.csv`
  - `pipeline/causal_diffusion_inference.py`
  - `pipeline/causal_diffusion_continuation.py`
  - `tests/test_causal_diffusion_continuation_session.py`
  - `tests/test_causal_wan_attention_continuation_sink.py`
  - `utils/stage1_continuation_inference.py`
  - `utils/stage1_continuation_report.py`
  - `scripts/run_stage1_continuation_validation.py`
  - `tests/test_stage1_continuation_artifacts.py`
  - `tests/test_stage1_continuation_inference_orchestration.py`
  - `tests/test_stage1_continuation_report.py`
  - `tests/test_stage1_continuation_runner.py`
  - `tests/test_stage1_continuation_entrypoint.py`
  - `infer_stage1_two_actions_continuation_10s.sh`
  - `docs/STAGE1_CONTINUATION_INFERENCE_H100_ZH.md`

### Phase 6：12-case 三风格实验改版
- **状态：** complete
- 执行的操作：
  - 读取既有 planning 文件并恢复已完成实现、测试与 Git 推送上下文。
  - 将用户新决策固化为 12-case、单一 `jump_then_toy`、三种 prompt 风格。
  - 将 runner/HTML 扩展为 `absolute_timeline`、`sequential`、`phase_relative` 三列，同时保持默认 checkpoint HTML helper 不变。
  - 删除旧 16-case CSV，生成新 12-case CSV，并同步 shell、runbook 和严格 metadata 契约测试。
  - 两份目标测试修改前后均为 27 passed；实际 12-case fake-carrier preparation、shell、CLI、语法、whitespace、旧 CSV hash 与默认 HTML 兼容性检查全部通过。
- 创建/修改的文件：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `testsets/metadata_12cases_two_actions_480x832_253frames.csv`
  - `scripts/run_stage1_training_checkpoints_validation.py`
  - `tests/test_stage1_checkpoint_inference_validation.py`
  - `infer_stage1_two_actions_10s.sh`
  - `docs/STAGE1_TWO_ACTION_LONG_INFERENCE_ZH.md`

### 错误日志
- 最终审计新增的 9 个路径/trace/并发测试先按预期失败；前移 manifest 路径门禁、统一 RLock 并收紧 trace validator 后全部转绿，随后扩展到 111 个 continuation tests。
- 指定 Anaconda Python 不含 `black` 模块；改用系统已安装的 Black 可执行文件，仅格式化本任务新文件，Ruff 与 py_compile 随后通过。
- continuation 正式 CSV 的 no-index whitespace 检查发现 CRLF 表头会被判为 trailing whitespace；仅标准化为 LF 后，metadata/artifact/HTML 48 tests 与全部新增文件 no-index 检查通过。
- 直接从仓库根运行无范围 `pytest -q` 会额外收集 `fouroversix/` 的可选 Modal 测试并发生顶层 `scripts` 包遮蔽，导致 10 个 collection errors；按仓库正式范围改为 `pytest -q tests` 后 260 passed、2 subtests passed。
- 首次同时更新 task plan 与任务文档 Step 3–5 时只给了 checkbox 前缀上下文，因实际整句同行而补丁未匹配；读取精确行后用完整上下文更新成功。
- session sink1 测试首次把 `[1,1,1,1]` cache slice 与 `[1,1,1,1,1]` initial slice直接比较，因 shape 不同失败；改为同形 scalar view 后重跑 16 passed。
- 首次记录 Phase 7 Step 1 完成状态的组合补丁因 progress 上下文校验失败；确认未部分应用后按文件拆分成功。
- 首次 Phase 7 planning 补丁的 findings 上下文与实际文本不一致；读取准确段落后拆分补丁成功。
- 首次 Phase 6 planning 补丁上下文混用了 task_plan/findings；读取实际段落后重新应用。

## 会话：2026-08-03

### 阶段 1：需求与仓库发现
- **状态：** complete
- 执行的操作：
  - 完整读取 `planning-with-files-zh` 技能说明与模板。
  - 检查历史规划文件和会话恢复状态；当前为新计划。
  - 完整阅读 252 行任务规格，提炼固定实验矩阵、CLI 扩展、metadata、HTML、shell、文档和验收要求。
  - 盘点仓库文件与 `git status --short --branch`，记录并隔离用户现有工作区改动。
  - 完整阅读现有 `infer_stage1.sh` 与 testset preparer，确认可复用的环境变量和参数透传路径。
  - 完整阅读 checkpoint validation runner 和 causal validation helper，定位固定 24-frame 参数、两次 metadata load、HTML row-id 映射及现成 253-frame 验证能力。
  - 完整阅读两份目标测试，运行修改前基线并审计测试导入链；隔离补入 OmegaConf 后确认还需 diffusers/easydict。
  - 使用隔离依赖与项目规定的 PYTHONPATH 完成修改前基线：7 passed，2 个既有 SWIG deprecation warnings。
  - 记录原 6-case CSV SHA256，阅读旧 checkpoint runbook，并确认新入口需保留现有空目录/失败现场/merged 清理约定。
- 创建/修改的文件：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### 阶段 2：方案与任务拆解
- **状态：** complete
- 执行的操作：
  - 完成需求审计与仓库审计交叉校验，确认最小改动范围和所有 fail-fast 边界。
  - 决定 HTML 通用 group 校验 + 正式 metadata 固定 8 组的分层验证方案。
- 创建/修改的文件：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`

### 阶段 3：实现
- **状态：** complete
- 执行的操作：
  - 确定专用 shell 的 Python、GPU、checkpoint 与显式空 work-dir 参数约定。
  - 新建专用 H100 shell：显式 3750 checkpoint、64 latent、重复图片 opt-in、prompt-style、50/5.0/seed1；默认 GPU 0 且允许覆盖，拒绝非空 work dir。
  - 新建中文 H100 快速运行文档，覆盖环境、命令、产物、253-frame 门禁、HTML 人工审阅与失败现场策略。
  - 运行 `bash -n` 和两份新文件的 `git diff --check`，均通过。
  - 完成两条并行代码 diff 主审；确认旧 HTML helper 原样保留、重复图片仅显式放宽、64 latent 全链透传、prompt-style schema/row-id/相对链接门禁完整。
  - 写入正式 16-case CSV，并用正式 loader + 独立静态规则验证 16/4/8×2 计数、geometry、prompt 契约和旧 CSV SHA256，全部通过。
  - 把正式 metadata 契约固化到目标测试中；runner 测试 14 passed，causal validation 测试 13 passed。
- 创建/修改的文件：
  - `infer_stage1_two_actions_10s.sh`
  - `docs/STAGE1_TWO_ACTION_LONG_INFERENCE_ZH.md`
  - `testsets/metadata_16cases_two_actions_480x832_253frames.csv`
  - `utils/stage1_causal_validation.py`
  - `scripts/prepare_stage1_causal_testsets.py`
  - `scripts/run_stage1_training_checkpoints_validation.py`
  - `tests/test_stage1_causal_validation.py`
  - `tests/test_stage1_checkpoint_inference_validation.py`

### 阶段 4：测试与验证
- **状态：** complete
- 执行的操作：
  - 完整目标测试联合运行：27 passed，只有 2 条第三方 SWIG deprecation warnings。
  - `bash -n`、runner `--help`、Python syntax、`git diff --check` 及三份新文件的 no-index whitespace 检查全部通过。
  - 用真实 16-case metadata 完成 64-latent fake-carrier preparation 集成验证，两个 bucket 的 shape/count/frame/attention/sink/sampling 全部通过。
  - 执行专用 shell dry contract：精确参数、无 forbidden override、固定 3750、非空 work dir 拒绝且现场未删除、文件模式 755，全部通过。
  - 对比 Git HEAD，旧 `_comparison_html` 源码逐字节一致；旧 6-case CSV SHA256 未变化。
  - 完成独立 prompt 与 shell/runbook 审计；修复隐藏 checkpoint override 和失败现场措辞边界后重新跑全套验收。
  - 修复后再做独立 production code review，结论为无 P0–P2 correctness/spec 问题。
- 创建/修改的文件：
  - 仅任务范围内的 production、test、metadata、shell、runbook 与 planning 文件。

### 阶段 5：交付
- **状态：** complete
- 执行的操作：
  - 审查最终 git status，确认用户既有 config 修改、删除文件和未跟踪文件均未被恢复或覆盖。
  - 准备中文交付摘要和内网 H100 下一步入口。
- 创建/修改的文件：
  - 无额外代码变更。

## 测试结果
| 测试 | 输入 | 预期结果 | 实际结果 | 状态 |
|------|------|---------|---------|------|
| 修改前目标 pytest 基线 | 两份 Stage-1 validation 测试 | 完成测试收集并报告既有状态 | 收集失败：缺少 `omegaconf`，同时未找到 `scripts` 包 | BLOCKED（环境） |
| 修改前目标 pytest 基线（正确环境） | `PYTHONPATH=$PWD:/tmp/... python3 -m pytest -q` 两份测试 | 全部既有测试通过 | 7 passed，2 warnings | PASS |
| causal validation 实现测试 | `tests/test_stage1_causal_validation.py` | repeat opt-in、旧门禁、24/64 preparation 全通过 | 13 passed，2 warnings | PASS |
| checkpoint runner/metadata 实现测试 | `tests/test_stage1_checkpoint_inference_validation.py` | CLI、透传、HTML/schema、正式 CSV 全通过 | 14 passed，2 warnings | PASS |
| 最终目标联合测试 | 规格指定的两份 pytest | 全部通过 | 27 passed，2 warnings | PASS |
| 真实 metadata preparation 集成 | 16-case CSV、64 latent、fake carrier | 两个 bucket 均为 253-frame 锁定配置 | PASS | PASS |
| shell/CLI/whitespace | `bash -n`、runner `--help`、Python syntax、diff checks | 全部成功 | PASS | PASS |
| 默认兼容性 | HEAD/current `_comparison_html` + 旧 CSV SHA256 | HTML helper 逐字节相同，CSV hash 相同 | PASS | PASS |
| Phase 6 修改前回归 | 两份 Stage-1 validation 测试 | 旧实现保持通过 | 27 passed，2 warnings | PASS |
| Phase 6 目标回归 | 12-case metadata、三风格 runner/HTML 与两份 validation 测试 | 全部通过 | 27 passed，2 warnings | PASS |
| Phase 6 实际 metadata preparation | 12-case CSV、64 latent、fake carrier | 3 landscape + 9 portrait，253 frames，锁定采样/attention 配置 | PASS | PASS |
| Phase 6 静态与兼容性 | shell、CLI help、py_compile、diff checks、默认 HTML、旧 6-case hash | 全部成功且旧行为不漂移 | PASS | PASS |
| Phase 7 continuation 目标测试 | 9 个 continuation/characterization/runner/HTML 测试模块 | session、sink、noise、trace、orchestration、DOM 全通过 | 111 passed | PASS |
| Phase 7 全仓正式范围回归 | `PYTHONPATH=$PWD /Users/zonghuiliu/anaconda3/bin/python -m pytest -q tests` | 所有项目 tests 通过 | 270 passed、2 subtests passed、14 warnings | PASS |
| Phase 7 最终静态验收 | shell syntax、runner help、py_compile、Ruff、tracked/untracked whitespace、旧 shell diff | 全部成功且旧入口不变 | PASS | PASS |

## 错误日志
| 时间戳 | 错误 | 尝试次数 | 解决方案 |
|--------|------|---------|---------|
| 2026-08-03 | 目标 pytest 收集阶段缺少 `omegaconf`/`scripts` | 1 | 检查解释器与已有环境，改用项目兼容 Python |
| 2026-08-03 | 正确 PYTHONPATH + OmegaConf 后缺少 `diffusers` | 2 | 审计导入链后补齐最小测试依赖 |
| 2026-08-03 | 规划日志补丁上下文不匹配 | 1 | 查看准确段落后重新应用 |
| 2026-08-03 | 第二次规划更新补丁把 task plan 清单上下文误放进 progress 文件 | 1 | 重新读取三份规划文件并按文件分别更新 |
| 2026-08-03 | 新 shell 的 no-index whitespace 检查发现 EOF 多余空行 | 1 | 删除空行后重新执行完整检查 |
| 2026-08-03 | 独立审计发现专用 shell 可被隐藏环境变量改到非 3750 checkpoint | 1 | 移除该 override 并精确化失败现场文档 |
| 2026-08-03 | planning 完成检查脚本与中文模板标记不兼容，首次显示 0/0 | 1 | 使用兼容标记后检查为 5/5 complete |
| 2026-08-04 | 12-case preparation 检查误读 sampling manifest 键名为 `steps` | 1 | 查看真实 sampling schema，改用 `sampling_steps` 后重跑 |
| 2026-08-04 | 记录上述错误的首次补丁上下文与 progress 实际行不一致 | 1 | 读取文件尾部后使用准确上下文补写 |
| 2026-08-04 | 最终审计暴露 manifest 写前路径与 inference/session 并发边界 | 1 | 先写失败测试，再实现写前路径门禁与全调用周期 RLock；独立复核无残留 P0/P1 |
| 2026-08-04 | Anaconda Python 无 `black` 模块 | 1 | 改用现有系统 Black 可执行文件并以 Ruff/py_compile 复核 |

## 五问重启检查
| 问题 | 答案 |
|------|------|
| 我在哪里？ | 本地实现、测试与交付准备已完成 |
| 我要去哪里？ | 用户按 runbook 在内网 H100 执行规格 Step 10 与 8.2 验收 |
| 目标是什么？ | 完整落实长视频推理实验任务文档 |
| 我学到了什么？ | 见 `findings.md` |
| 我做了什么？ | 见上方记录 |

---
*每个阶段完成后或遇到错误时更新此文件。*
