# 任务计划：Stage 1 模型长视频推理实验

## 目标
完整落实 `TASK-stage1-continuation-inference.md` 中的有状态 continuation inference 要求，形成可运行、可验证、可复现的实现、测试和内网 H100 快速部署文档，同时保持已有 uniform-prompt 推理语义不变。

## 当前阶段
Phase 7：本地实现与验证完成，等待用户执行 H100 Step 10

## 各阶段

### Phase 1：需求与仓库发现
- [x] 完整阅读任务文档
- [x] 盘点仓库结构、现有实现与项目约束
- [x] 提炼验收标准并记录到 findings.md
- **Status:** complete

### Phase 2：方案与任务拆解
- [x] 将文档要求映射到具体文件和实现步骤
- [x] 确定最小可靠的技术方案与验证方式
- [x] 吸收独立需求/仓库审计结果并完成交叉校验
- [x] 记录最终实现顺序与边界
- **Status:** complete

### Phase 3：实现
- [x] Step 2：重复首帧 opt-in 与正反测试
- [x] Step 3：runner 64-latent 参数化与默认回归测试
- [x] Step 4：prompt-style schema/HTML 与错误门禁测试
- [x] Step 5：16-case metadata 与静态契约测试
- [x] Step 6：专用 shell 与中文 H100 runbook
- [x] 每个子模块完成后执行针对性检查
- [x] 保留并兼容仓库中已有的用户改动
- **Status:** complete

### Phase 4：测试与验证
- [x] 运行规格列出的完整目标测试、shell、CLI help 和 diff 检查
- [x] 对照任务文档逐项验收
- [x] 修复验证中发现的问题
- **Status:** complete

### Phase 5：交付
- [x] 审查最终 diff 和交付文件
- [x] 更新使用说明与实验复现说明
- [x] 汇总实现结果、测试证据和剩余限制
- **Status:** complete

### Phase 6：12-case 三风格实验改版
- [x] 建立当前 27-test 回归基线
- [x] 将 prompt-style schema/HTML 从两种风格扩展为三种风格
- [x] 将 metadata 改为 4 cats × jump_then_toy × 3 styles = 12 rows
- [x] 同步 shell、runbook 与自动测试中的文件名、计数和布局
- [x] 运行目标测试、metadata/preparation、shell/CLI/diff 全量验收
- **Status:** complete

### Phase 7：Stage-1 有状态 Continuation Inference
- [x] 完整阅读 continuation 任务规格与当前 Git 状态
- [x] 7.1 阅读全部指定入口、pipeline/model/helper/test，建立普通 inference characterization 与回归基线
- [x] 7.2 实现并验证严格的 8-row continuation metadata loader
- [x] 7.3 抽取并验证普通 inference/session 共用的 block-generation 内核
- [x] 7.4 实现并验证 session/state 生命周期、cache/cursor/RoPE 连续性与 fail-fast
- [x] 7.5 实现并验证 initial anchor 与 B soft re-anchor
- [x] 7.6 实现并验证 sink0/1、共享 noise plan 与 identity hash
- [x] 7.7 实现统一 decode、session trace、技术门禁与失败原子报告
- [x] 7.8 实现独立 runner/shell/8x2 同步 HTML，保持旧入口不变
- [x] 7.9 运行 continuation 测试、相关全量回归与静态验收
- [x] 7.10 编写简洁中文 H100 快速部署/实验文档并完成最终审计
- **Status:** complete

## 关键问题
1. 任务文档规定了哪些明确交付物和验收指标？
2. 仓库当前已有多少可复用实现，哪些部分需要补齐？
3. 是否存在依赖外部模型、数据或算力而无法在本机完成的验证？

## 已做决策
| 决策 | 理由 |
|------|------|
| 先建立持久化计划，再阅读和实现 | 任务预计包含多阶段实验与多次验证，需防止遗漏 |
| prompt-style helper 通用验证“每组两种 style”，由本任务 metadata 测试严格断言 8 组 | 同时满足最小通用扩展和本实验固定矩阵，避免 runner 永久硬编码一份 CSV |
| 64 latent 继续传 `minimum_source_frames=97` | helper 使用 `max(97, pixel_frames)`，因此 64 latent 自然得到 253 carrier frames且不改变 24-latent 默认 |
| 新 shell 使用 `${CUDA_VISIBLE_DEVICES:-0}`，不传 solver/attention/sink/VAE 覆盖 | 未设置时单 GPU 0，保留用户覆盖；UniPC 和锁定 attention 继续由既有 config 语义保证 |
| 新增 `phase_relative` 作为第三种 prompt 风格 | 复用训练中见过的 0–4 秒局部时间词，同时保留 absolute/sequential 两个诊断对照 |
| 新矩阵只保留 `jump_then_toy` | 按用户要求将总样本从 16 降到 12，控制 H100 推理成本 |
| continuation 作为新独立入口实现 | 规格要求保留旧 64-latent uniform-prompt 实验语义，禁止静默改写 |
| 先用 characterization 锁住普通 inference 再抽内核 | session 重构涉及 KV/cache/scheduler/anchor，必须用测试约束旧路径兼容性 |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| 修改前目标 pytest 在收集阶段失败：当前 `pytest` 环境缺少 `omegaconf`，且未解析仓库 `scripts` 包 | 1 | 使用文档规定的 `PYTHONPATH="$PWD" python -m pytest`，并把缺失依赖装入 `/tmp` 隔离目录 |
| 隔离安装 OmegaConf 并按文档方式重跑后，收集继续因缺少 `diffusers` 失败 | 2 | 审计导入链后一次性补入 `diffusers==0.31.0`/easydict；基线最终 7 passed |
| 更新规划日志时一次补丁上下文误指向 `findings.md` 中不存在的“错误日志”段 | 1 | 读取三份规划文件的实际段落位置后拆分到正确文件 |
| 第二次规划更新补丁把 task plan 清单上下文误放进 `progress.md` | 1 | 重新读取三份规划文件并按文件分别应用正确上下文 |
| 新文件 whitespace 验收发现 shell 末尾多一个空行 | 1 | 删除多余 EOF 空行并重新运行全部验收命令 |
| 最终契约审计发现 shell 的未文档化 checkpoint override 可绕过固定 3750 | 1 | 删除 override，专用入口始终显式选择 `$LONG_LIVE_STAGE1_TRAIN_DIR/checkpoint_model_003750`；同时收紧 runbook 对 merged 保留时点的措辞 |
| planning 完成检查脚本不识别中文版 `阶段/状态` 标记，首次报告 0/0 | 1 | 将 plan 标题/状态改为脚本支持的 `Phase`/`Status`，复查为 5/5 complete |
| Phase 6 首次 planning 补丁把 findings 决策行误用为 task_plan 上下文 | 1 | 读取各文件实际段落后按文件分别应用 |
| Phase 7 首次规划补丁的 findings 上下文与实际文本不一致 | 1 | 读取三个文件的准确段落后拆分补丁并使用精确上下文 |
| Phase 7 Step 1 完成记录的组合补丁再次因 progress 上下文校验失败 | 1 | 确认组合补丁未部分应用后，按文件拆分为精确小补丁 |
| sink1 cache 保留测试首次比较的 tensor rank 不一致 | 1 | 将 initial scalar reshape 为 cache slice shape，验证真实数值后 16 tests 通过 |
| 新增 CSV 的 no-index whitespace 检查把 CRLF 表头判为 trailing whitespace | 1 | 仅将正式 continuation CSV 的换行标准化为 LF；严格 loader/artifact/HTML 48 tests 复跑通过，随后 no-index 检查通过 |
| 最终审计发现 manifest 写前路径校验与 ordinary/session 互斥存在边界缺口 | 1 | 先新增逃逸路径与并发红测，再前移 canonical/containment 校验并让普通 inference、session、clear_cache 复用同一 RLock；相关 71 tests 通过 |
| 指定 Anaconda Python 没有 `black` 模块 | 1 | 使用系统已安装的 Black 可执行文件格式化本任务新文件，随后 Ruff 与 py_compile 通过 |

## 备注
- 重大决策前重新读取本计划。
- 所有任务文档要求都要对应到实现或显式说明的外部限制。
- 规格 Step 10（8.2 技术验收）明确要求由用户在内网 H100 执行；本地没有伪造视频、pass 报告或视觉效果结论。
- Phase 6 的备注只针对上一轮 uniform-prompt 任务：其旧 16-case/双顺序矩阵已被 12-case 三风格覆盖；不影响本轮 continuation 固定的 8-row、双顺序、16-video 验收口径。
