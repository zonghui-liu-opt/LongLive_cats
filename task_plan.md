# 任务计划：LongLive‑2.0 Stage‑2 Self‑Forcing DMD/DFD

## 目标
完整落实 `TASK-stage2-self-forcing-dmd-dfd.md`：以Stage‑1 step3075 EMA为唯一Generator起点，连续完成训练、日志/权重/可视化、batch推理、trace与压缩/sink通用接口；保持Stage‑1与legacy DMD行为不回归，不再受旧检查点暂停规则约束。

## 当前阶段
Phase 19 已完成：内网异常已收敛为新trainer/旧Stage2DMD混合加载；v2 runtime API版本、精确签名双握手与wrapper source-path门禁已落地，完整Stage‑2 665项通过并重新发布`stage-2`。

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

### Phase 8：Stage-2 LongLive-2.0 Self-Forcing DMD/DFD
- [x] 只读审计 Stage-1 训练、推理、缓存、CFG、指标与现有 Stage-2 DMD 路径
- [x] 通过 grill-me 逐项锁定 baseline、数据、模型、损失、更新时钟、H100 与压缩消融方案
- [x] 审计并映射 Stage-1 loss/吞吐可视化实现
- [x] 生成可独立交给 Codex 执行的 Stage-2 任务文档
- [x] 复核任务文档覆盖全部锁定决策、P0 风险、验收标准与逐步验证
- 当时下一步（历史，现已完成）：按文档开始 Step 1；该条不再代表当前状态。
- **Status:** complete

### Phase 9：Stage‑2 按用户检查点实现
- [x] 9.1 完整阅读 Stage-2 任务文档、仓库状态与适用约束
- [x] 9.2 完成配置契约与 G/real/F 三角色初始化、LoRA/FSDP/manifest 基础
- [x] 9.3 创建独立 `stage-2` 分支，保护用户既有改动与 Stage-1 行为
- [x] 9.4 完成数据/negative/balanced sampler，以及合格F25原字节复用、F24从97帧源视频确定性重提的native F25准备链
- [x] 9.5 实现显式 1+24 pack、9750-token mixed timestep adapter（原 Step 4）
- [x] 9.6 原子完成 24-new rollout、W16/S1 reset、KV autograd 安全与 UniPC random exit（原 Steps 5–7）
- [x] 9.7 实现并解析验证 DMD、DFD、fake raw-flow 与 continuous-sigma loss（原 Step 8）
- [x] 9.8 重跑训练前准备全量本地测试和相关回归，完成两轮独立终审，并在第一个用户检查点停止
- [x] 9.9 用户检查通过后，写简洁中文 H100 指南并上传 `stage-2`；等待内网准备门禁结果
- [x] 9.10 实现严格5F→1G trainer、EMA/nonfinite、原子checkpoint、JSONL/plot（原 Steps 9–11），完成735项全仓本地回归并在第二个用户检查点停止
- [x] 9.11 旧的分批暂停/上传边界已由用户取消；完整H100 runbook已统一覆盖新版prepare与后续生命周期
- [x] 9.12 完成baseline batch推理、压缩/sink通用接口和全部本地验收（原 Steps 12–14）
- **Status:** complete（本地代码与验收完成；真实8×H100执行证据单列为外部边界）

### Phase 10：Stage‑1 LoRA/merged 四卡批量推理对比
- [x] 10.1 审计 Stage‑1 `adapter_ema.safetensors` 加载格式、现有 inference 分布式采样和6-case双分辨率约束
- [x] 10.2 设计并实现不会漏样本的4×H100数据并行调度，同时保持两种格式相同输入/seed/采样参数
- [x] 10.3 新增动态 LoRA batch inference，并输出 reference/merged/LoRA 映射与并排视频/HTML/report
- [x] 10.4 补齐正反测试、静态检查和 ffmpeg 调度/拼接验证
- [x] 10.5 审查最终改动范围并交付内网运行命令；不伪造正式H100推理结果
- [x] 10.6 将reference目录改为可选，缺省时从metadata和模型资产fresh prepare
- [x] 10.7 保持提供reference时的旧校验/历史视频兼容，并让无reference报告/HTML无悬空字段
- [x] 10.8 补齐fresh/reference双路径测试、shell/CLI/静态检查并交付新命令
- **Status:** complete（39 tests；正式4×H100性能/视频质量待用户内网执行）

### Phase 11：新增 Stage‑1 资产后的 Stage‑2 只读复审
- [x] 11.1 盘点新增 merged/runtime-LoRA 对比代码、配置文件与双向模型资产
- [x] 11.2 复核 real-score / fake-score 是否从同一双向 merged 权重独立初始化，并审计 manifest/provenance 门禁
- [x] 11.3 复核 Stage‑2 trainer、JSONL、checkpoint/resume 与 PNG/SVG/HTML 可视化是否受共享代码改动影响
- [x] 11.4 运行当前磁盘态的针对性与全量 CPU 回归、CLI/静态检查
- [x] 11.5 输出按 P0/P1 排序的结论和必须修改项；本轮不改生产代码或权重
- **Status:** complete（只读结论：核心训练闭环无新增P0/P1；正式YAML与teacher sidecar存在阻断项）

### Phase 12：重生成 real-score manifest 后的 Stage‑2 H100 指南
- [x] 12.1 审计当前 `prepare_stage2.sh`、600cats YAML、manifest/audit/preflight/train/plot CLI 与旧手册，建立真实命令和产物契约
- [x] 12.2 盘点所有面向操作者的 Stage‑2 指导入口，删除 manifest 后仍停留在旧检查点边界的过期步骤
- [x] 12.3 重写中文 H100 指南：manifest验收→正式资产审计→角色init-only→C0/C1/C2 smoke→全新正式训练→resume→九图/HTML验收
- [x] 12.4 增加或更新文档契约验证，确保文件名、参数、路径、成功标记和禁止事项与生产代码一致
- [x] 12.5 运行CLI help、shell syntax、文档命令静态审计、目标测试与最终diff检查
- **Status:** complete（443 passed；真实8×H100 smoke/formal仍待内网执行）

### Phase 13：前5项检查的一键入口
- [x] 13.1 将teacher、step3075 Generator、配置、F25/negative/formal cache、role init串为一个脚本
- [x] 13.2 支持安全复用已经验证过的昂贵产物，拒绝半成品或来源漂移
- [x] 13.3 把H100文档缩短为“填路径、运行、看5个PASS”
- [x] 13.4 更新契约测试并运行Stage-2相关回归
- **Status:** complete（437 passed；真实8×H100检查待内网执行）

### Phase 14：Stage‑2 F25 latent 专用重提入口
- [x] 14.1 核对 Stage‑1 F24 与 Stage‑2 F25 的像素帧、latent帧和训练slice契约
- [x] 14.2 审计两份新CSV的行数、字段、动作分布、video字符串和旧cache哈希兼容性
- [x] 14.3 新增只负责F25/attestation/negative/formal-audit的8×H100脚本
- [x] 14.4 增加无GPU输入契约与shell编排测试
- [x] 14.5 运行目标测试、shell/静态检查并交付内网命令
- **Status:** complete（443 passed；真实F25提取待内网600条完整输入与8×H100）

### Phase 15：Stage‑2 当前磁盘态重新严格审计
- [x] 15.1 完整重读任务书并把 Steps 9–14 映射到当前生产代码、测试和外部 H100 门禁
- [x] 15.2 独立审计训练状态机、loss/optimizer/EMA/nonfinite/checkpoint/resume 的语义一致性
- [x] 15.3 独立审计 JSONL、PNG/SVG/HTML 可视化和训练后 batch inference/trace 接口
- [x] 15.4 建立当前本地回归基线，修复所有可证实的 P0/P1/P2 缺陷并增加反例测试
- [x] 15.5 连续完成Stage‑2训练、可视化、推理与压缩接口的全部本地验收，不再按旧检查点停工
- **Status:** complete（Stage‑2 656 passed；正式tests/ 964 passed、2 subtests；最终只读审计P0=0/P1=0）

### Phase 16：Stage‑2 H100 单一指导脚本
- [x] 16.1 核对现有prepare、C0/C1/C2、formal、plot、inference的真实CLI与成功标志
- [x] 16.2 设计一个最小子命令接口，复用现有脚本和生产入口，不复制训练逻辑
- [x] 16.3 实现指导脚本及严格的环境/路径/断点/产物检查
- [x] 16.4 增加shell编排与文档契约测试，覆盖成功提示和危险误用
- [x] 16.5 运行shell、CLI、目标回归和diff门禁，交付只需复制执行的内网步骤
- **Status:** complete（目标回归27 passed；完整Stage‑2 662 passed；独立终审P0=0/P1=0；真实8×H100执行待内网）

### Phase 17：Stage‑2 H100 cross-KV smoke 故障定位
- [x] 记录异常栈、dirty worktree与真正中止点
- [x] 逐段核对`_preload_sink`、`_audit_cache`和cross-attention cache写入契约
- [x] 核对smoke配置、FSDP/BF16运行时与本地测试覆盖缺口
- [x] 用最小反例证明根因并评估修复风险
- [x] 给出修复方案、补丁建议与内网复验命令
- **Status:** complete（根因已用PyTorch FSDP2官方v2.8源码和本地最小反例双重证明；本轮按诊断请求未改production代码）

### Phase 18：实现 cross-KV FSDP2 修复并发布
- [x] 复核Git/GitHub分支、远程、认证和dirty worktree的精确提交边界
- [x] 先增加FSDP2双重容器重建反例测试
- [x] 实现轻量可变leaf state、严格验证与reset/audit语义
- [x] 运行聚焦、Stage-2全量与静态回归
- [x] 只暂存本轮文件，commit并push到GitHub `stage-2`
- **Status:** complete（聚焦55 passed；完整Stage‑2 663 passed；本轮文件精确发布；真实8×H100 smoke待内网复验）

### Phase 19：Stage‑2 smoke 全调用链接口闭环
- [x] 19.1 复现`timing_callback`失败并建立trainer→Stage2DMD全部调用/签名矩阵
- [x] 19.2 审计F/G、DMD/DFD、rollout/score、timing与返回值的所有动态接口漂移
- [x] 19.3 先补生产调用级红测，再最小修复全部已证实缺口
- [x] 19.4 运行聚焦、完整Stage‑2、训练状态机与静态回归
- [x] 19.5 精确审阅、commit并push GitHub `stage-2`
- **Status:** complete（联合142 passed；完整Stage‑2 665 passed；wrapper isolated API probe PASS；真实8×H100 smoke待内网复验）

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
| Stage‑2双动作每个样本/seed一次生成连续48个noise slots，A/B各取前/后24个 | 用户确认A/B使用独立随机起点，同时同一`(sample, seed)`必须确定性复现；禁止B前重置相同seed复用A noise |

## 遇到的错误
| 错误 | 尝试次数 | 解决方案 |
|------|---------|---------|
| 修改前目标 pytest 在收集阶段失败：当前 `pytest` 环境缺少 `omegaconf`，且未解析仓库 `scripts` 包 | 1 | 使用文档规定的 `PYTHONPATH="$PWD" python -m pytest`，并把缺失依赖装入 `/tmp` 隔离目录 |
| 隔离安装 OmegaConf 并按文档方式重跑后，收集继续因缺少 `diffusers` 失败 | 2 | 审计导入链后一次性补入 `diffusers==0.31.0`/easydict；基线最终 7 passed |
| 更新规划日志时一次补丁上下文误指向 `findings.md` 中不存在的“错误日志”段 | 1 | 读取三份规划文件的实际段落位置后拆分到正确文件 |
| 第二次规划更新补丁把 task plan 清单上下文误放进 `progress.md` | 1 | 重新读取三份规划文件并按文件分别应用正确上下文 |
| 新文件 whitespace 验收发现 shell 末尾多一个空行 | 1 | 删除多余 EOF 空行并重新运行全部验收命令 |
| 最终契约审计发现 shell 的未文档化 checkpoint override 可绕过固定 3750 | 1 | 删除 override，专用入口始终显式选择 `$LONG_LIVE_STAGE1_TRAIN_DIR/checkpoint_model_003750`；同时收紧 runbook 对 merged 保留时点的措辞 |
| Batch 2 搜索meta-init路径时首个`rg`组合正则缺少闭合括号 | 1 | 改用多个`-e`固定子表达式重新搜索，定位到仓库已有`accelerate.init_empty_weights`路径 |
| 直接导入`wan_5b.configs.WAN_CONFIGS`探查TI2V规格时本地缺少`easydict` | 1 | 不改变本地环境，改为只读`wan_5b/configs/wan_ti2v_5B.py`确认30层、dim3072、ffn14336、C48 |
| Stage‑2 exit RNG测试把“独立随机流”误写成“第一组排列必须不同” | 1 | 独立流仍可能偶然生成同一排列；改为比较RNG state，并分别验证分层覆盖 |
| 新增Stage‑2 rollout文件首次Black check需格式化 | 1 | 仅对两个新增文件运行Black机械格式化，再复跑静态与产品测试 |
| teacher checkpoint篡改负例同时改变了文件大小，先触发size门禁而非预期SHA门禁 | 1 | 将fixture改为等长字节篡改，分别独立验证size与SHA失败路径 |
| 首次格式检查误用当前Anaconda `python -m black`，该解释器未安装Black | 1 | 改用系统已安装的`black`/`ruff`可执行文件；根据报告机械格式化并清理新增代码lint |
| planning 完成检查脚本不识别中文版 `阶段/状态` 标记，首次报告 0/0 | 1 | 将 plan 标题/状态改为脚本支持的 `Phase`/`Status`，复查为 5/5 complete |
| 2026-08-11检查点B规划组合补丁误用了`findings.md`首行标题 | 1 | 读取三份文件真实首行后拆分补丁，改用`# 发现与决策`精确上下文 |
| Phase 6 首次 planning 补丁把 findings 决策行误用为 task_plan 上下文 | 1 | 读取各文件实际段落后按文件分别应用 |
| Phase 7 首次规划补丁的 findings 上下文与实际文本不一致 | 1 | 读取三个文件的准确段落后拆分补丁并使用精确上下文 |
| Phase 7 Step 1 完成记录的组合补丁再次因 progress 上下文校验失败 | 1 | 确认组合补丁未部分应用后，按文件拆分为精确小补丁 |
| sink1 cache 保留测试首次比较的 tensor rank 不一致 | 1 | 将 initial scalar reshape 为 cache slice shape，验证真实数值后 16 tests 通过 |
| 新增 CSV 的 no-index whitespace 检查把 CRLF 表头判为 trailing whitespace | 1 | 仅将正式 continuation CSV 的换行标准化为 LF；严格 loader/artifact/HTML 48 tests 复跑通过，随后 no-index 检查通过 |
| 最终审计发现 manifest 写前路径校验与 ordinary/session 互斥存在边界缺口 | 1 | 先新增逃逸路径与并发红测，再前移 canonical/containment 校验并让普通 inference、session、clear_cache 复用同一 RLock；相关 71 tests 通过 |
| 指定 Anaconda Python 没有 `black` 模块 | 1 | 使用系统已安装的 Black 可执行文件格式化本任务新文件，随后 Ruff 与 py_compile 通过 |
| Phase 9 首次组合 planning 补丁错误假设了 `findings.md` 的相邻标题 | 1 | 读取三个文件的真实标题位置后拆成精确小补丁，不重复原失败操作 |
| Batch 1 首次 Black check 发现两个新增Python文件需格式化 | 1 | 使用仓库现有Black机械格式化后再运行Black/Ruff/py_compile/diff-check；产品逻辑测试当时已71 passed |
| Batch 1 格式化后的首次Ruff检查发现测试文件未使用`math`导入 | 1 | 删除单个无用导入后重跑完整静态检查，不使用自动修复扩大修改面 |
| Batch 1 review修订的大组合补丁因Black后的精确上下文不匹配失败 | 1 | 确认未发生部分写入后，按职责拆成小型精确补丁逐项应用 |
| review修订后误用系统`pytest`可执行文件，Python3.11环境缺少OmegaConf而收集失败 | 1 | 核对shebang后改用项目依赖所在的Anaconda Python执行`python -m pytest`；96个Stage-2测试通过 |
| review修订后的Black check报告两个Python文件需重新格式化 | 1 | 仅运行Black机械格式化，再跑Ruff、py_compile、测试与diff检查 |
| 将`git diff --no-index --check`直接串入成功链时，正常“文件不同”退出码1被误当失败 | 1 | 对每个未跟踪文件单独接受0/1，仅把大于1视为检查异常；空白审计通过 |
| 文档编辑中的dirty checkout直接请求formal cache/F25 CLI `--help`，被物理clean-checkout前置门禁拒绝 | 1 | 不绕过门禁；读取当前parser源码并依赖CLI测试，正式命令只允许在clean clone/提交态运行 |
| 首次记录上述错误的组合补丁误用了task_plan中不存在的历史错误行上下文 | 1 | 先定位真实表格尾部，再用精确上下文补写；未发生部分修改 |
| prepare脚本首次补丁给`$STAGE2_PYTHON`命令名误加了单引号，形成字面量命令 | 1 | 静态复核后立即改为双引号变量展开，随后用`bash -n`和stub集成测试验证 |
| 新runbook测试首次运行因手册在“不要锁死历史计数”的反例句中仍出现`329 passed`而失败 | 1 | 删除具体历史数字、保留原则性说明；产品命令与流程未失败 |
| 新增runbook测试首次Black check要求格式化 | 1 | 仅对新增测试运行Black机械格式化，再复跑静态和目标测试 |
| Phase 18红测首次收集因`utils.stage2_cross_kv`尚不存在而失败 | 1 | 确认测试锁定缺失契约后新增非dataclass leaf state并复跑，聚焦测试通过 |
| Phase 18全文件Black/Ruff检查命中`causal_model.py`既有格式与7项lint债务 | 1 | 不扩大提交；修正本轮测试格式，严格检查其余文件，并对比HEAD证明causal_model当前仍是相同7项、无新增lint |
| planning完成检查脚本没有可执行位，直接调用被拒绝 | 1 | 保留脚本不改权限，改由`bash`显式执行同一只读检查 |
| Phase 19 runtime API红测首次收集因审计helper不存在而失败 | 1 | 确认启动期握手覆盖缺口后实现版本/签名双重审计，目标测试转为通过 |
| 一次组合pytest命令写了两个`-k`，后者覆盖前者而只运行3项loss测试 | 1 | 不把该结果冒充trainer覆盖；拆开命令后单独运行runtime API/smoke options 6项并通过 |
| Phase 19首次Black check要求格式化trainer新增审计代码 | 1 | 先记录再仅对该任务文件机械格式化，随后重跑全部静态门禁 |

## 备注
- 重大决策前重新读取本计划。
- 所有任务文档要求都要对应到实现或显式说明的外部限制。
- 规格 Step 10（8.2 技术验收）明确要求由用户在内网 H100 执行；本地没有伪造视频、pass 报告或视觉效果结论。
- Phase 6 的备注只针对上一轮 uniform-prompt 任务：其旧 16-case/双顺序矩阵已被 12-case 三风格覆盖；不影响本轮 continuation 固定的 8-row、双顺序、16-video 验收口径。
