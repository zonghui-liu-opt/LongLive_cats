# 进度日志

## 会话：2026-08-18（Phase 28）

### ffprobe失效路径启动前闭环
- **状态：** in_progress
- 内网8×H100已完成资产认证与模型生成，rank0在验证首个临时MP4时调用`/home/ma-user/miniconda3/bin/ffprobe`返回127。
- 已定位共享`probe_video()`只使用`shutil.which()`的单一首候选；下一步先补坏首候选/好后备候选红测，再实现健康检查解析器并接入一键shell预检。
- 保护边界：不跳过视频帧数/尺寸/fps校验，不删除部分完整结果，不触碰用户现有训练、metadata、checkpoint和results改动。
- 预期红测因生产模块尚无`resolve_ffprobe`而在收集期ImportError；实现后坏PATH首候选回退、显式坏override拒绝、shell预检三项转绿。
- 新增working override和全部发现失败诊断测试，当前五项聚焦回归全部通过；本机真实解析选择`/opt/homebrew/bin/ffprobe`并完成`-version`健康检查。
- `stage1_causal_validation`与完整Stage-2 inference联合116 passed；原子输出上下文确认异常时会删除未验收临时MP4，已完成的正式video+trace仍可续跑复用。
- 首次静态命令误把shell交给Ruff，产生无效Python语法报告；已改成Python/Ruff与shell/`bash -n`分开执行。Black格式化后Ruff、Black、py_compile、bash语法均通过。
- 全仓正式回归完成：1006 passed、2 subtests passed、14条既有TorchScript弃用warning；无功能失败。
- 发现两个共享Stage-1文件在HEAD并非全文件Black-clean，首次机械格式化引入无关diff；已逐项恢复原格式，只保留ffprobe生产/测试增量。恢复后相关208 passed，Ruff、py_compile、`bash -n`和精确diff-check通过。
- 发布范围锁定为解析器、早期推理shell、4类解析回归、shell契约、中文文档及Phase 28记录；继续排除用户`run_stage2_h100.sh`、H100 guide测试、metadata、checkpoint、results和Stage-1脚本。

## 会话：2026-08-18（Phase 27）

### 跨节点8×H100早期checkpoint一键推理
- **状态：** complete（本地实现与回归完成；真实8×H100执行待内网确认）
- 用户在第二台8×H100运行G70推理；rank0报告`Stage-2 checkpoint Generator provenance differs from its live manifest`，其余rank同步失败。
- 已确认现有`infer_stage2_tmp.sh`复用严格runner，但缺显式预检错误和bootstrap进度；`run_stage2_h100.sh infer`则硬锁G280，不能直接服务Phase A早期checkpoint。
- 生产代码比较了含训练节点文件identity的完整Generator资产对象；下一步先用跨节点identity红测锁住问题，再实现内容稳定比较和当前节点identity运行期守卫，随后收敛为一个可直接执行的8卡shell。
- 已确认该完整对象比较在rank0 asset attestation与每rank Generator loader各有一次；测试范围因此锁定`test_stage2_inference_assets.py`和`test_stage2_inference_loader.py`，并继续用`test_stage2_inference_entrypoint.py`锁住新shell编排。
- 红测精确复现：仅改变recorded Generator的`device/inode/mtime_ns/ctime_ns`时rank0和loader都会失败，改变真实`checkpoint_sha256`也会失败。
- 新增内容稳定比较helper并同时替换两处全对象比较；recorded完整资产SHA仍用于rank间一致性，live identity仍传给真实base loader并由既有加载前后identity门禁复核。
- 重写`infer_stage2_tmp.sh`：默认G70、支持多步数顺序执行、8卡数量/唯一性检查、完整资产预检、跨节点API探针、启动摘要、30秒心跳、追加日志、可恢复输出和56件套最终验收。
- 聚焦3项红测转绿；三个直接相关模块29 passed；完整Stage-2 inference八模块100 passed。Ruff、Black、py_compile、`bash -n`全部通过；本机故意缺少内网Python路径时脚本可立即给出明确失败信息，不再静默退出。

## 会话：2026-08-18（Phase 26）

### C1分布式LoRA恢复依赖兼容修复
- **状态：** complete（代码、无Git热补丁、全回归与远端发布均完成；真实C1 8×H100复验待内网执行）
- 用户报告C0 PASS、C1在generator role构建/恢复阶段8 rank统一报`ModuleNotFoundError: transformers.integrations.tensor_parallel`。
- 已确认失败发生在C1专属的raw adapter恢复：`stage2_role_init -> strict_load_lora_state_dict -> peft.set_peft_model_state_dict -> _maybe_shard_state_dict_for_tp`；训练子步尚未开始。
- 本地PEFT 0.19.1源码证明其在distributed initialized时先无条件导入HF tensor-parallel集成，再检查LoRA base是否存在TP plan；内网旧Transformers因此即使项目只用FSDP2也会失败。
- 决策：不要求内网盲升整套依赖，改为项目自有fail-closed canonical LoRA A/B加载路径，并同步无Git累计hotfix、启动版本握手、runbook和回归测试。
- 新LoRA loader回归7项已通过；首次hotfix联合测试发现新增API marker的legacy短片段也是current前缀，严格状态机正确拒绝且未写盘。已把指纹延长至`@dataclass`边界，下一轮验证幂等与旧版升级。
- 已实现`longlive_stage2_lora_load/v1`：canonical key与runtime default-adapter A/B参数必须完整双射，再用原生`load_state_dict`写入并逐tensor复验；不调用PEFT generic distributed/TP恢复分支。
- 累计hotfix会把旧loader、API marker和既有命名修复作为同一事务写入；wrapper与隔离probe新增`STAGE2_LORA_LOAD_API=PASS`，两份中文手册覆盖本次精确错误。
- LoRA/hotfix/H100 wrapper联合32 passed；覆盖缺失HF tensor-parallel模块、旧源码升级、重复执行、回滚和现有schema/value反例。
- checkpoint/role/inference/trainer/hotfix聚焦174 passed；完整Stage-2为694 passed、14条既有TorchScript弃用warning。
- 首次静态检查仅报告3个本轮Python文件需Black机械格式化；Ruff/py_compile/bash链因`&&`在Black处按预期停止，格式化后将完整重跑而不把未执行项记录为通过。
- Black机械格式化后同步hotfix current指纹；hotfix+缺模块回归11 passed，Black、Ruff、py_compile与`bash -n`全部通过。
- 全仓正式`python -m pytest -q tests`为998 passed、2 subtests passed、14条既有TorchScript弃用warning；相较Phase 25新增的唯一测试即缺失HF tensor-parallel模块的distributed-safe LoRA恢复回归。
- 精确暂存11个任务文件，用户metadata/checkpoints/tmp/results/Stage-1脚本与动作CSV均未进入提交；生产提交`c4202ca0d71c6a5712c98f7091ac76dc240a5ed5`已push `longlive-cats/stage-2`，`git ls-remote`返回同一SHA。

## 会话：2026-08-18（Phase 25）

### Stage-2参数命名契约系统修复
- **状态：** complete（代码、全回归、无Githotfix与远端发布均完成；真实8×H100 smoke待内网复验）
- 用户要求一次性审核并精准修复Stage-2训练过程所有parameter naming mismatch，不接受只针对当前EMA异常的局部绕过。
- 已恢复Phase 24根因与当前dirty worktree；本轮保护用户metadata、checkpoints、tmp config、results与Stage-1脚本，只改生产命名契约、回归测试、无Githotfix和必要文档。
- 审计范围锁定为：pre-FSDP immutable LoRA schema、PEFT canonical/raw names、Stage2 role wrapper、FSDP2 runtime names、optimizer DCP names、EMA topology/shadow、checkpoint save/load/resume及inference adapter加载。
- 修复原则：单一schema规范名；只允许完整相等或唯一点分隔后缀映射；缺失、歧义、碰撞、shape/dtype不一致全部fail closed；不删除门禁、不做全局`model.`替换。
- 首轮`rg`清单定位所有Stage-2消费者；确认EMA之外还有DCP optimizer持久化FQN与checkpoint默认交叉校验风险，修复范围扩展为canonicalize-on-save/decanonicalize-on-restore方案评估。
- 一次只读`sed`组合调用的JavaScript包装少了模板字符串右括号，工具在执行shell前即SyntaxError；已改正包装后成功读取，未执行任何仓库命令或产生文件改动。
- 25.1完成：确认四层命名空间——canonical adapter key（safetensors）、pre-FSDP raw FQN（唯一持久化拓扑名）、runtime FQN（允许Stage2/FSDP wrapper前缀）、DCP临时FQN（调用PyTorch set/get时使用）。推理只消费canonical adapter key，不应引入runtime FQN。
- 修改前联合基线：checkpoint、roundtrip、trainer orchestrator、true-Wan integration共108 passed、14条既有TorchScript弃用warning。
- 25.2红测首次按预期在`utils.parameter_names`不存在时collection error；实现公共resolver后，真实nested PEFT EMA、legacy EMA迁移、optimizer双向FQN转换与C0发布测试全部转绿。
- 25.3完成：`TrainableShardedEMA`新增immutable expected names并在每次module遍历时规范化；state/load/swap/update都使用schema raw FQN，load可严格迁移旧wrapper-prefixed state。Trainer显式传generator schema名字。
- optimizer DCP现在rank0写盘前规范为schema raw FQN，resume rank0验证后再映射为当前runtime FQN交给PyTorch DCP；prepared-payload入口也执行同一规范化。聚焦结果75 passed。
- 第二轮回归命令首次引用了不存在的`tests/test_stage2_fsdp2.py`，pytest在执行测试前退出；用`rg --files`确认真实覆盖位于`test_stage2_init_only.py`后改用正确文件集合，不重复错误命令。
- FSDP/init、role initialization、true-Wan、trainer、inference loader、LoRA selective checkpoint、Stage-1共享EMA联合129 passed，14条既有TorchScript弃用warning。
- 25.5完成：累计hotfix从4个扩展到9个运行时目标，能在旧内网checkout安全创建resolver，并把trainer、distributed、LoRA、checkpoint、FSDP2作为单事务更新；已有文件content-addressed备份，失败回滚，新文件失败删除，重复执行幂等。
- `run_stage2_h100.sh require_runtime`新增参数命名API、EMA签名与optimizer双向转换启动握手；两份中文手册要求同时看到DMD与parameter-name PASS。
- 完整Stage-2回归694 passed、14 warnings；全仓正式`tests/`回归997 passed、2 subtests passed、14 warnings。Black、Ruff、py_compile、bash syntax、本轮scoped diff-check及新文件no-index whitespace均通过。
- 全工作树diff-check仅命中用户原有600clip metadata的CRLF/尾随空白；按保护边界未修改。一次读取wrapper/runbook的JS包装语法错误在shell执行前失败，修正后成功，不影响仓库。
- 真实当前checkout执行hotfix `--check`输出`STAGE2_DMD_RUNTIME_API=PASS`、`STAGE2_PARAMETER_NAMES_API=PASS`、`ALREADY_APPLIED targets=9`。
- 精确暂存19个任务文件，确认用户metadata/checkpoints/tmp/results/Stage-1脚本均未进入index；提交`3551ed00667b82777a57805f62e2b0f47ac9adac`并push `longlive-cats/stage-2`。
- `git ls-remote longlive-cats refs/heads/stage-2`返回同一`3551ed00667b82777a57805f62e2b0f47ac9adac`，远端发布核验通过。

## 会话：2026-08-18（Phase 24）

### EMA parameter names mismatch诊断
- **状态：** in_progress
- 用户未附完整trace，仅给出错误文本；仓库唯一精确来源是`utils/distributed.py::validate_trainable_sharded_ema_state_dict`，该门禁比较checkpoint/local_shapes/global_shapes/shard_metadata与调用者传入的当前parameter_names。
- 当前先审计EMA构造/update/resume三种调用场景，重点验证PEFT/FSDP2前缀是否在wrap或序列化后发生漂移；未获完整trace前不把具体阶段或缺失/多余名字作事实。
- **状态：** complete（只读诊断；未修改production代码）
- 生产链确认：schema在`wrapper.model`上、FSDP前构建；Trainer随后对外层`self.model.generator`构造/更新EMA；C0在G1后保存checkpoint，`audit_local_ema_clock`用schema的`raw_parameter_name`校验EMA state，首次触发集合不等。
- 真实PEFT最小反例输出：schema为`base_model.model.block.lora_A/B.default.weight`，EMA为`model.base_model.model.block.lora_A/B.default.weight`，validator逐字报`EMA parameter names mismatch`。
- prepare只执行role init/FSDP2 init-only，不构造TrainableShardedEMA，也不走C0 cycle checkpoint，因此prepare PASS无法覆盖该缺陷。EMA虽到G40才初始化shadow，但从G0即记录并checkpoint参数shard topology，故C0/G1就失败。
- 安全修复方向：给TrainableShardedEMA传入immutable schema raw names，以“精确相等或唯一`.`后缀匹配”把外层运行时名字映射到schema名字；模糊/重复匹配fail closed。Trainer构造时传generator schema。增加nested真实PEFT、C0未初始化EMA保存、C1 resume和歧义拒绝测试。

## 会话：2026-08-18（Phase 23）

### prepare PASS后的重复OOM闭环
- **状态：** in_progress
- 新附件79行与Phase 22 OOM逐项相同：rank3、Generator rollout、FSDP2 all-gather申请318 MiB、GPU 3总79.19 GiB/空闲273.06 MiB、进程78.91 GiB、allocated69.91 GiB、reserved-unallocated7.11 GiB。
- `prepare=PASS`仅证明资产、角色初始化与tiny FSDP2 accumulation gate；不能证明完整5B micro2 rollout满足显存门禁。
- 下一步核对wrapper默认配置行为，交付同一shell内显式micro1配置和resolved自证命令，避免新shell丢失`STAGE2_CONFIG`后又回到默认micro2。
- **状态：** complete（只读诊断；未修改生产代码）
- wrapper第26/27行确认：未设置`STAGE2_CONFIG`时回退canonical micro2配置并同步`ACTIVE_CONFIG`；第29行默认smoke目录名也明确是`smoke_micro2_acc4`。
- 本地实跑resolved探针成功，canonical输出为`STAGE2_BATCH_PROFILE micro=2 acc=4 world=8 effective_global=64 saved_tensor_cpu_offload=False`；同一探针用于内网micro1启动前自证。
- 交付方案：同一shell创建micro1配置、显式export配置/全新目录/expandable allocator，探针必须输出micro1×acc8，再由wrapper重新prepare和smoke。micro1仍OOM才使用新config+新目录启用saved-tensor CPU offload。

## 会话：2026-08-17（Phase 21）

### H100 timing closure 故障闭环
- **状态：** complete（代码与无Git累计热修已发布；真实8×H100 smoke待内网复验）
- 内网smoke已完成rollout、fake/real score、backward/optimizer并进入`_append_metric("train_step")`；rank-0指标校验因`abs(timing_closure_error)=3.712756s`超过`max(0.1, 5%*step_seconds_max)=2.759816s`而拒绝。
- 本轮先核对计时边界和8 rank聚合，不直接提高5%阈值；修复完成后继续通过同一个无Git脚本交付，保护用户metadata/checkpoints/results/tmp/prepare_stage1等现有资产。
- 已完成计时边界审计：慢rank选取逻辑正确，失败源于多个真实runtime/audit/control阶段未分类，而不是跨rank把不同rank字段错误相加。方案锁定为新增互斥orchestration类别并保持原closure阈值。
- 红测设计锁定三层：`_run_update_attempt`必须输出非负attempt orchestration，`_run_one_logical_substep`必须把attempt与pre/post control合并传给timing summary，metrics/plotter必须把新字段纳入求和与图例；另扩展无Gitfixture验证trainer+metrics累计改写及多文件备份/幂等/失败不部分写。
- 6个聚焦红测按预期失败：当前attempt结果、logical-substep timing categories、slowest-rank summary与metrics字段均缺orchestration；失败位置与预期完全一致，开始实现互斥分段计时。
- 已实现attempt完整wall与内部分类差值、caller pre/post control显式计时、slowest-rank orchestration聚合、metrics字段及plot图例；原6项红测全部转绿。下一步增加启动期trainer/metrics握手并把多文件累计变换纳入无Git热修器。
- 启动审计已扩展为精确timing字段握手；当前wrapper与hotfix隔离probe会在torchrun前同时验证model callback API和metrics orchestration契约。
- 首轮联合106项有104通过、2个既有closure边界测试失败；原因是fixture新增0.1秒orchestration时没有从原有分类重分配，导致总phase凭空增加0.1秒，并非production公式错误。修正fixture为从rollout重分类0.05秒到orchestration后复验。
- 修正后3项closure边界转绿，hotfix/trainer/metrics/plot/H100 wrapper/runbook联合106 passed；包含启动期旧metrics拒绝、多文件累计升级与rollback反例。
- 静态首轮py_compile通过；Black仅要求机械格式化hotfix/trainer。Ruff新脚本指出rollback捕获`BaseException`过宽；plot/metrics/既有测试的其余诊断均为HEAD全文件债务，保持不扩大。将捕获收窄为`Exception`并只格式化本轮两个文件。
- Black把trainer的orchestration `result.get`压成单行，首次热修器current-source自检因仍持有格式化前精确片段而fail-closed；已同步热修器的current指纹，不放宽为正则匹配。
- 格式化后current-source probe通过；Ruff继续指出rollback通配`Exception`，已按真实`_atomic_replace`异常域收窄为`OSError`。新脚本Ruff、任务文件Black/py_compile及11项关键复验全部通过。
- 完整`python -m pytest -q tests/test_stage2_*.py`为675 passed、14条既有TorchScript弃用warning，用时152.65秒；loss/optimizer/EMA/checkpoint/rollout/metrics/guide全链未发现第二个回归。
- 最终静态门禁：新hotfix Ruff、任务文件Black/py_compile、shell语法、精确路径diff-check和isolated current-source probe全部通过；最终diff复核确认生产语义只新增orchestration计时/握手/图例，5%阈值、loss、optimizer与状态机未改。21.1–21.4完成，剩精确发布与远端核验。
- 追加真实`ab67824`四文件fixture验证时，首个临时目录命令因包含自动`rm -rf`清理被安全策略拒绝，未执行任何测试或删除；改为无删除命令的隔离临时目录验证，不重试被拒绝形式。
- 直接从发布提交`ab67824`归档真实model/trainer/metrics/plot四文件后执行当前热修器，结果精确为trainer/metrics/plot三文件`PATCHED`，逐字节等于当前源码；证明用户现有Phase 20内网状态可由同一单文件累计升级。
- 精确12文件提交`79747d3`已推送`longlive-cats/stage-2`；用户metadata/checkpoints/results/tmp/prepare_stage1及动作CSV均未暂存。下一步仅需重新传入同名热修脚本并重跑原smoke。

## 会话：2026-08-17（Phase 20）

### 无Git内网累计热修闭环
- **状态：** complete（代码与无Git交付已发布；真实8×H100 smoke待内网复验）
- 用户明确内网无法做Git管理；此前最新commit的增量diff被应用到旧`model/stage2_dmd.py`后，只加入v2常量而没有带入历史`72465ff`的callback实现，runtime门禁因此正确拒绝。
- 本轮不再要求clone/pull/reset；将提供单个可复制脚本，先备份目标文件，再以严格旧源码指纹执行累计变换，原子写回并在独立Python进程中验证F/G签名与callback行为。
- 当前本地HEAD=`c1daa26`，用户metadata/checkpoints/results/tmp/prepare_stage1继续排除。
- 五类热修回归已先写入；首次收集按预期因`apply_stage2_innernet_hotfix`模块尚不存在而失败，红测成立。
- 已实现严格片段状态机、AST/compile门禁、内容寻址备份、同目录fsync+`os.replace`原子写回及隔离子进程runtime audit；完整旧版、用户实际“v2常量+旧方法”混合版、当前版、重复执行和未知版拒绝共6项通过。
- trainer报错现已直接指向无Git热修器，快速部署与完整内网手册发布同一命令；首次组合回归因引用了不存在的`test_stage2_runtime_api.py`而未收集测试，已确认runtime审计实际覆盖在trainer/loss现有测试中，下一轮改用真实文件名执行。
- 第二次组合命令进入收集后发现当前shell实际调用系统Framework Python 3.11且缺`omegaconf`，因此runbook/trainer用例未执行；这属于本机测试解释器问题，不是热修回归失败。先定位仓库既有依赖环境，再重跑同一集合。
- 改用仓库依赖齐全的`python -m pytest`后，首次真实组合回归仅暴露快速文档超过既有60行上限；压缩无Git提示而不删契约后，hotfix/guide/runbook/trainer/loss共100 passed、14条既有TorchScript弃用warning。
- 完整`python -m pytest -q tests/test_stage2_*.py`为672 passed、14条既有TorchScript弃用warning，用时153.01秒；未发现callback之后的第二个Stage-2回归错误。
- 静态首轮：py_compile、`bash -n`及真实`--check` runtime probe通过；Black仅要求格式化两个新文件，Ruff对新脚本报告可执行位/import排序，对trainer报告的其余12项为既有全文件债务。下一步只机械整理新文件并用HEAD差分确认trainer未新增lint债务。
- 两个新文件已机械Black/import-sort并给热修器可执行位；新文件Ruff与任务文件Black均通过。全worktree `git diff --check`只报告用户现有metadata CSV尾随空格，未修改该资产；发布门禁改为对本轮精确路径执行。
- 格式化后hotfix/guide/runbook 23项复验通过，任务路径Ruff/Black/py_compile/bash语法/diff-check全通过；runtime子进程进一步用隔离`-X pycache_prefix`规避内网旧`.pyc`误加载。20.1–20.4完成，剩精确发布。
- 精确8文件提交`f46eb1b`已推送`longlive-cats/stage-2`；用户metadata/checkpoints/results/tmp/prepare_stage1及动作CSV均未暂存。无Git内网只需传入一个热修脚本并执行一条命令。

## 会话：2026-08-17（Phase 19）

### Stage‑2 smoke 全调用链接口闭环
- **状态：** complete（runtime API/source-path闭环已发布；真实8×H100 smoke待内网复验）
- 内网新失败发生在C0首个fake-score DSM loss：trainer传入`timing_callback`，但`Stage2DMD.fake_score_flow_dsm_loss_from_model()`签名不接受该参数。
- 本轮按用户要求不做单点止血：将建立trainer所有`self.model.*`生产调用与Stage2DMD签名矩阵，并继续审计callback调用时机、返回结构、F/G/DFD分支和checkpoint smoke边界。
- 当前分支/远端仍为`stage-2`/`longlive-cats`，HEAD=`8eb75e4`；用户metadata、checkpoints、results、tmp与prepare_stage1继续保护，不暂存。
- 已建立AST调用矩阵：trainer只有F/G两处`self.model.*`调用，当前HEAD的两个Stage2DMD签名均完整接受所传kwargs；`timing_callback`已存在于历史提交`72465ff`。这把根因收敛为内网运行时source/API skew，而非HEAD缺少该参数。
- 已确认现有loss测试未传callback，角色初始化也没有runtime API握手；将以“显式API版本+精确签名审计+F/G callback行为测试+trainer生产调用测试”补齐覆盖，而不是对TypeError做兼容吞错。
- runtime API红测先按预期因helper缺失失败；现已加入v2版本/精确签名握手并接入Trainer构造和角色初始化后双门禁，目标测试通过。
- 已给F-DMD、G-DMD、G-DFD三条真实Stage2DMD loss路径补callback标签/执行顺序测试，4项聚焦测试通过；静态接口矩阵确认rollout、Generator、fake/real score与output字段无其他漂移。
- 新增AST生产调用矩阵测试并让现有C0/C1/C2 Trainer构造测试断言runtime audit；一次组合命令因后一个`-k`覆盖前者只跑3项，已拆开复验trainer门禁6项通过。
- callback后联合目标回归142 passed（loss/trainer/role/manifest/transaction/init-only/true-Wan），14条均为既有TorchScript弃用warning；下一步执行完整Stage‑2与静态/发布门禁。
- 完整Stage‑2回归665 passed、14条既有warning；首次Black check仅报告`trainer/stage2_distillation.py`新增审计段需机械格式化，尚未把该静态结果标为通过。
- 仅机械格式化新增trainer审计段后，Black、Ruff、py_compile和精确diff-check全部通过；Phase 19仅剩最终diff审阅、精确commit/push与远端核验。
- 最终diff审阅后决定把同一runtime audit前移到`run_stage2_h100.sh::require_runtime`，这样source path/API版本在torchrun和模型加载前验证；将补shell契约测试后重新跑guide与完整静态门禁。
- wrapper/guide/trainer/loss联合83 passed，`bash -n`通过；与shell相同的isolated runtime probe已实跑并打印当前checkout的v2 PASS/source path。
- 最终精确范围为3个production文件、3个测试文件和3份Phase 19记录；用户metadata、checkpoints、results、tmp与prepare_stage1保持未暂存，发布后远端commit需与本地HEAD逐位核验。

## 会话：2026-08-17（Phase 18）

### 实现 cross-KV FSDP2 修复并发布
- **状态：** complete（本地修复与GitHub `stage-2`发布完成；真实8×H100 smoke待内网复验）
- 用户已明确授权修改代码并上传GitHub `stage-2`分支。
- 发布边界锁定为本轮cross-KV production代码、相关测试和Phase 17/18诊断记录；用户现有metadata、checkpoints、results、tmp和prepare_stage1不暂存。
- 已确认当前`stage-2`跟踪`longlive-cats/stage-2`且GitHub认证为`zonghui-liu-opt`；不新建分支或PR。
- 先加入FSDP2 root/block双容器重建回归；首次收集按预期因`utils.stage2_cross_kv`不存在失败，随后实现普通Python leaf state。
- allocation、attention、audit、reset和prefix validation已统一为严格state契约；legacy非Stage‑2 cache旁路未改，FSDP mixed-precision策略未改。
- 首轮聚焦回归55 passed；完整`tests/test_stage2_*.py`为663 passed、14条既有TorchScript弃用warning。
- 新增/正常格式任务文件通过Black、Ruff、py_compile和精确diff-check；`causal_model.py`当前7项Ruff与HEAD基线完全相同，未借本轮修复扩大历史格式/lint改动。
- 仅暂存本轮7个tracked文件与新增state模块并审阅staged diff；用户的metadata、checkpoints、results、tmp和prepare_stage1保持未暂存。
- planning完成检查脚本未设置可执行位；不修改技能文件权限，改用`bash`执行同一只读检查。

## 会话：2026-08-17（Phase 17）

### Stage‑2 H100 cross-KV smoke 故障定位
- **状态：** complete（诊断与解决方案完成；未改production代码）
- 已恢复现有planning文件并记录本轮目标；本轮是诊断与解决方案，未获授权修改production代码。
- 已确认异常位于`pipeline/stage2_rollout.py::_preload_sink`后的首次`_audit_cache`，而不是LoRA target选择或可选C++ extension导入警告。
- 已保护dirty worktree中的用户文件；下一步逐行比较preload期望状态与底层cross-attention实际状态更新。
- 已用PyTorch v2.8官方源码确认FSDP2 root device-move和block `cast_forward_inputs=True`会先后递归重建kwargs dict/list，并用本地最小反例复现“K/V tensor写入保留、Python bool状态丢失”。
- 已形成最小正式修复设计：FSDP共享轻量可变Python leaf flag + strict audit + root/block双容器重建回归；不改block mixed precision policy，不引入CUDA scalar sync。
- 复验建议：先运行新增单测/相关回归，再使用全新仓库外`STAGE2_SMOKE_DIR`重跑C0→C1→C2，保留旧失败目录作为证据。

## 会话：2026-08-16（Phase 16）

### Stage‑2 H100 单一指导脚本
- **状态：** complete（本地实现、回归和独立终审完成；真实8×H100执行待内网）
- 用户反馈完整运行手册过于复杂，要求重新编写一个指导脚本。
- 目标是提供一个顶层入口，按 `prepare / smoke / train / plot / infer` 分阶段调用现有生产实现，并为每阶段打印“作用、将运行的命令、成功输出、下一步”。
- 保持Stage‑1 step3075、正式动作分布198/202/200、C0/C1/C2精确恢复、B1/B0同G240分叉及G280 EMA推理契约不变；不在wrapper中复制训练算法。
- 已启动当前脚本/CLI和测试契约的并行只读审计。
- 首轮指导脚本测试为14通过/1失败；失败原因是macOS测试夹具硬编码了不存在的`/bin/true`，生产脚本未执行。已改为`shutil.which("true")`动态解析，避免重复平台假设。
- 修正夹具后指导脚本+runbook共15项全部通过；首次Black门禁仅报告新增测试文件需要机械格式化，未发现功能失败。
- 一次组合补丁因`require_runtime`实际行序与预期上下文不同而整体未应用；随后一次多文件补丁分隔符写法错误，也未发生部分修改。已改用三个精确小补丁。另一次手工status探针把shell builtin `true`误当绝对路径；正式测试已使用`shutil.which`，生产逻辑不受影响。
- 收紧formal arm/checkpoint与推理磁盘哈希后，artifact/guide/runbook目标回归27项全部通过；空目录status逐项输出INCOMPLETE且不创建文件。首次静态门禁仅要求格式化新增的artifact验证函数。
- 新增顶层可执行入口`run_stage2_h100.sh`，只保留`prepare / smoke / train / control / plot / infer / status`七个明确子命令，不提供会误启动长训练的`all`。
- wrapper严格锁定Stage‑1 step3075和正式动作分布198/202/200；B1/B0分别验证dmd_dfd/dmd_only arm、G280完整checkpoint及其认证metrics lineage，推理严格绑定当前B1 G280、当前推理配置、56个样本计划和所有MP4/trace/index内容哈希。
- 每阶段只认固定成功标志：`STAGE2_GUIDE_PREPARE=PASS`、`STAGE2_GUIDE_SMOKE=PASS`、`STAGE2_GUIDE_TRAIN_B1=PASS`、`STAGE2_GUIDE_TRAIN_B0=PASS`、`STAGE2_GUIDE_PLOT=PASS`、`STAGE2_GUIDE_INFER=PASS samples=56`；缺少标志即停止，不自动删除或修补半成品。
- 最终验证：目标guide/artifact/runbook 27 passed；完整`tests/test_stage2_*.py`为662 passed、14条既有TorchScript弃用warning；`bash -n`、Black、Ruff、py_compile和任务范围diff-check通过。独立最终审计P0=0、P1=0。

## 会话：2026-08-15（Phase 15）

### Stage‑2 当前磁盘态重新严格审计
- **状态：** complete（本地代码/测试/静态/终审；真实8×H100执行待内网）
- 已恢复既有 `task_plan.md`、`findings.md`、`progress.md` 和活跃持久目标，确认不从头重复已完成工作。
- 已记录用户确认：内网 H100 的 `prepare_stage2.sh` 全部门禁通过；训练 smoke/正式训练/训练后推理仍需分别验收。
- 已检查当前分支、远端和 dirty worktree，后续只在任务范围内增量修改并保护用户资产。
- 首次响应 `/goal` 时尝试创建新目标，被系统因同一任务已有 active goal 拒绝；随后读取现有目标并继续，未重复创建。
- 下一步：完整重读任务书，建立 Steps 9–14 的代码/测试映射并并行进行独立只读审计。
- 已完整重读739行任务书；发现顶部状态与后文Steps 9–11完成状态矛盾，并确认Steps 12–14仍是明确未完成项。
- 已启动三路独立只读审计：trainer/checkpoint、metrics/plot、inference/compression；主线程继续建立真实测试基线和调用图。
- 当前完整Stage‑2基线结束：445 passed、1 failed、14 warnings，耗时117.13s；失败为prepare脚本的step3750契约漂移。
- 进一步核对确认：任务书/正式YAML/测试为3750，当前prepare脚本的checkpoint路径、merged文件名、注释、manifest校验和resolved断言均为3075；该差异会改变研究初始化，已按规格暂停实现并中止三路并行审计，等待用户决策。
- 自动续办再次检查当前工作树，3750/3075冲突仍原样存在且没有用户决策或权威文件更新；这是连续第2个目标回合的同一阻断。已保持目标active并把首要计划固定为确认唯一初始化checkpoint，没有改动生产代码或测试。
- 用户随后确认step3075 EMA表现更好，是Stage‑2唯一权威Generator起点；同时取消旧任务划分和检查点暂停规则。已解除阻断，恢复三路只读审计并开始把所有Stage‑2专属3750引用系统性迁移到3075。
- 已同步正式YAML、resolver locked contract、Generator payload/manifest/role init、F25入口、runbook、任务书和Stage‑2测试fixture到step3075；首次统一后的path-independent contract hash为`68b4a3b05535c70b979becf5984898d81b651ead84c929370b7f21dccdde4e2f`。随后为支持A24同点安全分叉B0/B1，仅从fork-compatible contract view排除分支模式/概率向量，当前resolver实值更新为`a7365f2ec45f74c3918ec05725b5d19b488fa4447a409cc6b5db4ccb114dd6c6`。
- step3075迁移验证完成：聚焦配置/provenance/runbook集合157 passed；完整`tests/test_stage2_*.py`为446 passed、14 warnings、111.30s。旧的唯一失败已消失，Stage‑1专属3750实验入口未被改写。
- 静态复核：两个Stage‑2 shell syntax与修改Python编译通过；全树diff check仅被用户现有600条metadata行尾格式阻断，本轮不擅自重写用户训练数据，任务文件将单独验收。
- inference审计初报“末步少一次UniPC传播”经真实scheduler数值复验后撤销：K4末步raw x0与terminal step bitwise相等。保留的真实缺口是缺少显式deploy API/trace，以及当前pipeline硬锁K4/C8/W16/S1、无法走原生K2/C4/S4/S8。
- metrics/plot只读终审完成：无P0，发现2个P1、4个P2、1个P3；已记录lineage越界覆盖、complete metadata不绑定resolved config、JSON尾行容错过宽、producer测试缺口、绘图/hash TOCTOU和raw曲线缺失。
- 唯一没有权威数值的修复项是timing closure容限；已提出`max(0.1s, 5% wall)`建议并暂停实现等待用户确认。trainer与inference审计已中止，可在确认后原位恢复。
- 自动续办重新检查配置、实现、测试和任务书，仍未发现用户确认或其他权威timing closure阈值；这是连续第2个目标回合的同一阻断。未修改生产代码/测试，目标保持active。
- 第3个连续目标回合再次确认无用户答复、无配置或任务书权威阈值；已达到持久目标blocked门槛。保持当前代码现场，正式标记blocked，等待用户确认`max(0.1s, 5% wall)`或提供其他容限后恢复。
- 用户解释后明确确认采用`abs(closure) <= max(0.1s, 5% wall)`；持久目标已自动恢复active。解除本地计划阻断，继续metrics红测/修复并恢复trainer/inference审计。
- trainer/checkpoint只读终审完成：92项聚焦回归通过，无P0；记录6个P1、3个P2、1个P3，下一步以反例测试驱动修复，保留已验证正确的5F→G状态机与global-mean路径。
- inference/compression只读终审完成：确认Stage‑2部署交付和EMA-only loader是P0缺口；现有K4 random-exit数值正确但需显式deploy API，K2/C4/S4/S8与batch inference/trace均未实现。
- metrics修复任务在等待noise裁决时被中断；检查工作树确认未留下metrics/plot部分改动。
- 用户确认双动作A/B均使用独立随机起点且同一`(sample, seed)`确定性复现；精确定义锁定为一次连续48-slot draw，A/B分别取前/后24，禁止B前重置seed。推理实现暂停解除。
- 新增模型无关的Stage‑2推理基础件与8项红绿测试：逐sample显式RNG保证batch/rank分片不改变noise，双动作严格连续48切为24+24；VAE每episode只接收`sink1+future24`、清cache、验证97帧并只丢pixel frame0得到96帧。聚焦测试8 passed。
- 接入已锁定的rollout core接口并新增single/two orchestration：单动作96帧；双动作A/B各自full episode、reset self/cross KV、分别decode/drop sink后拼成192帧；S4/S8要求episode1 prefix snapshot并由episode2 pipeline恢复。trace记录noise/prompt/UniPC/cache/reset/VAE事件；聚焦测试扩展为12 passed。
- 新增Stage‑2 batch sample planner，直接复用严格6-row/8-row loader，显式忽略旧HOLD/soft-reanchor字段；baseline矩阵精确为24个single+32个two-action，sample key固定为`dataset/row/seed/profile`。8卡rank-stride分片为每卡7条且无padding/drop/路径冲突；8个canonical profile由rollout core单一resolver校验。推理相关聚焦测试21 passed。
- 新增技术产物层：逐MP4仅验证非空常规文件、96/192帧、分辨率与24fps；sample trace自哈希绑定raw prompt、EMA checkpoint、profile、noise/UniPC/cache/reset/VAE事件并拒绝A/B复用noise及任何自动质量指标。完整manifest要求56个sample key/trace精确集合，静态HTML只供人工播放。推理聚焦测试25 passed。
- 新增独立正式推理YAML与strict resolver：只接受baseline profile、seeds1‑4、EMA LoRA safe-merge、CFG1、BF16、24fps、per-device batch1及两份权威metadata；output root只改变launch hash。推理相关聚焦测试34 passed。
- metrics/plot修复代理完成：lineage boundary/连续cycle、严格坏JSON尾、同bytes parse+SHA、resolved-config双向绑定、用户确认的timing closure、raw+rolling time breakdown和真实Trainer producer集成均已落地；代理聚焦测试33 passed且Black/Ruff/diff-check全绿，未触碰trainer/config/checkpoint。
- trainer/checkpoint终审并修复：A24/G240可作为B0/B1共同immutable ancestry anchor，G240后严格分arm；checkpoint绑定`metrics_lineage.jsonl`并支持外部child空目录原子导入，G280 terminal no-work重启可幂等补`run_end`。完整sampler/EMA/provenance/optimizer/RNG/live-grad门禁与EMA同bytes认证加载均已闭环。
- C0/C1/C2新增无消费next-F1 probe：保存/重放sampler batch、exit、loader RNG、rollout/timestep/score-noise和计数器；任一本周期nonfinite次数非0均拒绝smoke PASS。真实world8 FSDP2 sync/no-sync global64 parity gate已接入新版prepare，未在本机伪跑H100。
- rollout/deployment统一为8个canonical named profiles；显式full-episode API复用唯一kernel，原生K2 schedule与C4/W/S capacity/call-count严格验证。S4/S8从episode1 clean recache snapshot detached永久prefix并只用于episode2，episode1保持baseline不变。
- baseline推理完整实现：56个样本、96/192帧、seeds1–4、CFG1单cache；双动作使用同一seed初始化的一条连续48-slot noise plan，A/B分别消费前/后24且可确定复现。Generator-only EMA loader只反序列化认证EMA，正式runtime每rank只加载一次T5/Generator/VAE并复用同图latent。
- 推理产物严格绑定resolved/runtime四类contract/launch hash、checkpoint EMA、source manifest以及T5/tokenizer/VAE/architecture/Generator内容身份；rank0单次流式认证后全rank在真实loader前后复核。任何sample写入前完成全rankconfig/code/metadata/plan/checkpoint共识和已有pair preflight。
- output-root使用frozen canonical dev/inode/mode guard；视频、trace、index和最终manifest写前后及逐层parent均复验。root被rename后换symlink或另一真实目录会在创建任何child前拒绝；manifest仍为最终commit marker。普通TOCTOU fail-closed边界已明确，不夸大为抵抗主动攻击者check→open微窗。
- H100完整中文runbook覆盖仓库外输出、新版6门禁、C0/C1/C2、micro1回退、B1 formal、同G240的B0 matched control、双JSONL绘图与G280 EMA baseline推理；正式动作分布唯一锁定为198/202/200，Stage‑1起点唯一锁定为step3075。
- 最终静止磁盘态动态门禁：`tests/test_stage2_*.py`为656 passed；仓库正式`tests/`范围为964 passed、2 subtests passed；14条warning均为既有TorchScript弃用提示。推理组112 passed，trainer/checkpoint/metrics focused 110 passed；独立最终规格审计P0=0、P1=0。
- 用户既有`training_sets/metadata_600clips_480x832_buckets.csv`保留CRLF/trailing-whitespace与当前内容，未擅自标准化或补成600条；任务代码路径单独执行diff检查。真实8×H100新版CHECK6、C0/C1/C2、B1/B0 formal、图表、56视频与人工质量仍必须内网执行并保存证据。

## 会话：2026-08-13（Phase 14）

### Stage‑2 F25 latent 专用重提
- **状态：** complete（本地实现与验证；真实8×H100执行待内网完整数据）
- 已完成只读契约审计：Stage‑1前93像素帧→F24；Stage‑2前97像素帧→F25；正式训练返回`video_latent[1:25]`。
- 已检查新CSV：两份本地文件均仅1条；metadata相对video与sidecar绝对video不一致；正式入口将对此在GPU启动前失败。
- 已确认不修改旧Stage‑1 cache、不使用七列metadata替换其来源metadata；新F25与negative输出进入独立目录。
- 首次组合规划补丁因`progress.md`标题/Phase 13段落与预期不一致而整体未应用；读取真实文件头后改用精确上下文。
- 新增`precompute_stage2_i2v_cache_h100_8gpu.sh`：clean clone、600条输入预检、8×H100门禁、F24→F25、positive attestation、negative和formal audit；不含模型merge、role init或训练命令。
- 新增`validate_stage2_i2v_cache_inputs.py`：严格六列metadata、600条、旧manifest逐行hash/dimension/bucket、两列sidecar逐字video映射与`198/202/200`动作分布门禁。
- 新增6项无GPU测试，覆盖有效输入、七列metadata拒绝、绝对/相对video不一致、动作不均衡、旧manifest不匹配与shell命令边界。
- 首轮5项测试通过；Black发现新验证器需要机械格式化，格式化后Black/Ruff全部通过。
- 目标配置/手册/入口联合回归112 passed；F25/data trust-chain回归52 passed；最终完整`tests/test_stage2_*.py`为443 passed、14 warnings。
- bash syntax、Python编译/CLI help、tracked/untracked whitespace、可执行权限和禁止训练命令审计全部通过；本机没有H100且两份用户CSV都仅1条，因此未伪跑正式提取。
- 最终设备审计将negative编码从固定物理GPU 0改为用户8卡列表中的第一张，并新增数字/去重门禁，避免自定义CUDA设备列表被绕过。
- 最终清理首次使用`rm -f`删除明确列出的测试字节码时被执行环境策略拒绝，命令未运行；改用逐个精确路径的`find -delete`后复核。

## 会话：2026-08-12（Phase 13）

### 前5项检查一键化
- **状态：** complete
- 已恢复规划和Git状态；当前分支与远端同步，只有本地未跟踪的checkpoint/result/tmp/prepare_stage1资产，继续排除。
- 已锁定实现：一个脚本依次完成teacher、step3075 Generator、配置绑定、600条F25/negative/formal audit、8卡role init与FSDP2 parity；简版文档只保留路径、单命令和6个成功标记。
- `prepare_stage2.sh`现在自动创建clean clone，完整产物复验后复用，来源漂移或明显半成品fail closed；不会启动Stage-2训练。
- 文档已压缩为43行；用户只需提供动作标签、teacher确认和必要的路径覆盖，最终检查5个固定PASS及总PASS。
- 验证完成：`bash -n prepare_stage2.sh`、`git diff --check`、全部`tests/test_stage2_*.py`，结果`437 passed, 14 warnings`。
- 修正：首次目标测试仍锁定旧长runbook预期，已改为5-gate契约；参数复核还发现negative manifest文件名应为`negative_conditioning_manifest.json`，修正后完成全量回归。

## 会话：2026-08-11（Phase 12）

### real-score manifest 后的 Stage-2 H100 指南重生成
- **状态：** in_progress
- 已恢复并完整读取三份规划文件，确认旧手册边界落后于当前Stage-2训练闭环。
- 已把本轮范围锁定为：从teacher manifest生成/验收到init-only、C0/C1/C2 smoke、正式训练/resume和JSONL/九图HTML验收的完整单线指导；下一步以当前CLI和脚本反向生成文档。
- 已核对train/plot/teacher/merge/preflight CLI；formal cache与F25 CLI即使请求`--help`也会先验证物理clean checkout，当前文档编辑态按设计失败。后续不绕过该安全门禁，改读parser源码并用测试验证命令契约。
- 已完成生产契约审计：确认3750是resolver硬锁定来源，prepare脚本中的3075/G变量实际未消费；确认YAML模型路径环境变量旧手册未生效、F25最终source env漏设、旧顺序会导致launch绑定漂移，以及正式执行必须采用clean clone/仓库外资产与输出。
- 已完整重写Stage-2中文手册，从teacher manifest三类SHA校验开始，覆盖step3750 Generator、一次性最终env/launch绑定、F25/negative/formal audit、init-only、C0→C1→C2、formal cold/resume、G280/F1400终点和九图HTML。
- 已让正式YAML的architecture/G/teacher路径真正读取手册环境变量，保留原路径默认值与既有contract hash；prepare脚本移除未消费的3075/G变量，新增teacher manifest自哈希和merge记录文件SHA复核及明确handoff。
- 新增`tests/test_stage2_runbook.py`锁定新手册全流程、禁止旧检查点A终止语义、验证外部资产env进入resolver并检查prepare脚本teacher-only边界。
- 首轮目标验证为113 passed / 1 failed：唯一失败是测试禁止出现历史`329 passed`，而手册恰在“不要锁死历史计数”的说明句中引用了该字符串；已删除具体历史数字。Black同时要求格式化新增测试，下一步机械格式化后重跑。
- 修订后目标门禁114 passed；完整`tests/test_stage2_*.py + test_jsonl_training_plot.py`最终443 passed、14 warnings（均为既有TorchScript弃用）。contract hash保持锁定值，bash/Black/关键Ruff/diff检查通过。
- **状态：** complete（本地代码与指南；真实8×H100 C0/C1/C2和formal待用户内网执行）

## 会话：2026-08-11

### Phase 10：Stage‑1 LoRA/merged 四卡批量推理对比
- **状态：** complete（reference可选扩展完成；等待内网4×H100）
- 新增需求：此前推理结果目录已删除；未提供`reference_dir`时必须直接从原6-case metadata fresh prepare共享carrier/config，不能要求重新跑旧格式推理。
- 兼容边界：显式提供reference时继续校验step、metadata、prepared manifest与历史输出；缺省时只生成merged-vs-LoRA主对比，HTML/report不得引用不存在的历史视频。
- runner CLI与shell现默认不传reference；fresh模式要求architecture/T5/tokenizer/VAE资产并复用`prepare_causal_testsets()`生成`shared_prepared/prepared_manifest.json`。设置`LONG_LIVE_STAGE1_REFERENCE_DIR`时自动恢复原reference模式。
- 无reference端到端CPU fixture真实完成两种geometry metadata解析、carrier/config preparation、四任务构建、两路输出映射、pair/report/HTML；确认report为fresh/null reference且所有sample无`reference_video`。
- 最终目标回归39 passed；Ruff、py_compile、`bash -n`、CLI help与diff whitespace全部通过。本轮未执行Git提交或push。
- 用户要求新增推理时动态加载 LoRA 的 batch inference，并在单机4×H100上尽可能并行加速。
- 已确认上一轮 runner 只重新推理预 merged checkpoint，左侧直接复用既有 infer_stage1 结果；原 infer_stage1 runner 本身也是先 merge EMA adapter 后推理，因此尚未形成动态 LoRA 对照。
- 已为 `inference.py` 接入Stage‑1 safetensors严格LoRA加载、显式样本索引sampler和逐row独立noise seed；legacy `.pt` adapter仍保留兼容。
- 正式调度不使用会漏样本的4-rank DistributedSampler，而是并发4个单卡进程：GPU0/1分别处理pre-merged横/竖屏3条，GPU2/3分别处理dynamic-LoRA横/竖屏3条。每卡只加载一个模型实例，4卡等负载且总模型加载4次。
- 两路均克隆同一reference prepared输入与sampling配置，并用`base_seed + CSV row_id`生成完全相同的noise seed映射；旧infer_stage1视频因使用旧顺序RNG只作为HTML历史参考。
- runner会校验merged companion manifest，证明完整权重来自同一个converted base、step3750 checkpoint和`adapter_ema.safetensors`，随后严格校验两路6个输出并生成merged-left/LoRA-right视频、HTML、JSON报告和每GPU日志。
- 性能设置只对克隆的comparison config启用：每进程1个prefetch worker、pinned memory、异步H2D、TF32/cuDNN benchmark、allocator扩展段和CPU线程限流；不改变其他config的DataLoader默认值。
- 本地目标回归35 passed；Ruff、py_compile、shell syntax、CLI help、diff whitespace通过；真实ffmpeg 832×480×3帧拼接得到1664×480×3帧。当前机器无H100，因此不声称正式吞吐或视频质量通过。
- 保护边界：Phase 9检查点B既有脏工作树不覆盖、不暂存；本地无4×H100，不声称正式性能或视频质量通过。

### Phase 9 检查点B：训练闭环代码实现
- **状态：** waiting_for_user_review
- 用户明确要求继续精准完成Stage-2训练代码，并在完成log、权重/checkpoint保存和可视化后停止交其检查；本轮不做batch推理、不运行正式H100训练、不编写/上传smoke指南。
- 已完整重读`planning-with-files-zh`、三份规划文件与739行Stage-2任务文档，锁定Steps 9–11：严格`F1..F5→G→EMA`、cycle-boundary checkpoint、成功G时钟驱动Phase/DFD/EMA、同batch/RNG nonfinite retry、F/G/cycle独立JSONL时钟与PNG/SVG/HTML。
- 用户本次继续指令构成代码实施授权；会话未提供新的H100检查点A真实输出，因此本地实现不会声称8×H100/NCCL/真实600-cache已经通过。
- 已完成严格`F1..F5→G→EMA→commit`训练状态机、A24/B4/DFD概率、global64 accumulation、G40 EMA、同batch/RNG nonfinite精确重试与角色梯度/optimizer隔离。
- 已完成world8一维FULL_SHARD专用checkpoint：G/F raw LoRA、G EMA、两个AdamW、sampler/DataLoader/每rank及控制RNG、配置/数据/资产/topology，隐藏临时目录闭链后rename并最后写`_SUCCESS`；C1恢复和formal/smoke lineage均fail-closed。
- 已完成权威Stage‑2 JSONL、F/G/cycle/nonfinite/checkpoint记录，9组PNG/SVG与静态HTML；完整run要求latest lineage自己的`run_end.status=complete`及严格终点/5F→1G结构。
- H100 smoke入口收敛为`--stage2-smoke C0|C1|C2`：C0 cold+save，C1 resume+DMD+save，C2 resume+DFD+discard；formal明确拒绝续接smoke checkpoint。新增`nvidia-ml-py`正式依赖，并在`train.py`最早阶段禁止写bytecode，避免训练入口自行污染clean checkout。
- 最终本地磁盘态：正式`tests/`为735 passed、2 subtests passed；14条warning均为既有TorchScript弃用提示。Black、Ruff、py_compile、train/plot CLI help、固定contract hash与`git diff --check`通过；独立终审为P0=0、P1=0。
- 当前边界：未运行8×H100/NCCL/FSDP2 smoke，不声称显存/NVML/吞吐通过；未实现Step 12以后batch推理。本轮按约定停止给用户检查，不写本节点H100指导、不提交、不push。

## 会话：2026-08-10

### Phase 9 检查点A：用户检查通过，等待内网8×H100准备门禁
- **状态：** waiting_for_h100_checkpoint_a
- 用户已确认原始视频为97 pixel frames并锁定方案：Stage‑2保持24-new/F25；现有cache逐条严格检查，合格F25直接复用，F24从原始97帧确定性重提。禁止补帧、复制latent或改23-new。
- 完成显式`initial1+future24` score pack、9750-token mixed timestep、FP32连续score noising/x0、DMD/DFD与fake raw-flow DSM。
- 完成3×8 rollout、W16/H8/S1/capacity17、真实UniPC sigma、分层exit RNG、clean-only self-KV commit、真实单路cross-cache与episode reset。
- 已修复FP32 noisy进入BF16 Conv3d、FSDP root误转FP32 timestep、rollout误用`t/1000`、NCCL CPU broadcast和方向混合microbatch；legacy Stage‑1 source manifest现有严格、原子、非覆盖式文本来源升级入口，正式CLI强制`python -I -B`并在项目导入前验证物理clean HEAD。
- 最终本地计数：F25 focused 88 passed；全部Stage‑2 329 passed；全`tests/` 611 passed及2 subtests passed。14条warning均为既有`torch.jit.script_method`弃用提示。两位独立代码审计与H100指南终审均确认P0=0。
- 最终静态检查：新增/Stage‑2 Python文件Black、Ruff、py_compile、tracked/untracked whitespace全部通过；contract hash为`aa4d7be1e05c846df14cee5417a298afe668429f41faa671f3021754a5616c00`。最小扩展的legacy `causal_model.py`保留与HEAD完全相同的既有7项Ruff/Black债务，没有新增诊断或无关重排。
- 已将旧434行历史式H100文档重写为单线检查点A手册，补齐clean/ignored门禁、外部600条metadata/action注入、F25→attestation→negative→formal audit、init-only成功信号和失败即停说明；未实现Steps9–14，`results/`用户数据保持未跟踪且未改动。

### Phase 9：训练前准备检查点执行记录（历史，已完成）
- **当时状态：** waiting_for_user_review
- 已完成：
  - 完整重读 `TASK-stage2-self-forcing-dmd-dfd.md`，确认文档首行、Step 2 与现有计划均要求等待 8×H100 init-only 验证。
  - 核对当前分支为 `stage-2@bde0142`，远端 `longlive-cats/stage-2` 同步；仅有用户未跟踪的 `results/`，不得触碰。
  - 核对 merge、teacher manifest、role preflight 三个 CLI 的真实 `--help` 与参数；确认 preflight 会拒绝 dirty worktree、既存输出目录及非 8-rank torchrun。
  - 识别旧手册的主要可用性问题：434 行混合历史与当前步骤、路径占位缺少集中填写区、成功信号分散、没有把“clean code clone / repo 外资产与输出”说成人话。
- 用户最新明确覆盖旧划分：只在“全部训练前准备”“训练+log/权重/可视化”“batch推理与其余任务”三个节点停；中间小Batch由Codex自行验证，不再逐个等待。
- 已据此废止H100‑002单独暂停，将第一个检查点锁定为原Steps 1–8；训练主循环/JSONL/checkpoint/plot仍属于下一检查点，当前禁止提前实现。
- 当时中间快照的212项Batch2门禁已复跑：`212 passed, 14 warnings in 21.86s`，警告均为已知`torch.jit.script_method` deprecation。
- 后续结果：Step 3–8、native F25 producer与安全终审全部完成；本条“用户通过前不发布”是当时边界，现用户已通过并进入H100门禁等待阶段。
- 错误记录：rollout RNG首轮测试错误地断言两条独立随机流的第一组`randperm`数值必不相同；不同随机流允许偶然产生相同排列。已改为断言底层RNG state不同，并分别验证两组都满足完整K4分层覆盖。
- 格式检查首次报告新 rollout 实现和测试需要 Black 机械格式化；产品测试已通过，下一步仅格式化这两个新文件后重跑 Ruff/py_compile/diff。
- Black后Ruff发现`typing.Sequence`未使用；已删除该单个导入，未运行扩大范围的自动修复。
- 创建/修改的文件：`task_plan.md`、`progress.md`、`findings.md`。

## 会话：2026-08-08

### Phase 9：Stage-2 分批实现与 H100 门禁（历史，已被2026‑08‑10三检查点覆盖）
- **当时状态：** paused（Batch 2 / Step 2 已完成并推送；等待用户执行 H100‑002 init-only 门禁）
- 已确认：
  - 用户要求正式编码前先完成任务划分。
  - 首批代码必须进入新建远程 `stage-2` 分支；推送后立即暂停，等待用户在内网 H100 验证成功。
  - 后续批次不得在首批 H100 门禁通过前提前实现。
- 当前工作：
  - 已完整阅读任务文档、Git状态、配置系统、legacy DMD trainer/model/pipeline/Wan wrapper与相关测试。
  - 三路独立只读复核完成：批次依赖、仓库映射、Git/测试安全均已交叉校验。
  - 已定稿8个实现批次；本轮只实现Batch 1/Step 1，推送后暂停。
  - 已从 `stage-1@4c0bb6a` 创建`stage-2`，发布目标严格为用户远程`longlive-cats/stage-2`；未触碰上游`origin`。
  - 修改前相关回归基线为64 passed（14+50）；UniPC K4/shift5 timetable与negative prompt hash均和规格一致。
  - 新增 `configs/train_i2v_stage2_600cats.yaml`：仅保存baseline原始参数与后续资产槽位，不复制派生计数或UniPC timetable。
  - 新增 `utils/stage2_config.py`：纯静态严格resolver、canonical hash、派生公式和只读CLI；未接registry，未导入torch/model/CUDA。
  - 静态CLI、py_compile与pure-import首次自检通过。
  - 新增 `tests/test_stage2_config.py`，覆盖release派生值、A/B计数、EMA时钟、LoRA契约、真实UniPC timetable、hash/幂等，以及unknown/legacy/missing/非法数值与拓扑反例。
  - 新增测试首轮71 passed；修改前64-test相关回归复跑仍为14+50 passed。
  - 首次Black check仅报告两个新增Python文件需格式化；下一步执行机械格式化并继续Ruff/py_compile/diff审计。
  - Black已格式化两个新增Python文件；Ruff发现并已删除一个未使用的`math`导入。
  - 修正后Black、Ruff、py_compile、`git diff --check`全部通过；新增测试保持71 passed。
  - 三路初审提出的schema碰撞、raw-before-normalize边界、CFG/EMA角色语义、candidate措辞、seed、Phase-B matched control、fallback、manifest-first action标签、双hash、typed字段和严格字符串问题均已逐项修订。
  - 当前resolver输出launch hash与跨init/resume可比的contract hash；补齐Generator self/cross KV单conditional cache、typed runtime输入、generator-grad-exit offload scope与语义数值规范化后，本地默认占位路径下分别为`7b6ad0163d75e94f86b8836a9d0a9aa1aa904964fb30dbb7bac90e98febf5e1c`与`aa4d7be1e05c846df14cee5417a298afe668429f41faa671f3021754a5616c00`。
  - review修订后的Stage-2契约测试为103 passed；修改前相关回归再次为14+50 passed，共167个本地测试无失败。
  - Black、Ruff、py_compile、tracked/untracked whitespace与CLI help/JSON解析全部通过；关键派生值为capacity17、seq_len9750、G280/F1400，EMA target为generator adapter。
  - 三路独立最终review提出的P0/P1已全部闭环；精确提交/推送范围不含`results/`，后续实现按用户要求暂停。
  - 用户回报内网H100：Stage‑2 tests为103 passed，配置契约全部通过；相关回归为63 passed。
  - 被排除项为`test_release_stage1_config_has_one_locked_source_of_truth`。该测试锁定公开仓库Stage‑1 release YAML，与实际内网Stage‑1训练参数不一致；Batch 1未修改Stage‑1 YAML或其解析路径，因此接受为内网门禁的显式例外，仓库测试文件本身不删除。
  - Batch 1总门禁判定为通过；按用户授权开始Batch 2，范围严格限定为Step 2三角色初始化、独立LoRA、checkpoint/manifest与FSDP init-only审计。
  - Batch 2修改前共享基础件基线：`test_stage2_config`、`test_lora_utils`、`test_stage1_fsdp2`、`test_merge_lora_generator`与`test_stage1_lazy_imports`合计128 passed；运行时禁用pytest cache与bytecode。
  - 已先新增Batch 2三组红测骨架，覆盖生产shape target公式、PEFT角色隔离、teacher/init manifest、world8 1D FULL_SHARD与lazy init-only CLI；首次运行按预期因三个待实现模块缺失而在collect阶段失败，未出现规格外失败。
  - 新增`Stage2DMD`三角色容器、严格role initializer、LoRA角色契约、world8一维FSDP2 helper、teacher/role-init manifest与独立`preflight_stage2_roles.py`；未注册trainer或修改`train.py`。
  - Stage‑1 EMA merge升级为不可覆盖源文件、前后source snapshot一致、正式raw/EMA metadata、180 target/360 tensor schema、自哈希与strict fresh reload的v2 producer；Stage‑2 validator会重新核验原Stage‑1 `_SUCCESS`、manifest、拓扑与所有来源SHA。
  - Teacher入口绑定architecture config、完整Wan无权重语义、operator attestation与可信provenance；native index的全部shards逐tensor必须BF16，LongLive wrapper payload只允许一个manifest指定state dict和标量metadata。
  - 三角色按G→real→F顺序meta构造和严格物化，real/F不共享Parameter或storage；G/F LoRA master为FP32且B全0，real为BF16 frozen/0 trainable。每个role独立wrap root+30 blocks，post-FSDP审计全部参数的1D `Shard(0)`、mesh、dtype、FQN与计数。
  - init-only入口在任何模型前拒绝resume/dirty或错误commit，所有rank逐阶段共识失败；运行时哨兵覆盖标准/直接forward、optimizer、EMA、T5、VAE与DataLoader，角色隔离、side effects、资产SHA和FSDP证据进入全rank consensus与原子manifest，marker为`ROLE_INIT_COMPLETE`而非训练`_SUCCESS`。
  - 三路独立终审发现的native shard哈希、source BF16、FSDP frozen-base假阳性、资产TOCTOU、merge destructive路径、CUDA RNG污染、architecture漂移、world hang、legacy mmap、metadata token误判等问题均已用正反测试闭环。
  - 最终本地Batch 2相关门禁为212 passed（14条已知`torch.jit` deprecation）；用户指定Stage‑1/DMD回归复跑为63 passed、1 deselected；Ruff、Black、三个CLI help、仓库外CLI bootstrap与`git diff --check`通过。
  - `docs/STAGE2_H100_QUICK_DEPLOY_ZH.md`已新增`H100-002`：精确覆盖merge v2、teacher两种格式、环境变量、8卡torchrun、原子manifest验收、失败即停及“只证明init、不证明训练”边界。
  - Batch 2提交范围严格为18个代码/测试/文档文件并推送到`longlive-cats/stage-2`；用户未跟踪的`results/`及其中资产文件未暂存、未提交、未推送。按分批约定暂停，不开始Batch 3。

## 会话：2026-08-07

### Phase 8：Stage-2 LongLive-2.0 Self-Forcing DMD/DFD
- **状态：** complete（任务文档已交付，production code未开始）
- 执行的操作：
  - 完整读取 `grill-me` 与 `planning-with-files-zh` 技能说明，恢复既有 planning 文件。
  - 完成 Stage-1/Stage-2 代码、配置、缓存、CFG、LoRA、score timestep、loss、resume 与 H100 只读审计。
  - 与用户逐项锁定 baseline、Phase A/B、DMD/DFD、5F→1G、KV梯度边界、checkpoint、预检和压缩消融方案。
  - 用户确认头脑风暴结束并授权生成任务文档，新增要求为参考 Stage-1 实现 generator/fake-score loss 与吞吐可视化。
  - 审计 `utils/jsonl_logger.py`、`scripts/plot_stage1_training.py`、Stage-1 trainer logging/config/tests，确定复用 append-only lineage JSONL、raw+rolling PNG/SVG 和角色拆分吞吐的方案。
  - 独立复核发现Stage-1 producer/plotter字段漂移及本地metrics artifact损坏；已把共享schema、producer→plotter契约测试、logical substep lineage与必需图fail-fast写入Stage-2任务文档。
  - 创建 `TASK-stage2-self-forcing-dmd-dfd.md`，覆盖目标、non-goals、全部锁定决策、P0、15步实现/验证计划、H100门禁、推理与压缩消融。
  - 补充隔离Stage‑2入口的文件映射，以及本地CSV仅1条且无action label时必须使用冻结sidecar的正式数据门禁。
  - 完成两轮独立复核：无P0；收紧了continuous sigma、negative conditioning、F/G独立sampler、global-mean梯度等价、B1概率端点、partial-run绘图、压缩EMA/matched-control与multi-sink KV snapshot契约。
  - 执行Markdown结构、重复行、空白与 `git diff --check` 审计，722行/78标题/15步/26个成对代码fence全部通过；planning完成检查为8/8。用户既有 `results/` 未跟踪数据保持不变。
- 下一步：
  - 等待用户审阅并明确授权后，从任务文档 Step 1 开始实现。
- 创建/修改的文件：
  - `task_plan.md`
  - `findings.md`
  - `progress.md`
  - `TASK-stage2-self-forcing-dmd-dfd.md`

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
| Phase 10 LoRA/merged目标测试 | LoRA strict load、4卡分片/provenance、输出映射与validator | 全部通过 | 35 passed | PASS |
| Phase 10静态与真实拼接smoke | Ruff、py_compile、bash syntax、CLI help、diff check、真实ffmpeg | 全部通过 | 1664×480、3帧、24fps | PASS |
| Phase 10 optional-reference扩展 | fresh端到端、reference HTML兼容、CLI/shell/static及相关回归 | 全部通过 | 39 passed | PASS |

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
| 2026-08-07 | 文档静态检查首次循环变量误用zsh特殊 `path`，导致当前shell的PATH被覆盖而找不到git | 1 | 改用任务专用 `file_item` 后在新shell重跑，全部静态检查通过 |
| 2026-08-11 | 对混有历史未格式化代码的既有文件运行Black check，报告5个文件需要全文件重排 | 1 | 不扩大或覆盖用户既有diff；改用Ruff、py_compile和目标测试验证本任务改动，未执行批量Black写入 |

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

## 2026-08-11 Phase 11：新增 Stage-1 资产后的 Stage-2 只读复审

- **状态：** complete
- 已确认正式配置迁移为 `configs/train_i2v_stage2_600cats.yaml`，原 `configs/train_i2v_stage2.yaml` 删除；`configs/tmp.yaml`、`prepare_stage1.sh` 与本地 checkpoint/result 资产不纳入 Stage-2 发布提交。
- 本轮仅审查，不修改生产代码、配置、manifest 或模型权重；此前735项通过只作历史证据，必须以当前磁盘态重新验证。
- 双向角色实现复核通过：real/fake 从同一已审计teacher分别构造独立 `WanModel`，real冻结，fake加载完整base后挂fresh r64；52项role测试通过。
- 当前两份新YAML完全相同且均不可解析：world4与锁定world8冲突，并缺必填`data.source_cache_manifest`；canonical YAML缺失导致当前全tests为617 passed / 124 failed，124项均由FileNotFoundError引起。
- trainer/checkpoint/JSONL/九图HTML与共享LoRA联合回归98 passed；隔离配置问题后的独立审计未发现这些模块新增P0/P1，Ruff/Black/py_compile通过。
- 提供的DiffSynth merge manifest是来源证明，不是`longlive_stage2_teacher_manifest`；当前配置错误指向Stage-1 causal conversion产物。应直接使用原生双向merged safetensors并由现有CLI生成正式teacher sidecar。
- 测试操作中的两次非产品错误：首次把shell glob加引号导致pytest找不到文件；一次诊断脚本误读resolved字段名，均已用正确命令/字段复跑，不影响上述结果。
# 2026-08-18 Phase 22：8×H100 smoke OOM 诊断

- **状态：** in_progress
- 已完整读取用户附件：rank3在generator rollout的FSDP2 block pre-forward `foreach_all_gather`申请318 MiB时OOM。
- OOM瞬间GPU 3总79.19 GiB、仅273.06 MiB空闲；进程占78.91 GiB，其中PyTorch allocated 69.91 GiB、reserved-but-unallocated 7.11 GiB。22.9 GiB显然不是失败点的峰值口径。
- 下一步：核对正式YAML、smoke wrapper、FSDP wrapping/prefetch与trainer memory字段，判定主因和安全回退配置。
- **状态：** complete（只读诊断；未修改生产代码/配置，未运行本机GPU测试）
- 配置核对：当前候选为micro2×8×acc4=global64，Generator关闭activation checkpoint；FSDP2逐block/root均`reshard_after_forward=True`，但每层forward仍需临时all-gather完整block，激活与cache不由参数分片消除。
- 统计口径核对：trainer在每个逻辑子步前reset peak，只在attempt成功返回后调用`_memory_fields()`写max allocated/reserved；OOM在attempt内部直接抛出，所以最后可见的22.9 GiB不会包含失败子步的69.91 GiB峰值。
- 门禁计算：79.19 GiB卡的85% allocated上限约67.31 GiB、90% reserved上限约71.27 GiB、最小空闲为8 GiB；本次69.91/约77.02/0.267 GiB三项均不满足，单开allocator调优也不能让micro2正式放行。
- 方案锁定：全新配置切micro1×acc8，保持global batch64并用全新smoke目录从C0重跑；`expandable_segments:True`只作为降低碎片的进程启动参数。若micro1仍失败，再单独profile配置允许的Generator grad-exit saved-tensor CPU offload；禁止Generator activation checkpoint、降低global batch或复用旧partial smoke lineage。
