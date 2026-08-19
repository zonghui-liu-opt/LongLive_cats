# 发现与决策

## 2026-08-19 Phase 30：旧失败遗留video-only恢复

- 用户栈停在`atomic_output_path(video_path)`成功退出后的`_assert_regular_parents()`；此时正式MP4已通过ffprobe并提交，但trace构建/写入尚未发生。
- 现有resume preflight要求video与trace同时存在或同时不存在，因此直接同步Phase 29后仍会对这一个单边MP4报`complete video+trace pair`，不能真正“一次性续跑”。
- generation trace包含noise/latent/cache等运行期审计，不能只凭MP4事后伪造；正确恢复是严格复验单边MP4后将其保留到输出根内隐藏隔离区，再用相同sample/seed/checkpoint完整重生成video+trace。
- 仅允许“video存在、trace不存在、最终manifest不存在”进入自动恢复；坏MP4、symlink、trace-only或已有最终manifest时继续fail-closed。隔离区位于root下但不在`videos/`/`traces/`，不会污染56件套验收。
- 恢复实现对原MP4移动前后各做一次ffprobe/size/SHA验收，并在两侧目录fsync后验证正式路径已空、隔离文件技术identity未变；原文件不删除，隔离名包含内容SHA前缀和随机nonce。
- 隔离后生成再次失败时，正式video/trace/manifest仍保持未完成，隔离原片保留；下一次启动会按“二者均不存在”正常生成。端到端测试已覆盖第二次失败后第三次成功。
- 精确生产提交的干净worktree完整通过Stage-2 inference八模块113项；`51b6da06c4f9c4dbc38cbd2177e1e82ba175877d`已发布到`longlive-cats/stage-2`并由ls-remote核验。

## 2026-08-19 Phase 29：输出根目录identity漂移

- 新错误发生在临时MP4已通过ffprobe并原子提交为正式视频之后；因此已有视频不是坏文件，失败点是提交后的输出路径安全复核。
- 当前root guard只保存并逐次精确比较`st_dev/st_ino/st_mode`，不包含mtime/ctime。正常创建子文件不会改变这些字段，内网现象说明共享/叠加存储返回了新的目录identity，或输出根确实被外部替换。
- 启动collective已经证明各rank最初看到同一identity；简单删除inode/device检查会使既有“rename旧root并在原路径放新目录”攻击反例失守，不能作为修复。
- 方案锁定为在输出根内原子建立随机、持久、各rank一致的内容anchor：每次写入前后同时验证canonical非symlink根、anchor类型/内容与单次检查稳定性；跨调用允许目录stat identity漂移，但真实新目录因缺少anchor仍被拒绝。
- anchor必须支持多rank首次并发竞争和失败后原目录断点续跑；最终artifact严格集合只枚举`videos/`与`traces/`，不会把根级隐藏anchor误计为额外视频或trace。
- 实现使用同目录完整临时文件加原子硬链接竞争发布`.stage2-output-root-anchor.json`；8个并发prepare只会有一个获胜nonce，所有rank的v2 collective identity统一为canonical path与anchor SHA256，不再传播易漂移的目录stat。
- 每次guard复核仍对root与anchor做lstat/resolve/open/fstat前后稳定性、类型、canonical JSON和SHA检查；合法漂移测试与“视频保存后才漂移”的端到端测试通过，目录替换、anchor缺失/非法/symlink/另一有效anchor反例继续拒绝。
- 精确提交生成的干净worktree中，Stage-2 inference八模块108 passed；因此标准远端shell契约、断点续跑和本轮root guard同时闭环，现场脚本的未提交7卡/路径定制不属于发布内容。
- 用户附带的`infer_stage2_tmp.sh`包含内网绝对路径和7卡定制，是现场脚本；本轮必须保留，不纳入生产提交。

## 2026-08-18 Phase 28：ffprobe exit 127

- 新错误已越过checkpoint/runtime asset认证、Generator EMA加载和至少一个样本生成；失败发生在临时MP4落盘后的技术验收，不是NCCL、权重、显存或生成模型错误。
- `probe_video()`当前只调用一次`shutil.which("ffprobe")`，内网PATH首候选为`/home/ma-user/miniconda3/bin/ffprobe`；该命令实际返回127，但代码没有启动健康检查、候选回退或stderr增强诊断。
- 修复不能跳过ffprobe或在失败时直接接受MP4。应在真实生成前执行`ffprobe -version`，支持显式环境override，并按PATH全部可执行候选及常见系统位置逐个实测；只有健康候选才能用于既有width/height/frame_count/fps门禁。
- 推理输出使用原子临时文件，当前失败不会把未验收的rank0 MP4冒充完整产物；其他rank已完成的完整video+trace对可由现有resumability在重跑时安全复用。
- 新解析器对普通发现依次实测PATH全部候选、当前Python环境及常见系统路径；坏候选返回127会记录stderr并继续。显式`LONG_LIVE_FFPROBE`采用fail-closed语义，避免操作者指定错误路径后被静默替换。
- 一键shell在任何5B模型加载前调用同一解析器并打印`STAGE2_FFPROBE=PASS (<path>)`；没有健康候选时直接给出全部尝试和override提示，避免再次生成完视频才发现系统工具失效。
- 真实共享回归证明修复没有改变视频验收：Stage-1 causal/continuation/merged comparison和Stage-2 inference联合208 passed；全仓1006 passed、2 subtests。完整技术门禁仍检查width/height/frame_count/fps并哈希正式MP4。
- 不应删除当前`inference_early_g000080`。原子上下文会删除失败的隐藏临时MP4，已完成的正式video+trace对在重跑时会逐项复核并跳过，剩余样本继续生成。

## 2026-08-18 Phase 27：跨节点早期checkpoint推理

- G70错误发生在rank0的`build_stage2_runtime_assets()`，其余rank只通过collective转发；尚未加载Generator EMA，也不是NCCL或8卡分片错误。
- checkpoint持久provenance中的Generator资产包含训练节点当时的`device/inode/mtime_ns/ctime_ns`；推理节点重新验证manifest后生成当前节点identity，现有代码对整个对象做canonical JSON相等，导致相同SHA内容也可能因跨节点stat identity不同而失败。
- 安全修复不能删除哈希/manifest门禁或修改checkpoint provenance。持久比较应验证manifest、checkpoint SHA/size/schema/lineage等内容稳定字段；rank0重新哈希后得到的当前identity仍必须进入runtime asset，并由`strict_load_stage2_role_base()`在加载前后复核，继续防止认证后文件被替换。
- `infer_stage2_tmp.sh`当前两个裸`test`会静默退出，且Python runner在资产认证/模型bootstrap期间无启动进度。新shell需给每个失败条件明确报错，并在长bootstrap阶段输出心跳。
- 同一完整对象比较在两处重复：rank0 `stage2_inference_assets` 的live-manifest认证，以及每rank `stage2_inference_loader` 对rank0 trusted asset与checkpoint recorded asset的复核；修复必须覆盖两处，否则rank0通过后仍会在各rank loader阶段再次失败。
- 正确的数据流是：checkpoint中的完整recorded asset及其SHA保持不变；rank0严格重验manifest/底座内容后生成live trusted asset；只在recorded-vs-live比较时忽略`checkpoint_files[*].identity`，后续loader始终携带live identity并在真实模型加载前后调用现有identity守卫。
- 已用统一`stage2_generator_asset_content_sha256()`修复rank0和每rank两处比较：只排除host-local identity，`checkpoint_sha256/size/path/manifest/schema/source_step`等内容契约仍保持fail-closed；跨节点identity测试转绿，SHA漂移反例继续拒绝。
- 新`infer_stage2_tmp.sh`默认G70并支持`60`或`60 70`顺序执行；固定快照目录，避免训练保留策略删除formal checkpoint；启动前校验输入、GPU数和跨节点API版本，bootstrap期间每30秒心跳，失败报告日志，成功强制验收56视频/56 trace/manifest/index。
- 56来自固定baseline矩阵：`(6个单动作 + 8个双动作) × 4个seed = 56`；8卡只做rank-stride数据并行，每卡7个样本，不会把总产物乘成448。
- 全部Stage-2 inference测试100 passed；Ruff、Black、py_compile和`bash -n`均通过。真实8×H100权重加载与生成需由内网执行。

## 2026-08-18 Phase 26：C1分布式LoRA恢复依赖错配

- C0 已通过而 C1 在 `initialize_stage2_roles()` 的 generator build/load/local audit 阶段失败；C1 尚未执行下一 F1、rollout、loss、backward、optimizer、EMA或新checkpoint。
- C0 fresh LoRA只配置adapter；C1额外调用`strict_load_lora_state_dict()`恢复C0的generator/fake-score raw adapter。当前实现进一步调用PEFT 0.19.1的`set_peft_model_state_dict()`。
- PEFT 0.19.1在任何已初始化`torch.distributed`进程中处理LoRA state时都会进入`_maybe_shard_state_dict_for_tp()`，并在检查模型是否真的使用HF tensor parallel之前无条件导入`transformers.integrations.tensor_parallel`。内网Transformers不含该模块，所以8个rank一致失败；本项目使用FSDP2而非HF tensor parallel，这个导入与实际恢复无关。
- 仅升级Transformers会改变Wan/diffusers/tokenizers整套运行依赖，且内网环境不一定可联网；仅降级PEFT又会改变已验证的LoRA注入语义。更安全的修复是让项目的严格LoRA loader把canonical A/B key一一映射到已存在的default-adapter runtime parameter key，执行完整schema/shape/dtype/finite/loaded-value校验后用PyTorch原生load，完全不调用PEFT的可选HF-TP恢复分支。
- 新loader必须只允许未FSDP/未DTensor的pre-shard模型，并继续拒绝missing/extra/duplicate/nonfinite/shape/dtype错误；这与C1 role初始化和inference merge的实际调用时点一致，不得通过捕获`ModuleNotFoundError`后静默跳过加载。
- 修复已发布到`longlive-cats/stage-2`提交`c4202ca0d71c6a5712c98f7091ac76dc240a5ed5`；内网只需同步最新版累计hotfix脚本，确认新增`STAGE2_LORA_LOAD_API=PASS`后可直接重跑C1并复用已严格验证的C0 checkpoint。

## 2026-08-17 Phase 21：timing closure 首始证据

- `step_seconds_max≈55.196s`，closure差值约3.713s，占6.73%，仅略过5%门槛；异常发生在训练更新已成功、JSONL append前，不是模型数值或optimizer失败。
- 必须先确认`step_seconds`的起止点是否包含跨rank barrier/all-gather、指标构造和CUDA event之外的CPU调度，而phase合计是否只覆盖rollout/score/backward/optimizer；若定义边界不同，固定5%相对阈值会把合法的FSDP/NCCL/CPU框架开销误判成漏计时。
- 生产代码证实wall计时从data materialize前持续到state commit/consensus后，但详细phase只计data、H2D、模型CUDA段、backward、clip/optimizer和EMA。未计入的明确工作包括exit/branch广播、micro间Python/diagnostic、全参数梯度finite审计、跨角色gradient审计、post-step参数finite审计、LoRA optimizer state审计、loss/diagnostic归约、state commit与consensus；5B×8卡上这些合法工作足以产生3.71秒差值。
- 正确修复不是把5%提高到7%或删除closure门禁，而是新增`orchestration_seconds_max`：用完整attempt wall减去其互斥的模型/backward/optimizer阶段得到attempt orchestration，再加caller显式测量的pre/post control段。closure仍独立比较完整step wall与所有互斥类别，继续保持0.1秒/5%严格门禁。
- metrics validator和plotter共享`STAGE2_TIMING_FIELDS`，因此新字段需同时进入schema契约；内网无Git热修器必须累计更新trainer与`utils/stage2_metrics.py`，不能只改producer，否则旧validator会因求和不包含新字段再次拒绝JSONL。
- 为防再次手工混版，现有启动期`_audit_stage2_dmd_runtime_api`应同时握手完整timing字段顺序；H100 wrapper本来就在torchrun前调用该审计，因此无需新增第二套入口即可在模型加载前拒绝“新trainer+旧metrics”。无Git脚本的隔离probe也复用同一审计。

## 2026-08-17 Phase 20：无Git内网部署事实

- 内网报错中的model source path是当前checkout，但方法实际参数仍为旧三参数；结合v2常量已存在，说明操作者应用的是“最新commit相对新基线的增量”，而不是完整文件。最新commit只触碰常量，Git历史中的callback实现不会随单个增量自动补入。
- 正确交付单元必须从“Git commit差异”切换为“累计热修器”：识别旧/新两种方法体，旧版补齐F/G callback签名与计时调用，新版只验证；未知或部分匹配源码拒绝写入。写前创建唯一备份，临时文件同目录fsync后`os.replace`，二次执行不再改文件。
- 现有完整手册已明确运行期不依赖Git，但缺少“手工同步造成source/API skew”后的恢复入口；应在快速部署和完整手册都固定为先运行单文件热修器、看到runtime API PASS，再重跑原smoke，避免操作者继续手改两个长方法。

## 2026-08-17 Phase 19：Stage‑2 smoke 新接口异常

- 新异常已越过上一轮cross-KV sink preload，说明FSDP2 cross-KV状态修复在真实H100路径上生效；当前中止点推进到`_compute_micro_loss`的fake-score DSM loss调用。
- 直接错误是调用者传入`timing_callback`而callee拒绝，但必须进一步核对callback契约是否也缺少内部阶段上报、其他model方法是否存在同类kwargs/返回值漂移，不能只在签名末尾机械加参数。
- 当前Git HEAD `8eb75e4`中的`model/stage2_dmd.py`其实已从提交`72465ff`起同时为F/G两个from-model方法接受并执行`timing_callback`；AST矩阵证明trainer两处`self.model.*`调用的全部kwargs在当前callee签名中均被接受。因此内网异常不是当前分支的静态接口缺口，而是运行时加载了与trainer不同步的旧`Stage2DMD`定义（source/API skew）。
- 仅删除callback或捕获`TypeError`会掩盖混合版本部署，并可能继续运行缺少其他训练修复的旧模型代码；正式方案应给Stage2DMD增加显式runtime API版本与精确签名握手，在角色初始化后、首个昂贵rollout前fail-fast，并用测试锁定trainer调用矩阵和callback标签/返回值。
- 现有loss测试验证了F/G/DFD的张量共享、梯度隔离和返回loss，但调用时都省略`timing_callback`；完整663项因而未覆盖生产trainer实际传参。角色初始化也直接返回`Stage2DMD`，没有对trainer所需runtime API版本/签名做握手，这是本次“单测全绿、H100首轮才失败”的具体覆盖缺口。
- 当前F路径callback应精确上报一次`fake_score`；G的DMD/DFD路径都应依次上报`fake_score`、`real_cond`、`real_uncond`并原样返回role forward结果。后续回归必须同时锁定标签、次数、执行顺序和loss/梯度不变。
- 静态矩阵继续覆盖到完整micro-loss链：trainer→rollout、rollout→Generator、Stage2DMD→fake/real score四处调用均无额外kwargs；trainer读取的F/G output字段与两个冻结dataclass完全一致。当前未发现callback之外的第二个HEAD接口漂移。
- 新门禁采用`longlive_stage2_dmd_runtime/v2`类版本+两个from-model方法的精确keyword-only参数顺序/default审计，并在Trainer构造期及角色初始化后各执行一次；旧callee会在任何模型forward/rollout前报告实际source path和缺失参数，不再消耗一次H100训练才能发现。
- callback之后的联合回归覆盖loss、Trainer 5F→1G、C0/C1/C2 options、role init/manifest、transaction、FSDP init-only与true-Wan rollout，共142项全部通过；这批测试同时覆盖optimizer/EMA/checkpoint smoke所依赖的本地状态机边界，未发现新的HEAD错误。
- 完整`tests/test_stage2_*.py`更新后为665 passed、14条既有TorchScript弃用warning；功能回归没有暴露第二个错误。首次Black静态门禁只要求机械格式化本轮trainer审计代码。
- 仅格式化新增trainer审计段后，Black、Ruff、py_compile与任务范围diff-check全部通过；生产修复范围收敛为`model/stage2_dmd.py`的API版本和trainer双握手，没有删除timing或放宽旧代码兼容。
- 为避免8个rank完成三角色5B加载后才发现source skew，顶层H100 wrapper的`require_runtime`也应使用隔离Python、显式插入当前`SCRIPT_ROOT`执行同一API审计，并要求实际`Stage2DMD` source恰为当前checkout的`model/stage2_dmd.py`；成功打印单一`STAGE2_DMD_RUNTIME_API=PASS`证据。
- wrapper前置门禁、guide、trainer与loss联合回归83项通过；本地以与shell相同的isolated import/path检查实跑并打印`STAGE2_DMD_RUNTIME_API=PASS version=longlive_stage2_dmd_runtime/v2`，证明当前checkout source解析正确。

## 2026-08-17 Phase 18：cross-KV FSDP2 修复实现

- 新增`Stage2CrossKVInitState`作为普通Python leaf，不使用dataclass/container，确保PyTorch FSDP2 root `_to_kwargs`和block `_apply_to_tensors`重建kwargs时保持同一对象identity。
- Stage‑2 allocation、Wan cross-attention写入、rollout audit/reset/prefix validation统一读写该state；缺失或类型错误会fail closed。legacy非Stage‑2 cache仍保留原`is_init`旁路行为。
- 回归测试直接执行PyTorch私有helper模拟两层真实容器重建，证明外层dict均被复制、leaf state仍共享且内层更新可被原rollout cache观察；没有关闭block `cast_forward_inputs=True`，也没有引入CUDA标量同步。
- 修改前红测因新模块不存在而按预期失败；实现后FSDP2反例、cross-cache行为、true-Wan rollout与init-only聚焦集合55项通过，完整Stage‑2回归663项通过。
- 新增/正常格式文件通过Black，任务文件Ruff/py_compile/diff-check通过；`causal_model.py`的全文件Black及7项Ruff债务均存在于HEAD，本轮对比未增加诊断，避免无关大规模格式化。

## 2026-08-17 Phase 17：Stage‑2 H100 cross-KV smoke 故障

- 用户内网`bash run_stage2_h100.sh smoke`已通过三角色LoRA构建，在C0的首个logical substep、首个rollout sink preload中止；异常是rank1的`Stage-2 layer 0 cross-KV init state mismatch`。
- `torch 2.8.0+cu128`与某可选C++ extension的`torch>=2.11`警告发生在模型初始化期，当前证据不支持它是此cross-KV契约异常的直接根因；不应先盲升Torch。
- 当前本地worktree有用户侧metadata/checkpoint/result/tmp等改动；本轮默认只读诊断，不触碰这些资产。
- 根因链路：PyTorch 2.8 FSDP2 root pre-forward在CUDA上先用`_to_kwargs`递归重建device kwargs，每个block又因`cast_forward_inputs=True`用`_apply_to_tensors`重建mixed-precision kwargs；两个helper都对dict/list创建新容器。BF16 K/V tensor已在目标device/dtype而仍共享对象，但Python `is_init=False`只被按值复制。cross attention的`crossattn_cache["is_init"] = True`只修改最内层FSDP副本，rollout原始state仍False，首层审计精确报错。因root device-move本身也会重建容器，仅将block `cast_forward_inputs=False`不是完整修复。
- 本地最小反例复现：FSDP式recursive move/cast后dict均不同一，K/V tensor仍共享；在forward dict内写K/V能反映到原dict，但赋值`is_init=True`后原dict仍False。轻量非dataclass可变Python flag对象在两次容器重建后仍保持同一identity，其`initialized`field可正确回写。
- 当前测试缺口：true-Wan rollout测试未经FSDP2 block wrapper；init-only测试反而锁定了block `cast_forward_inputs=True`，但不检查nested mutable kwargs的容器副本语义；H100 accumulation gate是toy Linear，因此prepare通过而首个真实Wan+FSDP forward失败。
- 推荐修复：新增一个非dataclass、非container的轻量`Stage2CrossKVInitState`叶子对象，内部只有`initialized: bool`；allocation将该对象放入每层cache；attention中严格验证类型并更新field；reset/audit/prefix validation通过field写/读。FSDP 2.8对未知Python对象按leaf保留identity，无CUDA `.item()`同步；保留`cast_forward_inputs=True`和现有BF16契约。0维device bool tensor可作正确性热修，但正式方案不建议承受每层标量读取的host/device同步。
- 禁止型“修复”：不要删除`_audit_cache`、不要将`expected_cross_initialized=False`、不要在forward返回后无条件手工将每层flag设True，不要只关掉block input cast，不要为该错盲升PyTorch。
- 环境警告是第二个独立问题：官方兼容表明PyTorch 2.8.0的C++ extension对应torchao 0.13.0，仓库`requirements.txt`也锁定0.13.0。内网应核对实际torchao版本并降/固定到0.13.0，而不是把已验证的Torch 2.8盲升到2.11。

## 2026-08-16 Phase 16：Stage‑2 H100 单一指导脚本

- 用户不需要再从完整runbook中人工拼接命令；唯一推荐入口是仓库根目录的`run_stage2_h100.sh`。运行顺序固定为`prepare`→`smoke`→`train`→`control`→`plot`→`infer`，`status`只读查看进度。
- 不提供`all`子命令，避免误触即启动两轮长训练。每个阶段独立重建所需环境、复用现有生产入口并在完成后做严格产物认证；wrapper不复制训练算法。
- `train`是正式B1（DMD+DFD）训练，`control`是从同一G240分叉的B0纯DMD对照；二者中断后都只需在同一目录重跑同一子命令，脚本会验证合法ancestry并精确恢复。
- `infer`只接受B1 G280 EMA，必须得到当前配置定义的56个完整video+trace pair、认证manifest和精确HTML index；技术PASS后仍要人工打开index审阅画质，不能把文件完整性等同于视觉质量。
- 所有工作/训练/推理输出必须位于Git checkout外，正式运行要求clean commit；任何固定PASS缺失都必须停下保留日志，禁止自动删除checkpoint、拼接partial产物或绕过hash/lineage门禁。
- 最终本地证据：guide/artifact/runbook目标集合27 passed，完整Stage‑2集合662 passed；shell syntax与Python静态门禁通过，两路独立审计P0=0/P1=0。真实8×H100 prepare/smoke/B1/B0/inference与人工画质仍需用户在内网执行，未伪造结论。

## 2026-08-15 Phase 15：当前磁盘态重新严格审计

- 用户明确确认已在内网 H100 上执行当时版本的 `prepare_stage2.sh` 且全部环境/准备门禁通过；本轮新增的真实FSDP2 `CHECK_6`、C0/C1/C2、正式训练和训练后推理仍必须在内网分别验收。
- 当前本地分支为 `stage-2`，跟踪 `longlive-cats/stage-2`；工作树包含 planning、`prepare_stage2.sh`、600条metadata及新增缓存入口测试等未提交改动，并有 checkpoints/results/tmp 等用户资产。所有审计和修复必须保留这些现有改动，不清理、不覆盖。
- 持久计划显示训练闭环（Steps 9–11）已有实现并停在检查点 B，Steps 12–14 的 Stage‑2 batch inference/trace/压缩接口仍未完成；本轮从任务书重新验证这些历史结论，不以旧测试通过数量替代代码证据。
- 任务书前520行重新锁定：输出必须是sink外24个全新latent；C/W/H/S/K=8/16/8/1/4、物理KV容量17、每chunk独立UniPC shift5、noisy/exit不提交KV且clean recache后持久KV无autograd；score pack严格1+24、video-global 9750 token timestep，score连续sigma与rollout scheduler不得混用。
- 训练闭环唯一顺序为F1..F5→G→EMA→commit；phase/DFD/EMA/checkpoint由成功G时钟驱动。Phase A/B为G240/40、F1200/200，checkpoint只能完整cycle边界原子保存，并完整恢复双optimizer、三adapter/EMA、双sampler、全rank RNG与来源hash。
- JSONL必须以F/G/cycle独立时钟作为权威数据源，loss numerator/count先跨rank/accum SUM再求global mean；九组PNG/SVG和HTML全部从resolved config推导phase/终点，producer→plotter共用schema并对关键字段缺失fail-fast。
- Stage‑2推理明确不同于旧continuation：单动作EMA/CFG1生成24 latent并decode 25→drop pixel0=96帧；双动作每个episode保留同一原始sink但清其余self/cross KV，分别decode/drop sink后拼成192帧，禁止HOLD/soft re-anchor语义。
- 任务书余下部分锁定H100 C0 cold-save、C1 resume+DMD、C2 forced DFD-discard三cycle门禁；micro2×acc4不满足显存/泄漏/straggler阈值时才退micro1×acc8，仍失败只允许尝试Generator grad-exit saved-tensor CPU offload，禁止改变算法或假装global32解决峰值。
- 压缩接口必须让baseline与W24/W16/W8、C4W12/C4W8、原生K2和S4/S8多帧sink走同一条resolver/cache/rollout/inference路径；S4/S8额外sink只从episode2启用，并恢复episode1时snapshot的detached KV，不能在新prompt下重算。
- 当前stage2文件清单仅包含训练、数据、role、checkpoint、metrics/plot和runbook；没有命名明确的Stage‑2 inference config/runner/shell/trace模块或测试，初步与任务书Steps 12–14仍未完成的状态一致，后续需确认是否有隐藏在通用文件中的未命名实现。
- `TASK-stage2-self-forcing-dmd-dfd.md` 顶部状态仍写“尚未进入trainer”，但同文Steps 9–11和验收表已标完成；这是明确的文档状态漂移，最终必须同步，不能让操作者误判代码边界。
- 当前Stage‑2全量回归基线为445 passed / 1 failed / 14 warnings（117.13s）。唯一失败是`tests/test_stage2_runbook.py`要求`prepare_stage2.sh`保留`expected_step=3750`，而脚本已改为3075；这不是测试环境问题。
- 初始化基线存在明确内在矛盾：任务书第74/220/221行、正式YAML第44–54行及runbook测试锁定step3750；当前脚本第23/33/142/160/188行则把checkpoint、产物名、manifest验证和resolved断言全部锁为3075。选择任一方都会改变Stage‑2 Generator起点和provenance，必须由用户确认，不能自行推断。
- 用户已明确裁决：实际对比发现step3075权重表现更好，Stage‑2所有配置、代码、manifest、测试、文档与产物命名必须统一到3075；与独立Stage‑1 3750历史实验相关的入口不属于Stage‑2，不应被无差别改写。
- 用户明确取消此前其他模型制定的Stage‑2任务划分/检查点暂停规则；任务书中的算法与验收要求继续有效，但实现应端到端连续推进，不再等待旧检查点确认。
- step3075同步后的Stage‑2专属生产代码、配置、任务书和测试中已无正向3750引用；唯一保留的Stage‑2测试字符串是禁止`stage1_step3750`重新出现的负契约。独立Stage‑1 continuation/comparison仍保留其自身3750实验语义。
- step3075首次统一后的path-independent Stage‑2 contract hash为`68b4a3b05535c70b979becf5984898d81b651ead84c929370b7f21dccdde4e2f`；修复A24同点安全分叉B0/B1后，当前hash更新为`a7365f2ec45f74c3918ec05725b5d19b488fa4447a409cc6b5db4ccb114dd6c6`。聚焦provenance/config/runbook回归曾为157 passed，完整Stage‑2回归曾为446 passed、14条既有TorchScript弃用warning；最终计数以当前磁盘态重跑为准。
- `bash -n`（prepare/F25入口）与修改过的Stage‑2 Python `py_compile`通过。全工作树`git diff --check`只报告用户现有600条metadata的CRLF/trailing-whitespace，不来自本轮step3075代码；后续使用任务文件范围的whitespace检查，未经确认不机械重写该数据文件。
- 训练闭环主体约6890行（trainer 1983、checkpoint 2532、train state/transaction 830、metrics/plot 1545），现有测试全绿不足以替代按状态机、事务、分布式归约和恢复顺序逐段核验；三路只读审计正在补充这一证据。
- 主线程已核对trainer主路径：cold/resume先建数据与三role/FSDP，再恢复双optimizer/EMA、逻辑时钟，最后恢复RNG；每substep先快照sampler与全部专用RNG，materialize独立global64 accumulation，FSDP2只在末micro同步，所有finite门禁通过后才optimizer.step，成功G再EMA并commit时钟。
- global mean backward使用`local_numerator * world_size / global_count`补偿FSDP平均归约；每个micro立即backward且只保留一个micro activation，符合global numerator/count与内存边界设计。该结论仍需审计不等数count、no-sync和真实FSDP API测试是否覆盖。
- nonfinite重试只覆盖optimizer前失败并恢复sampler/loader/rollout/t/noise/exit/branch及Python/NumPy/CPU/CUDA RNG；optimizer后参数非finite明确不可rollback并抛错，要求从上一个完整checkpoint恢复。训练入口异常会重新抛出，不会伪装成功。
- 推理审计初报曾把`exit_step=3`未调用最后一次`scheduler.step`列为P0；用真实FlowUniPC K4对多组随机tensor复验后更正：末步目标sigma=0时solver输出与raw `x0_pred=x-sigma*v` bitwise一致，且当前路径已执行4次DiT，因此无数值少步。仍需显式`generate_full_episode()`与真实scheduler parity/trace测试，避免部署语义依赖训练exit API。
- 原生K2 timetable确认为`(999,833)→0`，不能截取K4的`(999,937)`；当前rollout `_validate_contract`和`_new_scheduler`硬锁K4/C8/W16/S1，说明任务书要求的C4/K2/S4/S8通用接口尚未实现。
- metrics/plot独立审计发现2个P1：child JSONL lineage可写`logical_substep_id < resume boundary`并覆盖已提交父历史；plotter的complete终点/phase marker只信任重复metadata，没有与真实`resolved_config`交叉验证，篡改为G1/F5仍能把单cycle标为complete。
- 另有明确P2：完整换行终止的坏JSON尾行被误当“截断尾行”静默删掉；producer→plotter测试是手写字段而非Trainer真实producer；绘图后才重读JSONL计算hash导致partial run图与hash可能来自不同快照；time_breakdown缺少raw低alpha曲线。这些都有明确修复契约。
- timing closure当前validator只验证`parts + closure == wall`恒等式，能接受closure占wall 100%。仓库/任务书没有数值阈值；建议采用`abs(closure) <= max(0.1s, 5% * step_seconds_max)`，已按用户“不可臆断”要求暂停并询问。
- 用户已确认采用`abs(timing_closure_error_seconds) <= max(0.1s, 0.05 * step_seconds_max)`；该值成为共享metrics writer/reader/test的权威门禁，不改变训练算法或loss。
- trainer/checkpoint终审无P0，但确认6个必须修复的P1：A24同点无法按B0/B1合法分叉；rename后marker前崩溃会让latest拒绝回退完整旧点；resume不校验AdamW锁定超参；rank0 RNG验证会实例化其他rank CUDA generator；`_SUCCESS`前的sampler/EMA/provenance验证弱于真实loader；缺少真实FSDP2 sync/no-sync梯度parity门禁。
- trainer/checkpoint另有3个P2：生产provenance未保存代码版本；nonfinite精确重放测试只覆盖未被Trainer使用的transaction helper；写入checkpoint的RNG快照捕获晚于LoRA/optimizer/EMA gather，可能让continued与resumed下一draw分叉。pending live-state只写常量false且重复loader RNG未交叉校验列为P3。
- 已确认正确的trainer核心不变量不重写：严格5F→G→EMA→commit、B1概率、G40 EMA、rank0 CPU branch RNG+broadcast、global-mean缩放、末micro同步与optimizer角色隔离均与规格一致。
- inference终审确认P0交付缺口：尚无Stage‑2 inference config/runner/shell、EMA-only generator加载、96/192帧decode、技术trace/manifest/index；legacy causal pipeline具有双CFG cache和旧commit/continuation语义，禁止代用。正确实现必须复用`pipeline/stage2_rollout.py`。
- inference core需新增纯派生profile/spec、K2/K4真实scheduler、显式`generate_full_episode()`、S4/S8 episode1 detached prefix snapshot与episode2 restore、逐sample/chunk trace；single/two orchestration分别decode `[sink,A24]`/`[sink,B24]`并丢弃各自pixel frame0，双动作拼为192帧。
- 用户已裁决双动作noise语义：每个样本在显式seed上只初始化一次RNG并连续生成48个temporal noise slots，A使用前24、B使用后24；B前不得重置同seed，因此A/B随机起点不同，同时同一`(sample, seed)`可确定性复现。trace需保存总/A/B noise hash，测试需证明batch大小与rank分片不改变映射。
- 用户已裁决正式动作数量以当前内网冻结资产为准：`head_tilt_and_wink=198`、`jump=202`、`play_with_a_cat_wand=200`；任务书旧的均分口径废止，配置、cache门禁、sampler、测试与runbook只接受这一组精确映射。
- 最终推理链已新增strict config identity并贯穿跨rank startup、sample trace、resume与manifest；identity绑定checkpoint/architecture/T5/tokenizer/VAE/metadata/output/profile/seeds/runtime的canonical resolved配置及contract/launch hash，任一rank或旧产物漂移均fail closed。
- 真实Wan VAE encoder固定返回FP32；runtime现先验证FP32 raw latent的shape/device/finite，再显式转换为contiguous BF16交给rollout，避免正式推理首样本必失败。
- 完整内网runbook必须把prepare/smoke/formal/plot/inference输出放在checkout外；prepare默认工作根已迁出仓库，正式推理在任何模型加载前拒绝checkout内output root和dirty Git。
- metrics修复代理中断前没有修改production或test文件；当前工作树只有step3075/planning及用户原有资产改动，可以从红测开始安全恢复。
- Phase 15最终结论：正式动作分布只接受`head_tilt_and_wink=198`、`jump=202`、`play_with_a_cat_wand=200`；所有Stage‑2 Generator初始化/manifest/文件名只接受Stage‑1 step3075。人工fixture里的均分200和独立Stage‑1 step3750实验不属于正式Stage‑2契约，不能机械替换。
- checkpoint/resume最终采用双层身份：research contract允许A24/G240同点分叉B0/B1，G240之后严格绑定arm；explicit G240是immutable ancestry anchor，本地child必须逐边以canonical parent path和manifest SHA证明可达。checkpoint内认证的metrics JSONL前缀解决了外部分支空logdir与独立完整plot lineage。
- C0/C1/C2不能只证明“能加载”：C0/C1保存无消费next-F1 probe，下一阶段首个F1按生产顺序重放sampler/exit/loader/rollout/timestep/noise/计数器后只消费一次；任何cycle出现nonfinite即使精确重试成功也不得通过smoke。
- baseline部署推理使用Generator-only EMA快读路径，canonical EMA由同一bytes snapshot校验hash并反序列化；T5/tokenizer/VAE/architecture/Generator base由checkpoint绑定的source/role provenance认证。rank0只流式hash一次大资产，其余rank以stat identity在真实loader前后闭包，避免8倍大文件I/O。
- inference identity必须区分四类hash：resolved contract/launch只描述解析配置，runtime contract/launch再加入已认证模型资产内容；output root只影响两种launch，资产内容只影响两种runtime。沿用同名但不同含义的两个hash会破坏审计，现exact-schema已拒绝旧键。
- 正式双动作noise不是A/B重置同seed：每个sample seed只初始化一次generator并一次产生连续48 slots，A/B分别使用前/后24，因此起点独立且`(sample,seed)`确定复现，不受batch/rank-stride分片影响。
- 8个named rollout profiles复用唯一episode kernel；K4为`999,937,833,624`，原生K2为`999,833`。部署最后一步直接采用raw x0以保持训练exit bit pattern；不能声称BF16额外terminal scheduler step bitwise相等。C4/K2/S4/S8目前是技术压力接口，不是已训练的部署适配结论。
- 推理sample落盘前必须完成runtime assets、config/code/metadata/plan、Generator checkpoint和全部已有video+trace pair的全rank共识。output root还需固定canonical dev/inode/mode；root被换成symlink或同路径新目录时必须在创建child前拒绝，manifest继续作为最后commit marker。
- 本地最终动态证据为Stage‑2 656 passed、仓库正式tests 964 passed及2 subtests；14条warning均为既有TorchScript弃用提示。两路独立最终审计均为P0=0、P1=0。真实8×H100新版CHECK6、C0/C1/C2、B1/B0训练、绘图、56视频和人工质量仍是外部验收，不能用本地结果替代。
- 全工作树diff检查唯一已知例外是用户既有`training_sets/metadata_600clips_480x832_buckets.csv`的CRLF/trailing whitespace；该文件当前也不是正式600条内网资产。为保护用户数据不擅自改行尾或补记录，代码/文档范围单独执行whitespace门禁并在交付中明示。

## 2026-08-13 Phase 14：Stage‑2 F25 latent 专用重提

- Stage‑1配置的93是像素帧数，VAE输出是F24；Stage‑2必须固定读取像素帧0..96共97帧并输出F25，训练只取`video_latent[1:25]`作为24个真实未来目标。
- 新入口复用仓库已有`prepare_stage2_i2v_f25_cache.py`，它会从旧F24逐样本重新编码97帧并要求新F25前24帧与旧F24逐位相同；禁止padding、复制或截断。
- F25输出目录必须独立于旧Stage‑1 cache；YAML的`source_cache_manifest`指向新F25的`cache_manifest.attested.json`，`cache_dir`指向F25目录，negative字段指向独立negative目录内的manifest。
- 已确认的数据策略：F25迁移继续使用生成旧Stage‑1 cache时完全相同的六列metadata；动作标签使用独立`video,action_id` sidecar，video字符串必须与metadata逐字相同。
- 当前本地`metadata_600clips_480x832_buckets_action.csv`与`action_labels_600cats.csv`都只有1条，不能用于正式运行；七列metadata还会改变row hash，不能替换旧cache绑定的六列metadata。
- 专用脚本先做CPU级600条、动作数`198/202/200`、路径对应与manifest兼容预检，再允许8×H100 VAE重提；最终依次产生F25 base、attested source、negative conditioning和`stage2_i2v_manifest.json`，不加载DiT或启动训练。
- 正式cache manifest绑定完整launch hash，因此专用脚本使用与`prepare_stage2.sh`相同的architecture/generator/teacher默认路径环境，防止缓存完成后因模型路径不同而被训练启动门禁拒绝。
- F25逐行产物与completion sidecar支持安全断点续跑；attested/negative/final manifest存在时会重新严格验证，negative半成品目录会fail closed。
- 最终验证：新增6项入口/输入测试通过；完整`tests/test_stage2_*.py`为443 passed、14条既有TorchScript弃用警告；Black、Ruff、py_compile、bash syntax与whitespace检查通过。

## 2026-08-12 Phase 13：前5项检查一键化

- 用户明确只做正式训练前前5项，要求删除操作层面的冗长说明。
- 最小用户接口应是一个`prepare_stage2.sh`：顶部集中路径，单命令执行，成功只认`CHECK_1_TEACHER_PASS`到`CHECK_5_ROLE_INIT_PASS`。
- C0/C1/C2、正式训练、resume和plot全部移出本轮简版手册；它们的生产代码不删除。
- 昂贵F25和模型merge允许在严格复核后复用；只存在一半的产物或验证失败必须停止，不能自动覆盖。
- `prepare_stage2.sh` 从脚本所在clean checkout执行，并把产物限制在仓库外工作根；依次执行teacher、step3075 EMA Generator、配置、数据、role init和真实FSDP2六个gate，不会调用训练入口。
- teacher仍直接使用原始双向native checkpoint目录；`provenance.source_sha256`绑定完整DiffSynth `merge_manifest.json`文件SHA，不使用其内部`merged_state_sha256`。
- 数据gate调用仓库真实F25、`upgrade-source-manifest`、`prepare-negative`和正式audit CLI；negative正式文件名为`negative_conditioning_manifest.json`。
- 极简文档只有43行，保留动作标签要求、两个必需export、可选路径覆盖、单命令和最终PASS列表。

## 2026-08-11 Phase 12：real-score manifest 后的 Stage-2 H100 指南重生成

- 用户要求一次性重生成 teacher manifest 之后的全部操作指导，不能只修正 manifest 生成段。
- 当前已知漂移：旧手册仍以检查点A/训练前准备为终点，但仓库已经具备600cats正式配置、严格trainer、C0/C1/C2 smoke、checkpoint/resume、JSONL和九图/HTML实现。
- 本轮会以生产脚本和CLI为唯一真相，逐项核对参数、输出目录、成功信号和fail-closed边界；历史规划文字不作为可执行命令来源。
- 全仓盘点只发现一份正式Stage-2操作者手册：`docs/STAGE2_H100_QUICK_DEPLOY_ZH.md`；`TASK-stage2-self-forcing-dmd-dfd.md`是实现规格，不应被当作运行手册。`prepare_stage2.sh`是资产准备入口，也必须让其末尾输出明确指向手册中的下一步。
- 当前旧手册在formal cache audit/init-only后结束，没有覆盖已落地的`train.py --stage2-smoke C0|C1|C2`、formal冷启动/resume、checkpoint lineage和`plot_stage2_training.py`九图HTML，因此manifest后半段必须整体替换而非局部补丁。
- 生产入口存在必须先闭环的资产漂移：`prepare_stage2.sh`指向真实`checkpoint_model_003075`并命名step3075 Generator产物，但正式YAML/旧手册仍写`generator_stage1_step: 3750`和step3750文件名。
- 旧手册导出了`LONG_LIVE_STAGE2_ARCHITECTURE_ROOT/GENERATOR_BASE/GENERATOR_MANIFEST/REAL_SCORE_BASE/REAL_SCORE_MANIFEST`，但当前YAML这些字段是硬编码绝对路径，不读取这些环境变量；因此旧的“锁定同一配置”命令不能证明训练使用了刚生成的资产。
- `prepare_stage2.sh`当前只真正生成teacher manifest；其中`STAGE1_BASE/STAGE1_CKPT/G_MERGED/G_MANIFEST`只是未消费变量。后续指南必须明确Generator merge是已经完成的前置产物还是在本脚本中生成，不能继续制造“脚本已准备全部模型”的错觉。
- 本地未跟踪的旧`real_score_teacher.manifest.json`可验证历史错误：`provenance.source_sha256`为DiffSynth的`merged_state_sha256=d8ba...`，而新版脚本应写完整`merge_manifest.json`的文件SHA；它不能作为新手册示例中的合格产物，必须在内网删除旧sidecar后由新版脚本重新生成并校验。
- 本地存在一份完整、可审计的Stage-1 step3750 checkpoint manifest（`completed_step=3750`）；与此同时新版`prepare_stage2.sh`改指另一路step3075结果。两者是不同Generator候选，指南必须只允许与YAML声明一致的一路，不能靠文件名猜选。
- Trainer真实产物：JSONL默认为`<logdir>/metrics/stage2_train_metrics.jsonl`；checkpoint目录为`checkpoint_stage2_gNNNNNN`且`_SUCCESS`最后写入；默认成功/每次smoke结束会生成`<logdir>/plots/`中的9组PNG/SVG及`index.html`，除非显式`--no-visualize`。
- 当前resolver明确把Generator来源锁为Stage-1 step3750（代码、角色`base_source`和测试三重锁定）；因此step3075不是“换一条文档路径”即可合法使用的候选。本轮不擅自改变研究契约，而是从teacher-only脚本中删除未消费的step3075/G变量，并让Generator步骤以正式YAML的3750为准。
- YAML可以安全改为`oc.env`承载architecture/G/teacher的操作路径，同时保留当前绝对路径作为默认值；这些路径本来就被contract hash排除，默认解析和锁定contract hash不变，却能让clean clone真正使用手册刚生成的外部资产。
- `.gitignore`忽略`*.pt/*.pth/*.log/*.html`，而trainer同时拒绝tracked dirty、untracked和ignored文件；因此正式流程必须使用“代码clean clone + 仓库外资产/cache/logdir”的双目录布局。把checkpoint或训练输出写进执行clone会在下一次启动/resume时按设计失败。
- F25 producer只要求最终配置字符串可解析，不要求未来的attested source/negative文件已经存在；可在F25前就把`LONG_LIVE_STAGE2_SOURCE_MANIFEST`和`LONG_LIVE_STAGE2_NEGATIVE_MANIFEST`设为最终路径，从第一步起保持同一launch hash。formal audit/trainer会重新绑定该hash。
- 最终实现保持contract hash `aa4d7be1e05c846df14cee5417a298afe668429f41faa671f3021754a5616c00`不变；新增外部模型路径env只改变合法的launch-specific路径绑定。
- 最终本地证据为443 passed、14条既有TorchScript弃用warning；`bash -n`、Black、关键Ruff、config resolver、文档陈旧字符串搜索和`git diff --check`均通过。本地没有H100，不声称C0/C1/C2或formal真实训练已经成功。

## 2026-08-11 Stage‑1 LoRA/merged 四卡推理任务

- 当前 `run_stage1_merged_checkpoint_comparison.py` 只对预 merged `.pt` 执行新推理，并复用原 runner 视频；它没有对 `base + adapter_ema.safetensors` 执行动态 LoRA 推理。
- 原 `run_stage1_training_checkpoints_validation.py` 会先调用 `merge_stage1_ema_checkpoint()`，再生成不含 adapter section 的完整 generator 配置，所以原参考视频同样属于 merged 路径。
- 正式新目标是让动态 LoRA 与预 merged 两路共享同一 reference prepared carrier/config、相同seed/采样参数，并在报告中明确映射，避免把旧参考误称为动态 LoRA。
- 性能设计必须先解决6个case被两个3样本分辨率bucket拆分后的4卡调度；直接沿用当前 `DistributedSampler(..., drop_last=True)` 的4-rank torchrun会丢样本，不能作为正式方案。
- 对总计12次生成（2种格式×2个geometry bucket×3条样本），最均衡且加载开销最低的正式拓扑是4个并发单卡进程，每个进程固定一种格式和一个bucket并顺序生成3条；相比两轮各4进程，总模型加载从8次降为4次，且每卡总工作量一致。
- 动态LoRA必须从training checkpoint的`resolved_config.yaml`读取精确adapter schema，并通过`load_lora_safetensors_strict()`加载`adapter_ema.safetensors`；原inference中的`torch.load(lora_ckpt)`不能读取safetensors，也缺少完整key/shape/dtype/finite/value回读门禁。
- merge与dynamic-LoRA的全局RNG消耗不同，不能只调用相同`set_seed()`就宣称noise相同；正式实现使用`base_seed + row_id`创建独立CUDA Generator，因此两路每行noise与worker分片/模型初始化顺序解耦。
- merged companion manifest提供必要provenance：正式对比前同时绑定base SHA、training manifest SHA、step3750、EMA adapter SHA、merged output SHA/size/BF16/strict-reload；否则两个视频即使可生成也不能证明来自同一组权重。
- 原infer_stage1历史结果同样是先merge后推理且使用旧顺序RNG，因此不能冒充dynamic-LoRA或参与严格数值等价判断；它保留在HTML details中作为上下文，主并排固定为左pre-merged、右dynamic-LoRA。
- 用户确认历史推理目录已删除，因此reference只能是可选的加速/历史上下文输入，不能成为merged-vs-dynamic-LoRA对比的硬依赖。
- fresh路径应直接复用`prepare_causal_testsets()`：把共享prepared数据放在本次空work dir下，使用converted base、architecture、T5、tokenizer、VAE和锁定的24-latent/UniPC50/CFG5/seed1参数；两路仍从该manifest克隆以保持完全一致。
- 显式reference路径继续保留原step/metadata/技术输出门禁；fresh路径没有历史视频，因此HTML只显示主并排，report用明确`preparation_mode=fresh`而不是伪造reference validation。

## 2026-08-11 检查点B实现边界

- 用户明确授权继续Steps 9–11；本轮交付严格限定为trainer、optimizer/EMA/nonfinite、checkpoint/resume、JSONL与静态可视化，完成本地正反测试和终审后停止。
- 唯一合法训练cycle为`F1→F2→F3→F4→F5→G→EMA`；checkpoint只允许`next_substep=F1`且无pending grad/batch/branch/KV的完整cycle边界。
- Phase、generator epoch、B1十点DFD概率、milestone、EMA初始化/decay和checkpoint频率只能由成功`completed_generator_updates`派生；nonfinite attempt不得推进任何已提交时钟、sampler或RNG。
- JSONL是唯一权威metric source；必须沿用现有lineage/strict JSON/fsync语义，但使用Stage-2 typed schema和F/G/cycle独立横轴；plot fixture必须由真实producer生成。
- 本次继续实现不等于宣称内网H100准备门禁已通过；所有H100 profile、显存和正式训练结果继续保留为外部验证。

### 检查点B最终技术结论

- Trainer与checkpoint已形成真实接口闭环，不复用Stage‑1的world6/二维mesh假设：Stage‑2只选择性聚合world8一维FULL_SHARD的G/F LoRA与两个optimizer state，禁止聚合三份5B base。
- C0/C1/C2 lineage是硬门禁：C1只接C0，C2只接C1且不保存；正式训练拒绝任何带`smoke_probe`的checkpoint，避免预检权重进入正式run。正式训练必须另用全新输出目录。
- Generator rollout audit给出单样本query tokens；trainer现按实际microbatch乘一次，再按world size换成global logical tokens，micro2和micro1均有参数化测试，避免吞吐固定低报两倍。
- DMD/DFD diagnostic记录的是clamp前raw denominator；有限的0合法，logger按nonnegative校验，不能在optimizer/EMA/clock提交后因0误判失败。
- JSONL完整状态不仅看G/F终点，还必须看到latest lineage自己的`run_end.status=complete`及逐cycle严格5F→1G；plot输出9组PNG/SVG和静态HTML，核心loss/吞吐缺字段直接失败。
- H100 smoke的NVML余量门禁依赖`pynvml`，已把`nvidia-ml-py`加入正式依赖；`train.py`在导入项目模块前关闭bytecode，仍建议部署使用`-B`并要求checkout中ignored文件为0。
- 最终本地验证为`735 passed, 2 subtests passed`；14条warning都是既有TorchScript弃用提示。真实8×H100健康路径、异常watchdog、显存余量和DataLoader data-wait只能由用户C0→C1→C2实测，不能由CPU测试代替。

## 2026-08-08 Stage-2 分批执行约束

> **历史记录，非当前门禁。** 本节的逐Batch暂停与“H100‑002通过前不得进入Step 3”已被2026‑08‑10用户确认的三个检查点覆盖；当前状态只看本文后部“三个检查点”和 `task_plan.md`。

- 恢复上下文确认：Stage-2 任务规格已完成，production code 尚未开始。
- 本轮必须先产出任务拆分，再开始首批实现；拆分需把文档步骤映射到具体文件、CPU 本地测试、H100 门禁和暂停点。
- 第一批只能形成最小可验证闭环；推送远程 `stage-2` 分支后停止后续实现，直到用户明确反馈内网 H100 验证成功。
- Git 发布前必须核对并隔离工作树中已有用户改动，只暂存本轮确认范围。
- 已完整阅读 723 行任务规格；文档的 Step 1 明确要求只新增 baseline config、严格 config resolver 和 Stage-2 测试骨架，建立当前相关测试基线，不改 production 训练路径。
- 最终批次边界：Batch 1=Step 1配置契约；Batch 2=Step 2角色；Batch 3=Step 3数据/sampler；Batch 4=Step 4 score边界；Batch 5=Steps 5–7 cache-safety原子rollout；Batch 6=Step 8损失；Batch 7=Steps 9–11训练脊柱；Batch 8=Steps 12–14推理/压缩/runbook；Step 15 始终由用户执行。
- Batch 1 的合理 H100 门禁不是正式训练，而是确认配置能在内网解析真实路径/拓扑、修改前回归与新增契约测试行为一致；在该门禁通过前不进入任何模型角色或算法 production 实现。
- Git 基线：当前分支 `stage-1` 位于 `4c0bb6a`，跟踪 `longlive-cats/stage-1`；用户远程是 `longlive-cats=https://github.com/zonghui-liu-opt/LongLive_cats.git`，`origin` 是只应视为上游的 `NVlabs/LongLive.git`。新 `stage-2` 必须从当前 Stage-1 commit 分出并推到 `longlive-cats`，不能误推 upstream `origin`。
- 当前既有工作树包含三份 planning 文件修改、未跟踪的 Stage-2 任务文档以及未跟踪 `results/` 用户数据。`results/` 必须永不暂存；任务文档和 planning 文件属于本轮/前序 Stage-2 交付，可在最终审查后与首批契约代码一起显式暂存。
- `gh` 已安装且登录为 `zonghui-liu-opt`。用户只授权创建/推送分支，没有授权开 PR，因此首批只 commit/push，不扩大到 PR。
- 仓库目前没有 Stage-2 文件。`train.py` 只注册 `score_distillation`/`diffusion`，Step 1 不要求接入新 trainer；现有 legacy `configs/train_i2v_dmd.yaml` 使用旧 score-distillation 语义，正式 baseline 配置必须独立新增且不让 legacy normalize 默认值掩盖缺字段。
- 现有 Stage-1 正式配置是 `configs/train_i2v_ar.yaml`（不是 `configs/train_stage1.yaml`）；已有 `tests/test_stage1_config.py`、DMD conditioning、lazy import、JSONL plot 等可作为修改前回归集合。
- `utils/config.py::normalize_config()` 会把 grouped sections 展平，并对 `trainer=score_distillation` 自动注入 legacy DMD defaults（如 real guidance 3、all-causal 角色kwargs）。Stage-2 必须在调用legacy normalize之前识别raw YAML并走专用resolver；Batch 1用characterization测试明确锁住normalized mapping会被strict resolver拒绝，后续接`train.py`时不得颠倒顺序。
- `train.py` 当前只有 Stage-1 `stage1_i2v_cache`/FSDP2 gate 和两个 trainer 分支；按 Step 1 本批不改 registry。真正注册 `stage2_distillation` 应留到后续 trainer batch，避免一个不可执行的半接入口。
- Stage-1 config test的既有风格是：从正式 YAML 经 `normalize_config` 读取，断言锁定拓扑/shape/计数，并用独立 resolver 推导稳定时钟/hash。Stage-2 可沿用“frozen dataclass + derived fields + stable hash + fail-fast”的方式，但必须是独立 schema。
- Legacy I2V DMD config 明确是 32 latent、local_attn_size32、all-causal、共享 r128、旧 CFG 参数；Stage-2 config 测试应显式断言这些字段不存在或被拒绝，防止误启动旧路径。
- Legacy `trainer/distillation.py` 的结构进一步证明隔离必要：初始化时共享一份 `config.adapter` 扫描全部 attention-block Linear；LoRA resume只含G/F adapters与单一`step`；LoRA模式禁用EMA；dataset会现场加载/编码视频和T5；训练loop以 `step % ratio` 决定G并在同一accumulation batch中先G后F，和锁定的五个独立F成功更新后再G完全不同。
- Legacy trainer的异常处理只打印 traceback 而不重新抛出，可能让作业返回成功；checkpoint也不是完整cycle原子目录协议。Stage-2不得复用其主loop/checkpoint实现，首批 config 应把新 trainer 名称、init/resume互斥入口与完整原子checkpoint字段写死，真正接入留到后续批次。
- Legacy trainer会给text encoder和VAE分配训练进程资源，并为real/fake/G套旧FSDP1 wrapper；Stage-2 baseline明确要求T5/VAE不驻留、三role独立wrap、单机world8/full-shard。首批配置应把这些作为可静态断言的契约。
- Legacy `model/dmd.py` 的I2V helper会把initial覆盖进现有序列slot0；score timestep按block抽样/reshape；normalizer按block；real guidance用 `cond + legacy_scale*(cond-uncond)`；grad使用`nan_to_num`；fake loss把x0再转回flow。这些正是规格P0，Stage-2 config/契约测试必须禁止旧helper/旧CFG字段承担新语义。
- Legacy `pipeline/self_forcing_training.py` 由`num_max_frames * frame_seq_length`直接分配cache，random exit在pipeline内部按block抽样；noisy/exit forward默认会写共享cache；clean recache还会额外加`context_noise`。Stage-2需要显式派生`capacity=S+W=17`、外部exit、noisy discard与exact clean recache接口，不能仅参数化旧类即宣称正确。
- 现有pipeline在独立首帧I2V时把noise长度仍当整个24槽并将initial覆盖首槽，因此输出只有23个新latent；首批必须以静态配置/公式契约锁定`generated=24`、`score=25`、`decode=25→drop1→96`，实际修复推迟到rollout/pack批次。
- `WanDiffusionWrapper` 对noncausal模型固定 `uniform_timestep=True`，forward直接取`timestep[:,0]`；`seq_len`硬编码28160；flow↔x0通过离散scheduler nearest lookup并回到原dtype。后续Step 4/8需新增显式mixed-token timestep与continuous-sigma接口，同时保留旧uniform-time parity；首批只能锁定配置/公式，不能修改wrapper。
- Wrapper已有延迟cache-update环境变量协议，但仍会最终应用更新，不能表达Stage-2 noisy forward的discard语义；后续应设计正式`commit_self_kv`接口而非依赖环境变量。
- 本地环境与正式requirements有显著漂移（agent审计为macOS arm64/Python3.10、无CUDA、diffusers0.38/transformers5.9/peft0.19；requirements锁定diffusers0.31、transformers<5）。本机通过只能作为CPU契约证据，首批推送后仍需用户在内网H100正式环境验证依赖/解析。
- 修改前最小相关回归已由只读测试审计跑出两组合计64 passed；全`tests/` collect为262。第一批改后必须复跑相同集合，并新增Stage-2 config契约测试；不能把本机无CUDA结果包装成H100通过。
- 独立批次审查建议把Step 5–7作为不可拆的cache-safety原子批；Step 9–11完成前不得执行正式C0/C1/C2。该依赖已吸收进正式分批计划。
- 为最大限度缩小第一次变更面，本轮首批严格限定为文档Step 1，而不是提前实现角色或真实数据路径。首次H100门禁只核对正式环境中的配置/派生契约；真实600-cache和三角色init-only门禁分别留到Batch 2/3完成后。
- Balanced sampler尚有需在Batch 3固化的确定性细节：10个G batches消费640样本，必须显式定义每动作队列wrap/reshuffle、额外名额相位和resume state；Step 1只声明策略枚举，不实现或静默猜测数据标签。
- 主代理在修改前复跑两组正式首批相关回归：14 passed + 50 passed，共64 passed；运行时禁用pytest cache和bytecode，未改工作树。
- 当前仓库实际UniPC scheduler在CPU以K4/shift5解析为`[999,937,833,624]`且terminal sigma为0；`DEFAULT_NEGATIVE_PROMPT` UTF-8 SHA256实算为规格锁定的`ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e`。
- Batch 1最终配置只保存原始研究输入，所有chunks/history/capacity/token/update totals与UniPC timetable均不作为YAML第二真相；UniPC测试仅是仓库scheduler characterization，生产启动漂移门禁明确延期到Step 7。
- CFG执行路径已静态锁定：Generator与fake-score均为conditional-only单forward，Generator仅一套self-KV和一套cross-KV；real-score为顺序cond/uncond双forward并使用`uncond+5*(cond-uncond)`。EMA显式只指向Generator adapter，fake/real无EMA。
- H100 batch配置统一使用candidate而非approved措辞；训练seed允许任意非负值并纳入hash，fsync允许任意正整数；saved-tensor CPU offload只允许micro1×acc8候选，compile仅允许无cudagraph安全mode。
- Phase-B schema同时表达release `A24+B4 dmd_dfd`、A-only终止和相同40G/200F预算的`dmd_only` matched control，且明确A/B不重置optimizer。
- 配置fingerprint分为`launch_hash`与`contract_hash`：前者覆盖本次完整路径/初始化，后者排除operator-local路径和init/resume差异，供合法resume比较；资产内容SHA将在Step 2/3并入正式manifest门禁。
- Resolver现已typed暴露architecture/checkpoint/manifest/data/negative/jsonl等后续步骤运行输入，无需回钻raw dict；saved-tensor offload scope锁为`generator_grad_exit_only`。等价数值写法共享语义hash，超大数字与非字符串key统一fail-fast。
- review修订后的本地证据为Stage-2 103 passed、Stage-1/DMD相关回归64 passed；Black、Ruff、py_compile、CLI与空白检查通过。本机无CUDA，不能把这些结果表述为H100模型/算子/训练通过。
- Batch 1内网结果为Stage‑2 103 passed、配置契约全部通过、相关回归63 passed。用户显式排除`test_release_stage1_config_has_one_locked_source_of_truth`，因为它约束公开仓库Stage‑1 YAML而非真实内网Stage‑1训练参数；该排除不掩盖Batch 1改动，因为本批未改Stage‑1 YAML/解析路径，仍保留其余共享config/LoRA/FSDP/scheduler回归。
- Batch 1门禁据此通过。Batch 2只进入任务文档Step 2；不得提前实现数据、score adapter、rollout、loss、optimizer、EMA或trainer注册。
- Batch 2修改前共享基础件基线为128 passed。Stage‑1 FSDP2实现锁定DP2×SP3的6卡HSDP，不能复用其拓扑常量或wrapper到Stage‑2 world8/SP1/FULL_SHARD；可以复用通用LoRA exact-target、canonical adapter与DTensor审计思想，但Stage‑2必须有隔离的8卡FSDP2契约。
- Batch 2最终严格停在init-only：G读取step3750 EMA merge v2，real/F分别重新物化同一独立teacher SHA，G/F各自注入r32/r64 fresh LoRA，real无adapter；三角色分别以root+30 blocks执行1D FULL_SHARD并审计全部冻结/可训练参数为`Shard(0)` DTensor。
- Teacher native入口只接受BF16 safetensors单文件/index/目录，manifest绑定index、全部shards、逐tensor schema/dtype与architecture config；LongLive wrapper `.pt`接受`generator/real_score/model`唯一selector和受限legacy serialization，但未知格式、`.bin`、裸root `.pt`与训练状态payload均fail-closed。
- init-only preflight要求clean HEAD、单机world8 H100/BF16/NCCL；每个阶段做WORLD错误共识，运行时tripwire同时拦截标准Module调用与直接`.forward()`、optimizer/EMA/T5/VAE/DataLoader，角色对象/存储隔离和所有rank副作用结果明文写入原子`ROLE_INIT_COMPLETE` manifest。
- 本机最终Batch 2相关门禁为212 passed、14条已知`torch.jit` deprecation；用户指定Stage‑1/DMD回归为63 passed、1 deselected。无CUDA/真实5B资产，因此必须由`H100-002`验证实际strict load、8卡DTensor拓扑和主机内存峰值后才进入Step 3。
- 外部仍需内网确认：正式step3750 checkpoint是否具备新版raw/EMA metadata与完整源目录；real teacher真实物理格式、可信训练/转换来源SHA及cat-domain/video-global flow人工attestation。任一缺失均应停止，而不是自动猜格式或补写原件。

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

## 2026-08-10 H100‑002 历史手册审计（已失效）

> 2026‑08‑10 用户后续指令已覆盖本节的“必须单独停在H100‑002”结论。保留以下内容只作历史审计；当前执行以 task_plan.md 的三个用户检查点为准。

### 当时唯一允许范围
- 任务文档第3行、Step 2暂停边界和既有 Phase 9 计划一致：Batch 2代码已经推送，必须先由用户在单机8×H100运行 init-only 门禁。
- 在 H100‑002 成功前不得进入Step 3，不得实现cache loader/sampler，不得执行forward、backward、optimizer、EMA或训练，也不得声称C0/C1/C2通过。
- 当前可复现代码基线为`stage-2@bde0142`；本地`results/`是用户未跟踪数据，保持原样。

### 快速部署文档需要解决的问题
- 旧文档434行，把已完成的H100‑001历史、当前H100‑002、资产生成细节和通用模板混在一起；操作者无法快速判断“现在只做什么”。
- H100‑002真正需要操作者确认的输入应集中为：clean代码目录、Stage‑1 immutable base、Stage‑1 step3750 checkpoint、Wan2.2架构目录、cat-domain bidirectional teacher、teacher可信来源ID/SHA与仓库外资产/输出根目录。
- `scripts/preflight_stage2_roles.py`会检查整个worktree（含未跟踪文件）必须clean；因此代码应使用独立clean clone，权重、cache、日志和输出全部放仓库外。
- 成功不能靠“命令看起来跑完”判断；必须同时看到torchrun退出码0、`ROLE_INIT_COMPLETE`存在、`_SUCCESS`不存在、manifest审计脚本打印`H100-002 manifest audit: PASS`。
- 当前门禁只验证三模型资产、G/F LoRA角色隔离和8卡FSDP2初始化；不读600-cache，也不验证训练显存、rollout/loss/resume。

### AutoDL/H100边界
- 本任务没有提供AutoDL SSH或AutoDL路径，也没有要求从AutoDL执行远程部署；因此不运行AutoDL命令、不创建虚构路径映射。只采用其安全原则：大资产不进Git、输出放数据盘/仓库外、先验证再复用、记录可恢复的真实命令与hash。

## 2026-08-10 用户确认的 Stage-2 三个检查点

- 旧的“Batch 2必须先单独停、Batch 3再单独停”等内部划分不再作为用户门禁；内部步骤由Codex运行正反测试与回归自行保证。
- 检查点A覆盖原Steps 1–8：配置/角色、600-cache与balanced sampler、1+24 score pack、mixed timestep、24-new rollout、KV安全、UniPC random exit、DMD/DFD/fake-flow。完成后停止，不提前实现trainer。
- 检查点B覆盖原Steps 9–11：5F→1G trainer、phase/EMA/nonfinite、checkpoint/resume、JSONL与PNG/SVG/HTML可视化。完成后停止，交给用户H100 smoke。
- 检查点C在smoke通过后完成原Steps 12–14：batch推理、技术trace、压缩/sink通用接口及剩余验收。
- 每个检查点的GitHub流程是：先让用户检查本地代码；用户确认后再写该节点简洁中文快速部署文档、提交并push `stage-2`，然后根据内网结果继续。

## 2026-08-10 检查点A最终审计结论（用户已通过，等待内网H100门禁）

- 官方Stage‑1 producer实际保存`video_latent[24,48,H,W]`，但其slot0是首帧sink，Stage‑1训练会用独立initial覆盖且不计loss；有效future只有slot1..23共23帧。Stage‑2锁定3×8=24个全新latent，因此正确监督仍需F25的`video[1:25]`。把完整F24的slot0误称future虽能让shape测试全绿，却会重复sink并缺失最后target；该路径及289/571快照均作废。
- 三个要求无法同时成立：保留24-new算法、直接复用官方F24、不重编码600条视频。用户已选择保留24-new：合格F25原字节复用，F24从每条至少97帧的原视频固定取0..96并一次性编码F25。
- 用户已明确选择并锁定：保持24-new算法；现有cache若为严格合格F25则直接使用，若为F24则从对应97帧原视频重新提取F25。方案不可修改，后续实现与验收均以此为唯一口径。
- 数据侧不能只做22/21/21动作平衡；micro2还必须按30×52/52×30方向排成rank-local同形状组，否则collate会在正式训练首批失败。sampler状态现同时绑定action与shape hash，不可满足micro2时明确要求micro1×acc8。
- score noising必须保留FP32，但送入Wan patch Conv3d前只转换模型输入到BF16；FSDP2 root也必须禁止自动cast语义输入，否则999/937等timestep会被BF16舍入。真实Wan回归已锁住这两个边界。
- Generator exit x0只能使用UniPC运行时精确sigma；`t/1000`属于bidirectional score连续契约，两者不可混用。
- Stage-2 cross-attention cache已从“分配但底层绕过”改成显式opt-in真实缓存；legacy/cudagraph路径仍保持原绕过语义。self-KV只在sink和每chunk clean recache提交，持久K/V均detached。
- 正式训练启动不能只信任旧Stage-2 manifest自哈希；门禁现重新绑定当前metadata、source cache、negative artifact、action sidecar、config/code hash，并逐文件复核600个artifact SHA。legacy Stage‑1 source manifest使用严格、非覆盖式、带操作者证明的文本编码provenance升级；正式CLI强制`python -I -B`，dirty/untracked/ignored代码、Git环境重定向和读取期间文件替换均会fail-closed。
- native F25整链已闭合：base manifest与`_F25_SUCCESS`、文本attestation、negative、formal audit和Dataset/runtime只接受同一来源；proven F25不加载VAE，缺旧帧声明的F25仅做全25帧bitwise复验且仍发布原字节，F24发布前必须满足前24帧bitwise parity并保留其余tensor。加入外部正式metadata/action路径绑定后，最终证据为F25 focused 88、Stage‑2 329、全tests 611及2 subtests passed；两次独立代码终审与H100指南终审均为P0=0。
- 本地证据只能证明CPU/tiny逻辑和真实小模型接口；8×5B FSDP2/NCCL、H100 Triton/FlashAttention、真实600-cache及teacher猫域/video-global provenance仍必须由用户内网门禁确认。

## 2026-08-07 Stage-2 LongLive-2.0 任务文档

## 2026-08-11 Phase 11 复审初始事实

- 用户明确要求 real-score 与 fake-score 的初始化底座为 DiffSynth-Studio SFT LoRA merge 进 base 后的双向 attention 模型；Generator 仍应保持 Stage-1 causal merged 初始化，三者不可混用。
- 当前新增权威候选是 `checkpoints/bi_direction/merge_manifest.json` 和 `prepare_stage1.sh`，但“推理脚本能加载”不等价于通过 Stage-2 的 teacher provenance、BF16、架构、独立对象/存储与 checkpoint lineage 门禁。
- 正式配置已迁移为 `configs/train_i2v_stage2_600cats.yaml`；删除旧文件名后必须同步更新测试、preflight、cache 工具与运行手册中的所有入口，并重新复核严格 schema、路径绑定和 resume 兼容性。
- 本轮只读审查将分别验证：同一双向权重来源、real 冻结与 fake fresh r64 LoRA、两份模型对象/Parameter/storage 隔离、checkpoint 对 teacher 资产 hash 的绑定，以及 Stage-1 新共享 loader 改动是否影响 Stage-2。

### Phase 11 最终结论

- **无需改角色加载器：** `stage2_role_init` 对score角色固定构造非causal `WanModel`，real/fake分别strict load同一SHA；real无LoRA且冻结，fake再挂fresh r64，`Stage2DMD`审计三个wrapper/model/Parameter/storage全部互斥。
- **唯一正式配置：** canonical 配置为 `configs/train_i2v_stage2_600cats.yaml`；旧 `train_i2v_stage2.yaml` 删除，`tmp.yaml` 不发布。正式配置已恢复 world8/DP8、补齐 `source_cache_manifest`，并绑定 600cats Stage-2 资产路径。
- **必须换teacher输入：** `converted_causal_base.pt`及其manifest只服务Stage-1 causal推理，不是Stage-2双向teacher资产。正式cold start应直接指DiffSynth merged目录中的原生BF16 safetensors，并用`create_stage2_teacher_manifest.py`生成自哈希、架构绑定、cat-domain/bidirectional/video-global attestation齐全的sidecar。
- 用户提供的`checkpoints/bi_direction/merge_manifest.json`首个正式验证错误为缺`manifest_sha256`；即使补hash，top-level schema仍不兼容。它适合作为新teacher sidecar的source provenance，不可直接填`real_score_manifest`。
- 新Stage-1 comparison提交未被Stage-2 trainer/init/checkpoint导入；共享`lora_utils`的一维FSDP扩展保留Stage-1二维分支，相关联合回归通过。comparison当前证明两路视频分别有效，但没有merged-vs-runtime-LoRA跨路数值等价指标，不能把`pass`解释为严格等价。
- 当前本地没有约10GB merged权重，无法复核真实文件hash/BF16/tensor schema；这些由内网生成teacher manifest与8×H100 init-only继续验证。本轮没有伪造H100结论。

### 已锁定目标
- 使用 Stage-1 checkpoint 3750 EMA merged causal generator 初始化 Stage-2，训练 4-step UniPC、24 个全新 latent、3×8 chunk 的 Self-Forcing DMD/DFD baseline。
- 永久保留原始首帧 global sink；baseline 非 sink 局部窗口 W16 包含当前 chunk8 与最近历史8，物理 self-KV 容量17。
- 每个新动作清除上一动作 local self-KV 与 cross-KV，保留 sink，并输出24个新 latent；25-latent VAE输入只用于解码，丢弃 sink 对应像素帧后输出96帧。
- real-score 为冻结 bidirectional TI2V teacher、标准 CFG5；fake-score 为 teacher 副本上的 core r64 LoRA；generator 为 fresh core r32 LoRA，训练/部署 CFG1。
- Score 输入显式为 `[initial_latent, future24]`；future 使用 video-global timestep，sink t=0、不加噪且不参与任何 loss/normalizer。
- Phase A 为24个 generator epochs纯DMD；Phase B为4 epochs混合DMD/DFD；严格5个成功fake-score更新后1个generator更新，global batch64。
- Generator noisy forward不提交 self-KV；每chunk仅最终 clean recache 在 no_grad 下提交并 detach，禁止跨chunk BPTT；Generator cache path关闭activation checkpoint，fake-score保留。
- H100使用完整预检、micro2×8×acc4优先、micro1×8×acc8回退；global batch32不是OOM修复。
- 仅使用当前 testsets、seeds 1–4正确推理并人工评估；暂不实现自动视觉质量指标。

### 文档新增要求
- 任务必须包含训练可视化实现，参考 Stage-1：分别可视化 generator loss、fake-score loss、梯度/非有限状态、F/G吞吐、完整5F→1G cycle吞吐、显存与各phase/branch统计。
- 可视化必须使用独立的成功 F/G update 计数作为横轴，不能用混合 loop step；Phase A/B、DMD/DFD、raw/EMA与断点恢复边界必须可辨识。
- 当前阶段只生成任务文档，不实现代码；待用户审阅后再逐步执行。
- 当前本地600-clips CSV实际只有1条fixture且没有action label；正式22/21/21 balanced sampler必须依赖内网可靠cache字段或冻结的600行action sidecar，禁止用行顺序或模糊聚类猜标签。
- Stage‑2建议新增隔离的config/trainer/model/rollout/data/checkpoint/metrics入口，只复用并最小扩展Wan、UniPC、LoRA/FSDP/JSONL底座，避免继续扩大legacy DMD条件分支并破坏Stage‑1。

### 已知 P0
- 现有I2V路径用sink覆盖24槽位的第0帧，实际仅23个新latent。
- noncausal wrapper取首帧t=0作为整段t，必须实现9750-token TI2V timestep adapter和动态seq_len。
- 当前generator/fake-score共享LoRA配置、LoRA resume不含optimizer/EMA/RNG、更新顺序不是严格F×5→G。
- 当前带梯度KV提交会污染持久cache的autograd图；现有generator activation checkpoint还会在backward读取已变异cache。
- 当前fake-score raw flow经过BF16 x0 round-trip，在低sigma放大量化误差。

### Stage-1 可视化复用结论
- `utils/jsonl_logger.py` 已提供 append-only JSONL、截断尾行容错、run lineage、resume 后 child 覆盖 parent stale suffix、attempt 单调编号、跨 rank step elapsed/straggler 和逻辑吞吐字段；Stage-2 应复用通用机制，但使用新 schema/version，不能把混合 F/G 时钟硬塞进 Stage-1 单 optimizer-step schema。
- `trainer/diffusion.py` 的 Stage-1 正式路径每个成功 update 写一条 `train_step`，non-finite 重试另写 `nonfinite_attempt`；记录 phase、epoch、loss、pre-clip grad、EMA、吞吐、显存和 straggler，checkpoint I/O 前后恢复 RNG。Stage-2 应保持这一可审计风格。
- `scripts/plot_stage1_training.py` 已实现 latest-lineage读取、raw+rolling mean、PNG/SVG无头渲染与 loss/throughput/time/optimization/memory 图；其中 phase marker 目前硬编码 Stage-1 step 300/480，Stage-2 必须改为由日志/config派生 A/B 边界。
- `tests/test_jsonl_training_plot.py` 已覆盖 lineage覆盖、截断尾行、logical workload和12个PNG/SVG产物；Stage-2需新增双时钟、5F→1G cycle、DMD/DFD branch、resume重复/覆盖及图产物测试。
- Stage-2 建议单一 JSONL 内使用 `fake_update`、`generator_update`、`cycle_summary`、`nonfinite_attempt` 等 record_type；横轴分别为 `completed_fake_updates`、`completed_generator_updates` 和 `completed_cycles`，禁止使用旧的混合 `step`。
- Generator 图至少包含 DMD/DFD surrogate loss、raw score-difference/denominator诊断、grad norm、LR、DMD/DFD branch与exit histogram；Fake-score图至少包含raw-flow MSE、grad norm、LR和score timestep分布。
- 吞吐必须按角色拆分：F rollout/score/backward/optimizer、G rollout/real CFG cond+uncond/fake score/backward/optimizer、完整5F→1G cycle；同时记录global samples/s、generated latent/s、DiT forward calls/s、step max/mean、straggler及allocated/reserved/free显存。
- 需要生成一份可人工查看的静态汇总（PNG/SVG，建议再加HTML索引），但不实现自动视觉质量指标；训练可视化与推理质量评估是两个独立范围。
- 当前Stage-1 artifact只有损坏的4500-step run尾部3条记录且无run lineage，不对应checkpoint3750，不能用于趋势判断或作为Stage-2 plot输入。
- Stage-1 producer/plotter已有字段漂移：`preclip_grad_norm`/`pre_clip_grad_norm`、memory bytes/GiB、嵌套ER路径不一致，phase marker还硬编码300/480；Stage-2必须用共享schema helper和producer→plotter集成测试阻止同类空图。
- Stage-2成功substep建议统一 `record_type=train_step` 加 `role`，并以 `logical_substep_id=cycle*6+substep` 做resume stale-partial覆盖；F/G角色step仍分别记录。
- 最终文档复核无P0；已明确score连续sigma与rollout UniPC是独立契约、Stage‑1 negative prompt精确hash、F/G独立sample stream、global-mean梯度等价及B1十点概率序列。
- 可视化终点和phase marker必须京eresolved config推导：A24纯DMD为G240/F1200，完整A24+B4为G280/F1400，partial/preflight仍可绘制但不可伪装complete。
- 压缩适配的child raw G/EMA均从parent EMA启动，F从parent raw启动；branch/control成对使用A8+B2或A12+B2。W24 inference-only只是OOD full-context reference，不是质量上界。
# 2026-08-18 Phase 22：OOM首始证据

- 异常不是初始化OOM，也不是fake/real score路径OOM；它发生在rollout调用generator第5B transformer block前，FSDP2为该block执行all-gather/unshard并尝试申请318 MiB。
- 失败瞬间单卡79.19 GiB中只剩273.06 MiB；同一rank进程占78.91 GiB，PyTorch已分配69.91 GiB，缓存池保留但未分配7.11 GiB。因此“只需要22.9G”不能代表该rollout时刻的真实峰值。
- 7.11 GiB reserved-but-unallocated说明allocator碎片/分段可能是直接触发条件，但69.91 GiB live allocated也证明工作集本身已经很高；不能只用`empty_cache()`或把22.9G当容量依据。
- 8卡FSDP2不是把8张80GB拼成640GB统一显存：三角色参数各自分片，但当前block会在每个rank本地all-gather，rollout activation、24帧的3个chunk输出图、self/cross-KV及allocator临时buffer仍需单卡容纳。Generator activation checkpoint因cache正确性明确关闭，micro2会把这些激活近似翻倍。
- OOM记录显示GPU总量减去当前进程占用仅约0.28 GiB，与报告的0.267 GiB free一致；没有证据支持另一个进程占了大块显存。rank3只是最先报告失败的rank。
- trainer每个logical substep先`reset_peak_memory_stats()`，成功后才采集/写入max memory；CUDA OOM不属于其可恢复nonfinite分支，会直接越过指标写入。因此22.9 GiB若来自最后一条JSONL，代表上一条成功F/no-grad子步，而非失败rollout；若来自`nvidia-smi`，则只是采样瞬间。
- 79.19 GiB的门禁上限分别为allocated约67.31 GiB、reserved约71.27 GiB、free至少8 GiB；当前69.91 GiB allocated、约77.02 GiB reserved、0.267 GiB free即使不抛OOM也必须拒绝micro2。
- 唯一首选回退是micro1×8×acc8，仍为global batch64；新config需要重新prepare绑定launch hash，并在全新smoke目录完整重跑C0/C1/C2。`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`可减少动态shape/多阶段分配碎片，但不能替代micro1。micro1仍失败才允许Generator grad-exit `save_on_cpu`候选。

# 2026-08-18 Phase 23：重复附件首始判断

- 新附件不是prepare/manifest/launch-hash异常，而是与Phase 22逐项相同的5B Generator FSDP2 OOM；说明prepare成功后仍启动了一个无法通过显存门禁的候选。
- 最可能的操作原因是只运行默认`bash run_stage2_h100.sh prepare/smoke`，或生成micro1配置后换了shell，导致`STAGE2_CONFIG`未继承，wrapper重新使用仓库默认micro2×acc4。必须在昂贵smoke前打印resolved microbatch/accumulation/global batch作为自证。
- 源码确认wrapper的`${STAGE2_CONFIG:-canonical}`和默认`smoke_micro2_acc4`正是上述回退；`ACTIVE_CONFIG`每次启动都从`STAGE2_CONFIG`重建，因此在另一个shell只导出`ACTIVE_CONFIG`也不够，必须导出`STAGE2_CONFIG`。
- `prepare=PASS`绑定的是当次resolved config的launch hash；切micro1后必须再次执行wrapper `prepare`。只改YAML后直接沿用旧prepare证据会被严格复验拒绝或造成操作者误判。

# 2026-08-18 Phase 24：EMA名字异常首始证据

- 精确字符串`EMA parameter names mismatch`只在portable EMA state validator中出现，不是optimizer或模型forward报错。异常意味着EMA状态里记录的local/global shape名字集合与当前模型提供的parameter_names集合不完全相等。
- 需要完整trace区分两条路径：首次G update后的EMA内部自检，或C1/C2从C0/C1 checkpoint加载rank-local EMA state；二者根因和修复边界不同。
- 代码路径已证明C0即可触发：`build_lora_shard_schema(wrapper.model)`保存内层PEFT raw names；`TrainableShardedEMA(self.model.generator)`遍历外层Stage2DiTRole并给每个名字增加`model.`；`save_stage2_checkpoint`在rank-local EMA clock audit处要求EMA names等于schema raw names。
- `utils/lora_utils.py`的FSDP2导出早已正确允许`current == schema`或`current.endswith('.'+schema)`，说明wrapper前缀是预期运行时现象；EMA没有复用这条schema-aware映射才是缺口。不能简单全局删除字符串`model.`，因为PEFT合法名字自身包含`base_model.model.`，粗暴replace会损坏语义并可能碰撞。
- 真实PEFT最小反例不依赖FSDP即可稳定复现：`base_model.model.block...`对`model.base_model.model.block...`，missing/extra正好是一一加前缀关系。FSDP prefix cleanup不会移除Stage2 role的业务字段`model.`，因此8卡行为相同。
- 当前简化EMA测试在同一对象上同时构造schema/EMA，Stage2 checkpoint fixture又手工令两套名字相同；没有覆盖外层Stage2DiTRole+内层PeftModel的真实组合，是本地回归全绿而H100 C0暴露问题的具体原因。

# 2026-08-18 Phase 25：统一参数命名契约

- 本轮目标不是放宽`EMA parameter names mismatch`校验，而是让Stage-2所有持久化状态都使用pre-FSDP LoRA schema的`raw_parameter_name`作为唯一规范名。
- 运行时对象允许存在Stage2业务wrapper和FSDP/checkpoint/compiler wrapper前缀，但到schema边界必须通过精确相等或唯一`.`后缀映射；任何0匹配或多匹配都必须在训练/保存前失败。
- 必须审计的消费者包括EMA、optimizer DCP、LoRA raw gather/load、checkpoint C0/C1/C2、resume和Stage-2 inference；不得假设修复EMA即可代表其他路径安全。
- 首轮全仓清单确认optimizer也跨越同一边界：`audit_stage2_lora_optimizer()`接受外层运行时FQN并只验证其可唯一映射到schema；DCP full state因此仍可能以外层FQN持久化。`validate_stage2_checkpoint()`默认又用optimizer FQN校验rank EMA名字。若只把EMA改为schema名，C0发布后的checkpoint复验仍会再次发生names mismatch。
- `stage2_fsdp2.audit_stage2_fsdp2_role()`、optimizer audit、checkpoint optimizer-schema mapping和LoRA gather各自复制了相似但不完全一致的后缀逻辑；必须收敛到同一公共resolver，消除无点分隔endswith、重复匹配与错误消息漂移。
- 统一矩阵：`LoraTensorSpec`的mapping key是canonical adapter key，`raw_parameter_name`是pre-FSDP inner PEFT FQN；训练runtime FQN可在其前增加`model.`及技术wrapper；DCP只在调用PyTorch API时需要runtime FQN，写盘前应规范为raw FQN、恢复前再严格映射回当前runtime FQN；generator EMA topology/shadow同样只写raw FQN。
- Stage-2 inference从`generator_ema.safetensors`读取canonical adapter key并在新建PEFT模型上strict load/merge，不依赖EMA/optimizer raw FQN，因此无需改变推理artifact格式；只需保证训练checkpoint生成的canonical adapter本身不回归。
- EMA兼容策略无需升级payload schema：新checkpoint所有topology/shadow key均为schema raw FQN；加载旧payload时先用当前expected schema做全量双射重命名，再执行原schema version 2的shape/mesh/shadow严格校验，既兼容旧外层前缀又不放宽内容门禁。
- optimizer兼容策略同理：任何传入DCP state先规范为schema raw FQN并验证moments/step/hyperparameters；仅在`set_optimizer_state_dict`前映射到当前module runtime FQN。写盘始终canonical，旧wrapper-prefixed checkpoint仍可resume。
- 启动期现在有两层防混版：hotfix隔离probe同时验证DMD callback/timing与parameter-name API；H100 wrapper在任何torchrun前再次验证resolver样例、`TrainableShardedEMA.expected_parameter_names`及optimizer双向转换函数。缺任一文件会在8卡模型构造前失败。
- 完整回归证据：Stage-2 694 passed；全tests 997 passed、2 subtests passed。唯一全工作树whitespace失败来自用户原有600clip metadata，不属于代码补丁且未触碰。
- 生产修复已发布到`longlive-cats/stage-2`提交`3551ed00667b82777a57805f62e2b0f47ac9adac`；远端branch ref已逐字核验。内网无需Git，只需同步最新版累计hotfix脚本并确认两个API PASS后，用全新smoke目录从C0开始。
