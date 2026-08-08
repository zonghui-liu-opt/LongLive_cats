# LongLive‑2.0：猫咪 TI2V 4‑Step Self‑Forcing DMD/DFD Stage‑2

> 状态：Batch 1 / Step 1 已完成；暂停在内网配置契约验证门禁，未开始 Step 2
>
> 最后确认日期：2026‑08‑08
>
> 目标硬件：内网单机 8×H100；本地只完成代码、单测、静态与轻量集成验证
> 本文是 Stage‑2 的实现规格。不得用未记录的新假设替换已确认决策。

## 0. 执行纪律

- [x] 开始前完整阅读本文、`git status --short`、Stage‑1 配置/训练/指标实现、现有 DMD trainer、causal/bidirectional Wan wrapper、训练/推理 pipeline 与相关测试。
- [x] 保留工作区所有既有修改和未跟踪数据；不得恢复、覆盖、删除或顺手整理无关文件。当前 `results/` 属于用户数据。
- [ ] 复用现有 LongLive‑2.0 模型、FSDP、LoRA、scheduler、checkpoint、Stage‑1 JSONL 和推理基础设施；可以为正确性拆分接口，但不得复制第二套大体相同的系统。
- [ ] 每次只实现一个最小可验证步骤；先运行该步骤测试并记录证据，再勾选并继续。
- [ ] 先复用现有代码，再做最小扩展, 不要重复造轮子.
- [ ] 所有 shape、frame count、LoRA target/count、hash、role、时钟、cache、phase、RNG、checkpoint 与非有限状态不一致均 fail-fast；禁止 silent fallback。
- [ ] 代码事实若与本文冲突且会改变研究方案，停止实现并恢复一问一答；仅事实性文件映射可更新本文后继续。
- [ ] 不在本地伪造 H100 profile、正式训练成功或视觉质量结论；不提交、不推送，除非用户另行授权。

## 1. 目标与完成结果

以 Stage‑1 step 3750 EMA causal generator 为起点，训练一个适合端侧部署的 Stage‑2 baseline：

```text
输入：原始单张首帧 + 当前动作 prompt
输出：24 个全新 latent = 3 chunks × 8 latent
采样：每 chunk 独立重置 4-step UniPC，shift=5
历史：永久原始首帧 global sink + 当前/最近局部窗口
新动作：保留原始 sink，清除上一动作其余 self-KV 和全部 cross-KV
部署 CFG：1（仅 conditional 单路）
```

训练使用 Self‑Forcing on-policy rollout、DMD 和短 Phase‑B DFD post-training。代码完成必须同时满足：

- [ ] Generator 始终生成24个新 latent；sink不占输出槽位，score输入严格为 `1+24=25`。
- [ ] 4-step训练 rollout 与部署使用同一 LongLive‑2.0 UniPC算子、时间表、chunk scheduler reset 和 clean recache语义。
- [ ] DMD/DFD/fake-score三条梯度路径与下文公式一致，sink严格不进loss。
- [ ] `5 fake-score updates → 1 generator update`、Phase A/B、EMA与checkpoint使用独立且可审计的成功更新时钟。
- [ ] 8×H100可在global batch64下通过完整预检，并从原子checkpoint精确恢复。
- [ ] JSONL与静态图能分别展示Generator loss、fake-score loss、角色/周期吞吐、时间分解、显存、梯度、phase和DMD/DFD分支。
- [ ] 使用现有 `testsets/`、seeds `1,2,3,4` 正确产生可人工查看的视频；本任务不自动判断视觉质量。

## 2. 范围与明确不做

### 2.1 本任务范围

- Stage‑2 cached-latent loader、600样本审计与balanced sampler。
- Generator、real-score、fake-score三角色独立初始化/LoRA/FSDP/optimizer。
- 24-new-latent rollout、global sink、local window、episode reset、4-step UniPC random exit。
- Phase A DMD、Phase B DMD/DFD、fake-score flow DSM、严格5F→1G trainer。
- LoRA-only EMA、完整原子resume、JSONL、PNG/SVG/HTML训练可视化。
- Stage‑2 baseline推理入口、现有testsets编排和技术正确性门禁。
- 一次实现通用 `chunk/window/sink/steps` 接口及CPU/tiny测试，供后续消融复用。
- 内网8×H100 preflight/profile/formal runbook。

### 2.2 明确不做

- 不做任何权重、KV或activation量化；量化由专门同事负责。
- 不实现DINO、VBench、CLIP、动作分类或其他自动视觉质量指标，不自动选择best checkpoint。
- 不做raw flow-MSE anchor；真实数据只通过DFD teacher-input replacement进入Generator更新。
- 不重编码现有600条视频，不生成离线teacher trajectory，不缓存rollout/noise，不做数据增强、裁剪、翻转、颜色抖动或跨视频拼接。
- 不训练real-score，不给real-score挂LoRA。
- baseline不启用sequence parallel、Generator cache-path activation checkpoint、CPU model offload、NVFP4或部署CFG>1。
- 不在本任务中正式训练全部压缩变体；只完成通用接口、压力测试入口和后续实验规格。
- 不把Stage‑1的23-new-latent语义带入Stage‑2。

## 3. 已核实上下文与P0

### 3.1 Stage‑1事实

- Stage‑1 causal TI2V按8帧chunk训练，LoRA为self-attention q/k/v/o与FFN 0/2，rank32，共180个Linear、57,016,320个可训练参数。
- Stage‑1正式训练loss是conditional-only，等效标准CFG1；用户使用的推理路径是标准CFG5。
- step3750人工结果：动作能力约恢复到bidirectional teacher水平，但后段出现颜色过饱和与毛发细节下降。
- 当前本地 `infer_stage1.sh` 的checkpoint列表并不包含3750；内网启动前必须记录真实3750命令、EMA merge来源与hash，不能只凭脚本名推断。
- `results/stage1_600cats_3750steps/metrics/stage1_train_metrics.jsonl` 只有损坏的尾部摘录：首行不是合法JSON、没有`run_start`，仅余4500-step run的4498–4500记录，且与checkpoint3750不对应。不得据此绘制趋势或宣称质量/收敛结论。

### 3.2 当前Stage‑2路径必须修复的P0

1. 当前I2V路径把initial latent覆盖进24输出的第0槽，实际只有23个新latent。
2. noncausal wrapper取 `timestep[:,0]`；首帧t=0时会把整个bidirectional score forward误喂成t=0。
3. score `seq_len=28160` 硬编码于旧32×44×80场景；本任务25×30×52/52×30应为9750 tokens。
4. 当前I2V score timestep/per-block normalizer属于all-causal路径，25帧还会触发8帧reshape错误；本任务必须video-global且per-video holistic。
5. 当前fake-score丢弃raw flow，再经BF16 `flow→x0→flow`；低sigma误差会被放大。
6. Generator与fake-score共享adapter rank/targets配置，且target扫描过宽、无严格角色审计。
7. 当前更新循环不是显式 `F1…F5→G` 状态机；checkpoint/resume只保存LoRA与单一step。
8. 带梯度的exit forward会把有 `CopyBackwards` 的K/V写入持久cache；之后no-grad覆盖不会清除旧grad_fn。
9. Generator activation checkpoint闭包引用可变KV；backward重算时cache已被后续chunk改变。
10. 当前CFG1推理仍可能分配pos/neg两套KV，浪费约一倍self-KV显存。

### 3.3 实现隔离边界

保留LongLive‑2.0现有Wan/UniPC/attention底座，但不要继续在legacy `trainer/distillation.py`、`model/dmd.py` 和 `pipeline/self_forcing_training.py` 中堆叠大量I2V特殊分支。优先新增隔离的Stage‑2入口：

```text
configs/train_i2v_stage2.yaml
trainer/stage2_distillation.py
model/stage2_dmd.py
pipeline/stage2_rollout.py
utils/stage2_config.py
utils/stage2_i2v_data.py
utils/stage2_sampler.py
utils/stage2_checkpoint.py
utils/stage2_metrics.py
```

通过现有 `train.py` 与 `trainer/model/pipeline` registry注册。只对 `utils/wan_5b_wrapper.py`、`wan_5b/modules/causal_model.py`、通用LoRA/FSDP/JSONL基础件做最小且有回归测试的扩展。现有 `configs/train_i2v_dmd.yaml` 的32帧、all-causal、共享r128与legacy CFG语义不得作为正式baseline配置。

## 4. 已锁定baseline

### 4.1 时序、window与输出

统一记法：

```text
C = chunk_frames
W = local_window_frames（不含global sink，但包含当前chunk）
H = history_frames = W - C
S = global_sink_frames
K = num_denoising_steps
physical_kv_capacity_frames = S + W
```

Baseline：

| 项目 | 锁定值 |
|---|---:|
| generated_episode_frames | 24 |
| C / chunks | 8 / 3 |
| W / H | 16 / 8 |
| S | 1（最初原始首帧） |
| self-KV容量 | 17 latent frames |
| K / solver / shift | 4 / UniPC / 5 |
| rollout/deploy CFG | 标准CFG1 |

- Chunk1看 `sink1 + current8`；chunk2/3最多看 `sink1 + history8 + current8`。
- 公开配置不得直接复用含义混乱的legacy `local_attn_size`。由 `S+W=17` 派生内部attention span与cache容量，并在每次启动打印/断言 `C/W/H/S/capacity`。
- sink先用一次clean `t=0` no-grad forward写入；之后从3个全新8帧noise chunk生成24个新latent。
- 同一chunk的UniPC noisy forward不提交self-KV；只有最终clean latent做一次no-grad recache并提交。
- 同一动作内部RoPE位置：sink=0，future=1..24。

### 4.2 新动作episode reset

新动作到来时：

```text
保留每层 K/V[0:sink_tokens]
清除或失效其余 local self-KV
global_end_index = sink_tokens
local_end_index  = sink_tokens
pinned_start     = -1
pinned_len       = 0
清空 cross-attention K/V，is_init=False
新动作 future positions 重新从1开始
```

- 不刷新、不替换最初原始首帧sink。
- 不保留上一动作的局部KV、prompt cross-cache、VAE decoder cache或UniPC scheduler状态。
- 每个新动作仍输出24个新latent；sink不占输出位置。

### 4.3 UniPC与random exit

- 每个8帧chunk创建全新的 `FlowUniPCMultistepScheduler`；训练/部署都不跨chunk复用multistep state。
- 4-step、shift5的实际运行时timestep必须在启动时从scheduler读取并断言为当前LongLive‑2.0契约 `[999,937,833,624] → 0`；禁止同时维护第二份近似时间表。
- random exit均匀覆盖0..3；同一个microbatch的全部ranks、样本和3个chunks使用同一exit。
- baseline使用optimizer-update内 `stratified_uniform`：
  - micro2×8、acc4：rank0独立RNG生成 `randperm([0,1,2,3])`；
  - micro1×8、acc8：两次独立randperm；
  - F与G使用不同RNG流和独立schedule。
- pipeline接收显式 `exit_step`；禁止内部二次抽样。保留 `iid_uniform` 作为消融接口。
- exit之前的UniPC forwards为no-grad；exit forward有梯度并以该时刻的raw clean `x0_pred` 作为这个chunk的 `x_hat`；此后不再继续剩余UniPC steps。clean recache为no-grad；不跨solver step或chunk BPTT。部署推理不抽exit，每chunk始终执行完整4 steps。

### 4.4 KV autograd边界

- 所有noisy/prefix/exit forward使用 `commit_self_kv=false`：可临时拼接detached history与当前chunk K/V做attention，但forward结束丢弃更新信息。
- 每chunk仅clean recache使用 `commit_self_kv=true`；提交后所有持久K/V必须 `detach_()`，并断言 `requires_grad=False`、`grad_fn is None`。
- Generator cache-enabled forward禁用activation checkpoint。fake-score启用标准bidirectional activation checkpoint；real-score全程no-grad。
- Generator CFG1训练/部署只分配一套conditional self-KV/cross-KV；不得无条件创建negative cache。Stage‑1 CFG5路径必须保持兼容。

## 5. 数据与conditioning

### 5.1 数据源与启动门禁

| 项目 | 锁定值 |
|---|---|
| metadata | `training_sets/metadata_600clips_480x832_buckets.csv` |
| 样本 | 600；3动作×200 |
| cache dtype/channels | BF16 / C=48 |
| latent空间 | 30×52或52×30 |
| cached video latent | 必须每条恰好25帧 |
| initial latent | 每条恰好1帧，来自显式input image缓存 |
| prompt | cached positive T5 embedding；negative embedding只离线构造/缓存一次 |

内网正式启动前扫描全部600条：

- 600/600记录存在、可读、shape/dtype/channel/orientation合法；动作恰好200/200/200。
- 所有video latent必须统一F=25；F=24或混合长度立即失败，不能补帧或重新编码。
- `initial_latent` 与 `video_latent[:,0]` 的差异写入manifest，但永远不以video0静默替代initial。
- `real_future = video_latent[:,1:25]`；必须正好24帧。
- cached prompt padding位置必须为0，因为Wan模型不消费prompt mask。
- manifest记录每条shape/hash、公共schema hash、数据/config/code版本；T5/VAE不进入正式训练进程。
- real-score CFG5的negative conditioning必须逐字节复用 `utils/config.py::DEFAULT_NEGATIVE_PROMPT`（当前UTF‑8 SHA256为 `ce96e0324e4b54ce4b6e867f669ca520952e1a34cc116543516b1897f0d3c47e`）。使用与positive cache相同的T5/tokenizer checkpoint、revision、cleaning、special-token与padding设置离线编码一次；manifest同时校验文本、模型、tokenizer、embedding tensor与padding hash，正式trainer不现场加载T5。

当前本地CSV只有header和1条fixture，且schema没有 `action_id`。正式balanced sampler不得用模糊文本聚类或假设“前200/后200”的行顺序。内网预处理必须提供一个冻结、可审计的action label来源：优先读取现有cache manifest中的可靠字段；若不存在，则生成并由操作者确认一个精确sidecar（建议 `video,action_id`，600行，3个受限枚举值）。sidecar/hash写入Stage‑2 manifest，恰好200/200/200才放行；缺失、重复、未知或规则无法唯一分类时停止训练。

### 5.2 Balanced sampler与epoch口径

- 每个F optimizer update与每个G optimizer update的effective global batch均为64。
- 每个global batch按动作组成22/21/21，并轮转多出的一个名额，长期等权。
- F与G各自拥有独立的balanced sample stream与RNG；两者都对每个动作的200条队列做确定性shuffle并按index构造。checkpoint分别保存F/G的sampler epoch/cursor/generator state，禁止让5倍F消费扰动G的generator-epoch数据语义。
- 一个 **generator epoch** 固定定义为 `ceil(600/64)=10` 个成功G updates，不以F update或总loop iteration计数。
- 每次F/G update使用自己的4或8个accumulation microbatches；不得在G/F间偷共享一份需要梯度的rollout图。
- DFD real sample必须与on-policy fake来自同一CSV行、同一prompt与initial latent；不得随机配同动作视频。

## 6. 三个模型角色与初始化

### 6.1 Generator

- immutable causal full base：Stage‑1 step3750 EMA LoRA严格merge后的完整BF16权重。
- merge manifest记录Stage‑1 base、raw/EMA adapter、step3750、target schema与SHA256；Stage‑1 trainer state/step/optimizer/RNG不得导入Stage‑2。
- fresh Stage‑2 Generator LoRA：

```yaml
rank: 32
alpha: 32
dropout: 0.0
targets:
  - '^blocks\.[0-9]+\.self_attn\.(q|k|v|o)$'
  - '^blocks\.[0-9]+\.ffn\.(0|2)$'
expected_target_modules: 180
expected_trainable_parameters: 57016320
expected_adapter_tensors: 360
```

### 6.2 Real-score

- 冻结的猫域bidirectional Wan2.2‑TI2V‑5B teacher；加载路径、转换来源和SHA256独立记录。
- 无LoRA、无optimizer、无EMA、所有参数 `requires_grad=False`。
- 标准CFG5：`uncond + 5*(cond-uncond)`。如果兼容旧DMD `cond+g*(cond-uncond)` 参数，内部legacy值应为4；新配置只暴露无歧义的 `score_real_cfg_scale: 5.0`。
- cond/uncond两路共享完全相同的图像sink、noised latent、timestep和noise，只替换positive/negative T5 context。

### 6.3 Fake-score

- 从real-score immutable full checkpoint精确复制，挂独立fresh LoRA。
- baseline targets与Generator相同，rank64/alpha64/dropout0：180 modules、114,032,640参数、360 adapter tensors。
- conditional-only，标准CFG1；不部署、不做EMA。
- 保留配置化升级矩阵，不自动升级：

| targets | rank | 参数 | 用途 |
|---|---:|---:|---|
| self+FFN | 64 | 114,032,640 | baseline |
| self+FFN | 128 | 228,065,280 | rank消融 |
| self+cross+FFN | 64 | 161,218,560 | target消融 |
| self+cross+FFN | 128 | 322,437,120 | 质量优先上界 |

只有baseline fake loss长期不降、明显欠拟合或升级小规模profile有证据时，才按 `core r64 → core r128 → broad r64/r128` 实验；600条数据本身不是直接把rank升到128的理由。

### 6.4 角色隔离门禁

- 配置使用 `adapter.generator` 与 `adapter.fake_score` 两个独立节点；full-name regex解析、确定性排序和fail-fast count审计复用Stage‑1方式。
- 启动manifest列出每个target full name、rank/alpha、参数/tensor计数。
- F update后只有fake-score LoRA改变；G update后只有Generator LoRA改变；real-score永不改变。
- `init_from_stage1` 与 `resume_stage2` 是两个互斥入口；只有带 `_SUCCESS` 的Stage‑2完整checkpoint能恢复训练状态。

## 7. Score输入、timestep与噪声

### 7.1 25帧TI2V pack

定义：

```text
x_hat       : Generator输出 [B,24,48,H,W]
x_real      : video_latent[:,1:25]
score_fake  : concat(initial_latent, x_hat)  -> [B,25,48,H,W]
score_real  : concat(initial_latent, x_real) -> [B,25,48,H,W]
```

- TI2V模型不接收额外 `y`、mask或CLIP通道；首帧只在x的temporal frame0出现一次。
- 新建显式pack helper；禁止复用“覆盖slot0”的Stage‑1 I2V helper。
- sentinel测试必须证明future第0与第23帧均存在、唯一且均可得到Generator梯度。

### 7.2 Video-global score timestep

- 每个样本、每次G/F update独立采 `r ~ UniformInteger[0,1000)`：

```text
u = r / 1000
t = 1000 * (5*u) / (1 + 4*u)
t = clamp(t, 20, 980)
frame_t = [0, t, ..., t]  # 1 sink + 24 future
```

- baseline `ts_schedule=false`、`ts_schedule_max=false`；score t与rollout exit相互独立。
- 两种orientation均为每帧390 patch tokens；动态 `seq_len=25×390=9750`。
- bidirectional TI2V adapter把frame_t展开为token_t `[B,9750]`：前390为0，后9360为同一future t；禁止取frame0 scalar。
- score noising/x0换算helper使用frame-level `[B,25]` 连续sigma；模型time embedding使用token-level `[B,9750]`，不得混用，也不得调用离散scheduler做nearest-timestep lookup。
- future24每元素iid `epsilon~N(0,I)`；sink noise=0并在noising后强制保持exact initial latent。
- F update与G update使用相同边际采样器但fresh `(t,epsilon)`；不同DP rank不广播score t/noise。
- DFD的real/fake分支共享同一个future t和同一张iid epsilon tensor；“同noise”不是在24帧重复同一帧noise。

## 8. Loss与梯度

设连续 `sigma=t/1000`，flow目标 `v*=epsilon-x0`，`x_t=(1-sigma)*x0+sigma*epsilon`，`x0_pred=x_t-sigma*v_pred`。score noising与x0换算必须复用这一个FP32连续sigma；不得再把20/980映射到scheduler最近离散sigma。所有score→x0、差分、normalizer和loss reduction使用FP32。这与rollout的UniPC scheduler是两个独立契约。

### 8.1 Generator DMD

1. on-policy 4-step random-exit rollout得到 `x_hat`。
2. 对 `score_fake=[sink,x_hat]` 的future加噪。
3. fake-score与real-score都评估同一个noised fake；real使用标准CFG5，fake为CFG1。
4. 仅future24：

```text
g_raw = x0_fake(noisy_fake) - x0_real_cfg5(noisy_fake)
denom_b = mean(abs(x_hat - x0_real_cfg5), dims=[F,C,H,W])
g = g_raw / clamp_min(denom_b, 1e-6)
L_G = 0.5 * MSE(x_hat, stopgrad(x_hat - g))
```

- denom是每个video一个scalar，排除sink；若未来启用SP，sum/count必须在对应SP组聚合。
- 不额外乘/除sigma、SNR或timestep weight。
- real/fake模型在G update都冻结；梯度只沿 `x_hat → Generator LoRA`。

### 8.2 Generator DFD

Phase B的DFD分支只替换real teacher输入：

```text
fake teacher input = noise([sink,x_hat], t, epsilon)
real teacher input = noise([sink,x_real], t, epsilon)  # same t/epsilon/condition
g_raw = x0_fake(noisy_fake) - x0_real_cfg5(noisy_real)
denom = mean(abs(x_hat - x0_real_cfg5(noisy_real)))
```

- 其余surrogate、mask、FP32与stop-gradient完全同DMD。
- DFD不是raw flow anchor，也不是real video的逐点MSE。
- fake-score始终评估fake、学习Generator分布；DFD时不得把real video送进fake-score update。

### 8.3 Fake-score flow DSM

1. 使用当前Generator做fresh no-grad random-exit rollout并detach。
2. fresh video-global t/epsilon加噪 `[sink,x_hat]`。
3. 直接取fake-score原生 `raw_flow_pred`。
4. 仅future24计算：

```text
target_flow = epsilon - x_hat
L_F = mean_fp32((raw_flow_pred - target_flow)^2)
```

- 无CFG、无real-score、无real video、无sigma/SNR额外权重。
- 禁止 `raw flow → BF16 x0 → flow` round-trip。sink仅作为clean conditioning：固定 `t=0/noise=0`并强制覆写为exact initial latent，不进入loss、normalizer或 `target_flow`。

### 8.4 数值门禁

- denom clamp `1e-6`；loss、grad、prediction、denom全部finite assert，不能用 `nan_to_num`掩盖发散。
- 日志记录denom min/mean/max、raw score-difference norm、x0 prediction norm与score timestep edge mass，便于区分branch概率与梯度尺度。
- 每个substep开始前保存数据cursor与该substep所有专用RNG快照。nonfinite attempt不执行optimizer/EMA、不推进F/G/phase/sampler/RNG committed cursor；恢复快照后用完全相同的batch/exit/branch/t/noise重试，默认最多2次后fail-fast。

## 9. 训练状态机与超参数

### 9.1 严格更新顺序

唯一合法cycle：

```text
F1 → F2 → F3 → F4 → F5 → G1 → EMA → cycle committed
```

- 每个F/G optimizer update都各自完成global batch64与完整gradient accumulation。
- G必须看见5个已经完成的F updates；不允许同一iteration先算G再补F。
- phase、epoch、milestone、DFD概率和EMA全部由成功 `completed_generator_updates` 驱动。
- checkpoint只允许在完整cycle边界、grad accumulation cursor=0、next_substep=F1时写入。

### 9.2 Phase预算

| phase | generator epochs | G updates | F updates | real-data mode |
|---|---:|---:|---:|---|
| A | 24 | 240 | 1200 | DMD only |
| B | 4 | 40 | 200 | DMD/DFD mixture |
| total | 28 | 280 | 1400 | — |

- `phase_b_epochs=0` 表示纯DMD并在Phase A结束。
- Phase B第1个epoch的第 `j=0..9` 个成功G update使用 `p_DFD(j)=0.25*j/9`，端点精确为0与0.25；B2–B4固定0.25。resume从已完成G计数派生j，不保存/信任可漂移的隐式位置。
- 每个G update只选DMD或DFD之一，branch flag由rank0专用RNG生成并广播；不是同时加两个loss。
- 对照：从同一个A24 checkpoint分叉，B0继续纯DMD 4 epochs，B1执行上述DFD；不能用A24终点直接对比A24+B4。
- milestones：A8/A12/A16/A20/A24与B1/B2/B3/B4。

### 9.3 Optimizer与EMA

```text
Generator AdamW: lr=2e-6
Fake-score AdamW: lr=4e-7
both: betas=(0.0,0.999), eps=1e-8, weight_decay=0, max_grad_norm=10
schedule: constant, no warmup, no decay, A/B不重置
```

- LoRA master、grad reduction与optimizer moments为FP32；forward/activation为BF16。
- G/F clip与step分别执行，记录clip前global grad norm。
- Generator LoRA-only EMA：CPU FP32、decay0.99；completed G40后以当前raw参数初始化，从下一次成功G update开始decay更新。
- baseline推理/人工检查默认使用EMA；raw与EMA均保存。fake-score无EMA。

### 9.4 8×H100并行

- 单机world size8，single-node FULL_SHARD（现有 `hybrid_full` 若单机等价必须由runtime manifest明确证明）。
- baseline无SP；T5/VAE不驻留训练进程；real/fake/G三模型按role独立FSDP wrap。
- 优先每卡microbatch2、acc4；fallback每卡microbatch1、acc8；两者global batch均64且 `accumulation % K == 0`。
- gradient accumulation的loss scale必须使FSDP归约后梯度严格等于整个global batch上的 `total_numerator/total_count`；baseline等数microbatch下等价于每个local-mean loss除以accumulation steps。使用FSDP no-sync前，必须通过micro2×acc4、micro1×acc8、sync/no-sync与单一global-batch reference的loss及参数梯度一致性测试；禁止把多个microbatch activation同时保留。
- Generator无activation checkpoint；fake-score有；real-score全程no-grad并按cond/uncond顺序forward，避免同时保留activation。

## 10. 完整checkpoint与resume

### 10.1 保存边界与频率

- 只在完整 `5F→G→EMA` cycle后保存；崩溃在cycle中则丢弃partial cycle，从上一个完整checkpoint重放。
- 每个generator epoch（10 G/50 F）保存rolling resume checkpoint，保留最近2个；milestone永久保留。
- 保存前capture RNG，保存后restore RNG，checkpoint I/O不得改变下一次采样。

### 10.2 必须保存

- Generator raw LoRA、fake-score raw LoRA、Generator EMA LoRA。
- 两个AdamW完整FP32 state与role/schema审计。
- `completed_g`、`completed_f`、cycle、next_substep、successful attempts、nonfinite attempts。
- phase/epoch/DFD schedule版本与派生值；校验而非信任冗余字段。
- F/G两套balanced sampler的epoch/cursor、DataLoader generator state。
- 每rank Python/NumPy/torch CPU/current CUDA RNG；rank0独立G-exit/F-exit/DFD-branch RNG。
- resolved config/hash、代码版本、world/FSDP/microbatch/accum topology。
- G/real/fake immutable base hash、role LoRA schema、cache manifest与negative prompt/embedding hash。

### 10.3 原子性与恢复顺序

- 临时目录写入 → 全rank文件/hash校验 → manifest → 原子rename → `_SUCCESS`；`latest`只识别完整marker。
- 恢复前断言 `completed_f=5*completed_g`、`next_substep=F1`、无pending grads/branch/batch/KV、Adam step与角色计数一致、EMA last step一致。
- 恢复顺序：验证manifest/hash → 构造三base/LoRA → load raw adapters → FSDP → optimizer → EMA → counters/sampler → 最后恢复每rank RNG → 构造iterator并从F1继续。
- training入口捕获异常后必须重新抛出并返回非0；禁止打印后把失败作业标成成功。

## 11. Stage‑2 JSONL、loss与吞吐可视化

### 11.1 设计原则

- 参考并复用 `utils/jsonl_logger.py`、`trainer/diffusion.py` Stage‑1正式JSONL路径、`scripts/plot_stage1_training.py` 和 `tests/test_jsonl_training_plot.py`。
- JSONL是唯一权威持久metric source；baseline `disable_wandb=true`。如未来镜像到W&B，不得改变JSONL或时钟语义。
- 使用新schema，例如 `longlive_stage2_metrics/v1`；优先参数化/泛化现有logger与lineage parser，不复制第二套JSONL parser。保持严格JSON（`allow_nan=False`）、append+flush/fsync、仅末尾截断容错、run lineage与child覆盖stale suffix。
- 禁止用一个混合 `step` 同时表示F/G。所有图和记录显式携带：
  - `completed_fake_updates`
  - `completed_generator_updates`
  - `completed_cycles`
  - `cycle_substep` (`F1..F5|G`)
- 每个成功optimizer substep写 `record_type=train_step`、`role=fake_score|generator`，并使用稳定的 `logical_substep_id = cycle_index*6 + substep_index`。resume重放未完成cycle时，child按该id覆盖parent stale partial记录。
- `logging.jsonl_every_steps` 必须固定/断言为1，或删除该死配置；不得配置一个实际不生效的抽样间隔。

### 11.2 Record types与字段

至少实现：

1. `run_start`：role hashes、resolved config、world/topology、phase边界、workload定义。
2. `train_step, role=fake_score`：
   - raw-flow MSE、numerator/count、preclip grad norm、LR；
   - score t min/mean/max/hist/edge mass、exit histogram；
   - data wait、H2D、rollout、score、backward、clip+optimizer时间；
   - samples/s、generated latent/s、实际Generator DiT calls/tokens/s、fake-score tokens/s；
   - allocated/reserved/free显存与跨rankstraggler。
3. `train_step, role=generator`：
   - DMD或DFD branch、surrogate loss、denom与raw score-difference统计、grad norm、LR；
   - data wait、H2D、rollout、fake score、real cond、real uncond、surrogate backward、clip+optimizer、EMA时间；
   - exit/timestep分布、role throughput、显存、straggler。
4. `cycle_summary`：5个F加1个G的总时间、各role时间占比、F/G samples/s、cycle role-samples/s、成功/重试计数。
5. `nonfinite_attempt`：role/substep、batch identity、reason、耗时与显存；不伪装成成功update。
6. 可选 `checkpoint_event`：checkpoint step、耗时、大小、`_SUCCESS`与retention动作；不混入训练吞吐分母。

每条loss同时记录detached numerator/count；在所有ranks与全部accumulation microbatches上先SUM，再求global mean，禁止平均rank-local means。字段名由producer与plotter共用同一schema常量/typed helper，避免Stage‑1已有的 `preclip_grad_norm` vs `pre_clip_grad_norm`、bytes vs GiB等漂移。

吞吐从实际shape、patch size、batch、exit schedule与forward count动态推导，不得硬编整段token总量，也不能把DP/FSDP shard重复计数。动态记录各模型角色的实际 `forward_calls` 与 `logical_query_tokens`；score compute包含25帧sink token，loss count只含future24。计时在第一次取本update数据前reset peak、CUDA synchronize并启动，完成clip/optimizer/EMA后再次同步；使用跨rank MAX作为wall-clock分母，同时记录mean与straggler。data wait/H2D与其他分项必须在误差容限内闭合到update wall time；JSONL、plot、checkpoint、GC和console不计入训练update吞吐。

H100 preflight/dry-run记录必须有显式flag，plotter默认排除，避免把warmup或强制DFD诊断混入正式曲线。

### 11.3 静态图与汇总页

新增Stage‑2 plot CLI（建议 `scripts/plot_stage2_training.py`），读取latest lineage，输出raw低alpha曲线+rolling mean的PNG和SVG，至少包含：

- `generator_loss`：DMD/DFD着色，surrogate、denom、score-difference。
- `fake_score_loss`：flow MSE与rolling mean。
- `optimization`：G/F LR与grad norm，使用各自update横轴。
- `generator_throughput`：G update seconds、samples/s、generated latent/s和各模型logical query tokens/s。
- `fake_score_throughput`：F update seconds、samples/s、generated latent/s和Generator/fake-score query tokens/s。
- `cycle_throughput`：完整5F→1G wall time与role时间占比。
- `time_breakdown`：data wait、H2D、rollout、real CFG两路、fake score、backward、optimizer/EMA，并展示分项与wall-time闭合误差。
- `memory_straggler`：allocated/reserved/free GiB与max/mean ratio。
- `timestep_exit_phase`：score t直方图、exit覆盖、DMD/DFD累计次数及Phase A/B边界。

Phase marker与预期终点全部从run metadata/resolved config推导：完整baseline为G240/F1200进入B、G280/F1400结束；`phase_b_epochs=0` 的A24纯DMD run在G240/F1200结束且不画Phase B marker；partial/interrupted run仍可绘制已完成区间。禁止复制Stage‑1 plotter中硬编码的300/480。再生成一个轻量静态HTML索引，嵌入/链接全部PNG/SVG并显示run/config/hash摘要；这不是视觉质量dashboard。

### 11.4 可视化验收

- resume lineage有parent stale suffix时，child相同role/update记录覆盖曲线但原始JSONL不删除。
- F/G横轴严格单调；每个cycle恰好5个F和1个G。只在run标记complete时断言终点等于resolved config（完整baseline 1400/280，A24纯DMD 1200/240）；partial/preflight不伪装成完训。
- producer→plotter契约测试必须使用真实producer schema生成fixture，不能手写一套只迎合plotter的字段。Generator/Fake loss与两张角色吞吐图缺少必需字段时应fail-fast；只有明确可选的诊断panel可以显示“No matching metrics”，且绝不能把合法0值当成missing。
- Phase、DMD/DFD和nonfinite标记位置正确；plot不依赖CUDA或模型权重。

## 12. Stage‑2 baseline推理

### 12.1 单动作

- 使用 `testsets/metadata_6cases_480x832.csv`、seeds1–4、Generator EMA、CFG1。
- 每样本从原始image latent写入永久sink，再生成24个新latent。
- VAE输入 `[sink1,new24]` 共25 latent，得到97 pixel frames；丢弃pixel frame0，输出96个全新pixel frames（4s@24fps）。

### 12.2 两动作

- 复用 `testsets/metadata_8cases_two_actions_continuation_480x832_253frames.csv` 中image/action A/action B/order；忽略旧HOLD/soft-reanchor生成语义。
- A episode：保留原始sink，生成A的24新latent并独立decode/drop sink frame。
- 切换prompt：保留原始sink KV，清其余self-KV和cross-KV；B重新生成24新latent并独立decode/drop sink frame。
- 重置VAE cache后解码B；最终拼接A96+B96=192 pixel frames/8s，不插入重复source frame或旧动作帧。

### 12.3 技术门禁与产物

- 只检查shape、latent/pixel frame count、fps、resolution、seed/config/hash、cache reset/index、finite与输出文件完整性；不实现自动质量指标。
- CFG1只允许一套conditional cache；trace记录每chunk实际UniPC timestep、exit/step count、KV容量17、sink hash、prompt hash与reset事件。
- 生成可人工播放的静态索引/manifest；用户观察动作、ID、后段颜色、毛发细节和跨动作污染。

## 13. H100启动门禁与正式训练

### 13.1 三cycle临时预检

每个候选batch配置使用临时输出、结束后从原始初始化重新正式训练：

1. C0：完整 `5F→1G` warmup/allocator，cycle边界写checkpoint，并保存下一状态探针。
2. 重启C1：验证resume状态/RNG/sampler/next=F1，跑纯DMD cycle并profile。
3. C2：强制一次DFD G branch，覆盖Phase B路径；结果丢弃。

全程使用真实global batch64、stratified exits和真实三模型角色。每role记录forward/backward/optimizer/EMA、显存与跨rank时间。

### 13.2 microbatch接受门槛

micro2×8、acc4仅在以下全部满足时放行：

- 所有路径无OOM、NaN/Inf、梯度角色污染或cache invariant错误；
- 最坏 `max_allocated ≤ 85% total`、`max_reserved ≤ 90% total`；
- NVML最小空闲 `≥ max(8 GiB, 10% total)`；
- 重复cycle结束live allocated增长≤1 GiB，无单调泄漏；
- 跨rank step time `max/mean ≤ 1.15`。

否则切换micro1×8、acc8并重跑全部3 cycles。micro1仍失败时，只允许先试Generator grad-exit saved-tensor CPU offload并重profile；不得重新打开不安全activation checkpoint。仍失败则停止并报告，不能静默改window/chunk/score帧/CFG。global batch32不降低micro1峰值，不是OOM修复。

### 13.3 torch.compile

- eager correctness是准线。compile只能作为profile候选，使用无cudagraph的安全模式。
- 必须先通过forward/loss/grad/cache parity；只有完整cycle稳定且有明确吞吐收益时才用于正式训练。
- compile失败、graph break或显存越界时回到eager，不改变算法配置，并记录manifest。

## 14. 后续压缩与sink消融接口

### 14.1 实验拓扑

1. baseline：`C8/W16/H8/S1/K4`，capacity17。
2. 同一baseline EMA inference-only window压力测试：
   - `C8/W24/H16`，capacity25（full-context OOD stress/reference，不得称为质量上界）；
   - `C8/W16/H8`；
   - `C8/W8/H0`，capacity9（极限下界）。
3. 公平chunk消融：`C4/W12/H8/S1/K4`，capacity13；保持历史8不变。
4. 联合压缩：`C4/W8/H4/S1/K4`，capacity9。
5. 在选定C/W后训练专用K2；必须重建原生2-step、shift5 UniPC schedule，不能截断4步轨迹。K2 stratified exit均衡覆盖0/1。

端侧DiT调用数包含sink preload和每chunk clean recache：

```text
C8,K4: 1 + 3*(4+1) = 16
C4,K4: 1 + 6*(4+1) = 31
C4,K2: 1 + 6*(2+1) = 19
```

因此C4主要降低单次峰值而非总调用开销。

### 14.2 部署适配与科学对照

- 部署链：`C8/W16/K4 → C4/W12/K4 → C4/W8/K4 → C4/W8/K2`。
- 每阶段热启动规则唯一：child raw G ← parent EMA G，child EMA shadow ← 同一parent EMA并在child update0即有效、从第一个成功G开始decay；child F raw ← parent F raw。rank不变，optimizers/counters/sampler/RNG全部重置；不套用baseline的G40 EMA冷启动。
- 每个压缩branch与同parent、不压缩matched control成对训练：先同跑A8；若需延长，两者一起延至A12；然后两者都跑B2。A8+B2固定为100 G/500 F，A12+B2固定为140 G/700 F；B1仍为0→0.25 ramp，B2固定0.25。单阶段上限A12+B2，禁止无限续训或只延长掉点branch。
- 严格科学对照只对最终候选执行：共同Stage‑1初始化、fresh G r32/F r64、完整A24+B4与相同数据顺序。

### 14.3 多帧global sink

- S表示总永久sink帧数并包含原始首帧。
- S4/S8的额外S−1帧来自episode1最早生成的clean latents。在它们的clean recache后立即snapshot对应的逐层self-K/V slice（同时保存clean latent/hash用于trace），detach并断言无grad；episode2直接恢复这些KV而不在新prompt下重算，只清空cross-KV与非sink local self-KV。它们只从episode2启用，因此episode1完全一致。
- episode2保留这些sink、清其余local/cross KV，输出仍为24新latent。
- sink位置0..S−1，episode2 future位置S..S+23，避免RoPE重叠；capacity为S+W。
- 首轮仅做inference-only，观察ID收益与动作A姿态泄漏；真实视频额外帧或隐藏warmup不是部署输入。
- baseline仍为S1/future1..24。

## 15. 有序实现计划

### Step 1：建立修改前基线与规格测试骨架

- [x] **结果**：明确当前DMD/Stage‑1相关测试基线、工作区保护范围和所有新配置字段。
- **主要区域**：新增 `configs/train_i2v_stage2.yaml`、`utils/stage2_config.py` 与Stage‑2 test骨架；只读对照现有DMD/pipeline/wrapper。
- **验证**：运行相关现有测试；新增只描述baseline config/计数/公式的失败测试，不改production。
- **完成证据**：Stage‑2配置正反契约103项与相关Stage‑1/DMD回归64项合并为167 passed；Black、Ruff、py_compile、CLI与whitespace检查通过。path-independent contract hash为`aa4d7be1e05c846df14cee5417a298afe668429f41faa671f3021754a5616c00`。
- **暂停边界**：本步未接`train.py`/registry，未加载CUDA、模型、权重或600-cache；UniPC测试仅characterize仓库scheduler。内网确认raw配置解析、contract hash和本测试集后，才可开始Step 2。

### Step 2：角色配置、初始化manifest与独立LoRA

- [ ] **结果**：G/real/F独立checkpoint与adapter schema；strict target/count/hash审计；init与resume分离。
- **主要区域**：`model/stage2_dmd.py`、角色初始化helper、通用LoRA/FSDP基础件；不要把新角色语义塞回legacy trainer。
- **验证**：tiny/CPU模型断言G180/r32、F180/r64、real0 trainable；错误role/rank/key/hash全部失败。

### Step 3：Stage‑2 cache preflight与balanced loader

- [ ] **结果**：600条F25 cache gate、explicit initial、real future1:25、prompt/action/hash manifest、精确Stage‑1 negative conditioning、F/G独立deterministic 22/21/21 sampler。
- **主要区域**：`utils/stage2_i2v_data.py`、`utils/stage2_sampler.py`、`scripts/audit_stage2_i2v_cache.py` 与tests；不改Stage‑1 loader语义。
- **验证**：F25正例；F24/mixed/C/dtype/spatial/action-count/hash/padding/negative-text-or-encoder反例；F/G sampler独立性与resume cursor重复性。

### Step 4：显式1+24 pack与TI2V token timestep adapter

- [ ] **结果**：bidirectional score接收25帧、dynamic seq_len9750、token t `[0×390,t×9360]`。
- **主要区域**：`utils/wan_5b_wrapper.py`、Stage‑2 I2V conditioning helper与 `model/stage2_dmd.py`；legacy `model/dmd.py` 仅作对照和回归保护。
- **验证**：sentinel首尾future/梯度、spy model input、uniform-time parity、两orientation、无legacy8-frame reshape。

### Step 5：24-new-latent rollout、W16/S1与episode reset

- [ ] **结果**：sink预写+3×8新latent；derived capacity17；prompt切换只保留sink；CFG1单cache。
- **主要区域**：`pipeline/stage2_rollout.py`，对causal model/wrapper增加最小cache接口；Stage‑1/legacy pipeline保持回归。
- **验证**：tiny cache逐chunk可见帧、indices/RoPE、24输出、reset后无旧KV、Stage‑1旧推理回归。

### Step 6：只读noisy KV、clean-only commit与安全autograd

- [ ] **结果**：显式discard/commit接口；持久cache永远detached；Generator cache checkpoint关闭。
- **主要区域**：Wan causal model/wrapper、`pipeline/stage2_rollout.py` 与Stage‑2 inference session；legacy pipelines只接受通用底层扩展并必须回归。
- **验证**：`CopyBackwards`回归、跨chunkgrad隔离、同chunk多UniPC step不append、clean recache唯一commit。

### Step 7：4-step UniPC与stratified random exit

- [ ] **结果**：显式exit schedule、F/G独立RNG、acc4/acc8覆盖、scheduler timetable单一真相。
- **主要区域**：`pipeline/stage2_rollout.py`、`trainer/stage2_distillation.py` 的RNG/state helpers；legacy `pipeline/self_forcing_training.py` 仅作对照和回归保护。
- **验证**：每update exit histogram、跨rankbroadcast、resume、iid消融、K2通用接口。

### Step 8：DMD、DFD与fake raw-flow loss

- [ ] **结果**：严格实现第8节公式、video-global t/noise、CFG5 real与FP32 holistic normalizer。
- **主要区域**：`model/stage2_dmd.py`、Stage‑2 scheduler/loss helpers与通用wrapper扩展；legacy `model/dmd.py` 不承载新算法分支。
- **验证**：解析式tiny tensors、连续sigma在t=20/980边界的noising/x0一致性、DFD shared-noise恒等式、sink mask、无sigma额外权重、raw-flow路径、real/fake参数无grad。

### Step 9：严格5F→1G trainer、phase、EMA与nonfinite

- [ ] **结果**：双optimizer状态机、A24/B4、DFD branch、global64 accumulation、G40 EMA、成功时钟。
- **主要区域**：`trainer/stage2_distillation.py`、Stage‑2 config/schedule/sampler helpers。
- **验证**：调用顺序、参数变化归属、240/40/1200/200计数、B1精确10点概率序列与resume、B=0、matched control、nonfinite恢复batch/全部RNG且不推进；micro2×acc4、micro1×acc8、sync/no-sync与reference global-batch梯度一致。

### Step 10：原子checkpoint与精确resume

- [ ] **结果**：完整LoRA/optimizer/EMA/counter/sampler/RNG/hash状态；cycle-boundary原子保存。
- **主要区域**：新增Stage‑2 checkpoint helper、trainer integration与tests。
- **验证**：save/restart下一batch/exit/DFD/t/noise一致；stale/partial/hash/topology/role错误全部失败；异常返回非0。

### Step 11：Stage‑2 JSONL与训练可视化

- [ ] **结果**：F/G/cycle独立记录；Generator/Fake loss、角色/周期吞吐、时间/显存/phase图和HTML。
- **主要区域**：复用/兼容扩展 `utils/jsonl_logger.py`；新增Stage‑2 metrics helper、`scripts/plot_stage2_training.py`、tests。
- **验证**：synthetic lineage/nonfinite/resume fixture；全部PNG/SVG/HTML存在且字段/横轴/phase marker正确；Stage‑1 plot tests不变。

### Step 12：baseline推理、testsets与技术trace

- [ ] **结果**：单动作96帧、两动作192帧；seeds1–4；CFG1单cache；人工查看索引。
- **主要区域**：Stage‑2 inference config/runner/shell、现有inference helpers/testset loader。
- **验证**：无模型tiny orchestration测试、frame/cache/reset/hash/文件集合门禁；旧Stage‑1入口回归。

### Step 13：通用压缩/sink接口

- [ ] **结果**：C/W/H/S/K通用配置与capacity断言；C4、K2、episode1-prefix multi-sink可走同一代码路径。
- **主要区域**：config resolver、cache allocator、rollout/inference session、tests。
- **验证**：表中W24/W16/W8、C4W12/C4W8、K2、S4/S8的shape/index/call-count测试；不正式训练变体。

### Step 14：全量本地验收与内网H100 runbook

- [ ] **结果**：相关回归全绿；提供cache scan、preflight、resume、formal train、plot、baseline inference精确命令。
- **主要区域**：tests、shell、中文runbook、本文状态。
- **验证**：`pytest -q tests/test_stage2_*.py`、仓库正式范围 `pytest -q tests`、shell `bash -n`、CLI `--help`、py_compile/lint、`git diff --check`；不要用会额外收集 `fouroversix/` 可选环境测试的无范围根目录pytest冒充正式回归，不得伪造H100结果。

### Step 15：用户在内网执行H100预检和正式训练

- [ ] **结果**：600 cache gate、3-cycle profile、选定microbatch、正式A/B训练、图表与推理产物。
- **验证**：保存真实manifest/JSONL/plots/checkpoints/trace；用户人工检查视频并决定后续压缩实验。

## 16. 全任务验收标准

### 16.1 本地代码验收

- [ ] 24个新latent从G0到G23全部存在且可反传；score pack严格25帧。
- [ ] bidirectional score收到9750-token mixed timestep，不会整段t0。
- [ ] W16/S1真实容量17；chunk2/3为sink1+history8+current8。
- [ ] noisy KV不提交，clean唯一commit且persistent K/V无autograd；无跨chunkBPTT。
- [ ] UniPC训练/部署一致，random exit每update分层覆盖。
- [ ] DMD/DFD/fake-flow公式、CFG、noise/t共享与mask全部有解析测试。
- [ ] G/F role LoRA计数、梯度、optimizer与checkpoint严格隔离。
- [ ] 状态机恰为5F→1G；A/B、EMA与checkpoint只由成功G时钟推进。
- [ ] resume恢复下一数据、exit、DFD branch、score t/noise和参数状态。
- [ ] Stage‑2 JSONL/plot覆盖loss、吞吐、时间、显存、phase；Stage‑1 logger/plot回归不变。
- [ ] baseline inference输出96/192帧并正确reset；旧Stage‑1推理不回归。

### 16.2 H100验收

- [ ] 600条cache manifest通过；三模型base/LoRA/hash与negative prompt一致。
- [ ] C0/C1/C2全部通过，resume边界与DFD路径真实运行。
- [ ] 正式microbatch满足显存余量、无泄漏、straggler与finite门槛。
- [ ] 正式训练最终计数G=280、F=1400；checkpoint与JSONL lineage完整。
- [ ] plot CLI生成全部PNG/SVG/HTML；曲线可区分G/F、A/B和DMD/DFD。
- [ ] 当前testsets、seeds1–4推理全部通过技术门禁；用户完成视觉判断。

## 17. 风险与显式延期

- 600条数据可能令fake-score过拟合或DMD mode-seeking；baseline先用core r64和短DFD，升级r128/broad必须有证据。
- Stage‑1 50-step质量不保证4-step rollout初始即稳定；从Stage‑1初始化的baseline与严格科学对照只能在A24后启用DFD，不能从scratch开启。只有从已完成baseline A24+B4的parent热启动的部署压缩适配，才按第14.2节允许在child A8/A12后进入B2。
- Generator无activation checkpoint可能使micro2不满足余量；按H100门禁降micro1或saved-tensor CPU offload，不改变算法。
- bidirectional teacher实际内网训练provenance尚需启动前确认；若不是标准video-global TI2V flow contract，必须暂停而非静默改成per-block。
- 人工测试集规模小、两动作续写只覆盖有限动作组合；本任务不声称统计显著性。
- W8/C4/K2与S4/S8均可能掉点；inference-only只能作为压力测试，正式结论需本文定义的适配/对照。
- baseline BF16 self-KV容量17不是8帧量化block的整数倍。量化交接必须采用独立1帧sink区+16帧ring区或支持尾块；本任务不实现量化。
- S4/S8会携带episode1动作姿态并增加KV/RoPE范围；只作为后续研究上界。

## 18. 参考

- Self‑Forcing: <https://arxiv.org/html/2506.08009>
- DMD2: <https://arxiv.org/html/2405.14867>
- Data‑Forcing Distillation: <https://arxiv.org/html/2606.18478>
- DFD Self‑Forcing reference code: <https://github.com/csy2077/DFD-self-forcing>
- 本仓库Stage‑1规格：`TASK-stage1.md`
- Stage‑1指标实现：`utils/jsonl_logger.py`、`scripts/plot_stage1_training.py`
