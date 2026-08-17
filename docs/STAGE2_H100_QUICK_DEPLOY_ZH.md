# Stage-2 训练前：只做 6 个检查

推荐直接使用新的单一入口；它会逐步说明作用并只在严格复验后打印 PASS：

```bash
bash run_stage2_h100.sh help
```

通过后继续运行脚本显示的下一条命令；复杂故障才查看[完整手册](STAGE2_H100_TRAINING_INFERENCE_RUNBOOK_ZH.md)。下面保留旧的手工 prepare 说明。

手工拷贝后若报告 API mismatch 或 timing closure error，只同步 `scripts/apply_stage2_innernet_hotfix.py` 并运行 `python scripts/apply_stage2_innernet_hotfix.py --project-root "$PWD"`；看到 `STAGE2_DMD_RUNTIME_API=PASS` 与 `STAGE2_INNERNET_HOTFIX=PATCHED`（或 `ALREADY_APPLIED`）后重跑 smoke。脚本自动备份、多文件事务写回且不需要 Git。

先准备一个 600 行的动作标签文件，表头必须是：

```csv
video,action_id
```

要求正好 3 类受限动作：`head_tilt_and_wink=198`、`jump=202`、
`play_with_a_cat_wand=200`。

然后只运行：

```bash
cd /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0

export ACTION_SIDECAR_600=/你的绝对路径/action_labels_600.csv
export ATTEST_STAGE2_TEACHER=1
# 推荐使用独立工作目录，便于容量管理、归档和恢复。
export STAGE2_WORK_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/stage2_runs/LongLive-2.0_stage2_new

bash prepare_stage2.sh
```

如果服务器文件位置和脚本默认值不同，再覆盖对应路径；相同就不用设置：

```bash
export TEACHER_CKPT=/双向合并权重/ckpts
export TEACHER_PROVENANCE_RECORD=/双向合并权重/merge_manifest.json
export STAGE1_BASE=/checkpoints/stage1/converted_causal_base.pt
export STAGE1_CKPT=/stage1训练目录/checkpoint_model_003075
export METADATA_600=/metadata_600clips_480x832_buckets.csv
export STAGE1_CACHE_MANIFEST=/stage1缓存/cache_manifest.json
```

最后看到下面 7 行，就说明训练前 6 项检查全部通过：

```text
CHECK_1_TEACHER_PASS
CHECK_2_GENERATOR_PASS
CHECK_3_CONFIG_PASS
CHECK_4_DATA_PASS
CHECK_5_ROLE_INIT_PASS
CHECK_6_FSDP2_ACCUMULATION_PASS
STAGE2_PRETRAIN_PASS
```

第 6 项使用 tiny 参数在真实 8×H100 FSDP2 上对比 micro2×acc4、micro1×acc8、
sync/no-sync 与 global-batch reference 的 loss、梯度和更新后参数。这个脚本不会启动 Stage-2 训练。
