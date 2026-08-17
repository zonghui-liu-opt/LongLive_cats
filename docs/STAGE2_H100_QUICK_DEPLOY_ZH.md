# Stage-2 训练前：只做 6 个检查

推荐直接使用新的单一入口；它会逐步说明作用并只在严格复验后打印 PASS：

```bash
bash run_stage2_h100.sh help
```

通过后继续运行该脚本显示的下一条命令。复杂故障排查才需要查看
[完整训练与推理手册](STAGE2_H100_TRAINING_INFERENCE_RUNBOOK_ZH.md)。下面保留旧的手工 prepare 说明。

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
# 必须位于 Git checkout 外；正式推理会拒绝含 untracked 产物的仓库。
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
