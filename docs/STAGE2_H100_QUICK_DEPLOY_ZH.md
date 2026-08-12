# Stage-2 训练前：只做 5 个检查

先准备一个 600 行的动作标签文件，表头必须是：

```csv
video,action_id
```

要求正好 3 类动作，每类 200 条。

然后只运行：

```bash
cd /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0

export ACTION_SIDECAR_600=/你的绝对路径/action_labels_600.csv
export ATTEST_STAGE2_TEACHER=1

bash prepare_stage2.sh
```

如果服务器文件位置和脚本默认值不同，再覆盖对应路径；相同就不用设置：

```bash
export TEACHER_CKPT=/双向合并权重/ckpts
export TEACHER_PROVENANCE_RECORD=/双向合并权重/merge_manifest.json
export STAGE1_BASE=/checkpoints/stage1/converted_causal_base.pt
export STAGE1_CKPT=/stage1_600cats_3750steps
export METADATA_600=/metadata_600clips_480x832_buckets.csv
export STAGE1_CACHE_MANIFEST=/stage1缓存/cache_manifest.json
```

最后看到下面 6 行，就说明训练前 5 项检查全部通过：

```text
CHECK_1_TEACHER_PASS
CHECK_2_GENERATOR_PASS
CHECK_3_CONFIG_PASS
CHECK_4_DATA_PASS
CHECK_5_ROLE_INIT_PASS
STAGE2_PRETRAIN_PASS
```

这个脚本不会启动 Stage-2 训练。
