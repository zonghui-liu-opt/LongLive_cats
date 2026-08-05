CUDA_VISIBLE_DEVICES=6,7 torchrun --standalone --nproc_per_node=2 \
  scripts/infer_wan22_ti2v_batch.py \
  --metadata testsets/metadata_6cases_480x832.csv \
  --checkpoint-dir /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/ckpts \
  --auxiliary-dir /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B \
  --output-dir /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/results/baseline/pred_videos/merged_lora_rank64_euler_seed1 \
  --parallel-mode data \
  --batch-size 2 \
  --solver euler \
  --sampling-steps 50 \
  --frame-num 97 \
  --t-shift 5.0 \
  --negative-prompt '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走' \
  --seed 1
