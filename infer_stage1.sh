export LONG_LIVE_STAGE1_PROJECT_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0
export LONG_LIVE_STAGE1_AUXILIARY_ROOT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/shared_checkpoints/Wan2.2-TI2V-5B
export LONG_LIVE_STAGE1_SOURCE_CHECKPOINT=/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/DiffSynth-Studio_cats_LoRA/results/merged_bi-direct_Wan2.2-5B-cats/ckpts
export LONG_LIVE_STAGE1_ARCHITECTURE_ROOT="$LONG_LIVE_STAGE1_AUXILIARY_ROOT"
export LONG_LIVE_STAGE1_T5_CHECKPOINT="$LONG_LIVE_STAGE1_AUXILIARY_ROOT/models_t5_umt5-xxl-enc-bf16.pth"
export LONG_LIVE_STAGE1_TOKENIZER_DIR="$LONG_LIVE_STAGE1_AUXILIARY_ROOT/google/umt5-xxl"
export LONG_LIVE_STAGE1_VAE_CHECKPOINT="$LONG_LIVE_STAGE1_AUXILIARY_ROOT/Wan2.2_VAE.pth"
export LONG_LIVE_STAGE1_BASE_CHECKPOINT="$LONG_LIVE_STAGE1_PROJECT_ROOT/checkpoints/stage1/converted_causal_base.pt"
export LONG_LIVE_STAGE1_BASE_MANIFEST="$LONG_LIVE_STAGE1_PROJECT_ROOT/checkpoints/stage1/converted_causal_base.manifest.json"
export LONG_LIVE_STAGE1_TRAIN_DIR="$LONG_LIVE_STAGE1_PROJECT_ROOT/results/stage1_600cats"
export CUDA_VISIBLE_DEVICES=0 

/srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/condaenv/longlive2/bin/python scripts/run_stage1_training_checkpoints_validation.py \
  --training-root /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/results/stage1_600cats \
  --metadata /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/testsets/metadata_6cases_480x832.csv \
  --work-dir  /srv/workspace/Kirin_AI_Workspace/TMG_I/l00832862/LongLive-2.0/work_dir/stage1_600cats \
  --steps 75,150,225,300,375,450,525,600,675,750 \
  --sampling-steps 50 \
  --guidance-scale 5.0 \
  --seed 1