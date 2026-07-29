torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py \
  --config_path configs/train_i2v_ar.yaml \
  --logdir logs/train_i2v_ar \
  --wandb-save-dir wandb