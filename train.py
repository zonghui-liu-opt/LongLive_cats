# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from omegaconf import OmegaConf

from utils.config import normalize_config, section_get


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no-save", "--no_save", dest="no_save", action="store_true")
    parser.add_argument("--no-visualize", "--no_visualize", dest="no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="", help="Path to the directory to save logs")
    parser.add_argument("--wandb-save-dir", type=str, default="", help="Path to the directory to save wandb logs")
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument("--no-auto-resume", action="store_true", help="Disable auto resume from latest checkpoint in logdir")
    parser.add_argument("--generate-before-train", action="store_true", help="Run one evaluation inference before training starts")
    parser.add_argument(
        "--stage1-dry-run-one-update",
        action="store_true",
        help="Run exactly two Stage-1 micro-steps and one isolated optimizer update.",
    )

    args = parser.parse_args()

    config = normalize_config(OmegaConf.load(args.config_path))
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    # CLI is an opt-out switch. An omitted False flag must never turn a YAML
    # `disable_wandb: true` back on.
    config.disable_wandb = bool(config.get("disable_wandb", False)) or args.disable_wandb
    config.auto_resume = not args.no_auto_resume  # Default to True unless --no-auto-resume is specified
    config.generate_before_train = args.generate_before_train
    config.stage1_dry_run_one_update = args.stage1_dry_run_one_update

    if section_get(config, "data", "backend", None) == "stage1_i2v_cache":
        if section_get(config, "infra", "fsdp_backend", None) != "fsdp2":
            raise ValueError("Stage-1 cache training requires infra.fsdp_backend=fsdp2.")
        from utils.distributed import validate_stage1_fsdp2_api

        validate_stage1_fsdp2_api()

    if config.trainer == "score_distillation":
        from trainer import ScoreDistillationTrainer

        trainer = ScoreDistillationTrainer(config)
    elif config.trainer == "diffusion":
        from trainer import DiffusionTrainer

        trainer = DiffusionTrainer(config)
    else:
        raise ValueError(f"Unsupported trainer: {config.trainer!r}")
    try:
        trainer.train()
    finally:
        if not config.disable_wandb:
            from utils.optional_wandb import wandb

            wandb.finish()


if __name__ == "__main__":
    main()
