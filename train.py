# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
import argparse
import os
from pathlib import Path
import sys

# Disable Python bytecode before importing any project module so deployment
# runs do not create ``__pycache__`` beside the source files. ``-B`` remains a
# harmless deployment defense-in-depth.
sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from omegaconf import OmegaConf  # noqa: E402

from utils.config import normalize_config, section_get  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no-save", "--no_save", dest="no_save", action="store_true")
    parser.add_argument(
        "--no-visualize", "--no_visualize", dest="no_visualize", action="store_true"
    )
    parser.add_argument(
        "--logdir", type=str, default="", help="Path to the directory to save logs"
    )
    parser.add_argument(
        "--wandb-save-dir",
        type=str,
        default="",
        help="Path to the directory to save wandb logs",
    )
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Disable auto resume from latest checkpoint in logdir",
    )
    parser.add_argument(
        "--generate-before-train",
        action="store_true",
        help="Run one evaluation inference before training starts",
    )
    parser.add_argument(
        "--stage1-dry-run-one-update",
        action="store_true",
        help="Run exactly two Stage-1 micro-steps and one isolated optimizer update.",
    )
    parser.add_argument(
        "--stage2-smoke",
        choices=("C0", "C1", "C2"),
        default=None,
        help=(
            "Run one Stage-2 H100 smoke cycle: C0=cold+save, "
            "C1=resume+DMD+save, C2=resume+forced-DFD+discard."
        ),
    )

    args = parser.parse_args()

    raw_config = OmegaConf.load(args.config_path)
    raw_trainer = OmegaConf.select(raw_config, "algorithm.trainer", default=None)
    if raw_trainer == "stage2_distillation":
        # Stage-2 owns a strict nested schema.  Legacy normalize_config flattens
        # sections and must never run before its authoritative resolver.
        from trainer import Stage2DistillationTrainer
        from utils.stage2_config import resolve_stage2_config

        if args.generate_before_train:
            raise ValueError(
                "Stage-2 training does not run inference before training; use the "
                "separate post-training batch-inference checkpoint."
            )
        config = resolve_stage2_config(raw_config)
        smoke_mode = args.stage2_smoke
        smoke_no_save = smoke_mode == "C2"
        smoke_auto_resume = smoke_mode in {"C1", "C2"}
        trainer = Stage2DistillationTrainer(
            config,
            output_dir=args.logdir,
            no_save=(smoke_no_save if smoke_mode else args.no_save),
            no_visualize=args.no_visualize,
            auto_resume=(smoke_auto_resume if smoke_mode else not args.no_auto_resume),
            smoke_mode=smoke_mode,
        )
        trainer.train()
        return

    config = normalize_config(raw_config)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    config_name = os.path.splitext(os.path.basename(args.config_path))[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.wandb_save_dir = args.wandb_save_dir
    # CLI is an opt-out switch. An omitted False flag must never turn a YAML
    # `disable_wandb: true` back on.
    config.disable_wandb = (
        bool(config.get("disable_wandb", False)) or args.disable_wandb
    )
    config.auto_resume = (
        not args.no_auto_resume
    )  # Default to True unless --no-auto-resume is specified
    config.generate_before_train = args.generate_before_train
    config.stage1_dry_run_one_update = args.stage1_dry_run_one_update

    if section_get(config, "data", "backend", None) == "stage1_i2v_cache":
        if section_get(config, "infra", "fsdp_backend", None) != "fsdp2":
            raise ValueError(
                "Stage-1 cache training requires infra.fsdp_backend=fsdp2."
            )
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
