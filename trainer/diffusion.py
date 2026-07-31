# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0

import gc
import hashlib
import json
import logging
import math
from pathlib import Path
import types

from model import CausalDiffusion
from wan_5b.distributed.sp_training import SequenceParallelHelper
from utils.dataset import MultiVideoConcatDataset, MultiTextConcatDataset, cycle, multi_video_collate_fn, eval_collate_fn
from utils.config import section_get, wan_default_config
from utils.misc import set_seed
from utils.sampler import build_training_sampler
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
from utils.optional_wandb import wandb
import time
import os
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, launch_distributed_job, FSDP
from torch.distributed.fsdp import (
    StateDictType, FullStateDictConfig, FullOptimStateDictConfig
)

def save_prompts_to_txt(prompts_for_sample, prompt_txt_path: str, is_main_process: bool):
    """
    Save prompts for one generated video to a txt file.
    Consecutive identical prompts are merged, e.g.:
        [0] a, [1] a, [2] b  =>  [0,1] a\n[2] b\n
    """
    try:
        with open(prompt_txt_path, "w", encoding="utf-8") as f:
            if len(prompts_for_sample) == 0:
                return

            current_prompt = prompts_for_sample[0]
            current_indices = [0]
            for seg_idx in range(1, len(prompts_for_sample)):
                p = prompts_for_sample[seg_idx]
                if p == current_prompt:
                    current_indices.append(seg_idx)
                else:
                    indices_str = ",".join(str(i) for i in current_indices)
                    f.write(f"[{indices_str}] {current_prompt}\n")
                    current_prompt = p
                    current_indices = [seg_idx]
            # flush the last run
            indices_str = ",".join(str(i) for i in current_indices)
            f.write(f"[{indices_str}] {current_prompt}\n")
    except Exception as e:
        if is_main_process:
            print(f"Warning: failed to save prompts to {prompt_txt_path}: {e}")


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        self.stage1_mode = (
            section_get(config, "data", "backend", None) == "stage1_i2v_cache"
        )
        if self.stage1_mode and not self.disable_wandb:
            raise ValueError(
                "Stage-1 is JSONL-only; set logging.disable_wandb=true or pass "
                "--disable-wandb."
            )

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            if getattr(config, "wandb_key", None):
                wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir
        auto_resume = getattr(config, "auto_resume", True)
        self.gradient_accumulation_steps = getattr(config, "gradient_accumulation_steps", 1)

        # Sequence Parallel is supported only for the 5B model; world_size must
        # equal sp_size * dp_size.
        self.sequence_parallel_size = getattr(config, "sequence_parallel_size", 1)
        world_size = dist.get_world_size()
        self.data_parallel_size = world_size // self.sequence_parallel_size if self.sequence_parallel_size > 1 else world_size
        self.sp_group = None
        self.dp_group = None

        if self.is_main_process and self.gradient_accumulation_steps > 1:
            eff_batch = config.batch_size * self.gradient_accumulation_steps * self.data_parallel_size
            print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}, effective batch size: {eff_batch}")

        if self.sequence_parallel_size > 1:
            assert config.model_kwargs.model_name == "Wan2.2-TI2V-5B", (
                f"sequence_parallel_size is only supported for Wan2.2-TI2V-5B model, but got {config.model_kwargs.model_name}"
            )
            assert world_size % self.sequence_parallel_size == 0, (
                f"world_size ({world_size}) must be divisible by sequence_parallel_size ({self.sequence_parallel_size})"
            )
            from wan_5b.distributed.sp_training import (
                validate_sequence_parallel_training_config,
            )
            validate_sequence_parallel_training_config(
                config,
                self.sequence_parallel_size,
                config.num_frame_per_block,
            )
            # Create SP process groups: each DP group contains sp_size ranks,
            # and all_to_all runs only within that group.
            from wan_5b.distributed.sp_training import (
                set_data_parallel_group,
                set_sequence_parallel_group,
            )
            sp_size = self.sequence_parallel_size
            dp_size = self.data_parallel_size
            sp_groups = []
            for g in range(dp_size):
                ranks_g = list(range(g * sp_size, (g + 1) * sp_size))
                sp_groups.append(dist.new_group(ranks=ranks_g))
            self.sp_group = sp_groups[global_rank // sp_size]
            set_sequence_parallel_group(self.sp_group)

            # Also create DP groups: ranks with the same SP rank across DP
            # replicas own the same sequence chunk. For sp_rank=k, the DP group
            # is [k, sp+k, 2*sp+k, ..., (dp-1)*sp+k]. This lets warmup gather
            # different batches of errors for the same block efficiently.
            dp_groups = []
            for k in range(sp_size):
                ranks_k = [g * sp_size + k for g in range(dp_size)]
                dp_groups.append(dist.new_group(ranks=ranks_k))
            self.dp_group = dp_groups[global_rank % sp_size]
            set_data_parallel_group(self.dp_group)
            if self.is_main_process:
                print(f"[SP] Sequence Parallel enabled, sp_size={sp_size}, dp_size={dp_size}, world_size={world_size}")

        # Stage-1 is a deliberately isolated FSDP2/cache-only branch.  Return
        # before any legacy FSDP1, T5/VAE, dataset, EMA, or resume setup runs.
        if self.stage1_mode:
            self._initialize_stage1_fsdp2(global_rank=global_rank)
            return

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(config, device=self.device)
        self.sp_helper = SequenceParallelHelper(self)

        # 2D mode only: print which GLOBAL block-position slice this rank is
        # responsible for. The LAST SP rank carries the most error-accumulated
        # tail blocks, useful when debugging position-bucketed error recycling.
        if self.model.error_buffer is not None and self.model.er_num_blocks > 0:
            lo = self.model.er_block_offset
            hi = lo + self.model.er_num_blocks
            global_rank_id = dist.get_rank()
            sp_rk = global_rank_id % max(self.sequence_parallel_size, 1)
            print(
                f"[ErrorBuffer] rank={global_rank_id} sp_rank={sp_rk} "
                f"covers GLOBAL blocks [{lo},{hi}) ({self.model.er_num_blocks} local blocks)"
            )

        # Bind the SP forward path before FSDP wrapping.
        model_name = getattr(getattr(config, "model_kwargs", None), "model_name", "") or ""
        if self.sequence_parallel_size > 1 and "Wan2.2-TI2V-5B" in model_name:
            from wan_5b.distributed.sequence_parallel import (
                sp_dit_causal_forward_train,
                sp_causal_attn_forward,
            )
            model = self.model.generator.model
            # Use the SP forward implementation in the training path.
            model._forward_train = types.MethodType(sp_dit_causal_forward_train, model)

            # Keep the original self_attn.forward so inference can temporarily
            # disable SP.
            self._sp_attn_blocks = []
            for block in model.blocks:
                sa = block.self_attn
                if not hasattr(sa, "_orig_forward"):
                    sa._orig_forward = sa.forward
                sa.forward = types.MethodType(sp_causal_attn_forward, sa)
                self._sp_attn_blocks.append(sa)

            if self.is_main_process:
                print("[SP] sp_dit_causal_forward_train and sp_causal_attn_forward are enabled")
                print("[SP] natural TF layout is the default training layout")
                if getattr(config, "load_raw_video", False):
                    print(f"[SP-VAE] chunk-halo VAE enabled, halo_latents={self.sp_helper.vae_halo_latents}")

        # ================================= NVFP4 Quantized Training =================================
        self.model_quant = getattr(config, "model_quant", False)
        if self.model_quant:
            from utils.quant import ModelQuantizationConfig, quantize_model_with_filter

            quant_cfg = ModelQuantizationConfig(
                scale_rule=getattr(config, "model_quant_scale_rule", "static_6"),
                activation_scale_rule=getattr(config, "model_quant_activation_scale_rule", "static_6"),
                weight_scale_rule=getattr(config, "model_quant_weight_scale_rule", None),
                gradient_scale_rule=getattr(config, "model_quant_gradient_scale_rule", None),
                keep_master_weights=True,
                weight_scale_2d=True,
            )
            self.model.generator.model, matched_modules = quantize_model_with_filter(
                self.model.generator.model,
                quant_config=quant_cfg,
                filtered_modules=getattr(config, "model_quant_filtered_modules", None),
                use_default_filtered_modules=getattr(config, "model_quant_use_default_filtered_modules", True),
                cast_model_to_bf16=False,
                materialize_for_inference=False,
                verbose=self.is_main_process,
            )
            if self.is_main_process:
                from fouroversix.matmul.cutlass.backend import CUTLASSMatmulBackend

                print(f"[NVFP4] CUTLASS available: {CUTLASSMatmulBackend.is_available()}")
                print(
                    "[NVFP4] Quantized AR training enabled "
                    "(keep_master_weights=True, weight_scale_2d=True)"
                )
                print(f"[NVFP4] {len(matched_modules)} modules excluded from quantization")

        # ================================= Load model weights (before FSDP) =================================
        # Load model weights before FSDP wrapping, while keys still match the
        # raw nn.Module. Optimizer, EMA, and step state are restored after FSDP
        # and the related objects are created, so keep raw_state.
        #
        # Priority: auto_resume from logdir > generator_ckpt for a
        # cold start > random initialization. This allows configs to keep
        # generator_ckpt set while interrupted training still resumes from the
        # latest step. The style mirrors trainer/distillation.py.
        raw_state = None

        checkpoint_path = None

        if auto_resume and self.output_path:
            latest_checkpoint = self.find_latest_checkpoint(self.output_path)
            if latest_checkpoint:
                checkpoint_path = latest_checkpoint
                if self.is_main_process:
                    print(f"Auto resume: Found latest checkpoint at {checkpoint_path}")
            else:
                if self.is_main_process:
                    print("Auto resume: No checkpoint found in logdir, starting from scratch")
        elif auto_resume:
            if self.is_main_process:
                print("Auto resume enabled but no logdir specified, starting from scratch")
        else:
            if self.is_main_process:
                print("Auto resume disabled, starting from scratch")

        if checkpoint_path is None and getattr(config, "generator_ckpt", False):
            checkpoint_path = config.generator_ckpt
            if self.is_main_process:
                print(f"Using explicit checkpoint: {checkpoint_path}")

        if checkpoint_path:
            if self.is_main_process:
                print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            if "generator" in checkpoint:
                if self.is_main_process:
                    print(f"Loading pretrained generator from {checkpoint_path}")
                self.model.generator.load_state_dict(checkpoint["generator"], strict=True)
                del checkpoint["generator"]
            elif "model" in checkpoint:
                if self.is_main_process:
                    print(f"Loading pretrained generator from {checkpoint_path}")
                self.model.generator.load_state_dict(checkpoint["model"], strict=True)
                del checkpoint["model"]
            else:
                if self.is_main_process:
                    print(f"No 'generator'/'model' key found in {checkpoint_path}, treating as raw state_dict")
                self.model.generator.load_state_dict(checkpoint, strict=True)

            gc.collect()

            raw_state = checkpoint
            if "step" in raw_state:
                self.step = raw_state["step"]
                if self.is_main_process:
                    print(f"Resuming from step {self.step}")
            else:
                if self.is_main_process:
                    print("Warning: Step not found in checkpoint, starting from step 0.")

        # ================================= FSDP Wrap =================================
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        frame_raw_height = list(config.image_or_video_shape)[3] * wan_default_config[config.model_kwargs.model_name]["spatial_compression_ratio"]
        frame_raw_width = list(config.image_or_video_shape)[4] * wan_default_config[config.model_kwargs.model_name]["spatial_compression_ratio"]
        total_frames = (list(config.image_or_video_shape)[1] - 1) * wan_default_config[config.model_kwargs.model_name]["temporal_compression_ratio"] + 1
        num_frame_per_block = config.num_frame_per_block
        self.fps = wan_default_config[config.model_kwargs.model_name].get("fps", 16)

        allow_padding = getattr(config, "allow_padding", False)
        min_latent_frames = getattr(config, "min_latent_frames", 0)
        single_video_only = getattr(config, "uniform_prompt", False)
        max_chunks_per_shot = getattr(config, "max_chunks_per_shot", 0)
        dataset_sample_warning_seconds = getattr(config, "dataset_sample_warning_seconds", 60.0)
        dataset_sample_warning_interval_seconds = getattr(
            config, "dataset_sample_warning_interval_seconds", 60.0
        )
        dataset = MultiVideoConcatDataset(
            data_dir=config.data_path,
            video_size=(frame_raw_height, frame_raw_width),
            total_frames=total_frames,
            deterministic=False,
            num_frame_per_block=num_frame_per_block,
            temporal_compression_ratio=wan_default_config[config.model_kwargs.model_name]["temporal_compression_ratio"],
            target_fps=self.fps,
            allow_padding=allow_padding,
            min_latent_frames=min_latent_frames,
            single_video_only=single_video_only,
            independent_first_frame=getattr(config, "independent_first_frame", False),
            return_image=getattr(config, "i2v", False),
            max_chunks_per_shot=max_chunks_per_shot,
            sample_warning_seconds=dataset_sample_warning_seconds,
            sample_warning_interval_seconds=dataset_sample_warning_interval_seconds,
        )
        if allow_padding and self.is_main_process:
            print(f"[Padding] Variable-length training enabled: short videos will be padded with loss masking"
                  f" (min_latent_frames={min_latent_frames})")
        if single_video_only and self.is_main_process:
            print(f"[uniform_prompt] single_video_only enabled: each sample uses one video only")
        # SP ranks in the same SP group need the same batch because they shard
        # the sequence dimension. Use dp_rank for data parallel sampling.
        if self.sequence_parallel_size > 1:
            dp_rank = global_rank // self.sequence_parallel_size
            sampler = build_training_sampler(
                dataset,
                seed=config.seed,
                rank=dp_rank, num_replicas=self.data_parallel_size,
            )
        else:
            sampler = build_training_sampler(
                dataset,
                seed=config.seed,
            )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=2,
            prefetch_factor=1,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=multi_video_collate_fn,
        )

        # Eval dataloader: batch size defaults to 1 to keep validation memory predictable.
        eval_data_path = getattr(config, "eval_data_path", config.data_path)
        inference_num_frames = section_get(config, "evaluation", "num_frames", getattr(config, "inference_num_frames", 0))
        if isinstance(inference_num_frames, (list, tuple)):
            inference_num_frames = inference_num_frames[0] if len(inference_num_frames) > 0 else 0
        eval_total_frames = (
            (inference_num_frames - 1) * wan_default_config[config.model_kwargs.model_name]["temporal_compression_ratio"] + 1
            if inference_num_frames > 0 else total_frames
        )
        temporal_compression_ratio = wan_default_config[config.model_kwargs.model_name]["temporal_compression_ratio"]
        first_chunk_frames = 1 + (num_frame_per_block - 1) * temporal_compression_ratio
        subsequent_chunk_frames = num_frame_per_block * temporal_compression_ratio
        num_blocks = 1 + (eval_total_frames - first_chunk_frames) // subsequent_chunk_frames
        chunks_per_shot = getattr(config, "chunks_per_shot", 0)
        scene_cut_prefix = getattr(config, "scene_cut_prefix", "The scene transitions. ")
        if getattr(config, "i2v", False):
            eval_dataset = MultiVideoConcatDataset(
                data_dir=eval_data_path,
                video_size=(frame_raw_height, frame_raw_width),
                total_frames=eval_total_frames,
                deterministic=True,
                num_frame_per_block=num_frame_per_block,
                temporal_compression_ratio=temporal_compression_ratio,
                target_fps=self.fps,
                allow_padding=allow_padding,
                min_latent_frames=min_latent_frames,
                single_video_only=single_video_only,
                independent_first_frame=getattr(config, "independent_first_frame", False),
                return_image=True,
                max_chunks_per_shot=max_chunks_per_shot,
                scene_cut_prefix=scene_cut_prefix,
                sample_warning_seconds=dataset_sample_warning_seconds,
                sample_warning_interval_seconds=dataset_sample_warning_interval_seconds,
            )
            eval_collate = multi_video_collate_fn
        else:
            eval_dataset = MultiTextConcatDataset(
                data_path=eval_data_path,
                num_blocks=num_blocks,
                chunks_per_shot=chunks_per_shot,
                scene_cut_prefix=scene_cut_prefix,
                deterministic=True,
            )
            eval_collate = eval_collate_fn
        if dist.get_rank() == 0:
            print(f"Using {eval_dataset.__class__.__name__} for eval: {eval_data_path}, num_blocks={num_blocks}")
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, shuffle=False, drop_last=False
        )
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=section_get(config, "evaluation", "val_batch_size", 1),
            sampler=eval_sampler,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=eval_collate,
        )

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
            print("EVAL DATASET SIZE %d" % len(eval_dataset))

        self.dataloader = cycle(dataloader)
        self.eval_dataloader = eval_dataloader

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0) and (self.step >= config.ema_start_step):
            if self.is_main_process:
                print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load optimizer and EMA from checkpoint
        #    Model weights were loaded before FSDP wrapping; restore only
        #    optimizer and EMA state that depend on FSDP here.

        if raw_state is not None:
            if "generator_ema" in raw_state and self.generator_ema is not None:
                self.generator_ema.load_state_dict(raw_state["generator_ema"])
                if self.is_main_process:
                    print("Resuming generator EMA...")
            else:
                if self.is_main_process:
                    print("Warning: Generator EMA checkpoint not found.")

            if "generator_optimizer" in raw_state:
                gen_osd = FSDP.optim_state_dict_to_load(
                    self.model.generator,
                    self.generator_optimizer,
                    raw_state["generator_optimizer"],
                )
                del raw_state["generator_optimizer"]
                self.generator_optimizer.load_state_dict(gen_osd)
                del gen_osd
                if self.is_main_process:
                    print("Resuming generator optimizer...")
            else:
                if self.is_main_process:
                    print("Warning: Generator optimizer checkpoint not found.")

            del raw_state
            gc.collect()

        ##############################################################################################################

        self.max_grad_norm = getattr(config, "max_grad_norm", 10.0)
        self.previous_time = None

        # Resume error buffer from checkpoint.
        #   Try ``*_sp{sp_rank}.pt`` first, fall back to ``*.pt`` (legacy).
        if self.model.error_buffer is not None and auto_resume:
            ckpt_dir = self.find_latest_checkpoint(self.output_path)
            if ckpt_dir is not None:
                ckpt_root = os.path.dirname(ckpt_dir)
                sp_size_ = max(self.sequence_parallel_size, 1)
                global_rank = dist.get_rank() if dist.is_initialized() else 0
                sp_rank = global_rank % sp_size_

                def _resolve_buf_file(stem):
                    if sp_size_ > 1:
                        p = os.path.join(ckpt_root, f"{stem}_sp{sp_rank}.pt")
                        if os.path.exists(p):
                            return p
                    p = os.path.join(ckpt_root, f"{stem}.pt")
                    return p if os.path.exists(p) else None

                for stem, buffer in [("error_buffer", self.model.error_buffer),
                                     ("noise_error_buffer", self.model.noise_error_buffer)]:
                    if buffer is None:
                        continue
                    bf = _resolve_buf_file(stem)
                    if bf is not None:
                        bf_state = torch.load(bf, map_location="cpu")
                        buffer.load_state_dict(bf_state)
                        del bf_state
                        s = buffer.stats()
                        rng = s.get('global_block_range', '')
                        shard = s.get('shard', '')
                        print(f"[{stem}] rank={global_rank} Resumed from "
                              f"{os.path.basename(bf)}: {s['total_entries']} entries, "
                              f"{s['filled_buckets']} buckets, "
                              f"total_added={s['total_added']} {rng} {shard}".rstrip())
                    elif self.is_main_process:
                        print(f"[{stem}] No saved buffer found, starting fresh.")

    def _stage1_rank0_call(self, label, callback):
        """Run heavyweight read-only validation once and broadcast its result."""

        payload = [None]
        if self.is_main_process:
            try:
                payload[0] = {"ok": True, "value": callback()}
            except Exception as exc:
                payload[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(payload, src=0)
        result = payload[0]
        if not isinstance(result, dict) or not result.get("ok", False):
            detail = result if isinstance(result, dict) else {"error": repr(result)}
            raise RuntimeError(
                f"Stage-1 rank-0 {label} failed: "
                f"{detail.get('error_type', 'RuntimeError')}: {detail.get('error')}"
            )
        return result.get("value")

    def _stage1_world_checked_call(self, label, callback):
        """Run rank-local work, then turn any failure into a WORLD failure."""

        value = None
        local_error = None
        try:
            value = callback()
        except Exception as exc:
            local_error = exc
        if not self._stage1_world_consensus(local_error is None):
            raise RuntimeError(f"Stage-1 WORLD stage failed: {label}") from local_error
        return value

    def _validate_stage1_runtime_config(self) -> None:
        config = self.config
        exact = {
            "infra.expected_world_size": (
                int(section_get(config, "infra", "expected_world_size", -1)), 6
            ),
            "infra.sequence_parallel_size": (self.sequence_parallel_size, 3),
            "infra.data_parallel_size": (self.data_parallel_size, 2),
            "data.batch_size": (int(section_get(config, "data", "batch_size", -1)), 1),
            "training.gradient_accumulation_steps": (
                self.gradient_accumulation_steps,
                2,
            ),
            "model.num_frame_per_block": (int(config.num_frame_per_block), 8),
        }
        wrong = {
            name: {"expected": expected, "actual": actual}
            for name, (actual, expected) in exact.items()
            if actual != expected
        }
        if wrong:
            raise ValueError(f"Stage-1 locked topology/config mismatch: {wrong}")
        if dist.get_world_size() != 6:
            raise RuntimeError(
                f"Stage-1 requires WORLD_SIZE=6, got {dist.get_world_size()}"
            )
        if section_get(config, "infra", "fsdp_backend", None) != "fsdp2":
            raise ValueError("Stage-1 requires infra.fsdp_backend=fsdp2")
        if section_get(config, "infra", "sharding_strategy", None) != "hsdp":
            raise ValueError("Stage-1 requires infra.sharding_strategy=hsdp")
        if tuple(section_get(config, "infra", "device_mesh_shape", ())) != (2, 3):
            raise ValueError("Stage-1 requires infra.device_mesh_shape=[2,3]")
        if tuple(section_get(config, "infra", "device_mesh_dim_names", ())) != (
            "replicate",
            "shard",
        ):
            raise ValueError(
                "Stage-1 requires device_mesh_dim_names=[replicate,shard]"
            )
        boolean_contract = {
            "mixed_precision": bool(config.mixed_precision),
            "gradient_checkpointing": bool(config.gradient_checkpointing),
            "i2v": bool(config.i2v),
            "causal": bool(config.causal),
            "teacher_forcing": bool(config.teacher_forcing),
            "independent_first_frame": bool(config.independent_first_frame),
        }
        disabled = [name for name, enabled in boolean_contract.items() if not enabled]
        if disabled:
            raise ValueError(f"Stage-1 required flags are disabled: {disabled}")
        forbidden = {
            "model_quant": bool(getattr(config, "model_quant", False)),
            "cpu_offload": bool(getattr(config, "cpu_offload", False)),
        }
        enabled_forbidden = [name for name, enabled in forbidden.items() if enabled]
        if enabled_forbidden:
            raise ValueError(f"Stage-1 forbidden flags are enabled: {enabled_forbidden}")
        if section_get(config, "evaluation", "interval", None) != 0:
            raise ValueError("Stage-1 evaluation.interval must be 0")
        if list(config.image_or_video_shape) != [1, 24, 48, 30, 52]:
            raise ValueError(
                "Stage-1 canonical image_or_video_shape must be [1,24,48,30,52]"
            )

    def _audit_stage1_inputs_once(self) -> dict:
        """Hash base/source/cache exactly once on rank 0."""

        config = self.config

        def audit():
            from omegaconf import OmegaConf
            from utils.stage1_i2v_data import (
                audit_stage1_cache,
                build_source_fingerprint,
                load_stage1_i2v_manifest,
            )
            from utils.stage1_io import sha256_file

            base_path = Path(config.generator_ckpt).expanduser().resolve()
            base_manifest_path = Path(config.base_manifest).expanduser().resolve()
            if not base_path.is_file() or not base_manifest_path.is_file():
                raise FileNotFoundError(
                    f"missing converted base/manifest: {base_path}, {base_manifest_path}"
                )
            with base_manifest_path.open("r", encoding="utf-8") as handle:
                base_manifest = json.load(handle)
            if base_manifest.get("checkpoint_format") != "longlive_causal_base_init":
                raise RuntimeError("base manifest checkpoint_format is not causal init")
            if int(base_manifest.get("checkpoint_version", -1)) != 1:
                raise RuntimeError("unsupported converted causal base version")
            base_sha256 = sha256_file(base_path)
            output_meta = base_manifest.get("output", {})
            if output_meta.get("sha256") != base_sha256:
                raise RuntimeError(
                    "converted base SHA256 differs from its manifest: "
                    f"manifest={output_meta.get('sha256')}, actual={base_sha256}"
                )
            if int(output_meta.get("size", -1)) != base_path.stat().st_size:
                raise RuntimeError("converted base size differs from its manifest")

            records = load_stage1_i2v_manifest(
                config.data.metadata_path,
                expected_num_samples=int(config.data.expected_num_samples),
                require_files=True,
                validate_images=True,
            )
            source_fingerprint = build_source_fingerprint(
                records,
                model_paths={
                    "vae_checkpoint": config.model_paths.vae_checkpoint,
                    "t5_checkpoint": config.model_paths.t5_checkpoint,
                    "tokenizer_dir": config.model_paths.tokenizer_dir,
                },
                preprocessing=OmegaConf.to_container(
                    config.preprocessing, resolve=True
                ),
            )
            cache_audit = audit_stage1_cache(
                config.data.cache_dir,
                source_fingerprint=source_fingerprint,
                expected_num_samples=int(config.data.expected_num_samples),
                allowed_latent_spatial_shapes=tuple(
                    tuple(int(value) for value in shape)
                    for shape in config.data.allowed_latent_spatial_shapes
                ),
            )
            return {
                "base_checkpoint": str(base_path),
                "base_manifest": str(base_manifest_path),
                "base_sha256": base_sha256,
                "base_manifest_sha256": sha256_file(base_manifest_path),
                "cache_manifest_sha256": cache_audit["manifest_sha256"],
                "cache_num_samples": cache_audit["num_samples"],
                "cache_total_bytes": cache_audit["total_bytes"],
                "cache_bucket_counts": cache_audit["bucket_counts"],
                "source_fingerprint": source_fingerprint["aggregate_sha256"],
            }

        return self._stage1_rank0_call("base/cache audit", audit)

    @staticmethod
    def _stage1_base_transformer(peft_model):
        getter = getattr(peft_model, "get_base_model", None)
        transformer = getter() if callable(getter) else peft_model
        if transformer.__class__.__name__ != "CausalWanModel":
            raise TypeError(
                "Stage-1 PEFT base must be CausalWanModel, got "
                f"{transformer.__class__.__name__}"
            )
        return transformer

    def _bind_stage1_sequence_parallel(self, transformer) -> None:
        from wan_5b.distributed.sequence_parallel import (
            sp_causal_attn_forward,
            sp_dit_causal_forward_train,
        )

        transformer._forward_train = types.MethodType(
            sp_dit_causal_forward_train, transformer
        )
        blocks = tuple(transformer.blocks)
        if len(blocks) != 30:
            raise RuntimeError(f"Stage-1 expected 30 Wan blocks, got {len(blocks)}")
        self._sp_attn_blocks = []
        for block in blocks:
            attention = block.self_attn
            if not hasattr(attention, "_orig_forward"):
                attention._orig_forward = attention.forward
            attention.forward = types.MethodType(sp_causal_attn_forward, attention)
            self._sp_attn_blocks.append(attention)

    def _initialize_stage1_fsdp2(self, *, global_rank: int) -> None:
        """Initialize the cache-only Stage-1 branch without legacy FSDP1."""

        from utils.distributed import (
            STAGE1_FSDP2_RANK_LAYOUT,
            TrainableShardedEMA,
            build_stage1_fsdp2_device_mesh,
            fsdp2_wrap_stage1,
        )
        from utils.inference_utils import load_generator_checkpoint
        from utils.jsonl_logger import JsonlLogger
        from utils.lora_utils import (
            audit_fsdp2_lora_dtensor_topology,
            build_lora_shard_schema,
            configure_lora_for_model,
            load_lora_safetensors_strict,
        )
        from utils.stage1_checkpoint import (
            checkpoint_step,
            validate_checkpoint,
        )
        from utils.stage1_i2v_data import (
            Stage1I2VCacheDataset,
            stage1_i2v_cache_collate,
        )
        from utils.stage1_schedule import resolve_stage1_schedule

        self._stage1_world_checked_call(
            "runtime config validation", self._validate_stage1_runtime_config
        )
        # Reject wrong GPU count/type/memory, backend, host layout, or BF16
        # support before scanning the dataset or loading the 5B base.
        self.stage1_mesh = build_stage1_fsdp2_device_mesh()
        self.stage1_input_audit = self._audit_stage1_inputs_once()
        self.stage1_base_sha256 = self.stage1_input_audit["base_sha256"]

        def resolve_config_payload():
            resolved_yaml = OmegaConf.to_yaml(
                self.config, resolve=True, sort_keys=True
            )
            if not resolved_yaml.endswith("\n"):
                resolved_yaml += "\n"
            payload = resolved_yaml.encode("utf-8")
            return payload, hashlib.sha256(payload).hexdigest()

        (
            self.stage1_resolved_config_bytes,
            self.stage1_resolved_config_sha256,
        ) = self._stage1_world_checked_call(
            "resolved config serialization", resolve_config_payload
        )

        self.stage1_resume_checkpoint = None
        if bool(getattr(self.config, "auto_resume", True)):
            if not self.output_path:
                raise ValueError("Stage-1 auto-resume requires an explicit --logdir")

            def discover_resume_checkpoint():
                from utils.stage1_io import sha256_file

                root = Path(self.output_path).expanduser()
                nonempty = root.is_dir() and any(root.iterdir())
                markers = sorted(root.rglob("_RESUMABLE_SUCCESS")) if root.is_dir() else []
                marked_checkpoints = []
                for marker in markers:
                    candidate = marker.parent
                    if candidate.parent.resolve() != root.resolve():
                        raise RuntimeError(
                            "unexpected nested resumable marker in Stage-1 output: "
                            f"{marker}"
                        )
                    try:
                        checkpoint_step(candidate)
                    except ValueError as exc:
                        raise RuntimeError(
                            "resumable marker is not inside checkpoint_model_XXXXXX: "
                            f"{marker}"
                        ) from exc
                    validate_checkpoint(
                        candidate,
                        require_resumable=True,
                        expected_base_sha256=self.stage1_base_sha256,
                        expected_topology=(6, 3, 2),
                    )
                    marked_checkpoints.append(candidate)
                if root.is_dir():
                    for candidate in sorted(root.iterdir()):
                        if not candidate.is_dir():
                            continue
                        try:
                            checkpoint_step(candidate)
                        except ValueError:
                            continue
                        if (candidate / "_RESUMABLE_SUCCESS").is_file():
                            continue
                        if not (candidate / "_SUCCESS").is_file():
                            raise RuntimeError(
                                "incomplete uncommitted Stage-1 checkpoint directory: "
                                f"{candidate}"
                            )
                        manifest = validate_checkpoint(
                            candidate,
                            require_resumable=False,
                            expected_base_sha256=self.stage1_base_sha256,
                            expected_topology=(6, 3, 2),
                        )
                        if bool(manifest.get("resumable", False)):
                            raise RuntimeError(
                                "checkpoint advertises resumable heavy state but "
                                f"is missing _RESUMABLE_SUCCESS: {candidate}"
                            )
                if nonempty and not marked_checkpoints:
                    raise RuntimeError(
                        "Stage-1 output directory is non-empty but contains no "
                        "valid resumable checkpoint; refusing a silent cold start: "
                        f"{root}"
                    )
                checkpoint = max(
                    marked_checkpoints,
                    key=checkpoint_step,
                    default=None,
                )
                if checkpoint is None:
                    return None
                resolved_path = checkpoint / "resolved_config.yaml"
                resolved_hash = sha256_file(resolved_path)
                if resolved_hash != self.stage1_resolved_config_sha256:
                    raise RuntimeError(
                        "latest resumable checkpoint resolved config differs from "
                        f"this launch: checkpoint={resolved_hash}, "
                        f"current={self.stage1_resolved_config_sha256}"
                    )
                return str(checkpoint.resolve())

            resume_value = self._stage1_rank0_call(
                "resumable checkpoint discovery", discover_resume_checkpoint
            )
            if resume_value is not None:
                self.stage1_resume_checkpoint = Path(resume_value)
        if (
            self.stage1_resume_checkpoint is not None
            and bool(self.config.stage1_dry_run_one_update)
        ):
            raise ValueError("Stage-1 dry-run must be isolated and cannot auto-resume")

        def construct_cache_only_model():
            model = CausalDiffusion(self.config, device=self.device)
            if model.text_encoder is not None or model.vae is not None:
                raise RuntimeError(
                    "Stage-1 cache-only init must not construct T5/VAE"
                )
            return model

        self.model = self._stage1_world_checked_call(
            "cache-only causal architecture construction",
            construct_cache_only_model,
        )
        self.sp_helper = SequenceParallelHelper(self)

        # Architecture-only construction is FP32. Cast the entire immutable
        # base first so strict loading cannot silently retain FP32 storage.
        def load_and_freeze_base():
            self.model.generator.to(device="cpu", dtype=torch.bfloat16)
            incompatible = load_generator_checkpoint(
                self.model.generator,
                self.stage1_input_audit["base_checkpoint"],
                strict=True,
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "converted causal base strict load returned incompatible keys"
                )
            self.model.generator.requires_grad_(False)

        self._stage1_world_checked_call(
            "immutable BF16 base strict load/freeze", load_and_freeze_base
        )

        # All ranks must start from the same LoRA A initialization even though
        # their training RNG streams are intentionally rank-specific.
        def configure_stage1_lora():
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(self.config.seed))
                return configure_lora_for_model(
                    self.model.generator.model,
                    model_name="generator",
                    lora_config=self.config.adapter,
                    is_main_process=self.is_main_process,
                )

        self.model.generator.model = self._stage1_world_checked_call(
            "pre-FSDP LoRA configuration/audit", configure_stage1_lora
        )
        if self.stage1_resume_checkpoint is not None:
            self._stage1_world_checked_call(
                "pre-FSDP raw adapter strict resume load",
                lambda: load_lora_safetensors_strict(
                    self.model.generator.model,
                    self.stage1_resume_checkpoint / "adapter_raw.safetensors",
                    expected_dtype=torch.float32,
                    require_finite=True,
                    verify_tensors=True,
                ),
            )

        def build_and_validate_lora_schema():
            schema = build_lora_shard_schema(
                self.model.generator.model, expected_dtype=torch.float32
            )
            if len(schema) != int(self.config.adapter.expected_adapter_tensors):
                raise RuntimeError(
                    "Stage-1 pre-FSDP LoRA schema tensor count mismatch"
                )
            return schema

        self.stage1_lora_schema = self._stage1_world_checked_call(
            "pre-FSDP LoRA schema audit", build_and_validate_lora_schema
        )

        def bind_sequence_parallel():
            transformer = self._stage1_base_transformer(
                self.model.generator.model
            )
            self._bind_stage1_sequence_parallel(transformer)
            return transformer

        self.stage1_transformer = self._stage1_world_checked_call(
            "sequence-parallel forward binding", bind_sequence_parallel
        )

        self.model.generator = fsdp2_wrap_stage1(
            self.model.generator,
            transformer=self.model.generator.model,
            mesh=self.stage1_mesh,
            expected_blocks=30,
            expected_trainable_tensors=int(
                self.config.adapter.expected_adapter_tensors
            ),
            expected_trainable_numel=int(
                self.config.adapter.expected_trainable_parameters
            ),
        )
        def audit_post_fsdp_topology():
            topology_audit = audit_fsdp2_lora_dtensor_topology(
                self.model.generator,
                expected_schema=self.stage1_lora_schema,
            )
            if topology_audit["global_trainable_parameters"] != int(
                self.config.adapter.expected_trainable_parameters
            ):
                raise RuntimeError(
                    "post-FSDP2 LoRA global parameter audit failed"
                )
            return topology_audit

        self._stage1_world_checked_call(
            "post-FSDP2 LoRA topology audit", audit_post_fsdp_topology
        )
        self.stage1_authoritative_shard_group = self.stage1_mesh.get_group("shard")
        self.stage1_replica_group = self.stage1_mesh.get_group("replicate")

        optimizer_config = self.config.optimizer
        from utils.stage1_checkpoint import audit_stage1_lora_optimizer

        def build_lora_optimizer():
            if str(optimizer_config.type).lower() != "adamw":
                raise ValueError("Stage-1 optimizer.type must be adamw")
            trainable = [
                parameter
                for parameter in self.model.generator.parameters()
                if parameter.requires_grad
            ]
            optimizer = torch.optim.AdamW(
                trainable,
                lr=float(self.config.training.phases[0].lr.start),
                betas=(
                    float(optimizer_config.beta1),
                    float(optimizer_config.beta2),
                ),
                eps=float(optimizer_config.eps),
                weight_decay=float(optimizer_config.weight_decay),
            )
            audit_stage1_lora_optimizer(
                self.model.generator,
                optimizer,
                expected_schema=self.stage1_lora_schema,
            )
            return optimizer

        self.generator_optimizer = self._stage1_world_checked_call(
            "post-FSDP LoRA-only optimizer construction/audit",
            build_lora_optimizer,
        )
        self.max_grad_norm = float(optimizer_config.max_grad_norm)
        ema_config = self.config.ema

        def build_trainable_ema():
            if not (
                bool(ema_config.enabled)
                and str(ema_config.dtype) == "float32"
                and str(ema_config.device) == "cpu"
                and bool(ema_config.trainable_only)
            ):
                raise ValueError(
                    "Stage-1 requires enabled trainable-only CPU FP32 EMA"
                )
            return TrainableShardedEMA(
                self.model.generator,
                decay=float(ema_config.decay),
                start_step=int(ema_config.start_step),
                topology={
                    "rank_layout": STAGE1_FSDP2_RANK_LAYOUT,
                    "mesh_dim_names": ("replicate", "shard"),
                },
            )

        self.generator_ema = self._stage1_world_checked_call(
            "LoRA-only EMA construction", build_trainable_ema
        )

        def build_data_runtime():
            dataset = Stage1I2VCacheDataset(
                self.config.data.cache_dir,
                expected_num_samples=int(self.config.data.expected_num_samples),
                verify_on_read=False,
                allowed_latent_spatial_shapes=tuple(
                    tuple(int(value) for value in shape)
                    for shape in self.config.data.allowed_latent_spatial_shapes
                ),
            )
            dp_rank = global_rank // self.sequence_parallel_size
            warmup_microbatches = int(
                self.config.error_recycling.buffer_warmup_steps
            ) * self.gradient_accumulation_steps
            sampler = build_training_sampler(
                dataset,
                seed=int(self.config.seed),
                rank=dp_rank,
                num_replicas=self.data_parallel_size,
                resolution_aware=True,
                spatial_shapes=dataset.spatial_shapes,
                local_batch_size=int(self.config.batch_size),
                warmup_microbatches=warmup_microbatches,
            )
            dataloader_generator = torch.Generator()
            dataloader_generator.manual_seed(int(self.config.seed) + 10_000)
            loader_kwargs = {
                "dataset": dataset,
                "batch_size": int(self.config.batch_size),
                "sampler": sampler,
                "num_workers": int(self.config.num_workers),
                "pin_memory": False,
                "persistent_workers": False,
                "collate_fn": stage1_i2v_cache_collate,
                "generator": dataloader_generator,
            }
            if int(self.config.num_workers) > 0:
                loader_kwargs["prefetch_factor"] = 1
            dataloader = torch.utils.data.DataLoader(**loader_kwargs)
            schedule = resolve_stage1_schedule(
                self.config, dataloader_length=len(dataloader)
            )
            return dataset, sampler, dataloader_generator, dataloader, schedule

        (
            dataset,
            self.stage1_sampler,
            self.stage1_dataloader_generator,
            self.stage1_dataloader,
            self.stage1_schedule,
        ) = self._stage1_world_checked_call(
            "cache dataset/sampler/DataLoader/schedule construction",
            build_data_runtime,
        )
        self.stage1_epoch = 0
        self.stage1_microbatch_cursor = 0
        self.stage1_iterator = None
        self.step = 0
        self.stage1_nonfinite_attempt_count = 0
        self.stage1_max_attempts = int(self.config.nonfinite_max_attempts_per_update)
        if self.stage1_max_attempts != 2:
            raise ValueError("Stage-1 nonfinite_max_attempts_per_update must be 2")

        resume_rng_state = None
        parent_run_id = None
        checkpoint_next_attempt_index = 0
        if self.stage1_resume_checkpoint is not None:
            (
                resume_rng_state,
                parent_run_id,
                checkpoint_next_attempt_index,
            ) = self._restore_stage1_fsdp2_checkpoint()
        else:
            self.stage1_sampler.set_epoch(self.stage1_epoch)
            self.stage1_iterator = self._stage1_world_checked_call(
                "cold-start DataLoader iterator construction",
                lambda: iter(self.stage1_dataloader),
            )

        if not self.output_path:
            raise ValueError("Stage-1 requires an explicit --logdir")
        metrics_path = Path(self.config.jsonl_path)
        if not metrics_path.is_absolute():
            metrics_path = Path(self.output_path) / metrics_path
        self.stage1_logger = self._stage1_world_checked_call(
            "JSONL lineage initialization",
            lambda: JsonlLogger(
                metrics_path,
                experiment_id=str(self.config.config_name),
                parent_run_id=parent_run_id,
                resume_from_step=self.step,
                checkpoint_next_attempt_index=checkpoint_next_attempt_index,
                fsync_every_steps=int(self.config.fsync_every_steps),
                enabled=self.is_main_process,
                run_metadata={
                    "seed": int(self.config.seed),
                    "schedule_sha256": self.stage1_schedule.resolved_hash(),
                    "base_sha256": self.stage1_base_sha256,
                    "cache_manifest_sha256": self.stage1_input_audit[
                        "cache_manifest_sha256"
                    ],
                    "dry_run": bool(self.config.stage1_dry_run_one_update),
                    "fsdp_backend": "fsdp2",
                    "device_mesh": [[0, 1, 2], [3, 4, 5]],
                },
            ),
        )
        self._stage1_world_checked_call(
            "generator train-mode transition", self.model.generator.train
        )
        if self.is_main_process:
            resume_suffix = (
                f", resumed from {self.stage1_resume_checkpoint}"
                if self.stage1_resume_checkpoint is not None
                else ""
            )
            print(
                "[Stage-1] FSDP2 HSDP initialized: DP2×SP3, "
                f"{len(dataset)} cache records, {self.stage1_schedule.total_updates} "
                f"updates{resume_suffix}"
            )
        # Iterator construction/skip and JSONL lineage setup are deliberately
        # complete before restoring the per-rank model RNG stream. Nothing
        # below this point may consume Python/NumPy/Torch/CUDA training RNG.
        if resume_rng_state is not None:
            from utils.stage1_checkpoint import restore_rng_state

            self._stage1_world_checked_call(
                "per-rank RNG restore",
                lambda: restore_rng_state(
                    resume_rng_state, require_cuda_topology=True
                ),
            )

    def _restore_stage1_fsdp2_checkpoint(self):
        """Restore raw-LoRA-adjacent state and rebuild the committed cursor."""

        from utils.stage1_checkpoint import (
            load_stage1_resume_payloads,
            restore_stage1_optimizer_state,
        )

        checkpoint = self.stage1_resume_checkpoint
        if checkpoint is None:
            raise RuntimeError("Stage-1 resume restore called without a checkpoint")
        payload = load_stage1_resume_payloads(
            checkpoint,
            replica_group=self.stage1_replica_group,
            expected_base_sha256=self.stage1_base_sha256,
            expected_resolved_config_sha256=self.stage1_resolved_config_sha256,
        )
        metadata = payload.trainer_state

        def validate_metadata():
            completed_step = int(metadata["completed_step"])
            if completed_step > self.stage1_schedule.total_updates:
                raise RuntimeError(
                    "resume completed step exceeds resolved schedule: "
                    f"{completed_step}>{self.stage1_schedule.total_updates}"
                )
            expected_epoch, expected_cursor, _ = (
                self.stage1_schedule.epoch_position(
                    completed_step * self.gradient_accumulation_steps
                )
            )
            actual_position = (
                int(metadata["global_epoch"]),
                int(metadata["committed_microbatch_cursor_in_epoch"]),
            )
            if actual_position != (expected_epoch, expected_cursor):
                raise RuntimeError(
                    "resume epoch/cursor differs from completed-step derivation: "
                    f"checkpoint={actual_position}, "
                    f"expected={(expected_epoch, expected_cursor)}"
                )

            phase = metadata["phase_derivation"]
            expected_schedule = self.stage1_schedule.to_dict()
            if phase.get("schedule_sha256") != self.stage1_schedule.resolved_hash():
                raise RuntimeError("resume schedule hash differs from current schedule")
            if phase.get("schedule") != expected_schedule:
                raise RuntimeError(
                    "resume resolved schedule payload differs from current schedule"
                )
            if phase.get("cache_manifest_sha256") != self.stage1_input_audit[
                "cache_manifest_sha256"
            ]:
                raise RuntimeError(
                    "resume cache manifest hash differs from the current audited cache"
                )
            ema_last_step = payload.ema_state.get("last_completed_step")
            if ema_last_step is None or int(ema_last_step) != completed_step:
                raise RuntimeError(
                    "resume EMA step differs from trainer completed_step: "
                    f"ema={ema_last_step}, trainer={completed_step}"
                )
            expected_ema_initialized = completed_step >= int(
                self.config.ema.start_step
            )
            if bool(payload.ema_state.get("initialized")) != expected_ema_initialized:
                raise RuntimeError(
                    "resume EMA initialization state differs from completed_step"
                )
            parent_run_id = phase.get("run_id")
            if not isinstance(parent_run_id, str) or not parent_run_id:
                raise RuntimeError(
                    "resume checkpoint is missing its JSONL parent run_id"
                )

            expected_sampler_state = {
                "schema_version": 1,
                "seed": int(self.config.seed),
                "global_epoch": expected_epoch,
                "committed_microbatch_cursor_in_epoch": expected_cursor,
                "micro_batches_per_epoch": len(self.stage1_dataloader),
                "data_parallel_size": self.data_parallel_size,
                "resolution_aware": True,
            }
            if dict(metadata["sampler_state"]) != expected_sampler_state:
                raise RuntimeError(
                    "resume sampler state differs from the deterministic Stage-1 "
                    f"contract: checkpoint={dict(metadata['sampler_state'])}, "
                    f"expected={expected_sampler_state}"
                )
            return (
                completed_step,
                expected_epoch,
                expected_cursor,
                parent_run_id,
                int(metadata["next_attempt_index"]),
                int(metadata["nonfinite_attempt_count"]),
                metadata["dataloader_generator_state"].clone(),
            )

        (
            completed_step,
            expected_epoch,
            expected_cursor,
            parent_run_id,
            next_attempt_index,
            nonfinite_attempt_count,
            saved_loader_state,
        ) = self._stage1_world_checked_call(
            "resume metadata/schedule validation", validate_metadata
        )

        restore_stage1_optimizer_state(
            self.model.generator,
            self.generator_optimizer,
            payload.optimizer_state,
            expected_schema=self.stage1_lora_schema,
            expected_completed_step=completed_step,
        )

        def restore_rank_local_state():
            self.generator_ema.load_state_dict(
                payload.ema_state, self.model.generator
            )
            self.model.error_buffer.load_state_dict(
                payload.error_buffer_state,
                strict_offset=True,
                strict_schema=True,
            )
            self.step = completed_step
            self.stage1_epoch = expected_epoch
            self.stage1_microbatch_cursor = expected_cursor
            self.stage1_nonfinite_attempt_count = nonfinite_attempt_count
            self._stage1_rebuild_iterator_at_committed_cursor(
                saved_loader_state
            )

        self._stage1_world_checked_call(
            "rank-local EMA/error-buffer/cursor restore",
            restore_rank_local_state,
        )
        return (
            payload.rng_state,
            parent_run_id,
            next_attempt_index,
        )

    def _stage1_rebuild_iterator_at_committed_cursor(
        self, saved_loader_state: torch.Tensor
    ) -> None:
        """Recreate the deterministic epoch iterator and skip committed reads."""

        if not isinstance(saved_loader_state, torch.Tensor):
            raise TypeError("saved DataLoader generator state must be a Tensor")
        if saved_loader_state.device.type != "cpu":
            raise TypeError("saved DataLoader generator state must be on CPU")
        if not 0 <= self.stage1_microbatch_cursor <= len(self.stage1_dataloader):
            raise RuntimeError(
                "resume microbatch cursor is outside the DataLoader: "
                f"{self.stage1_microbatch_cursor}/{len(self.stage1_dataloader)}"
            )
        self.stage1_dataloader_generator.set_state(saved_loader_state)
        self.stage1_sampler.set_epoch(self.stage1_epoch)
        if self.step == self.stage1_schedule.total_updates:
            if self.stage1_microbatch_cursor != 0:
                raise RuntimeError(
                    "completed Stage-1 schedule must resume at an epoch boundary"
                )
            self.stage1_iterator = None
            return
        self.stage1_iterator = iter(self.stage1_dataloader)
        for skipped in range(self.stage1_microbatch_cursor):
            try:
                next(self.stage1_iterator)
            except StopIteration as exc:
                raise RuntimeError(
                    "resume DataLoader ended while skipping committed cursor: "
                    f"skipped={skipped}, cursor={self.stage1_microbatch_cursor}"
                ) from exc
        if self.stage1_microbatch_cursor:
            # The saved generator state is already post-current-epoch iterator
            # construction. Recreating that iterator advances it once more;
            # restore the saved value so the next epoch receives the same
            # worker base seed as an uninterrupted run. Cache reads themselves
            # are deterministic and do not use worker RNG.
            self.stage1_dataloader_generator.set_state(saved_loader_state)

    def _stage1_world_consensus(self, local_success: bool) -> bool:
        flag = torch.tensor(
            1 if local_success else 0,
            dtype=torch.int32,
            device=self.device,
        )
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    def _stage1_fetch_update_batches(self):
        microbatches_per_epoch = len(self.stage1_dataloader)
        if self.stage1_microbatch_cursor == 0 and self.stage1_iterator is None:
            self.stage1_sampler.set_epoch(self.stage1_epoch)
            self.stage1_iterator = iter(self.stage1_dataloader)
        if self.stage1_microbatch_cursor < 0 or (
            self.stage1_microbatch_cursor + self.gradient_accumulation_steps
            > microbatches_per_epoch
        ):
            raise RuntimeError(
                "Stage-1 committed microbatch cursor is not on a complete "
                f"accumulation boundary: {self.stage1_microbatch_cursor}/"
                f"{microbatches_per_epoch}"
            )
        batches = []
        for _ in range(self.gradient_accumulation_steps):
            try:
                batches.append(next(self.stage1_iterator))
            except StopIteration as exc:
                raise RuntimeError(
                    "Stage-1 DataLoader ended before its exact 300-microbatch epoch"
                ) from exc
        return tuple(batches)

    def _stage1_validate_logical_batch(self, batch, *, update_index: int) -> dict:
        sample_ids = batch.get("sample_id")
        if not isinstance(sample_ids, torch.Tensor) or tuple(sample_ids.shape) != (1,):
            raise ValueError("Stage-1 local batch must contain one sample_id")
        sample_id = int(sample_ids.item())
        clean = batch.get("clean_latent")
        initial = batch.get("initial_latent")
        prompt = batch.get("prompt_embeds")
        prompt_mask = batch.get("prompt_mask")
        if not isinstance(clean, torch.Tensor) or clean.ndim != 5:
            raise ValueError("Stage-1 clean_latent must have shape [1,24,48,H,W]")
        if tuple(clean.shape[:3]) != (1, 24, 48):
            raise ValueError(f"invalid Stage-1 clean latent shape: {tuple(clean.shape)}")
        if not isinstance(initial, torch.Tensor) or tuple(initial.shape) != (
            1,
            1,
            48,
            clean.shape[-2],
            clean.shape[-1],
        ):
            raise ValueError(f"invalid Stage-1 initial latent shape: {getattr(initial, 'shape', None)}")
        if not torch.equal(clean[:, :1], initial):
            raise RuntimeError("cached clean_latent[0] is not the explicit input image latent")
        if not isinstance(prompt, torch.Tensor) or tuple(prompt.shape) != (1, 512, 4096):
            raise ValueError(f"invalid cached prompt shape: {getattr(prompt, 'shape', None)}")
        if not isinstance(prompt_mask, torch.Tensor) or tuple(prompt_mask.shape) != (1, 512):
            raise ValueError("invalid cached prompt mask shape")
        if clean.dtype != torch.bfloat16 or initial.dtype != torch.bfloat16:
            raise TypeError("Stage-1 cached latents must be BF16")
        if prompt.dtype != torch.bfloat16 or prompt_mask.dtype != torch.bool:
            raise TypeError("Stage-1 cached prompt must be BF16 with bool mask")
        metadata = {
            "sample_id": sample_id,
            "bucket": str(batch["bucket"][0]),
            "height": int(clean.shape[-2]),
            "width": int(clean.shape[-1]),
        }
        sp_values = [None] * self.sequence_parallel_size
        dist.all_gather_object(sp_values, metadata, group=self.sp_group)
        if any(value != sp_values[0] for value in sp_values):
            raise RuntimeError(f"SP ranks loaded different cache records: {sp_values}")

        if update_index < int(self.config.error_recycling.buffer_warmup_steps):
            dp_values = [None] * self.data_parallel_size
            dist.all_gather_object(dp_values, metadata, group=self.dp_group)
            if dp_values[0]["sample_id"] == dp_values[1]["sample_id"]:
                raise RuntimeError(f"DP replicas duplicated a warmup sample: {dp_values}")
            for field in ("bucket", "height", "width"):
                if dp_values[0][field] != dp_values[1][field]:
                    raise RuntimeError(
                        f"DP warmup replicas differ in {field}: {dp_values}"
                    )
        return metadata

    def _stage1_prepare_microbatch(self, batch, *, schedule_values, update_index: int):
        from utils.stage1_error_recycling import broadcast_error_recycling_gate

        metadata = self._stage1_validate_logical_batch(
            batch, update_index=update_index
        )
        clean_latent = batch["clean_latent"].to(
            device=self.device, dtype=torch.bfloat16, non_blocking=False
        )
        initial_latent = batch["initial_latent"].to(
            device=self.device, dtype=torch.bfloat16, non_blocking=False
        )
        conditional_dict = {
            "prompt_embeds": batch["prompt_embeds"].to(
                device=self.device, dtype=torch.bfloat16, non_blocking=False
            )
        }
        loss_mask = self.sp_helper.build_loss_mask(
            batch, clean_latent, clean_latent_is_sharded=False
        )
        shape = list(clean_latent.shape)
        clean_latent, conditional_dict, shape = self.sp_helper.partition_training_inputs(
            image_or_video_shape=shape,
            clean_latent=clean_latent,
            conditional_dict=conditional_dict,
            clean_latent_is_sharded=False,
        )
        initial_latent = self.sp_helper.local_i2v_initial_latent(initial_latent)
        loss_mask, global_valid = self.sp_helper.partition_loss_mask(
            loss_mask, already_sharded=False
        )
        if global_valid is None or int(global_valid.item()) != 23:
            raise RuntimeError(
                f"Stage-1 SP global valid-frame count must be 23, got {global_valid}"
            )
        sp_root_global_rank = (
            dist.get_rank() // self.sequence_parallel_size
        ) * self.sequence_parallel_size
        gate = broadcast_error_recycling_gate(
            schedule_values,
            clean_buffer_update_prob=float(
                self.config.error_recycling.clean_buffer_update_prob
            ),
            group=self.sp_group,
            root_global_rank=sp_root_global_rank,
            device=torch.device("cuda", self.device),
        )
        return {
            "metadata": metadata,
            "image_or_video_shape": shape,
            "conditional_dict": conditional_dict,
            "clean_latent": clean_latent,
            "initial_latent": initial_latent,
            "loss_mask": loss_mask,
            "loss_mask_global_valid_count": global_valid,
            "er_gate": gate.to_dict(),
        }

    @staticmethod
    def _stage1_local_gradients_finite(module) -> tuple[bool, str]:
        try:
            from torch.distributed.tensor import DTensor
        except (ImportError, AttributeError):
            DTensor = ()  # type: ignore[assignment]
        missing = []
        nonfinite = []
        for name, parameter in module.named_parameters():
            if not parameter.requires_grad:
                continue
            gradient = parameter.grad
            if gradient is None:
                missing.append(name)
                continue
            local = gradient.to_local() if isinstance(gradient, DTensor) else gradient
            if local.numel() and not bool(torch.isfinite(local).all().item()):
                nonfinite.append(name)
        if missing or nonfinite:
            return False, f"missing_grad={missing[:4]}, nonfinite_grad={nonfinite[:4]}"
        return True, ""

    def _stage1_attempt_timing(
        self, started_at: float, finished_at: float, batches
    ) -> dict:
        from utils.jsonl_logger import logical_workload, throughput_fields

        elapsed_local = float(finished_at) - float(started_at)
        if not math.isfinite(elapsed_local) or elapsed_local <= 0.0:
            raise RuntimeError(
                f"invalid Stage-1 attempt duration: {elapsed_local}"
            )
        maximum = torch.tensor(elapsed_local, dtype=torch.float64, device=self.device)
        total = maximum.clone()
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        elapsed_max = float(maximum.item())
        elapsed_mean = float(total.item()) / dist.get_world_size()

        patch_tokens = set()
        for batch in batches:
            height = int(batch["clean_latent"].shape[-2])
            width = int(batch["clean_latent"].shape[-1])
            if height % 2 or width % 2:
                raise RuntimeError("Stage-1 latent H/W must be divisible by patch 2×2")
            patch_tokens.add((height // 2) * (width // 2))
        if patch_tokens != {390}:
            raise RuntimeError(f"Stage-1 patch tokens/frame must be 390, got {patch_tokens}")
        first = batches[0]["clean_latent"]
        workload = logical_workload(
            latent_height=int(first.shape[-2]),
            latent_width=int(first.shape[-1]),
            global_samples=(
                self.data_parallel_size * self.gradient_accumulation_steps
            ),
        )
        if (
            workload.dit_tokens != 74_880
            or workload.supervised_tokens != 35_880
            or workload.logical_source_frames != 372
        ):
            raise RuntimeError(f"Stage-1 logical workload mismatch: {workload}")
        fields = throughput_fields(workload, elapsed_max)
        allocated = torch.tensor(
            float(torch.cuda.max_memory_allocated(self.device)), device=self.device
        )
        reserved = torch.tensor(
            float(torch.cuda.max_memory_reserved(self.device)), device=self.device
        )
        dist.all_reduce(allocated, op=dist.ReduceOp.MAX)
        dist.all_reduce(reserved, op=dist.ReduceOp.MAX)
        return {
            **fields,
            "logical_dit_tokens": workload.dit_tokens,
            "supervised_tokens": workload.supervised_tokens,
            "global_samples": workload.global_samples,
            "logical_source_frames": workload.logical_source_frames,
            "step_seconds_max": elapsed_max,
            "step_seconds_mean": elapsed_mean,
            "straggler_ratio": elapsed_max / elapsed_mean,
            "max_allocated_bytes": int(allocated.item()),
            "max_reserved_bytes": int(reserved.item()),
        }

    def _stage1_reduce_success_metrics(self, micro_logs) -> tuple[dict, dict]:
        from utils.stage1_loss import aggregate_loss_metrics

        device = torch.device("cuda", self.device)
        values = torch.zeros(8, dtype=torch.float64, device=device)
        gate_counts = torch.zeros(7, dtype=torch.float64, device=device)
        for log_dict in micro_logs:
            values[0] += log_dict["loss_numerator_local"].double()
            values[1] += log_dict["loss_count_local"].double()
            values[2:5] += log_dict["block_numerators_local"].double()
            values[5:8] += log_dict["block_counts_local"].double()
            if dist.get_rank() % self.sequence_parallel_size == 0:
                gate_counts[0] += 1
                gate_counts[1] += int(log_dict.get("er_logical_active", False))
                gate_counts[2] += int(log_dict.get("er_logical_context", False))
                gate_counts[3] += int(log_dict.get("er_logical_latent", False))
            gate_counts[4] += int(log_dict.get("er_context_applied_blocks", 0))
            gate_counts[5] += int(log_dict.get("er_latent_applied_blocks", 0))
            gate_counts[6] += int(log_dict.get("er_noise_injected", False))
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        dist.all_reduce(gate_counts, op=dist.ReduceOp.SUM)
        losses = aggregate_loss_metrics(
            values[0], values[1], values[2:5], values[5:8]
        )
        expected_counts = (92.0, 28.0, 32.0, 32.0)
        actual_counts = (
            losses["loss_count"],
            losses["blocks"]["block0"]["count"],
            losses["blocks"]["block1"]["count"],
            losses["blocks"]["block2"]["count"],
        )
        if actual_counts != expected_counts:
            raise RuntimeError(
                f"Stage-1 reduced valid-frame counts mismatch: {actual_counts}"
            )
        logical_samples = float(gate_counts[0].item())
        if logical_samples != 4.0:
            raise RuntimeError(
                f"Stage-1 logical gate denominator must be 4, got {logical_samples}"
            )
        er = {
            "logical_samples": int(logical_samples),
            "logical_active_count": int(gate_counts[1].item()),
            "logical_context_count": int(gate_counts[2].item()),
            "logical_latent_count": int(gate_counts[3].item()),
            "logical_active_rate": float(gate_counts[1].item()) / logical_samples,
            "logical_context_rate": float(gate_counts[2].item()) / logical_samples,
            "logical_latent_rate": float(gate_counts[3].item()) / logical_samples,
            "context_applied_blocks": int(gate_counts[4].item()),
            "latent_applied_blocks": int(gate_counts[5].item()),
            "noise_applied_blocks": int(gate_counts[6].item()),
            "context_applied_rate": float(gate_counts[4].item()) / 12.0,
            "latent_applied_rate": float(gate_counts[5].item()) / 12.0,
        }
        return losses, er

    def _stage1_buffer_metrics(self) -> dict:
        local = self.model.error_buffer.stats()
        all_stats = [None] * dist.get_world_size()
        dist.all_gather_object(all_stats, local)

        def summarize(values):
            entries = [int(value["total_entries"]) for value in values]
            filled = [
                int(str(value["filled_buckets"]).split("/", 1)[0])
                for value in values
            ]
            shapes = sorted(
                {
                    shape
                    for value in values
                    for shape in value.get("entries_by_spatial_shape", {})
                }
            )
            by_shape = {}
            for shape in shapes:
                counts = [
                    int(
                        value.get("entries_by_spatial_shape", {}).get(shape, 0)
                    )
                    for value in values
                ]
                by_shape[shape] = {
                    "min": min(counts),
                    "mean": sum(counts) / len(counts),
                    "max": max(counts),
                }
            return {
                "entries_min": min(entries),
                "entries_mean": sum(entries) / len(entries),
                "entries_max": max(entries),
                "filled_buckets_min": min(filled),
                "filled_buckets_mean": sum(filled) / len(filled),
                "filled_buckets_max": max(filled),
                "entries_by_resolution": by_shape,
            }

        aggregate = summarize(all_stats)
        by_sp_position = {
            f"sp{sp_position}": summarize(
                [
                    all_stats[sp_position],
                    all_stats[sp_position + self.sequence_parallel_size],
                ]
            )
            for sp_position in range(self.sequence_parallel_size)
        }
        return {
            **aggregate,
            "by_sp_position": by_sp_position,
        }

    def _stage1_dry_run_ema_smoke(self) -> None:
        from utils.distributed import TrainableShardedEMA
        from utils.stage1_checkpoint import gather_stage1_raw_and_ema_adapters

        smoke = TrainableShardedEMA(
            self.model.generator,
            decay=float(self.config.ema.decay),
            start_step=1,
            topology={
                "rank_layout": ((0, 1, 2), (3, 4, 5)),
                "mesh_dim_names": ("replicate", "shard"),
                "dry_run_only": True,
            },
        )
        if smoke.update_after_step(self.model.generator, 1) != "initialized":
            raise RuntimeError("dry-run EMA smoke did not initialize at step 1")
        if smoke.update_after_step(self.model.generator, 2) != "updated":
            raise RuntimeError("dry-run EMA smoke did not exercise decay update")
        pair = gather_stage1_raw_and_ema_adapters(
            self.model.generator,
            smoke,
            expected_schema=self.stage1_lora_schema,
            authoritative_shard_group=self.stage1_authoritative_shard_group,
            replica_group=self.stage1_replica_group,
            expected_adapter_tensors=int(
                self.config.adapter.expected_adapter_tensors
            ),
            expected_global_numel=int(
                self.config.adapter.expected_trainable_parameters
            ),
        )
        if self.is_main_process and (pair.raw is None or pair.ema is None):
            raise RuntimeError("dry-run EMA smoke did not gather canonical adapters")

    def _run_stage1_update(
        self,
        batches,
        schedule_values,
        *,
        attempt_number: int,
    ):
        from torch.distributed.tensor import DTensor
        from utils.distributed import stage1_fsdp2_accumulation
        from utils.stage1_checkpoint import audit_stage1_lora_optimizer

        self.generator_optimizer.zero_grad(set_to_none=True)
        local_losses_finite = True
        micro_logs = []
        pending_buffer_items = []
        for micro_step, batch in enumerate(batches):
            sync_gradients = micro_step == self.gradient_accumulation_steps - 1
            prepared = self._stage1_prepare_microbatch(
                batch,
                schedule_values=schedule_values,
                update_index=schedule_values.update_index,
            )
            with stage1_fsdp2_accumulation(
                self.model.generator, sync_gradients=sync_gradients
            ):
                generator_loss, log_dict = self.model.generator_loss(
                    image_or_video_shape=prepared["image_or_video_shape"],
                    conditional_dict=prepared["conditional_dict"],
                    unconditional_dict={},
                    clean_latent=prepared["clean_latent"],
                    initial_latent=prepared["initial_latent"],
                    loss_mask=prepared["loss_mask"],
                    loss_mask_global_valid_count=prepared[
                        "loss_mask_global_valid_count"
                    ],
                    global_step=schedule_values.update_index,
                    stage1_schedule_values=schedule_values,
                    er_gate=prepared["er_gate"],
                    defer_error_buffer_commit=True,
                )
                local_losses_finite = local_losses_finite and bool(
                    torch.isfinite(generator_loss.detach()).all().item()
                )
                backward_loss = generator_loss * (
                    self.sequence_parallel_size
                    / self.gradient_accumulation_steps
                )
                backward_loss.backward()
            micro_logs.append(log_dict)
            pending_buffer_items.extend(
                log_dict.get("pending_error_buffer_items", ())
            )

        gradients_finite, gradient_detail = self._stage1_local_gradients_finite(
            self.model.generator
        )
        invalid_residuals = []
        for item_index, item in enumerate(pending_buffer_items):
            if not isinstance(item, tuple) or len(item) != 3:
                invalid_residuals.append(f"item{item_index}:invalid_schema")
                continue
            residual = item[0]
            if not isinstance(residual, torch.Tensor) or (
                residual.numel()
                and not bool(torch.isfinite(residual).all().item())
            ):
                invalid_residuals.append(f"item{item_index}:nonfinite")
        residuals_finite = not invalid_residuals
        locally_finite = (
            local_losses_finite and gradients_finite and residuals_finite
        )
        if not self._stage1_world_consensus(locally_finite):
            self.generator_optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(self.device)
            details = []
            if not local_losses_finite:
                details.append("nonfinite_loss")
            if gradient_detail:
                details.append(gradient_detail)
            if invalid_residuals:
                details.append(f"invalid_residuals={invalid_residuals[:4]}")
            return {
                "success": False,
                "reason": "nonfinite loss/gradient/residual",
                "detail": "; ".join(details),
                "compute_finished_at": time.perf_counter(),
            }

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.generator.parameters(),
            max_norm=self.max_grad_norm,
            error_if_nonfinite=False,
        )
        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()
        grad_norm_value = float(grad_norm.item())
        if not self._stage1_world_consensus(math.isfinite(grad_norm_value)):
            self.generator_optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(self.device)
            return {
                "success": False,
                "reason": "nonfinite pre-clip gradient norm",
                "detail": str(grad_norm_value),
                "compute_finished_at": time.perf_counter(),
            }

        completed_step = self.step + 1
        ema_action = self._stage1_world_checked_call(
            "optimizer/EMA update",
            lambda: (
                self.generator_optimizer.step(),
                self.generator_ema.update_after_step(
                    self.model.generator, completed_step
                ),
            )[1],
        )
        torch.cuda.synchronize(self.device)
        compute_finished_at = time.perf_counter()

        def commit_rank_local_state():
            for items in pending_buffer_items:
                # `pending_error_buffer_items` is itself a list of collected
                # item tuples. Extend above flattens the two microstep lists,
                # so replay each tuple through the local commit primitive.
                self.model._apply_gathered_items(
                    self.model.error_buffer, [items]
                )
            self.step = completed_step
            self.stage1_microbatch_cursor += self.gradient_accumulation_steps
            if self.stage1_microbatch_cursor == len(self.stage1_dataloader):
                self.stage1_epoch += 1
                self.stage1_microbatch_cursor = 0
                self.stage1_sampler.set_epoch(self.stage1_epoch)
                # Do not construct the next epoch iterator before a boundary
                # checkpoint captures the DataLoader generator state.
                self.stage1_iterator = None
            if self.step == 1:
                audit_stage1_lora_optimizer(
                    self.model.generator,
                    self.generator_optimizer,
                    expected_schema=self.stage1_lora_schema,
                    require_initialized_moments=True,
                )

        self._stage1_world_checked_call(
            "error-buffer/cursor commit", commit_rank_local_state
        )
        return {
            "success": True,
            "grad_norm": grad_norm_value,
            "ema_action": ema_action,
            "micro_logs": micro_logs,
            "attempt_number": attempt_number,
            "compute_finished_at": compute_finished_at,
        }

    def _stage1_replicated_loader_generator_state(self) -> torch.Tensor:
        """Return the common loader RNG state after auditing all six ranks."""

        state = self.stage1_dataloader_generator.get_state().cpu().clone()
        digest = hashlib.sha256(state.numpy().tobytes()).hexdigest()
        digests = [None] * dist.get_world_size()
        dist.all_gather_object(digests, digest)
        if len(set(digests)) != 1:
            raise RuntimeError(
                "Stage-1 DataLoader generator state diverged across ranks: "
                f"{digests}"
            )
        return state

    def _save_stage1_fsdp2_checkpoint(self) -> None:
        """Collectively publish one adapter-only, fully resumable checkpoint."""

        from utils.stage1_checkpoint import (
            apply_stage1_resume_retention_collective,
            build_stage1_trainer_state,
            capture_rng_state,
            checkpoint_directory,
            finalize_stage1_checkpoint,
            gather_stage1_optimizer_state,
            gather_stage1_raw_and_ema_adapters,
            restore_rng_state,
            write_base_reference,
            write_stage1_adapter_pair,
            write_stage1_resume_payloads,
        )
        from utils.stage1_io import atomic_write_bytes

        if self.step <= 0 or not self.stage1_schedule.is_checkpoint_step(self.step):
            raise RuntimeError(
                f"Stage-1 checkpoint requested off schedule at completed step {self.step}"
            )
        if self.stage1_microbatch_cursor % self.gradient_accumulation_steps:
            raise RuntimeError("Stage-1 checkpoint is not on an accumulation boundary")
        if not self.generator_ema.initialized:
            raise RuntimeError(
                "Stage-1 adapter checkpoint requires initialized EMA; "
                f"completed_step={self.step}, ema_start={self.generator_ema.start_step}"
            )

        directory = checkpoint_directory(self.output_path, self.step)
        local_rng_state = self._stage1_world_checked_call(
            "per-rank checkpoint RNG capture",
            lambda: capture_rng_state(include_cuda=True),
        )
        try:
            def prepare_directory():
                if directory.exists():
                    entries = sorted(path.name for path in directory.iterdir())
                    if entries:
                        raise RuntimeError(
                            "refusing to overwrite a non-empty Stage-1 checkpoint "
                            f"directory {directory}: {entries}"
                        )
                directory.mkdir(parents=True, exist_ok=True)
                return str(directory.resolve())

            self._stage1_rank0_call(
                "checkpoint directory preflight", prepare_directory
            )
            barrier()

            adapters = gather_stage1_raw_and_ema_adapters(
                self.model.generator,
                self.generator_ema,
                expected_schema=self.stage1_lora_schema,
                authoritative_shard_group=self.stage1_authoritative_shard_group,
                replica_group=self.stage1_replica_group,
                expected_adapter_tensors=int(
                    self.config.adapter.expected_adapter_tensors
                ),
                expected_global_numel=int(
                    self.config.adapter.expected_trainable_parameters
                ),
            )
            write_stage1_adapter_pair(
                directory,
                adapters,
                expected_schema=self.stage1_lora_schema,
                completed_step=self.step,
                expected_adapter_tensors=int(
                    self.config.adapter.expected_adapter_tensors
                ),
                expected_global_numel=int(
                    self.config.adapter.expected_trainable_parameters
                ),
            )
            optimizer_state = gather_stage1_optimizer_state(
                self.model.generator,
                self.generator_optimizer,
                expected_schema=self.stage1_lora_schema,
                expected_completed_step=self.step,
            )
            loader_generator_state = (
                self._stage1_replicated_loader_generator_state()
            )

            def build_rank_zero_trainer_state():
                if not self.is_main_process:
                    return None
                if optimizer_state is None:
                    raise RuntimeError("rank 0 did not gather optimizer state")
                return build_stage1_trainer_state(
                    optimizer_state=optimizer_state,
                    completed_step=self.step,
                    global_epoch=self.stage1_epoch,
                    committed_microbatch_cursor_in_epoch=(
                        self.stage1_microbatch_cursor
                    ),
                    phase_derivation={
                        "schedule_sha256": self.stage1_schedule.resolved_hash(),
                        "schedule": self.stage1_schedule.to_dict(),
                        "cache_manifest_sha256": self.stage1_input_audit[
                            "cache_manifest_sha256"
                        ],
                        "run_id": self.stage1_logger.run_id,
                    },
                    sampler_state={
                        "schema_version": 1,
                        "seed": int(self.config.seed),
                        "global_epoch": self.stage1_epoch,
                        "committed_microbatch_cursor_in_epoch": (
                            self.stage1_microbatch_cursor
                        ),
                        "micro_batches_per_epoch": len(self.stage1_dataloader),
                        "data_parallel_size": self.data_parallel_size,
                        "resolution_aware": True,
                    },
                    dataloader_generator_state=loader_generator_state,
                    next_attempt_index=self.stage1_logger.next_attempt_index,
                    nonfinite_attempt_count=self.stage1_nonfinite_attempt_count,
                    resolved_config_sha256=self.stage1_resolved_config_sha256,
                )

            trainer_state = self._stage1_world_checked_call(
                "rank-0 trainer-state build", build_rank_zero_trainer_state
            )
            ema_state, error_buffer_state = self._stage1_world_checked_call(
                "rank-local EMA/error-buffer snapshot",
                lambda: (
                    self.generator_ema.state_dict(),
                    self.model.error_buffer.state_dict()
                    if dist.get_rank() < self.sequence_parallel_size
                    else None,
                ),
            )

            write_stage1_resume_payloads(
                directory,
                trainer_state=trainer_state,
                ema_state=ema_state,
                rng_state=local_rng_state,
                error_buffer_state=error_buffer_state,
            )

            def write_rank_zero_metadata():
                atomic_write_bytes(
                    directory / "resolved_config.yaml",
                    self.stage1_resolved_config_bytes,
                )
                return write_base_reference(
                    directory,
                    base_checkpoint=self.stage1_input_audit["base_checkpoint"],
                    base_manifest=self.stage1_input_audit["base_manifest"],
                    base_sha256=self.stage1_base_sha256,
                )

            self._stage1_rank0_call(
                "resolved config/base reference write", write_rank_zero_metadata
            )
            finalize_stage1_checkpoint(
                directory,
                completed_step=self.step,
                resumable=True,
            )
            apply_stage1_resume_retention_collective(
                self.output_path,
                keep_last=int(
                    self.config.checkpointing.keep_last_resumable
                ),
                keep_steps=tuple(
                    int(value)
                    for value in self.config.checkpointing.keep_resumable_steps
                ),
            )
            if self.is_main_process:
                print(f"[Stage-1] checkpoint committed: {directory}")
        finally:
            # Checkpoint I/O/collectives must be transparent to the next model
            # RNG stream, both after success and on a fail-fast exception.
            self._stage1_world_checked_call(
                "post-checkpoint RNG restoration",
                lambda: restore_rng_state(
                    local_rng_state, require_cuda_topology=True
                ),
            )
        barrier()

    def _train_stage1_fsdp2(self):
        console_every = int(self.config.console_every_steps)
        dry_run = bool(self.config.stage1_dry_run_one_update)
        try:
            while self.step < self.stage1_schedule.total_updates:
                update_index = self.step
                expected_epoch, expected_cursor, _ = self.stage1_schedule.epoch_position(
                    update_index * self.gradient_accumulation_steps
                )
                if (
                    self.stage1_epoch != expected_epoch
                    or self.stage1_microbatch_cursor != expected_cursor
                ):
                    raise RuntimeError(
                        "Stage-1 step/epoch/cursor drift: "
                        f"step={self.step}, actual=({self.stage1_epoch},"
                        f"{self.stage1_microbatch_cursor}), expected=({expected_epoch},"
                        f"{expected_cursor})"
                    )
                schedule_values = self.stage1_schedule.values_at(update_index)
                for group in self.generator_optimizer.param_groups:
                    group["lr"] = float(schedule_values.lr)

                cached_batches = None
                success = None
                update_epoch = self.stage1_epoch
                start_cursor = self.stage1_microbatch_cursor
                for attempt_number in range(1, self.stage1_max_attempts + 1):
                    torch.cuda.synchronize(self.device)
                    torch.cuda.reset_peak_memory_stats(self.device)
                    started_at = time.perf_counter()
                    if cached_batches is None:
                        cached_batches = self._stage1_fetch_update_batches()
                    result = self._run_stage1_update(
                        cached_batches,
                        schedule_values,
                        attempt_number=attempt_number,
                    )
                    timing = self._stage1_attempt_timing(
                        started_at,
                        result["compute_finished_at"],
                        cached_batches,
                    )
                    if not result["success"]:
                        self.stage1_nonfinite_attempt_count += 1
                        if self.is_main_process:
                            self.stage1_logger.append_attempt(
                                "nonfinite_attempt",
                                {
                                    "optimizer_step": update_index + 1,
                                    "update_index": update_index,
                                    "attempt_number_for_update": attempt_number,
                                    "phase": schedule_values.phase_name,
                                    "lr": schedule_values.lr,
                                    "reason": result["reason"],
                                    "detail": result["detail"],
                                    "nonfinite": True,
                                    "skipped": True,
                                    **timing,
                                },
                            )
                        if attempt_number == self.stage1_max_attempts:
                            raise FloatingPointError(
                                "Stage-1 update remained non-finite for two attempts: "
                                f"update_index={update_index}, detail={result['detail']}"
                            )
                        continue
                    success = result
                    break
                if success is None:
                    raise AssertionError("Stage-1 attempt loop exited without a result")

                losses, er_metrics = self._stage1_reduce_success_metrics(
                    success["micro_logs"]
                )
                buffer_metrics = self._stage1_buffer_metrics()
                dry_ema_smoke = False
                if dry_run:
                    self._stage1_dry_run_ema_smoke()
                    dry_ema_smoke = True
                epoch_progress = (
                    start_cursor + self.gradient_accumulation_steps
                ) / len(self.stage1_dataloader)
                record = {
                    "optimizer_step": self.step,
                    "update_index": update_index,
                    "epoch_index": update_epoch,
                    "epoch_progress": epoch_progress,
                    "phase": schedule_values.phase_name,
                    "phase_index": schedule_values.phase_index,
                    "ramp_u": schedule_values.ramp_u,
                    "ramp_s": schedule_values.ramp_s,
                    "lr": schedule_values.lr,
                    "loss_total": losses["loss_total"],
                    "loss_numerator": losses["loss_numerator"],
                    "loss_count": losses["loss_count"],
                    "blocks": losses["blocks"],
                    "preclip_grad_norm": success["grad_norm"],
                    "nonfinite": False,
                    "skipped": False,
                    "attempt_number_for_update": success["attempt_number"],
                    "ema_action": success["ema_action"],
                    "scheduled_active_probability": schedule_values.active_probability,
                    "scheduled_context_probability": schedule_values.effective_context_probability,
                    "scheduled_latent_probability": schedule_values.effective_latent_probability,
                    "scheduled_noise_probability": schedule_values.effective_noise_probability,
                    "error_recycling": er_metrics,
                    "error_buffer": buffer_metrics,
                    "dry_run": dry_run,
                    "dry_run_ema_component_check": dry_ema_smoke,
                    **timing,
                }
                if self.is_main_process:
                    self.stage1_logger.append_attempt("train_step", record)
                    if self.step % console_every == 0 or dry_run or self.step == 1:
                        print(
                            f"[Stage-1 step {self.step:06d}/"
                            f"{self.stage1_schedule.total_updates:06d}] "
                            f"phase={schedule_values.phase_name} "
                            f"loss={losses['loss_total']:.6f} "
                            f"grad={success['grad_norm']:.4f} "
                            f"lr={schedule_values.lr:.3e} "
                            f"sec={timing['step_seconds_max']:.3f}"
                        )
                if dry_run:
                    barrier()
                    return

                # Checkpoint integration is deliberately dispatched only on a
                # committed accumulation boundary and never from dry-run.
                if (
                    not self.config.no_save
                    and self.stage1_schedule.is_checkpoint_step(self.step)
                ):
                    self._save_stage1_fsdp2_checkpoint()
                barrier()
        finally:
            self.stage1_logger.close()

    def _move_optimizer_to_device(self, optimizer, device):
        """Move optimizer state to the specified device."""
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    def find_latest_checkpoint(self, logdir):
        """Find the latest checkpoint in the logdir."""
        if not os.path.exists(logdir):
            return None

        checkpoint_dirs = []
        for item in os.listdir(logdir):
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
                try:
                    # Extract step number from directory name
                    step_str = item.replace("checkpoint_model_", "")
                    step = int(step_str)
                    checkpoint_path = os.path.join(logdir, item, "model.pt")
                    if os.path.exists(checkpoint_path):
                        checkpoint_dirs.append((step, checkpoint_path))
                except ValueError:
                    continue
        
        if not checkpoint_dirs:
            return None
        
        # Sort by step number and return the latest one
        checkpoint_dirs.sort(key=lambda x: x[0])
        latest_step, latest_path = checkpoint_dirs[-1]
        return latest_path

    def get_all_checkpoints(self, logdir):
        """Get all checkpoints in the logdir sorted by step number."""
        if not os.path.exists(logdir):
            return []
        
        checkpoint_dirs = []
        for item in os.listdir(logdir):
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
                try:
                    # Extract step number from directory name
                    step_str = item.replace("checkpoint_model_", "")
                    step = int(step_str)
                    checkpoint_dir_path = os.path.join(logdir, item)
                    checkpoint_file_path = os.path.join(checkpoint_dir_path, "model.pt")
                    if os.path.exists(checkpoint_file_path):
                        checkpoint_dirs.append((step, checkpoint_dir_path, item))
                except ValueError:
                    continue
        
        # Sort by step number (ascending order)
        checkpoint_dirs.sort(key=lambda x: x[0])
        return checkpoint_dirs

    def cleanup_old_checkpoints(self, logdir, max_checkpoints):
        """Remove old checkpoints if the number exceeds max_checkpoints.
        
        Only the main process performs the actual deletion to avoid race conditions
        in distributed training.
        """
        if max_checkpoints <= 0:
            return
        
        # Only main process should perform cleanup to avoid race conditions
        if not self.is_main_process:
            return
            
        checkpoints = self.get_all_checkpoints(logdir)
        if len(checkpoints) > max_checkpoints:
            # Calculate how many to remove
            num_to_remove = len(checkpoints) - max_checkpoints
            checkpoints_to_remove = checkpoints[:num_to_remove]  # Remove oldest ones
            
            print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints, removing {num_to_remove} oldest ones (keeping {max_checkpoints})")
            
            import shutil
            removed_count = 0
            for step, checkpoint_dir_path, dir_name in checkpoints_to_remove:
                try:
                    print(f"  Removing: {dir_name} (step {step})")
                    shutil.rmtree(checkpoint_dir_path)
                    removed_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to remove checkpoint {dir_name}: {e}")
            
            print(f"Checkpoint cleanup completed: removed {removed_count}/{num_to_remove} old checkpoints")
        else:
            if len(checkpoints) > 0:
                print(f"Checkpoint cleanup: Found {len(checkpoints)} checkpoints (max: {max_checkpoints}, no cleanup needed)")

    def save(self):
        print("Start gathering distributed model states...")

        # Release large inference caches before saving when possible.
        if hasattr(self.model, "inference_pipeline") and self.model.inference_pipeline is not None:
            clear_fn = getattr(self.model.inference_pipeline, "clear_cache", None)
            if clear_fn is not None:
                try:
                    clear_fn()
                except Exception as e:
                    print(f"Warning: failed to clear inference cache before save: {e}")
            # Drop the inference pipeline reference so GC / empty_cache can
            # reclaim memory.
            self.model.inference_pipeline = None
            torch.cuda.empty_cache()
        
        with FSDP.state_dict_type(
            self.model.generator,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            generator_state_dict  = self.model.generator.state_dict()
            generator_opim_state_dict = FSDP.optim_state_dict(self.model.generator,
                                            self.generator_optimizer)

        if self.config.ema_start_step < self.step and self.generator_ema is not None:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.state_dict(),
                "generator_optimizer": generator_opim_state_dict,
                "step": self.step,
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
                "generator_optimizer": generator_opim_state_dict,
                "step": self.step,
            }

        checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        if self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, "model.pt")
            torch.save(state_dict, checkpoint_file)
            print("Model saved to", checkpoint_file)

        # Save error buffer — unified per-sp_rank pattern:
        #   Each SP rank owns a different t-bucket shard (and different
        #   positions in 2D mode).  The first DP rank in each SP group
        #   writes ``error_buffer_sp{sp_rank}.pt``.
        #   Fallback (sp_size<=1): main_process writes ``error_buffer.pt``.
        if self.model.error_buffer is not None:
            sp_size_ = max(self.sequence_parallel_size, 1)
            _global_rank = dist.get_rank() if dist.is_initialized() else 0
            _sp_rank = _global_rank % sp_size_
            _is_first_dp = (_global_rank // sp_size_) == 0

            if dist.is_initialized():
                dist.barrier()

            should_save = _is_first_dp if sp_size_ > 1 else self.is_main_process
            if should_save:
                for stem, buffer in [("error_buffer", self.model.error_buffer),
                                     ("noise_error_buffer", self.model.noise_error_buffer)]:
                    if buffer is None:
                        continue
                    fname = f"{stem}_sp{_sp_rank}.pt" if sp_size_ > 1 else f"{stem}.pt"
                    fpath = os.path.join(checkpoint_dir, fname)
                    torch.save(buffer.state_dict(), fpath)
                    s = buffer.stats()
                    rng = s.get('global_block_range', '')
                    shard = s.get('shard', '')
                    print(f"[rank={_global_rank}] {stem} saved to {fname} "
                          f"({s['total_entries']} entries, {s['filled_buckets']} buckets) "
                          f"{rng} {shard}".rstrip())

        if self.is_main_process:
            # Cleanup old checkpoints if max_checkpoints is set
            max_checkpoints = getattr(self.config, "max_checkpoints", 0)
            if max_checkpoints > 0:
                self.cleanup_old_checkpoints(self.output_path, max_checkpoints)

        # Keep all ranks in sync so non-rank0 workers don't kick off the next
        # training iteration (and trigger NCCL watchdog timeouts) while rank0
        # is still writing the checkpoint to disk.
        if dist.is_initialized():
            dist.barrier()

        torch.cuda.empty_cache()
        import gc
        gc.collect()

    def train_one_step(self, batch, accumulation_step=0, accumulation_steps=None):
        accumulation_steps = accumulation_steps or getattr(self, "gradient_accumulation_steps", 1)
        self.log_iters = 1

        if self.step % 20 == 0:
            torch.cuda.empty_cache()
        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)
        clean_latent_is_sp_sharded = False
        if not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["ode_latent"][:, -1].to(
                device=self.device, dtype=self.dtype)
            image_latent = clean_latent[:, 0:1]
        else:  # encode raw video to latent
            (
                clean_latent,
                image_latent,
                clean_latent_is_sp_sharded,
            ) = self.sp_helper.encode_raw_video_latents(
                batch,
                batch_size=batch_size,
            )

        loss_mask = self.sp_helper.build_loss_mask(
            batch, clean_latent, clean_latent_is_sp_sharded
        )
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        # Step 2: Extract the conditional infos
        with torch.no_grad():
            # turn text prompts: List[List[str]] into List[str]
            text_prompts_flat = [prompt for sublist in text_prompts for prompt in sublist]

            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts_flat)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 2.5: Sequence Parallel partitions sequence-owned tensors.
        if self.sequence_parallel_size > 1:
            clean_latent, conditional_dict, image_or_video_shape = (
                self.sp_helper.partition_training_inputs(
                    image_or_video_shape=image_or_video_shape,
                    clean_latent=clean_latent,
                    conditional_dict=conditional_dict,
                    clean_latent_is_sharded=clean_latent_is_sp_sharded,
                )
            )
            image_latent = self.sp_helper.local_i2v_initial_latent(image_latent)
        loss_mask, loss_mask_global_valid_count = self.sp_helper.partition_loss_mask(
            loss_mask,
            already_sharded=clean_latent_is_sp_sharded,
        )

        # Step 3: Train the generator
        gen_kwargs = dict(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent,
            loss_mask=loss_mask,
            loss_mask_global_valid_count=loss_mask_global_valid_count,
            global_step=self.step,
        )
        generator_loss, log_dict = self.model.generator_loss(**gen_kwargs)
        if accumulation_step == 0:
            self.generator_optimizer.zero_grad(set_to_none=True)
        scaled_loss = generator_loss / accumulation_steps
        scaled_loss.backward()
        if accumulation_step == accumulation_steps - 1:
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm)

            self.generator_optimizer.step()
            self.step += 1
        else:
            generator_grad_norm = torch.tensor(0.0, device=self.device)

        # Run the remaining logic only after a full gradient-accumulation cycle.
        if accumulation_step != accumulation_steps - 1:
            return

        # Step 4: Update EMA (if enabled and after start step)
        if (self.step >= self.config.ema_start_step) and \
                (self.generator_ema is None) and \
                (getattr(self.config, "ema_weight", None) is not None) and \
                (self.config.ema_weight > 0):
            self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

        # Update EMA after optimizer step
        if self.generator_ema is not None and self.step >= self.config.ema_start_step:
            self.generator_ema.update(self.model.generator)

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }

        # Error buffer stats
        er_log_str = ""
        if "er_total_added" in log_dict:
            wandb_loss_dict["er_total_entries"] = log_dict["er_total_entries"]
            wandb_loss_dict["er_total_added"] = log_dict["er_total_added"]
            wandb_loss_dict["er_injected"] = int(log_dict["er_injected"])
            wandb_loss_dict["er_latent_injected"] = int(log_dict["er_latent_injected"])
            wandb_loss_dict["er_noise_injected"] = int(log_dict.get("er_noise_injected", False))
            wandb_loss_dict["er_noise_total_entries"] = log_dict.get("er_noise_total_entries", 0)
            ctx_flag = 'Y' if log_dict['er_injected'] else 'N'
            lat_flag = 'Y' if log_dict['er_latent_injected'] else 'N'
            noise_flag = 'Y' if log_dict.get('er_noise_injected', False) else 'N'
            er_log_str = (
                f", er_buf={log_dict['er_total_entries']}|"
                f"{log_dict.get('er_noise_total_entries', 0)} "
                f"({log_dict['er_filled_buckets']} buckets), "
                f"ctx={ctx_flag} lat={lat_flag} noise={noise_flag}"
            )

        # Step 5: Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)
            print(
                f"[step {self.step:07d}] "
                f"generator_loss={wandb_loss_dict['generator_loss']:.6f}, "
                f"generator_grad_norm={wandb_loss_dict['generator_grad_norm']:.6f}"
                f"{er_log_str}"
            )

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

    def _set_sp_attn(self, enabled: bool):
        """
        Toggle SP self-attention between training and inference.
        This only applies to 5B runs with SP enabled.
        """
        if not hasattr(self, "_sp_attn_blocks"):
            return
        if self.sequence_parallel_size <= 1:
            return

        # Lazy import to avoid failures under non-5B configurations.
        try:
            from wan_5b.distributed.sequence_parallel import sp_causal_attn_forward
        except Exception:
            return

        for sa in self._sp_attn_blocks:
            if not hasattr(sa, "_orig_forward"):
                continue
            if enabled:
                sa.forward = types.MethodType(sp_causal_attn_forward, sa)
            else:
                sa.forward = sa._orig_forward

    @torch.no_grad()
    def _swap_ema_weights(self):
        """
        Bidirectionally swap model weights with EMA shadow weights.
        Calling this twice restores both the model and EMA to their original state.
        """
        with FSDP.summon_full_params(self.model.generator, writeback=True):
            for n, p in self.model.generator.module.named_parameters():
                cleaned_name = EMA_FSDP._clean_param_name(n)
                if cleaned_name in self.generator_ema.shadow:
                    ema_val = self.generator_ema.shadow[cleaned_name]
                    tmp = p.data.clone().float().cpu()
                    p.data.copy_(ema_val.to(dtype=p.dtype, device=p.device))
                    self.generator_ema.shadow[cleaned_name] = tmp

    def _run_evaluation_inference(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        if self.model.inference_pipeline is None:
            self.model._initialize_inference_pipeline()

        out_dir = os.path.join(self.output_path, f"generated_video_{self.step:06d}")
        if self.is_main_process:
            os.makedirs(out_dir, exist_ok=True)
        barrier()

        rank = dist.get_rank()
        vis_ema = section_get(self.config, "evaluation", "use_ema", getattr(self.config, "vis_ema", False))
        vis_ema = vis_ema and self.generator_ema is not None

        for eval_batch in self.eval_dataloader:
            eval_prompts = eval_batch["prompts"]
            eval_idx = eval_batch["idx"]
            eval_images = eval_batch.get("image", None)

            batch_size_eval = len(eval_prompts)
            for b in range(batch_size_eval):
                prompts_for_sample = eval_prompts[b]

                if self.is_main_process:
                    print(f"prompts_for_sample: {prompts_for_sample}")
                    print(len(prompts_for_sample))
                    print(prompts_for_sample[0][:60])

                sample_idx = (
                    eval_idx[b].item()
                    if hasattr(eval_idx, "shape")
                    else int(eval_idx[b])
                )

                save_latents_only = section_get(
                    self.config,
                    "evaluation",
                    "save_latents_only",
                    self.config.get("return_latents", False),
                    aliases=("return_latents", "save_latent_only"),
                )

                run_modes = [("", False)]
                if vis_ema:
                    run_modes.append(("_ema", True))

                for suffix, use_ema in run_modes:
                    generated_video = self.generate_video(
                        self.model.inference_pipeline,
                        [prompts_for_sample],
                        eval_images[b:b + 1] if eval_images is not None else None,
                        use_ema=use_ema,
                    )

                    if not save_latents_only:
                        from torchvision.io import write_video

                        video_path = os.path.join(
                            out_dir,
                            f"video{suffix}_rank{rank:02d}_idx{sample_idx:06d}.mp4",
                        )
                        write_video(video_path, generated_video[0], fps=self.fps)
                    else:
                        video_path = os.path.join(
                            out_dir,
                            f"latents{suffix}_rank{rank:02d}_idx{sample_idx:06d}.pt",
                        )
                        torch.save(generated_video[0], video_path)

                    if (not self.disable_wandb) and self.is_main_process and not save_latents_only:
                        caption = prompts_for_sample[0] if len(prompts_for_sample) > 0 else ""
                        log_key = f"generated_video{suffix}"
                        wandb.log(
                            {
                                log_key: wandb.Video(
                                    generated_video[0].transpose(0, 3, 1, 2),
                                    caption=f"{caption}",
                                    fps=self.fps,
                                    format="mp4",
                                ),
                            },
                            step=self.step,
                        )

                    del generated_video

                prompt_txt_path = os.path.join(
                    out_dir,
                    f"prompt_rank{rank:02d}_idx{sample_idx:06d}.txt",
                )
                save_prompts_to_txt(
                    prompts_for_sample,
                    prompt_txt_path,
                    self.is_main_process,
                )
        barrier()

        if hasattr(self.model, "inference_pipeline") and self.model.inference_pipeline is not None:
            clear_fn = getattr(self.model.inference_pipeline, "clear_cache", None)
            if clear_fn is not None:
                clear_fn()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def generate_video(self, pipeline, prompts, image=None, use_ema=False):
        # Temporarily disable SP self-attention during inference to avoid
        # interfering with KV-cache logic.
        self._set_sp_attn(False)
        ema_applied = use_ema and self.generator_ema is not None
        if ema_applied:
            self._swap_ema_weights()
        try:
            batch_size = len(prompts)
            noise_shape = list(self.config.image_or_video_shape[1:])
            inference_num_frames = section_get(
                self.config, "evaluation", "num_frames", getattr(self.config, "inference_num_frames", 0)
            )
            if isinstance(inference_num_frames, (list, tuple)):
                inference_num_frames = inference_num_frames[0] if len(inference_num_frames) > 0 else 0
            if inference_num_frames > 0:
                noise_shape[0] = inference_num_frames
            initial_latent = None
            if image is not None:
                image = image.to(device="cuda", dtype=self.dtype)
                if image.ndim == 4:
                    image = image.unsqueeze(2)
                elif image.ndim != 5:
                    raise ValueError(f"Expected i2v image with shape [B,C,H,W] or [B,C,T,H,W], got {tuple(image.shape)}")
                initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=self.dtype)
                if initial_latent.shape[0] != batch_size:
                    initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
                if noise_shape[0] <= initial_latent.shape[1]:
                    raise ValueError(
                        f"evaluation.num_frames must exceed the i2v conditioning frames; "
                        f"got {inference_num_frames} and {initial_latent.shape[1]}"
                    )
            sampled_noise = torch.randn(
                [batch_size] + noise_shape, device="cuda", dtype=self.dtype
            )

            save_latents_only = section_get(
                self.config,
                "evaluation",
                "save_latents_only",
                self.config.get("return_latents", False),
                aliases=("return_latents", "save_latent_only"),
            )
            video = pipeline.inference(
                noise=sampled_noise,
                text_prompts=prompts,
                initial_latent=initial_latent,
                return_latents=save_latents_only
            )
            if not save_latents_only:
                current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
            else:
                current_video = video
        finally:
            if ema_applied:
                self._swap_ema_weights()
            # Restore SP self-attention for training.
            self._set_sp_attn(True)

        return current_video

    def _sync_batch_for_sequence_parallel(self, batch, accumulation_step: int = 0):
        return self.sp_helper.sync_batch(batch, step=self.step)

    def train(self):
        if getattr(self, "stage1_mode", False):
            return self._train_stage1_fsdp2()

        if getattr(self.config, "generate_before_train", False):
            if self.is_main_process:
                print("[generate_before_train] Running evaluation inference before training starts...")
            self._run_evaluation_inference()
            if self.is_main_process:
                print("[generate_before_train] Inference done. Exiting.")
            barrier()
            return

        acc_steps = getattr(self, "gradient_accumulation_steps", 1)
        while True:
            for acc in range(acc_steps):
                batch = next(self.dataloader)

                # Synchronize batch contents across ranks under Sequence Parallel.
                if self.sequence_parallel_size > 1:
                    batch = self._sync_batch_for_sequence_parallel(batch, accumulation_step=acc)

                self.train_one_step(batch, accumulation_step=acc, accumulation_steps=acc_steps)
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            evaluation_interval = section_get(self.config, "evaluation", "interval", getattr(self.config, "generate_interval", 0))
            if evaluation_interval > 0 and self.step % evaluation_interval == 0:
                self._run_evaluation_inference()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

            if self.step >= self.config.max_iters:
                break
