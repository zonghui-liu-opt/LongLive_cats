import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from PIL import Image
from safetensors.torch import save_file

import wan_5b.textimage2video as ti2v_module
from scripts.infer_wan22_ti2v_batch import (
    Sample,
    make_batches,
    validate_checkpoint_dir,
)
from wan_5b.distributed import sequence_parallel as sp_module
from wan_5b.modules.model import WanModel, rope_apply, rope_params


class _FakeScheduler:
    def __init__(self, **_kwargs):
        self.timesteps = None

    def set_timesteps(self, _steps, device, shift):
        del shift
        self.timesteps = torch.tensor([1.0], device=device)

    def step(self, _prediction, _timestep, sample, **_kwargs):
        return (sample,)


class _Movable:
    def to(self, *_args, **_kwargs):
        return self

    def cpu(self):
        return self


class _FakeTextEncoder:
    def __init__(self):
        self.model = _Movable()
        self.calls = []

    def __call__(self, texts, device):
        self.calls.append(list(texts))
        return [torch.ones(2, 4, device=device) for _ in texts]


class _FakeVAEModel:
    z_dim = 2


class _FakeVAE:
    def __init__(self):
        self.model = _FakeVAEModel()

    def encode(self, videos):
        return [
            torch.zeros(2, 1, video.shape[-2] // 16, video.shape[-1] // 16)
            for video in videos
        ]

    def decode(self, latents):
        return [latent.clone() for latent in latents]


class _FakeWanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, values, **_kwargs):
        self.batch_sizes.append(len(values))
        return torch.stack([torch.zeros_like(value) for value in values])


def _make_pipeline():
    pipe = ti2v_module.WanTI2V.__new__(ti2v_module.WanTI2V)
    pipe.device = torch.device("cpu")
    pipe.rank = 0
    pipe.t5_cpu = True
    pipe.init_on_cpu = False
    pipe.use_sp = False
    pipe.sp_size = 1
    pipe.num_train_timesteps = 1000
    pipe.param_dtype = torch.bfloat16
    pipe.vae_stride = (4, 16, 16)
    pipe.patch_size = (1, 2, 2)
    pipe.sample_neg_prompt = "negative"
    pipe.text_encoder = _FakeTextEncoder()
    pipe.vae = _FakeVAE()
    pipe.model = _FakeWanModel()
    return pipe


class WanTI2VBatchTest(unittest.TestCase):
    def test_euler_uses_diffsynth_wan_shifted_schedule(self):
        scheduler, timesteps = ti2v_module._prepare_sampling_scheduler(
            sample_solver="euler",
            sampling_steps=4,
            shift=3.0,
            num_train_timesteps=1000,
            device=torch.device("cpu"),
        )

        torch.testing.assert_close(
            timesteps, torch.tensor([1000.0, 900.0, 750.0, 500.0]))
        torch.testing.assert_close(
            scheduler.sigmas,
            torch.tensor([1.0, 0.9, 0.75, 0.5, 0.0]),
        )

    def test_explicit_empty_negative_prompt_does_not_use_model_default(self):
        pipe = _make_pipeline()
        image = Image.new("RGB", (32, 32), "white")
        with mock.patch.object(
                ti2v_module.torch.amp,
                "autocast",
                side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(),
        ):
            pipe.i2v_batch(
                input_prompts=["positive"],
                imgs=[image],
                max_area=32 * 32,
                frame_num=5,
                sample_solver="euler",
                sampling_steps=1,
                n_prompts="",
                seeds=[1],
                offload_model=False,
            )

        self.assertEqual(pipe.text_encoder.calls, [["positive"], [""]])

    def test_configless_flat_checkpoint_loads_strictly(self):
        config = SimpleNamespace(
            model_type="ti2v",
            patch_size=(1, 2, 2),
            text_len=8,
            in_dim=4,
            dim=16,
            ffn_dim=32,
            freq_dim=8,
            text_dim=12,
            out_dim=4,
            num_heads=2,
            num_layers=1,
            window_size=(-1, -1),
            qk_norm=True,
            cross_attn_norm=True,
            eps=1e-6,
        )
        source = WanModel(**ti2v_module._wan_model_kwargs(config))
        source_state = source.state_dict()
        for sharded in (False, True):
            with self.subTest(sharded=sharded), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                if sharded:
                    keys = list(source_state)
                    midpoint = len(keys) // 2
                    shard_names = (
                        "diffusion_pytorch_model-00001-of-00002.safetensors",
                        "diffusion_pytorch_model-00002-of-00002.safetensors",
                    )
                    save_file(
                        {key: source_state[key] for key in keys[:midpoint]},
                        root / shard_names[0],
                    )
                    save_file(
                        {key: source_state[key] for key in keys[midpoint:]},
                        root / shard_names[1],
                    )
                    weight_map = {
                        key: shard_names[int(index >= midpoint)]
                        for index, key in enumerate(keys)
                    }
                    (root / "diffusion_pytorch_model.safetensors.index.json").write_text(
                        json.dumps({"weight_map": weight_map}), encoding="utf-8")
                else:
                    save_file(
                        source_state,
                        root / "diffusion_pytorch_model.safetensors",
                    )
                loaded = ti2v_module._load_wan_model(directory, config)

            self.assertFalse(any(
                parameter.is_meta for parameter in loaded.parameters()))
            for key, expected in source_state.items():
                torch.testing.assert_close(loaded.state_dict()[key], expected)

    def test_checkpoint_validation_allows_merged_layout_without_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "diffusion_pytorch_model.safetensors").touch()
            (root / "Wan2.2_VAE.pth").touch()
            (root / "models_t5_umt5-xxl-enc-bf16.pth").touch()
            (root / "google" / "umt5-xxl").mkdir(parents=True)

            validate_checkpoint_dir(root)

    def test_checkpoint_validation_supports_separate_auxiliary_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dit_root = root / "merged"
            aux_root = root / "base"
            dit_root.mkdir()
            (dit_root / "diffusion_pytorch_model.safetensors").touch()
            aux_root.mkdir()
            (aux_root / "Wan2.2_VAE.pth").touch()
            (aux_root / "models_t5_umt5-xxl-enc-bf16.pth").touch()
            (aux_root / "google" / "umt5-xxl").mkdir(parents=True)

            validate_checkpoint_dir(dit_root, aux_root)

    def test_resolution_bucketing_never_mixes_orientations(self):
        samples = [
            Sample(i, Path(f"{i}.png"), "prompt", height, width, "")
            for i, (height, width) in enumerate([
                (480, 832), (480, 832), (832, 480),
                (480, 832), (832, 480), (832, 480),
            ])
        ]
        batches = make_batches(samples, batch_size=3)

        self.assertEqual([[item.index for item in batch] for batch in batches],
                         [[0, 1, 3], [2, 4, 5]])
        self.assertTrue(all(
            len({(item.height, item.width) for item in batch}) == 1
            for batch in batches
        ))

    def test_batch_uses_one_model_call_per_cfg_branch(self):
        pipe = _make_pipeline()
        images = [Image.new("RGB", (32, 32), "white") for _ in range(2)]
        with (
            mock.patch.object(
                ti2v_module, "FlowUniPCMultistepScheduler", _FakeScheduler),
            mock.patch.object(
                ti2v_module.torch.amp,
                "autocast",
                side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(),
            ),
        ):
            videos = pipe.i2v_batch(
                input_prompts=["a", "b"],
                imgs=images,
                max_area=32 * 32,
                frame_num=5,
                sampling_steps=1,
                seeds=[11, 12],
                offload_model=False,
            )

        self.assertEqual(len(videos), 2)
        self.assertEqual(pipe.model.batch_sizes, [2, 2])

    def test_per_sample_seed_is_independent_of_batching(self):
        pipe = _make_pipeline()
        image = Image.new("RGB", (32, 32), "white")
        patches = (
            mock.patch.object(
                ti2v_module, "FlowUniPCMultistepScheduler", _FakeScheduler),
            mock.patch.object(
                ti2v_module.torch.amp,
                "autocast",
                side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(),
            ),
        )
        with patches[0], patches[1]:
            batched = pipe.i2v_batch(
                input_prompts=["a", "b"],
                imgs=[image, image],
                max_area=32 * 32,
                frame_num=5,
                sampling_steps=1,
                seeds=[11, 12],
                offload_model=False,
            )

        pipe = _make_pipeline()
        with (
            mock.patch.object(
                ti2v_module, "FlowUniPCMultistepScheduler", _FakeScheduler),
            mock.patch.object(
                ti2v_module.torch.amp,
                "autocast",
                side_effect=lambda *_args, **_kwargs: contextlib.nullcontext(),
            ),
        ):
            single = pipe.i2v(
                input_prompt="b",
                img=image,
                max_area=32 * 32,
                frame_num=5,
                sampling_steps=1,
                seed=12,
                offload_model=False,
            )

        torch.testing.assert_close(batched[1], single)

    def test_sequence_parallel_rope_matches_global_rope_with_padding(self):
        torch.manual_seed(0)
        full = torch.randn(1, 12, 2, 12)
        grid_sizes = torch.tensor([[2, 2, 2]])  # 8 valid + 4 padded tokens
        head_dim = full.size(-1)
        freqs = torch.cat([
            rope_params(32, head_dim - 4 * (head_dim // 6)),
            rope_params(32, 2 * (head_dim // 6)),
            rope_params(32, 2 * (head_dim // 6)),
        ], dim=1)
        expected = rope_apply(full, grid_sizes, freqs)

        chunks = []
        for rank, local in enumerate(full.chunk(3, dim=1)):
            with (
                mock.patch.object(sp_module, "get_world_size", return_value=3),
                mock.patch.object(sp_module, "get_rank", return_value=rank),
            ):
                chunks.append(sp_module.sp_rope_apply(
                    local, grid_sizes, freqs, global_token_shard=True))

        actual = torch.cat(chunks, dim=1)
        torch.testing.assert_close(actual, expected)

    def test_frame_sharded_rope_keeps_training_semantics(self):
        torch.manual_seed(1)
        full = torch.randn(1, 24, 2, 12)
        local_grid_sizes = torch.tensor([[2, 2, 2]])
        global_grid_sizes = torch.tensor([[6, 2, 2]])
        head_dim = full.size(-1)
        freqs = torch.cat([
            rope_params(32, head_dim - 4 * (head_dim // 6)),
            rope_params(32, 2 * (head_dim // 6)),
            rope_params(32, 2 * (head_dim // 6)),
        ], dim=1)
        expected = rope_apply(full, global_grid_sizes, freqs)

        chunks = []
        for rank, local in enumerate(full.chunk(3, dim=1)):
            with (
                mock.patch.object(sp_module, "get_world_size", return_value=3),
                mock.patch.object(sp_module, "get_rank", return_value=rank),
            ):
                chunks.append(sp_module.sp_rope_apply(
                    local, local_grid_sizes, freqs))

        actual = torch.cat(chunks, dim=1)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
