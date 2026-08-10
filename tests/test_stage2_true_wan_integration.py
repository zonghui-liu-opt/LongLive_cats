from __future__ import annotations

import pytest
import torch


def test_true_bf16_wan_score_adapter_preserves_fp32_noising_and_token_time(
    monkeypatch,
):
    """Exercise the Stage-2 score adapter through a real tiny WanModel."""

    import wan_5b.modules.model as wan_model_module
    from model.stage2_dmd import Stage2DiTRole
    from utils.stage2_i2v_conditioning import pack_stage2_i2v_score_input
    from utils.stage2_score_math import noise_stage2_score_input
    from wan_5b.modules.model import WanModel

    # Keep the real Wan projections/patch embedding/time path while replacing
    # only quadratic attention, which is unrelated to this adapter contract.
    monkeypatch.setattr(
        wan_model_module,
        "flash_attention",
        lambda q, k, v, **_kwargs: torch.zeros_like(q),
    )
    model = WanModel(
        model_type="ti2v",
        patch_size=(1, 2, 2),
        text_len=2,
        in_dim=48,
        out_dim=48,
        dim=12,
        ffn_dim=24,
        freq_dim=8,
        text_dim=4,
        num_heads=1,
        num_layers=1,
        qk_norm=False,
        cross_attn_norm=False,
    ).to(dtype=torch.bfloat16)
    role = Stage2DiTRole(model, role="fake_score", is_causal=False)

    captured: dict[str, torch.Tensor] = {}

    def capture_true_wan_inputs(_module, args, kwargs):
        captured["latent"] = args[0].detach()
        captured["token_timestep"] = kwargs["t"].detach()

    model.register_forward_pre_hook(capture_true_wan_inputs, with_kwargs=True)
    initial = torch.zeros(1, 1, 48, 30, 52, dtype=torch.bfloat16)
    future = torch.ones(1, 24, 48, 30, 52, dtype=torch.bfloat16)
    clean_score = pack_stage2_i2v_score_input(initial, future)
    frame_timestep = torch.cat(
        (torch.zeros(1, 1), torch.full((1, 24), 317.25)),
        dim=1,
    )
    noised = noise_stage2_score_input(
        clean_score,
        frame_timestep,
        future_epsilon=torch.zeros_like(future, dtype=torch.float32),
    )
    assert noised.noisy_score.dtype == torch.float32

    with torch.no_grad():
        raw_flow, x0_pred = role.forward_score(
            noisy_image_or_video=noised.noisy_score,
            conditional_dict={
                "prompt_embeds": torch.zeros(1, 2, 4, dtype=torch.bfloat16)
            },
            frame_timestep=frame_timestep,
        )

    assert captured["latent"].shape == (1, 48, 25, 30, 52)
    assert captured["latent"].dtype == torch.bfloat16
    token_timestep = captured["token_timestep"]
    assert token_timestep.shape == (1, 9_750)
    assert token_timestep.dtype == torch.float32
    assert torch.count_nonzero(token_timestep[:, :390]).item() == 0
    assert torch.equal(
        token_timestep[:, 390:],
        torch.full((1, 9_360), 317.25, dtype=torch.float32),
    )
    assert raw_flow.shape == noised.noisy_score.shape
    assert raw_flow.dtype == torch.bfloat16
    assert x0_pred.shape == noised.noisy_score.shape
    assert x0_pred.dtype == torch.float32
    expected_x0 = (
        noised.noisy_score
        - (frame_timestep.view(1, 25, 1, 1, 1) / 1000.0) * raw_flow.float()
    )
    assert torch.equal(x0_pred, expected_x0)
    assert bool(torch.isfinite(x0_pred).all())


def test_true_peft_causal_wan_unipc_rollout_commits_only_detached_clean_kv(
    monkeypatch,
):
    """Exercise real PEFT proxies, Wan deferred KV, and the UniPC rollout."""

    # Importing the module does not import Triton kernels. Doing this before
    # the scoped environment override lets monkeypatch restore the module's
    # actual process default after the test, even when this is its first import.
    import wan_5b.modules.causal_model as causal_model_module

    # These environment changes are scoped to this test by pytest's
    # monkeypatch fixture. H100 keeps the production defaults (both enabled).
    monkeypatch.setenv("LLV2_TRITON_ADALN", "0")
    monkeypatch.setenv("LLV2_TRITON_ROPE", "0")

    from model.stage2_dmd import Stage2DiTRole
    from peft import LoraConfig, get_peft_model
    from pipeline.stage2_rollout import (
        STAGE2_K4_SHIFT5_TIMESTEPS,
        Stage2RolloutPipeline,
    )
    from wan_5b.modules.causal_model import CausalWanModel

    # The module may already have been imported by another test file, so also
    # scope its cached feature flags explicitly. This does not reload or alter
    # production defaults.
    monkeypatch.setattr(causal_model_module, "_TRITON_ADALN_ENABLED", False)
    monkeypatch.setattr(causal_model_module, "_TRITON_ROPE_ENABLED", False)
    monkeypatch.setattr(
        causal_model_module,
        "causal_rope_apply",
        lambda value, *_args, **_kwargs: value,
    )
    monkeypatch.setattr(
        causal_model_module,
        "attention",
        lambda query, _key, _value: query,
    )
    monkeypatch.setattr(
        causal_model_module,
        "flash_attention",
        lambda query, _key, _value, **_kwargs: query,
    )

    committed_global_ends: list[int] = []
    original_apply_cache_updates = CausalWanModel._apply_cache_updates

    def count_apply_cache_updates(self, kv_cache, cache_update_infos):
        assert len(cache_update_infos) == 1
        committed_global_ends.append(int(cache_update_infos[0][1][0]))
        return original_apply_cache_updates(self, kv_cache, cache_update_infos)

    monkeypatch.setattr(
        CausalWanModel,
        "_apply_cache_updates",
        count_apply_cache_updates,
    )

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        base_model = CausalWanModel(
            model_type="ti2v",
            patch_size=(1, 2, 2),
            text_len=2,
            in_dim=48,
            out_dim=48,
            dim=12,
            ffn_dim=24,
            freq_dim=8,
            text_dim=4,
            num_heads=1,
            num_layers=1,
            local_attn_size=17,
            sink_size=1,
            qk_norm=False,
            cross_attn_norm=False,
        ).to(dtype=torch.bfloat16)
        peft_model = get_peft_model(
            base_model,
            LoraConfig(
                r=2,
                lora_alpha=2,
                lora_dropout=0.0,
                bias="none",
                target_modules=["blocks.0.self_attn.q"],
            ),
            autocast_adapter_dtype=True,
        )
        # Fresh Wan constructors intentionally zero the diffusion head. Real
        # checkpoints do not; use a deterministic nonzero head so this tiny
        # integration test can require a nonzero LoRA gradient.
        with torch.no_grad():
            base_model.head.head.weight.fill_(0.01)

        assert tuple(peft_model.patch_size) == (1, 2, 2)
        assert peft_model.patch_embedding is base_model.patch_embedding
        assert callable(peft_model._apply_cache_updates)
        trainable = [
            (name, parameter)
            for name, parameter in peft_model.named_parameters()
            if parameter.requires_grad
        ]
        assert len(trainable) == 2
        assert all(parameter.dtype == torch.float32 for _, parameter in trainable)

        cross_k_calls: list[torch.Size] = []

        def record_cross_k(_module, args, _output):
            cross_k_calls.append(args[0].shape)

        base_model.blocks[0].cross_attn.k.register_forward_hook(record_cross_k)
        role = Stage2DiTRole(peft_model, role="generator", is_causal=True)
        pipeline = Stage2RolloutPipeline(role)
        initial = torch.zeros(1, 1, 48, 30, 52, dtype=torch.bfloat16)
        noise = torch.full(
            (1, 24, 48, 30, 52),
            0.25,
            dtype=torch.bfloat16,
        )
        result, state = pipeline.rollout(
            initial_latent=initial,
            noise=noise,
            conditional_dict={
                "prompt_embeds": torch.ones(1, 2, 4, dtype=torch.bfloat16)
            },
            exit_step=1,
            requires_grad=True,
        )

        assert result.latents.shape == noise.shape
        assert result.latents.dtype == torch.bfloat16
        assert result.latents.requires_grad
        assert result.scheduler_timesteps == STAGE2_K4_SHIFT5_TIMESTEPS
        assert result.scheduler_sigmas[-1] == 0.0
        assert result.scheduler_sigmas[0] == pytest.approx(
            0.9997998476028442, rel=0.0, abs=1.0e-8
        )
        assert result.scheduler_sigmas[0] != 999.0 / 1000.0
        assert result.chunk_timesteps == ((999, 937),) * 3
        assert result.cache_audit["noisy_forward_calls"] == 6
        assert result.cache_audit["clean_recache_forward_calls"] == 3
        assert result.cache_audit["sink_preload_forward_calls"] == 1
        assert result.cache_audit["generator_forward_calls"] == 10

        # Exactly sink + three clean recaches commit. The three exit/noisy
        # forwards return deferred updates that are discarded.
        assert committed_global_ends == [390, 3_510, 6_630, 9_750]
        self_cache = state.self_kv[0]
        assert self_cache["k"].shape[1] == 17 * 390
        assert int(self_cache["global_end_index"].item()) == 25 * 390
        assert int(self_cache["local_end_index"].item()) == 17 * 390
        assert self_cache["k"].grad_fn is None
        assert self_cache["v"].grad_fn is None
        assert not self_cache["k"].requires_grad
        assert not self_cache["v"].requires_grad

        # The real Stage-2 cross-cache is populated once during sink preload
        # and reused for all nine later forwards.
        assert cross_k_calls == [torch.Size([1, 2, 12])]
        cross_cache = state.cross_kv[0]
        assert cross_cache["stage2_enabled"] is True
        assert cross_cache["is_init"] is True
        assert torch.count_nonzero(cross_cache["k"]).item() > 0
        assert cross_cache["k"].grad_fn is None
        assert cross_cache["v"].grad_fn is None

        result.latents.float().mean().backward()
        gradients = [parameter.grad for _, parameter in trainable]
        assert all(gradient is not None for gradient in gradients)
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        assert any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients)
