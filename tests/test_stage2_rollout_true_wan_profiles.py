from __future__ import annotations

import torch


def test_true_causal_wan_uses_named_c4_k2_runtime_and_restores_it(monkeypatch):
    """A non-baseline spec must reach real Wan attention only for the episode."""

    # Import before applying the scoped environment override so pytest restores
    # the module's actual process default even when this is its first import.
    import wan_5b.modules.causal_model as causal_model_module

    monkeypatch.setenv("LLV2_TRITON_ADALN", "0")
    monkeypatch.setenv("LLV2_TRITON_ROPE", "0")

    from model.stage2_dmd import Stage2DiTRole
    from pipeline.stage2_rollout import (
        STAGE2_K2_SHIFT5_TIMESTEPS,
        Stage2RolloutPipeline,
    )
    from wan_5b.modules.causal_model import CausalWanModel

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

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(941)
        model = CausalWanModel(
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
        attention = model.blocks[0].self_attn
        original = {
            "model_local": model.local_attn_size,
            "model_sink": model.sink_size,
            "model_has_global": hasattr(model, "global_sink_size"),
            "block_local": model.blocks[0].local_attn_size,
            "attention_local": attention.local_attn_size,
            "attention_max": attention.max_attention_size,
            "attention_sink": attention.sink_size,
            "attention_global": attention.global_sink_size,
        }
        observed: list[tuple[int, int, int, int]] = []

        def record_attention_runtime(module, _args):
            observed.append(
                (
                    module.local_attn_size,
                    module.max_attention_size,
                    module.sink_size,
                    module.global_sink_size,
                )
            )

        attention.register_forward_pre_hook(record_attention_runtime)
        role = Stage2DiTRole(model, role="generator", is_causal=True)
        pipeline = Stage2RolloutPipeline(role, spec="c4w8k2s1")
        initial = torch.zeros(1, 1, 48, 30, 52, dtype=torch.bfloat16)
        noise = torch.full(
            (1, 24, 48, 30, 52),
            0.25,
            dtype=torch.bfloat16,
        )
        result, state = pipeline.generate_full_episode(
            initial_latent=initial,
            noise=noise,
            conditional_dict={
                "prompt_embeds": torch.ones(1, 2, 4, dtype=torch.bfloat16)
            },
        )

    assert result.latents.shape == noise.shape
    assert not result.latents.requires_grad
    assert result.scheduler_timesteps == STAGE2_K2_SHIFT5_TIMESTEPS
    assert result.scheduler_sigmas[-1] == 0.0
    assert result.cache_audit["profile"]["name"] == "c4w8k2s1"
    assert result.cache_audit["capacity_frames"] == 9
    assert result.cache_audit["scheduler_instances"] == 6
    assert result.cache_audit["generator_forward_calls"] == 19
    assert int(state.self_kv[0]["global_end_index"].item()) == 25 * 390
    assert int(state.self_kv[0]["local_end_index"].item()) == 9 * 390

    # Sink preload + six chunks * (two noisy forwards + one clean recache).
    assert observed == [(9, 9 * 390, 1, 1)] * 19

    # The real shared Wan object may be reused immediately by another profile.
    # Every attribute, including one created only by the rollout context, must
    # therefore return to its constructor value after the episode.
    assert model.local_attn_size == original["model_local"]
    assert model.sink_size == original["model_sink"]
    assert hasattr(model, "global_sink_size") is original["model_has_global"]
    assert model.blocks[0].local_attn_size == original["block_local"]
    assert attention.local_attn_size == original["attention_local"]
    assert attention.max_attention_size == original["attention_max"]
    assert attention.sink_size == original["attention_sink"]
    assert attention.global_sink_size == original["attention_global"]
