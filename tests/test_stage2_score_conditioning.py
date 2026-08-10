from __future__ import annotations

import torch
from torch import nn
import pytest

from model.stage2_dmd import Stage2DiTRole
from utils.stage2_i2v_conditioning import (
    STAGE2_PATCH_TOKENS_PER_FRAME,
    STAGE2_SCORE_SEQ_LEN,
    pack_stage2_i2v_score_input,
    prepare_stage2_i2v_score_model_inputs,
)
from utils.wan_forward_adapter import (
    call_wan_model_with_cache_policy,
    legacy_wan_model_timestep,
)


def _latent(
    frames: int,
    *,
    batch: int = 1,
    orientation: tuple[int, int] = (30, 52),
    fill: float = 0.0,
) -> torch.Tensor:
    return torch.full(
        (batch, frames, 48, *orientation),
        fill,
        dtype=torch.float32,
    )


def _frame_timestep(*future_values: float) -> torch.Tensor:
    return torch.tensor(
        [[0.0] + [float(value)] * 24 for value in future_values],
        dtype=torch.float32,
    )


class _ScoreSpy(nn.Module):
    patch_size = (1, 2, 2)
    text_len = 3
    text_dim = 4

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.calls: list[dict[str, object]] = []

    def forward(self, x, *, t, context, seq_len):
        self.calls.append(
            {
                "x_shape": tuple(x.shape),
                "t": t.detach().clone(),
                "context": context,
                "seq_len": seq_len,
            }
        )
        return x * self.anchor


def test_score_pack_is_explicit_initial_plus_all_24_unique_future_frames():
    initial = _latent(1, fill=-1.0)
    future = torch.arange(24, dtype=torch.float32).view(1, 24, 1, 1, 1)
    future = future.expand(1, 24, 48, 30, 52).clone().requires_grad_(True)

    packed = pack_stage2_i2v_score_input(initial, future)

    assert packed.shape == (1, 25, 48, 30, 52)
    temporal_sentinels = packed[0, :, 0, 0, 0].tolist()
    assert temporal_sentinels == [-1.0, *map(float, range(24))]
    assert temporal_sentinels.count(0.0) == 1
    assert temporal_sentinels.count(23.0) == 1
    endpoint_loss = packed[:, 1].sum() + packed[:, 24].sum()
    endpoint_loss.backward()
    assert torch.count_nonzero(future.grad[:, 0]).item() == future[:, 0].numel()
    assert torch.count_nonzero(future.grad[:, 23]).item() == future[:, 23].numel()
    assert torch.count_nonzero(future.grad[:, 1:23]).item() == 0


@pytest.mark.parametrize("orientation", [(30, 52), (52, 30)])
def test_score_role_passes_dynamic_9750_token_mixed_timestep_to_wan(orientation):
    spy = _ScoreSpy()
    role = Stage2DiTRole(spy, role="fake_score", is_causal=False)
    score_input = _latent(25, orientation=orientation)
    frame_timestep = _frame_timestep(317.25)
    prompt_embeds = torch.zeros(1, 3, 4, dtype=torch.bfloat16)

    raw_flow, x0_pred = role.forward_score(
        noisy_image_or_video=score_input,
        conditional_dict={"prompt_embeds": prompt_embeds},
        frame_timestep=frame_timestep,
    )

    assert raw_flow.shape == score_input.shape
    assert x0_pred.shape == score_input.shape
    assert x0_pred.dtype == torch.float32
    assert len(spy.calls) == 1
    call = spy.calls[0]
    assert call["x_shape"] == (1, 48, 25, *orientation)
    assert call["context"] is prompt_embeds
    assert call["seq_len"] == STAGE2_SCORE_SEQ_LEN
    token_timestep = call["t"]
    assert token_timestep.shape == (1, STAGE2_SCORE_SEQ_LEN)
    assert torch.equal(
        token_timestep[:, :STAGE2_PATCH_TOKENS_PER_FRAME],
        torch.zeros(1, STAGE2_PATCH_TOKENS_PER_FRAME),
    )
    assert torch.equal(
        token_timestep[:, STAGE2_PATCH_TOKENS_PER_FRAME:],
        torch.full((1, 9_360), 317.25),
    )


def test_score_token_timestep_is_holistic_per_video_not_frame_zero_scalar():
    score_input = _latent(25).expand(2, -1, -1, -1, -1)
    frame_timestep = _frame_timestep(20.0, 980.0)

    prepared = prepare_stage2_i2v_score_model_inputs(score_input, frame_timestep)

    assert prepared.patch_tokens_per_frame == 390
    assert prepared.seq_len == 9_750
    assert prepared.token_timestep.shape == (2, 9_750)
    assert torch.count_nonzero(prepared.token_timestep[:, :390]).item() == 0
    assert torch.equal(prepared.token_timestep[0, 390:], torch.full((9_360,), 20.0))
    assert torch.equal(prepared.token_timestep[1, 390:], torch.full((9_360,), 980.0))


def test_score_adapter_preserves_generator_endpoint_gradients():
    spy = _ScoreSpy()
    role = Stage2DiTRole(spy, role="fake_score", is_causal=False)
    initial = _latent(1, fill=-3.0)
    future = _latent(24).requires_grad_(True)
    score_input = pack_stage2_i2v_score_input(initial, future)

    raw_flow, _ = role.forward_score(
        noisy_image_or_video=score_input,
        conditional_dict={"prompt_embeds": torch.zeros(1, 2, 4, dtype=torch.bfloat16)},
        frame_timestep=_frame_timestep(500.0),
    )
    (raw_flow[:, 1].sum() + raw_flow[:, 24].sum()).backward()

    assert torch.count_nonzero(future.grad[:, 0]).item() == future[:, 0].numel()
    assert torch.count_nonzero(future.grad[:, 23]).item() == future[:, 23].numel()
    assert torch.count_nonzero(future.grad[:, 1:23]).item() == 0


@pytest.mark.parametrize(
    "prompt_embeds, match",
    [
        (torch.zeros(1, 4, 4, dtype=torch.bfloat16), "sequence length"),
        (torch.zeros(1, 3, 5, dtype=torch.bfloat16), "text_dim"),
    ],
)
def test_score_adapter_rejects_prompt_geometry_outside_wan_contract(
    prompt_embeds, match
):
    role = Stage2DiTRole(_ScoreSpy(), role="fake_score", is_causal=False)
    with pytest.raises(ValueError, match=match):
        role.forward_score(
            noisy_image_or_video=_latent(25),
            conditional_dict={"prompt_embeds": prompt_embeds},
            frame_timestep=_frame_timestep(500.0),
        )


def test_public_wan_wrapper_stage2_score_entry_uses_same_bf16_model_cast():
    from utils.wan_5b_wrapper import WanDiffusionWrapper

    class _WrapperScoreSpy(_ScoreSpy):
        def __init__(self):
            super().__init__()
            self.patch_embedding = nn.Conv3d(
                48, 4, kernel_size=(1, 2, 2), stride=(1, 2, 2), bias=False
            ).to(dtype=torch.bfloat16)

        def forward(self, x, *, t, context, seq_len):
            assert x.dtype == torch.bfloat16
            return super().forward(x, t=t, context=context, seq_len=seq_len)

    wrapper = WanDiffusionWrapper.__new__(WanDiffusionWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = _WrapperScoreSpy()
    wrapper.uniform_timestep = True
    wrapper._compiled_model_call = None

    raw_flow = wrapper.forward_stage2_score(
        noisy_image_or_video=_latent(25),
        conditional_dict={"prompt_embeds": torch.zeros(1, 3, 4, dtype=torch.bfloat16)},
        frame_timestep=_frame_timestep(317.25),
    )
    assert raw_flow.dtype == torch.bfloat16
    assert raw_flow.shape == _latent(25).shape


@pytest.mark.parametrize(
    ("frame_timestep", "match"),
    [
        (torch.full((1, 25), 200.0), "sink timestep"),
        (
            torch.tensor([[0.0] + [200.0] * 23 + [201.0]]),
            "video-global",
        ),
    ],
)
def test_score_adapter_rejects_non_stage2_timestep_contract(frame_timestep, match):
    with pytest.raises(ValueError, match=match):
        prepare_stage2_i2v_score_model_inputs(_latent(25), frame_timestep)


def test_score_pack_rejects_legacy_23_or_slot_overwrite_shapes():
    with pytest.raises(ValueError, match="exactly 24 future"):
        pack_stage2_i2v_score_input(_latent(1), _latent(23))
    with pytest.raises(ValueError, match="exactly one explicit"):
        pack_stage2_i2v_score_input(_latent(2), _latent(24))


def test_legacy_uniform_timestep_resolution_is_unchanged():
    timestep = torch.tensor([[101.0, 101.0, 101.0], [707.0, 707.0, 707.0]])
    uniform = legacy_wan_model_timestep(timestep, uniform_timestep=True)
    causal = legacy_wan_model_timestep(timestep, uniform_timestep=False)
    assert torch.equal(uniform, torch.tensor([101.0, 707.0]))
    assert causal is timestep


class _CacheUpdateOwner(nn.Module):
    def __init__(self):
        super().__init__()
        self.applied = 0
        self.received_update_infos = None

    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        self.applied += 1
        self.received_update_infos = cache_update_infos
        for block_index, (current_end, local_end, update) in cache_update_infos:
            cache = kv_cache[block_index]
            cache["k"][:, :1] = update["new_k"]
            cache["v"][:, :1] = update["new_v"]
            cache["global_end_index"].fill_(current_end)
            cache["local_end_index"].fill_(local_end)


def _empty_cache():
    return [
        {
            "k": torch.zeros(1, 1, 1, 1),
            "v": torch.zeros(1, 1, 1, 1),
            "global_end_index": torch.tensor(0),
            "local_end_index": torch.tensor(0),
        }
    ]


@pytest.mark.parametrize("commit_self_kv", [False, True])
def test_explicit_self_kv_policy_discards_or_detached_commits(commit_self_kv):
    owner = _CacheUpdateOwner()
    kv_cache = _empty_cache()
    source = torch.tensor([[[[3.0]]]], requires_grad=True)
    model_input = torch.tensor([2.0], requires_grad=True)

    def deferred_model(value, *, defer_cache_updates=False, **kwargs):
        assert defer_cache_updates is True
        del kwargs
        update = {
            "action": "direct_insert",
            "local_start_index": 0,
            "local_end_index": 1,
            "new_k": source,
            "new_v": source * 2,
        }
        return value * source.flatten()[0], [(0, (1, 1, update))]

    output = call_wan_model_with_cache_policy(
        deferred_model,
        model_input,
        cache_update_owner=owner,
        commit_self_kv=commit_self_kv,
        kv_cache=kv_cache,
    )
    output.sum().backward()

    assert source.grad is not None
    assert model_input.grad is not None
    if commit_self_kv:
        assert owner.applied == 1
        assert torch.equal(kv_cache[0]["k"], source.detach())
        assert torch.equal(kv_cache[0]["v"], (source * 2).detach())
        assert kv_cache[0]["k"].requires_grad is False
        assert kv_cache[0]["k"].grad_fn is None
        assert kv_cache[0]["v"].requires_grad is False
        assert kv_cache[0]["v"].grad_fn is None
        update = owner.received_update_infos[0][1][2]
        assert update["new_k"].grad_fn is None
        assert update["new_v"].grad_fn is None
    else:
        assert owner.applied == 0
        assert torch.count_nonzero(kv_cache[0]["k"]).item() == 0
        assert torch.count_nonzero(kv_cache[0]["v"]).item() == 0


def test_default_self_kv_policy_preserves_legacy_direct_call(monkeypatch):
    monkeypatch.delenv("LLV2_DEFER_KV_UPDATES", raising=False)
    owner = _CacheUpdateOwner()
    kv_cache = _empty_cache()
    observed = {}

    def legacy_model(value, **kwargs):
        observed.update(kwargs)
        return value + 1

    output = call_wan_model_with_cache_policy(
        legacy_model,
        torch.tensor([2.0]),
        cache_update_owner=owner,
        commit_self_kv=None,
        kv_cache=kv_cache,
    )

    assert torch.equal(output, torch.tensor([3.0]))
    assert "defer_cache_updates" not in observed
    assert owner.applied == 0
