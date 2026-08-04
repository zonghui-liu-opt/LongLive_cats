from __future__ import annotations

from types import SimpleNamespace

import torch

import wan_5b.modules.causal_model as causal_model_module
from wan_5b.modules.causal_model import CausalWanModel, CausalWanSelfAttention


def _run_real_attention_rolling(monkeypatch, *, sink_size: int):
    frame_seq_length = 3
    block_frames = 8
    block_tokens = block_frames * frame_seq_length
    capacity = 24 * frame_seq_length
    attention = CausalWanSelfAttention(
        dim=1,
        num_heads=1,
        local_attn_size=24,
        sink_size=sink_size,
        qk_norm=False,
    )
    with torch.no_grad():
        for projection in (attention.q, attention.k, attention.v, attention.o):
            projection.weight.fill_(1)
            projection.bias.zero_()

    monkeypatch.setattr(
        causal_model_module,
        "causal_rope_apply",
        lambda value, *_args, **_kwargs: value,
    )
    monkeypatch.setattr(
        causal_model_module,
        "attention",
        lambda query, _key, _value: torch.zeros_like(query),
    )
    cache = {
        "k": torch.zeros(1, capacity, 1, 1),
        "v": torch.zeros(1, capacity, 1, 1),
        "quantized": False,
        "block_token_size": block_tokens,
        "max_blocks": 3,
        "global_end_index": torch.tensor([0], dtype=torch.long),
        "local_end_index": torch.tensor([0], dtype=torch.long),
        "pinned_start": torch.tensor([-1], dtype=torch.long),
        "pinned_len": torch.tensor([0], dtype=torch.long),
    }
    first_k = None
    first_v = None
    for block_index in range(8):
        causal_model_module._CURRENT_GRID_META.clear()
        inputs = torch.full((1, block_tokens, 1), float(block_index + 1))
        _output, cache_update = attention(
            inputs,
            seq_lens=torch.tensor([block_tokens]),
            grid_sizes=torch.tensor([[block_frames, 1, frame_seq_length]]),
            freqs=None,
            block_mask=None,
            kv_cache=cache,
            current_start=block_index * block_tokens,
            cache_start=block_index * block_tokens,
        )
        CausalWanModel._apply_cache_updates(
            SimpleNamespace(),
            [cache],
            [(0, cache_update)],
        )
        if block_index == 0:
            first_k = cache["k"][:, :frame_seq_length].clone()
            first_v = cache["v"][:, :frame_seq_length].clone()
    return cache, first_k, first_v


def test_real_causal_attention_legacy_leading_sink_preserves_initial_frame_tokens(
    monkeypatch,
):
    sink0, sink0_first_k, sink0_first_v = _run_real_attention_rolling(
        monkeypatch,
        sink_size=0,
    )
    sink1, sink1_first_k, sink1_first_v = _run_real_attention_rolling(
        monkeypatch,
        sink_size=1,
    )

    assert not torch.equal(sink0["k"][:, :3], sink0_first_k)
    assert not torch.equal(sink0["v"][:, :3], sink0_first_v)
    assert torch.equal(sink1["k"][:, :3], sink1_first_k)
    assert torch.equal(sink1["v"][:, :3], sink1_first_v)
    for cache in (sink0, sink1):
        assert int(cache["global_end_index"].item()) == 64 * 3
        assert int(cache["local_end_index"].item()) == 24 * 3
        assert int(cache["pinned_start"].item()) == -1
        assert int(cache["pinned_len"].item()) == 0
