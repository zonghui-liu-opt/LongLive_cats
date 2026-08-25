import pytest
import torch

from utils.wan_5b_wrapper import (
    WanVAEWrapper,
    audit_wan_vae_runtime,
    configure_wan_vae_runtime,
)


class _TinyConvVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(3, 48, kernel_size=1)
        self.register_buffer("scale", torch.ones(1, dtype=torch.float32))
        self.register_buffer("indices", torch.arange(1, dtype=torch.int64))

    def encode(self, pixels, _scale):
        return self.conv(pixels)


def _tiny_wrapper() -> WanVAEWrapper:
    wrapper = WanVAEWrapper.__new__(WanVAEWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.mean = torch.zeros(48, dtype=torch.float32)
    wrapper.std = torch.ones(48, dtype=torch.float32)
    wrapper.model = _TinyConvVAE().eval().requires_grad_(False)
    wrapper.eval().requires_grad_(False)
    return wrapper


def test_encode_rejects_bfloat16_input_with_float32_vae_before_conv3d():
    wrapper = _tiny_wrapper()
    pixels = torch.zeros(1, 3, 1, 2, 2, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="Wan VAE runtime contract"):
        wrapper.encode_to_latent(pixels)


def test_configure_casts_all_floating_state_and_preserves_integer_buffers():
    wrapper = configure_wan_vae_runtime(
        _tiny_wrapper(), device=torch.device("cpu"), dtype=torch.bfloat16
    )

    assert {parameter.dtype for parameter in wrapper.parameters()} == {torch.bfloat16}
    assert wrapper.model.scale.dtype == torch.bfloat16
    assert wrapper.model.indices.dtype == torch.int64
    assert not wrapper.training
    assert all(not module.training for module in wrapper.modules())
    assert all(not parameter.requires_grad for parameter in wrapper.parameters())

    pixels = torch.zeros(1, 3, 1, 2, 2, dtype=torch.bfloat16)
    encoded = wrapper.encode_to_latent(pixels)
    assert encoded.dtype == torch.float32
    assert tuple(encoded.shape) == (1, 1, 48, 2, 2)


@pytest.mark.parametrize("state_kind", ["parameter", "buffer"])
def test_runtime_audit_rejects_each_mixed_floating_state(state_kind):
    wrapper = configure_wan_vae_runtime(
        _tiny_wrapper(), device=torch.device("cpu"), dtype=torch.bfloat16
    )
    if state_kind == "parameter":
        wrapper.model.conv.bias.data = wrapper.model.conv.bias.data.float()
        expected_name = "model.conv.bias"
    else:
        wrapper.model.scale = wrapper.model.scale.float()
        expected_name = "model.scale"

    with pytest.raises(RuntimeError, match=expected_name):
        audit_wan_vae_runtime(
            wrapper,
            expected_device=torch.device("cpu"),
            expected_dtype=torch.bfloat16,
            operation="test",
            require_eval=True,
            require_frozen=True,
        )
