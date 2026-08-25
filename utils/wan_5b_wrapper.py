import types
from typing import List, Optional
import json
import os
import torch

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from utils.wan_forward_adapter import (
    call_wan_model_with_cache_policy,
    forward_stage2_i2v_score_model,
    legacy_wan_model_timestep,
    wan_patch_embedding_dtype,
)

from wan_5b.modules.tokenizers import HuggingfaceTokenizer
from wan_5b.modules.model import WanModel
from wan_5b.modules.vae2_2 import _video_vae
from wan_5b.modules.t5 import umt5_xxl
from wan_5b.modules.causal_model import CausalWanModel

DEFAULT_WAN_ARCHITECTURE_ROOT = "wan_models/Wan2.2-TI2V-5B"
DEFAULT_WAN_T5_CHECKPOINT = "wan_models/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
DEFAULT_WAN_TOKENIZER_DIR = "wan_models/Wan2.2-TI2V-5B/google/umt5-xxl"
DEFAULT_WAN_VAE_CHECKPOINT = "wan_models/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"

# These values are part of the immutable positive/negative Stage-2 text-cache
# contract.  Keeping WanTextEncoder itself wired to the same constants makes
# the provenance validator below a check of the production path rather than a
# second, test-only description of it.
WAN_TEXT_SEQUENCE_LENGTH = 512
WAN_TEXT_CLEANING = "whitespace"
WAN_TEXT_ADD_SPECIAL_TOKENS = True
WAN_TEXT_PADDING_SIDE = "right"
WAN_TEXT_EMBEDDING_PADDING_VALUE = 0.0


def _resolved_runtime_device(device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    return resolved


def audit_wan_vae_runtime(
    vae: torch.nn.Module,
    *,
    expected_device,
    expected_dtype: torch.dtype,
    operation: str,
    require_eval: bool = False,
    require_frozen: bool = False,
) -> torch.nn.Module:
    """Fail before a VAE kernel when model state cannot consume its input.

    Wan's VAE checkpoint loader preserves the checkpoint dtype.  Callers must
    therefore make the model/input contract explicit instead of relying on
    autocast or on the dtype of a single representative parameter.
    """

    if not isinstance(vae, torch.nn.Module):
        raise TypeError("Wan VAE runtime audit requires a torch.nn.Module")
    if not isinstance(expected_dtype, torch.dtype):
        raise TypeError("Wan VAE expected_dtype must be a torch.dtype")
    if not torch.empty((), dtype=expected_dtype).is_floating_point():
        raise ValueError("Wan VAE runtime requires a floating-point input dtype")

    device = _resolved_runtime_device(expected_device)
    issues: list[str] = []

    if require_eval:
        training_modules = [
            name or "<root>" for name, module in vae.named_modules() if module.training
        ]
        if training_modules:
            issues.append(
                "modules still in training mode: " + ", ".join(training_modules[:4])
            )

    for name, parameter in vae.named_parameters():
        label = f"parameter {name!r}"
        if parameter.is_meta:
            issues.append(f"{label} is still on the meta device")
            continue
        if parameter.device != device:
            issues.append(f"{label} is on {parameter.device}, expected {device}")
        if parameter.is_floating_point() and parameter.dtype != expected_dtype:
            issues.append(
                f"{label} has dtype {parameter.dtype}, expected {expected_dtype}"
            )
        if require_frozen and parameter.requires_grad:
            issues.append(f"{label} still has requires_grad=True")

    for name, buffer in vae.named_buffers():
        label = f"buffer {name!r}"
        if buffer.is_meta:
            issues.append(f"{label} is still on the meta device")
            continue
        if buffer.device != device:
            issues.append(f"{label} is on {buffer.device}, expected {device}")
        if buffer.is_floating_point() and buffer.dtype != expected_dtype:
            issues.append(
                f"{label} has dtype {buffer.dtype}, expected {expected_dtype}"
            )

    if issues:
        visible = "; ".join(issues[:8])
        omitted = len(issues) - 8
        suffix = f"; plus {omitted} more issue(s)" if omitted else ""
        raise RuntimeError(
            f"Wan VAE runtime contract failed before {operation}: {visible}{suffix}"
        )
    return vae


def configure_wan_vae_runtime(
    vae: torch.nn.Module,
    *,
    device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """Move, freeze, and audit a Wan VAE for deterministic execution."""

    resolved_device = _resolved_runtime_device(device)
    vae.eval().requires_grad_(False).to(device=resolved_device, dtype=dtype)
    return audit_wan_vae_runtime(
        vae,
        expected_device=resolved_device,
        expected_dtype=dtype,
        operation="initialization",
        require_eval=True,
        require_frozen=True,
    )


def audit_wan_text_encoding_tokenizer_contract(tokenizer_dir):
    """Validate the tokenizer half of Wan's locked cache-encoding contract.

    This intentionally loads only the tokenizer, never T5.  Besides checking
    the configured values, it executes a short positive/negative special-token
    probe and verifies that the returned attention mask is a right-padded
    prefix.  Embedding padding is bound to the production constant used by
    :class:`WanTextEncoder`; the actual cached tensors are independently
    checked for exact zero padding by the Stage-2 cache audit.
    """

    if WAN_TEXT_SEQUENCE_LENGTH != 512:
        raise RuntimeError("Wan text sequence length contract drifted")
    if WAN_TEXT_CLEANING != "whitespace":
        raise RuntimeError("Wan text cleaning contract drifted")
    if WAN_TEXT_ADD_SPECIAL_TOKENS is not True:
        raise RuntimeError("Wan add_special_tokens contract drifted")
    if WAN_TEXT_PADDING_SIDE != "right":
        raise RuntimeError("Wan tokenizer padding-side contract drifted")
    if WAN_TEXT_EMBEDDING_PADDING_VALUE != 0.0:
        raise RuntimeError("Wan embedding padding must remain exact zero")

    tokenizer = HuggingfaceTokenizer(
        name=os.path.abspath(os.path.expanduser(os.fspath(tokenizer_dir))),
        seq_len=WAN_TEXT_SEQUENCE_LENGTH,
        clean=WAN_TEXT_CLEANING,
    )
    if tokenizer.seq_len != WAN_TEXT_SEQUENCE_LENGTH:
        raise RuntimeError("Wan tokenizer sequence length differs from 512")
    if tokenizer.clean != WAN_TEXT_CLEANING:
        raise RuntimeError("Wan tokenizer cleaning differs from whitespace")
    padding_side = getattr(tokenizer.tokenizer, "padding_side", None)
    if padding_side != WAN_TEXT_PADDING_SIDE:
        raise RuntimeError(
            "Wan tokenizer must use right padding, got " f"{padding_side!r}"
        )

    probe = ["stage2 provenance probe", "stage2"]
    _, mask_with_special = tokenizer(
        probe,
        return_mask=True,
        add_special_tokens=WAN_TEXT_ADD_SPECIAL_TOKENS,
    )
    _, mask_without_special = tokenizer(
        probe,
        return_mask=True,
        add_special_tokens=False,
    )
    expected_shape = (len(probe), WAN_TEXT_SEQUENCE_LENGTH)
    if (
        tuple(mask_with_special.shape) != expected_shape
        or tuple(mask_without_special.shape) != expected_shape
    ):
        raise RuntimeError("Wan tokenizer did not return the locked [B,512] mask")
    valid_with_special = mask_with_special.to(dtype=torch.bool).sum(dim=1)
    valid_without_special = mask_without_special.to(dtype=torch.bool).sum(dim=1)
    if not bool(torch.all(valid_with_special > valid_without_special).item()):
        raise RuntimeError(
            "Wan tokenizer add_special_tokens=True did not add a special token"
        )
    positions = torch.arange(WAN_TEXT_SEQUENCE_LENGTH).view(1, -1)
    expected_mask = positions < valid_with_special.view(-1, 1).cpu()
    if not torch.equal(mask_with_special.to(dtype=torch.bool).cpu(), expected_mask):
        raise RuntimeError("Wan tokenizer attention mask is not right padded")

    return {
        "cleaning": WAN_TEXT_CLEANING,
        "add_special_tokens": WAN_TEXT_ADD_SPECIAL_TOKENS,
        "sequence_length": WAN_TEXT_SEQUENCE_LENGTH,
        "padding_side": WAN_TEXT_PADDING_SIDE,
        "embedding_padding_value": WAN_TEXT_EMBEDDING_PADDING_VALUE,
        "validated_special_token_growth": True,
        "validated_right_padding_mask": True,
    }


_WAN_ARCHITECTURE_FIELDS = (
    "model_type",
    "patch_size",
    "text_len",
    "in_dim",
    "dim",
    "ffn_dim",
    "freq_dim",
    "text_dim",
    "out_dim",
    "num_heads",
    "num_layers",
    "window_size",
    "qk_norm",
    "cross_attn_norm",
    "eps",
)


def _architecture_config_from_root(architecture_root, model_name):
    """Load architecture values without loading a model checkpoint."""
    root = os.path.abspath(os.path.expanduser(os.fspath(architecture_root)))
    config_path = os.path.join(root, "config.json")
    raw_config = {}
    if model_name == "Wan2.2-TI2V-5B":
        # Some Diffusers configs omit constructor values listed in
        # ``ignore_for_config``.  Seed those locked TI2V values from the same
        # canonical config used by the flat-checkpoint inference loader, then
        # let an explicit config.json override them.
        from wan_5b.configs import WAN_CONFIGS

        raw_config.update(dict(WAN_CONFIGS["ti2v-5B"]))
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as handle:
            file_config = json.load(handle)
        if not isinstance(file_config, dict):
            raise ValueError(
                f"Wan architecture config must be an object: {config_path}"
            )
        raw_config.update(file_config)
    elif (
        model_name == "Wan2.2-TI2V-5B"
        and architecture_root == DEFAULT_WAN_ARCHITECTURE_ROOT
    ):
        # Preserve the legacy default while allowing architecture-only tools to
        # run against config-less TI2V weight layouts.
        pass
    else:
        raise FileNotFoundError(
            f"Wan architecture_root must contain config.json: {config_path}"
        )

    missing = [field for field in _WAN_ARCHITECTURE_FIELDS if field not in raw_config]
    if missing:
        raise ValueError(
            f"Wan architecture config is missing required fields {missing}: {config_path}"
        )
    return {field: raw_config[field] for field in _WAN_ARCHITECTURE_FIELDS}


def build_wan_model(
    *,
    model_name="Wan2.2-TI2V-5B",
    is_causal=False,
    architecture_root=None,
    init_weights=True,
    local_attn_size=-1,
    sink_size=0,
    num_frame_per_block=1,
):
    """Build a Wan DiT with explicit architecture and weight-loading policy.

    ``init_weights=True`` preserves the legacy ``from_pretrained`` behavior.
    ``False`` constructs only the architecture from ``config.json`` (or the
    built-in legacy TI2V config) and never reads a model weight file.
    """
    architecture_root = os.fspath(
        architecture_root
        if architecture_root is not None
        else f"wan_models/{model_name}"
    )
    model_cls = CausalWanModel if is_causal else WanModel
    if init_weights:
        overrides = {}
        if is_causal:
            overrides = {
                "local_attn_size": local_attn_size,
                "sink_size": sink_size,
                "num_frame_per_block": num_frame_per_block,
            }
        return model_cls.from_pretrained(architecture_root, **overrides)

    architecture_kwargs = _architecture_config_from_root(architecture_root, model_name)
    if is_causal:
        architecture_kwargs.pop("window_size")
        architecture_kwargs.update(
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
        )
    return model_cls(**architecture_kwargs)


class WanTextEncoder(torch.nn.Module):
    def __init__(
        self,
        t5_checkpoint=None,
        tokenizer_dir=None,
        device=None,
    ) -> None:
        super().__init__()

        t5_checkpoint = os.path.abspath(
            os.path.expanduser(os.fspath(t5_checkpoint or DEFAULT_WAN_T5_CHECKPOINT))
        )
        tokenizer_dir = os.path.abspath(
            os.path.expanduser(os.fspath(tokenizer_dir or DEFAULT_WAN_TOKENIZER_DIR))
        )
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self._device = torch.device(device)

        self.text_encoder = (
            umt5_xxl(
                encoder_only=True,
                return_tokenizer=False,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )
            .eval()
            .requires_grad_(False)
        )
        self.text_encoder.load_state_dict(
            torch.load(t5_checkpoint, map_location="cpu", weights_only=False)
        )

        self.text_encoder = self.text_encoder.to(self._device)

        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_dir,
            seq_len=WAN_TEXT_SEQUENCE_LENGTH,
            clean=WAN_TEXT_CLEANING,
        )

    @property
    def device(self):
        try:
            return next(self.text_encoder.parameters()).device
        except StopIteration:
            return self._device

    def forward(self, text_prompts: List[str], return_mask: bool = False) -> dict:
        ids, mask = self.tokenizer(
            text_prompts,
            return_mask=True,
            add_special_tokens=WAN_TEXT_ADD_SPECIAL_TOKENS,
        )
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        prompt_mask = mask.gt(0)
        seq_lens = prompt_mask.sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        for u, v in zip(context, seq_lens):
            u[v:] = WAN_TEXT_EMBEDDING_PADDING_VALUE

        result = {"prompt_embeds": context}
        if return_mask:
            result["prompt_mask"] = prompt_mask
        return result


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, vae_checkpoint=None):
        super().__init__()
        mean = [
            -0.2289,
            -0.0052,
            -0.1323,
            -0.2339,
            -0.2799,
            0.0174,
            0.1838,
            0.1557,
            -0.1382,
            0.0542,
            0.2813,
            0.0891,
            0.1570,
            -0.0098,
            0.0375,
            -0.1825,
            -0.2246,
            -0.1207,
            -0.0698,
            0.5109,
            0.2665,
            -0.2108,
            -0.2158,
            0.2502,
            -0.2055,
            -0.0322,
            0.1109,
            0.1567,
            -0.0729,
            0.0899,
            -0.2799,
            -0.1230,
            -0.0313,
            -0.1649,
            0.0117,
            0.0723,
            -0.2839,
            -0.2083,
            -0.0520,
            0.3748,
            0.0152,
            0.1957,
            0.1433,
            -0.2944,
            0.3573,
            -0.0548,
            -0.1681,
            -0.0667,
        ]
        std = [
            0.4765,
            1.0364,
            0.4514,
            1.1677,
            0.5313,
            0.4990,
            0.4818,
            0.5013,
            0.8158,
            1.0344,
            0.5894,
            1.0901,
            0.6885,
            0.6165,
            0.8454,
            0.4978,
            0.5759,
            0.3523,
            0.7135,
            0.6804,
            0.5833,
            1.4146,
            0.8986,
            0.5659,
            0.7069,
            0.5338,
            0.4889,
            0.4917,
            0.4069,
            0.4999,
            0.6866,
            0.4093,
            0.5709,
            0.6065,
            0.6415,
            0.4944,
            0.5726,
            1.2042,
            0.5458,
            1.6887,
            0.3971,
            1.0600,
            0.3943,
            0.5537,
            0.5444,
            0.4089,
            0.7468,
            0.7744,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = (
            _video_vae(
                pretrained_path=os.path.abspath(
                    os.path.expanduser(
                        os.fspath(vae_checkpoint or DEFAULT_WAN_VAE_CHECKPOINT)
                    )
                ),
            )
            .eval()
            .requires_grad_(False)
        )

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        audit_wan_vae_runtime(
            self,
            expected_device=device,
            expected_dtype=dtype,
            operation="encode_to_latent",
        )

        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(
        self, latent: torch.Tensor, use_cache: bool = False
    ) -> torch.Tensor:
        audit_wan_vae_runtime(
            self,
            expected_device=latent.device,
            expected_dtype=latent.dtype,
            operation="decode_to_pixel",
        )
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(
                decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
            )
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel_chunk(
        self, latent: torch.Tensor, use_cache: bool = False, chunk_size: int = 1
    ) -> torch.Tensor:
        """
        Decode latent frames to pixel space.

        Args:
            latent: Latent tensor with shape [batch_size, num_frames, num_channels, height, width]
            use_cache: Whether to use cached decoding (for streaming)
            chunk_size: Number of latent frames to decode at once (default 240 to avoid OOM)

        Returns:
            Decoded video tensor with shape [batch_size, num_frames, num_channels, height, width]
        """
        audit_wan_vae_runtime(
            self,
            expected_device=latent.device,
            expected_dtype=latent.dtype,
            operation="decode_to_pixel_chunk",
        )
        # latent shape: [batch_size, num_frames, num_channels, height, width]
        # zs shape after permute: [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            num_frames = u.shape[1]
            if num_frames <= chunk_size:
                # Decode short clips in one pass.
                if use_cache:
                    # Start this segment from a clean cache.
                    self.model.clear_cache()
                decoded = (
                    decode_function(u.unsqueeze(0), scale)
                    .float()
                    .clamp_(-1, 1)
                    .squeeze(0)
                )
                decoded = decoded.cpu()
                if use_cache:
                    # Clear after this segment so it cannot affect the next video.
                    self.model.clear_cache()
            else:
                # Decode longer clips in temporal chunks.
                decoded_chunks = []
                if use_cache:
                    # Clear once at the segment start; later chunks share the
                    # internal cache.
                    self.model.clear_cache()
                for start_idx in range(0, num_frames, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_frames)
                    chunk = u[:, start_idx:end_idx, :, :]  # [C, chunk_frames, H, W]
                    decoded_chunk = (
                        decode_function(chunk.unsqueeze(0), scale)
                        .float()
                        .clamp_(-1, 1)
                        .squeeze(0)
                    )
                    decoded_chunks.append(decoded_chunk.cpu())

                    del decoded_chunk
                    torch.cuda.empty_cache()
                decoded = torch.cat(decoded_chunks, dim=1)
                if use_cache:
                    # Clear the cache after the full segment.
                    self.model.clear_cache()
            output.append(decoded)

        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output


class WanDiffusionWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name="Wan2.2-TI2V-5B",
        timestep_shift=8.0,
        is_causal=False,
        local_attn_size=-1,
        sink_size=0,
        num_frame_per_block=1,
        t_scale=1.0,
        rope_method="linear",
        original_seq_len=None,
        architecture_root=None,
        init_weights=True,
    ):
        super().__init__()

        self.model = build_wan_model(
            model_name=model_name,
            is_causal=is_causal,
            architecture_root=architecture_root,
            init_weights=init_weights,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
        )
        self.model.eval()
        self.model.t_scale = t_scale
        self.model.rope_method = rope_method
        self.model.original_seq_len = original_seq_len

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 28160  # [1, 32, 48, 44, 80]

        self.post_init()
        self._compiled_model_call = None

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def configure_torch_compile(
        self,
        *,
        backend: str = "inductor",
        mode: str | None = "max-autotune-no-cudagraphs",
        fullgraph: bool = False,
        dynamic: bool | None = False,
        options: dict | None = None,
        suppress_errors: bool = True,
    ) -> bool:
        from utils.torch_compile_utils import configure_module_call_torch_compile

        self._compiled_model_call = configure_module_call_torch_compile(
            self.model,
            name="WanDiffusionWrapper5B.model",
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
            options=options,
            suppress_errors=suppress_errors,
        )
        return self._compiled_model_call is not None

    def _call_model(self, *args, **kwargs):
        commit_self_kv = kwargs.pop("commit_self_kv", None)

        if self._compiled_model_call is not None:
            # iter-25: signal cudagraph allocator that a new "step" starts.
            # Required for mode=reduce-overhead when modules cache state
            # (KV cache rolling buffers, fp4-quant scale tensors) so the
            # cudagraph pool knows it can safely reuse step-N memory now
            # that step-(N+1) is starting.
            mark_step = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
            if mark_step is not None:
                mark_step()
            model_call = self._compiled_model_call
        else:
            model_call = self.model
        return call_wan_model_with_cache_policy(
            model_call,
            *args,
            cache_update_owner=self.model,
            commit_self_kv=commit_self_kv,
            **kwargs,
        )

    @staticmethod
    def _legacy_model_timestep(
        timestep: torch.Tensor, *, uniform_timestep: bool
    ) -> torch.Tensor:
        """Preserve the pre-Stage-2 timestep contract for existing callers."""
        return legacy_wan_model_timestep(timestep, uniform_timestep=uniform_timestep)

    def forward_stage2_score(
        self,
        *,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        frame_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Return raw flow for the strict Stage-2 25-frame score path."""
        if not self.uniform_timestep:
            raise RuntimeError("Stage-2 score adapter requires a bidirectional Wan")
        model_input = noisy_image_or_video.to(
            dtype=wan_patch_embedding_dtype(
                self.model, fallback=noisy_image_or_video.dtype
            )
        )
        return forward_stage2_i2v_score_model(
            self._call_model,
            noisy_image_or_video=model_input,
            conditional_dict=conditional_dict,
            frame_timestep=frame_timestep,
            patch_size=tuple(int(value) for value in self.model.patch_size),
            maximum_text_length=getattr(self.model, "text_len", None),
            expected_text_dim=getattr(self.model, "text_dim", None),
        )

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """
        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps],
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(
        scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
        )
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
        rope_temporal_offset: Optional[torch.Tensor] = None,
        commit_self_kv: bool | None = None,
    ) -> torch.Tensor:
        prompt_embeds = conditional_dict["prompt_embeds"]

        # Stage-2 score calls use ``forward_stage2_score`` and never reduce
        # their mixed token timestep through this legacy frame-zero path.
        input_timestep = self._legacy_model_timestep(
            timestep, uniform_timestep=self.uniform_timestep
        )
        if commit_self_kv is not None and kv_cache is None:
            raise ValueError("commit_self_kv requires kv_cache")

        logits = None
        rope_offset_was_set = rope_temporal_offset is not None and hasattr(
            self.model, "rope_temporal_offset"
        )
        if rope_offset_was_set:
            prev_rope_temporal_offset = self.model.rope_temporal_offset
            self.model.rope_temporal_offset = rope_temporal_offset

        # X0 prediction
        if kv_cache is not None:
            flow_pred = self._call_model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                commit_self_kv=commit_self_kv,
            ).permute(0, 2, 1, 3, 4)
        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self._call_model(
                    noisy_image_or_video.permute(0, 2, 1, 3, 4),
                    t=input_timestep,
                    context=prompt_embeds,
                    seq_len=self.seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self._call_model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=self.seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings,
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    flow_pred = self._call_model(
                        noisy_image_or_video.permute(0, 2, 1, 3, 4),
                        t=input_timestep,
                        context=prompt_embeds,
                        seq_len=self.seq_len,
                    ).permute(0, 2, 1, 3, 4)

        if rope_offset_was_set:
            self.model.rope_temporal_offset = prev_rope_temporal_offset

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        if logits is not None:
            return flow_pred, pred_x0, logits

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler
        )
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler
        )
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler
        )
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()


_MG_LIGHTVAE_DEFAULT_PATHS = {
    "mg_lightvae": os.path.join("wan_models", "Matrix-Game-3.0", "MG-LightVAE.pth"),
    "mg_lightvae_v2": os.path.join(
        "wan_models", "Matrix-Game-3.0", "MG-LightVAE_v2.pth"
    ),
}


def build_vae_5b(args):
    """Return the 5B VAE wrapper requested by args.vae_type."""
    vae_type = str(getattr(args, "vae_type", "wan")).lower().strip()
    model_paths = getattr(args, "model_paths", {}) or {}
    vae_checkpoint = model_paths.get("vae_checkpoint", None)

    if vae_type in ("wan", "wan2.2", ""):
        return WanVAEWrapper(vae_checkpoint=vae_checkpoint)

    if vae_type in _MG_LIGHTVAE_DEFAULT_PATHS:
        from utils.lightvae_5b_wrapper import LightVAE5BWrapper

        return LightVAE5BWrapper(vae_path=_MG_LIGHTVAE_DEFAULT_PATHS[vae_type])

    raise ValueError(
        f"Unknown vae_type '{vae_type}'. "
        "Expected one of: wan, mg_lightvae, mg_lightvae_v2."
    )
