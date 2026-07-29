"""Shared GenPolar model, latent, and physics utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DDPMScheduler,
    UNet2DConditionModel,
)
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
HF_REPO_ID = "Roydon728/GenPolar"
ONE_STEP_FILENAME = "genpolar_one_step.pth"
TEACHER_FILENAME = "genpolar_stage1_teacher.pth"
ONE_STEP_URL = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{ONE_STEP_FILENAME}"
TEACHER_URL = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{TEACHER_FILENAME}"


def adapt_unet_io(
    unet: UNet2DConditionModel, channels: int = 8
) -> UNet2DConditionModel:
    """Expand an SD1.5 UNet from four to eight latent input/output channels."""
    if unet.config.in_channels != channels:
        old = unet.conv_in
        if channels % old.in_channels:
            raise ValueError(
                f"Cannot expand {old.in_channels} input channels to {channels}"
            )
        new = nn.Conv2d(
            channels,
            old.out_channels,
            old.kernel_size,
            stride=old.stride,
            padding=old.padding,
        )
        repeat = channels // old.in_channels
        with torch.no_grad():
            new.weight.copy_(old.weight.repeat(1, repeat, 1, 1) / repeat)
            new.bias.copy_(old.bias)
        unet.conv_in = new
        unet.register_to_config(in_channels=channels)

    if unet.config.out_channels != channels:
        old = unet.conv_out
        if channels % old.out_channels:
            raise ValueError(
                f"Cannot expand {old.out_channels} output channels to {channels}"
            )
        new = nn.Conv2d(
            old.in_channels,
            channels,
            old.kernel_size,
            stride=old.stride,
            padding=old.padding,
        )
        repeat = channels // old.out_channels
        with torch.no_grad():
            new.weight.copy_(old.weight.repeat(repeat, 1, 1, 1))
            new.bias.copy_(old.bias.repeat(repeat))
        unet.conv_out = new
        unet.register_to_config(out_channels=channels)
    return unet


def build_components(
    base_model: str = BASE_MODEL,
    *,
    dtype: torch.dtype = torch.float32,
    create_controlnet: bool = True,
) -> tuple[
    CLIPTokenizer,
    CLIPTextModel,
    AutoencoderKL,
    UNet2DConditionModel,
    ControlNetModel | None,
    DDPMScheduler,
]:
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        base_model, subfolder="text_encoder", torch_dtype=dtype
    )
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype)
    unet = adapt_unet_io(
        UNet2DConditionModel.from_pretrained(
            base_model, subfolder="unet", torch_dtype=dtype
        )
    )
    scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")
    controlnet = ControlNetModel.from_unet(unet) if create_controlnet else None
    return tokenizer, text_encoder, vae, unet, controlnet, scheduler


def latent_scale(vae: AutoencoderKL) -> float:
    return float(getattr(vae.config, "scaling_factor", 0.18215))


def encode_stokes(
    vae: AutoencoderKL,
    stokes: torch.Tensor,
    *,
    sample: bool = True,
) -> tuple[torch.Tensor, Any]:
    """Encode Bx6 S1/S2 as two VAE batches and merge them into Bx8."""
    if stokes.shape[1] != 6:
        raise ValueError(f"Expected six Stokes channels, got {tuple(stokes.shape)}")
    batch = stokes.shape[0]
    posterior = vae.encode(torch.cat(stokes.chunk(2, dim=1), dim=0)).latent_dist
    latent = posterior.sample() if sample else posterior.mode()
    first, second = (latent * latent_scale(vae)).split(batch, dim=0)
    return torch.cat([first, second], dim=1), posterior


def decode_stokes(vae: AutoencoderKL, latent: torch.Tensor) -> torch.Tensor:
    """Decode Bx8 latent into Bx6 S1/S2 while preserving gradients to latent."""
    first, second = latent.chunk(2, dim=1)
    decoded = vae.decode(torch.cat([first, second], dim=0) / latent_scale(vae)).sample
    batch = latent.shape[0]
    return torch.cat(decoded.split(batch, dim=0), dim=1)


def alpha_sigma(
    scheduler: DDPMScheduler,
    timesteps: torch.Tensor,
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha_bar = scheduler.alphas_cumprod.to(
        device=reference.device, dtype=reference.dtype
    )[timesteps].view(-1, 1, 1, 1)
    return alpha_bar.sqrt(), (1.0 - alpha_bar).sqrt()


def predict_model_output(
    unet: UNet2DConditionModel,
    controlnet: ControlNetModel,
    sample: torch.Tensor,
    timesteps: torch.Tensor,
    text_embeddings: torch.Tensor,
    condition: torch.Tensor,
    *,
    conditioning_scale: float = 1.0,
) -> torch.Tensor:
    down, mid = controlnet(
        sample=sample,
        timestep=timesteps,
        encoder_hidden_states=text_embeddings,
        controlnet_cond=condition,
        conditioning_scale=conditioning_scale,
        return_dict=False,
    )
    return unet(
        sample,
        timesteps,
        encoder_hidden_states=text_embeddings,
        down_block_additional_residuals=down,
        mid_block_additional_residual=mid,
    ).sample


def output_to_clean(
    noisy: torch.Tensor,
    output: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    if prediction_type == "sample":
        return output
    if prediction_type == "epsilon":
        return (noisy - sigma * output) / alpha.clamp_min(1e-6)
    if prediction_type == "v_prediction":
        return alpha * noisy - sigma * output
    raise ValueError(f"Unsupported prediction type: {prediction_type}")


def score_from_clean(
    noisy: torch.Tensor,
    clean: torch.Tensor,
    alpha: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    return -(noisy - alpha * clean) / sigma.square().clamp_min(1e-6)


def polarization_maps(
    stokes: torch.Tensor,
    condition: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return channel-wise DoLP and pi-periodic AoP in radians."""
    s1, s2 = stokes.chunk(2, dim=1)
    s0 = condition + 1.0
    dolp = (s1.square() + s2.square() + 1e-8).sqrt() / s0.clamp_min(1e-6)
    aop = 0.5 * torch.atan2(s2, s1)
    return dolp.clamp(0.0, 1.0), aop


def physics_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    condition: torch.Tensor,
    *,
    aop_weight: float = 0.5,
    dolp_threshold: float = 0.05,
    stokes_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Equations (8)-(10): global Stokes L1 plus observable, pi-periodic AoP."""
    prediction = prediction / stokes_scale
    target = target / stokes_scale
    pred_s1, pred_s2 = prediction.chunk(2, dim=1)
    target_s1, target_s2 = target.chunk(2, dim=1)
    stokes_l1 = F.l1_loss(pred_s1, target_s1) + F.l1_loss(pred_s2, target_s2)

    _, pred_aop = polarization_maps(prediction, condition)
    target_dolp, target_aop = polarization_maps(target, condition)
    mask = target_dolp > dolp_threshold
    angular = (
        0.5
        * torch.atan2(
            torch.sin(2.0 * (pred_aop - target_aop)),
            torch.cos(2.0 * (pred_aop - target_aop)),
        ).abs()
    )
    aop = angular[mask].mean() if mask.any() else angular.sum() * 0.0
    total = stokes_l1 + aop_weight * aop
    return total, {"stokes": stokes_l1, "aop": aop}


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must be a dictionary: {path}")
    return payload


def load_model_weights(
    unet: UNet2DConditionModel,
    controlnet: ControlNetModel,
    checkpoint: dict[str, Any],
    *,
    unet_key: str = "unet_state_dict",
) -> None:
    unet_state = checkpoint.get(unet_key, checkpoint)
    if not isinstance(unet_state, dict):
        raise KeyError(f"Missing {unet_key}")
    unet.load_state_dict(strip_module_prefix(unet_state), strict=True)
    control_state = checkpoint.get("controlnet_state_dict")
    if not isinstance(control_state, dict):
        raise KeyError("Missing controlnet_state_dict")
    controlnet.load_state_dict(strip_module_prefix(control_state), strict=True)


def resolve_release_checkpoint(path: str | None, filename: str) -> Path:
    if path:
        return Path(path)
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO_ID, filename))


def checkpoint_metadata(
    checkpoint: dict[str, Any],
    *,
    legacy_prediction_type: str = "epsilon",
    legacy_stokes_scale: float = 5.0,
) -> tuple[str, float]:
    """Read release metadata; old uploaded checkpoints use known legacy defaults."""
    prediction_type = str(checkpoint.get("prediction_type", legacy_prediction_type))
    stokes_scale = float(checkpoint.get("stokes_scale", legacy_stokes_scale))
    return prediction_type, stokes_scale
