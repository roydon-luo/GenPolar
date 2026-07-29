"""One-step GenPolar inference from an RGB intensity image."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import DDIMScheduler
from PIL import Image
from tqdm.auto import tqdm

from module.genpolar import (
    BASE_MODEL,
    ONE_STEP_FILENAME,
    TEACHER_FILENAME,
    alpha_sigma,
    build_components,
    checkpoint_metadata,
    decode_stokes,
    load_checkpoint,
    load_model_weights,
    output_to_clean,
    polarization_maps,
    predict_model_output,
    resolve_release_checkpoint,
)

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="RGB image or a directory of images")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--checkpoint", type=Path, help="Omit to download the public weight"
    )
    parser.add_argument("--model", choices=("one-step", "stage1"), default="one-step")
    parser.add_argument("--steps", type=int, default=20, help="Stage-I DDIM steps")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--prompt", default="denoised polarized images")
    parser.add_argument("--conditioning-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument(
        "--prediction-type",
        choices=("auto", "sample", "epsilon", "v_prediction"),
        default="auto",
    )
    parser.add_argument(
        "--stokes-scale",
        type=float,
        help="Override checkpoint output normalization.",
    )
    return parser.parse_args()


def image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    paths = sorted(
        item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No supported images under {path}")
    return paths


def preprocess(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    width, height = width // 8 * 8, height // 8 * 8
    if width == 0 or height == 0:
        raise ValueError(f"Image is smaller than 8x8: {path}")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device, dtype)


def save_rgb(path: Path, tensor: torch.Tensor) -> None:
    array = (
        tensor.detach()
        .float()
        .clamp(0.0, 1.0)
        .squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    Image.fromarray(np.round(array * 255.0).astype(np.uint8)).save(path)


def save_result(
    output_dir: Path,
    stokes: torch.Tensor,
    condition: torch.Tensor,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    s1, s2 = stokes.chunk(2, dim=1)
    dolp, aop = polarization_maps(stokes, condition)
    np.savez_compressed(
        output_dir / "polarization.npz",
        s0=(condition + 1.0).squeeze(0).permute(1, 2, 0).float().cpu().numpy(),
        s1=s1.squeeze(0).permute(1, 2, 0).float().cpu().numpy(),
        s2=s2.squeeze(0).permute(1, 2, 0).float().cpu().numpy(),
        dolp=dolp.squeeze(0).permute(1, 2, 0).float().cpu().numpy(),
        aop=aop.squeeze(0).permute(1, 2, 0).float().cpu().numpy(),
    )
    save_rgb(output_dir / "s1.png", (s1 + 1.0) / 2.0)
    save_rgb(output_dir / "s2.png", (s2 + 1.0) / 2.0)
    save_rgb(output_dir / "dolp.png", dolp)
    save_rgb(output_dir / "aop.png", aop / torch.pi + 0.5)


@torch.inference_mode()
def infer(
    image: torch.Tensor,
    tokenizer: torch.nn.Module,
    text_encoder: torch.nn.Module,
    vae: torch.nn.Module,
    unet: torch.nn.Module,
    controlnet: torch.nn.Module,
    scheduler: torch.nn.Module,
    *,
    prompt: str,
    prediction_type: str,
    stokes_scale: float,
    conditioning_scale: float,
    seed: int,
    steps: int,
    one_step: bool,
) -> torch.Tensor:
    tokens = tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(image.device)
    text = text_encoder(tokens)[0]
    batch, _, height, width = image.shape
    generator = torch.Generator(device=image.device).manual_seed(seed)
    noisy = torch.randn(
        (batch, 8, height // 8, width // 8),
        generator=generator,
        device=image.device,
        dtype=image.dtype,
    )
    timesteps = torch.full(
        (batch,),
        scheduler.config.num_train_timesteps - 1,
        device=image.device,
        dtype=torch.long,
    )
    if one_step:
        output = predict_model_output(
            unet,
            controlnet,
            noisy,
            timesteps,
            text,
            image,
            conditioning_scale=conditioning_scale,
        )
        alpha, sigma = alpha_sigma(scheduler, timesteps, noisy)
        clean = output_to_clean(noisy, output, alpha, sigma, prediction_type)
    else:
        sampler = DDIMScheduler.from_config(scheduler.config)
        sampler.register_to_config(prediction_type=prediction_type)
        sampler.set_timesteps(steps, device=image.device)
        noisy = noisy * sampler.init_noise_sigma
        for timestep in sampler.timesteps:
            timestep_batch = timestep.expand(batch)
            output = predict_model_output(
                unet,
                controlnet,
                noisy,
                timestep_batch,
                text,
                image,
                conditioning_scale=conditioning_scale,
            )
            noisy = sampler.step(output, timestep, noisy).prev_sample
        clean = noisy
    # Restore the physical Stokes range used by legacy release checkpoints.
    # Legacy release weights used a 5x Stokes target; convert it back here.
    return decode_stokes(vae, clean) / stokes_scale


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = (
        torch.float16
        if args.precision == "fp16" and device.type == "cuda"
        else torch.float32
    )
    filename = ONE_STEP_FILENAME if args.model == "one-step" else TEACHER_FILENAME
    checkpoint_path = resolve_release_checkpoint(
        str(args.checkpoint) if args.checkpoint else None, filename
    )
    checkpoint = load_checkpoint(checkpoint_path)
    metadata_prediction, metadata_scale = checkpoint_metadata(checkpoint)
    prediction_type = (
        metadata_prediction if args.prediction_type == "auto" else args.prediction_type
    )
    stokes_scale = metadata_scale if args.stokes_scale is None else args.stokes_scale

    tokenizer, text_encoder, vae, unet, controlnet, scheduler = build_components(
        args.base_model, dtype=dtype
    )
    assert controlnet is not None
    load_model_weights(unet, controlnet, checkpoint)
    for model in (text_encoder, vae, unet, controlnet):
        model.to(device).eval()

    paths = image_paths(args.input)
    for index, path in enumerate(tqdm(paths, desc="GenPolar")):
        image = preprocess(path, device, dtype)
        prediction = infer(
            image,
            tokenizer,
            text_encoder,
            vae,
            unet,
            controlnet,
            scheduler,
            prompt=args.prompt,
            prediction_type=prediction_type,
            stokes_scale=stokes_scale,
            conditioning_scale=args.conditioning_scale,
            seed=args.seed + index,
            steps=args.steps,
            one_step=args.model == "one-step",
        )
        relative = (
            path.stem
            if args.input.is_file()
            else str(path.relative_to(args.input).with_suffix(""))
        )
        save_result(args.output / relative, prediction, image)

    print(f"Saved {len(paths)} result(s) to {args.output.resolve()}")
    print(
        f"checkpoint={checkpoint_path} model={args.model} "
        f"prediction_type={prediction_type} "
        f"stokes_scale={stokes_scale:g}"
    )


if __name__ == "__main__":
    main()
