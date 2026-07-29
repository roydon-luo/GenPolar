"""Stage-I training for the Stokes-informed diffusion teacher."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataloader.Polarization_Dataset import Polarization_Dataset
from module.genpolar import (
    BASE_MODEL,
    alpha_sigma,
    build_components,
    decode_stokes,
    encode_stokes,
    load_checkpoint,
    load_model_weights,
    output_to_clean,
    physics_loss,
    predict_model_output,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--data-config", type=Path, default=Path("data_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/stage1"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--physics-weight", type=float, default=0.5)
    parser.add_argument("--aop-weight", type=float, default=0.5)
    parser.add_argument("--dolp-threshold", type=float, default=0.05)
    parser.add_argument(
        "--prediction-type",
        choices=("sample", "epsilon", "v_prediction"),
        default="sample",
        help="The paper uses direct clean-latent ('sample') prediction.",
    )
    parser.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-xformers", action="store_true")
    return parser.parse_args()


def save_training_checkpoint(
    path: Path,
    accelerator: Accelerator,
    unet: torch.nn.Module,
    controlnet: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model_role": "stage1_teacher",
        "prediction_type": args.prediction_type,
        "stokes_scale": args.stokes_scale,
        "epoch": epoch,
        "unet_state_dict": accelerator.unwrap_model(unet).state_dict(),
        "controlnet_state_dict": accelerator.unwrap_model(controlnet).state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    torch.manual_seed(args.seed + accelerator.process_index)

    tokenizer, text_encoder, vae, unet, controlnet, scheduler = build_components(
        args.base_model
    )
    assert controlnet is not None
    text_encoder.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    if args.enable_xformers:
        unet.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()

    optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(controlnet.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        load_model_weights(unet, controlnet, checkpoint)
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1

    config = OmegaConf.load(args.data_config)
    args.stokes_scale = float(config.data.train.params.get("stokes_scale", 1.0))
    dataset = Polarization_Dataset(config.data.train.params, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if __import__("os").name == "nt" else args.num_workers,
        pin_memory=True,
    )
    unet, controlnet, optimizer, dataloader = accelerator.prepare(
        unet, controlnet, optimizer, dataloader
    )
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    for epoch in range(start_epoch, args.epochs):
        unet.train()
        controlnet.train()
        progress = tqdm(
            dataloader,
            disable=not accelerator.is_local_main_process,
            desc=f"Stage I {epoch + 1}/{args.epochs}",
        )
        for batch in progress:
            if batch is None:
                continue
            target = batch["polarization"].to(accelerator.device)
            condition = batch["rgb"].to(accelerator.device)
            input_ids = batch["input_ids"].to(accelerator.device)

            with torch.no_grad():
                text = text_encoder(input_ids)[0]
                clean, _ = encode_stokes(vae, target)
                noise = torch.randn_like(clean)
                timesteps = torch.randint(
                    scheduler.config.num_train_timesteps,
                    (clean.shape[0],),
                    device=clean.device,
                )
                noisy = scheduler.add_noise(clean, noise, timesteps)

            with accelerator.accumulate(unet, controlnet):
                model_output = predict_model_output(
                    unet, controlnet, noisy, timesteps, text, condition
                )
                alpha, sigma = alpha_sigma(scheduler, timesteps, noisy)
                clean_prediction = output_to_clean(
                    noisy, model_output, alpha, sigma, args.prediction_type
                )
                diffusion = F.mse_loss(clean_prediction.float(), clean.float())
                decoded = decode_stokes(vae, clean_prediction)
                physical, terms = physics_loss(
                    decoded,
                    target,
                    condition,
                    aop_weight=args.aop_weight,
                    dolp_threshold=args.dolp_threshold,
                    stokes_scale=args.stokes_scale,
                )
                loss = diffusion + args.physics_weight * physical
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        list(unet.parameters()) + list(controlnet.parameters()), 1.0
                    )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            progress.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                diff=f"{diffusion.detach().item():.4f}",
                aop=f"{terms['aop'].detach().item():.4f}",
            )

        save_training_checkpoint(
            args.output_dir / "checkpoint_latest.pth",
            accelerator,
            unet,
            controlnet,
            optimizer,
            epoch,
            args,
        )


if __name__ == "__main__":
    main()
