"""Stage-II one-step DMD training with end-to-end VAE-encoder LoRA."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import UNet2DConditionModel
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from omegaconf import OmegaConf
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataloader.Polarization_Dataset import Polarization_Dataset
from module.genpolar import (
    BASE_MODEL,
    TEACHER_FILENAME,
    adapt_unet_io,
    alpha_sigma,
    build_components,
    checkpoint_metadata,
    decode_stokes,
    latent_scale,
    load_checkpoint,
    load_model_weights,
    output_to_clean,
    physics_loss,
    predict_model_output,
    resolve_release_checkpoint,
    score_from_clean,
    strip_module_prefix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument(
        "--teacher",
        type=Path,
        help="Stage-I checkpoint. Omit to download the public teacher.",
    )
    parser.add_argument("--data-config", type=Path, default=Path("data_config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/stage2"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-lora", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dmd-weight", type=float, default=1.0)
    parser.add_argument("--physics-weight", type=float, default=1.0)
    parser.add_argument("--aop-weight", type=float, default=0.5)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--dolp-threshold", type=float, default=0.05)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument(
        "--mixed-precision", choices=("no", "fp16", "bf16"), default="fp16"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable-xformers", action="store_true")
    return parser.parse_args()


def build_unet(base_model: str) -> UNet2DConditionModel:
    return adapt_unet_io(
        UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    )


def encode_with_lora(
    encoder: torch.nn.Module,
    vae: torch.nn.Module,
    stokes: torch.Tensor,
) -> tuple[torch.Tensor, DiagonalGaussianDistribution]:
    batch = stokes.shape[0]
    images = torch.cat(stokes.chunk(2, dim=1), dim=0)
    moments = vae.quant_conv(encoder(images))
    posterior = DiagonalGaussianDistribution(moments)
    first, second = (posterior.sample() * latent_scale(vae)).split(batch, dim=0)
    return torch.cat([first, second], dim=1), posterior


def predict_clean(
    unet: torch.nn.Module,
    controlnet: torch.nn.Module,
    scheduler: torch.nn.Module,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    text: torch.Tensor,
    condition: torch.Tensor,
    prediction_type: str,
) -> torch.Tensor:
    output = predict_model_output(unet, controlnet, noisy, timesteps, text, condition)
    alpha, sigma = alpha_sigma(scheduler, timesteps, noisy)
    return output_to_clean(noisy, output, alpha, sigma, prediction_type)


def freeze(module: torch.nn.Module) -> torch.nn.Module:
    return module.requires_grad_(False).eval()


def save_checkpoint(
    accelerator: Accelerator,
    output_dir: Path,
    epoch: int,
    student: torch.nn.Module,
    fake: torch.nn.Module,
    controlnet: torch.nn.Module,
    encoder: torch.nn.Module,
    optimizer_generator: torch.optim.Optimizer,
    optimizer_fake: torch.optim.Optimizer,
    prediction_type: str,
    stokes_scale: float,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 2,
        "model_role": "one_step_inference",
        "prediction_type": prediction_type,
        "stokes_scale": stokes_scale,
        "epoch": epoch,
        "unet_state_dict": accelerator.unwrap_model(student).state_dict(),
        "controlnet_state_dict": controlnet.state_dict(),
        "fake_unet_state_dict": accelerator.unwrap_model(fake).state_dict(),
        "optimizer_generator_state_dict": optimizer_generator.state_dict(),
        "optimizer_fake_state_dict": optimizer_fake.state_dict(),
    }
    target = output_dir / "checkpoint_latest.pth"
    temporary = target.with_suffix(".pth.tmp")
    torch.save(payload, temporary)
    temporary.replace(target)

    adapter_target = output_dir / "vae_encoder_lora"
    accelerator.unwrap_model(encoder).save_pretrained(adapter_target)


def main() -> None:
    args = parse_args()
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    torch.manual_seed(args.seed + accelerator.process_index)

    tokenizer, text_encoder, vae, teacher, controlnet, scheduler = build_components(
        args.base_model
    )
    assert controlnet is not None
    teacher_path = resolve_release_checkpoint(
        str(args.teacher) if args.teacher else None, TEACHER_FILENAME
    )
    teacher_checkpoint = load_checkpoint(teacher_path)
    load_model_weights(teacher, controlnet, teacher_checkpoint)
    prediction_type, stokes_scale = checkpoint_metadata(teacher_checkpoint)

    student = build_unet(args.base_model)
    fake = build_unet(args.base_model)
    teacher_state = strip_module_prefix(teacher_checkpoint["unet_state_dict"])
    student.load_state_dict(teacher_state, strict=True)
    fake.load_state_dict(teacher_state, strict=True)
    start_epoch = 0
    resume_checkpoint = None
    if args.resume:
        resume_checkpoint = load_checkpoint(args.resume)
        student.load_state_dict(
            strip_module_prefix(resume_checkpoint["unet_state_dict"]), strict=True
        )
        fake.load_state_dict(
            strip_module_prefix(resume_checkpoint["fake_unet_state_dict"]), strict=True
        )
        resume_prediction, resume_scale = checkpoint_metadata(resume_checkpoint)
        if (resume_prediction, resume_scale) != (prediction_type, stokes_scale):
            raise ValueError("Resume checkpoint metadata does not match the teacher")
        start_epoch = int(resume_checkpoint.get("epoch", -1)) + 1

    freeze(text_encoder)
    freeze(teacher)
    freeze(controlnet)
    freeze(vae)
    lora = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        lora_dropout=0.0,
        bias="none",
        target_modules=[
            "conv_in",
            "conv1",
            "conv2",
            "conv_shortcut",
            "q",
            "k",
            "v",
            "to_out.0",
        ],
    )
    if args.resume:
        adapter_path = args.resume_lora or args.resume.parent / "vae_encoder_lora"
        encoder = PeftModel.from_pretrained(
            vae.encoder, adapter_path, is_trainable=True
        )
    else:
        encoder = get_peft_model(vae.encoder, lora)

    if args.enable_xformers:
        teacher.enable_xformers_memory_efficient_attention()
        student.enable_xformers_memory_efficient_attention()
        fake.enable_xformers_memory_efficient_attention()
        controlnet.enable_xformers_memory_efficient_attention()

    generator_parameters = list(student.parameters()) + [
        parameter for parameter in encoder.parameters() if parameter.requires_grad
    ]
    optimizer_generator = torch.optim.AdamW(
        generator_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    optimizer_fake = torch.optim.AdamW(
        fake.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    if resume_checkpoint is not None:
        optimizer_generator.load_state_dict(
            resume_checkpoint["optimizer_generator_state_dict"]
        )
        optimizer_fake.load_state_dict(resume_checkpoint["optimizer_fake_state_dict"])

    config = OmegaConf.load(args.data_config)
    config.data.train.params.stokes_scale = stokes_scale
    dataset = Polarization_Dataset(config.data.train.params, tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if __import__("os").name == "nt" else args.num_workers,
        pin_memory=True,
    )
    student, fake, encoder, optimizer_generator, optimizer_fake, dataloader = (
        accelerator.prepare(
            student,
            fake,
            encoder,
            optimizer_generator,
            optimizer_fake,
            dataloader,
        )
    )
    for module in (text_encoder, vae, teacher, controlnet):
        module.to(accelerator.device)

    max_timestep = scheduler.config.num_train_timesteps - 1
    for epoch in range(start_epoch, args.epochs):
        student.train()
        fake.train()
        encoder.train()
        progress = tqdm(
            dataloader,
            disable=not accelerator.is_local_main_process,
            desc=f"Stage II {epoch + 1}/{args.epochs}",
        )
        for batch in progress:
            if batch is None:
                continue
            target = batch["polarization"].to(accelerator.device)
            condition = batch["rgb"].to(accelerator.device)
            input_ids = batch["input_ids"].to(accelerator.device)
            batch_size = target.shape[0]

            with torch.no_grad():
                text = text_encoder(input_ids)[0]
            clean_encoded, posterior = encode_with_lora(encoder, vae, target)
            initial_noise = torch.randn_like(clean_encoded)
            t_star = torch.full(
                (batch_size,), max_timestep, device=target.device, dtype=torch.long
            )
            noisy_encoded = scheduler.add_noise(clean_encoded, initial_noise, t_star)

            # Train the online fake score model on the current generated distribution.
            with accelerator.accumulate(fake):
                with torch.no_grad():
                    generated_target = predict_clean(
                        student,
                        controlnet,
                        scheduler,
                        noisy_encoded.detach(),
                        t_star,
                        text,
                        condition,
                        prediction_type,
                    )
                    fake_t = torch.randint(
                        max_timestep + 1, (batch_size,), device=target.device
                    )
                    fake_noise = torch.randn_like(generated_target)
                    fake_noisy = scheduler.add_noise(
                        generated_target, fake_noise, fake_t
                    )
                fake_clean = predict_clean(
                    fake,
                    controlnet,
                    scheduler,
                    fake_noisy,
                    fake_t,
                    text,
                    condition,
                    prediction_type,
                )
                fake_loss = F.mse_loss(fake_clean.float(), generated_target.float())
                accelerator.backward(fake_loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(fake.parameters(), 1.0)
                optimizer_fake.step()
                optimizer_fake.zero_grad(set_to_none=True)

            # DMD affects theta only; physics affects theta and phi; KL affects phi only.
            with accelerator.accumulate(student, encoder):
                generated_dmd = predict_clean(
                    student,
                    controlnet,
                    scheduler,
                    noisy_encoded.detach(),
                    t_star,
                    text,
                    condition,
                    prediction_type,
                )
                score_t = torch.randint(
                    1, max_timestep + 1, (batch_size,), device=target.device
                )
                score_noise = torch.randn_like(generated_dmd)
                score_noisy = scheduler.add_noise(generated_dmd, score_noise, score_t)
                alpha, sigma = alpha_sigma(scheduler, score_t, score_noisy)
                with torch.no_grad():
                    real_clean = predict_clean(
                        teacher,
                        controlnet,
                        scheduler,
                        score_noisy,
                        score_t,
                        text,
                        condition,
                        prediction_type,
                    )
                    fake_clean = predict_clean(
                        fake,
                        controlnet,
                        scheduler,
                        score_noisy,
                        score_t,
                        text,
                        condition,
                        prediction_type,
                    )
                    score_gap = score_from_clean(
                        score_noisy, fake_clean, alpha, sigma
                    ) - score_from_clean(score_noisy, real_clean, alpha, sigma)
                dmd = (score_noisy * score_gap.detach()).mean()

                generated_physics = predict_clean(
                    student,
                    controlnet,
                    scheduler,
                    noisy_encoded,
                    t_star,
                    text,
                    condition,
                    prediction_type,
                )
                decoded = decode_stokes(vae, generated_physics)
                physical, terms = physics_loss(
                    decoded,
                    target,
                    condition,
                    aop_weight=args.aop_weight,
                    dolp_threshold=args.dolp_threshold,
                    stokes_scale=stokes_scale,
                )
                kl = posterior.kl().mean()
                loss = (
                    args.dmd_weight * dmd
                    + args.physics_weight * physical
                    + args.kl_weight * kl
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(generator_parameters, 1.0)
                optimizer_generator.step()
                optimizer_generator.zero_grad(set_to_none=True)

            progress.set_postfix(
                loss=f"{loss.detach().item():.4f}",
                dmd=f"{dmd.detach().item():.4f}",
                phy=f"{physical.detach().item():.4f}",
                aop=f"{terms['aop'].detach().item():.4f}",
                fake=f"{fake_loss.detach().item():.4f}",
            )

        save_checkpoint(
            accelerator,
            args.output_dir,
            epoch,
            student,
            fake,
            controlnet,
            encoder,
            optimizer_generator,
            optimizer_fake,
            prediction_type,
            stokes_scale,
        )


if __name__ == "__main__":
    main()
