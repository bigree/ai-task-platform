"""
cranio_diff/train.py

Training loop for Cranio-Diff.

Hyperparameters from Section 5.1:
    lr           = 1e-4
    batch_size   = 14
    weight_decay = 1e-5
    optimizer    = AdamW
    epochs       = 350
    image_size   = 512
    λ1 = λ2      = 0.20

Usage:
    python -m cranio_diff.train --data-root ./data/s2f --epochs 350
"""

from __future__ import annotations
import argparse
import logging
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .model import CranioDiff
from .losses import CranioDiffLoss
from .dataset import S2FDataset, SyntheticS2FDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ── Training step ─────────────────────────────────────────────────────────────

def train_step(
    model: CranioDiff,
    criterion: CranioDiffLoss,
    batch: dict,
    optimizer: AdamW,
    device: str,
    decode_for_perceptual: bool = True,
) -> dict[str, float]:
    """
    Single training step.

    1. Forward pass: L_diff from UNet noise prediction
    2. Optionally decode ẑ_0 → x̂_0 for perceptual + identity losses
    3. Compute total loss L = L_diff + λ1·L_LPIPS + λ2·L_id
    4. Backward + optimizer step
    """
    skulls  = batch["skull"].to(device)
    faces   = batch["face"].to(device)
    prompts = batch["prompt"]

    optimizer.zero_grad()

    # Diffusion loss (Eq. 8)
    l_diff = model(faces, skulls, prompts)

    if decode_for_perceptual:
        # Decode to image space for perceptual/identity losses
        # In full training: run partial denoising to get ẑ_0
        # PoC simplification: add noise → one-step denoise → decode
        with torch.no_grad():
            z0 = model.encode_image(faces)
            noise = torch.randn_like(z0)
            t = torch.full((z0.shape[0],), 50, device=device, dtype=torch.long)
            z_t = model.add_noise(z0, noise, t)
            text_emb = model.encode_text(prompts)
            down_res, mid_res = model.controlnet(
                z_t, t,
                encoder_hidden_states=text_emb,
                controlnet_cond=skulls.to(model.dtype),
                return_dict=False,
            )
            noise_pred = model.unet(
                z_t, t,
                encoder_hidden_states=text_emb,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            ).sample
            # One-step estimate of z_0 from noise prediction
            alpha_t = model.noise_scheduler.alphas_cumprod[50].to(device)
            z0_est = (z_t - (1 - alpha_t).sqrt() * noise_pred) / alpha_t.sqrt()
        generated_faces = model.decode_latent(z0_est).float()
        real_faces = faces.float()

        total_loss, breakdown = criterion(l_diff, generated_faces, real_faces)
    else:
        # Fast mode: skip perceptual/identity (useful for early training)
        total_loss = l_diff
        breakdown = {"loss_diff": l_diff.item(), "loss_total": l_diff.item()}

    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        list(model.controlnet.parameters()) + list(model.unet.parameters()),
        max_norm=1.0,
    )
    optimizer.step()

    return breakdown


# ── Main training loop ────────────────────────────────────────────────────────

def train(
    data_root: str | None,
    output_dir: str = "./checkpoints",
    epochs: int = 350,
    batch_size: int = 14,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    lambda_lpips: float = 0.20,
    lambda_id: float = 0.20,
    device: str = "cuda",
    synthetic: bool = False,
    save_every: int = 50,
    log_every: int = 10,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ── Dataset ──────────────────────────────────────────────────────────────
    if synthetic or data_root is None:
        log.info("Using synthetic dataset (no real data).")
        train_dataset = SyntheticS2FDataset(n_samples=128)
        val_dataset   = SyntheticS2FDataset(n_samples=16)
    else:
        train_dataset = S2FDataset(data_root, split="train", augment=True)
        val_dataset   = S2FDataset(data_root, split="test",  augment=False)

    # num_workers=0 on macOS/CPU/MPS to avoid multiprocessing hang
    num_workers = 0 if device in ("cpu", "mps") else 4
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False,
    )
    log.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    log.info("Loading Cranio-Diff model...")
    model = CranioDiff(device=device, dtype=torch.float16)

    # ── Optimizer (AdamW, Section 5.1) ────────────────────────────────────────
    trainable_params = (
        list(model.controlnet.parameters()) +
        list(model.unet.parameters())
    )
    optimizer = AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = CranioDiffLoss(
        lambda_lpips=lambda_lpips,
        lambda_id=lambda_id,
        device=device,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info(f"Starting training: {epochs} epochs, lr={lr}, batch={batch_size}")
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for step, batch in enumerate(train_loader):
            # Use perceptual loss from epoch 10 onward (training stability)
            use_perceptual = epoch >= 10
            breakdown = train_step(
                model, criterion, batch, optimizer, device,
                decode_for_perceptual=use_perceptual,
            )
            epoch_losses.append(breakdown["loss_total"])
            global_step += 1

            if global_step % log_every == 0:
                log.info(
                    f"Epoch {epoch:03d} Step {step:04d} | "
                    f"total={breakdown['loss_total']:.4f} "
                    f"diff={breakdown.get('loss_diff', 0):.4f} "
                    f"lpips={breakdown.get('loss_lpips', 0):.4f} "
                    f"id={breakdown.get('loss_id', 0):.4f}"
                )

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        log.info(f"[Epoch {epoch:03d}] avg_loss={avg_loss:.4f}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if epoch % save_every == 0 or epoch == epochs:
            ckpt_dir = output_path / f"epoch_{epoch:04d}"
            model.controlnet.save_pretrained(str(ckpt_dir / "controlnet"))
            model.unet.save_pretrained(str(ckpt_dir / "unet"))
            log.info(f"Saved checkpoint → {ckpt_dir}")

    log.info("Training complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Cranio-Diff")
    parser.add_argument("--data-root",    type=str, default=None)
    parser.add_argument("--output-dir",   type=str, default="./checkpoints")
    parser.add_argument("--epochs",       type=int, default=350)
    parser.add_argument("--batch-size",   type=int, default=14)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-lpips", type=float, default=0.20)
    parser.add_argument("--lambda-id",    type=float, default=0.20)
    parser.add_argument("--device",       type=str, default="cuda")
    parser.add_argument("--synthetic",    action="store_true",
                        help="Use synthetic data (for pipeline testing without real dataset)")
    parser.add_argument("--save-every",   type=int, default=50)
    args = parser.parse_args()

    train(
        data_root    = args.data_root,
        output_dir   = args.output_dir,
        epochs       = args.epochs,
        batch_size   = args.batch_size,
        lr           = args.lr,
        weight_decay = args.weight_decay,
        lambda_lpips = args.lambda_lpips,
        lambda_id    = args.lambda_id,
        device       = args.device,
        synthetic    = args.synthetic,
        save_every   = args.save_every,
    )
