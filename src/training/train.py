"""Main training loop for spray chart diffusion."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SprayChartDataset
from src.model.diffusion import SprayChartDiffusion
from src.model.unet import ConditionalUNet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr_with_warmup(
    optimizer: optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> optim.lr_scheduler.LambdaLR:
    def _schedule(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, _schedule)


def save_checkpoint(
    path: Path,
    model: SprayChartDiffusion,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    val_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: SprayChartDiffusion,
    optimizer: optim.Optimizer,
    scheduler,
    device: torch.device,
) -> tuple[int, int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt["epoch"], ckpt["global_step"], ckpt["val_loss"]


def log_sample_images(
    model: SprayChartDiffusion,
    sample_batch: dict,
    step: int,
    out_dir: Path,
    device: torch.device,
    num: int = 4,
) -> None:
    import matplotlib.pyplot as plt

    model.eval()
    bat = sample_batch["batter_idx"][:num].to(device)
    sit = sample_batch["situation_code"][:num].to(device)
    with torch.no_grad():
        imgs = model.sample(bat, sit, num_inference_steps=50, device=device)
    imgs_np = imgs.cpu().numpy()[:, 0]   # (N, 64, 64)

    fig, axes = plt.subplots(1, num, figsize=(num * 3, 3))
    for i, ax in enumerate(axes):
        ax.imshow(imgs_np[i], cmap="hot", origin="upper")
        ax.axis("off")
        ax.set_title(f"batter {bat[i].item()}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"samples_step{step:07d}.png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    model.train()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    model: SprayChartDiffusion,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            bat = batch["batter_idx"].to(device)
            sit = batch["situation_code"].to(device)
            partial = batch.get("partial_image")
            if partial is not None:
                partial = partial.to(device)
            with autocast("cuda", enabled=use_amp):
                loss = model.loss(imgs, bat, sit, partial)
            total += loss.item() * imgs.size(0)
            count += imgs.size(0)
    model.train()
    return total / count


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config_path: str | Path = "configs/default.yaml") -> None:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    processed_dir = Path(cfg["data"]["processed_dir"])
    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    sample_dir = Path("samples")

    # Datasets
    inpaint = cfg["data"].get("inpaint_k_min", 5) > 0
    image_scale = float(cfg["data"].get("image_scale", 1.0))
    train_ds = SprayChartDataset(
        processed_dir, split="train", inpaint_mode=inpaint,
        inpaint_k_min=cfg["data"]["inpaint_k_min"],
        inpaint_k_max=cfg["data"]["inpaint_k_max"],
        image_scale=image_scale,
    )
    val_ds = SprayChartDataset(processed_dir, split="val", inpaint_mode=inpaint,
                               inpaint_k_min=cfg["data"]["inpaint_k_min"],
                               inpaint_k_max=cfg["data"]["inpaint_k_max"],
                               image_scale=image_scale)

    # Embedding table size = number of training batters + 1.
    # Index 0 is reserved as the population-prior / unknown-batter vector.
    # Held-out batters are NOT in the table (they map to index 0 at inference).
    with open(processed_dir / "batter_id_map.json") as f:
        id_map = json.load(f)
    # max index + 1 gives the correct table size whether or not held-out entries
    # (which map to 0) are present in the JSON.
    num_batters = max(int(v) for v in id_map.values()) + 1
    print(f"Num batters: {num_batters}")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=cfg["training"]["num_workers"],
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"] * 2,
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
    )

    # Model
    mcfg = cfg["model"]
    unet = ConditionalUNet(
        num_batters=num_batters,
        base_channels=mcfg["base_channels"],
        channel_multipliers=tuple(mcfg["channel_multipliers"]),
        batter_embed_dim=mcfg["batter_embedding_dim"],
        situation_embed_dim=mcfg["situation_embedding_dim"],
        time_embed_dim=mcfg["timestep_embedding_dim"],
        num_heads=mcfg["attention_heads"],
        dropout=mcfg["dropout"],
        inpaint_mode=inpaint,
    ).to(device)

    dcfg = cfg["diffusion"]
    model = SprayChartDiffusion(
        unet=unet,
        num_timesteps=dcfg["num_timesteps"],
        noise_schedule="squaredcos_cap_v2",
        cfg_dropout_prob=dcfg["cfg_dropout_prob"],
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {param_count / 1e6:.1f}M")

    # Optimiser and scheduler
    tcfg = cfg["training"]
    optimizer = optim.AdamW(
        model.parameters(),
        lr=tcfg["learning_rate"],
        weight_decay=tcfg["weight_decay"],
    )
    total_steps = tcfg["epochs"] * len(train_loader)
    lr_scheduler = cosine_lr_with_warmup(optimizer, tcfg["warmup_steps"], total_steps)

    # Mixed precision (CUDA only — MPS and CPU don't support torch.cuda.amp)
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda", enabled=use_amp)

    # Resume from latest checkpoint if present
    start_epoch, global_step, best_val = 0, 0, float("inf")
    ckpts = sorted(ckpt_dir.glob("epoch_*.pt")) if ckpt_dir.exists() else []
    if ckpts:
        print(f"Resuming from {ckpts[-1]}")
        start_epoch, global_step, best_val = load_checkpoint(
            ckpts[-1], model, optimizer, lr_scheduler, device
        )
        start_epoch += 1

    # Grab a fixed sample batch for visualisation
    sample_batch = next(iter(val_loader))

    # Training loop
    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        epoch_loss, t0 = 0.0, time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        for batch in pbar:
            imgs = batch["image"].to(device)
            bat = batch["batter_idx"].to(device)
            sit = batch["situation_code"].to(device)
            partial = batch.get("partial_image")
            if partial is not None:
                partial = partial.to(device)

            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=use_amp):
                loss = model.loss(imgs, bat, sit, partial)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            epoch_loss += loss.item()
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

            if global_step % tcfg["sample_every_n_steps"] == 0:
                log_sample_images(model, sample_batch, global_step, sample_dir, device)

        avg_loss = epoch_loss / len(train_loader)
        val_loss = validate(model, val_loader, device, use_amp)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d} | train={avg_loss:.4f} | val={val_loss:.4f} | "
            f"time={elapsed:.0f}s | step={global_step}"
        )

        if (epoch + 1) % tcfg["save_every_n_epochs"] == 0 or val_loss < best_val:
            tag = "best" if val_loss < best_val else f"epoch_{epoch}"
            if val_loss < best_val:
                best_val = val_loss
                tag = "best"
            save_checkpoint(
                ckpt_dir / f"{tag}.pt",
                model, optimizer, lr_scheduler,
                epoch, global_step, val_loss,
            )

    print("Training complete.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(args.config)
