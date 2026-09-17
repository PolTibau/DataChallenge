"""
lstm/src/trainer.py — v3.1
--------------------------
Entrenament i inferència del Transformer.

v3.1:
  - Pèrdua MAE per defecte (alineada amb la mètrica del challenge)
  - Màscares de validació congelades (early stopping estable)
  - TTA per soroll als punts observats (no shifts cíclics)
"""

import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from lstm.src.config import (
    BATCH_SIZE, N_EPOCHS, LR, LR_WARMUP_STEPS,
    WEIGHT_DECAY, GRAD_CLIP, PATIENCE, VAL_SPLIT,
    BEST_CKPT, RANDOM_SEED,
    MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN, MASK_BLOCK_PROB,
    EMPIRICAL_MASK_PROB,
    TTA_N_PASSES, TTA_NOISE_STD,
)
from lstm.src.dataset import TimeSeriesDataset, generate_mask
from lstm.src.model   import TemporalMaskedAutoencoder, count_parameters


# ── Pèrdues ───────────────────────────────────────────────────────────────────

def masked_mae_loss(pred: torch.Tensor,
                    target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """MAE calculat NOMÉS sobre les posicions emmascarades."""
    errors = torch.abs(pred - target) * mask.float()
    return errors.sum() / mask.float().sum().clamp(min=1)


def masked_hybrid_loss(pred: torch.Tensor,
                       target: torch.Tensor,
                       mask: torch.Tensor,
                       alpha: float = 0.7) -> torch.Tensor:
    """Pèrdua híbrida opcional: alpha × MAE + (1-alpha) × RMSE."""
    diff = pred - target
    mae  = (torch.abs(diff) * mask.float()).sum() / mask.float().sum().clamp(min=1)
    mse  = ((diff ** 2)     * mask.float()).sum() / mask.float().sum().clamp(min=1)
    return alpha * mae + (1 - alpha) * torch.sqrt(mse)


# ── Scheduler ─────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Warmup lineal + cosine annealing."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, lr: float):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.lr           = lr
        self.step_count   = 0

    def step(self):
        self.step_count += 1
        s = self.step_count
        if s <= self.warmup_steps:
            lr = self.lr * s / self.warmup_steps
        else:
            progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = max(self.lr * 0.5 * (1 + np.cos(np.pi * progress)), self.lr / 100)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr


def build_fixed_masks(n: int, t: int, mask_dist: dict | None,
                      seed: int) -> np.ndarray:
    """Genera un conjunt fix de màscares (N, T) per a validació estable."""
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, t), dtype=bool)
    for i in range(n):
        masks[i] = generate_mask(
            t, MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN, MASK_BLOCK_PROB,
            rng, mask_dist, EMPIRICAL_MASK_PROB,
        )
    return masks


# ── Entrenament ───────────────────────────────────────────────────────────────

def train(curves_norm: np.ndarray,
          device: torch.device,
          n_epochs_override: int | None = None,
          ckpt_path: Path | None = None,
          use_hybrid_loss: bool = False,
          curves_raw: np.ndarray | None = None,
          global_stats: dict | None = None,
          mask_dist: dict | None = None) -> TemporalMaskedAutoencoder:
    """
    Entrena el Transformer.

    use_hybrid_loss=False per defecte: la mètrica del challenge és MAE en watts.
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n_epochs  = n_epochs_override if n_epochs_override is not None else N_EPOCHS
    best_ckpt = ckpt_path if ckpt_path is not None else BEST_CKPT
    last_ckpt = best_ckpt.parent / f"last_{best_ckpt.name}"
    loss_fn   = masked_hybrid_loss if use_hybrid_loss else masked_mae_loss
    loss_name = "MAE+RMSE" if use_hybrid_loss else "MAE"

    # DataLoaders
    print("  Building training dataset (this includes profile computation)...")
    full_ds = TimeSeriesDataset(curves_norm, augment=True, seed=RANDOM_SEED,
                                curves_raw=curves_raw, global_stats=global_stats,
                                mask_dist=mask_dist)
    n_val   = max(1, int(len(full_ds) * VAL_SPLIT))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED))

    # Validació: màscares FIXES (mateixes cada època → early stopping fiable)
    print("  Building validation dataset with frozen masks...")
    fixed_val_masks = build_fixed_masks(
        len(curves_norm), curves_norm.shape[1], mask_dist, seed=RANDOM_SEED + 7)
    val_full = TimeSeriesDataset(
        curves_norm, augment=False, seed=RANDOM_SEED + 1,
        curves_raw=curves_raw, global_stats=global_stats,
        mask_dist=mask_dist, fixed_masks=fixed_val_masks)
    val_ds.dataset = val_full

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"\nTrain: {n_train:,} corbes  |  Val: {n_val:,} corbes")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Loss: {loss_name}")
    print(f"Mask distribution: {'empirical (60%) + uniform (40%)' if mask_dist else 'uniform'}")
    print("Val masks: FROZEN (reproducible early stopping)")

    # Model
    model = TemporalMaskedAutoencoder().to(device)
    n_params = count_parameters(model)
    print(f"Paràmetres entrenables: {n_params:,}  ({n_params/1e6:.1f}M)")

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR,
                                    weight_decay=WEIGHT_DECAY)
    total_steps = n_epochs * len(train_loader)
    scheduler   = WarmupCosineScheduler(optimizer, LR_WARMUP_STEPS, total_steps, LR)

    best_val_mae  = float("inf")
    patience_left = PATIENCE
    best_ckpt.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Val Loss':>10} "
          f"{'LR':>10} {'Time':>7} {'':>10}")
    print("─" * 64)

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            inp, tgt, mask = (batch["input"].to(device),
                              batch["target"].to(device),
                              batch["mask"].to(device))
            profile = batch.get("profile")
            if profile is not None:
                profile = profile.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inp, mask, profile), tgt, mask)
            loss.backward()
            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # Validació
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                inp, tgt, mask = (batch["input"].to(device),
                                  batch["target"].to(device),
                                  batch["mask"].to(device))
                profile = batch.get("profile")
                if profile is not None:
                    profile = profile.to(device)
                val_losses.append(loss_fn(model(inp, mask, profile), tgt, mask).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        lr_now     = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        if val_loss < best_val_mae:
            best_val_mae  = val_loss
            patience_left = PATIENCE
            torch.save(model.state_dict(), best_ckpt)
            kaggle_backup = Path('/kaggle/working') / best_ckpt.name
            if Path('/kaggle/working').exists():
                import shutil
                shutil.copy(best_ckpt, kaggle_backup)
            status = "✓ millor"
        else:
            patience_left -= 1
            status = f"({patience_left} left)"

        torch.save(model.state_dict(), last_ckpt)
        print(f"{epoch:>6} {train_loss:>11.4f} {val_loss:>10.4f} "
              f"{lr_now:>10.2e} {elapsed:>6.1f}s {status:>10}")

        if epoch % 5 == 0 or status == "✓ millor":
            backup_dir = Path("/kaggle/working")
            if backup_dir.exists():
                import shutil
                shutil.copy(best_ckpt, backup_dir / best_ckpt.name)
                shutil.copy(last_ckpt, backup_dir / last_ckpt.name)

        if patience_left == 0:
            print(f"\nEarly stopping a l'època {epoch}.")
            break

    print(f"\nMillor Val Loss ({loss_name}): {best_val_mae:.4f}")
    print(f"Checkpoint guardat a: {best_ckpt}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device,
                                     weights_only=True))
    return model


# ── Inferència ────────────────────────────────────────────────────────────────

def predict_batch(model: TemporalMaskedAutoencoder,
                  loader: DataLoader,
                  device: torch.device) -> np.ndarray:
    """Inferència simple (sense TTA)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            inp     = batch["input"].to(device)
            mask    = batch["mask"].to(device)
            profile = batch.get("profile")
            if profile is not None:
                profile = profile.to(device)
            preds.append(model(inp, mask, profile).cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_batch_tta(model: TemporalMaskedAutoencoder,
                      loader: DataLoader,
                      device: torch.device,
                      n_passes: int = TTA_N_PASSES,
                      noise_std: float = TTA_NOISE_STD) -> np.ndarray:
    """
    Test-Time Augmentation amb soroll als punts observats.

    NO usa shifts cíclics: aquests desalinejen el consum amb els embeddings
    de calendari (hora × tipus_dia), que són fixos per índex de patch.

    Passada 0: predicció neta.
    Passades 1..n-1: afegeix N(0, noise_std) només on mask=False (observats).
    Promitja totes les passades.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            inp     = batch["input"].to(device)
            mask    = batch["mask"].to(device)
            profile = batch.get("profile")
            if profile is not None:
                profile = profile.to(device)

            batch_preds = []
            for p in range(n_passes):
                if p == 0 or noise_std <= 0:
                    inp_p = inp
                else:
                    noise = torch.randn_like(inp) * noise_std
                    # Només pertorbem punts observats (mask=False)
                    inp_p = inp + noise * (~mask).float()
                batch_preds.append(model(inp_p, mask, profile).cpu().numpy())

            all_preds.append(np.mean(batch_preds, axis=0))

    return np.concatenate(all_preds, axis=0)
