"""
lstm/src/trainer.py
-------------------
Entrenament i inferència del Transformer.

  - Loss: MAE sobre posicions emmascarades únicament
  - Scheduler: warmup lineal + cosine annealing
  - Early stopping per validació MAE
  - Checkpointing del millor i de l'últim model
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from lstm.src.config import (
    BATCH_SIZE, N_EPOCHS, LR, LR_WARMUP_STEPS,
    WEIGHT_DECAY, GRAD_CLIP, PATIENCE, VAL_SPLIT,
    BEST_CKPT, LAST_CKPT, RANDOM_SEED,
)
from lstm.src.dataset import TimeSeriesDataset
from lstm.src.model   import TemporalMaskedAutoencoder, count_parameters


def masked_mae_loss(pred: torch.Tensor,
                    target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """MAE calculat NOMÉS sobre les posicions emmascarades."""
    errors = torch.abs(pred - target) * mask.float()
    return errors.sum() / mask.float().sum().clamp(min=1)


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
            progress = (s - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = max(self.lr * 0.5 * (1 + np.cos(np.pi * progress)), self.lr / 100)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr


def train(curves_norm: np.ndarray,
          device: torch.device) -> TemporalMaskedAutoencoder:
    """
    Entrena el Transformer sobre les corbes de referència.

    Paràmetres
    ----------
    curves_norm : (N_ref, T) corbes normalitzades sense NaN
    device      : torch.device

    Retorna
    -------
    Millor model (carregat des del checkpoint)
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # DataLoaders
    full_ds = TimeSeriesDataset(curves_norm, augment=True, seed=RANDOM_SEED)
    n_val   = max(1, int(len(full_ds) * VAL_SPLIT))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED))
    val_ds.dataset = TimeSeriesDataset(curves_norm, augment=False, seed=RANDOM_SEED + 1)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train: {n_train:,} corbes  |  Val: {n_val:,} corbes")
    print(f"Batches per epoch: {len(train_loader)}")

    # Model
    model    = TemporalMaskedAutoencoder().to(device)
    print(f"Paràmetres entrenables: {count_parameters(model):,}")

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR,
                                    weight_decay=WEIGHT_DECAY)
    total_steps = N_EPOCHS * len(train_loader)
    scheduler   = WarmupCosineScheduler(optimizer, LR_WARMUP_STEPS, total_steps, LR)

    best_val_mae  = float("inf")
    patience_left = PATIENCE
    BEST_CKPT.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>6} {'Train MAE':>10} {'Val MAE':>10} "
          f"{'LR':>10} {'Time':>7} {'':>10}")
    print("─" * 62)

    for epoch in range(1, N_EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            inp, tgt, mask = (batch["input"].to(device),
                              batch["target"].to(device),
                              batch["mask"].to(device))
            optimizer.zero_grad()
            pred = model(inp, mask)
            loss = masked_mae_loss(pred, tgt, mask)
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
                val_losses.append(masked_mae_loss(model(inp, mask), tgt, mask).item())

        train_mae = np.mean(train_losses)
        val_mae   = np.mean(val_losses)
        lr_now    = optimizer.param_groups[0]["lr"]
        elapsed   = time.time() - t0

        if val_mae < best_val_mae:
            best_val_mae  = val_mae
            patience_left = PATIENCE
            torch.save(model.state_dict(), BEST_CKPT)
            status = "✓ millor"
        else:
            patience_left -= 1
            status = f"({patience_left} left)"

        torch.save(model.state_dict(), LAST_CKPT)
        print(f"{epoch:>6} {train_mae:>10.4f} {val_mae:>10.4f} "
              f"{lr_now:>10.2e} {elapsed:>6.1f}s {status:>10}")

        if patience_left == 0:
            print(f"\nEarly stopping a l'època {epoch}.")
            break

    print(f"\nMillor Val MAE (normalitzat): {best_val_mae:.4f}")
    model.load_state_dict(torch.load(BEST_CKPT, map_location=device))
    return model


def predict_batch(model: TemporalMaskedAutoencoder,
                  loader: DataLoader,
                  device: torch.device) -> np.ndarray:
    """
    Inferència sobre un DataLoader.

    Retorna
    -------
    preds : (N, T) prediccions en espai normalitzat
    """
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            inp  = batch["input"].to(device)
            mask = batch["mask"].to(device)
            preds.append(model(inp, mask).cpu().numpy())
    return np.concatenate(preds, axis=0)
