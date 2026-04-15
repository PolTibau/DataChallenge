"""
lstm/src/trainer.py
-------------------
Entrenament i inferència del Transformer.

Millores respecte la versió anterior:
  - Checkpoint personalitzat: cada script pot guardar a un path diferent
  - Pèrdua híbrida MAE + MSE: penalitza errors grans quadràticament
    → millora especialment les corbes difícils com holed_644
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
    BEST_CKPT, LAST_CKPT, RANDOM_SEED,
)
from lstm.src.dataset import TimeSeriesDataset
from lstm.src.model   import TemporalMaskedAutoencoder, count_parameters


# ── Pèrdua híbrida MAE + MSE ──────────────────────────────────────────────────

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
    """
    Pèrdua híbrida: alpha × MAE + (1-alpha) × MSE

    El MAE és robust a outliers però no penalitza errors grans.
    El MSE penalitza errors grans quadràticament → millora corbes difícils.

    alpha=0.7: 70% MAE + 30% MSE (equilibri entre robustesa i penalització)
    """
    diff = pred - target
    mae  = (torch.abs(diff) * mask.float()).sum() / mask.float().sum().clamp(min=1)
    mse  = ((diff ** 2)     * mask.float()).sum() / mask.float().sum().clamp(min=1)
    return alpha * mae + (1 - alpha) * torch.sqrt(mse)   # RMSE per mantenir escala


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
            progress = (s - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = max(self.lr * 0.5 * (1 + np.cos(np.pi * progress)), self.lr / 100)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr


# ── Entrenament ───────────────────────────────────────────────────────────────

def train(curves_norm: np.ndarray,
          device: torch.device,
          n_epochs_override: int | None = None,
          ckpt_path: Path | None = None,
          use_hybrid_loss: bool = True) -> TemporalMaskedAutoencoder:
    """
    Entrena el Transformer sobre les corbes de referència.

    Paràmetres
    ----------
    curves_norm       : (N_ref, T) corbes normalitzades sense NaN
    device            : torch.device
    n_epochs_override : sobreescriu N_EPOCHS (útil per debug)
    ckpt_path         : path on guardar el millor model
                        (per defecte usa BEST_CKPT del config)
    use_hybrid_loss   : si True usa MAE+MSE, si False usa MAE pur
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    n_epochs  = n_epochs_override if n_epochs_override is not None else N_EPOCHS
    best_ckpt = ckpt_path if ckpt_path is not None else BEST_CKPT
    last_ckpt = best_ckpt.parent / f"last_{best_ckpt.name}"
    loss_fn   = masked_hybrid_loss if use_hybrid_loss else masked_mae_loss
    loss_name = "MAE+MSE" if use_hybrid_loss else "MAE"

    # DataLoaders
    full_ds = TimeSeriesDataset(curves_norm, augment=True, seed=RANDOM_SEED)
    n_val   = max(1, int(len(full_ds) * VAL_SPLIT))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED))
    val_ds.dataset = TimeSeriesDataset(curves_norm, augment=False, seed=RANDOM_SEED + 1)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"Train: {n_train:,} corbes  |  Val: {n_val:,} corbes")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Loss: {loss_name}")

    # Model
    model = TemporalMaskedAutoencoder().to(device)
    print(f"Paràmetres entrenables: {count_parameters(model):,}")

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
            optimizer.zero_grad()
            loss = loss_fn(model(inp, mask), tgt, mask)
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
                val_losses.append(loss_fn(model(inp, mask), tgt, mask).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        lr_now     = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        if val_loss < best_val_mae:
            best_val_mae  = val_loss
            patience_left = PATIENCE
            torch.save(model.state_dict(), best_ckpt)
            status = "✓ millor"
        else:
            patience_left -= 1
            status = f"({patience_left} left)"

        torch.save(model.state_dict(), last_ckpt)
        print(f"{epoch:>6} {train_loss:>11.4f} {val_loss:>10.4f} "
              f"{lr_now:>10.2e} {elapsed:>6.1f}s {status:>10}")

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
