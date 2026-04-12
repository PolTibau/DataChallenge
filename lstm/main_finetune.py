"""
lstm/main_finetune.py
---------------------
Fine-tuning del model entrenat sobre les corbes de referència del TEST.

Per què?
--------
El model s'ha entrenat sobre X_train (20.000 corbes).
Però X_test té 37.140 corbes de referència pròpies que el model
no ha vist mai. Fer fine-tuning sobre aquestes corbes permet que
el model s'adapti al context específic del test abans d'imputar.

Flux:
  1. Carregar best_model.pt (entrenat sobre X_train)
  2. Carregar les 37.140 corbes de referència de X_test
  3. Fine-tuning breu (N_FINETUNE_EPOCHS èpoques, LR baix)
  4. Guardar el model fine-tuned
  5. Imputar les 1000 corbes holed_ del test
  6. Guardar y_pred_test.csv

Execució:
    cd data_challenge
    python -m lstm.main_finetune
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Evitar repòs a Windows
try:
    import ctypes
    ES_CONTINUOUS       = 0x80000000
    ES_SYSTEM_REQUIRED  = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    _windows = True
except AttributeError:
    _windows = False   # Linux/Mac (Colab) — no cal fer res

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from lstm.src.config  import (
    X_TEST_PATH, OUT_DIR,
    BEST_CKPT, CKPT_DIR, BATCH_SIZE,
    RANDOM_SEED, GRAD_CLIP, VAL_SPLIT,
)
from lstm.src.model   import TemporalMaskedAutoencoder
from lstm.src.trainer import masked_mae_loss, WarmupCosineScheduler
from lstm.src.dataset import TimeSeriesDataset
from src.data_loader  import load_test



# ── Configuració del fine-tuning ──────────────────────────────────────────────
N_FINETUNE_EPOCHS = 25     # èpoques de fine-tuning (menys que l'entrenament original)
LR_FINETUNE       = 3e-5   # LR molt baix per no destruir el que ha après
PATIENCE_FT       = 6      # early stopping del fine-tuning
FINETUNE_CKPT     = CKPT_DIR / "finetuned_model.pt"


def normalize_matrix(values: np.ndarray) -> np.ndarray:
    """
    Normalitza cada fila (corba) per zscore.
    Elimina corbes amb NaN o std=0 (planes).
    """
    has_nan = np.isnan(values).any(axis=1)
    is_flat = values.std(axis=1) < 1e-6
    bad     = has_nan | is_flat

    if bad.sum() > 0:
        print(f"[finetune] Eliminant {bad.sum()} corbes problemàtiques "
              f"({has_nan.sum()} amb NaN, {is_flat.sum()} planes)")
    values = values[~bad]

    N, T = values.shape
    norm = np.zeros_like(values)
    for i in range(N):
        mu, sig  = values[i].mean(), values[i].std()
        norm[i]  = (values[i] - mu) / (sig if sig > 1e-6 else 1.0)

    print(f"[finetune] Corbes vàlides: {N} | NaN restants: {np.isnan(norm).sum()}")
    return norm


def normalize_holed(X_holed: pd.DataFrame) -> tuple:
    """Normalitza les corbes holed_ i guarda stats per desnormalitzar."""
    values    = X_holed.values.T.astype(np.float32)
    N, T      = values.shape
    nan_masks = np.isnan(values)
    norm      = np.zeros_like(values)
    mus       = np.zeros(N, dtype=np.float32)
    sigmas    = np.ones(N,  dtype=np.float32)

    for i in range(N):
        mu, sig   = np.nanmean(values[i]), np.nanstd(values[i])
        sig       = sig if sig > 1e-6 else 1.0
        mus[i]    = mu
        sigmas[i] = sig
        norm[i]   = np.nan_to_num((values[i] - mu) / sig, nan=0.0)

    return norm, nan_masks, mus, sigmas


def finetune(model: TemporalMaskedAutoencoder,
             ref_norm: np.ndarray,
             device: torch.device) -> TemporalMaskedAutoencoder:
    """
    Fine-tunes el model sobre les corbes de referència del test.

    Usa un LR molt baix per preservar el que el model ha après
    i adaptar-lo subtilment al context del test.
    """
    torch.manual_seed(RANDOM_SEED)

    # Dataset
    full_ds = TimeSeriesDataset(ref_norm, augment=True, seed=RANDOM_SEED + 99)
    n_val   = max(1, int(len(full_ds) * VAL_SPLIT))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED))
    val_ds.dataset = TimeSeriesDataset(ref_norm, augment=False, seed=RANDOM_SEED + 100)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"Fine-tuning: {n_train:,} corbes train | {n_val:,} corbes val")
    print(f"LR: {LR_FINETUNE:.1e}  |  Èpoques màx: {N_FINETUNE_EPOCHS}  |  "
          f"Patience: {PATIENCE_FT}")

    # Optimizer amb LR molt baix
    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE,
                                    weight_decay=1e-5)
    total_steps = N_FINETUNE_EPOCHS * len(train_loader)
    # Warmup molt curt — el model ja està pre-entrenat
    scheduler   = WarmupCosineScheduler(optimizer, warmup_steps=50,
                                        total_steps=total_steps, lr=LR_FINETUNE)

    best_val_mae  = float("inf")
    patience_left = PATIENCE_FT
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Epoch':>6} {'Train MAE':>10} {'Val MAE':>10} "
          f"{'LR':>10} {'Time':>7} {'':>10}")
    print("─" * 62)

    for epoch in range(1, N_FINETUNE_EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            inp, tgt, mask = (batch["input"].to(device),
                              batch["target"].to(device),
                              batch["mask"].to(device))
            optimizer.zero_grad()
            loss = masked_mae_loss(model(inp, mask), tgt, mask)
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
                val_losses.append(
                    masked_mae_loss(model(inp, mask), tgt, mask).item())

        train_mae = np.mean(train_losses)
        val_mae   = np.mean(val_losses)
        lr_now    = optimizer.param_groups[0]["lr"]
        elapsed   = time.time() - t0

        if val_mae < best_val_mae:
            best_val_mae  = val_mae
            patience_left = PATIENCE_FT
            torch.save(model.state_dict(), FINETUNE_CKPT)
            status = "✓ millor"
        else:
            patience_left -= 1
            status = f"({patience_left} left)"

        print(f"{epoch:>6} {train_mae:>10.4f} {val_mae:>10.4f} "
              f"{lr_now:>10.2e} {elapsed:>6.1f}s {status:>10}")

        if patience_left == 0:
            print(f"\nEarly stopping al fine-tuning (època {epoch}).")
            break

    print(f"\nMillor Val MAE fine-tuning (normalitzat): {best_val_mae:.4f}")

    # Carregar el millor model fine-tuned
    model.load_state_dict(
        torch.load(FINETUNE_CKPT, map_location=device, weights_only=True))
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Carregar model pre-entrenat ───────────────────────────────────────────
    print(f"\nCarregant model pre-entrenat: {BEST_CKPT}")
    if not BEST_CKPT.exists():
        raise FileNotFoundError(
            f"No s'ha trobat '{BEST_CKPT}'.\n"
            f"Executa primer: python -m lstm.main_train")

    model = TemporalMaskedAutoencoder().to(device)
    model.load_state_dict(
        torch.load(BEST_CKPT, map_location=device, weights_only=True))
    print("Model carregat correctament.")

    # ── Carregar dades del test ───────────────────────────────────────────────
    print("\n── Carregant dades de test ──")
    X_ref, X_holed = load_test(X_TEST_PATH)

    # Normalitzar corbes de referència del TEST
    ref_values = X_ref.values.T.astype(np.float32)   # (37140, T)
    ref_norm   = normalize_matrix(ref_values)

    # Normalitzar corbes holed_
    holed_norm, nan_masks, mus, sigmas = normalize_holed(X_holed)

    # ── Fine-tuning sobre X_test ref ─────────────────────────────────────────
    print("\n── Fine-tuning sobre corbes de referència del test ──")
    model = finetune(model, ref_norm, device)

    print(f"\nFine-tuning completat.")
    print(f"Model guardat a: {FINETUNE_CKPT}")
    print(f"\nAra executa: python -m lstm.main_predict")

    # Restaurar repòs Windows
    if _windows:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
