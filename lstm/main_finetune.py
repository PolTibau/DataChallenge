"""
lstm/main_finetune.py — v3.1
----------------------------
Fine-tuning del model v3 (best_model_v3.pt) sobre corbes de referència
del TEST, amb la mateixa pipeline que l'entrenament combinat:

  - Normalització robusta (mediana / MAD)
  - Profiles enriquits (9 dims) + global_stats
  - Màscares empíriques (si es pot estimar de holed_)
  - Pèrdua MAE + profiles al forward

Flux:
  1. Carregar best_model_v3.pt (entrenat amb main_train_combined)
  2. Fine-tuning breu sobre refs de X_test
  3. Guardar finetuned_model_v3.pt

Execució:
    cd data_challenge
    python -m lstm.main_finetune
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import ctypes
    ES_CONTINUOUS       = 0x80000000
    ES_SYSTEM_REQUIRED  = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
    _windows = True
except AttributeError:
    _windows = False

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from lstm.src.config  import (
    X_TEST_PATH, X_TRAIN_PATH, Y_TRAIN_PATH,
    CKPT_DIR, BATCH_SIZE,
    RANDOM_SEED, GRAD_CLIP, VAL_SPLIT,
)
from lstm.src.model   import TemporalMaskedAutoencoder
from lstm.src.trainer import masked_mae_loss, WarmupCosineScheduler, build_fixed_masks
from lstm.src.dataset import (
    TimeSeriesDataset, compute_global_stats, fit_mask_distribution, robust_stats,
)
from src.data_loader  import load_test, load_train


N_FINETUNE_EPOCHS = 15
LR_FINETUNE       = 3e-5
PATIENCE_FT       = 5
PRETRAIN_CKPT     = CKPT_DIR / "best_model_v3.pt"
FINETUNE_CKPT     = CKPT_DIR / "finetuned_model_v3.pt"


def normalize_robust(values: np.ndarray, label: str) -> tuple:
    """Normalització mediana/MAD; elimina NaN i planes."""
    has_nan = np.isnan(values).any(axis=1)
    is_flat = values.std(axis=1) < 1e-6
    bad     = has_nan | is_flat
    if bad.sum() > 0:
        print(f"[{label}] Eliminant {bad.sum()} corbes problemàtiques "
              f"({has_nan.sum()} amb NaN, {is_flat.sum()} planes)")
    values = values[~bad]

    norm = np.zeros_like(values)
    for i in range(len(values)):
        med, mad = robust_stats(values[i])
        norm[i]  = (values[i] - med) / mad

    print(f"[{label}] Corbes vàlides: {len(values):,} | "
          f"NaN restants: {np.isnan(norm).sum()}")
    return norm, values


def finetune(model: TemporalMaskedAutoencoder,
             ref_norm: np.ndarray,
             device: torch.device,
             curves_raw: np.ndarray | None = None,
             global_stats: dict | None = None,
             mask_dist: dict | None = None) -> TemporalMaskedAutoencoder:
    """Fine-tuning amb profiles, màscares empíriques i val masks congelades."""
    torch.manual_seed(RANDOM_SEED)

    full_ds = TimeSeriesDataset(
        ref_norm, augment=True, seed=RANDOM_SEED + 99,
        curves_raw=curves_raw, global_stats=global_stats, mask_dist=mask_dist)
    n_val   = max(1, int(len(full_ds) * VAL_SPLIT))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED))

    fixed_val_masks = build_fixed_masks(
        len(ref_norm), ref_norm.shape[1], mask_dist, seed=RANDOM_SEED + 101)
    val_full = TimeSeriesDataset(
        ref_norm, augment=False, seed=RANDOM_SEED + 100,
        curves_raw=curves_raw, global_stats=global_stats,
        mask_dist=mask_dist, fixed_masks=fixed_val_masks)
    val_ds.dataset = val_full

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"Fine-tuning: {n_train:,} corbes train | {n_val:,} corbes val")
    print(f"LR: {LR_FINETUNE:.1e}  |  Èpoques màx: {N_FINETUNE_EPOCHS}  |  "
          f"Patience: {PATIENCE_FT}")

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR_FINETUNE,
                                    weight_decay=1e-5)
    total_steps = N_FINETUNE_EPOCHS * len(train_loader)
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
            loss = masked_mae_loss(model(inp, mask, profile), tgt, mask)
            loss.backward()
            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

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
                val_losses.append(
                    masked_mae_loss(model(inp, mask, profile), tgt, mask).item())

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
    model.load_state_dict(
        torch.load(FINETUNE_CKPT, map_location=device, weights_only=True))
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nCarregant model pre-entrenat: {PRETRAIN_CKPT}")
    if not PRETRAIN_CKPT.exists():
        raise FileNotFoundError(
            f"No s'ha trobat '{PRETRAIN_CKPT}'.\n"
            f"Executa primer: python -m lstm.main_train_combined")

    model = TemporalMaskedAutoencoder().to(device)
    model.load_state_dict(
        torch.load(PRETRAIN_CKPT, map_location=device, weights_only=True))
    print("Model carregat correctament.")

    print("\n── Carregant dades ──")
    # global_stats sobre la mateixa base que l'entrenament combinat
    X_ref_train, _, _ = load_train(X_TRAIN_PATH, Y_TRAIN_PATH)
    X_ref_test, X_holed = load_test(X_TEST_PATH)

    ref_train_vals = X_ref_train.select_dtypes("number").to_numpy(dtype=np.float32).T
    ref_test_vals  = X_ref_test.select_dtypes("number").to_numpy(dtype=np.float32).T

    def clean_raw(vals):
        bad = np.isnan(vals).any(axis=1) | (vals.std(axis=1) < 1e-6)
        return vals[~bad]

    ref_all_raw  = np.concatenate([clean_raw(ref_train_vals),
                                   clean_raw(ref_test_vals)], axis=0)
    global_stats = compute_global_stats(ref_all_raw)

    # Fine-tune només sobre refs del test (adaptació al context)
    ref_norm, ref_raw = normalize_robust(ref_test_vals, "X_test ref")

    print("\n── Aprenent distribució de forats (test holed_) ──")
    mask_dist = fit_mask_distribution(X_holed.values.T.astype(np.float32))

    print("\n── Fine-tuning sobre corbes de referència del test ──")
    model = finetune(model, ref_norm, device,
                     curves_raw=ref_raw,
                     global_stats=global_stats,
                     mask_dist=mask_dist)

    print(f"\nFine-tuning completat.")
    print(f"Model guardat a: {FINETUNE_CKPT}")
    print(f"\nPer inferir amb aquest checkpoint, actualitza "
          f"main_predict_combined.py o carrega '{FINETUNE_CKPT.name}'.")

    if _windows:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
