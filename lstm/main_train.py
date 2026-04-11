"""
lstm/main_train.py
------------------
Execució:
    cd data_challenge
    python -m lstm.main_train
"""

import sys
import ctypes
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Impedeix que Windows entri en repòs durant l'entrenament
ctypes.windll.kernel32.SetThreadExecutionState(0x80000002)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lstm.src.config  import (
    X_TRAIN_PATH, Y_TRAIN_PATH, PRED_TRAIN_PATH, OUT_DIR, BATCH_SIZE,
)
from lstm.src.model   import TemporalMaskedAutoencoder
from lstm.src.trainer import train, predict_batch
from lstm.src.dataset import InferenceDataset
from src.data_loader  import load_train, load_csv, save_submission
from src.evaluator    import print_summary


def load_and_normalize_ref(X_ref: pd.DataFrame):
    """Normalitza les corbes de referència per zscore."""
    values = X_ref.values.T.astype(np.float32)   # (N_ref, T)
    N, T   = values.shape
    norm   = np.zeros_like(values)
    for i in range(N):
        mu, sig   = np.nanmean(values[i]), np.nanstd(values[i])
        norm[i]   = (values[i] - mu) / (sig if sig > 1e-6 else 1.0)
    return norm


def load_and_normalize_holed(X_holed: pd.DataFrame):
    """Normalitza les corbes holed_ (NaN → 0.0) i guarda stats per desnormalitzar."""
    values    = X_holed.values.T.astype(np.float32)   # (N_holed, T)
    N, T      = values.shape
    nan_masks = np.isnan(values)
    norm      = np.zeros_like(values)
    mus       = np.zeros(N, dtype=np.float32)
    sigmas    = np.ones(N, dtype=np.float32)

    for i in range(N):
        mu, sig  = np.nanmean(values[i]), np.nanstd(values[i])
        sig      = sig if sig > 1e-6 else 1.0
        mus[i]   = mu
        sigmas[i]= sig
        n        = (values[i] - mu) / sig
        norm[i]  = np.nan_to_num(n, nan=0.0)

    return norm, nan_masks, mus, sigmas


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Carregar dades ────────────────────────────────────────────────────────
    print("\n── Carregant dades ──")
    X_ref, X_holed, y_train = load_train(X_TRAIN_PATH, Y_TRAIN_PATH)

    ref_norm                             = load_and_normalize_ref(X_ref)
    holed_norm, nan_masks, mus, sigmas   = load_and_normalize_holed(X_holed)

    # ── Entrenar ──────────────────────────────────────────────────────────────
    print("\n── Entrenament ──")
    model = train(ref_norm, device)

    # ── Inferència ────────────────────────────────────────────────────────────
    print("\n── Inferència sobre holed_ ──")
    inf_loader = DataLoader(
        InferenceDataset(holed_norm, nan_masks),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    preds_norm = predict_batch(model, inf_loader, device)   # (N_holed, T)

    # Desnormalitzar
    N, T      = preds_norm.shape
    preds_raw = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        preds_raw[i] = preds_norm[i] * sigmas[i] + mus[i]

    y_pred = pd.DataFrame(preds_raw.T,
                          columns=X_holed.columns,
                          index=X_holed.index)

    # ── Avaluació ─────────────────────────────────────────────────────────────
    print("\n── Resultats ──")
    nan_mask_df = pd.DataFrame(nan_masks.T,
                               columns=X_holed.columns,
                               index=X_holed.index)
    print_summary(y_train, y_pred, nan_mask_df, model_name="Transformer")

    # ── Guardar ───────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamps = X_holed.index.strftime("%Y-%m-%d %H:%M:%S").values
    save_submission(y_pred, PRED_TRAIN_PATH, timestamps)

    # Restaurar comportament normal de repòs
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
    print("\nEntrenament completat. El sistema pot tornar a entrar en repòs.")


if __name__ == "__main__":
    main()
