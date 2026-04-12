"""
lstm/main_train.py
------------------
Execució normal:
    cd data_challenge
    python -m lstm.main_train

Mode debug (prova rapida ~30-60s en CPU):
    Posa DEBUG=True a lstm/src/config.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# # Evitar que Windows entri en repòs durant l'entrenament
# import ctypes
# ES_CONTINUOUS      = 0x80000000
# ES_SYSTEM_REQUIRED = 0x00000001
# ES_DISPLAY_REQUIRED = 0x00000002
# ctypes.windll.kernel32.SetThreadExecutionState(
#     ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
# )

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lstm.src.config import (
    X_TRAIN_PATH, Y_TRAIN_PATH, PRED_TRAIN_PATH, OUT_DIR, BATCH_SIZE,
    DEBUG, DEBUG_N_CURVES, DEBUG_N_EPOCHS,
)
from lstm.src.model   import TemporalMaskedAutoencoder
from lstm.src.trainer import train, predict_batch
from lstm.src.dataset import InferenceDataset
from src.data_loader  import load_train, save_submission
from src.evaluator    import print_summary


def load_and_normalize_ref(X_ref: pd.DataFrame) -> np.ndarray:
    """
    Normalitza les corbes de referència per zscore.
    Elimina automàticament corbes amb NaN o std=0 (planes).
    """
    values  = X_ref.values.T.astype(np.float32)   # (N_ref, T)

    # Detectar i eliminar corbes problemàtiques
    has_nan = np.isnan(values).any(axis=1)
    is_flat = values.std(axis=1) < 1e-6
    bad     = has_nan | is_flat
    if bad.sum() > 0:
        print(f"[normalize_ref] Eliminant {bad.sum()} corbes problemàtiques "
              f"({has_nan.sum()} amb NaN, {is_flat.sum()} planes)")
    values = values[~bad]

    # Mode debug: usar subconjunt petit
    if DEBUG:
        values = values[:DEBUG_N_CURVES]
        print(f"[DEBUG] Usant {len(values)} corbes de referència")

    N, T = values.shape
    norm = np.zeros_like(values)
    for i in range(N):
        mu, sig  = values[i].mean(), values[i].std()
        norm[i]  = (values[i] - mu) / (sig if sig > 1e-6 else 1.0)

    print(f"[normalize_ref] Corbes vàlides: {N} | "
          f"NaN restants: {np.isnan(norm).sum()}")
    return norm


def load_and_normalize_holed(X_holed: pd.DataFrame) -> tuple:
    """
    Normalitza les corbes holed_ (NaN → 0.0).
    Guarda mu i sigma per desnormalitzar després.
    """
    values    = X_holed.values.T.astype(np.float32)   # (N_holed, T)
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


def main():
    if DEBUG:
        print("=" * 50)
        print("  MODE DEBUG — entrenament rapid de prova")
        print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Carregar dades ────────────────────────────────────────────────────────
    print("\n── Carregant dades ──")
    X_ref, X_holed, y_train = load_train(X_TRAIN_PATH, Y_TRAIN_PATH)

    ref_norm                           = load_and_normalize_ref(X_ref)
    holed_norm, nan_masks, mus, sigmas = load_and_normalize_holed(X_holed)

    # ── Entrenar ──────────────────────────────────────────────────────────────
    print("\n── Entrenament ──")
    model = train(ref_norm, device, n_epochs_override=DEBUG_N_EPOCHS if DEBUG else None)

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
    if not DEBUG:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamps = X_holed.index.strftime("%Y-%m-%d %H:%M:%S").values
        save_submission(y_pred, PRED_TRAIN_PATH, timestamps)
        print(f"\nPrediccions guardades a: {PRED_TRAIN_PATH}")
    else:
        print("\n[DEBUG] Mode debug — prediccions no guardades.")
        print("Posa DEBUG=False a config.py per l'entrenament real.")

    # Restaurar comportament normal de repòs
    # ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
