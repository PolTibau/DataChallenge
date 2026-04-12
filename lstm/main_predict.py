"""
lstm/main_predict_finetuned.py
------------------------------
Inferència usant el model fine-tunat sobre X_test.

Prerequisits:
  1. python -m lstm.main_train      → genera best_model.pt
  2. python -m lstm.main_finetune   → genera finetuned_model.pt

Execució:
    cd data_challenge
    python -m lstm.main_predict_finetuned

Genera:
    outputs/lstm/y_pred_test.csv  ← submissió final
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lstm.src.config  import (
    X_TEST_PATH, PRED_TEST_PATH, OUT_DIR,
    CKPT_DIR, BATCH_SIZE,
)
from lstm.src.model   import TemporalMaskedAutoencoder
from lstm.src.trainer import predict_batch
from lstm.src.dataset import InferenceDataset
from src.data_loader  import load_test, save_submission

FINETUNED_CKPT = CKPT_DIR / "finetuned_model.pt"


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Carregar model fine-tunat ─────────────────────────────────────────────
    if not FINETUNED_CKPT.exists():
        raise FileNotFoundError(
            f"No s'ha trobat '{FINETUNED_CKPT}'.\n"
            f"Executa primer: python -m lstm.main_finetune")

    print(f"\nCarregant model fine-tunat: {FINETUNED_CKPT}")
    model = TemporalMaskedAutoencoder().to(device)
    model.load_state_dict(
        torch.load(FINETUNED_CKPT, map_location=device, weights_only=True))
    print("Model carregat correctament.")

    # ── Carregar dades de test ────────────────────────────────────────────────
    print("\n── Carregant dades de test ──")
    X_ref, X_holed = load_test(X_TEST_PATH)
    holed_norm, nan_masks, mus, sigmas = normalize_holed(X_holed)

    # ── Inferència ────────────────────────────────────────────────────────────
    print("\n── Inferència ──")
    inf_loader = DataLoader(
        InferenceDataset(holed_norm, nan_masks),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    preds_norm = predict_batch(model, inf_loader, device)   # (N_holed, T)

    # ── Desnormalitzar ────────────────────────────────────────────────────────
    N, T      = preds_norm.shape
    preds_raw = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        preds_raw[i] = preds_norm[i] * sigmas[i] + mus[i]

    # Clippejar negatius a 0
    preds_raw = np.clip(preds_raw, 0, None)

    # ── Restaurar valors CONEGUTS originals ───────────────────────────────────
    # Usar la predicció NOMÉS als forats reals (posicions NaN originals)
    # Els valors que ja coneixíem es preserven tal qual
    holed_original = X_holed.values.T.astype(np.float32)
    known_mask     = ~np.isnan(holed_original)
    preds_raw[known_mask] = holed_original[known_mask]

    print(f"Forats imputats   : {nan_masks.sum():,}")
    print(f"Valors preservats : {known_mask.sum():,}")

    # ── Guardar CSV de submissió ──────────────────────────────────────────────
    y_pred = pd.DataFrame(preds_raw.T, columns=X_holed.columns)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamps = X_holed.index.strftime("%Y-%m-%d %H:%M:%S").values
    save_submission(y_pred, PRED_TEST_PATH, timestamps)


if __name__ == "__main__":
    main()
