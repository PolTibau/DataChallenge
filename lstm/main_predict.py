"""
lstm/main_predict.py
--------------------
Prerequisit: haver executat lstm/main_train.py

Execució:
    cd data_challenge
    python -m lstm.main_predict
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lstm.src.config  import X_TEST_PATH, PRED_TEST_PATH, OUT_DIR, BEST_CKPT, BATCH_SIZE
from lstm.src.model   import TemporalMaskedAutoencoder
from lstm.src.trainer import predict_batch
from lstm.src.dataset import InferenceDataset
from src.data_loader  import load_test, save_submission


def load_and_normalize_holed(X_holed: pd.DataFrame):
    values    = X_holed.values.T.astype(np.float32)
    N, T      = values.shape
    nan_masks = np.isnan(values)
    norm      = np.zeros_like(values)
    mus       = np.zeros(N, dtype=np.float32)
    sigmas    = np.ones(N, dtype=np.float32)
    for i in range(N):
        mu, sig   = np.nanmean(values[i]), np.nanstd(values[i])
        sig       = sig if sig > 1e-6 else 1.0
        mus[i]    = mu; sigmas[i] = sig
        norm[i]   = np.nan_to_num((values[i] - mu) / sig, nan=0.0)
    return norm, nan_masks, mus, sigmas


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Carregar model ────────────────────────────────────────────────────────
    if not BEST_CKPT.exists():
        raise FileNotFoundError(
            f"No s'ha trobat '{BEST_CKPT}'.\nExecuta primer: python -m lstm.main_train")

    model = TemporalMaskedAutoencoder().to(device)
    model.load_state_dict(torch.load(BEST_CKPT, map_location=device))
    print(f"Model carregat des de: {BEST_CKPT}")

    # ── Carregar dades ────────────────────────────────────────────────────────
    print("\n── Carregant dades de test ──")
    X_ref, X_holed = load_test(X_TEST_PATH)
    holed_norm, nan_masks, mus, sigmas = load_and_normalize_holed(X_holed)

    # ── Inferència ────────────────────────────────────────────────────────────
    print("\n── Inferència ──")
    inf_loader = DataLoader(
        InferenceDataset(holed_norm, nan_masks),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    preds_norm = predict_batch(model, inf_loader, device)

    N, T      = preds_norm.shape
    preds_raw = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        preds_raw[i] = preds_norm[i] * sigmas[i] + mus[i]

    y_pred = pd.DataFrame(preds_raw.T, columns=X_holed.columns)

    # ── Guardar ───────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamps = X_holed.index.strftime("%Y-%m-%d %H:%M:%S").values
    save_submission(y_pred, PRED_TEST_PATH, timestamps)


if __name__ == "__main__":
    main()
