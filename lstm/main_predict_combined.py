"""
lstm/main_predict_combined.py
-----------------------------
Inferència usant el model entrenat sobre X_train + X_test ref.

No cal fine-tuning — el model ja ha vist les corbes de X_test ref.

Prerequisit:
    python -m lstm.main_train_combined

Execució:
    cd data_challenge
    python -m lstm.main_predict_combined

Genera:
    outputs/lstm/y_pred_test.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lstm.src.config import (
    X_TEST_PATH, X_TRAIN_PATH, Y_TRAIN_PATH, PRED_TEST_PATH, OUT_DIR, CKPT_DIR, BATCH_SIZE,
)
from lstm.src.model   import TemporalMaskedAutoencoder
from lstm.src.trainer import predict_batch
from lstm.src.dataset import InferenceDataset, compute_global_stats
from src.data_loader  import load_test, load_train, save_submission

COMBINED_CKPT = CKPT_DIR / "best_model_combined.pt"
PRED_TEST_PATH = PRED_TEST_PATH


def normalize_holed(X_holed: pd.DataFrame) -> tuple:
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

    # ── Carregar model combinat ───────────────────────────────────────────────
    import lstm.main_predict_combined as _self
    ckpt = _self.COMBINED_CKPT
    pred_path = _self.PRED_TEST_PATH

    if not ckpt.exists():
        raise FileNotFoundError(
            f"No s'ha trobat '{ckpt}'.\n"
            f"Executa primer: python -m lstm.main_train_combined")

    print(f"\nCarregant model: {ckpt}")
    model = TemporalMaskedAutoencoder().to(device)
    model.load_state_dict(
        torch.load(ckpt, map_location=device, weights_only=True))
    print("Model carregat correctament.")

    # ── Carregar dades per calcular global_stats (necessari per profiles) ─────
    print("\n── Carregant dades per global_stats ──")
    X_ref_train, _, _ = load_train(X_TRAIN_PATH, Y_TRAIN_PATH)
    X_ref_test, X_holed = load_test(X_TEST_PATH)

    ref_train_raw = X_ref_train.select_dtypes("number").to_numpy(dtype=np.float32).T
    ref_test_raw  = X_ref_test.select_dtypes("number").to_numpy(dtype=np.float32).T

    # Eliminar corbes planes/NaN per calcular stats
    def clean_raw(vals):
        bad = np.isnan(vals).any(axis=1) | (vals.std(axis=1) < 1e-6)
        return vals[~bad]

    ref_all_raw  = np.concatenate([clean_raw(ref_train_raw),
                                   clean_raw(ref_test_raw)], axis=0)
    global_stats = compute_global_stats(ref_all_raw)
    print(f"  global_stats calculats sobre {len(ref_all_raw):,} corbes")

    # ── Normalitzar holed_ ────────────────────────────────────────────────────
    holed_norm, nan_masks, mus, sigmas = normalize_holed(X_holed)
    holed_raw = X_holed.values.T.astype(np.float32)

    # ── Inferència ────────────────────────────────────────────────────────────
    print("\n── Inferència ──")
    inf_loader = DataLoader(
        InferenceDataset(holed_norm, nan_masks,
                         curves_raw=holed_raw,
                         global_stats=global_stats),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    preds_norm = predict_batch(model, inf_loader, device)

    # Desnormalitzar
    N, T      = preds_norm.shape
    preds_raw = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        preds_raw[i] = preds_norm[i] * sigmas[i] + mus[i]

    # Clippejar negatius
    preds_raw = np.clip(preds_raw, 0, None)

    # Preservar valors coneguts originals
    holed_original = X_holed.values.T.astype(np.float32)
    known_mask     = ~np.isnan(holed_original)
    preds_raw[known_mask] = holed_original[known_mask]

    print(f"Forats imputats   : {nan_masks.sum():,}")
    print(f"Valors preservats : {known_mask.sum():,}")

    # ── Guardar CSV ───────────────────────────────────────────────────────────
    y_pred = pd.DataFrame(preds_raw.T, columns=X_holed.columns)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamps = X_holed.index.strftime("%Y-%m-%d %H:%M:%S").values
    save_submission(y_pred, pred_path, timestamps)


if __name__ == "__main__":
    main()
