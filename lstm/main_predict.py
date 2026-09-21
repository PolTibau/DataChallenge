"""
lstm/main_predict.py — v3.1
---------------------------
Inferència amb el model fine-tunat (finetuned_model_v3.pt).

Prerequisits:
  1. python -m lstm.main_train_combined  → best_model_v3.pt
  2. python -m lstm.main_finetune        → finetuned_model_v3.pt

Execució:
    python -m lstm.main_predict

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

from lstm.src.config  import (
    X_TEST_PATH, X_TRAIN_PATH, Y_TRAIN_PATH,
    PRED_TEST_PATH, OUT_DIR, CKPT_DIR, BATCH_SIZE,
)
from lstm.src.model   import TemporalMaskedAutoencoder
from lstm.src.trainer import predict_batch_tta, load_model_weights
from lstm.src.dataset import (
    InferenceDataset, compute_global_stats, robust_stats,
)
from src.data_loader  import load_test, load_train, save_submission

FINETUNED_CKPT = CKPT_DIR / "finetuned_model_v3.pt"


def normalize_holed_robust(X_holed: pd.DataFrame) -> tuple:
    values    = X_holed.values.T.astype(np.float32)
    N, T      = values.shape
    nan_masks = np.isnan(values)
    norm      = np.zeros_like(values)
    meds      = np.zeros(N, dtype=np.float32)
    mads      = np.ones(N,  dtype=np.float32)

    for i in range(N):
        med, mad   = robust_stats(values[i])
        meds[i]    = med
        mads[i]    = mad
        norm[i]    = np.nan_to_num((values[i] - med) / mad, nan=0.0)

    return norm, nan_masks, meds, mads


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not FINETUNED_CKPT.exists():
        raise FileNotFoundError(
            f"No s'ha trobat '{FINETUNED_CKPT}'.\n"
            f"Executa: python -m lstm.main_train_combined\n"
            f"         python -m lstm.main_finetune")

    print(f"\nCarregant model fine-tunat: {FINETUNED_CKPT}")
    model = TemporalMaskedAutoencoder().to(device)
    load_model_weights(FINETUNED_CKPT, model, device)
    print("Model carregat.")

    print("\n── Carregant dades per global_stats ──")
    X_ref_train, _, _ = load_train(X_TRAIN_PATH, Y_TRAIN_PATH)
    X_ref_test, X_holed = load_test(X_TEST_PATH)

    ref_train_raw = X_ref_train.select_dtypes("number").to_numpy(dtype=np.float32).T
    ref_test_raw  = X_ref_test.select_dtypes("number").to_numpy(dtype=np.float32).T

    def clean_raw(vals):
        bad = np.isnan(vals).any(axis=1) | (vals.std(axis=1) < 1e-6)
        return vals[~bad]

    ref_all_raw  = np.concatenate([clean_raw(ref_train_raw),
                                   clean_raw(ref_test_raw)], axis=0)
    global_stats = compute_global_stats(ref_all_raw)

    holed_norm, nan_masks, meds, mads = normalize_holed_robust(X_holed)
    holed_raw = X_holed.values.T.astype(np.float32)

    print("\n── Inferència amb TTA (soroll) ──")
    inf_loader = DataLoader(
        InferenceDataset(holed_norm, nan_masks,
                         curves_raw=holed_raw,
                         global_stats=global_stats),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    preds_norm = predict_batch_tta(model, inf_loader, device)

    N, T      = preds_norm.shape
    preds_raw = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        preds_raw[i] = preds_norm[i] * mads[i] + meds[i]

    preds_raw = np.clip(preds_raw, 0, None)

    holed_original = X_holed.values.T.astype(np.float32)
    known_mask     = ~np.isnan(holed_original)
    preds_raw[known_mask] = holed_original[known_mask]

    print(f"Forats imputats   : {nan_masks.sum():,}")
    print(f"Valors preservats : {known_mask.sum():,}")

    y_pred = pd.DataFrame(preds_raw.T, columns=X_holed.columns)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamps = X_holed.index.strftime("%Y-%m-%d %H:%M:%S").values
    save_submission(y_pred, PRED_TEST_PATH, timestamps)


if __name__ == "__main__":
    main()
