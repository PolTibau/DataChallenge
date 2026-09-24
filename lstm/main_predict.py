"""
lstm/main_predict.py — v3.1
---------------------------
Inferència sobre X_test holed_ amb el millor checkpoint v3.1 disponible.

Selecció automàtica del checkpoint:
  1. finetuned_model_v3.pt  (si existeix: best_model_v3.pt + main_finetune.py)
  2. best_model_v3.pt       (sortida directa de main_train_combined.py)

Prerequisit (com a mínim un dels dos):
    python -m lstm.main_train_combined   → best_model_v3.pt
    python -m lstm.main_finetune         → finetuned_model_v3.pt (opcional)

Execució:
    cd data_challenge
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

from lstm.src.config import (
    X_TEST_PATH, X_TRAIN_PATH, Y_TRAIN_PATH, PRED_TEST_PATH,
    OUT_DIR, CKPT_DIR, BATCH_SIZE,
)
from lstm.src.model   import TemporalMaskedAutoencoder
from lstm.src.trainer import predict_batch_tta, load_model_weights
from lstm.src.dataset import (
    InferenceDataset, compute_global_stats, robust_stats,
)
from src.data_loader  import load_test, load_train, save_submission

COMBINED_CKPT  = CKPT_DIR / "best_model_v3.pt"
FINETUNED_CKPT = CKPT_DIR / "finetuned_model_v3.pt"


def resolve_checkpoint() -> Path:
    """Prioritza el checkpoint fine-tunat si existeix; sinó el combinat."""
    if FINETUNED_CKPT.exists():
        return FINETUNED_CKPT
    if COMBINED_CKPT.exists():
        return COMBINED_CKPT
    raise FileNotFoundError(
        f"No s'ha trobat cap checkpoint v3.1.\n"
        f"Executa primer: python -m lstm.main_train_combined  (→ {COMBINED_CKPT.name})\n"
        f"Opcionalment després: python -m lstm.main_finetune  (→ {FINETUNED_CKPT.name})")


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

    ckpt_path = resolve_checkpoint()
    print(f"\nCarregant checkpoint: {ckpt_path.name}")
    model = TemporalMaskedAutoencoder().to(device)
    load_model_weights(ckpt_path, model, device)
    print("Model carregat.")

    # ── global_stats sobre la mateixa base que l'entrenament ─────────────────
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

    # ── Normalitzar holed_ ────────────────────────────────────────────────────
    holed_norm, nan_masks, meds, mads = normalize_holed_robust(X_holed)
    holed_raw = X_holed.values.T.astype(np.float32)

    # ── Inferència amb TTA ────────────────────────────────────────────────────
    print("\n── Inferència amb Test-Time Augmentation ──")
    inf_loader = DataLoader(
        InferenceDataset(holed_norm, nan_masks,
                         curves_raw=holed_raw,
                         global_stats=global_stats),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    preds_norm = predict_batch_tta(model, inf_loader, device)

    # Desnormalitzar
    N, T      = preds_norm.shape
    preds_raw = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        preds_raw[i] = preds_norm[i] * mads[i] + meds[i]

    preds_raw = np.clip(preds_raw, 0, None)

    # Preservar valors coneguts
    holed_original = X_holed.values.T.astype(np.float32)
    known_mask     = ~np.isnan(holed_original)
    preds_raw[known_mask] = holed_original[known_mask]

    print(f"Forats imputats   : {nan_masks.sum():,}")
    print(f"Valors preservats : {known_mask.sum():,}")

    # Guardar
    y_pred = pd.DataFrame(preds_raw.T, columns=X_holed.columns)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamps = X_holed.index.strftime("%Y-%m-%d %H:%M:%S").values
    save_submission(y_pred, PRED_TEST_PATH, timestamps)


if __name__ == "__main__":
    main()
