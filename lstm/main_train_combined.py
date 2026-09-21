"""
lstm/main_train_combined.py — v3.1
----------------------------------
Entrena el Transformer v3.1 sobre X_train ref + X_test ref junts.

v3.1:
  - Profile weekend/weekday load coherent
  - Pèrdua MAE (mètrica del challenge)
  - Model més petit (256 / 4+4) per reduir temps d'entrenament
  - Val masks congelades + TTA per soroll

Execució:
    cd data_challenge
    python -m lstm.main_train_combined
"""
import numpy as np
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
    _windows = False

import pandas as pd
import torch
from torch.utils.data import DataLoader

from lstm.src.config import (
    X_TRAIN_PATH, X_TEST_PATH, Y_TRAIN_PATH,
    PRED_TRAIN_PATH, OUT_DIR, CKPT_DIR,
    BATCH_SIZE, DEBUG, DEBUG_N_CURVES, DEBUG_N_EPOCHS,
)
from lstm.src.model   import TemporalMaskedAutoencoder
from lstm.src.trainer import train, predict_batch_tta
from lstm.src.dataset import (
    InferenceDataset, compute_global_stats, fit_mask_distribution,
    robust_stats,
)
from src.data_loader  import load_train, load_test, save_submission
from src.evaluator    import print_summary

COMBINED_CKPT = CKPT_DIR / "best_model_v3.pt"

# ── Resume (Kaggle / local) ───────────────────────────────────────────────────
# False = entrenament des de zero. True = continua des de RESUME_FROM.
RESUME = False
RESUME_FROM = None  # p.ex. "/kaggle/working/DataChallenge_checkpoints/latest.pt"


def resolve_ckpt_dir() -> Path:
    """Kaggle → /kaggle/working/DataChallenge_checkpoints ; si no, lstm/checkpoints."""
    kaggle_working = Path("/kaggle/working")
    if kaggle_working.exists():
        d = kaggle_working / "DataChallenge_checkpoints"
    else:
        d = CKPT_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def normalize_robust(values: np.ndarray, label: str) -> tuple:
    """
    NOU v3: Normalització robusta amb mediana i MAD per corba.

    Elimina corbes amb NaN o massa planes. Retorna (norm, raw).
    """
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


def normalize_holed_robust(X_holed: pd.DataFrame) -> tuple:
    """NOU v3: Normalització robusta de corbes holed_."""
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
    if DEBUG:
        print("=" * 50)
        print("  MODE DEBUG")
        print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. Carregar X_train ──────────────────────────────────────────────────
    print("\n── Carregant X_train ──")
    X_ref_train, X_holed_train, y_train = load_train(X_TRAIN_PATH, Y_TRAIN_PATH)

    ref_train_vals = X_ref_train.select_dtypes("number").to_numpy(dtype=np.float32).T
    ref_train_norm, ref_train_raw = normalize_robust(ref_train_vals, "X_train ref")

    # ── 2. Carregar X_test ───────────────────────────────────────────────────
    print("\n── Carregant X_test ──")
    X_ref_test, X_holed_test = load_test(X_TEST_PATH)

    ref_test_vals = X_ref_test.select_dtypes("number").to_numpy(dtype=np.float32).T
    ref_test_norm, ref_test_raw = normalize_robust(ref_test_vals, "X_test ref")

    # ── 3. Combinar ──────────────────────────────────────────────────────────
    print("\n── Combinant corbes de referència ──")
    ref_combined     = np.concatenate([ref_train_norm, ref_test_norm], axis=0)
    ref_combined_raw = np.concatenate([ref_train_raw,  ref_test_raw],  axis=0)
    print(f"Total corbes d'entrenament: {len(ref_combined):,}")

    # ── 4. Estadístiques globals per profiles ───────────────────────────────
    global_stats = compute_global_stats(ref_combined_raw)

    # ── 5. NOU v3: Distribució empírica de forats — TRAIN + TEST holed_ ──────
    # Combinem les holed_ del train (1000) i del test (1000) per tenir 2000
    # mostres en lloc de 1000. Redueix el risc d'overfitting a una distribució
    # estimada amb poques corbes.
    print("\n── Aprenent distribució de forats (train + test holed_) ──")
    holed_train_vals = X_holed_train.values.T.astype(np.float32)
    holed_test_vals  = X_holed_test.values.T.astype(np.float32)
    holed_all_vals   = np.concatenate([holed_train_vals, holed_test_vals], axis=0)
    print(f"  Total holed_ per estimar la distribució: {len(holed_all_vals):,} "
          f"({len(holed_train_vals)} train + {len(holed_test_vals)} test)")
    mask_dist = fit_mask_distribution(holed_all_vals)
    print(f"  Ratios per corba: "
          f"min={mask_dist['ratios'].min():.3f} "
          f"med={np.median(mask_dist['ratios']):.3f} "
          f"max={mask_dist['ratios'].max():.3f}")
    print(f"  Longituds de blocs: "
          f"min={mask_dist['block_lengths'].min()} "
          f"med={int(np.median(mask_dist['block_lengths']))} "
          f"max={mask_dist['block_lengths'].max()}")

    if DEBUG:
        ref_combined     = ref_combined[:DEBUG_N_CURVES]
        ref_combined_raw = ref_combined_raw[:DEBUG_N_CURVES]
        print(f"[DEBUG] Usant {len(ref_combined)} corbes")

    # ── 6. Entrenament ───────────────────────────────────────────────────────
    ckpt_save_dir = resolve_ckpt_dir()
    resume_from = None
    if RESUME:
        resume_from = Path(RESUME_FROM) if RESUME_FROM else (ckpt_save_dir / "latest.pt")
        print(f"\nRESUME=True → {resume_from}")

    print("\n── Entrenament v3.1 (MAE, model 256/4+4) ──")
    model = train(ref_combined, device,
                  n_epochs_override=DEBUG_N_EPOCHS if DEBUG else None,
                  ckpt_path=COMBINED_CKPT,
                  use_hybrid_loss=False,  # mètrica del challenge = MAE
                  curves_raw=ref_combined_raw,
                  global_stats=global_stats,
                  mask_dist=mask_dist,
                  resume_from=resume_from,
                  ckpt_dir=ckpt_save_dir)

    print(f"\nPesos millor model (predicció): {COMBINED_CKPT}")
    print(f"Checkpoints complets:           {ckpt_save_dir}")

    # ── 7. Avaluació sobre X_train holed_ ────────────────────────────────────
    print("\n── Avaluació sobre X_train holed_ ──")
    holed_norm, nan_masks, meds, mads = normalize_holed_robust(X_holed_train)
    holed_raw = X_holed_train.values.T.astype(np.float32)

    inf_loader = DataLoader(
        InferenceDataset(holed_norm, nan_masks,
                         curves_raw=holed_raw,
                         global_stats=global_stats),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("  Inferència amb Test-Time Augmentation...")
    preds_norm = predict_batch_tta(model, inf_loader, device)

    N, T      = preds_norm.shape
    preds_raw = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        preds_raw[i] = preds_norm[i] * mads[i] + meds[i]

    preds_raw = np.clip(preds_raw, 0, None)

    # Preservar valors coneguts
    holed_original = X_holed_train.values.T.astype(np.float32)
    known_mask     = ~np.isnan(holed_original)
    preds_raw[known_mask] = holed_original[known_mask]

    y_pred = pd.DataFrame(preds_raw.T,
                          columns=X_holed_train.columns,
                          index=X_holed_train.index)

    nan_mask_df = pd.DataFrame(nan_masks.T,
                               columns=X_holed_train.columns,
                               index=X_holed_train.index)

    print_summary(y_train, y_pred, nan_mask_df, model_name="Transformer v3.1")

    if not DEBUG:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamps = X_holed_train.index.strftime("%Y-%m-%d %H:%M:%S").values
        save_submission(y_pred, PRED_TRAIN_PATH, timestamps)

    print(f"\nAra executa: python -m lstm.main_predict_combined")

    if _windows:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
