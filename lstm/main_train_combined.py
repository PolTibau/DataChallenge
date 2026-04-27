"""
lstm/main_train_combined.py
---------------------------
Entrena el Transformer sobre X_train ref + X_test ref junts (57.140 corbes).

Per què?
--------
X_train ref (20.000) i X_test ref (37.140) són corbes completament
independents del mateix període. Entrenar-les juntes dona al model
molta més diversitat de patrons sense cap risc de data leakage —
les corbes holed_ del test mai entren a l'entrenament.

Els CSV originals NO es modifiquen — la combinació es fa en memòria.

Flux:
  1. Carregar corbes ref de X_train (20.000 senceres)
  2. Carregar corbes ref de X_test  (37.140 senceres)
  3. Concatenar en memòria          (57.140 senceres)
  4. Entrenar el Transformer sobre les 57.140 corbes
  5. Guardar best_model_combined.pt

Després:
  python -m lstm.main_predict   ← usa best_model_combined.pt

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
from lstm.src.trainer import train, predict_batch
from lstm.src.dataset import InferenceDataset, compute_global_stats
from src.data_loader  import load_train, load_test, load_csv, save_submission
from src.evaluator    import print_summary

COMBINED_CKPT = CKPT_DIR / "best_model_combined.pt"

HOLED_PREFIX = "holed_"


def normalize_and_clean(values: np.ndarray, label: str) -> tuple:
    """
    Normalitza per zscore i elimina corbes amb NaN o std=0.
    Retorna (norm, raw) — raw per calcular profiles.
    No modifica l'array original.
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
        mu, sig  = values[i].mean(), values[i].std()
        norm[i]  = (values[i] - mu) / (sig if sig > 1e-6 else 1.0)

    print(f"[{label}] Corbes vàlides: {len(values):,} | "
          f"NaN restants: {np.isnan(norm).sum()}")
    return norm, values   # (norm, raw)


def normalize_holed(X_holed: pd.DataFrame) -> tuple:
    """Normalitza corbes holed_ i guarda stats per desnormalitzar."""
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
    if DEBUG:
        print("=" * 50)
        print("  MODE DEBUG")
        print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── 1. Carregar corbes de referència de X_train ───────────────────────────
    print("\n── Carregant X_train ──")
    X_ref_train, X_holed_train, y_train = load_train(X_TRAIN_PATH, Y_TRAIN_PATH)

    ref_train_vals = X_ref_train.select_dtypes("number").to_numpy(dtype=np.float32).T
    ref_train_norm, ref_train_raw = normalize_and_clean(ref_train_vals, "X_train ref")

    # ── 2. Carregar corbes de referència de X_test ────────────────────────────
    print("\n── Carregant X_test ──")
    X_ref_test, X_holed_test = load_test(X_TEST_PATH)

    ref_test_vals = X_ref_test.select_dtypes("number").to_numpy(dtype=np.float32).T
    ref_test_norm, ref_test_raw = normalize_and_clean(ref_test_vals, "X_test ref")

    # ── 3. Concatenar en memòria ──────────────────────────────────────────────
    print("\n── Combinant corbes de referència ──")
    ref_combined     = np.concatenate([ref_train_norm, ref_test_norm], axis=0)
    ref_combined_raw = np.concatenate([ref_train_raw,  ref_test_raw],  axis=0)
    print(f"Total corbes d'entrenament: {len(ref_combined):,}")
    print(f"  X_train ref: {len(ref_train_norm):,}")
    print(f"  X_test  ref: {len(ref_test_norm):,}")

    # Calcular estadístiques globals per normalitzar profiles
    print("Calculant estadístiques globals per profiles...")
    global_stats = compute_global_stats(ref_combined_raw)
    print(f"  mu_global={global_stats[0]:.1f}W  "
          f"sigma_global={global_stats[1]:.1f}W  "
          f"cv_global={global_stats[2]:.3f}")

    if DEBUG:
        ref_combined     = ref_combined[:DEBUG_N_CURVES]
        ref_combined_raw = ref_combined_raw[:DEBUG_N_CURVES]
        print(f"[DEBUG] Usant {len(ref_combined)} corbes")

    # ── 4. Entrenar ───────────────────────────────────────────────────────────
    print("\n── Entrenament combinat ──")
    import lstm.main_train_combined as _self
    ckpt = _self.COMBINED_CKPT
    model = train(ref_combined, device,
                  n_epochs_override=DEBUG_N_EPOCHS if DEBUG else None,
                  ckpt_path=ckpt,
                  use_hybrid_loss=True,
                  curves_raw=ref_combined_raw,
                  global_stats=global_stats)

    print(f"\nModel combinat guardat a: {ckpt}")

    # ── 5. Avaluació sobre X_train holed_ (tenim y_train) ────────────────────
    print("\n── Avaluació sobre X_train holed_ ──")
    holed_norm, nan_masks, mus, sigmas = normalize_holed(X_holed_train)
    holed_raw = X_holed_train.values.T.astype(np.float32)

    inf_loader = DataLoader(
        InferenceDataset(holed_norm, nan_masks,
                         curves_raw=holed_raw,
                         global_stats=global_stats),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    preds_norm = predict_batch(model, inf_loader, device)

    N, T      = preds_norm.shape
    preds_raw = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        preds_raw[i] = preds_norm[i] * sigmas[i] + mus[i]

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

    print_summary(y_train, y_pred, nan_mask_df, model_name="Transformer Combined")

    if not DEBUG:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamps = X_holed_train.index.strftime("%Y-%m-%d %H:%M:%S").values
        save_submission(y_pred, PRED_TRAIN_PATH, timestamps)
        print(f"Prediccions train guardades a: {PRED_TRAIN_PATH}")

    print(f"\nAra executa: python -m lstm.main_predict_combined")

    if _windows:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
