"""
lstm/main_train_combined.py — v3.1
----------------------------------
Entrena el Transformer v3.1 sobre X_train ref + X_test ref junts.

v3.1:
  - Profile weekend/weekday load coherent
  - Pèrdua MAE (mètrica del challenge)
  - Model 192 / 4+4 / D_FF=768 per reduir temps d'entrenament
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
    BATCH_SIZE, DEBUG, DEBUG_N_CURVES, DEBUG_N_EPOCHS, RANDOM_SEED,
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

# Quantes de les 1.000 corbes holed_ de train es reserven com a validació
# HONESTA (mai vistes en entrenament, ni com a corba ni com a estadística
# de màscara). La resta (1000 - N_REAL_VAL) s'afegeixen com a corbes de
# referència addicionals, ja que en coneixem el valor complet via y_train.
N_REAL_VAL = 200

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

    # ── 5b. Validació real + corbes extra d'entrenament ───────────────────────
    # Fins ara el "millor checkpoint" es triava per la loss de validació sobre
    # màscares SINTÈTIQUES aplicades a corbes completes (ref_). És una bona
    # aproximació però no és la mètrica del challenge: no usa mai els 1.000
    # parells (holed_, y_train) que sí que tenim etiquetats de veritat.
    #
    # Aquí en reservem 200 com a validació honesta (holed_ real + ground
    # truth real, MAI usades en entrenament) i afegim les 800 restants com a
    # corbes de referència addicionals -en sabem el valor complet via
    # y_train, així que es poden usar igual que qualsevol altra corba ref_,
    # emmascarant-les sintèticament a cada època.
    print(f"\n── Separant validació real ({N_REAL_VAL}) de corbes extra "
          f"d'entrenament ──")
    holed_cols = X_holed_train.columns.to_numpy()

    # Algunes corbes (p.ex. holed_192) tenen y_train gairebé tot NaN — el
    # ground truth és incomplet i no es pot fer servir per validar. Si una
    # d'aquestes caigués al grup de validació honesta, el MAE real donaria
    # NaN a cada època i el "improved" no seria mai True → cap checkpoint
    # es guardaria mai. Les excloem del sorteig de validació (encara poden
    # anar al grup "extra", on normalize_robust ja les descarta soles).
    y_nan_cols = y_train.columns[y_train.isna().any(axis=0)].to_numpy()
    if len(y_nan_cols) > 0:
        print(f"  Avís: {len(y_nan_cols)} corba(es) amb y_train incomplet, "
              f"excloses de la validació real: {list(y_nan_cols)}")
    clean_cols = holed_cols[~np.isin(holed_cols, y_nan_cols)]
    bad_cols   = holed_cols[np.isin(holed_cols, y_nan_cols)]

    perm       = np.random.default_rng(RANDOM_SEED + 123).permutation(len(clean_cols))
    val_cols   = clean_cols[perm[:N_REAL_VAL]]
    extra_cols = np.concatenate([clean_cols[perm[N_REAL_VAL:]], bad_cols])

    X_val_holed = X_holed_train[val_cols]
    y_val       = y_train[val_cols]
    val_norm, val_nanmask, val_meds, val_mads = normalize_holed_robust(X_val_holed)

    real_val = {
        "norm":         val_norm,
        "nanmask":      val_nanmask,
        "raw":          X_val_holed.values.T.astype(np.float32),
        "meds":         val_meds,
        "mads":         val_mads,
        "y_raw":        y_val.values.T.astype(np.float32),
        "global_stats": global_stats,
    }
    print(f"  Validació real: {len(val_cols)} corbes (mai vistes en l'entrenament)")

    y_extra_raw            = y_train[extra_cols].values.T.astype(np.float32)
    extra_norm, extra_raw  = normalize_robust(y_extra_raw, "y_train holed_ (extra)")
    ref_combined            = np.concatenate([ref_combined, extra_norm], axis=0)
    ref_combined_raw        = np.concatenate([ref_combined_raw, extra_raw], axis=0)
    print(f"  Entrenament addicional: {len(extra_cols)} corbes completades (y_train)")
    print(f"  Total corbes d'entrenament: {len(ref_combined):,}")

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

    print("\n── Entrenament v3.1 (MAE, D_MODEL=192, 4+4, D_FF=768) ──")
    model = train(ref_combined, device,
                  n_epochs_override=DEBUG_N_EPOCHS if DEBUG else None,
                  ckpt_path=COMBINED_CKPT,
                  use_hybrid_loss=False,  # mètrica del challenge = MAE
                  curves_raw=ref_combined_raw,
                  global_stats=global_stats,
                  mask_dist=mask_dist,
                  resume_from=resume_from,
                  ckpt_dir=ckpt_save_dir,
                  real_val=real_val)

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

    print("\n  NOTA: 800 d'aquestes 1.000 corbes s'han fet servir com a corbes "
          "de referència ADDICIONALS a l'entrenament (secció 5b) — el model ja "
          "n'ha après el valor real sota altres màscares, així que el seu MAE "
          "aquí és optimista. Les 200 restants no s'han fet servir mai i són "
          "la mesura fiable (és la mateixa validació usada per triar el "
          "millor checkpoint durant l'entrenament).")

    print_summary(y_train[val_cols], y_pred[val_cols], nan_mask_df[val_cols],
                  model_name="Transformer v3.1 — validació HONESTA (200)")
    print_summary(y_train[extra_cols], y_pred[extra_cols], nan_mask_df[extra_cols],
                  model_name="Transformer v3.1 — vistes en training (800, optimista)")
    print_summary(y_train, y_pred, nan_mask_df,
                  model_name="Transformer v3.1 — total 1000 (biaixat a l'alça)")

    if not DEBUG:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamps = X_holed_train.index.strftime("%Y-%m-%d %H:%M:%S").values
        save_submission(y_pred, PRED_TRAIN_PATH, timestamps)

    print(f"\nAra executa: python -m lstm.main_predict")

    if _windows:
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
