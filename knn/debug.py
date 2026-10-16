"""
debug.py
--------
Script de diagnòstic. Executa'l i envia'm la sortida completa.
    python debug.py
"""

import pandas as pd
import numpy as np
from knn.core.config import X_TRAIN_PATH, Y_TRAIN_PATH

print("=" * 60)
print("DIAGNÒSTIC — llegint fitxers directament")
print("=" * 60)

# ── 1. Llegir en cru ──────────────────────────────────────────
X_raw = pd.read_parquet(X_TRAIN_PATH)
y_raw = pd.read_parquet(Y_TRAIN_PATH)

print(f"\nX_train shape brut : {X_raw.shape}")
print(f"y_train shape brut : {y_raw.shape}")
print(f"\nX_train index dtype: {X_raw.index.dtype}")
print(f"X_train index[:3]  : {X_raw.index[:3].tolist()}")
print(f"\ny_train index dtype: {y_raw.index.dtype}")
print(f"y_train index[:3]  : {y_raw.index[:3].tolist()}")

print(f"\nX_train columnes[:3]: {X_raw.columns[:3].tolist()}")
print(f"X_train columnes[-3:]: {X_raw.columns[-3:].tolist()}")
print(f"\ny_train columnes[:3]: {y_raw.columns[:3].tolist()}")
print(f"y_train columnes[-3:]: {y_raw.columns[-3:].tolist()}")

# ── 2. Dtypes de les columnes ─────────────────────────────────
non_num_X = X_raw.select_dtypes(exclude="number").columns.tolist()
non_num_y = y_raw.select_dtypes(exclude="number").columns.tolist()
print(f"\nColumnes no numèriques X_train: {non_num_X}")
print(f"Columnes no numèriques y_train: {non_num_y}")

# ── 3. Passar pel data_loader i veure el resultat ─────────────
print("\n" + "=" * 60)
print("Passant pel data_loader...")
print("=" * 60)
from src.data_loader import load_train
X_ref, X_holed, y_train = load_train(X_TRAIN_PATH, Y_TRAIN_PATH)

print(f"\nX_ref index[:3]   : {X_ref.index[:3].tolist()}")
print(f"X_holed index[:3] : {X_holed.index[:3].tolist()}")
print(f"y_train index[:3] : {y_train.index[:3].tolist()}")

print(f"\nX_ref index dtype  : {X_ref.index.dtype}")
print(f"X_holed index dtype: {X_holed.index.dtype}")
print(f"y_train index dtype: {y_train.index.dtype}")

# ── 4. Comprovar alineació d'índexs ──────────────────────────
idx_match = (X_holed.index == y_train.index).all()
print(f"\nX_holed.index == y_train.index: {idx_match}")

# ── 5. Comprovar la nan_mask ──────────────────────────────────
from knn.core.preprocessor import get_nan_mask
nan_mask = get_nan_mask(X_holed)
print(f"\nnan_mask shape: {nan_mask.shape}")
print(f"nan_mask True count: {nan_mask.sum().sum():,}")
print(f"nan_mask columnes[:3]: {nan_mask.columns[:3].tolist()}")

# ── 6. Simular compute_mae amb dades reals ────────────────────
print("\n" + "=" * 60)
print("Simulant compute_mae manualment...")
print("=" * 60)

# Agafem 3 columnes per provar
col = nan_mask.columns[0]
mask_col = nan_mask[col].values
print(f"\nColumna de prova: {col}")
print(f"  NaN al mask: {mask_col.sum()}")
print(f"  y_train[col] NaN: {y_train[col].isna().sum()}")
print(f"  y_train[col] valors on mask=True: {y_train[col].values[mask_col][:5]}")

# ── 7. Provar impute_column sobre una sola corba ─────────────
print("\n" + "=" * 60)
print("Provant impute_column sobre holed_1...")
print("=" * 60)
from knn.core.preprocessor import normalize
from knn.core.config import K_NEIGHBORS, DISTANCE_METRIC, WEIGHTING, GAUSSIAN_SIGMA, MIN_OVERLAP

X_ref_norm = normalize(X_ref).values.T.astype(float) if False else \
             X_ref.select_dtypes("number").to_numpy(dtype=float).T
X_ref_vals = X_ref.select_dtypes("number").to_numpy(dtype=float).T

test_col = "holed_1"
holed_s = X_holed[test_col]
print(f"NaN en {test_col}: {holed_s.isna().sum()}")
print(f"Valors coneguts: {holed_s.notna().sum()}")
print(f"MIN_OVERLAP configurat: {MIN_OVERLAP}")

valid_mask = holed_s.notna().values
print(f"Timesteps vàlids: {valid_mask.sum()}")
print(f"valid_mask.sum() >= MIN_OVERLAP: {valid_mask.sum() >= MIN_OVERLAP}")

print("\nDIAGNÒSTIC COMPLET.")
