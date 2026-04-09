"""
main_train.py
-------------
Script principal per entrenar/avaluar sobre el conjunt de train.

Execució:
    python main_train.py

Genera:
    outputs/y_pred_train.parquet   → prediccions de les 1.000 corbes holed_
    (imprimeix MAE vs. ground truth y_train)
"""

import pandas as pd
from pathlib import Path

from src.config import (
    X_TRAIN_PATH, Y_TRAIN_PATH, PRED_TRAIN_PATH, OUT_DIR,
    K_NEIGHBORS, DISTANCE_METRIC, WEIGHTING
)
from src.data_loader   import load_train
from src.preprocessor  import get_nan_mask
from src.knn_imputer   import impute_all
from src.evaluator     import print_summary


def main():
    # ── 1. Carregar dades ────────────────────────────────────────────────────
    print("\n── Carregant dades de train ──")
    X_ref, X_holed, y_train = load_train(X_TRAIN_PATH, Y_TRAIN_PATH)

    # ── 2. Màscara NaN (necessitem saber quins punts hem imputat) ────────────
    nan_mask = get_nan_mask(X_holed)
    print(f"Punts a imputar: {nan_mask.sum().sum():,}")

    # ── 3. Imputació KNN ─────────────────────────────────────────────────────
    print("\n── Imputació KNN ──")
    y_pred = impute_all(
        X_holed  = X_holed,
        X_ref    = X_ref,
        k        = K_NEIGHBORS,
        metric   = DISTANCE_METRIC,
        weighting= WEIGHTING,
    )

    # ── 4. Avaluació ─────────────────────────────────────────────────────────
    print("\n── Resultats ──")
    # Alinear columnes (y_train, nan_mask i y_pred han de tenir el mateix ordre)
    y_true_aligned = y_train[y_pred.columns]
    nan_mask_aligned = nan_mask[y_pred.columns]
    print_summary(y_true_aligned, y_pred, nan_mask_aligned)

    # ── 5. Guardar prediccions ───────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y_pred.to_parquet(PRED_TRAIN_PATH)
    print(f"\nPrediccions guardades a: {PRED_TRAIN_PATH}")


if __name__ == "__main__":
    main()
