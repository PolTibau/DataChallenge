"""
main_train.py
-------------
Avaluació sobre el conjunt de train.

Execució:
    python main_train.py

Genera:
    outputs/y_pred_train.csv
"""

import pandas as pd
import numpy as np

from src.config import (
    X_TRAIN_PATH, Y_TRAIN_PATH, PRED_TRAIN_PATH, OUT_DIR,
    K_NEIGHBORS, DISTANCE_METRIC, WEIGHTING, GAUSSIAN_SIGMA, INDEX_COL,
)
from src.data_loader  import load_train
from src.preprocessor import get_nan_mask
from src.knn_imputer  import impute_all


def compute_mae(y_train, y_pred, nan_mask):
    """MAE sobre els punts imputats, ignorant posicions on y_train és NaN."""
    cols = [c for c in y_pred.columns if c in y_train.columns]
    errors = []
    skipped = 0
    for col in cols:
        mask = nan_mask[col].values.astype(bool)
        if mask.sum() == 0:
            continue
        tv = y_train[col].values[mask]
        pv = y_pred[col].values[mask]
        valid = ~(np.isnan(tv) | np.isnan(pv))
        skipped += (~valid).sum()
        if valid.sum() > 0:
            errors.extend(np.abs(tv[valid] - pv[valid]).tolist())
    return errors, int(skipped)


def main():
    print("\n── Carregant dades de train ──")
    X_ref, X_holed, y_train = load_train(X_TRAIN_PATH, Y_TRAIN_PATH)

    nan_mask = get_nan_mask(X_holed)
    print(f"Punts a imputar: {nan_mask.sum().sum():,}")

    print("\n── Imputació KNN ──")
    y_pred = impute_all(
        X_holed       = X_holed,
        X_ref         = X_ref,
        k             = K_NEIGHBORS,
        metric        = DISTANCE_METRIC,
        weighting     = WEIGHTING,
        gaussian_sigma= GAUSSIAN_SIGMA,
    )

    print("\n── Resultats ──")
    y_pred_r   = y_pred.reset_index(drop=True)
    y_train_r  = y_train.reset_index(drop=True)
    nan_mask_r = nan_mask.reset_index(drop=True)

    errors, skipped = compute_mae(y_train_r, y_pred_r, nan_mask_r)

    mae_curves = {}
    for col in [c for c in y_pred_r.columns if c in y_train_r.columns]:
        mask = nan_mask_r[col].values.astype(bool)
        if mask.sum() == 0:
            continue
        tv = y_train_r[col].values[mask]
        pv = y_pred_r[col].values[mask]
        valid = ~(np.isnan(tv) | np.isnan(pv))
        if valid.sum() > 0:
            mae_curves[col] = float(np.mean(np.abs(tv[valid] - pv[valid])))
    s = pd.Series(mae_curves)

    print("=" * 50)
    print(f"  MAE total        : {np.mean(errors):.2f} W")
    print(f"  MAE mitjà/corba  : {s.mean():.2f} W")
    print(f"  MAE mediana/corba: {s.median():.2f} W")
    print(f"  Pitjor corba     : {s.idxmax()} ({s.max():.2f} W)")
    print(f"  Millor corba     : {s.idxmin()} ({s.min():.2f} W)")
    if skipped > 0:
        print(f"  Punts omesos (NaN al ground truth): {skipped}")
    print("=" * 50)

    # Guardar — Horodate com a primera columna, index=False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y_pred_csv = y_pred.copy()
    y_pred_csv.index = X_holed.index.strftime("%Y-%m-%d %H:%M:%S")
    y_pred_csv.index.name = INDEX_COL
    y_pred_csv.to_csv(PRED_TRAIN_PATH, index=True)
    print(f"\nPrediccions guardades a: {PRED_TRAIN_PATH}")


if __name__ == "__main__":
    main()
