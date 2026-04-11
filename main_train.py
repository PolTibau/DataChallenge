"""
main_train.py
-------------
Script principal per entrenar/avaluar sobre el conjunt de train.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

for mod in list(sys.modules.keys()):
    if mod.startswith("src."):
        del sys.modules[mod]

from src.config import (
    X_TRAIN_PATH, Y_TRAIN_PATH, PRED_TRAIN_PATH, OUT_DIR,
    K_NEIGHBORS, DISTANCE_METRIC, WEIGHTING, GAUSSIAN_SIGMA,
)
from src.data_loader  import load_train
from src.preprocessor import get_nan_mask
from src.knn_imputer  import impute_all
from src.evaluator    import print_summary


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

    # Resetar índex per garantir alineació per posició
    y_pred_r   = y_pred.reset_index(drop=True)
    y_train_r  = y_train.reset_index(drop=True)
    nan_mask_r = nan_mask.reset_index(drop=True)

    zero_curves = (y_pred_r == 0).all()
    print(f"  Corbes 100% zeros: {zero_curves.sum()}")

    # MAE inline — comprova NaN a y_train i y_pred
    cols = [c for c in y_pred_r.columns if c in y_train_r.columns]
    errors = []
    nan_ytrain, nan_ypred = 0, 0
    for col in cols:
        mask = nan_mask_r[col].values.astype(bool)
        if mask.sum() == 0:
            continue
        tv = y_train_r[col].values[mask]
        pv = y_pred_r[col].values[mask]
        nan_ytrain += int(np.isnan(tv).sum())
        nan_ypred  += int(np.isnan(pv).sum())
        valid = ~(np.isnan(tv) | np.isnan(pv))
        if valid.sum() > 0:
            errors.extend(np.abs(tv[valid] - pv[valid]).tolist())

    print(f"  NaN a y_train on mask=True: {nan_ytrain}")
    print(f"  NaN a y_pred  on mask=True: {nan_ypred}")
    print(f"  Punts avaluats: {len(errors):,}")
    if errors:
        print(f"  MAE total (inline): {np.mean(errors):.2f} W")
        mae_curves = {}
        for col in cols:
            mask = nan_mask_r[col].values.astype(bool)
            if mask.sum() == 0:
                continue
            tv = y_train_r[col].values[mask]
            pv = y_pred_r[col].values[mask]
            valid = ~(np.isnan(tv) | np.isnan(pv))
            if valid.sum() > 0:
                mae_curves[col] = float(np.mean(np.abs(tv[valid] - pv[valid])))
        s = pd.Series(mae_curves)
        print(f"  MAE mitjà/corba  : {s.mean():.2f} W")
        print(f"  MAE mediana/corba: {s.median():.2f} W")
        print(f"  Pitjor corba     : {s.idxmax()} ({s.max():.2f} W)")
        print(f"  Millor corba     : {s.idxmin()} ({s.min():.2f} W)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y_pred.to_parquet(PRED_TRAIN_PATH, index=True)
    print(f"\nPrediccions guardades a: {PRED_TRAIN_PATH}")


if __name__ == "__main__":
    main()
