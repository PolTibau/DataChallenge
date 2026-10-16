"""
main_predict.py
---------------
Genera la predicció final sobre el conjunt de test.

Execució:
    python -m knn.main_predict

Genera:
    outputs/y_pred_test.csv  ← fitxer de submissió al challenge
"""

import pandas as pd

from knn.core.config import (
    X_TEST_PATH, PRED_TEST_PATH, OUT_DIR,
    K_NEIGHBORS, DISTANCE_METRIC, WEIGHTING, GAUSSIAN_SIGMA, INDEX_COL,
)
from src.data_loader import load_test
from knn.core.knn_imputer import impute_all


def main():
    print("\n── Carregant dades de test ──")
    # load_test ara retorna també els timestamps originals
    X_ref, X_holed, horodates = load_test(X_TEST_PATH)

    print("\n── Imputació KNN ──")
    y_pred = impute_all(
        X_holed       = X_holed,
        X_ref         = X_ref,
        k             = K_NEIGHBORS,
        metric        = DISTANCE_METRIC,
        weighting     = WEIGHTING,
        gaussian_sigma= GAUSSIAN_SIGMA,
    )

    # Guardar CSV amb format exacte del challenge:
    # Horodate | holed_1 | holed_2 | ... | holed_1000
    # sense índex de pandas (index=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    y_pred_csv = y_pred.reset_index(drop=True).copy()
    y_pred_csv.insert(0, INDEX_COL, horodates)
    y_pred_csv.to_csv(PRED_TEST_PATH, index=False)

    print(f"\nCSV de submissió guardat a: {PRED_TEST_PATH}")
    print(f"Shape: {y_pred_csv.shape}")
    print(f"Columnes[:3]: {y_pred_csv.columns[:3].tolist()}")
    print(f"Primers Horodate: {list(y_pred_csv[INDEX_COL][:3])}")
    print(f"NaN restants: {y_pred.isna().sum().sum()}")


if __name__ == "__main__":
    main()
