"""
main_predict.py
---------------
Script principal per generar la predicció final sobre el conjunt de test.

Execució:
    python main_predict.py

Genera:
    outputs/y_pred_test.parquet  → fitxer de submissió final
"""

import pandas as pd
from src.config import (
    X_TEST_PATH, PRED_TEST_PATH, OUT_DIR,
    K_NEIGHBORS, DISTANCE_METRIC, WEIGHTING
)
from src.data_loader  import load_test
from src.knn_imputer  import impute_all


def main():
    # ── 1. Carregar dades de test ────────────────────────────────────────────
    print("\n── Carregant dades de test ──")
    X_ref, X_holed = load_test(X_TEST_PATH)

    # ── 2. Imputació KNN ─────────────────────────────────────────────────────
    print("\n── Imputació KNN ──")
    y_pred = impute_all(
        X_holed  = X_holed,
        X_ref    = X_ref,
        k        = K_NEIGHBORS,
        metric   = DISTANCE_METRIC,
        weighting= WEIGHTING,
    )

    # ── 3. Guardar prediccions ───────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y_pred.to_parquet(PRED_TEST_PATH)
    print(f"\nFitxer de submissió guardat a: {PRED_TEST_PATH}")
    print(f"Shape: {y_pred.shape}  |  NaN restants: {y_pred.isna().sum().sum()}")


if __name__ == "__main__":
    main()
