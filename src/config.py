"""
config.py
---------
Paràmetres globals del projecte. Toca aquí per experimentar.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
OUT_DIR  = ROOT_DIR / "outputs"

# Fitxers d'entrada en format CSV
X_TRAIN_PATH = DATA_DIR / "X_train.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
X_TEST_PATH  = DATA_DIR / "X_test.csv"

# Fitxers de sortida
PRED_TRAIN_PATH = OUT_DIR / "y_pred_train.csv"
PRED_TEST_PATH  = OUT_DIR / "y_pred_test.csv"

# ── Identificació de columnes ─────────────────────────────────────────────────
HOLED_PREFIX  = "holed_"       # prefix de les columnes a predir
INDEX_COL     = "Horodate"     # columna de timestamps als CSV

# ── KNN ───────────────────────────────────────────────────────────────────────
K_NEIGHBORS     = 8
MIN_OVERLAP     = 33
DISTANCE_METRIC = "manhattan"  # "euclidean" | "cosine" | "manhattan" | "chebyshev"
WEIGHTING       = "rank"       # "inverse" | "uniform" | "softmax" | "rank" | "gaussian"
GAUSSIAN_SIGMA  = 1.0

# ── Preprocessament ───────────────────────────────────────────────────────────
NORMALIZE   = True
NORM_METHOD = "none"           # "zscore" | "robust" | "minmax" | "none"

# ── Reproducibilitat ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
