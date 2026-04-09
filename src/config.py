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

X_TRAIN_PATH    = DATA_DIR / "X_train.parquet"
Y_TRAIN_PATH    = DATA_DIR / "y_train.parquet"
X_TEST_PATH     = DATA_DIR / "X_test.parquet"
SAMPLE_PATH     = DATA_DIR / "new_output_sample.parquet"

PRED_TRAIN_PATH = OUT_DIR / "y_pred_train.parquet"
PRED_TEST_PATH  = OUT_DIR / "y_pred_test.parquet"

# ── Identificació de columnes ─────────────────────────────────────────────────
HOLED_PREFIX = "holed_"       # prefix de les columnes a predir

# ── KNN ───────────────────────────────────────────────────────────────────────
K_NEIGHBORS     = 10          # nombre de veïns més propers
MIN_OVERLAP     = 50          # mínim de timesteps comuns per calcular distància vàlida
DISTANCE_METRIC = "euclidean" # "euclidean" | "cosine" | "manhattan" | "chebyshev"
WEIGHTING       = "inverse"   # "inverse" | "uniform" | "softmax" | "rank" | "gaussian"
GAUSSIAN_SIGMA  = 1.0         # σ del kernel gaussian (només si WEIGHTING="gaussian")

# ── Preprocessament ───────────────────────────────────────────────────────────
NORMALIZE   = True            # normalitzar cada corba abans de calcular distàncies
NORM_METHOD = "zscore"        # "zscore" | "robust" | "minmax" | "none"

# ── Reproducibilitat ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
