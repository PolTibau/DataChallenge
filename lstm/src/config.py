"""
lstm/src/config.py
------------------
Paràmetres del Transformer. Modifica aquí per experimentar.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]   # data_challenge/
DATA_DIR = ROOT_DIR / "data"
OUT_DIR  = ROOT_DIR / "outputs" / "lstm"
CKPT_DIR = ROOT_DIR / "lstm" / "checkpoints"

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
X_TEST_PATH  = DATA_DIR / "X_test.csv"

PRED_TRAIN_PATH = OUT_DIR / "y_pred_train.csv"
PRED_TEST_PATH  = OUT_DIR / "y_pred_test.csv"
BEST_CKPT       = CKPT_DIR / "best_model.pt"
LAST_CKPT       = CKPT_DIR / "last_model.pt"

# ── Seqüència ─────────────────────────────────────────────────────────────────
SEQ_LEN = 1057    # timesteps totals (30 min × 1057 ≈ 22 dies)

# ── Masking ───────────────────────────────────────────────────────────────────
MASK_RATIO      = 0.15
MASK_MIN_LEN    = 1
MASK_MAX_LEN    = 96     # 48h en timesteps de 30 min
MASK_BLOCK_PROB = 0.7

# ── Arquitectura ──────────────────────────────────────────────────────────────
D_MODEL      = 128
N_HEADS      = 8
N_ENC_LAYERS = 4
N_DEC_LAYERS = 4
D_FF         = 512
DROPOUT      = 0.1
MAX_SEQ_LEN  = 2048

# ── Entrenament ───────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
N_EPOCHS        = 50
LR              = 1e-4
LR_WARMUP_STEPS = 500
WEIGHT_DECAY    = 1e-5
GRAD_CLIP       = 1.0
PATIENCE        = 8
VAL_SPLIT       = 0.1

# ── Reproducibilitat ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
