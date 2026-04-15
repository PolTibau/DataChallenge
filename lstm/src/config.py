"""
lstm/src/config.py
------------------
Paràmetres del Transformer. Modifica aquí per experimentar.

Configuració actual: model gran per entrenament nocturn en CPU.
  - D_MODEL=192 (era 128) — model més gran, millor generalització
  - Entrenament combinat X_train + X_test ref (55.843 corbes)
  - MASK_RATIO=0.20 — lleugerament més desafiant que 0.15
  - Early stopping PATIENCE=6 — atura si no millora
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
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
SEQ_LEN = 1057

# ── Masking ───────────────────────────────────────────────────────────────────
MASK_RATIO      = 0.22   # lleugerament més que 0.15, sense ser excessiu
MASK_MIN_LEN    = 1
MASK_MAX_LEN    = 96
MASK_BLOCK_PROB = 0.7

# ── Arquitectura ──────────────────────────────────────────────────────────────
# Model gran — compromís CPU/qualitat
D_MODEL      = 192   # era 128 — +50% capacitat
N_HEADS      = 8     # 192 / 8 = 24 dims per cap (valid)
N_ENC_LAYERS = 4
N_DEC_LAYERS = 4
D_FF         = 768   # típicament 4 × D_MODEL
DROPOUT      = 0.1
MAX_SEQ_LEN  = 2048

# ── Entrenament ───────────────────────────────────────────────────────────────
BATCH_SIZE      = 32     # reduït per compensar model més gran en CPU
N_EPOCHS        = 60
LR              = 1e-4
LR_WARMUP_STEPS = 500
WEIGHT_DECAY    = 1e-5
GRAD_CLIP       = 1.0
PATIENCE        = 6      # para abans per evitar overfitting
VAL_SPLIT       = 0.1

# ── Fine-tuning ───────────────────────────────────────────────────────────────
N_FINETUNE_EPOCHS = 20
LR_FINETUNE       = 3e-5
PATIENCE_FT       = 6

# ── Mode de prova ─────────────────────────────────────────────────────────────
DEBUG          = False    # ← canvia a False per l'entrenament real nocturn
DEBUG_N_CURVES = 200
DEBUG_N_EPOCHS = 2

# ── Reproducibilitat ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
