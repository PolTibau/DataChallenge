"""
lstm/src/config.py — v3.1
-------------------------
Paràmetres del Transformer. Modifica aquí per experimentar.

v3.1 (correccions + velocitat):
  - Model més petit (256 / 4+4) — 8+8 @ 384 costava >12h amb millora marginal
  - Batch 64, warmup 1000, patience 8 — menys steps i parada més aviat
  - TTA per soroll (no shifts cíclics) — no desalineja embeddings de calendari
  - DEBUG=False per defecte
  - Profile [3] = weekend/weekday load ratio (coherent train/infer)
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
# NOU v3: les màscares ara s'aprenen del patró real del test, no aleatori uniforme.
# Aquests paràmetres són fallback si no es proporciona distribució empírica.
MASK_RATIO      = 0.25
MASK_MIN_LEN    = 1
MASK_MAX_LEN    = 96
MASK_BLOCK_PROB = 0.7

# NOU v3: probabilitat de fer servir el patró empíric vs aleatori uniforme
# A l'entrenament barregem els dos per donar robustesa
EMPIRICAL_MASK_PROB = 0.6   # 60% màscares realistes, 40% aleatòries

# ── Arquitectura ──────────────────────────────────────────────────────────────
D_MODEL      = 192
N_HEADS      = 8         # 256/8 = 32 dims per head
N_ENC_LAYERS = 4
N_DEC_LAYERS = 4
D_FF         = 768      # 4 × D_MODEL
DROPOUT      = 0.1
MAX_SEQ_LEN  = 2048

# NOU v3: profile dim ampliat
PROFILE_DIM  = 9

# ── Entrenament ───────────────────────────────────────────────────────────────
BATCH_SIZE      = 64     # el model més petit cap amb batch més gran → menys steps
N_EPOCHS        = 80
LR              = 1e-4
LR_WARMUP_STEPS = 1000
WEIGHT_DECAY    = 1e-5
GRAD_CLIP       = 1.0
PATIENCE        = 8
VAL_SPLIT       = 0.1

# ── Test-time augmentation ───────────────────────────────────────────────────
# Passades a inferència (1 = sense TTA). Soroll gaussità als punts observats;
# NO shifts cíclics: desalinearien hora-del-dia vs. embeddings de calendari.
TTA_N_PASSES  = 3
TTA_NOISE_STD = 0.02     # en espai normalitzat (mediana/MAD)

# ── Mode de prova ─────────────────────────────────────────────────────────────
DEBUG          = False
DEBUG_N_CURVES = 200
DEBUG_N_EPOCHS = 2

# ── Reproducibilitat ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
