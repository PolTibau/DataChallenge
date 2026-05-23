"""
lstm/src/config.py — v3
-----------------------
Paràmetres del Transformer. Modifica aquí per experimentar.

Canvis v3 respecte la versió anterior:
  - PATCH_SIZE=2 (1h per token) — màxima resolució temporal
    Cada token representa 1 hora, granularitat natural del comportament humà
  - PROFILE_DIM ampliat de 4 a 9 — més estadístiques discriminatives
    [mu, sigma_robust, cv, weekend_ratio, peak_day_ratio, skewness,
     autocorr_lag48, p95, q3_q1_iqr]
  - Hora × tipus_dia com a embedding conjunt (96 entrades) en lloc de
    sumar hour + weekday + workday per separat
  - Normalització robusta (mediana / MAD) en lloc de mean / std
  - Màscares realistes durant l'entrenament basades en patrons reals del test
  - Bigger model: D_MODEL=384, més capacitat per aprofitar GPU

Configuració actual: model gran per GPU.
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
# NOU v3: model més gran per aprofitar GPU. Amb PATCH_SIZE=2 tenim ~530 tokens,
# així que cal capacitat suficient per processar aquesta seqüència.
D_MODEL      = 384
N_HEADS      = 8         # 384/8 = 48 dims per head — múltiple net
N_ENC_LAYERS = 8
N_DEC_LAYERS = 8
D_FF         = 1536      # 4 × D_MODEL com és habitual
DROPOUT      = 0.1
MAX_SEQ_LEN  = 2048

# NOU v3: profile dim ampliat
PROFILE_DIM  = 9

# ── Entrenament ───────────────────────────────────────────────────────────────
BATCH_SIZE      = 48     # reduït lleugerament pel model més gran (cabrà a GPU típica)
N_EPOCHS        = 100    # més èpoques, el model és més gran
LR              = 1e-4
LR_WARMUP_STEPS = 2000   # warmup més llarg per al model gran
WEIGHT_DECAY    = 1e-5
GRAD_CLIP       = 1.0
PATIENCE        = 12     # més paciència, el model gran convergeix més lent
VAL_SPLIT       = 0.1

# ── Test-time augmentation ───────────────────────────────────────────────────
# NOU v3: nombre de passades per corba a l'inferència (1 = sense TTA)
# El TTA promedia múltiples prediccions amb shifts cíclics petits
TTA_N_PASSES = 4
TTA_SHIFTS   = [0, 12, 24, 36]   # shifts en timesteps (0h, 6h, 12h, 18h)

# ── Mode de prova ─────────────────────────────────────────────────────────────
DEBUG          = False
DEBUG_N_CURVES = 200
DEBUG_N_EPOCHS = 2

# ── Reproducibilitat ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
