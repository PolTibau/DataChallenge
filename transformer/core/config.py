"""
transformer/core/config.py
--------------------------
Paràmetres del Transformer d'imputació. Modifica aquí per experimentar.

Inclou:
  - Entrades de context per instant (màscara, interpolació, mateixa hora ±1 dia
    i ±1 setmana, perfil habitual, població, consumidors semblants)
  - Pèrdua ponderada per l'escala de la corba (≈ MAE en watts)
  - Validació amb totes les corbes holed_ etiquetades de train
  - Ensemble dels millors checkpoints + barreja amb interpolació als forats curts
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
OUT_DIR  = ROOT_DIR / "outputs" / "transformer"
CKPT_DIR = ROOT_DIR / "transformer" / "checkpoints"
RUNS_DIR = CKPT_DIR               # un subdirectori per entrenament: seed42/, seed43/...

X_TRAIN_PATH = DATA_DIR / "X_train.csv"
Y_TRAIN_PATH = DATA_DIR / "y_train.csv"
X_TEST_PATH  = DATA_DIR / "X_test.csv"

PRED_VAL_PATH  = OUT_DIR / "y_pred_train.csv"
PRED_TEST_PATH = OUT_DIR / "y_pred_test.csv"

# ── Seqüència i calendari ─────────────────────────────────────────────────────
SEQ_LEN       = 1057
STEPS_PER_DAY = 48
START_WEEKDAY = 0        # 2023-01-09 és dilluns (comprovat a les dades)

# ── Masking d'entrenament ─────────────────────────────────────────────────────
# Fallback uniforme; la major part de màscares segueixen el patró empíric.
MASK_RATIO      = 0.25
MASK_MIN_LEN    = 1
MASK_MAX_LEN    = 96
MASK_BLOCK_PROB = 0.7
EMPIRICAL_MASK_PROB = 0.6   # 60% màscares realistes, 40% aleatòries

# ── Context: consumidors semblants ────────────────────────────────────────────
N_NEIGHBOURS = 50        # veïns per construir la corba de "consumidors semblants"

# ── Arquitectura ──────────────────────────────────────────────────────────────
# 2 = un token per hora; 1 = un token per mitja hora (~4x més lent i més memòria).
# Es pot canviar sense editar:  TRANSFORMER_PATCH_SIZE=1 TRANSFORMER_BATCH_SIZE=32 python -m ...
PATCH_SIZE   = int(os.environ.get("TRANSFORMER_PATCH_SIZE", "2"))
D_MODEL      = 256       # 16 canals d'entrada sense sobreajust: hi ha marge per a més capacitat
N_HEADS      = 8         # 32 dimensions per cap
N_ENC_LAYERS = 4
N_DEC_LAYERS = 4
D_FF         = 1024      # 4 × D_MODEL
DROPOUT      = 0.1
MAX_SEQ_LEN  = 2048
PROFILE_DIM  = 10        # 9 indicadors de consum + sensibilitat a la temperatura

# ── Entrenament ───────────────────────────────────────────────────────────────
BATCH_SIZE      = int(os.environ.get("TRANSFORMER_BATCH_SIZE", "64"))
N_EPOCHS        = 80
LR              = 1e-4
LR_WARMUP_STEPS = 1000
WEIGHT_DECAY    = 1e-5
GRAD_CLIP       = 1.0
PATIENCE        = 10
VAL_SPLIT       = 0.03   # validació proxy (màscares sintètiques), només per monitorar
# Processos que preparen batches en paral·lel (0 = cap). A Linux comparteixen
# memòria; a Windows cadascun copia les dades (~1 GB), millor TRANSFORMER_WORKERS=0.
NUM_WORKERS     = int(os.environ.get("TRANSFORMER_WORKERS", "4"))

# Pes de cada corba a la pèrdua = (escala_corba / escala_mitjana) ** LOSS_SCALE_POWER
# 1.0 → la pèrdua equival a l'MAE en watts (el que puntua el concurs)
# 0.0 → totes les corbes pesen igual
LOSS_SCALE_POWER = 1.0

# ── Ensemble ──────────────────────────────────────────────────────────────────
N_SNAPSHOTS = 3          # es guarden els N millors checkpoints i es promitgen

# ── Mode de prova ─────────────────────────────────────────────────────────────
# S'activa sense editar res:  TRANSFORMER_DEBUG=1 python -m transformer.main_train
DEBUG          = os.environ.get("TRANSFORMER_DEBUG", "0") == "1"
DEBUG_N_CURVES = 300
DEBUG_N_EPOCHS = 2
DEBUG_N_VAL    = 100

# ── Reproducibilitat ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
