"""
lstm/main_predict_ensemble.py
-----------------------------
Ensemble de dos models: CPU (D_MODEL=192) + GPU (D_MODEL=256).

Combina les prediccions dels dos models amb un pes configurable.
Els models falla en casos diferents → la combinació redueix l'error global.

Prerequisits:
    lstm/checkpoints/best_model_combined.pt   ← model CPU (D_MODEL=192)
    lstm/checkpoints/best_model_gpu_256.pt    ← model GPU (D_MODEL=256)

Execució:
    cd data_challenge
    python -m lstm.main_predict_ensemble

Genera:
    outputs/lstm/y_pred_test_ensemble.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lstm.src.dataset import InferenceDataset
from lstm.src.trainer import predict_batch
from src.data_loader  import load_test, save_submission

# ── Configuració ensemble ─────────────────────────────────────────────────────
# Pes del model GPU (256) vs CPU (192)
# 0.6/0.4 → confiem més en el model gran però deixem contribuir l'altre
WEIGHT_GPU = 0.6
WEIGHT_CPU = 1 - WEIGHT_GPU

CKPT_DIR     = Path(__file__).resolve().parents[1] / "lstm" / "checkpoints"
OUT_DIR      = Path(__file__).resolve().parents[1] / "outputs" / "lstm"
X_TEST_PATH  = Path(__file__).resolve().parents[1] / "data" / "X_test.csv"
PRED_PATH    = OUT_DIR / "y_pred_test_ensemble.csv"

GPU_CKPT = CKPT_DIR / "best_model_gpu_256.pt"
CPU_CKPT = CKPT_DIR / "best_model_combined.pt"


def normalize_holed(X_holed: pd.DataFrame) -> tuple:
    values    = X_holed.values.T.astype(np.float32)
    N, T      = values.shape
    nan_masks = np.isnan(values)
    norm      = np.zeros_like(values)
    mus       = np.zeros(N, dtype=np.float32)
    sigmas    = np.ones(N,  dtype=np.float32)

    for i in range(N):
        mu, sig   = np.nanmean(values[i]), np.nanstd(values[i])
        sig       = sig if sig > 1e-6 else 1.0
        mus[i]    = mu
        sigmas[i] = sig
        norm[i]   = np.nan_to_num((values[i] - mu) / sig, nan=0.0)

    return norm, nan_masks, mus, sigmas


def load_model(ckpt_path: Path, d_model: int, n_heads: int,
               n_enc: int, n_dec: int, d_ff: int,
               device: torch.device):
    """Carrega un model amb arquitectura específica."""
    from lstm.src.config import DROPOUT, MAX_SEQ_LEN, SEQ_LEN
    from lstm.src.model  import TemporalMaskedAutoencoder, PATCH_SIZE

    # Instanciar directament amb els paràmetres correctes
    model = TemporalMaskedAutoencoder(
        seq_len   = SEQ_LEN,
        patch_size = PATCH_SIZE,
        d_model   = d_model,
        n_heads   = n_heads,
        n_enc     = n_enc,
        n_dec     = n_dec,
        d_ff      = d_ff,
        dropout   = DROPOUT,
    ).to(device)

    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Paràmetres: {n_params:,}")
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Verificar checkpoints
    for ckpt, name in [(GPU_CKPT, "GPU 256"), (CPU_CKPT, "CPU 192")]:
        if not ckpt.exists():
            raise FileNotFoundError(
                f"No s'ha trobat el model {name}: {ckpt}\n"
                f"Assegura't que el fitxer és a lstm/checkpoints/")

    # ── Carregar dades ────────────────────────────────────────────────────────
    print("\n── Carregant dades de test ──")
    _, X_holed = load_test(X_TEST_PATH)
    holed_norm, nan_masks, mus, sigmas = normalize_holed(X_holed)

    inf_loader = DataLoader(
        InferenceDataset(holed_norm, nan_masks),
        batch_size=64, shuffle=False, num_workers=0)

    # ── Inferència model GPU (D_MODEL=256, 6 capes) ───────────────────────────
    print(f"\n── Model GPU (D_MODEL=256, 6+6 capes) — pes: {WEIGHT_GPU} ──")
    model_gpu = load_model(GPU_CKPT,
                           d_model=256, n_heads=8,
                           n_enc=6, n_dec=6, d_ff=1024,
                           device=device)
    preds_gpu = predict_batch(model_gpu, inf_loader, device)
    del model_gpu
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print(f"  Prediccions shape: {preds_gpu.shape}")

    # ── Inferència model CPU (D_MODEL=192, 4 capes) ───────────────────────────
    print(f"\n── Model CPU (D_MODEL=192, 4+4 capes) — pes: {WEIGHT_CPU} ──")
    model_cpu = load_model(CPU_CKPT,
                           d_model=192, n_heads=8,
                           n_enc=4, n_dec=4, d_ff=768,
                           device=device)
    preds_cpu = predict_batch(model_cpu, inf_loader, device)
    del model_cpu
    print(f"  Prediccions shape: {preds_cpu.shape}")

    # ── Combinar prediccions ──────────────────────────────────────────────────
    print(f"\n── Ensemble ({WEIGHT_GPU}×GPU + {WEIGHT_CPU}×CPU) ──")
    preds_ensemble = WEIGHT_GPU * preds_gpu + WEIGHT_CPU * preds_cpu

    # ── Desnormalitzar ────────────────────────────────────────────────────────
    N, T      = preds_ensemble.shape
    preds_raw = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        preds_raw[i] = preds_ensemble[i] * sigmas[i] + mus[i]

    # Clippejar negatius
    preds_raw = np.clip(preds_raw, 0, None)

    # Preservar valors coneguts originals
    holed_original = X_holed.values.T.astype(np.float32)
    known_mask     = ~np.isnan(holed_original)
    preds_raw[known_mask] = holed_original[known_mask]

    print(f"Forats imputats   : {nan_masks.sum():,}")
    print(f"Valors preservats : {known_mask.sum():,}")

    # ── Guardar ───────────────────────────────────────────────────────────────
    y_pred = pd.DataFrame(preds_raw.T, columns=X_holed.columns)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamps = X_holed.index.strftime("%Y-%m-%d %H:%M:%S").values
    save_submission(y_pred, PRED_PATH, timestamps)
    print(f"\nEnsemble guardat a: {PRED_PATH}")
    print(f"Pesos usats: GPU×{WEIGHT_GPU} + CPU×{WEIGHT_CPU}")


if __name__ == "__main__":
    main()
