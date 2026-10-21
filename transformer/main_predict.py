"""
transformer/main_predict.py
---------------------------
Genera la submissió combinant tots els entrenaments disponibles.

  1. Troba els entrenaments a transformer/checkpoints/seed*/ (o els de --runs)
  2. Fa la mitjana de les prediccions de tots els seus millors checkpoints
  3. Ajusta la barreja amb interpolació (forats d'1-2 punts) sobre les 999
     corbes holed_ etiquetades de train, i n'informa l'MAE
  4. Aplica el mateix a X_test i desa la submissió

Execució:
    python -m transformer.main_predict                       # tots els seed*/
    python -m transformer.main_predict --runs seed42 seed43  # només aquests

Genera:
    outputs/transformer/y_pred_test.csv   ← fitxer a penjar
    outputs/transformer/y_pred_train.csv  ← prediccions de validació (per analitzar)
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from transformer.core.config import RUNS_DIR, OUT_DIR, PRED_TEST_PATH, PRED_VAL_PATH
from transformer.core.dataset import InferenceDataset
from transformer.core.pipeline import (
    load_data, labeled_validation, build_pool, holed_context, gap_lengths,
    linear_interp, to_watts, tune_blend, apply_blend, breakdown, print_breakdown,
)
from transformer.core.trainer import make_loader, predict_ensemble
from src.data_loader import save_submission


def collect_specs(run_names: list[str] | None) -> list:
    runs = [RUNS_DIR / r for r in run_names] if run_names else sorted(RUNS_DIR.glob("seed*"))
    specs = []
    for run in runs:
        meta_path, snap_path = run / "meta.json", run / "snapshots.json"
        if not (meta_path.exists() and snap_path.exists()):
            print(f"  (s'ignora {run.name}: falta meta.json o snapshots.json)")
            continue
        kwargs = json.loads(meta_path.read_text(encoding="utf-8"))["model_kwargs"]
        snaps = json.loads(snap_path.read_text(encoding="utf-8"))
        specs += [(kwargs, run / s["file"]) for s in snaps if (run / s["file"]).exists()]
        print(f"  {run.name}: {len(snaps)} checkpoints "
              f"(MAE val: {[round(s['mae'], 2) for s in snaps]})")
    if not specs:
        raise FileNotFoundError(
            f"No hi ha cap entrenament a {RUNS_DIR}. Executa: python -m transformer.main_train")
    return specs


def run_holed(raw: np.ndarray, pool: dict, specs: list, device: torch.device):
    ctx = holed_context(raw, pool, device)
    ds = InferenceDataset(ctx["norm"], ctx["mask"], ctx["profiles"],
                          ctx["nb_norm"], ctx["nb_anom"], pool["population"])
    pred = to_watts(predict_ensemble(specs, make_loader(ds), device), ctx, raw)
    return pred, gap_lengths(ctx["mask"]), linear_interp(raw)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("\n── Entrenaments trobats ──")
    specs = collect_specs(args.runs)
    print(f"Total: {len(specs)} checkpoints a promitjar")

    data = load_data()
    pool = build_pool(data["refs_raw"])

    # ── Validació: ajustar la barreja i informar de l'MAE ────────────────────
    val = labeled_validation(data["X_holed_train"], data["y_train"])
    pred_v, runlen_v, interp_v = run_holed(val["raw"], pool, specs, device)
    blend_w = tune_blend(pred_v, interp_v, runlen_v, val["y_raw"])
    final_v = apply_blend(pred_v, interp_v, runlen_v, blend_w)
    print_breakdown({
        "interpolació lineal (benchmark)": breakdown(interp_v, val["y_raw"], runlen_v),
        f"ensemble ({len(specs)} checkpoints)": breakdown(pred_v, val["y_raw"], runlen_v),
        "  + barreja forats curts": breakdown(final_v, val["y_raw"], runlen_v),
    })
    print(f"Pesos del model a la barreja: {blend_w}")

    # ── Test ─────────────────────────────────────────────────────────────────
    X_test = data["X_holed_test"]
    raw_t = X_test.to_numpy(dtype=np.float32).T
    pred_t, runlen_t, interp_t = run_holed(raw_t, pool, specs, device)
    final_t = apply_blend(pred_t, interp_t, runlen_t, blend_w)
    final_t[~np.isnan(raw_t)] = raw_t[~np.isnan(raw_t)]
    print(f"\nForats imputats: {int(np.isnan(raw_t).sum()):,} | NaN a la sortida: "
          f"{int(np.isnan(final_t).sum())}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamps = X_test.index.strftime("%Y-%m-%d %H:%M:%S").values
    save_submission(pd.DataFrame(final_t.T, columns=X_test.columns), PRED_TEST_PATH, timestamps)
    pd.DataFrame(final_v.T, index=data["X_holed_train"].index,
                 columns=val["cols"]).to_csv(PRED_VAL_PATH)


if __name__ == "__main__":
    main()
