"""
lstm/main_train_combined.py — v4
--------------------------------
Entrena el Transformer v4 sobre les referències de train + test.

Execució (des de l'arrel del repositori):
    python -m lstm.main_train_combined                 # llavor per defecte (42)
    python -m lstm.main_train_combined --seed 43       # un altre entrenament per a l'ensemble
    python -m lstm.main_train_combined --resume        # reprèn lstm/checkpoints/v4/seed42/latest.pt

Cada entrenament es desa a lstm/checkpoints/v4/seed<N>/ (best_model.pt,
snap_eXXX.pt, meta.json). La predicció final (main_predict) combina tots els
entrenaments que trobi en aquesta carpeta.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from lstm.src.config import (
    RUNS_DIR, OUT_DIR, SEQ_LEN, VAL_SPLIT, RANDOM_SEED, LOSS_SCALE_POWER,
    N_NEIGHBOURS, DEBUG, DEBUG_N_CURVES, DEBUG_N_EPOCHS, DEBUG_N_VAL,
)
from lstm.src.dataset import (
    TimeSeriesDataset, InferenceDataset, fit_mask_distribution,
)
from lstm.src.model import default_model_kwargs
from lstm.src.pipeline import (
    load_data, labeled_validation, build_pool, pool_neighbour_channels,
    holed_context, gap_lengths, linear_interp, to_watts, tune_blend,
    apply_blend, breakdown, print_breakdown,
)
from lstm.src.trainer import (
    train, build_fixed_masks, make_loader, predict, predict_ensemble, _write_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--resume", action="store_true",
                        help="Reprèn des de latest.pt del mateix --seed")
    args = parser.parse_args()
    seed = args.seed

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if DEBUG:
        print("=" * 50 + "\n  MODE DEBUG\n" + "=" * 50)

    # ── 1. Dades i pool de referència ────────────────────────────────────────
    data = load_data()
    pool = build_pool(data["refs_raw"])
    print("[train] Buscant consumidors semblants per a cada referència...")
    nb_norm, nb_anom = pool_neighbour_channels(pool, device)
    weights = (pool["mads"] / pool["mads"].mean()) ** LOSS_SCALE_POWER
    print(f"[train] Pes de les corbes a la pèrdua (p50/p90/max): "
          f"{np.round(np.percentile(weights, [50, 90, 100]), 2)}")

    # ── 2. Validació real: totes les holed_ de train etiquetades ─────────────
    val = labeled_validation(data["X_holed_train"], data["y_train"])
    if DEBUG:
        val = {k: v[:DEBUG_N_VAL] for k, v in val.items()}
    ctx = holed_context(val["raw"], pool, device)
    real_val = {
        "dataset": InferenceDataset(ctx["norm"], ctx["mask"], ctx["profiles"],
                                    ctx["nb_norm"], ctx["nb_anom"], pool["population"]),
        "meds": ctx["meds"], "mads": ctx["mads"],
        "nanmask": ctx["mask"], "y_raw": val["y_raw"],
    }

    # ── 3. Distribució de forats (train + test holed_) ───────────────────────
    holed_all = np.concatenate([data["X_holed_train"].to_numpy(np.float32).T,
                                data["X_holed_test"].to_numpy(np.float32).T])
    mask_dist = fit_mask_distribution(holed_all)

    # ── 4. Datasets ──────────────────────────────────────────────────────────
    n = len(pool["norm"])
    perm = np.random.default_rng(seed).permutation(n)
    n_proxy = max(1, int(n * VAL_SPLIT))
    proxy_idx, train_idx = perm[:n_proxy], perm[n_proxy:]
    if DEBUG:
        train_idx, proxy_idx = train_idx[:DEBUG_N_CURVES], proxy_idx[:64]
    common = dict(curves_norm=pool["norm"], profiles=pool["profiles"], weights=weights,
                  nb_norm=nb_norm, nb_anom=nb_anom, population=pool["population"],
                  mask_dist=mask_dist)
    train_ds = TimeSeriesDataset(indices=train_idx, augment=True, seed=seed, **common)
    val_ds = TimeSeriesDataset(
        indices=proxy_idx, augment=False, seed=seed + 1,
        fixed_masks=build_fixed_masks(len(proxy_idx), SEQ_LEN, mask_dist, seed + 7), **common)

    # ── 5. Entrenament ───────────────────────────────────────────────────────
    model_kwargs = default_model_kwargs()
    run_name = f"seed{seed}" + (f"_p{model_kwargs['patch_size']}"
                                if model_kwargs["patch_size"] != 2 else "")
    run_dir = RUNS_DIR / (f"debug_{run_name}" if DEBUG else run_name)
    meta = {"seed": seed, "model_kwargs": model_kwargs,
            "loss_scale_power": LOSS_SCALE_POWER, "n_neighbours": N_NEIGHBOURS}
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "meta.json", meta)

    model, snapshots = train(
        train_ds, val_ds, real_val, device, run_dir, model_kwargs,
        n_epochs=DEBUG_N_EPOCHS if DEBUG else None,
        resume_from=(run_dir / "latest.pt") if args.resume else None, seed=seed)

    # ── 6. Avaluació final sobre la validació real ───────────────────────────
    loader = make_loader(real_val["dataset"])
    raw, y = val["raw"], val["y_raw"]
    runlen = gap_lengths(ctx["mask"])
    interp = linear_interp(raw)
    best   = to_watts(predict(model, loader, device), ctx, raw)
    specs  = [(model_kwargs, run_dir / s["file"]) for s in snapshots]
    ens    = to_watts(predict_ensemble(specs, loader, device), ctx, raw)
    blend_w = tune_blend(ens, interp, runlen, y)
    final  = apply_blend(ens, interp, runlen, blend_w)

    rows = {
        "interpolació lineal (benchmark)": breakdown(interp, y, runlen),
        "millor checkpoint": breakdown(best, y, runlen),
        f"mitjana de {len(specs)} checkpoints": breakdown(ens, y, runlen),
        "  + barreja forats curts": breakdown(final, y, runlen),
    }
    print_breakdown(rows)
    print(f"Pesos del model a la barreja (forats d'1 i 2 punts): {blend_w}")

    meta.update({"snapshots": snapshots, "blend_weights": blend_w,
                 "val_breakdown": rows, "n_val_curves": len(val["cols"])})
    _write_json(run_dir / "meta.json", meta)

    if not DEBUG:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        idx = data["X_holed_train"].index
        pd.DataFrame(final.T, index=idx, columns=val["cols"]).to_csv(
            OUT_DIR / f"y_pred_train_v4_seed{seed}.csv")
    print(f"\nFet. Per generar la submissió: python -m lstm.main_predict")


if __name__ == "__main__":
    main()
