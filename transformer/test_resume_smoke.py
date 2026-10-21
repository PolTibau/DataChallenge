"""Smoke test: pipeline amb dades sintètiques, 2 èpoques, reprendre fins a 4, ensemble."""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformer.core.config import SEQ_LEN
from transformer.core.dataset import TimeSeriesDataset, InferenceDataset, fit_mask_distribution
from transformer.core.model import default_model_kwargs
from transformer.core.pipeline import (
    build_pool, pool_neighbour_channels, holed_context, to_watts, gap_lengths,
    linear_interp, tune_blend, apply_blend,
)
from transformer.core.trainer import (
    train, build_fixed_masks, load_training_checkpoint, make_loader, predict_ensemble,
)


def synthetic_curves(n: int, rng) -> np.ndarray:
    t = np.arange(SEQ_LEN)
    daily = 1 + 0.6 * np.sin(2 * np.pi * (t % 48) / 48)
    level = rng.lognormal(5, 1, size=(n, 1))
    return (level * daily * rng.gamma(2, 0.5, size=(n, SEQ_LEN))).astype(np.float32)


def main() -> None:
    device = torch.device("cpu")
    rng = np.random.default_rng(0)
    refs = synthetic_curves(120, rng)
    holed = synthetic_curves(20, rng)
    truth = holed.copy()
    holed[rng.random(holed.shape) < 0.12] = np.nan
    holed[:, 480:528] = np.nan                      # un dia sencer

    pool = build_pool(refs)
    nb_norm, nb_anom = pool_neighbour_channels(pool, device)
    mask_dist = fit_mask_distribution(holed)
    ctx = holed_context(holed, pool, device)
    real_val = {"dataset": InferenceDataset(ctx["norm"], ctx["mask"], ctx["profiles"],
                                            ctx["nb_norm"], ctx["nb_anom"], pool["population"]),
                "meds": ctx["meds"], "mads": ctx["mads"], "nanmask": ctx["mask"], "y_raw": truth}
    common = dict(curves_norm=pool["norm"], profiles=pool["profiles"],
                  weights=pool["mads"] / pool["mads"].mean(), nb_norm=nb_norm, nb_anom=nb_anom,
                  population=pool["population"], mask_dist=mask_dist)
    train_ds = TimeSeriesDataset(indices=np.arange(100), **common)
    val_ds = TimeSeriesDataset(indices=np.arange(100, 120), augment=False,
                               fixed_masks=build_fixed_masks(20, SEQ_LEN, mask_dist, 1), **common)
    sample = train_ds[0]
    print("input", tuple(sample["input"].shape), "profile", tuple(sample["profile"].shape))

    kwargs = default_model_kwargs()
    kwargs.update(d_model=32, n_heads=4, n_enc=1, n_dec=1, d_ff=64)
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp)
        _, snaps = train(train_ds, val_ds, real_val, device, run, kwargs, n_epochs=2)
        ck = load_training_checkpoint(run / "latest.pt", device)
        assert ck["epoch"] == 2 and (run / "best_model.pt").exists()
        _, snaps = train(train_ds, val_ds, real_val, device, run, kwargs, n_epochs=4,
                         resume_from=run / "latest.pt")
        ck = load_training_checkpoint(run / "latest.pt", device)
        assert ck["epoch"] == 4, ck["epoch"]
        assert 1 <= len(snaps) <= 3 and all((run / s["file"]).exists() for s in snaps)

        loader = make_loader(real_val["dataset"])
        pred = to_watts(predict_ensemble([(kwargs, run / s["file"]) for s in snaps],
                                         loader, device), ctx, holed)
        runlen, interp = gap_lengths(ctx["mask"]), linear_interp(holed)
        w = tune_blend(pred, interp, runlen, truth)
        out = apply_blend(pred, interp, runlen, w)
        assert out.shape == holed.shape and not np.isnan(out).any()
        print("blend weights", w)
    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
