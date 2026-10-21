"""
transformer/core/pipeline.py
----------------------------
Preparació de dades i postprocés compartits entre entrenament i predicció,
perquè tots dos construeixin exactament el mateix context.
"""

import numpy as np
import pandas as pd
import torch

from transformer.core.config import X_TRAIN_PATH, Y_TRAIN_PATH, X_TEST_PATH
from transformer.core.context import (
    build_population, typical_profiles, expand_profiles, descriptors,
    find_neighbours, neighbour_channels,
)
from transformer.core.dataset import (
    robust_stats, profile_features, fit_profile_stats, normalize_profiles,
)
from src.data_loader import load_train, load_test


# ── Càrrega ───────────────────────────────────────────────────────────────────

def load_data() -> dict:
    """Referències de train + test (netes) i corbes holed_ de train i test."""
    X_ref_train, X_holed_train, y_train = load_train(X_TRAIN_PATH, Y_TRAIN_PATH)
    X_ref_test, X_holed_test = load_test(X_TEST_PATH)
    refs = np.concatenate([X_ref_train.to_numpy(dtype=np.float32).T,
                           X_ref_test.to_numpy(dtype=np.float32).T])
    del X_ref_train, X_ref_test
    bad = np.isnan(refs).any(axis=1) | (refs.std(axis=1) < 1e-6)
    print(f"[pipeline] Referències: {len(refs):,} ({bad.sum()} descartades: NaN o planes)")
    return {"refs_raw": np.ascontiguousarray(refs[~bad]),
            "X_holed_train": X_holed_train, "y_train": y_train,
            "X_holed_test": X_holed_test}


def labeled_validation(X_holed_train: pd.DataFrame, y_train: pd.DataFrame) -> dict:
    """
    Totes les corbes holed_ de train amb ground truth complet (999; holed_192
    queda fora perquè el seu y_train és gairebé tot NaN).
    """
    cols = [c for c in X_holed_train.columns if not y_train[c].isna().any()]
    skipped = [c for c in X_holed_train.columns if c not in cols]
    if skipped:
        print(f"[pipeline] Validació: excloses per y_train incomplet: {skipped}")
    return {"cols": cols,
            "raw": X_holed_train[cols].to_numpy(dtype=np.float32).T,
            "y_raw": y_train[cols].to_numpy(dtype=np.float32).T}


# ── Context ───────────────────────────────────────────────────────────────────

def robust_normalize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, T) amb possibles NaN → (norm amb NaN→0, medianes, MADs)."""
    n = len(values)
    meds = np.zeros(n, dtype=np.float32)
    mads = np.ones(n, dtype=np.float32)
    for i in range(n):
        meds[i], mads[i] = robust_stats(values[i])
    norm = (values - meds[:, None]) / mads[:, None]
    return np.nan_to_num(norm, nan=0.0).astype(np.float32), meds, mads


def build_pool(refs_raw: np.ndarray) -> dict:
    """
    Tot el que es calcula a partir de les referències completes: població,
    estadístiques de la fitxa, curves normalitzades, anomalies respecte al
    perfil habitual de cada corba i descriptors per buscar semblants.
    """
    print("[pipeline] Construint el pool de referència...")
    population = build_population(refs_raw)
    prof_raw   = profile_features(refs_raw, population["daily"])
    prof_stats = fit_profile_stats(prof_raw)
    norm, meds, mads = robust_normalize(refs_raw)
    typ  = expand_profiles(typical_profiles(refs_raw))
    anom = ((refs_raw - typ) / mads[:, None]).astype(np.float32)
    del typ
    print(f"[pipeline] Sensibilitat a la temperatura (p10/p50/p90): "
          f"{np.round(np.nanpercentile(prof_raw[:, 9], [10, 50, 90]), 2)}")
    return {
        "norm": norm, "meds": meds, "mads": mads, "anom": anom,
        "desc": descriptors(refs_raw),
        "profiles": normalize_profiles(prof_raw, prof_stats),
        "profile_stats": prof_stats,
        "population": population,
    }


def pool_neighbour_channels(pool: dict, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Canals de semblants per a les pròpies referències (sense comptar-se a si mateixes)."""
    nb = find_neighbours(pool["desc"], pool["desc"], exclude_self=True, device=device)
    return neighbour_channels(nb, pool["norm"], pool["anom"], device=device)


def holed_context(raw: np.ndarray, pool: dict, device: torch.device) -> dict:
    """Context per a corbes amb forats reals (només es fan servir els punts coneguts)."""
    norm, meds, mads = robust_normalize(raw)
    prof_raw = profile_features(raw, pool["population"]["daily"])
    nb = find_neighbours(descriptors(raw), pool["desc"], device=device)
    nb_norm, nb_anom = neighbour_channels(nb, pool["norm"], pool["anom"], device=device)
    return {"norm": norm, "mask": np.isnan(raw), "meds": meds, "mads": mads,
            "profiles": normalize_profiles(prof_raw, pool["profile_stats"]),
            "nb_norm": nb_norm, "nb_anom": nb_anom}


# ── Postprocés ────────────────────────────────────────────────────────────────

def gap_lengths(mask: np.ndarray) -> np.ndarray:
    """(N, T) bool → longitud del forat al qual pertany cada punt (0 si és conegut)."""
    out = np.zeros(mask.shape, dtype=np.int32)
    for i, m in enumerate(mask):
        d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
        for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
            out[i, s:e] = e - s
    return out


def linear_interp(raw: np.ndarray) -> np.ndarray:
    """Interpolació lineal (extrem: valor constant), com el benchmark del concurs."""
    out = np.zeros_like(raw)
    t = np.arange(raw.shape[1])
    for i, v in enumerate(raw):
        obs = ~np.isnan(v)
        if obs.any():
            out[i] = np.interp(t, t[obs], v[obs])
    return out


def to_watts(preds_norm: np.ndarray, ctx: dict, raw: np.ndarray) -> np.ndarray:
    """Desnormalitza, talla a ≥0 i conserva els valors coneguts."""
    out = np.clip(preds_norm * ctx["mads"][:, None] + ctx["meds"][:, None], 0, None)
    known = ~np.isnan(raw)
    out[known] = raw[known]
    return out


BLEND_LENGTHS = (1, 2)


def apply_blend(pred: np.ndarray, interp: np.ndarray, runlen: np.ndarray,
                weights: dict) -> np.ndarray:
    """Als forats de longitud L: w_L × model + (1 − w_L) × interpolació."""
    out = pred.copy()
    for L, w in weights.items():
        sel = runlen == int(L)
        out[sel] = w * pred[sel] + (1 - w) * interp[sel]
    return out


def tune_blend(pred: np.ndarray, interp: np.ndarray, runlen: np.ndarray,
               y: np.ndarray) -> dict:
    """Pes del model als forats d'1 i 2 punts que minimitza l'MAE de validació."""
    valid = ~np.isnan(y)
    weights = {}
    for L in BLEND_LENGTHS:
        sel = (runlen == L) & valid
        if not sel.any():
            weights[L] = 1.0
            continue
        grid = np.linspace(0, 1, 21)
        errs = [np.abs(w * pred[sel] + (1 - w) * interp[sel] - y[sel]).mean() for w in grid]
        weights[L] = float(grid[int(np.argmin(errs))])
    return weights


def breakdown(pred: np.ndarray, y: np.ndarray, runlen: np.ndarray) -> dict:
    """MAE total i per tipus de forat."""
    valid = (runlen > 0) & ~np.isnan(y)
    err = np.abs(pred - y)
    cats = {"total": valid, "1-2": valid & (runlen <= 2),
            "3-7": valid & (runlen >= 3) & (runlen <= 7), ">=8": valid & (runlen >= 8)}
    return {k: float(err[m].mean()) if m.any() else float("nan") for k, m in cats.items()}


def print_breakdown(rows: dict) -> None:
    print(f"\n{'':32s} {'total':>8} {'1-2':>8} {'3-7':>8} {'>=8':>8}")
    for name, b in rows.items():
        print(f"{name:32s} {b['total']:8.2f} {b['1-2']:8.2f} {b['3-7']:8.2f} {b['>=8']:8.2f}")
