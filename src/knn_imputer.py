"""
knn_imputer.py
--------------
Lògica principal del KNN per a imputació de sèries temporals.

Algorisme
---------
Per a cada corba holed_i:
  1. Trobar els timesteps vàlids (no-NaN) de holed_i.
  2. Calcular la distància entre holed_i i TOTES les corbes de referència,
     usant només els timesteps vàlids compartits.
  3. Seleccionar els K veïns més propers.
  4. Omplir cada forat amb la mitjana ponderada (1/distància) dels K veïns.

Nota sobre rendiment
--------------------
X_ref pot tenir 20.000-37.000 columnes × 1.057 timesteps.
Treballem amb arrays NumPy transposats (n_curves × T) per vectoritzar
el càlcul de distàncies i evitar loops Python per fila.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    K_NEIGHBORS, MIN_OVERLAP, DISTANCE_METRIC,
    WEIGHTING, GAUSSIAN_SIGMA,
)
from src.preprocessor import normalize


def _euclidean_distances(ref_norm: np.ndarray,
                         target: np.ndarray,
                         valid_mask: np.ndarray) -> np.ndarray:
    if valid_mask.sum() < 1:
        return np.full(ref_norm.shape[0], np.inf)
    diff = ref_norm[:, valid_mask] - target[valid_mask]
    return np.sqrt((diff ** 2).sum(axis=1))


def _manhattan_distances(ref_norm: np.ndarray,
                         target: np.ndarray,
                         valid_mask: np.ndarray) -> np.ndarray:
    """L1: menys sensible a outliers que L2."""
    if valid_mask.sum() < 1:
        return np.full(ref_norm.shape[0], np.inf)
    diff = ref_norm[:, valid_mask] - target[valid_mask]
    return np.abs(diff).sum(axis=1)


def _chebyshev_distances(ref_norm: np.ndarray,
                         target: np.ndarray,
                         valid_mask: np.ndarray) -> np.ndarray:
    """L∞: distància del pitjor timestep. Molt sensible a pics."""
    if valid_mask.sum() < 1:
        return np.full(ref_norm.shape[0], np.inf)
    diff = ref_norm[:, valid_mask] - target[valid_mask]
    return np.abs(diff).max(axis=1)


def _cosine_distances(ref_norm: np.ndarray,
                      target: np.ndarray,
                      valid_mask: np.ndarray) -> np.ndarray:
    if valid_mask.sum() < 1:
        return np.full(ref_norm.shape[0], np.inf)
    ref_v    = ref_norm[:, valid_mask]
    tgt_v    = target[valid_mask]
    dot      = ref_v @ tgt_v
    norm_ref = np.linalg.norm(ref_v, axis=1)
    norm_tgt = np.linalg.norm(tgt_v)
    denom    = norm_ref * norm_tgt
    denom[denom == 0] = 1e-10
    return 1.0 - dot / denom


def _get_distances(ref_norm: np.ndarray,
                   target: np.ndarray,
                   valid_mask: np.ndarray,
                   metric: str = DISTANCE_METRIC) -> np.ndarray:
    """Dispatcher de mètriques de distància."""
    if valid_mask.sum() < MIN_OVERLAP:
        return np.full(ref_norm.shape[0], np.inf)
    if metric == "euclidean":
        return _euclidean_distances(ref_norm, target, valid_mask)
    elif metric == "manhattan":
        return _manhattan_distances(ref_norm, target, valid_mask)
    elif metric == "chebyshev":
        return _chebyshev_distances(ref_norm, target, valid_mask)
    elif metric == "cosine":
        return _cosine_distances(ref_norm, target, valid_mask)
    else:
        raise ValueError(f"Mètrica desconeguda: '{metric}'. "
                         f"Usa 'euclidean', 'manhattan', 'chebyshev' o 'cosine'.")


def _compute_weights(distances: np.ndarray,
                     weighting: str = WEIGHTING,
                     gaussian_sigma: float = GAUSSIAN_SIGMA) -> np.ndarray:
    """
    Calcula els pesos dels K veïns a partir de les seves distàncies.

    Mètodes
    -------
    inverse   : pes = 1/dist
    uniform   : tots els veïns pesen igual
    softmax   : pes = softmax(-dist), versió suavitzada d'inverse
    rank      : pes proporcional al rang invers, robust a escales
    gaussian  : pes = exp(-dist²/(2σ²)), decau suaument
    """
    k = len(distances)
    zero_mask = distances == 0
    if zero_mask.any():
        w = np.zeros(k)
        w[zero_mask] = 1.0 / zero_mask.sum()
        return w

    if weighting == "uniform":
        return np.ones(k) / k
    elif weighting == "inverse":
        w = 1.0 / distances
        return w / w.sum()
    elif weighting == "softmax":
        logits = -distances
        logits -= logits.max()   # estabilitat numèrica
        w = np.exp(logits)
        return w / w.sum()
    elif weighting == "rank":
        order = np.argsort(distances)
        ranks = np.empty(k)
        ranks[order] = np.arange(k, 0, -1, dtype=float)
        return ranks / ranks.sum()
    elif weighting == "gaussian":
        w = np.exp(-(distances ** 2) / (2 * gaussian_sigma ** 2))
        return w / w.sum() if w.sum() > 0 else np.ones(k) / k
    else:
        raise ValueError(f"Weighting desconegut: '{weighting}'. "
                         f"Usa 'inverse', 'uniform', 'softmax', 'rank' o 'gaussian'.")


def impute_column(holed_series: pd.Series,
                  X_ref_norm: np.ndarray,
                  X_ref_values: np.ndarray,
                  k: int = K_NEIGHBORS,
                  metric: str = DISTANCE_METRIC,
                  weighting: str = WEIGHTING,
                  gaussian_sigma: float = GAUSSIAN_SIGMA) -> pd.Series:
    """
    Imputa una sola columna holed_ usant KNN.
    """
    result      = holed_series.copy()
    nan_mask    = holed_series.isna().values
    valid_mask  = ~nan_mask

    if not nan_mask.any():
        return result

    target = holed_series.values.copy().astype(float)
    mu  = np.nanmean(target)
    sig = np.nanstd(target)
    if sig == 0:
        sig = 1.0
    target_norm = (target - mu) / sig

    distances = _get_distances(X_ref_norm, target_norm, valid_mask, metric)

    finite_mask = np.isfinite(distances)
    k_use = min(k, finite_mask.sum())

    if k_use == 0:
        result = result.interpolate(method="linear", limit_direction="both")
        return result

    top_k_idx  = np.argpartition(distances, k_use)[:k_use]
    top_k_dist = distances[top_k_idx]
    weights    = _compute_weights(top_k_dist, weighting, gaussian_sigma)

    for t in np.where(nan_mask)[0]:
        result.iloc[t] = float(np.dot(weights, X_ref_values[top_k_idx, t]))

    return result


def impute_all(X_holed: pd.DataFrame,
               X_ref: pd.DataFrame,
               k: int = K_NEIGHBORS,
               metric: str = DISTANCE_METRIC,
               weighting: str = WEIGHTING,
               gaussian_sigma: float = GAUSSIAN_SIGMA) -> pd.DataFrame:
    """
    Imputa totes les columnes holed_ de X_holed usant KNN sobre X_ref.
    """
    print(f"[knn_imputer] Normalitzant {X_ref.shape[1]:,} corbes de referència...")
    X_ref_norm_df = normalize(X_ref)
    X_ref_norm    = X_ref_norm_df.values.T.astype(float)
    X_ref_values  = X_ref.values.T.astype(float)

    print(f"[knn_imputer] Imputant {X_holed.shape[1]:,} corbes  "
          f"K={k} | metric='{metric}' | weighting='{weighting}'...")

    results = {}
    for col in tqdm(X_holed.columns, desc="KNN imputation"):
        results[col] = impute_column(
            holed_series  = X_holed[col],
            X_ref_norm    = X_ref_norm,
            X_ref_values  = X_ref_values,
            k             = k,
            metric        = metric,
            weighting     = weighting,
            gaussian_sigma= gaussian_sigma,
        )

    y_pred = pd.DataFrame(results, index=X_holed.index)
    print(f"[knn_imputer] Fet. NaN restants: {y_pred.isna().sum().sum()}")
    return y_pred
