"""
transformer/core/dataset.py
---------------------------
Fitxa del consumidor (profile), màscares d'entrenament i Datasets de PyTorch.

Inclou:
  - Fitxa de 10 indicadors: s'hi afegeix la sensibilitat a la temperatura
    (com varia el consum diari de la corba quan varia el de la població)
  - Els Datasets retornen 16 canals d'entrada per instant (veure context.py)
    i el pes de la corba a la pèrdua
"""

import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

from transformer.core.config import (
    STEPS_PER_DAY, MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN,
    MASK_BLOCK_PROB, EMPIRICAL_MASK_PROB, PROFILE_DIM,
)
from transformer.core.context import (
    SLOT, WEEKEND, N_DAYS_FULL, sample_features,
)

PEAK_HOUR_START = 36     # 18h
PEAK_HOUR_END   = 44     # 22h


# ─── Robust scaling (mediana / MAD) ───────────────────────────────────────────

def robust_stats(values: np.ndarray) -> tuple[float, float]:
    """Mediana i MAD (×1.4826) dels punts coneguts. Fallback a std o 1."""
    arr = values[~np.isnan(values)]
    if len(arr) == 0:
        return 0.0, 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    if mad < 1e-6:
        mad = float(np.std(arr)) if np.std(arr) > 1e-6 else 1.0
    return med, mad


# ─── Fitxa del consumidor ─────────────────────────────────────────────────────

def temperature_sensitivity(values: np.ndarray, pop_daily: np.ndarray) -> float:
    """
    Pendent de log(consum diari de la corba) respecte log(consum diari de la
    població), només amb dies prou observats. ~0 = no reacciona al fred,
    ~1 = reacciona com la mitjana, >2 = molt sensible (calefacció elèctrica).
    A les dades va de -0.9 (p10) a 2.5 (p90).
    """
    v = values[:N_DAYS_FULL * STEPS_PER_DAY].reshape(N_DAYS_FULL, STEPS_PER_DAY)
    ok = (~np.isnan(v)).sum(axis=1) >= STEPS_PER_DAY // 2
    if ok.sum() < 5:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cm = np.nanmean(v[ok], axis=1)
    x = np.log(pop_daily[ok] / pop_daily[ok].mean())
    y = np.log((cm + 1.0) / (cm.mean() + 1.0))
    denom = float((x * x).sum())
    if denom < 1e-9:
        return 0.0
    return float(np.clip((x * y).sum() / denom, -1.0, 4.0))


def _profile_raw(values: np.ndarray, pop_daily: np.ndarray) -> np.ndarray:
    """
    10 indicadors en brut (només punts coneguts):
      [0] mitjana  [1] MAD  [2] CV  [3] cap de setmana / laborable
      [4] pic vespre / resta  [5] asimetria  [6] autocorrelació a 1 dia
      [7] p95  [8] IQR  [9] sensibilitat a la temperatura
    """
    obs = ~np.isnan(values)
    arr = values[obs]
    if len(arr) < 10:
        return np.full(PROFILE_DIM, np.nan)

    mu = float(arr.mean())
    med, mad = robust_stats(values)
    cv = mad / med if med > 1e-6 else 0.0
    std = float(arr.std())
    skew = float(((arr - mu) ** 3).mean() / (std ** 3 + 1e-6)) if std > 1e-6 else 0.0

    vc = np.where(obs, values, mu) - mu
    autocorr = float((vc[:-STEPS_PER_DAY] * vc[STEPS_PER_DAY:]).sum() / ((vc ** 2).sum() + 1e-6))

    p25, p75, p95 = np.percentile(arr, [25, 75, 95])

    wd   = ~WEEKEND & obs
    we   = WEEKEND & obs
    peak = (SLOT >= PEAK_HOUR_START) & (SLOT < PEAK_HOUR_END)
    peak_ratio = 1.0
    if (wd & peak).sum() > 5 and (wd & ~peak).sum() > 5:
        off = values[wd & ~peak].mean()
        if off > 1e-6:
            peak_ratio = float(values[wd & peak].mean() / off)
    we_wd = 1.0
    if we.sum() >= 5 and wd.sum() >= 5 and values[wd].mean() > 1e-6:
        we_wd = float(values[we].mean() / values[wd].mean())

    beta = temperature_sensitivity(values, pop_daily)
    return np.array([mu, mad, cv, we_wd, peak_ratio, skew, autocorr,
                     p95, p75 - p25, beta], dtype=np.float64)


def profile_features(curves: np.ndarray, pop_daily: np.ndarray) -> np.ndarray:
    """(N, T) amb NaN → (N, PROFILE_DIM) indicadors en brut (NaN si no n'hi ha prou dades)."""
    return np.stack([_profile_raw(c, pop_daily) for c in curves])


def fit_profile_stats(raw: np.ndarray) -> dict:
    """Mitjana i desviació de cada indicador sobre les referències."""
    return {"mean": np.nanmean(raw, axis=0), "std": np.nanstd(raw, axis=0) + 1e-6}


def normalize_profiles(raw: np.ndarray, stats: dict) -> np.ndarray:
    """Estandarditza; les corbes sense prou dades queden a 0 (valor mitjà)."""
    out = (raw - stats["mean"]) / stats["std"]
    return np.nan_to_num(np.clip(out, -10, 10), nan=0.0).astype(np.float32)


# ─── Màscares realistes ───────────────────────────────────────────────────────

def fit_mask_distribution(holed_curves: np.ndarray) -> dict:
    """
    Aprèn la distribució empírica de forats de les corbes holed_:
      - ratios: % de NaN per corba
      - block_lengths / block_start_hours: longitud i franja d'inici de cada
        bloc real, en parells (els blocs llargs comencen sovint a mitjanit)
    """
    nan_mask = np.isnan(holed_curves)
    N, T     = nan_mask.shape
    ratios   = nan_mask.sum(axis=1) / T

    block_lengths, block_start_hours = [], []
    for i in range(N):
        m = nan_mask[i]
        if m.sum() == 0:
            continue
        diff   = np.diff(np.concatenate([[False], m, [False]]).astype(int))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            block_lengths.append(e - s)
            block_start_hours.append(int(s % STEPS_PER_DAY))

    if not block_lengths:
        block_lengths, block_start_hours = [10], [0]
    return {
        "ratios": ratios,
        "block_lengths": np.array(block_lengths),
        "block_start_hours": np.array(block_start_hours),
    }


def generate_mask_uniform(T: int, ratio: float, min_len: int, max_len: int,
                          block_prob: float, rng: np.random.Generator) -> np.ndarray:
    mask     = np.zeros(T, dtype=bool)
    n_target = max(1, int(T * ratio))
    while mask.sum() < n_target:
        if rng.random() < block_prob:
            length = int(rng.integers(min_len, max_len + 1))
            start  = int(rng.integers(0, T))
            mask[start:min(start + length, T)] = True
        else:
            mask[int(rng.integers(0, T))] = True
    return mask


def generate_mask_empirical(T: int, mask_dist: dict,
                            rng: np.random.Generator) -> np.ndarray:
    """Mostreja ratio real + parells (longitud, franja d'inici) de blocs reals."""
    mask = np.zeros(T, dtype=bool)
    target_count = max(1, int(T * float(rng.choice(mask_dist["ratios"]))))
    lengths  = mask_dist["block_lengths"]
    starts_h = mask_dist["block_start_hours"]
    n_days   = max(1, T // STEPS_PER_DAY)
    while mask.sum() < target_count:
        i      = int(rng.integers(0, len(lengths)))
        start  = int(rng.integers(0, n_days)) * STEPS_PER_DAY + int(starts_h[i])
        mask[start:min(start + int(lengths[i]), T)] = True
    return mask


def generate_mask(T: int, rng: np.random.Generator,
                  mask_dist: dict | None = None) -> np.ndarray:
    if mask_dist is not None and rng.random() < EMPIRICAL_MASK_PROB:
        return generate_mask_empirical(T, mask_dist, rng)
    return generate_mask_uniform(T, MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN,
                                 MASK_BLOCK_PROB, rng)


# ─── Datasets ─────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Entrenament: corbes de referència completes amb forats simulats.

    Els arrays grans es comparteixen (no es copien); `indices` tria quines
    corbes formen part d'aquest dataset (train o validació proxy).
    fixed_masks (len(indices), T): màscares congelades per a validació estable.
    """

    def __init__(self, curves_norm: np.ndarray, profiles: np.ndarray,
                 weights: np.ndarray, nb_norm: np.ndarray, nb_anom: np.ndarray,
                 population: dict, indices: np.ndarray,
                 mask_dist: dict | None = None, augment: bool = True,
                 seed: int = 42, fixed_masks: np.ndarray | None = None):
        self.curves      = curves_norm
        self.profiles    = profiles
        self.weights     = weights.astype(np.float32)
        self.nb_norm     = nb_norm
        self.nb_anom     = nb_anom
        self.population  = population
        self.indices     = np.asarray(indices)
        self.mask_dist   = mask_dist
        self.augment     = augment
        self.fixed_masks = fixed_masks
        self.T           = curves_norm.shape[1]
        self.rng         = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> dict:
        j = self.indices[i]
        x = self.curves[j].astype(np.float32, copy=True)
        if self.augment:
            x += self.rng.normal(0, 0.01, self.T).astype(np.float32)
        if self.fixed_masks is not None:
            mask = self.fixed_masks[i]
        else:
            mask = generate_mask(self.T, self.rng, self.mask_dist)
        feats = sample_features(x, mask, self.population, self.nb_norm[j], self.nb_anom[j])
        return {
            "input":   torch.from_numpy(feats),
            "target":  torch.from_numpy(x),
            "mask":    torch.from_numpy(mask.copy()),
            "profile": torch.from_numpy(self.profiles[j]),
            "weight":  torch.tensor(self.weights[j]),
        }


class InferenceDataset(Dataset):
    """Corbes holed_ amb els seus forats reals (validació etiquetada o test)."""

    def __init__(self, curves_norm: np.ndarray, nan_masks: np.ndarray,
                 profiles: np.ndarray, nb_norm: np.ndarray, nb_anom: np.ndarray,
                 population: dict):
        self.curves     = curves_norm.astype(np.float32)
        self.nan_masks  = nan_masks.astype(bool)
        self.profiles   = profiles
        self.nb_norm    = nb_norm
        self.nb_anom    = nb_anom
        self.population = population

    def __len__(self) -> int:
        return len(self.curves)

    def __getitem__(self, idx: int) -> dict:
        feats = sample_features(self.curves[idx], self.nan_masks[idx], self.population,
                                self.nb_norm[idx], self.nb_anom[idx])
        return {
            "input":   torch.from_numpy(feats),
            "mask":    torch.from_numpy(self.nan_masks[idx].copy()),
            "profile": torch.from_numpy(self.profiles[idx]),
        }


def worker_init_fn(worker_id: int) -> None:
    """Cada procés de càrrega rep una llavor diferent (si no, generarien les mateixes màscares)."""
    info = torch.utils.data.get_worker_info()
    info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2 ** 32)
