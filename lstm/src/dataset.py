"""
lstm/src/dataset.py — v3
------------------------
PyTorch Dataset per al Transformer d'imputació.

Canvis v3:
  - Profile ampliat de 4 a 9 dimensions amb estadístiques de forma:
      [mu, sigma_robust, cv, weekend_ratio, peak_day_ratio,
       skewness, autocorr_lag48, p95_norm, iqr_norm]
  - Normalització robusta amb mediana / MAD en lloc de mean / std
  - Compute global stats sobre els sigmas (bug fix v2 — abans usava mus)
  - Màscares realistes: opció d'usar patrons empírics del test
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from lstm.src.config import (
    SEQ_LEN, MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN, MASK_BLOCK_PROB,
    EMPIRICAL_MASK_PROB, PROFILE_DIM,
)

# Constants temporals
STEPS_PER_DAY = 48      # timesteps de 30min per dia
START_WEEKDAY = 0       # 2023-01-09 és dilluns

# Hora pic: vespre (18-22h = timesteps 36-44 dins d'un dia)
PEAK_HOUR_START = 36
PEAK_HOUR_END   = 44


# ─── Robust scaling (mediana / MAD) ───────────────────────────────────────────

def robust_stats(values: np.ndarray) -> tuple[float, float]:
    """
    Calcula mediana i MAD (Median Absolute Deviation) ajustada.

    Més robust a outliers que mean/std. La constant 1.4826 fa que el MAD
    sigui un estimador consistent de la desviació estàndard sota la
    distribució normal.

    Paràmetres
    ----------
    values : (T,) — pot contenir NaN

    Retorna
    -------
    median, mad : floats
    """
    arr = values[~np.isnan(values)]
    if len(arr) == 0:
        return 0.0, 1.0
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.4826
    if mad < 1e-6:
        mad = float(np.std(arr)) if np.std(arr) > 1e-6 else 1.0
    return med, mad


# ─── Càlcul de profile enriquit ───────────────────────────────────────────────

def compute_weekend_ratio(T: int = SEQ_LEN) -> float:
    """Proporció de timesteps de cap de setmana a la seqüència."""
    days     = np.arange(T) // STEPS_PER_DAY
    weekdays = (days + START_WEEKDAY) % 7
    return float((weekdays >= 5).mean())


def is_weekend_flag(T: int = SEQ_LEN) -> np.ndarray:
    """(T,) bool — True si el timestep és cap de setmana."""
    days     = np.arange(T) // STEPS_PER_DAY
    weekdays = (days + START_WEEKDAY) % 7
    return weekdays >= 5


def is_peak_hour_flag(T: int = SEQ_LEN) -> np.ndarray:
    """(T,) bool — True si és hora pic del vespre (18-22h)."""
    hour_idx = np.arange(T) % STEPS_PER_DAY
    return (hour_idx >= PEAK_HOUR_START) & (hour_idx < PEAK_HOUR_END)


def compute_profile(values: np.ndarray,
                    global_stats: dict) -> np.ndarray:
    """
    Calcula el vector de perfil enriquit (9 dimensions).

    Cada feature aporta informació discriminativa diferent:

      [0] mu_norm           — nivell de consum (industrial alt vs domèstic baix)
      [1] sigma_robust_norm — variabilitat robusta (corbes planes vs variables)
      [2] cv_norm           — coeficient variació (estabilitat relativa)
      [3] weekend_ratio     — proporció timesteps coneguts en cap de setmana
      [4] peak_day_ratio    — ratio consum_pic_vespre / consum_dia
                              ALT → domèstic (pic vespre)
                              BAIX → industrial (planet o pic diürn)
      [5] skewness          — asimetria de la distribució de valors
                              ALT → pics esporàdics
                              BAIX → consum estable
      [6] autocorr_lag48    — correlació amb si mateixa amb un dia de lag
                              ALT → patró diari fort (domèstic)
                              BAIX → comportament caòtic
      [7] p95_norm          — percentil 95 normalitzat (pic màxim típic)
      [8] iqr_norm          — interquartile range normalitzat (spread robust)

    Paràmetres
    ----------
    values       : (T,) valors de la corba (pot tenir NaN)
    global_stats : dict amb estadístiques globals per normalitzar

    Retorna
    -------
    profile : (9,) vector float32 normalitzat
    """
    arr = values[~np.isnan(values)]
    if len(arr) < 10:
        return np.zeros(PROFILE_DIM, dtype=np.float32)

    # Estadístiques bàsiques robustes
    mu  = float(np.nanmean(values))
    med, mad = robust_stats(values)
    cv  = mad / med if med > 1e-6 else 0.0

    # Skewness (Fisher-Pearson)
    diffs = arr - mu
    std   = arr.std()
    skew  = float((diffs ** 3).mean() / (std ** 3 + 1e-6)) if std > 1e-6 else 0.0

    # Autocorrelació a lag 48 (un dia)
    if len(values) > STEPS_PER_DAY + 10:
        v = np.where(np.isnan(values), mu, values)  # imputar amb mu per calcular
        v_centered = v - mu
        num = (v_centered[:-STEPS_PER_DAY] * v_centered[STEPS_PER_DAY:]).sum()
        den = (v_centered ** 2).sum() + 1e-6
        autocorr = float(num / den)
    else:
        autocorr = 0.0

    # Percentil 95 i IQR
    p95 = float(np.percentile(arr, 95))
    p25 = float(np.percentile(arr, 25))
    p75 = float(np.percentile(arr, 75))
    iqr = p75 - p25

    # Peak/day ratio: consum mitjà a hores pic vs fora-pic (durant dies laborables)
    weekend_mask = is_weekend_flag(len(values))
    peak_mask    = is_peak_hour_flag(len(values))
    not_nan      = ~np.isnan(values)
    workday      = ~weekend_mask & not_nan

    if (workday & peak_mask).sum() > 5 and (workday & ~peak_mask).sum() > 5:
        peak_mean = values[workday & peak_mask].mean()
        off_mean  = values[workday & ~peak_mask].mean()
        peak_ratio = float(peak_mean / off_mean) if off_mean > 1e-6 else 1.0
    else:
        peak_ratio = 1.0

    # Normalitzar contra global_stats
    mu_norm      = (mu  - global_stats["mu_mu"])   / (global_stats["mu_sigma"]   + 1e-6)
    sig_norm     = (mad - global_stats["mad_mu"])  / (global_stats["mad_sigma"]  + 1e-6)
    cv_norm      = (cv  - global_stats["cv_mu"])   / (global_stats["cv_sigma"]   + 1e-6)
    skew_norm    = (skew - global_stats["skew_mu"]) / (global_stats["skew_sigma"] + 1e-6)
    p95_norm     = (p95 - global_stats["p95_mu"])  / (global_stats["p95_sigma"]  + 1e-6)
    iqr_norm     = (iqr - global_stats["iqr_mu"])  / (global_stats["iqr_sigma"]  + 1e-6)
    peak_norm    = (peak_ratio - global_stats["peak_mu"]) / (global_stats["peak_sigma"] + 1e-6)

    return np.array([
        mu_norm, sig_norm, cv_norm, 0.0,  # weekend_ratio omplert després
        peak_norm, skew_norm, autocorr, p95_norm, iqr_norm
    ], dtype=np.float32)


def compute_global_stats(curves: np.ndarray) -> dict:
    """
    Calcula estadístiques globals sobre TOTES les corbes per normalitzar profiles.

    Important v3: ara calculem mu i sigma de cada feature (mu, mad, cv, ...)
    sobre la població de corbes, no només de la mu (bug fix).
    """
    print("  Calculant estadístiques globals per profiles (v3)...")
    mus, mads, cvs, skews, p95s, iqrs, peaks = [], [], [], [], [], [], []

    weekend_mask = is_weekend_flag(curves.shape[1])
    peak_mask    = is_peak_hour_flag(curves.shape[1])

    for c in curves:
        arr = c[~np.isnan(c)]
        if len(arr) < 10:
            continue

        mu = float(np.nanmean(c))
        med, mad = robust_stats(c)
        cv = mad / med if med > 1e-6 else 0.0

        diffs = arr - mu
        std   = arr.std()
        skew  = float((diffs ** 3).mean() / (std ** 3 + 1e-6)) if std > 1e-6 else 0.0

        p95 = float(np.percentile(arr, 95))
        p25 = float(np.percentile(arr, 25))
        p75 = float(np.percentile(arr, 75))
        iqr = p75 - p25

        not_nan = ~np.isnan(c)
        workday = ~weekend_mask & not_nan
        if (workday & peak_mask).sum() > 5 and (workday & ~peak_mask).sum() > 5:
            pm = c[workday & peak_mask].mean()
            om = c[workday & ~peak_mask].mean()
            peak_ratio = float(pm / om) if om > 1e-6 else 1.0
        else:
            peak_ratio = 1.0

        mus.append(mu);     mads.append(mad);  cvs.append(cv)
        skews.append(skew); p95s.append(p95);  iqrs.append(iqr)
        peaks.append(peak_ratio)

    def _stats(lst):
        a = np.array(lst)
        return float(a.mean()), float(a.std() + 1e-6)

    mu_mu,   mu_sigma   = _stats(mus)
    mad_mu,  mad_sigma  = _stats(mads)
    cv_mu,   cv_sigma   = _stats(cvs)
    skew_mu, skew_sigma = _stats(skews)
    p95_mu,  p95_sigma  = _stats(p95s)
    iqr_mu,  iqr_sigma  = _stats(iqrs)
    peak_mu, peak_sigma = _stats(peaks)

    stats = {
        "mu_mu": mu_mu,     "mu_sigma": mu_sigma,
        "mad_mu": mad_mu,   "mad_sigma": mad_sigma,
        "cv_mu": cv_mu,     "cv_sigma": cv_sigma,
        "skew_mu": skew_mu, "skew_sigma": skew_sigma,
        "p95_mu": p95_mu,   "p95_sigma": p95_sigma,
        "iqr_mu": iqr_mu,   "iqr_sigma": iqr_sigma,
        "peak_mu": peak_mu, "peak_sigma": peak_sigma,
    }
    print(f"    mu={mu_mu:.1f}±{mu_sigma:.1f}W  "
          f"peak_ratio={peak_mu:.2f}±{peak_sigma:.2f}  "
          f"skew={skew_mu:.2f}±{skew_sigma:.2f}")
    return stats


# ─── Màscares realistes ───────────────────────────────────────────────────────

def fit_mask_distribution(holed_curves: np.ndarray) -> dict:
    """
    Aprèn la distribució empírica de forats a partir de les corbes holed_ del test.

    Estima:
      - ratio_per_curve: distribució del % de NaN per corba
      - block_lengths: distribució de longituds de blocs consecutius

    Aquesta distribució es fa servir per generar màscares d'entrenament que
    imiten les del test, en lloc d'usar masking uniforme.
    """
    nan_mask = np.isnan(holed_curves)
    N, T     = nan_mask.shape

    # Ratio per corba
    ratios = nan_mask.sum(axis=1) / T

    # Longituds de blocs consecutius
    block_lengths = []
    for i in range(N):
        m = nan_mask[i]
        # Buscar runs de True
        if m.sum() == 0:
            continue
        diff   = np.diff(np.concatenate([[False], m, [False]]).astype(int))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]
        for s, e in zip(starts, ends):
            block_lengths.append(e - s)

    block_lengths = np.array(block_lengths) if len(block_lengths) > 0 else np.array([10])

    return {
        "ratios": ratios,
        "block_lengths": block_lengths,
    }


def generate_mask_uniform(T: int, ratio: float, min_len: int, max_len: int,
                          block_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Màscara uniforme (versió antiga)."""
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
    """
    Genera una màscara mostrejant del patró empíric del test.

    Mostreja:
      1. Un ratio target (de la distribució real de ratios)
      2. Longituds de blocs (de la distribució real de blocs)
    """
    mask = np.zeros(T, dtype=bool)

    # Mostrejar ratio target
    target_ratio = float(rng.choice(mask_dist["ratios"]))
    target_count = max(1, int(T * target_ratio))

    block_lengths = mask_dist["block_lengths"]

    while mask.sum() < target_count:
        # Cada bloc d'una longitud mostrejada de la distribució real
        length = int(rng.choice(block_lengths))
        start  = int(rng.integers(0, T))
        mask[start:min(start + length, T)] = True

    return mask


def generate_mask(T: int, ratio: float, min_len: int, max_len: int,
                  block_prob: float, rng: np.random.Generator,
                  mask_dist: dict | None = None,
                  empirical_prob: float = 0.0) -> np.ndarray:
    """
    Genera una màscara. Si mask_dist és proporcionat, amb probabilitat
    empirical_prob usa el patró empíric, sinó l'uniforme.
    """
    if mask_dist is not None and rng.random() < empirical_prob:
        return generate_mask_empirical(T, mask_dist, rng)
    return generate_mask_uniform(T, ratio, min_len, max_len, block_prob, rng)


# ─── Datasets ─────────────────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """
    Dataset d'entrenament: corbes senceres amb forats simulats.

    v3:
      - Profiles enriquits (9 dims)
      - Suport per màscares empíriques (basades en distribució real del test)
    """

    def __init__(self, curves_norm: np.ndarray,
                 curves_raw: np.ndarray | None = None,
                 global_stats: dict | None = None,
                 mask_dist: dict | None = None,
                 augment: bool = True, seed: int = 42):
        self.curves    = curves_norm.astype(np.float32)
        self.N, self.T = self.curves.shape
        self.augment   = augment
        self.rng       = np.random.default_rng(seed)
        self.mask_dist = mask_dist

        # Calcular profiles si tenim les dades originals
        if curves_raw is not None and global_stats is not None:
            print(f"  Computing {self.N:,} profiles...")
            self.profiles = np.stack([
                compute_profile(curves_raw[i], global_stats)
                for i in range(self.N)
            ]).astype(np.float32)
        else:
            self.profiles = np.zeros((self.N, PROFILE_DIM), dtype=np.float32)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        original = self.curves[idx].copy()
        if self.augment:
            original += self.rng.normal(0, 0.01, self.T).astype(np.float32)

        mask = generate_mask(
            self.T, MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN, MASK_BLOCK_PROB,
            self.rng, self.mask_dist, EMPIRICAL_MASK_PROB,
        )
        masked_input = original.copy()
        masked_input[mask] = 0.0

        return {
            "input"  : torch.from_numpy(masked_input),
            "target" : torch.from_numpy(original),
            "mask"   : torch.from_numpy(mask),
            "profile": torch.from_numpy(self.profiles[idx]),
        }


class InferenceDataset(Dataset):
    """
    Dataset d'inferència: corbes holed_ amb les màscares reals.

    v3: profiles enriquits + weekend_ratio calculat sobre punts coneguts.
    """

    def __init__(self, curves_norm: np.ndarray,
                 nan_masks: np.ndarray,
                 curves_raw: np.ndarray | None = None,
                 global_stats: dict | None = None):
        self.curves    = curves_norm.astype(np.float32)
        self.nan_masks = nan_masks.astype(bool)
        self.N, self.T = self.curves.shape

        if curves_raw is not None and global_stats is not None:
            weekend_mask = is_weekend_flag(self.T)
            profiles = []
            for i in range(self.N):
                p = compute_profile(curves_raw[i], global_stats)
                known = ~nan_masks[i]
                if known.sum() > 0:
                    p[3] = float(weekend_mask[known].mean())
                profiles.append(p)
            self.profiles = np.stack(profiles).astype(np.float32)
        else:
            self.profiles = np.zeros((self.N, PROFILE_DIM), dtype=np.float32)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        return {
            "input"  : torch.from_numpy(self.curves[idx]),
            "mask"   : torch.from_numpy(self.nan_masks[idx]),
            "profile": torch.from_numpy(self.profiles[idx]),
        }
