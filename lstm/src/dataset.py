"""
lstm/src/dataset.py
-------------------
PyTorch Dataset per al Transformer d'imputació.

Estratègia de masking
---------------------
Usem corbes de referència SENCERES i hi simulem forats durant l'entrenament.
El model aprèn a reconstruir els valors emmascarats sense veure mai els forats reals.

Dos tipus de màscara combinats:
  1. Blocs consecutius  — simula fallades prolongades
  2. Punts aleatoris    — simula pèrdues esporàdiques

Novetats v2
-----------
  - Profile embedding: mu i sigma normalitzats + flag cap de setmana
    El model sap si la corba és d'alt o baix consum, i si és cap de setmana
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from lstm.src.config import (
    SEQ_LEN, MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN, MASK_BLOCK_PROB,
)

# Constants temporals
STEPS_PER_DAY = 48      # timesteps de 30min per dia
START_WEEKDAY = 0       # 2023-01-09 és dilluns


def compute_weekend_ratio(T: int = SEQ_LEN) -> float:
    """
    Calcula la proporció de timesteps de cap de setmana a la seqüència.
    Usada per normalitzar el flag de cap de setmana per corba.
    """
    days     = np.arange(T) // STEPS_PER_DAY
    weekdays = (days + START_WEEKDAY) % 7
    return float((weekdays >= 5).mean())


def is_weekend_flag(T: int = SEQ_LEN) -> np.ndarray:
    """Retorna array booleà (T,): True si el timestep és cap de setmana."""
    days     = np.arange(T) // STEPS_PER_DAY
    weekdays = (days + START_WEEKDAY) % 7
    return weekdays >= 5   # 5=dissabte, 6=diumenge


def compute_profile(values: np.ndarray,
                    mu_global: float, sigma_global: float,
                    cv_global: float) -> np.ndarray:
    """
    Calcula el vector de perfil d'una corba: [mu_norm, sigma_norm, cv_norm, weekend_ratio].

    Paràmetres
    ----------
    values      : (T,) valors de la corba (pot tenir NaN)
    mu_global   : mitjana global de les mus de totes les corbes
    sigma_global: desviació estàndard global de les mus
    cv_global   : desviació estàndard global dels cvs

    Retorna
    -------
    profile : (4,) vector normalitzat
        [0] mu_norm    — consum mitjà normalitzat (indica si és industrial/domèstic)
        [1] sigma_norm — variabilitat normalitzada
        [2] cv_norm    — coeficient de variació normalitzat (ratio variabilitat/consum)
        [3] is_weekend — sempre 0.0 per corbes de referència (senceres)
                         serà la proporció de cap de setmana per corbes holed_
    """
    mu    = float(np.nanmean(values))
    sigma = float(np.nanstd(values))
    cv    = sigma / mu if mu > 1e-6 else 0.0

    mu_norm    = (mu    - mu_global)    / (sigma_global + 1e-6)
    sigma_norm = (sigma - mu_global)    / (sigma_global + 1e-6)
    cv_norm    = (cv    - cv_global)    / (cv_global    + 1e-6)

    return np.array([mu_norm, sigma_norm, cv_norm, 0.0], dtype=np.float32)


def compute_global_stats(curves: np.ndarray) -> tuple:
    """
    Calcula estadístiques globals per normalitzar els profiles.

    Retorna
    -------
    mu_global    : mitjana de les mus
    sigma_global : std de les mus
    cv_global    : std dels cvs
    """
    mus    = np.array([np.nanmean(c) for c in curves])
    sigmas = np.array([np.nanstd(c)  for c in curves])
    cvs    = np.where(mus > 1e-6, sigmas / mus, 0.0)

    return float(mus.mean()), float(mus.std() + 1e-6), float(cvs.std() + 1e-6)


def generate_mask(T: int, ratio: float, min_len: int, max_len: int,
                  block_prob: float, rng: np.random.Generator) -> np.ndarray:
    """Genera màscara booleana combinant blocs i punts aleatoris."""
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


class TimeSeriesDataset(Dataset):
    """
    Dataset d'entrenament: corbes senceres amb forats simulats.

    Paràmetres
    ----------
    curves_norm  : (N, T) corbes normalitzades sense NaN
    curves_raw   : (N, T) corbes originals (pre-normalització) per calcular profiles
    global_stats : (mu_global, sigma_global, cv_global) per normalitzar profiles
    augment      : si True, afegeix soroll gaussià lleuger
    """

    def __init__(self, curves_norm: np.ndarray,
                 curves_raw: np.ndarray | None = None,
                 global_stats: tuple | None = None,
                 augment: bool = True, seed: int = 42):
        self.curves  = curves_norm.astype(np.float32)
        self.N, self.T = self.curves.shape
        self.augment = augment
        self.rng     = np.random.default_rng(seed)

        # Calcular profiles si tenim les dades originals
        if curves_raw is not None and global_stats is not None:
            mu_g, sigma_g, cv_g = global_stats
            self.profiles = np.stack([
                compute_profile(curves_raw[i], mu_g, sigma_g, cv_g)
                for i in range(self.N)
            ]).astype(np.float32)
        else:
            # Profiles neutres si no tenim dades originals
            self.profiles = np.zeros((self.N, 4), dtype=np.float32)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        original = self.curves[idx].copy()
        if self.augment:
            original += self.rng.normal(0, 0.01, self.T).astype(np.float32)

        mask         = generate_mask(self.T, MASK_RATIO, MASK_MIN_LEN,
                                     MASK_MAX_LEN, MASK_BLOCK_PROB, self.rng)
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

    Paràmetres
    ----------
    curves_norm  : (N, T) corbes normalitzades (NaN → 0.0)
    nan_masks    : (N, T) bool, True = NaN original
    curves_raw   : (N, T) valors originals per calcular profiles (pot tenir NaN)
    global_stats : (mu_global, sigma_global, cv_global)
    """

    def __init__(self, curves_norm: np.ndarray,
                 nan_masks: np.ndarray,
                 curves_raw: np.ndarray | None = None,
                 global_stats: tuple | None = None):
        self.curves    = curves_norm.astype(np.float32)
        self.nan_masks = nan_masks.astype(bool)
        self.N, self.T = self.curves.shape

        # Calcular profiles
        if curves_raw is not None and global_stats is not None:
            mu_g, sigma_g, cv_g = global_stats
            # Flag cap de setmana: proporció de timesteps coneguts que són cap de setmana
            weekend_mask = is_weekend_flag(self.T)
            profiles = []
            for i in range(self.N):
                p = compute_profile(curves_raw[i], mu_g, sigma_g, cv_g)
                # Calcular proporció cap de setmana sobre punts coneguts
                known = ~nan_masks[i]
                if known.sum() > 0:
                    p[3] = float(weekend_mask[known].mean())
                profiles.append(p)
            self.profiles = np.stack(profiles).astype(np.float32)
        else:
            self.profiles = np.zeros((self.N, 4), dtype=np.float32)

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        return {
            "input"  : torch.from_numpy(self.curves[idx]),
            "mask"   : torch.from_numpy(self.nan_masks[idx]),
            "profile": torch.from_numpy(self.profiles[idx]),
        }
