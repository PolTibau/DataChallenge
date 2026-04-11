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
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from lstm.src.config import (
    SEQ_LEN, MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN, MASK_BLOCK_PROB,
)


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
    curves_norm : (N, T) corbes normalitzades sense NaN
    augment     : si True, afegeix soroll gaussià lleuger
    """

    def __init__(self, curves_norm: np.ndarray,
                 augment: bool = True, seed: int = 42):
        self.curves  = curves_norm.astype(np.float32)
        self.N, self.T = self.curves.shape
        self.augment = augment
        self.rng     = np.random.default_rng(seed)

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
            "input" : torch.from_numpy(masked_input),
            "target": torch.from_numpy(original),
            "mask"  : torch.from_numpy(mask),
        }


class InferenceDataset(Dataset):
    """
    Dataset d'inferència: corbes holed_ amb les màscares reals.

    Paràmetres
    ----------
    curves_norm : (N, T) corbes normalitzades (NaN → 0.0)
    nan_masks   : (N, T) bool, True = NaN original
    """

    def __init__(self, curves_norm: np.ndarray, nan_masks: np.ndarray):
        self.curves    = curves_norm.astype(np.float32)
        self.nan_masks = nan_masks.astype(bool)
        self.N, self.T = self.curves.shape

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> dict:
        return {
            "input": torch.from_numpy(self.curves[idx]),
            "mask" : torch.from_numpy(self.nan_masks[idx]),
        }
