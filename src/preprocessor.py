"""
preprocessor.py
---------------
Normalització de corbes per millorar el càlcul de distàncies al KNN.

Per què normalitzem?
  Les corbes de consum tenen magnituds molt diverses (un client industrial
  vs. un habitatge). Si no normalitzem, el KNN s'assemblarà per "volum"
  en lloc de per "forma", i trobarem veïns poc útils.

Normalitzem cada columna (corba) de forma independent.
"""

import numpy as np
import pandas as pd
from src.config import NORMALIZE, NORM_METHOD


def zscore_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalitza cada corba (columna) restant-li la seva mitjana i
    dividint per la seva desviació estàndard.
    Els NaN es mantenen com a NaN (es calcula la stats ignorant-los).
    """
    mean = df.mean(axis=0)       # (n_curves,)
    std  = df.std(axis=0)        # (n_curves,)
    std  = std.replace(0, 1)     # evitar divisió per zero en corbes constants
    return (df - mean) / std


def minmax_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalitza cada corba entre 0 i 1 usant min i max.
    Els NaN es mantenen com a NaN.
    """
    mn = df.min(axis=0)
    mx = df.max(axis=0)
    rng = (mx - mn).replace(0, 1)
    return (df - mn) / rng


def robust_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalitza cada corba per mediana i IQR (rang interquartílic).
    Molt més robust a pics puntuals de consum (forns, aires condicionats...)
    que el zscore, que es veu arrossegat per valors extrems.
    Els NaN es mantenen com a NaN.
    """
    median = df.median(axis=0)
    q75    = df.quantile(0.75, axis=0)
    q25    = df.quantile(0.25, axis=0)
    iqr    = (q75 - q25).replace(0, 1)
    return (df - median) / iqr


def normalize(df: pd.DataFrame, method: str = NORM_METHOD) -> pd.DataFrame:
    """
    Aplica la normalització configurada a config.py.
    Si NORMALIZE=False o method='none', retorna el DataFrame original.
    """
    if not NORMALIZE or method == "none":
        return df

    if method == "zscore":
        return zscore_normalize(df)
    elif method == "minmax":
        return minmax_normalize(df)
    elif method == "robust":
        return robust_normalize(df)
    else:
        raise ValueError(f"Mètode de normalització desconegut: '{method}'. "
                         f"Usa 'zscore', 'robust', 'minmax' o 'none'.")


def get_nan_mask(X_holed: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna una màscara booleana (True = NaN) per a les corbes holed_.
    Útil per saber exactament quins punts hem d'imputar i avaluar.
    """
    return X_holed.isna()


def describe_gaps(X_holed: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna un resum per columna del nombre i longitud dels forats.
    Útil per a exploració i debug.
    """
    records = []
    for col in X_holed.columns:
        series = X_holed[col]
        n_nan  = series.isna().sum()
        # trobar longituds de blocs consecutius de NaN
        in_gap, gap_len, gaps = False, 0, []
        for v in series:
            if pd.isna(v):
                in_gap = True
                gap_len += 1
            elif in_gap:
                gaps.append(gap_len)
                gap_len = 0
                in_gap = False
        if in_gap:
            gaps.append(gap_len)

        records.append({
            "col":       col,
            "n_missing": n_nan,
            "n_gaps":    len(gaps),
            "max_gap":   max(gaps) if gaps else 0,
            "mean_gap":  round(np.mean(gaps), 1) if gaps else 0,
        })
    return pd.DataFrame(records).set_index("col")
