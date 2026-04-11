"""
src/data_loader.py
------------------
Càrrega de dades compartida entre KNN i Transformer.

Estructura dels CSV:
  - Primera columna "Horodate": timestamps
  - Columnes numèriques: corbes de referència + columnes holed_
"""

import numpy as np
import pandas as pd
from pathlib import Path

INDEX_COL    = "Horodate"
HOLED_PREFIX = "holed_"


def load_csv(path: Path | str) -> pd.DataFrame:
    """Carrega un CSV usant Horodate com a índex DatetimeIndex."""
    df = pd.read_csv(path, index_col=INDEX_COL)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def split_reference_holed(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa corbes de referència de les holed_."""
    holed_mask = df.columns.str.startswith(HOLED_PREFIX)
    return df.loc[:, ~holed_mask], df.loc[:, holed_mask]


def load_train(x_path: Path | str,
               y_path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Carrega les dades de train.

    Retorna
    -------
    X_ref   : (T, N_ref)   corbes de referència senceres
    X_holed : (T, N_holed) corbes amb forats
    y_train : (T, N_holed) ground truth
    """
    X_train = load_csv(x_path)
    y_train = load_csv(y_path)
    X_ref, X_holed = split_reference_holed(X_train)

    print(f"[data_loader] X_ref   : {X_ref.shape}")
    print(f"[data_loader] X_holed : {X_holed.shape}  "
          f"(NaN: {X_holed.isna().sum().sum():,})")
    print(f"[data_loader] y_train : {y_train.shape}")
    print(f"[data_loader] Índex   : {X_holed.index[0]}  →  {X_holed.index[-1]}")
    return X_ref, X_holed, y_train


def load_test(x_path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega les dades de test.

    Retorna
    -------
    X_ref   : corbes de referència
    X_holed : corbes amb forats a imputar
    """
    X_test = load_csv(x_path)
    X_ref, X_holed = split_reference_holed(X_test)

    print(f"[data_loader] X_ref   : {X_ref.shape}")
    print(f"[data_loader] X_holed : {X_holed.shape}  "
          f"(NaN: {X_holed.isna().sum().sum():,})")
    return X_ref, X_holed


def save_submission(y_pred: pd.DataFrame,
                    out_path: Path | str,
                    timestamps) -> None:
    """
    Guarda el CSV de submissió en el format exacte del challenge:
      Horodate | holed_1 | ... | holed_1000   (index=False)

    Paràmetres
    ----------
    y_pred     : DataFrame amb columnes holed_
    out_path   : ruta del CSV de sortida
    timestamps : array de strings amb els timestamps originals del test
    """
    out      = y_pred.reset_index(drop=True).copy()
    out.insert(0, INDEX_COL, timestamps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[data_loader] Submissió guardada a: {out_path}")
    print(f"              Shape: {out.shape}  |  "
          f"Primers Horodate: {list(out[INDEX_COL][:2])}")
