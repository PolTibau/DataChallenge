"""
data_loader.py
--------------
Càrrega de les dades en format CSV.

Estructura dels CSV:
  - Primera columna: "Horodate" (timestamps, es converteix a índex)
  - Resta de columnes: corbes numèriques (referència o holed_)
"""

import pandas as pd
from pathlib import Path
from src.config import HOLED_PREFIX, INDEX_COL


def load_csv(path: Path | str) -> pd.DataFrame:
    """
    Carrega un CSV, usa Horodate com a índex i assegura que
    totes les columnes restants són numèriques.
    """
    df = pd.read_csv(path, index_col=INDEX_COL)
    # Convertir index a DatetimeIndex (els timestamps del challenge són strings)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def split_reference_holed(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa les columnes de referència (senceres) de les columnes holed_ (amb NaN).
    """
    holed_mask = df.columns.str.startswith(HOLED_PREFIX)
    X_ref   = df.loc[:, ~holed_mask]
    X_holed = df.loc[:,  holed_mask]
    return X_ref, X_holed


def load_train(x_path: Path | str, y_path: Path | str) -> tuple:
    """
    Carrega les dades de train.

    Retorna
    -------
    X_ref   : (T, 20_000) corbes de referència
    X_holed : (T, 1_000)  corbes amb forats
    y_train : (T, 1_000)  corbes completes (ground truth)
    """
    X_train = load_csv(x_path)
    y_train = load_csv(y_path)
    X_ref, X_holed = split_reference_holed(X_train)

    print(f"[data_loader] X_ref:   {X_ref.shape}")
    print(f"[data_loader] X_holed: {X_holed.shape}  "
          f"(NaN total: {X_holed.isna().sum().sum():,})")
    print(f"[data_loader] y_train: {y_train.shape}")
    print(f"[data_loader] Index[:2]: {X_holed.index[:2].tolist()}")
    return X_ref, X_holed, y_train


def load_test(x_path: Path | str) -> tuple:
    """
    Carrega les dades de test.

    Retorna
    -------
    X_ref   : corbes de referència
    X_holed : corbes amb forats a imputar
    horodates : array de timestamps originals (per al CSV de submissió)
    """
    X_test = load_csv(x_path)
    X_ref, X_holed = split_reference_holed(X_test)

    print(f"[data_loader] X_ref:   {X_ref.shape}")
    print(f"[data_loader] X_holed: {X_holed.shape}  "
          f"(NaN total: {X_holed.isna().sum().sum():,})")
    print(f"[data_loader] Index[:2]: {X_holed.index[:2].tolist()}")
    # Retornem els timestamps com a strings per al CSV de submissió
    horodates = X_holed.index.strftime("%Y-%m-%d %H:%M:%S").values
    return X_ref, X_holed, horodates
