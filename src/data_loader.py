"""
data_loader.py
--------------
Càrrega i validació bàsica de les dades en format parquet.
Retorna sempre DataFrames amb l'índex com a DatetimeIndex.
"""

import pandas as pd
from pathlib import Path
from src.config import HOLED_PREFIX


def load_parquet(path: Path | str) -> pd.DataFrame:
    """Carrega un fitxer parquet i assegura que l'índex és DatetimeIndex."""
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def split_reference_holed(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separa les columnes de referència (senceres) de les columnes holed_ (amb NaN).

    Retorna
    -------
    X_ref   : DataFrame amb les corbes de referència
    X_holed : DataFrame amb les corbes a imputar
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
    X_holed : (T, 1_000)  corbes amb forats (input)
    y_train : (T, 1_000)  corbes completes (ground truth per avaluar MAE)
    """
    X_train = load_parquet(x_path)
    y_train = load_parquet(y_path)
    X_ref, X_holed = split_reference_holed(X_train)

    print(f"[data_loader] X_ref:   {X_ref.shape}")
    print(f"[data_loader] X_holed: {X_holed.shape}  "
          f"(NaN total: {X_holed.isna().sum().sum():,})")
    print(f"[data_loader] y_train: {y_train.shape}")
    return X_ref, X_holed, y_train


def load_test(x_path: Path | str) -> tuple:
    """
    Carrega les dades de test.

    Retorna
    -------
    X_ref   : corbes de referència del test
    X_holed : corbes amb forats a imputar
    """
    X_test = load_parquet(x_path)
    X_ref, X_holed = split_reference_holed(X_test)

    print(f"[data_loader] X_ref:   {X_ref.shape}")
    print(f"[data_loader] X_holed: {X_holed.shape}  "
          f"(NaN total: {X_holed.isna().sum().sum():,})")
    return X_ref, X_holed
