"""
src/data_loader.py
------------------
Càrrega de dades compartida.
"""

import numpy as np
import pandas as pd
from pathlib import Path

INDEX_COL    = "Horodate"
HOLED_PREFIX = "holed_"


def load_csv(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=INDEX_COL)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def split_reference_holed(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    holed_mask = df.columns.str.startswith(HOLED_PREFIX)
    return df.loc[:, ~holed_mask], df.loc[:, holed_mask]


def load_train(x_path: Path | str,
               y_path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = load_csv(x_path)
    y_train = load_csv(y_path)
    X_ref, X_holed = split_reference_holed(X_train)
    print(f"[data_loader] X_ref   : {X_ref.shape}")
    print(f"[data_loader] X_holed : {X_holed.shape}  "
          f"(NaN: {X_holed.isna().sum().sum():,})")
    print(f"[data_loader] y_train : {y_train.shape}")
    return X_ref, X_holed, y_train


def load_test(x_path: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_test = load_csv(x_path)
    X_ref, X_holed = split_reference_holed(X_test)
    print(f"[data_loader] X_ref   : {X_ref.shape}")
    print(f"[data_loader] X_holed : {X_holed.shape}  "
          f"(NaN: {X_holed.isna().sum().sum():,})")
    return X_ref, X_holed


def save_submission(y_pred: pd.DataFrame,
                    out_path: Path | str,
                    timestamps) -> None:
    out      = y_pred.reset_index(drop=True).copy()
    out.insert(0, INDEX_COL, timestamps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[data_loader] Submissió guardada a: {out_path}")
    print(f"              Shape: {out.shape}")
