"""
evaluator.py
------------
Càlcul del MAE i visualitzacions per avaluar la qualitat de la imputació.

El MAE es calcula NOMÉS sobre els punts que eren NaN originalment,
tal com especifica el challenge.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.config import HOLED_PREFIX


def compute_mae(y_true: pd.DataFrame,
                y_pred: pd.DataFrame,
                nan_mask: pd.DataFrame) -> float:
    """
    MAE sobre els punts imputats (on nan_mask és True).
    Treballa columna per columna per posició, independent de l'índex.
    """
    cols = [c for c in y_pred.columns
            if c in y_true.columns and c in nan_mask.columns]
    if not cols:
        print("[evaluator] AVÍS: cap columna en comú entre y_true i y_pred.")
        return float("nan")

    true_list, pred_list = [], []
    nan_in_pred = 0

    for col in cols:
        mask = nan_mask[col].values.astype(bool)
        if mask.sum() == 0:
            continue
        tv = y_true[col].values[mask]
        pv = y_pred[col].values[mask]
        # Ignorar posicions on y_pred és NaN (fallback fallit)
        valid = ~np.isnan(pv)
        nan_in_pred += (~valid).sum()
        if valid.sum() == 0:
            continue
        true_list.append(tv[valid])
        pred_list.append(pv[valid])

    if nan_in_pred > 0:
        print(f"[evaluator] AVÍS: {nan_in_pred} valors NaN a y_pred "
              f"(fallback incomplet en algunes posicions).")

    if not true_list:
        print("[evaluator] AVÍS: cap punt imputat vàlid per calcular MAE.")
        return float("nan")

    return float(np.mean(np.abs(
        np.concatenate(true_list) - np.concatenate(pred_list)
    )))


def compute_mae_per_curve(y_true: pd.DataFrame,
                          y_pred: pd.DataFrame,
                          nan_mask: pd.DataFrame) -> pd.Series:
    """MAE individual per cada corba holed_."""
    maes = {}
    for col in y_pred.columns:
        if col not in y_true.columns or col not in nan_mask.columns:
            continue
        mask = nan_mask[col].values.astype(bool)
        if mask.sum() == 0:
            maes[col] = np.nan
            continue
        tv = y_true[col].values[mask]
        pv = y_pred[col].values[mask]
        valid = ~np.isnan(pv)
        if valid.sum() == 0:
            maes[col] = np.nan
            continue
        maes[col] = float(np.mean(np.abs(tv[valid] - pv[valid])))
    return pd.Series(maes)


def print_summary(y_true: pd.DataFrame,
                  y_pred: pd.DataFrame,
                  nan_mask: pd.DataFrame) -> None:
    """Imprimeix un resum complet de l'avaluació."""
    mae_total  = compute_mae(y_true, y_pred, nan_mask)
    mae_curves = compute_mae_per_curve(y_true, y_pred, nan_mask)

    print("=" * 50)
    print(f"  MAE total        : {mae_total:.2f} W")
    print(f"  MAE mitjà/corba  : {mae_curves.mean():.2f} W")
    print(f"  MAE mediana/corba: {mae_curves.median():.2f} W")
    print(f"  Pitjor corba     : {mae_curves.idxmax()} ({mae_curves.max():.2f} W)")
    print(f"  Millor corba     : {mae_curves.idxmin()} ({mae_curves.min():.2f} W)")
    print("=" * 50)


def plot_curve(col: str,
               X_holed: pd.DataFrame,
               y_pred: pd.DataFrame,
               y_true: pd.DataFrame | None = None,
               n_points: int = 200) -> None:
    """Visualitza una corba holed_ amb la imputació i el ground truth."""
    fig, ax = plt.subplots(figsize=(14, 4))
    idx  = X_holed.index[:n_points]
    orig = X_holed[col].iloc[:n_points]
    pred = y_pred[col].iloc[:n_points]

    known_mask = orig.notna()
    ax.plot(idx, pred, color="steelblue", lw=1.2, label="KNN imputació", zorder=2)
    ax.scatter(idx[known_mask], orig[known_mask],
               color="black", s=6, zorder=3, label="Valors coneguts")

    if y_true is not None and col in y_true.columns:
        ax.plot(idx, y_true[col].iloc[:n_points],
                color="tomato", lw=1.0, ls="--", alpha=0.8,
                label="Ground truth", zorder=1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    plt.xticks(rotation=30, ha="right")
    ax.set_title(f"Imputació KNN — {col}", fontsize=12)
    ax.set_ylabel("Consum (W)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_mae_distribution(mae_curves: pd.Series) -> None:
    """Histograma de la distribució de MAE per corba."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mae_curves.dropna(), bins=50, color="steelblue", edgecolor="white")
    ax.axvline(mae_curves.median(), color="tomato", lw=2,
               label=f"Mediana: {mae_curves.median():.1f} W")
    ax.set_xlabel("MAE per corba (W)")
    ax.set_ylabel("Nombre de corbes")
    ax.set_title("Distribució del MAE per corba holed_")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
