"""
src/evaluator.py
----------------
Avaluació compartida entre KNN i Transformer.

El MAE es calcula NOMÉS sobre els punts que eren NaN originalment,
tal com especifica el challenge. Ignora posicions on y_train és NaN
(ground truth incomplet — cas de holed_192).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def compute_mae(y_true: pd.DataFrame,
                y_pred: pd.DataFrame,
                nan_mask: pd.DataFrame) -> float:
    """
    MAE global sobre tots els punts imputats.

    Paràmetres
    ----------
    y_true   : ground truth (pot tenir NaN en algunes posicions)
    y_pred   : prediccions (no hauria de tenir NaN)
    nan_mask : màscara booleana, True = posició original NaN a X_holed
    """
    cols   = [c for c in y_pred.columns if c in y_true.columns]
    errors = []
    for col in cols:
        mask  = nan_mask[col].values.astype(bool)
        if mask.sum() == 0:
            continue
        tv    = y_true[col].values[mask]
        pv    = y_pred[col].values[mask]
        valid = ~(np.isnan(tv) | np.isnan(pv))
        if valid.sum() > 0:
            errors.extend(np.abs(tv[valid] - pv[valid]).tolist())

    return float(np.mean(errors)) if errors else float("nan")


def compute_mae_per_curve(y_true: pd.DataFrame,
                          y_pred: pd.DataFrame,
                          nan_mask: pd.DataFrame) -> pd.Series:
    """MAE individual per cada corba holed_."""
    cols = [c for c in y_pred.columns if c in y_true.columns]
    maes = {}
    for col in cols:
        mask  = nan_mask[col].values.astype(bool)
        if mask.sum() == 0:
            maes[col] = np.nan
            continue
        tv    = y_true[col].values[mask]
        pv    = y_pred[col].values[mask]
        valid = ~(np.isnan(tv) | np.isnan(pv))
        maes[col] = float(np.mean(np.abs(tv[valid] - pv[valid]))) \
                    if valid.sum() > 0 else np.nan
    return pd.Series(maes)


def print_summary(y_true: pd.DataFrame,
                  y_pred: pd.DataFrame,
                  nan_mask: pd.DataFrame,
                  model_name: str = "") -> None:
    """Imprimeix el resum complet de l'avaluació."""
    # Alinear per posició (evita problemes d'índex)
    y_true_r   = y_true.reset_index(drop=True)
    y_pred_r   = y_pred.reset_index(drop=True)
    nan_mask_r = nan_mask.reset_index(drop=True)

    mae_total  = compute_mae(y_true_r, y_pred_r, nan_mask_r)
    mae_curves = compute_mae_per_curve(y_true_r, y_pred_r, nan_mask_r)

    title = f" {model_name} " if model_name else ""
    print(f"\n{'=' * 50}")
    print(f"  Resultats{title}")
    print(f"{'=' * 50}")
    print(f"  MAE total        : {mae_total:.2f} W")
    print(f"  MAE mitjà/corba  : {mae_curves.mean():.2f} W")
    print(f"  MAE mediana/corba: {mae_curves.median():.2f} W")
    print(f"  Pitjor corba     : {mae_curves.idxmax()} ({mae_curves.max():.2f} W)")
    print(f"  Millor corba     : {mae_curves.idxmin()} ({mae_curves.min():.2f} W)")
    print(f"{'=' * 50}")
    return mae_total, mae_curves


def compare_models(mae_knn: float,
                   mae_lstm: float,
                   mae_curves_knn: pd.Series,
                   mae_curves_lstm: pd.Series) -> None:
    """
    Compara els resultats de KNN i Transformer en un sol gràfic.
    Útil per decidir quin model és millor per a la submissió final.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Comparació KNN vs Transformer", fontsize=13, fontweight="bold")

    # MAE global
    ax = axes[0]
    bars = ax.bar(["KNN", "Transformer"], [mae_knn, mae_lstm],
                  color=["steelblue", "seagreen"], alpha=0.85, width=0.4)
    ax.bar_label(bars, fmt="%.1f W", fontsize=10)
    ax.set_title("MAE total (W)")
    ax.set_ylabel("MAE (W)")
    ax.grid(True, alpha=0.3, axis="y")

    # Distribució per corba
    ax = axes[1]
    ax.hist(mae_curves_knn.dropna(), bins=40, alpha=0.6,
            color="steelblue", label=f"KNN (med={mae_curves_knn.median():.0f}W)")
    ax.hist(mae_curves_lstm.dropna(), bins=40, alpha=0.6,
            color="seagreen", label=f"Transformer (med={mae_curves_lstm.median():.0f}W)")
    ax.set_xlabel("MAE per corba (W)")
    ax.set_ylabel("Nombre de corbes")
    ax.set_title("Distribució del MAE per corba")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_curve(col: str,
               X_holed: pd.DataFrame,
               predictions: dict,
               y_true: pd.DataFrame | None = None,
               n_points: int = 200) -> None:
    """
    Visualitza una corba amb les prediccions de tots els models.

    Paràmetres
    ----------
    predictions : dict amb format {"KNN": df_knn, "Transformer": df_lstm}
    """
    colors = {"KNN": "steelblue", "Transformer": "seagreen"}
    fig, ax = plt.subplots(figsize=(14, 4))

    idx        = X_holed.index[:n_points]
    orig       = X_holed[col].iloc[:n_points]
    known_mask = orig.notna()

    # Valors coneguts
    ax.scatter(idx[known_mask], orig[known_mask],
               color="black", s=6, zorder=4, label="Valors coneguts")

    # Prediccions de cada model
    for name, df in predictions.items():
        if col in df.columns:
            ax.plot(idx, df[col].iloc[:n_points],
                    color=colors.get(name, "gray"),
                    lw=1.2, label=name, zorder=2)

    # Ground truth
    if y_true is not None and col in y_true.columns:
        ax.plot(idx, y_true[col].iloc[:n_points],
                color="tomato", lw=1.0, ls="--", alpha=0.7,
                label="Ground truth", zorder=1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
    plt.xticks(rotation=30, ha="right")
    ax.set_title(f"Imputació — {col}", fontsize=12)
    ax.set_ylabel("Consum (W)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
