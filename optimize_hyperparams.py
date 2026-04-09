"""
optimize_hyperparams.py
-----------------------
Bayesian Optimization (via Optuna) sobre tots els hiperparàmetres del KNN.

Per què Bayesian en lloc de Grid Search?
-----------------------------------------
  El Grid Search explora l'espai de forma uniforme i cega: si K=10 dona bon
  resultat, el Grid Search no "sap" que K=11 o K=15 podrien ser millors.
  Optuna construeix un model probabilístic (Tree-structured Parzen Estimator,
  TPE) que aprèn quines zones de l'espai són prometedores i hi focalitza les
  iteracions. Amb el mateix nombre de proves cobreix molt millor l'espai, i
  especialment troba combinacions sinèrgiques que una graella mai veuria.

Espai de cerca
--------------
  k              → [1 .. 300]         enter, continu
  metric         → euclidean | cosine | manhattan | chebyshev
  weighting      → inverse | uniform | softmax | rank | gaussian
  norm_method    → zscore | robust | minmax | none
  min_overlap    → [10 .. 300]        enter, continu
  gaussian_sigma → [0.01 .. 10]       float, log-scale (només si weighting=gaussian)

Estratègia de validació
-----------------------
  Per a cada trial:
    1. Agafem N_EVAL_CURVES corbes holed_ (amb ground truth disponible).
    2. Simulem forats addicionals (SIMULATED_HOLE_RATIO) sobre punts coneguts.
    3. Imputem amb KNN usant els paràmetres del trial.
    4. Mesurem MAE sobre els forats simulats → valor a minimitzar.

Funcions de ponderació (weighting)
-----------------------------------
  inverse   : pes = 1/dist.  Veïns molt propers dominen clarament.
  uniform   : tots els veïns pesen igual.  Efecte suavitzant màxim.
  softmax   : pes = softmax(-dist).  Versió suavitzada d'inverse; evita
              que un sol veí amb dist≈0 monopolitzi tot el pes.
  rank      : pes proporcional al rang invers (K, K-1 ... 1).  Robust
              a escales de distància, no depèn de valors absoluts.
  gaussian  : pes = exp(-dist²/(2σ²)).  Decau suaument; σ és optimitzable.

Mètriques de distància (metric)
---------------------------------
  euclidean  : L2.  Clàssica, sensible a pics de magnitud.
  cosine     : angle entre vectors.  Compara forma independentment d'escala.
  manhattan  : L1.  Menys sensible a outliers que L2.
  chebyshev  : L∞.  Distància del pitjor timestep; molt sensible a pics.

Normalitzacions (norm_method)
------------------------------
  zscore  : (x - mitjana) / std.  Estàndard, sensible a outliers extrems.
  robust  : (x - mediana) / IQR.  Molt robust a pics puntuals de consum.
  minmax  : (x - min) / (max - min).  Escala [0,1].
  none    : sense normalitzar.  Útil per veure si la normalització ajuda.

Execució:
    pip install optuna
    python optimize_hyperparams.py

Genera:
    outputs/hyperparam_results.csv  → historial complet de trials
    outputs/hyperparam_results.png  → gràfic resum
    outputs/optuna_study.pkl        → estudi serialitzat (resumible si s'interromp)
"""

import warnings
warnings.filterwarnings("ignore")

import pickle
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from src.config      import X_TRAIN_PATH, Y_TRAIN_PATH, OUT_DIR
from src.data_loader import load_train

# ── Configuració ──────────────────────────────────────────────────────────────

N_TRIALS             = 200   # iteracions Bayesianes
N_EVAL_CURVES        = 80    # corbes holed_ avaluades per trial
SIMULATED_HOLE_RATIO = 0.15  # fracció de punts coneguts que simulem com a forats
RANDOM_SEED          = 42
STUDY_PATH           = OUT_DIR / "optuna_study.pkl"


# ── Normalització ─────────────────────────────────────────────────────────────

def normalize_series(arr: np.ndarray, method: str) -> np.ndarray:
    """Normalitza un array 1D ignorant NaN."""
    if method == "none":
        return arr.copy()
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return arr.copy()
    if method == "zscore":
        mu, sig = valid.mean(), valid.std()
        return (arr - mu) / (sig if sig > 0 else 1.0)
    elif method == "robust":
        med = np.median(valid)
        q75, q25 = np.percentile(valid, [75, 25])
        iqr = q75 - q25
        return (arr - med) / (iqr if iqr > 0 else 1.0)
    elif method == "minmax":
        mn, mx = valid.min(), valid.max()
        rng = mx - mn
        return (arr - mn) / (rng if rng > 0 else 1.0)
    else:
        raise ValueError(f"norm_method desconegut: '{method}'")


def normalize_matrix_rows(mat: np.ndarray, method: str) -> np.ndarray:
    """Normalitza cada fila (corba) d'una matriu (n_curves, T)."""
    if method == "none":
        return mat.copy()
    result = np.empty_like(mat, dtype=float)
    for i in range(mat.shape[0]):
        result[i] = normalize_series(mat[i], method)
    return result


# ── Distàncies ────────────────────────────────────────────────────────────────

def compute_distances(ref_norm: np.ndarray,
                      target_norm: np.ndarray,
                      valid_mask: np.ndarray,
                      metric: str,
                      min_overlap: int) -> np.ndarray:
    """
    Distància entre target_norm i cada fila de ref_norm,
    usant només els timesteps de valid_mask.
    """
    if valid_mask.sum() < min_overlap:
        return np.full(ref_norm.shape[0], np.inf)

    ref_v = ref_norm[:, valid_mask]
    tgt_v = target_norm[valid_mask]
    diff  = ref_v - tgt_v

    if metric == "euclidean":
        return np.sqrt((diff ** 2).sum(axis=1))
    elif metric == "manhattan":
        return np.abs(diff).sum(axis=1)
    elif metric == "chebyshev":
        return np.abs(diff).max(axis=1)
    elif metric == "cosine":
        dot      = ref_v @ tgt_v
        norm_ref = np.linalg.norm(ref_v, axis=1)
        norm_tgt = np.linalg.norm(tgt_v)
        denom    = norm_ref * norm_tgt
        denom[denom == 0] = 1e-10
        return 1.0 - dot / denom
    else:
        raise ValueError(f"metric desconeguda: '{metric}'")


# ── Ponderació ────────────────────────────────────────────────────────────────

def compute_weights(distances: np.ndarray,
                    weighting: str,
                    gaussian_sigma: float = 1.0) -> np.ndarray:
    """Calcula els pesos dels K veïns a partir de les seves distàncies."""
    k = len(distances)
    zero_mask = distances == 0
    if zero_mask.any():
        w = np.zeros(k)
        w[zero_mask] = 1.0 / zero_mask.sum()
        return w

    if weighting == "uniform":
        return np.ones(k) / k
    elif weighting == "inverse":
        w = 1.0 / distances
        return w / w.sum()
    elif weighting == "softmax":
        logits = -distances
        logits -= logits.max()
        w = np.exp(logits)
        return w / w.sum()
    elif weighting == "rank":
        order = np.argsort(distances)
        ranks = np.empty(k)
        ranks[order] = np.arange(k, 0, -1, dtype=float)
        return ranks / ranks.sum()
    elif weighting == "gaussian":
        w = np.exp(-(distances ** 2) / (2 * gaussian_sigma ** 2))
        return w / w.sum() if w.sum() > 0 else np.ones(k) / k
    else:
        raise ValueError(f"weighting desconegut: '{weighting}'")


# ── Imputació d'una corba ─────────────────────────────────────────────────────

def impute_one_curve(holed: np.ndarray,
                     ref_norm: np.ndarray,
                     ref_values: np.ndarray,
                     k: int,
                     metric: str,
                     weighting: str,
                     min_overlap: int,
                     gaussian_sigma: float = 1.0) -> np.ndarray:
    """Imputa una sola corba amb els paràmetres donats."""
    result     = holed.copy()
    nan_mask   = np.isnan(holed)
    valid_mask = ~nan_mask

    if not nan_mask.any():
        return result

    target_norm = np.nan_to_num(holed, nan=0.0)

    dists = compute_distances(ref_norm, target_norm, valid_mask, metric, min_overlap)

    finite_mask = np.isfinite(dists)
    k_use = min(k, finite_mask.sum())

    if k_use == 0:
        s = pd.Series(holed)
        return s.interpolate(method="linear", limit_direction="both").values

    top_k_idx  = np.argpartition(dists, k_use)[:k_use]
    top_k_dist = dists[top_k_idx]
    weights    = compute_weights(top_k_dist, weighting, gaussian_sigma)

    for t in np.where(nan_mask)[0]:
        result[t] = float(np.dot(weights, ref_values[top_k_idx, t]))

    return result


# ── Simulació de forats ───────────────────────────────────────────────────────

def simulate_holes(arr: np.ndarray,
                   ratio: float,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    valid_idx      = np.where(~np.isnan(arr))[0]
    n_sim          = max(1, int(len(valid_idx) * ratio))
    sim_idx        = rng.choice(valid_idx, size=n_sim, replace=False)
    holed          = arr.copy()
    holed[sim_idx] = np.nan
    sim_mask       = np.zeros(len(arr), dtype=bool)
    sim_mask[sim_idx] = True
    return holed, sim_mask


# ── Funció d'avaluació ────────────────────────────────────────────────────────

def evaluate_params(params: dict,
                    X_ref: pd.DataFrame,
                    X_holed: pd.DataFrame,
                    y_train: pd.DataFrame,
                    eval_cols: list,
                    rng: np.random.Generator) -> float:
    """Avalua un conjunt de paràmetres. Retorna MAE (a minimitzar)."""
    k              = params["k"]
    metric         = params["metric"]
    weighting      = params["weighting"]
    norm_method    = params["norm_method"]
    min_overlap    = params["min_overlap"]
    gaussian_sigma = params.get("gaussian_sigma", 1.0)

    # Normalitzar la matriu de referència amb el norm_method del trial
    ref_norm   = normalize_matrix_rows(X_ref.values.T.astype(float), norm_method)
    ref_values = X_ref.values.T.astype(float)

    all_errors = []
    for col in eval_cols:
        original   = y_train[col].values.astype(float)
        holed_real = X_holed[col].values.astype(float)

        holed_sim, sim_mask = simulate_holes(holed_real, SIMULATED_HOLE_RATIO, rng)
        if sim_mask.sum() == 0:
            continue

        imputed = impute_one_curve(
            holed          = holed_sim,
            ref_norm       = ref_norm,
            ref_values     = ref_values,
            k              = k,
            metric         = metric,
            weighting      = weighting,
            min_overlap    = min_overlap,
            gaussian_sigma = gaussian_sigma,
        )
        all_errors.extend(np.abs(original[sim_mask] - imputed[sim_mask]).tolist())

    return float(np.mean(all_errors)) if all_errors else np.inf


# ── Objectiu Optuna ───────────────────────────────────────────────────────────

def make_objective(X_ref, X_holed, y_train, eval_cols, rng):
    """Retorna la funció objectiu que Optuna crida en cada trial."""
    def objective(trial: optuna.Trial) -> float:
        metric      = trial.suggest_categorical("metric",
                          ["euclidean", "cosine", "manhattan", "chebyshev"])
        norm_method = trial.suggest_categorical("norm_method",
                          ["zscore", "robust", "minmax", "none"])
        weighting   = trial.suggest_categorical("weighting",
                          ["inverse", "uniform", "softmax", "rank", "gaussian"])
        k           = trial.suggest_int("k", 1, 300)
        min_overlap = trial.suggest_int("min_overlap", 10, 300)

        gaussian_sigma = (
            trial.suggest_float("gaussian_sigma", 0.01, 10.0, log=True)
            if weighting == "gaussian" else 1.0
        )

        return evaluate_params(
            params={
                "k": k, "metric": metric, "weighting": weighting,
                "norm_method": norm_method, "min_overlap": min_overlap,
                "gaussian_sigma": gaussian_sigma,
            },
            X_ref=X_ref, X_holed=X_holed, y_train=y_train,
            eval_cols=eval_cols, rng=rng,
        )
    return objective


# ── Visualitzacions ───────────────────────────────────────────────────────────

def plot_results(study: optuna.Study, df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Bayesian Optimization — KNN Imputation (Optuna)",
                 fontsize=13, fontweight="bold")

    # 1. Convergència
    ax = axes[0, 0]
    maes = df["mae"].values
    ax.plot(maes, color="steelblue", alpha=0.4, lw=0.8, label="MAE trial")
    ax.plot(np.minimum.accumulate(maes), color="tomato", lw=2,
            label="Millor acumulat")
    ax.set_title("Convergència dels trials")
    ax.set_xlabel("Trial")
    ax.set_ylabel("MAE (W)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. Importància dels hiperparàmetres
    ax = axes[0, 1]
    try:
        importances = optuna.importance.get_param_importances(study)
        ax.barh(list(importances.keys()), list(importances.values()),
                color="steelblue", alpha=0.85)
        ax.set_title("Importància dels hiperparàmetres")
        ax.set_xlabel("Importància relativa")
        ax.grid(True, alpha=0.3, axis="x")
    except Exception:
        ax.text(0.5, 0.5, "No disponible\n(calen ≥20 trials completats)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title("Importància dels hiperparàmetres")

    # 3. MAE per categoria
    ax = axes[1, 0]
    palette = {"metric": "steelblue", "weighting": "seagreen",
               "norm_method": "darkorange"}
    x_pos, xticks, xlabels = 0, [], []
    for cat, color in palette.items():
        if cat not in df.columns:
            continue
        for val, mae in df.groupby(cat)["mae"].mean().sort_values().items():
            ax.bar(x_pos, mae, color=color, alpha=0.82, width=0.6)
            xticks.append(x_pos)
            xlabels.append(f"{val}\n({cat.replace('_method','')})")
            x_pos += 1
        x_pos += 0.6
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7.5)
    ax.set_title("MAE per categoria")
    ax.set_ylabel("MAE mitjà (W)")
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Top 10
    ax = axes[1, 1]
    top10 = df.nsmallest(10, "mae").copy()
    top10["label"] = top10.apply(
        lambda r: (f"K={int(r.k)} | {r.metric[:4]} | "
                   f"{r.weighting[:4]} | {r.norm_method[:3]} | "
                   f"ov={int(r.min_overlap)}"), axis=1)
    ax.barh(range(len(top10)), top10["mae"].values, color="steelblue", alpha=0.85)
    ax.set_yticks(range(len(top10)))
    ax.set_yticklabels(top10["label"].values, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_title("Top 10 combinacions")
    ax.set_xlabel("MAE (W)")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "hyperparam_results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Gràfic guardat a: {path}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n── Carregant dades ──")
    X_ref, X_holed, y_train = load_train(X_TRAIN_PATH, Y_TRAIN_PATH)

    rng = np.random.default_rng(RANDOM_SEED)
    eval_cols = rng.choice(
        X_holed.columns.tolist(),
        size=min(N_EVAL_CURVES, X_holed.shape[1]),
        replace=False
    ).tolist()

    print(f"Corbes d'avaluació : {len(eval_cols)}")
    print(f"Forats simulats    : {SIMULATED_HOLE_RATIO*100:.0f}% dels punts coneguts")
    print(f"Trials Bayesians   : {N_TRIALS}\n")

    # Carregar o crear l'estudi (permet reprendre si s'interromp)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if STUDY_PATH.exists():
        print(f"Reprenent estudi existent: {STUDY_PATH}")
        with open(STUDY_PATH, "rb") as f:
            study = pickle.load(f)
        n_done      = len(study.trials)
        n_remaining = max(0, N_TRIALS - n_done)
        print(f"Trials ja fets: {n_done}  |  Restants: {n_remaining}\n")
    else:
        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
        )
        n_remaining = N_TRIALS

    objective = make_objective(X_ref, X_holed, y_train, eval_cols, rng)

    # Bucle amb progrés manual i guardats intermedis
    t_start = time.time()
    for _ in range(n_remaining):
        study.optimize(objective, n_trials=1, show_progress_bar=False)
        trial_n = len(study.trials)
        elapsed = (time.time() - t_start) / 60
        print(
            f"  Trial {trial_n:>3}/{N_TRIALS}  |  "
            f"MAE actual: {study.trials[-1].value:.2f} W  |  "
            f"Millor: {study.best_value:.2f} W  |  "
            f"{elapsed:.1f} min",
            end="\r"
        )
        if trial_n % 10 == 0:
            with open(STUDY_PATH, "wb") as f:
                pickle.dump(study, f)
    print()

    # Guardar estudi final
    with open(STUDY_PATH, "wb") as f:
        pickle.dump(study, f)

    # Resultats
    total_min = (time.time() - t_start) / 60
    results_df = pd.DataFrame([
        {**t.params, "mae": t.value, "trial": t.number}
        for t in study.trials if t.value is not None
    ]).sort_values("mae").reset_index(drop=True)

    csv_path = OUT_DIR / "hyperparam_results.csv"
    results_df.to_csv(csv_path, index=False)

    best = study.best_params
    print(f"\n── Optimització completada en {total_min:.1f} min ──")
    print(f"Millor MAE estimat : {study.best_value:.2f} W\n")
    print("Millors paràmetres (copia a src/config.py):")
    print(f"  K_NEIGHBORS     = {best['k']}")
    print(f"  DISTANCE_METRIC = \"{best['metric']}\"")
    print(f"  WEIGHTING       = \"{best['weighting']}\"")
    print(f"  NORM_METHOD     = \"{best['norm_method']}\"")
    print(f"  MIN_OVERLAP     = {best['min_overlap']}")
    if best.get("weighting") == "gaussian":
        print(f"  GAUSSIAN_SIGMA  = {best.get('gaussian_sigma', 1.0):.4f}")
    print(f"\nResultats: {csv_path}")
    print(f"Estudi:    {STUDY_PATH}")

    plot_results(study, results_df)


if __name__ == "__main__":
    main()
