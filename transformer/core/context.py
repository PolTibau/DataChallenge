"""
transformer/core/context.py
---------------------------
Informació de context que acompanya cada corba, instant a instant.

A més del valor de la corba, el model rep per a cada mitja hora:

  canal  contingut
  -----  ------------------------------------------------------------------
    0    valor de la corba (normalitzat; 0 on falta)
    1    màscara (1 = valor desconegut)
    2    interpolació lineal entre els punts coneguts
   3-10  mateixa hora el dia abans / dia després / setmana abans / setmana
         després (valor + indicador de si és conegut)
   11    perfil habitual del consumidor a aquesta franja (laborable o festiu)
   12    corba de població (mitjana de totes les referències), estandarditzada
   13    anomalia de la població respecte al seu perfil habitual ("fa fred?")
   14    mitjana dels consumidors semblants (normalitzada)
   15    anomalia dels consumidors semblants respecte al seu perfil habitual

Tot (excepte població) està en les unitats normalitzades de la corba
(mediana / MAD), igual que el valor.
"""

import warnings

import numpy as np
import torch

from transformer.core.config import SEQ_LEN, STEPS_PER_DAY, START_WEEKDAY, N_NEIGHBOURS

N_CHANNELS  = 16
N_DAYS_FULL = SEQ_LEN // STEPS_PER_DAY          # 22 dies complets (+1 punt final)
LAGS        = (-STEPS_PER_DAY, STEPS_PER_DAY, -7 * STEPS_PER_DAY, 7 * STEPS_PER_DAY)
LEVEL_WEIGHT = 3.0                              # pes del nivell de consum a la cerca de veïns

_t          = np.arange(SEQ_LEN)
SLOT        = _t % STEPS_PER_DAY                               # franja de mitja hora 0..47
WEEKEND     = ((_t // STEPS_PER_DAY + START_WEEKDAY) % 7) >= 5
WEEKEND_INT = WEEKEND.astype(np.int64)
DAY_IS_WEEKEND = ((np.arange(N_DAYS_FULL) + START_WEEKDAY) % 7) >= 5


# ── Perfil habitual ───────────────────────────────────────────────────────────

def typical_profiles(values: np.ndarray) -> np.ndarray:
    """
    (N, T) amb NaN on falta → (N, 2, 48): mediana de cada franja horària,
    [0] = dies laborables, [1] = cap de setmana. Només usa punts coneguts.
    Franges sense cap dada: s'omplen amb l'altre tipus de dia, i si tampoc
    n'hi ha, amb la mediana global de la corba.
    """
    n = len(values)
    v = values[:, :N_DAYS_FULL * STEPS_PER_DAY].reshape(n, N_DAYS_FULL, STEPS_PER_DAY)
    out = np.empty((n, 2, STEPS_PER_DAY), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out[:, 0] = np.nanmedian(v[:, ~DAY_IS_WEEKEND], axis=1)
        out[:, 1] = np.nanmedian(v[:, DAY_IS_WEEKEND], axis=1)
        out = np.where(np.isnan(out), out[:, ::-1], out)
        overall = np.nanmedian(values, axis=1)
    out = np.where(np.isnan(out), overall[:, None, None], out)
    return np.nan_to_num(out, nan=0.0).astype(np.float32)


def expand_profiles(profiles: np.ndarray) -> np.ndarray:
    """(N, 2, 48) → (N, T): el valor habitual de cada instant segons franja i tipus de dia."""
    return profiles[:, WEEKEND_INT, SLOT]


# ── Població ──────────────────────────────────────────────────────────────────

def build_population(refs_raw: np.ndarray) -> dict:
    """
    Corba mitjana de totes les referències (completes). Captura el que afecta
    tothom alhora, sobretot la temperatura: el consum mitjà diari va de ~280 W
    a ~406 W dins del període.
    """
    pop = refs_raw.mean(axis=0).astype(np.float64)
    typ = typical_profiles(pop[None].astype(np.float32))[0]
    std = pop.std() + 1e-6
    daily = pop[:N_DAYS_FULL * STEPS_PER_DAY].reshape(N_DAYS_FULL, STEPS_PER_DAY).mean(axis=1)
    return {
        "pop_z":    ((pop - pop.mean()) / std).astype(np.float32),
        "pop_anom": ((pop - typ[WEEKEND_INT, SLOT]) / std).astype(np.float32),
        "daily":    daily,
    }


# ── Consumidors semblants ─────────────────────────────────────────────────────

def descriptors(values: np.ndarray) -> np.ndarray:
    """
    (N, T) amb NaN → (N, 97): forma del perfil habitual (dividit pel nivell)
    + log del nivell. Resumeix "quin tipus de consumidor és" i és robust als
    forats, a diferència de comparar punt a punt.
    """
    prof = typical_profiles(values).reshape(len(values), -1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        level = np.nan_to_num(np.nanmean(values, axis=1), nan=0.0)
    level = np.clip(level, 0, None)
    shape = prof / (level[:, None] + 1.0)
    return np.concatenate(
        [shape, LEVEL_WEIGHT * np.log1p(level)[:, None]], axis=1).astype(np.float32)


def find_neighbours(query: np.ndarray, pool: np.ndarray,
                    k: int = N_NEIGHBOURS, exclude_self: bool = False,
                    device: torch.device | None = None,
                    chunk: int = 2048) -> np.ndarray:
    """
    Índexs (Nq, k) dels k descriptors del pool més propers a cada consulta.
    exclude_self=True quan query i pool són el mateix conjunt: una corba no
    pot ser veïna de si mateixa (seria una fuita del valor que cal predir).
    """
    device = device or torch.device("cpu")
    P  = torch.from_numpy(pool).to(device)
    p2 = (P * P).sum(dim=1)
    out = np.empty((len(query), k), dtype=np.int64)
    for s in range(0, len(query), chunk):
        Q = torch.from_numpy(query[s:s + chunk]).to(device)
        d = (Q * Q).sum(dim=1, keepdim=True) + p2[None, :] - 2.0 * (Q @ P.T)
        if exclude_self:
            rows = torch.arange(len(Q), device=device)
            d[rows, rows + s] = float("inf")
        out[s:s + len(Q)] = torch.topk(d, k, dim=1, largest=False).indices.cpu().numpy()
    return out


def neighbour_channels(nb_idx: np.ndarray, pool_norm: np.ndarray,
                       pool_anom: np.ndarray,
                       device: torch.device | None = None,
                       chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Mitjana dels veïns (valor normalitzat i anomalia) a cada instant → (N, T) ×2."""
    device = device or torch.device("cpu")
    Pn = torch.from_numpy(pool_norm).to(device)
    Pa = torch.from_numpy(pool_anom).to(device)
    n, T = len(nb_idx), pool_norm.shape[1]
    nb_norm = np.empty((n, T), dtype=np.float32)
    nb_anom = np.empty((n, T), dtype=np.float32)
    for s in range(0, n, chunk):
        idx = torch.from_numpy(nb_idx[s:s + chunk]).to(device)
        nb_norm[s:s + len(idx)] = Pn[idx].mean(dim=1).cpu().numpy()
        nb_anom[s:s + len(idx)] = Pa[idx].mean(dim=1).cpu().numpy()
    del Pn, Pa
    return nb_norm, nb_anom


# ── Entrades per mostra ───────────────────────────────────────────────────────

def sample_features(x: np.ndarray, mask: np.ndarray, population: dict,
                    nb_norm: np.ndarray, nb_anom: np.ndarray) -> np.ndarray:
    """
    x    : (T,) corba normalitzada (els valors on mask=True s'ignoren)
    mask : (T,) bool, True = desconegut
    Retorna (T, N_CHANNELS) float32. Només fa servir punts coneguts de la
    pròpia corba, així que és igual a l'entrenament que al test.
    """
    T   = len(x)
    obs = ~mask
    feats = np.zeros((T, N_CHANNELS), dtype=np.float32)
    feats[obs, 0] = x[obs]
    feats[:, 1]   = mask

    idx = np.flatnonzero(obs)
    if len(idx) > 0:
        feats[:, 2] = np.interp(np.arange(T), idx, x[idx])

    c = 3
    for lag in LAGS:
        src = np.arange(T) + lag
        inb = (src >= 0) & (src < T)
        ok  = np.zeros(T, dtype=bool)
        ok[inb] = obs[src[inb]]
        feats[ok, c]    = x[src[ok]]
        feats[:, c + 1] = ok
        c += 2

    xn = np.where(obs, x, np.nan).astype(np.float32)[None]
    feats[:, 11] = expand_profiles(typical_profiles(xn))[0]
    feats[:, 12] = population["pop_z"]
    feats[:, 13] = population["pop_anom"]
    feats[:, 14] = nb_norm
    feats[:, 15] = nb_anom
    return feats
