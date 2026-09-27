"""
lstm/src/trainer.py — v4
------------------------
Entrenament i inferència del Transformer.

v4:
  - Pèrdua MAE ponderada per l'escala de cada corba (≈ MAE en watts)
  - Selecció del millor model per MAE real (W) sobre les corbes holed_ etiquetades
  - Es conserven els N_SNAPSHOTS millors checkpoints (snap_eXXX.pt) per fer-ne
    la mitjana a la predicció
  - Checkpoint complet (latest.pt) cada època per reprendre l'entrenament
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lstm.src.config import (
    BATCH_SIZE, N_EPOCHS, LR, LR_WARMUP_STEPS, WEIGHT_DECAY, GRAD_CLIP,
    PATIENCE, RANDOM_SEED, NUM_WORKERS, N_SNAPSHOTS,
)
from lstm.src.dataset import generate_mask, worker_init_fn
from lstm.src.model import TemporalMaskedAutoencoder, count_parameters


# ── Pèrdua ────────────────────────────────────────────────────────────────────

def masked_mae_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                    weight: torch.Tensor | None = None) -> torch.Tensor:
    """MAE sobre les posicions emmascarades; weight (B,) pondera cada corba."""
    m   = mask.float()
    err = (pred - target).abs() * m
    if weight is not None:
        err = err * weight[:, None]
    return err.sum() / m.sum().clamp(min=1)


# ── Scheduler ─────────────────────────────────────────────────────────────────

class WarmupCosineScheduler:
    """Warmup lineal + cosine annealing."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, lr: float):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.lr           = lr
        self.step_count   = 0

    def step(self):
        self.step_count += 1
        s = self.step_count
        if s <= self.warmup_steps:
            lr = self.lr * s / self.warmup_steps
        else:
            progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = max(self.lr * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0))), self.lr / 100)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def state_dict(self) -> dict:
        return {"warmup_steps": self.warmup_steps, "total_steps": self.total_steps,
                "lr": self.lr, "step_count": self.step_count}

    def load_state_dict(self, state: dict) -> None:
        self.warmup_steps = int(state["warmup_steps"])
        self.lr           = float(state["lr"])
        self.step_count   = int(state["step_count"])


def build_fixed_masks(n: int, t: int, mask_dist: dict | None, seed: int) -> np.ndarray:
    """Màscares (n, t) fixes per a la validació proxy."""
    rng = np.random.default_rng(seed)
    return np.stack([generate_mask(t, rng, mask_dist) for _ in range(n)])


# ── Checkpoints ───────────────────────────────────────────────────────────────

def is_full_training_checkpoint(obj) -> bool:
    return (isinstance(obj, dict) and "model_state_dict" in obj
            and "optimizer_state_dict" in obj and "scheduler_state_dict" in obj
            and "epoch" in obj)


def load_model_weights(path: str | Path, model: nn.Module, device: torch.device) -> None:
    """Carrega pesos d'un state_dict o d'un checkpoint complet."""
    obj = torch.load(Path(path), map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model_state_dict" in obj:
        obj = obj["model_state_dict"]
    model.load_state_dict(obj)


def load_training_checkpoint(path: str | Path, device: torch.device) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No s'ha trobat el checkpoint: {path}")
    obj = torch.load(path, map_location=device, weights_only=False)
    if not is_full_training_checkpoint(obj):
        raise ValueError(f"'{path}' no és un checkpoint reanudable (només pesos).")
    return obj


def _atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _capture_rng_state() -> dict:
    state = {"python_random": random.getstate(), "numpy_random": np.random.get_state(),
             "torch_random": torch.get_rng_state(), "cuda_random": None}
    if torch.cuda.is_available():
        state["cuda_random"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_random"])
    if state.get("cuda_random") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda_random"])
        except Exception as exc:
            print(f"  Avís: no s'ha pogut restaurar RNG CUDA ({exc})")


def _write_json(path: Path, payload) -> None:
    """JSON auxiliar: mai ha d'aturar l'entrenament."""
    def _default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"{type(obj).__name__} no serialitzable")
    try:
        path.write_text(json.dumps(payload, indent=2, default=_default), encoding="utf-8")
    except Exception as exc:
        print(f"  Avís: no s'ha pogut escriure {path.name}: {exc}")


def _update_snapshots(snapshots: list, model: nn.Module, epoch: int, mae: float,
                      run_dir: Path, k: int) -> bool:
    """Manté els k checkpoints amb millor MAE real. Retorna True si s'ha desat."""
    if len(snapshots) >= k and mae >= max(s["mae"] for s in snapshots):
        return False
    fname = f"snap_e{epoch:03d}.pt"
    _atomic_torch_save(model.state_dict(), run_dir / fname)
    snapshots.append({"epoch": epoch, "mae": float(mae), "file": fname})
    snapshots.sort(key=lambda s: s["mae"])
    for s in snapshots[k:]:
        (run_dir / s["file"]).unlink(missing_ok=True)
    del snapshots[k:]
    _write_json(run_dir / "snapshots.json", snapshots)
    return True


# ── Inferència ────────────────────────────────────────────────────────────────

def make_loader(dataset, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      worker_init_fn=worker_init_fn if num_workers > 0 else None,
                      persistent_workers=False)


def predict(model: TemporalMaskedAutoencoder, loader: DataLoader,
            device: torch.device) -> np.ndarray:
    """Prediccions normalitzades (N, T)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch["input"].to(device), batch["mask"].to(device),
                               batch["profile"].to(device)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_ensemble(specs: list, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Mitjana de les prediccions de diversos pesos. specs: [(model_kwargs, path), ...]"""
    total = None
    for kwargs, path in specs:
        model = TemporalMaskedAutoencoder(**kwargs).to(device)
        load_model_weights(path, model, device)
        p = predict(model, loader, device)
        total = p if total is None else total + p
    return total / len(specs)


def real_val_mae(preds_norm: np.ndarray, real_val: dict) -> float:
    """MAE en watts sobre els forats reals amb ground truth conegut."""
    raw = np.clip(preds_norm * real_val["mads"][:, None] + real_val["meds"][:, None], 0, None)
    m = real_val["nanmask"] & ~np.isnan(real_val["y_raw"])
    if not m.any():
        return float("nan")
    return float(np.abs(raw - real_val["y_raw"])[m].mean())


# ── Entrenament ───────────────────────────────────────────────────────────────

def _run_epoch_loss(model, loader, device, optimizer=None, scheduler=None) -> float:
    training = optimizer is not None
    model.train(training)
    losses = []
    with torch.set_grad_enabled(training):
        for batch in loader:
            inp    = batch["input"].to(device, non_blocking=True)
            tgt    = batch["target"].to(device, non_blocking=True)
            mask   = batch["mask"].to(device, non_blocking=True)
            prof   = batch["profile"].to(device, non_blocking=True)
            weight = batch["weight"].to(device, non_blocking=True)
            loss = masked_mae_loss(model(inp, mask, prof), tgt, mask, weight)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if GRAD_CLIP > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                scheduler.step()
            losses.append(loss.item())
    return float(np.mean(losses))


def train(train_ds, val_ds, real_val: dict | None, device: torch.device,
          run_dir: Path, model_kwargs: dict,
          n_epochs: int | None = None, resume_from: str | Path | None = None,
          seed: int = RANDOM_SEED) -> tuple[TemporalMaskedAutoencoder, list]:
    """
    train_ds / val_ds : TimeSeriesDataset (validació proxy amb màscares fixes)
    real_val          : {"dataset", "meds", "mads", "nanmask", "y_raw"} — corbes
                        holed_ etiquetades; decideixen el millor model. Sense
                        això, es fa servir la validació proxy.
    run_dir           : on es desen latest.pt, best_model.pt, snap_eXXX.pt
    Retorna (model amb els millors pesos, llista de snapshots).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    n_epochs    = n_epochs or N_EPOCHS
    latest_path = run_dir / "latest.pt"
    best_path   = run_dir / "best_model.pt"

    train_loader = make_loader(train_ds, shuffle=True, num_workers=NUM_WORKERS)
    val_loader   = make_loader(val_ds, num_workers=NUM_WORKERS)
    real_loader  = make_loader(real_val["dataset"]) if real_val is not None else None

    model = TemporalMaskedAutoencoder(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = WarmupCosineScheduler(optimizer, LR_WARMUP_STEPS,
                                      n_epochs * len(train_loader), LR)

    print(f"\nTrain: {len(train_ds):,} corbes | Val proxy: {len(val_ds):,} | "
          f"Val real: {len(real_val['dataset']) if real_val else 0:,}")
    print(f"Batches per època: {len(train_loader)} | workers: {NUM_WORKERS}")
    print(f"Paràmetres entrenables: {count_parameters(model):,}")
    print(f"Directori: {run_dir}")

    best, patience_left, start_epoch, snapshots = float("inf"), PATIENCE, 1, []
    if resume_from is not None:
        ckpt = load_training_checkpoint(resume_from, device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best          = float(ckpt["best_val_loss"])
        patience_left = int(ckpt["patience_left"])
        start_epoch   = int(ckpt["epoch"]) + 1
        snapshots     = list(ckpt.get("snapshots", []))
        _restore_rng_state(ckpt.get("rng_state"))
        print(f"\n── Reprenent des de {resume_from}: època {start_epoch}, "
              f"millor={best:.4f}, paciència={patience_left} ──")

    print(f"\n{'Època':>6} {'Train':>9} {'ValProxy':>9} {'MAE real(W)':>12} "
          f"{'LR':>9} {'Temps':>7}")
    print("─" * 70)

    for epoch in range(start_epoch, n_epochs + 1):
        t0 = time.time()
        train_loss = _run_epoch_loss(model, train_loader, device, optimizer, scheduler)
        val_loss   = _run_epoch_loss(model, val_loader, device)
        real_mae   = real_val_mae(predict(model, real_loader, device), real_val) \
            if real_val is not None else None
        monitor = real_mae if real_mae is not None else val_loss

        improved = bool(monitor < best)
        if improved:
            best, patience_left = monitor, PATIENCE
            _atomic_torch_save(model.state_dict(), best_path)
            status = "✓ millor"
        else:
            patience_left -= 1
            status = f"({patience_left} left)"
        if _update_snapshots(snapshots, model, epoch, monitor, run_dir, N_SNAPSHOTS) \
                and not improved:
            status += " +snap"

        _atomic_torch_save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best,
            "patience_left": patience_left,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "real_val_mae": real_mae,
            "snapshots": snapshots,
            "rng_state": _capture_rng_state(),
        }, latest_path)
        _write_json(run_dir / "training_status.json", {
            "last_completed_epoch": epoch, "n_epochs": n_epochs,
            "best_real_mae": best, "patience_left": patience_left,
            "train_loss": train_loss, "val_proxy": val_loss, "real_mae": real_mae,
            "snapshots": snapshots,
        })

        real_str = f"{real_mae:12.2f}" if real_mae is not None else f"{'-':>12}"
        print(f"{epoch:>6} {train_loss:>9.4f} {val_loss:>9.4f} {real_str} "
              f"{optimizer.param_groups[0]['lr']:>9.2e} {time.time() - t0:>6.0f}s  {status}",
              flush=True)

        if patience_left <= 0:
            print(f"\nEarly stopping a l'època {epoch}.")
            break

    print(f"\nMillor MAE real: {best:.2f} W | snapshots: "
          f"{[(s['epoch'], round(s['mae'], 2)) for s in snapshots]}")
    if best_path.exists():
        load_model_weights(best_path, model, device)
    return model, snapshots
