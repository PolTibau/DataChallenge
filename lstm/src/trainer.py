"""
lstm/src/trainer.py — v3.1
--------------------------
Entrenament i inferència del Transformer.

v3.1:
  - Pèrdua MAE per defecte (alineada amb la mètrica del challenge)
  - Màscares de validació congelades (early stopping estable)
  - TTA per soroll als punts observats (no shifts cíclics)
  - Checkpoints complets (latest.pt / best.pt) per reanudar entrenament
"""

import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader, random_split

from lstm.src.config import (
    BATCH_SIZE, N_EPOCHS, LR, LR_WARMUP_STEPS,
    WEIGHT_DECAY, GRAD_CLIP, PATIENCE, VAL_SPLIT,
    BEST_CKPT, RANDOM_SEED, CKPT_DIR,
    MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN, MASK_BLOCK_PROB,
    EMPIRICAL_MASK_PROB,
    TTA_N_PASSES, TTA_NOISE_STD,
)
from lstm.src.dataset import TimeSeriesDataset, generate_mask
from lstm.src.model   import TemporalMaskedAutoencoder, count_parameters


# ── Pèrdues ───────────────────────────────────────────────────────────────────

def masked_mae_loss(pred: torch.Tensor,
                    target: torch.Tensor,
                    mask: torch.Tensor) -> torch.Tensor:
    """MAE calculat NOMÉS sobre les posicions emmascarades."""
    errors = torch.abs(pred - target) * mask.float()
    return errors.sum() / mask.float().sum().clamp(min=1)


def masked_hybrid_loss(pred: torch.Tensor,
                       target: torch.Tensor,
                       mask: torch.Tensor,
                       alpha: float = 0.7) -> torch.Tensor:
    """Pèrdua híbrida opcional: alpha × MAE + (1-alpha) × RMSE."""
    diff = pred - target
    mae  = (torch.abs(diff) * mask.float()).sum() / mask.float().sum().clamp(min=1)
    mse  = ((diff ** 2)     * mask.float()).sum() / mask.float().sum().clamp(min=1)
    return alpha * mae + (1 - alpha) * torch.sqrt(mse)


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
            lr = max(self.lr * 0.5 * (1 + np.cos(np.pi * progress)), self.lr / 100)
        for g in self.optimizer.param_groups:
            g["lr"] = lr
        return lr

    def state_dict(self) -> dict:
        return {
            "warmup_steps": self.warmup_steps,
            "total_steps":  self.total_steps,
            "lr":           self.lr,
            "step_count":   self.step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restaura el punt del schedule. Conserva total_steps actual
        (n_epochs d'aquesta execució) perquè el cosine no es tanqui abans d'hora."""
        self.warmup_steps = int(state["warmup_steps"])
        self.lr           = float(state["lr"])
        self.step_count   = int(state["step_count"])
        # total_steps es queda el de __init__ (n_epochs * batches d'aquesta sessió)


def build_fixed_masks(n: int, t: int, mask_dist: dict | None,
                      seed: int) -> np.ndarray:
    """Genera un conjunt fix de màscares (N, T) per a validació estable."""
    rng = np.random.default_rng(seed)
    masks = np.zeros((n, t), dtype=bool)
    for i in range(n):
        masks[i] = generate_mask(
            t, MASK_RATIO, MASK_MIN_LEN, MASK_MAX_LEN, MASK_BLOCK_PROB,
            rng, mask_dist, EMPIRICAL_MASK_PROB,
        )
    return masks


# ── Checkpoints ───────────────────────────────────────────────────────────────

def is_full_training_checkpoint(obj) -> bool:
    """True si el fitxer és un checkpoint reanudable (no només pesos)."""
    return (
        isinstance(obj, dict)
        and "model_state_dict" in obj
        and "optimizer_state_dict" in obj
        and "scheduler_state_dict" in obj
        and "epoch" in obj
    )


def load_model_weights(path: str | Path,
                       model: nn.Module,
                       device: torch.device) -> None:
    """
    Carrega pesos tant d'un state_dict clàssic com d'un checkpoint complet.

    best_model_v3.pt → OrderedDict de pesos
    latest.pt / best.pt → dict amb model_state_dict
    """
    path = Path(path)
    try:
        obj = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        obj = torch.load(path, map_location=device, weights_only=False)

    if is_full_training_checkpoint(obj) or (
            isinstance(obj, dict) and "model_state_dict" in obj):
        model.load_state_dict(obj["model_state_dict"])
    else:
        model.load_state_dict(obj)


def load_training_checkpoint(path: str | Path, device: torch.device) -> dict:
    """Carrega un checkpoint complet. Falla amb missatge clar si és antic."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No s'ha trobat el checkpoint: {path}")

    try:
        obj = torch.load(path, map_location=device, weights_only=False)
    except Exception as exc:
        raise RuntimeError(
            f"No s'ha pogut llegir '{path}'. Fitxer corrupte o incompatible.\n{exc}"
        ) from exc

    if not is_full_training_checkpoint(obj):
        raise ValueError(
            f"'{path}' és un checkpoint antic / no reanudable: només conté "
            "els pesos del model (state_dict), sense optimizer ni scheduler.\n"
            "Per reanudar cal un latest.pt (o best.pt) generat amb el trainer v3.1+.\n"
            "Aquest fitxer sí es pot usar per predicció amb load_model_weights()."
        )
    return obj


def _atomic_torch_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _capture_rng_state() -> dict:
    state = {
        "python_random": random.getstate(),
        "numpy_random":  np.random.get_state(),
        "torch_random":  torch.get_rng_state(),
        "cuda_random":   None,
    }
    if torch.cuda.is_available():
        state["cuda_random"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict | None) -> None:
    if not state:
        return
    if state.get("python_random") is not None:
        random.setstate(state["python_random"])
    if state.get("numpy_random") is not None:
        np.random.set_state(state["numpy_random"])
    if state.get("torch_random") is not None:
        torch.set_rng_state(state["torch_random"])
    if state.get("cuda_random") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["cuda_random"])
        except Exception as exc:
            print(f"  Avís: no s'ha pogut restaurar RNG CUDA ({exc})")


def _write_status(ckpt_dir: Path, payload: dict) -> None:
    """
    Escriu l'estat del training sense permetre que un tipus NumPy
    faci fallar l'entrenament. Els valors NumPy (np.bool_, np.integer,
    np.floating) es converteixen a tipus natius de Python abans del JSON.
    """
    path = ckpt_dir / "training_status.json"

    def _json_default(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(
            f"Object of type {obj.__class__.__name__} is not JSON serializable"
        )

    try:
        path.write_text(
            json.dumps(payload, indent=2, default=_json_default),
            encoding="utf-8",
        )
    except Exception as exc:
        # El JSON és només informació auxiliar: mai ha d'aturar el training.
        print(f"  Avís: no s'ha pogut escriure training_status.json: {exc}")


# ── Entrenament ───────────────────────────────────────────────────────────────

def train(curves_norm: np.ndarray,
          device: torch.device,
          n_epochs_override: int | None = None,
          ckpt_path: Path | None = None,
          use_hybrid_loss: bool = False,
          curves_raw: np.ndarray | None = None,
          global_stats: dict | None = None,
          mask_dist: dict | None = None,
          resume_from: str | Path | None = None,
          ckpt_dir: str | Path | None = None) -> TemporalMaskedAutoencoder:
    """
    Entrena el Transformer.

    use_hybrid_loss=False per defecte: la mètrica del challenge és MAE en watts.

    Checkpoints
    -----------
    ckpt_path : pesos del millor model per predicció (best_model_v3.pt, state_dict)
    ckpt_dir  : directori de latest.pt / best.pt (checkpoints complets)
    resume_from : ruta a latest.pt (o best.pt) per continuar; None = des de zero
    """
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    n_epochs  = n_epochs_override if n_epochs_override is not None else N_EPOCHS
    best_weights_path = Path(ckpt_path) if ckpt_path is not None else BEST_CKPT
    if ckpt_dir is None:
        ckpt_dir_path = best_weights_path.parent if best_weights_path.parent else CKPT_DIR
    else:
        ckpt_dir_path = Path(ckpt_dir)
    ckpt_dir_path.mkdir(parents=True, exist_ok=True)
    best_weights_path.parent.mkdir(parents=True, exist_ok=True)

    latest_path = ckpt_dir_path / "latest.pt"
    best_full_path = ckpt_dir_path / "best.pt"

    loss_fn   = masked_hybrid_loss if use_hybrid_loss else masked_mae_loss
    loss_name = "MAE+RMSE" if use_hybrid_loss else "MAE"

    # DataLoaders
    print("  Building training dataset (this includes profile computation)...")
    full_ds = TimeSeriesDataset(curves_norm, augment=True, seed=RANDOM_SEED,
                                curves_raw=curves_raw, global_stats=global_stats,
                                mask_dist=mask_dist)
    n_val   = max(1, int(len(full_ds) * VAL_SPLIT))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(RANDOM_SEED))

    # Validació: màscares FIXES (mateixes cada època → early stopping fiable)
    print("  Building validation dataset with frozen masks...")
    fixed_val_masks = build_fixed_masks(
        len(curves_norm), curves_norm.shape[1], mask_dist, seed=RANDOM_SEED + 7)
    val_full = TimeSeriesDataset(
        curves_norm, augment=False, seed=RANDOM_SEED + 1,
        curves_raw=curves_raw, global_stats=global_stats,
        mask_dist=mask_dist, fixed_masks=fixed_val_masks)
    val_ds.dataset = val_full

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0, pin_memory=True)

    print(f"\nTrain: {n_train:,} corbes  |  Val: {n_val:,} corbes")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"Loss: {loss_name}")
    print(f"Mask distribution: {'empirical (60%) + uniform (40%)' if mask_dist else 'uniform'}")
    print("Val masks: FROZEN (reproducible early stopping)")
    print(f"ckpt_dir: {ckpt_dir_path}")
    print(f"best weights (predicció): {best_weights_path}")

    # Model
    model = TemporalMaskedAutoencoder().to(device)
    n_params = count_parameters(model)
    print(f"Paràmetres entrenables: {n_params:,}  ({n_params/1e6:.1f}M)")

    optimizer   = torch.optim.AdamW(model.parameters(), lr=LR,
                                    weight_decay=WEIGHT_DECAY)
    total_steps = n_epochs * len(train_loader)
    scheduler   = WarmupCosineScheduler(optimizer, LR_WARMUP_STEPS, total_steps, LR)

    best_val_mae  = float("inf")
    patience_left = PATIENCE
    start_epoch   = 1
    last_completed_epoch = 0

    if resume_from is not None:
        resume_path = Path(resume_from)
        print(f"\n── Reanudant des de: {resume_path} ──")
        ckpt = load_training_checkpoint(resume_path, device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_mae = float(ckpt["best_val_loss"])
        if "patience_left" in ckpt:
            patience_left = int(ckpt["patience_left"])
        else:
            patience_left = PATIENCE - int(ckpt.get("patience_counter", 0))
        last_completed_epoch = int(ckpt["epoch"])
        start_epoch = last_completed_epoch + 1

        ds_rng = ckpt.get("dataset_rng_state")
        if ds_rng is not None:
            full_ds.rng.bit_generator.state = ds_rng

        _restore_rng_state(ckpt.get("rng_state"))

        print(f"  checkpoint epoch     : {last_completed_epoch}")
        print(f"  start_epoch          : {start_epoch}")
        print(f"  best_val_loss        : {best_val_mae:.6f}")
        print(f"  patience_left        : {patience_left}")
        print(f"  scheduler.step_count : {scheduler.step_count}")
        print(f"  optimizer lr         : {optimizer.param_groups[0]['lr']:.2e}")

        if start_epoch > n_epochs:
            print(f"\nEl checkpoint ja ha completat {last_completed_epoch}/{n_epochs} "
                  "èpoques. No cal entrenar més.")
            if best_weights_path.exists():
                load_model_weights(best_weights_path, model, device)
            return model

    print(f"\n{'Epoch':>6} {'Train Loss':>11} {'Val Loss':>10} "
          f"{'LR':>10} {'Time':>7} {'':>10}")
    print("─" * 64)

    for epoch in range(start_epoch, n_epochs + 1):
        t0 = time.time()

        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            inp, tgt, mask = (batch["input"].to(device),
                              batch["target"].to(device),
                              batch["mask"].to(device))
            profile = batch.get("profile")
            if profile is not None:
                profile = profile.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(inp, mask, profile), tgt, mask)
            loss.backward()
            if GRAD_CLIP > 0:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # Validació
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                inp, tgt, mask = (batch["input"].to(device),
                                  batch["target"].to(device),
                                  batch["mask"].to(device))
                profile = batch.get("profile")
                if profile is not None:
                    profile = profile.to(device)
                val_losses.append(loss_fn(model(inp, mask, profile), tgt, mask).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        lr_now     = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        improved = bool(val_loss < best_val_mae)
        if improved:
            best_val_mae  = val_loss
            patience_left = PATIENCE
            status = "✓ millor"
        else:
            patience_left -= 1
            status = f"({patience_left} left)"

        last_completed_epoch = epoch
        patience_counter = PATIENCE - patience_left

        full_ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_mae,
            "patience_counter": patience_counter,
            "patience_left": patience_left,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "rng_state": _capture_rng_state(),
            "dataset_rng_state": full_ds.rng.bit_generator.state,
        }

        # latest.pt: CADA època (per reanudar)
        _atomic_torch_save(full_ckpt, latest_path)

        # best.pt + best_model_v3.pt: només si millora la val loss
        if improved:
            _atomic_torch_save(full_ckpt, best_full_path)
            _atomic_torch_save(model.state_dict(), best_weights_path)

        # Històric opcional cada 10 èpoques
        if epoch % 10 == 0:
            _atomic_torch_save(full_ckpt, ckpt_dir_path / f"epoch_{epoch:03d}.pt")

        _write_status(ckpt_dir_path, {
            "last_completed_epoch": int(epoch),
            "n_epochs": int(n_epochs),
            "best_val_loss": float(best_val_mae),
            "patience_left": int(patience_left),
            "latest": str(latest_path),
            "best_full": str(best_full_path),
            "best_weights": str(best_weights_path),
            "improved_this_epoch": bool(improved),
        })

        print(f"{epoch:>6} {train_loss:>11.4f} {val_loss:>10.4f} "
              f"{lr_now:>10.2e} {elapsed:>6.1f}s {status:>10}")

        if patience_left == 0:
            print(f"\nEarly stopping a l'època {epoch}.")
            break

    print(f"\nMillor Val Loss ({loss_name}): {best_val_mae:.4f}")
    print(f"Checkpoint complet (resume): {latest_path}")
    print(f"Millor checkpoint complet:   {best_full_path}")
    print(f"Pesos predicció:             {best_weights_path}")
    print("\nTraining finished")
    print(f"Last checkpoint: {latest_path}")
    print(f"Best checkpoint/model: {best_weights_path}")
    print(f"Last completed epoch: {last_completed_epoch}")

    if best_weights_path.exists():
        load_model_weights(best_weights_path, model, device)
    return model


# ── Inferència ────────────────────────────────────────────────────────────────

def predict_batch(model: TemporalMaskedAutoencoder,
                  loader: DataLoader,
                  device: torch.device) -> np.ndarray:
    """Inferència simple (sense TTA)."""
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            inp     = batch["input"].to(device)
            mask    = batch["mask"].to(device)
            profile = batch.get("profile")
            if profile is not None:
                profile = profile.to(device)
            preds.append(model(inp, mask, profile).cpu().numpy())
    return np.concatenate(preds, axis=0)


def predict_batch_tta(model: TemporalMaskedAutoencoder,
                      loader: DataLoader,
                      device: torch.device,
                      n_passes: int = TTA_N_PASSES,
                      noise_std: float = TTA_NOISE_STD) -> np.ndarray:
    """
    Test-Time Augmentation amb soroll als punts observats.

    NO usa shifts cíclics: aquests desalinejen el consum amb els embeddings
    de calendari (hora × tipus_dia), que són fixos per índex de patch.

    Passada 0: predicció neta.
    Passades 1..n-1: afegeix N(0, noise_std) només on mask=False (observats).
    Promitja totes les passades.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            inp     = batch["input"].to(device)
            mask    = batch["mask"].to(device)
            profile = batch.get("profile")
            if profile is not None:
                profile = profile.to(device)

            batch_preds = []
            for p in range(n_passes):
                if p == 0 or noise_std <= 0:
                    inp_p = inp
                else:
                    noise = torch.randn_like(inp) * noise_std
                    # Només pertorbem punts observats (mask=False)
                    inp_p = inp + noise * (~mask).float()
                batch_preds.append(model(inp_p, mask, profile).cpu().numpy())

            all_preds.append(np.mean(batch_preds, axis=0))

    return np.concatenate(all_preds, axis=0)
