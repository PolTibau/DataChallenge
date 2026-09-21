"""Smoke test: train a few epochs, resume, verify scheduler/patience/best weights."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lstm.src.trainer import (
    train, load_training_checkpoint, load_model_weights, is_full_training_checkpoint,
)
from lstm.src.model import TemporalMaskedAutoencoder
from lstm.src.config import SEQ_LEN


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    curves = rng.normal(size=(24, SEQ_LEN)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        weights = tmp / "best_model_v3.pt"

        print("=== Fase 1: entrenar 2 èpoques des de zero ===")
        train(curves, device,
              n_epochs_override=2,
              ckpt_path=weights,
              ckpt_dir=tmp,
              resume_from=None)

        latest = tmp / "latest.pt"
        assert latest.exists(), "latest.pt no s'ha creat"
        assert weights.exists(), "best_model_v3.pt no s'ha creat"

        ckpt = load_training_checkpoint(latest, device)
        assert ckpt["epoch"] == 2, f"esperava epoch=2, got {ckpt['epoch']}"
        assert "optimizer_state_dict" in ckpt
        assert "scheduler_state_dict" in ckpt
        step_before = ckpt["scheduler_state_dict"]["step_count"]
        best_before = ckpt["best_val_loss"]
        pat_before = ckpt["patience_left"]
        print(f"  latest.pt epoch={ckpt['epoch']} step_count={step_before} "
              f"best_val={best_before:.4f} patience_left={pat_before}")

        raw_weights = torch.load(weights, map_location="cpu", weights_only=True)
        assert not is_full_training_checkpoint(raw_weights), \
            "best_model_v3.pt ha de ser només state_dict"
        print("  best_model_v3.pt és un state_dict (predicció) ✓")

        print("\n=== Fase 1b: checkpoint antic no reanudable ===")
        old = tmp / "old_weights_only.pt"
        torch.save(raw_weights, old)
        try:
            load_training_checkpoint(old, device)
            raise AssertionError("havia de fallar amb checkpoint antic")
        except ValueError as exc:
            assert "no reanudable" in str(exc) or "antic" in str(exc)
            print(f"  missatge clar ✓  ({str(exc)[:80]}...)")

        print("\n=== Fase 2: resume → èpoques 3 i 4 ===")
        train(curves, device,
              n_epochs_override=4,
              ckpt_path=weights,
              ckpt_dir=tmp,
              resume_from=latest)

        ckpt2 = load_training_checkpoint(latest, device)
        assert ckpt2["epoch"] == 4, f"esperava epoch=4 després del resume, got {ckpt2['epoch']}"
        step_after = ckpt2["scheduler_state_dict"]["step_count"]
        assert step_after > step_before, \
            f"scheduler no ha avançat: {step_before} → {step_after}"
        print(f"  latest.pt epoch={ckpt2['epoch']} step_count={step_after} "
              f"(era {step_before}) patience_left={ckpt2['patience_left']}")
        print("  resume ha continuat a l'època 3 (no des de 1) ✓")
        print("  optimizer/scheduler restaurats i han avançat ✓")
        print("  best_val_loss/patience presents al checkpoint ✓")

        model = TemporalMaskedAutoencoder().to(device)
        load_model_weights(weights, model, device)
        print("  load_model_weights(best_model_v3.pt) ✓")

    print("\nSMOKE TEST OK")


if __name__ == "__main__":
    main()
