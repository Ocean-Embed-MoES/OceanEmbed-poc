"""
trainer.py
----------
Training and validation loop for OceanEmbed PoC.

Responsibilities
----------------
- Run train / val epochs with progress bars (tqdm)
- Track and log per-epoch metrics (loss + per-zone RMSE)
- Save checkpoints: latest + best-val-loss
- Cosine annealing LR schedule
- Clean early stopping (optional, configurable)
- Write training history to outputs/metrics/history.json

Checkpoint format
-----------------
Each checkpoint is a dict saved with torch.save:
  {
    "epoch":       int,
    "val_loss":    float,
    "model":       model.state_dict(),
    "optimizer":   optimizer.state_dict(),
    "scheduler":   scheduler.state_dict(),
    "config":      cfg dict (for reproducibility)
  }
"""
from __future__ import annotations

import json
import pathlib
import time
from typing import Any

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.loss import DepthStratifiedLoss, LossOutput


class Trainer:
    """
    Training orchestrator for OceanEmbedModel.

    Parameters
    ----------
    model          : OceanEmbedModel (or any nn.Module with matching I/O).
    loss_fn        : DepthStratifiedLoss instance.
    optimizer      : Torch optimizer (Adam recommended).
    scheduler      : LR scheduler (CosineAnnealingLR recommended).
    device         : torch.device to train on.
    checkpoint_dir : Directory for saving checkpoints.
    metrics_dir    : Directory for saving history.json.
    """

    def __init__(
        self,
        model:          nn.Module,
        loss_fn:        DepthStratifiedLoss,
        optimizer:      torch.optim.Optimizer,
        scheduler:      torch.optim.lr_scheduler._LRScheduler,
        device:         torch.device,
        checkpoint_dir: pathlib.Path,
        metrics_dir:    pathlib.Path,
    ) -> None:
        self.model          = model
        self.loss_fn        = loss_fn
        self.optimizer      = optimizer
        self.scheduler      = scheduler
        self.device         = device
        self.checkpoint_dir = checkpoint_dir
        self.metrics_dir    = metrics_dir

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        self._history: list[dict[str, Any]] = []
        self._best_val_loss = float("inf")

    # ── Single epoch loops ─────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        """One full pass over the training set with gradient updates."""
        self.model.train()
        accum = _MetricAccumulator()

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [train]", leave=False, ncols=90)
        for x, y, mask in pbar:
            x    = x.to(self.device, non_blocking=True)
            y    = y.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(x)
            out: LossOutput = self.loss_fn(pred, y, mask)
            out.total.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            accum.update(out, x.size(0))
            pbar.set_postfix(loss=f"{out.total.item():.4f}")

        return accum.mean()

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader, epoch: int) -> dict[str, float]:
        """One full pass over the validation set (no gradients)."""
        self.model.eval()
        accum = _MetricAccumulator()

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [val]  ", leave=False, ncols=90)
        for x, y, mask in pbar:
            x    = x.to(self.device, non_blocking=True)
            y    = y.to(self.device, non_blocking=True)
            mask = mask.to(self.device, non_blocking=True)

            pred = self.model(x)
            out: LossOutput = self.loss_fn(pred, y, mask)
            accum.update(out, x.size(0))
            pbar.set_postfix(loss=f"{out.total.item():.4f}")

        return accum.mean()

    # ── Fit ────────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        n_epochs:     int,
    ) -> list[dict[str, Any]]:
        """
        Run the full training loop.

        Parameters
        ----------
        train_loader : DataLoader for the training split.
        val_loader   : DataLoader for the validation split.
        n_epochs     : Total number of epochs to train.

        Returns
        -------
        list[dict]  Per-epoch metric history (also saved to history.json).
        """
        print(f"\n{'═' * 65}")
        print(f"  OceanEmbed PoC — Training")
        print(f"  Device:  {self.device}")
        print(f"  Epochs:  {n_epochs}")
        print(f"  Train batches / epoch: {len(train_loader)}")
        print(f"  Val   batches / epoch: {len(val_loader)}")
        print(f"{'═' * 65}\n")

        for epoch in range(1, n_epochs + 1):
            t0 = time.perf_counter()

            train_m = self._train_epoch(train_loader, epoch)
            val_m   = self._val_epoch(val_loader, epoch)
            self.scheduler.step()

            elapsed = time.perf_counter() - t0
            lr      = self.optimizer.param_groups[0]["lr"]

            # ── Log ────────────────────────────────────────────────────────
            record = {
                "epoch":          epoch,
                "lr":             lr,
                "elapsed_s":      round(elapsed, 1),
                "train_loss":     train_m["total"],
                "val_loss":       val_m["total"],
                "train_mixed":    train_m["mixed_layer"],
                "train_thermo":   train_m["thermocline"],
                "train_deep":     train_m["deep_ocean"],
                "val_mixed":      val_m["mixed_layer"],
                "val_thermo":     val_m["thermocline"],
                "val_deep":       val_m["deep_ocean"],
            }
            self._history.append(record)
            self._save_history()

            # ── Console summary ────────────────────────────────────────────
            is_best = val_m["total"] < self._best_val_loss
            marker  = " ★" if is_best else ""
            print(
                f"  Epoch {epoch:03d}/{n_epochs:03d} "
                f"| train {train_m['total']:.4f} "
                f"| val {val_m['total']:.4f}{marker} "
                f"| mix {val_m['mixed_layer']:.4f} "
                f"| thermo {val_m['thermocline']:.4f} "
                f"| deep {val_m['deep_ocean']:.4f} "
                f"| lr {lr:.2e} "
                f"| {elapsed:.0f}s"
            )

            # ── Checkpoints ────────────────────────────────────────────────
            self._save_checkpoint(epoch, val_m["total"])
            if is_best:
                self._best_val_loss = val_m["total"]
                self._save_checkpoint(epoch, val_m["total"], name="best.pt")

        print(f"\n  Training complete. Best val loss: {self._best_val_loss:.4f}")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        return self._history

    # ── Checkpoint helpers ─────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        epoch:    int,
        val_loss: float,
        name:     str = "latest.pt",
    ) -> None:
        path = self.checkpoint_dir / name
        torch.save(
            {
                "epoch":     epoch,
                "val_loss":  val_loss,
                "model":     self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
            },
            path,
        )

    def load_checkpoint(self, path: pathlib.Path) -> int:
        """
        Load a checkpoint and restore model, optimizer, and scheduler state.

        Returns the epoch number stored in the checkpoint.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.scheduler.load_state_dict(ckpt["scheduler"])
        self._best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"  Resumed from {path}  (epoch {ckpt['epoch']}, val_loss {ckpt['val_loss']:.4f})")
        return ckpt["epoch"]

    def _save_history(self) -> None:
        path = self.metrics_dir / "history.json"
        path.write_text(json.dumps(self._history, indent=2))


# ── Internal helper ────────────────────────────────────────────────────────────

class _MetricAccumulator:
    """Weighted running mean of loss metrics across batches."""

    def __init__(self) -> None:
        self._sums:   dict[str, float] = {}
        self._n_total = 0

    def update(self, out: LossOutput, batch_size: int) -> None:
        metrics = {
            "total":       float(out.total),
            "mixed_layer": out.mixed_layer,
            "thermocline": out.thermocline,
            "deep_ocean":  out.deep_ocean,
        }
        for k, v in metrics.items():
            self._sums[k] = self._sums.get(k, 0.0) + v * batch_size
        self._n_total += batch_size

    def mean(self) -> dict[str, float]:
        n = max(self._n_total, 1)
        return {k: v / n for k, v in self._sums.items()}


# ── Factory ────────────────────────────────────────────────────────────────────

def build_trainer(
    model:   nn.Module,
    cfg:     dict,
    device:  torch.device,
    root:    pathlib.Path,
) -> tuple[Trainer, int]:
    """
    Build a Trainer from a YAML config dict.

    Parameters
    ----------
    model  : OceanEmbedModel instance.
    cfg    : Full YAML config (from poc.yaml).
    device : Training device.
    root   : Project root (for resolving relative paths in cfg).

    Returns
    -------
    (trainer, start_epoch)
        start_epoch is 0 normally, or the resumed epoch if --resume was set.
    """
    tc = cfg["training"]

    optimizer = Adam(
        model.parameters(),
        lr=tc["learning_rate"],
        weight_decay=tc["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=tc["epochs"], eta_min=1e-6)

    dw = tc["depth_weights"]
    loss_fn = DepthStratifiedLoss(
        mixed_layer_weight=dw["mixed_layer"],
        thermocline_weight=dw["thermocline"],
        deep_ocean_weight =dw["deep_ocean"],
    )

    trainer = Trainer(
        model          = model,
        loss_fn        = loss_fn,
        optimizer      = optimizer,
        scheduler      = scheduler,
        device         = device,
        checkpoint_dir = root / tc["checkpoint_dir"],
        metrics_dir    = root / "outputs/metrics",
    )
    return trainer, 0
