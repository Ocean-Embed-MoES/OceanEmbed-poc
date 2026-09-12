#!/usr/bin/env python3
"""
scripts/train.py
----------------
Training entry point for OceanEmbed PoC.

Usage
-----
  python scripts/train.py                        # full run, uses configs/poc.yaml
  python scripts/train.py --config path/to.yaml  # custom config
  python scripts/train.py --resume outputs/checkpoints/best.pt

What happens
------------
  1. Load config from poc.yaml.
  2. Build train/val DataLoaders from preprocessed NetCDF files.
  3. Instantiate OceanEmbedModel (CBAM U-Net, 858 k params).
  4. Build DepthStratifiedLoss, Adam optimiser, CosineAnnealingLR scheduler.
  5. Run training for cfg.training.epochs epochs.
  6. Save best.pt and latest.pt checkpoints to outputs/checkpoints/.
  7. Save per-epoch history to outputs/metrics/history.json.

Pre-requisites
--------------
  scripts/prepare_data.py must have been run first to generate:
    outputs/processed/train/inputs_YYYY.nc
    outputs/processed/train/targets_YYYY.nc
    outputs/processed/val/inputs_YYYY.nc
    outputs/processed/val/targets_YYYY.nc
    outputs/processed/norm_stats.json
"""
from __future__ import annotations

import argparse
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.dataset   import OceanEmbedDataset
from src.model          import OceanEmbedModel
from src.training.trainer import build_trainer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OceanEmbed PoC — training script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="configs/poc.yaml",
                        help="Path to YAML config (default: configs/poc.yaml)")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = yaml.safe_load(cfg_path.read_text())

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    proc_dir = _ROOT / cfg["data"]["processed_dir"]
    tw       = cfg["data"]["temporal_window"]

    print(f"\nLoading datasets from {proc_dir} ...")
    train_ds = OceanEmbedDataset("train", proc_dir, temporal_window=tw)
    val_ds   = OceanEmbedDataset("val",   proc_dir, temporal_window=tw)
    print(f"  {train_ds}")
    print(f"  {val_ds}")

    tc         = cfg["training"]
    batch_size = tc["batch_size"]

    train_loader = DataLoader(
        train_ds,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = (device.type == "cuda"),
        drop_last   = True,   # keeps batch sizes uniform for loss averaging
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = batch_size * 2,   # larger batch for eval (no gradients)
        shuffle     = False,
        num_workers = 0,
        pin_memory  = (device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = OceanEmbedModel(
        in_channels  = tw * len(cfg["channels"]["names"]),   # 3 × 8 = 24
        enc_channels = cfg["model"]["enc_channels"],
        n_depths     = len(cfg["standard_depths"]),
    ).to(device)
    print(f"\n{model}")

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer, start_epoch = build_trainer(model, cfg, device, _ROOT)

    # Resume from checkpoint if requested
    if args.resume:
        resume_path = pathlib.Path(args.resume)
        if not resume_path.exists():
            print(f"Checkpoint not found: {resume_path}", file=sys.stderr)
            sys.exit(1)
        start_epoch = trainer.load_checkpoint(resume_path)

    # ── Train ─────────────────────────────────────────────────────────────────
    n_epochs = tc["epochs"] - start_epoch
    history  = trainer.fit(train_loader, val_loader, n_epochs=n_epochs)

    print(f"\nDone. History saved to {_ROOT / 'outputs/metrics/history.json'}")
    print(f"Best checkpoint: {_ROOT / tc['checkpoint_dir'] / 'best.pt'}")


if __name__ == "__main__":
    main()
