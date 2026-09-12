#!/usr/bin/env python3
"""
scripts/evaluate.py
-------------------
OceanEmbed PoC — full evaluation pipeline.

Usage
-----
  python scripts/evaluate.py                        # uses best.pt + configs/poc.yaml
  python scripts/evaluate.py --checkpoint outputs/checkpoints/best.pt
  python scripts/evaluate.py --config path/to.yaml

What this does
--------------
  1. Processes the test split (2019–2020) if not already done.
  2. Loads the best checkpoint and runs inference on every test day.
  3. Compares predictions (in °C) against:
       PRIMARY   — INCOIS LAS VAM + INCOIS McCreary (10-day, 1°)
       SECONDARY — ARMOR3D (daily, 0.125°)
  4. Reports per-depth RMSE / Bias / R² / Pearson-r.
  5. Saves outputs/metrics/validation_results.json
          and outputs/metrics/validation_summary.txt.
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

from src.data.dataset       import OceanEmbedDataset
from src.validation.evaluate import OceanEmbedEvaluator


def _ensure_test_split(cfg: dict, root: pathlib.Path) -> None:
    """
    Check that test-split NetCDF files exist.
    If any are missing, run prepare_data.py for the test split.
    """
    proc = root / cfg["data"]["processed_dir"] / "test"
    d    = cfg["data"]
    years = list(range(int(d["test_start"][:4]), int(d["test_end"][:4]) + 1))
    missing = [y for y in years if not (proc / f"inputs_{y}.nc").exists()]

    if not missing:
        print(f"  Test split already processed: {proc}")
        return

    print(f"  Test split missing for years: {missing}")
    print("  Running prepare_data.py --splits test ...")

    # Import and run the preprocessing logic directly
    import subprocess
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "prepare_data.py"),
         "--splits", "test", "--config", str(root / "configs" / "poc.yaml")],
        cwd=str(root),
        check=True,
    )
    print("  Test split preprocessing complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OceanEmbed PoC — evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",     default="configs/poc.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best.pt")
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────────
    cfg_path = pathlib.Path(args.config)
    if not cfg_path.exists():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)
    cfg = yaml.safe_load(cfg_path.read_text())

    ckpt_path = pathlib.Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nOceanEmbed PoC — Evaluation")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Device:     {device}")
    if device.type == "cuda":
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")

    # ── Ensure test split is preprocessed ────────────────────────────────────
    print("\nChecking test split ...")
    _ensure_test_split(cfg, _ROOT)

    # ── Test DataLoader ───────────────────────────────────────────────────────
    proc_dir = _ROOT / cfg["data"]["processed_dir"]
    tw       = cfg["data"]["temporal_window"]

    print("\nBuilding test DataLoader ...")
    test_ds = OceanEmbedDataset("test", proc_dir, temporal_window=tw)
    print(f"  {test_ds}")

    test_loader = DataLoader(
        test_ds,
        batch_size  = cfg["training"]["batch_size"] * 2,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = (device.type == "cuda"),
    )

    # ── Evaluator ─────────────────────────────────────────────────────────────
    print("\nLoading model ...")
    evaluator = OceanEmbedEvaluator(ckpt_path, cfg, device)

    # ── Run evaluation ────────────────────────────────────────────────────────
    out_dir = _ROOT / "outputs" / "metrics"
    results = evaluator.evaluate(test_loader=test_loader, test_dataset=test_ds, out_dir=out_dir)

    print("\n" + "=" * 65)
    print("Evaluation complete.")
    print(f"  JSON:    {out_dir / 'validation_results.json'}")
    print(f"  Summary: {out_dir / 'validation_summary.txt'}")


if __name__ == "__main__":
    main()
