#!/usr/bin/env python3
"""
scripts/plot_results.py
-----------------------
OceanEmbed PoC — Stage 6 visualization.

Generates four figures saved to outputs/figures/:
  1. depth_profile_rmse.png     — RMSE vs depth (all sources)
  2. depth_profile_r.png        — Pearson r vs depth
  3. training_history.png       — training / val loss curves
  4. spatial_bias_map.png       — mean bias at 3 representative depths
"""
from __future__ import annotations

import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe on all systems
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import xarray as xr

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.data.dataset      import OceanEmbedDataset
from src.model             import OceanEmbedModel
from src.preprocessing.normalize import load_stats
from src.preprocessing.regrid    import make_target_grid
import yaml

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":   "DejaVu Sans",
    "font.size":     10,
    "axes.grid":     True,
    "grid.alpha":    0.3,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

STANDARD_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
DEPTH_LABELS    = [f"d{d}m" for d in STANDARD_DEPTHS]
COMMON_DEPTHS   = [5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
COMMON_LABELS   = [f"d{d}m" for d in COMMON_DEPTHS]

OUT_DIR = _ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_results() -> dict:
    p = _ROOT / "outputs" / "metrics" / "validation_results.json"
    return json.loads(p.read_text())


# ── Figure 1 & 2 — Depth profiles ─────────────────────────────────────────────

def plot_depth_profiles(results: dict) -> None:
    """RMSE and Pearson-r vs depth from all validation sources (2019+2020 averaged)."""

    fig, axes = plt.subplots(1, 2, figsize=(11, 7), sharey=True)

    # ── ARMOR3D (supplementary) — all 15 depths ───────────────────────────────
    def avg_years(dataset_dict: dict, prefix: str, labels: list[str]) -> tuple[list, list]:
        """Average metric across 2019 and 2020."""
        rmse_vals, r_vals = [], []
        for lbl in labels:
            vals_rmse, vals_r = [], []
            for yr in ["2019", "2020"]:
                m = dataset_dict.get(f"{prefix}_{yr}", {}).get(lbl, {})
                if m.get("n_valid", 0) > 0 and not np.isnan(m.get("rmse", float("nan"))):
                    vals_rmse.append(m["rmse"])
                    vals_r.append(m["r"])
            rmse_vals.append(np.mean(vals_rmse) if vals_rmse else float("nan"))
            r_vals.append(np.mean(vals_r)    if vals_r    else float("nan"))
        return rmse_vals, r_vals

    # ARMOR3D — all 15 depths
    armor_rmse, armor_r = avg_years(results["supplementary"], "armor3d", DEPTH_LABELS)
    # McCreary — 14 common depths (no 0m)
    mc_rmse,   mc_r    = avg_years(results["primary"], "incois_mccreary", COMMON_LABELS)
    # VAM — 14 common depths
    vam_rmse,  vam_r   = avg_years(results["primary"], "incois_vam",     COMMON_LABELS)

    depths_all    = np.array(STANDARD_DEPTHS)
    depths_common = np.array(COMMON_DEPTHS)

    ax_rmse, ax_r = axes

    # ── RMSE plot ─────────────────────────────────────────────────────────────
    ax_rmse.plot(armor_rmse, depths_all, "o-", color="#2563eb", lw=2,
                 label="ARMOR3D (supp.)", markersize=5)
    ax_rmse.plot(mc_rmse,  depths_common, "s--", color="#16a34a", lw=2,
                 label="INCOIS McCreary (primary)", markersize=5)
    ax_rmse.plot(vam_rmse, depths_common, "^:", color="#dc2626", lw=1.5,
                 label="INCOIS VAM (primary)", markersize=5)

    # Target RMSE shading from architecture doc
    ax_rmse.axvspan(0, 0.5,  alpha=0.08, color="green",  label="Target: mixed <0.5°C")
    ax_rmse.axvspan(0, 1.0,  alpha=0.05, color="orange", label="Target: thermocline <1.0°C")

    ax_rmse.set_xlabel("RMSE (°C)")
    ax_rmse.set_ylabel("Depth (m)")
    ax_rmse.invert_yaxis()
    ax_rmse.set_xlim(left=0)
    ax_rmse.set_title("RMSE vs Depth", fontweight="bold", pad=8)
    ax_rmse.legend(fontsize=8, loc="lower right")

    # Zone dividers
    for d, label in [(30, "Mixed\nLayer"), (200, "Thermo-\ncline"), (1000, "Deep\nOcean")]:
        ax_rmse.axhline(d, color="gray", lw=0.8, ls="--", alpha=0.5)

    # ── r plot ────────────────────────────────────────────────────────────────
    ax_r.plot(armor_r, depths_all,    "o-", color="#2563eb", lw=2, markersize=5,
              label="ARMOR3D (supp.)")
    ax_r.plot(mc_r,  depths_common,   "s--", color="#16a34a", lw=2, markersize=5,
              label="INCOIS McCreary (primary)")
    ax_r.plot(vam_r, depths_common,   "^:", color="#dc2626", lw=1.5, markersize=5,
              label="INCOIS VAM (primary)")
    ax_r.axvline(0.97, color="green", ls="--", lw=1, alpha=0.6, label="Full-system target r>0.97")
    ax_r.axvline(0.80, color="orange", ls=":", lw=1, alpha=0.6, label="PoC acceptable r>0.80")

    ax_r.set_xlabel("Pearson r")
    ax_r.set_xlim(0, 1)
    ax_r.set_title("Pearson r vs Depth", fontweight="bold", pad=8)
    ax_r.legend(fontsize=8, loc="lower right")

    for d in [30, 200]:
        ax_r.axhline(d, color="gray", lw=0.8, ls="--", alpha=0.5)

    fig.suptitle(
        "OceanEmbed PoC — Test-Set Validation (2019–2020)\n"
        "Training: 2015–2017 · Grid: 0.5° 50×120 NIO · 858k params",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out = OUT_DIR / "depth_profile.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 2 — Training history ───────────────────────────────────────────────

def plot_training_history() -> None:
    """Training and validation loss curves from training_history.json."""
    hist_path = _ROOT / "outputs" / "metrics" / "history.json"
    if not hist_path.exists():
        print(f"  Skipping training history plot: {hist_path} not found")
        return

    hist = json.loads(hist_path.read_text())
    epochs    = [h["epoch"]    for h in hist]
    train_l   = [h["train_loss"] for h in hist]
    val_l     = [h["val_loss"]   for h in hist]
    best_ep   = min(hist, key=lambda h: h["val_loss"])["epoch"]
    best_val  = min(h["val_loss"] for h in hist)

    # Extract zone losses from val (if available)
    ml_l  = [h.get("val_mixed",  float("nan")) for h in hist]
    th_l  = [h.get("val_thermo", float("nan")) for h in hist]
    dp_l  = [h.get("val_deep",   float("nan")) for h in hist]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax1, ax2 = axes
    ax1.plot(epochs, train_l, lw=2, label="Train loss", color="#2563eb")
    ax1.plot(epochs, val_l,   lw=2, label="Val loss",   color="#dc2626")
    ax1.axvline(best_ep, color="green", ls="--", lw=1.5,
                label=f"Best epoch {best_ep} (val={best_val:.4f})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Depth-stratified RMSE (normalized)")
    ax1.set_title("Training & Validation Loss", fontweight="bold")
    ax1.legend(fontsize=9)

    ax2.plot(epochs, ml_l, lw=1.8, label="Mixed layer (w=1.0)",  color="#2563eb")
    ax2.plot(epochs, th_l, lw=1.8, label="Thermocline (w=2.0)",  color="#f59e0b")
    ax2.plot(epochs, dp_l, lw=1.8, label="Deep ocean (w=0.5)",   color="#16a34a")
    ax2.axvline(best_ep, color="gray", ls="--", lw=1, alpha=0.6)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Zone RMSE (normalized)")
    ax2.set_title("Per-Zone Validation Loss", fontweight="bold")
    ax2.legend(fontsize=9)

    fig.suptitle("OceanEmbed PoC — Training History (40 epochs, RTX 4050)",
                 fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "training_history.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Figure 3 — Spatial bias maps ──────────────────────────────────────────────

def plot_spatial_bias() -> None:
    """
    Mean prediction bias at 0 m, 100 m, 300 m across 2019 test predictions.
    """
    cfg      = yaml.safe_load((_ROOT / "configs" / "poc.yaml").read_text())
    proc_dir = _ROOT / cfg["data"]["processed_dir"]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tw       = cfg["data"]["temporal_window"]

    # Load model
    ckpt = torch.load(_ROOT / "outputs" / "checkpoints" / "best.pt",
                      map_location=device, weights_only=False)
    model = OceanEmbedModel(
        in_channels  = tw * len(cfg["channels"]["names"]),
        enc_channels = cfg["model"]["enc_channels"],
        n_depths     = len(cfg["standard_depths"]),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    stats = load_stats(proc_dir / "norm_stats.json")
    tlats, tlons = make_target_grid(
        cfg["data"]["lat_min"], cfg["data"]["lat_max"],
        cfg["data"]["lon_min"], cfg["data"]["lon_max"],
        cfg["data"]["grid_res"],
    )

    # Load 2019 test data only (365 days)
    ds = OceanEmbedDataset("test", proc_dir, temporal_window=tw)
    # Only use 2019 days (first 365 - tw + 1 valid samples)
    n_2019 = 365 - tw + 1

    from torch.utils.data import DataLoader, Subset
    sub   = Subset(ds, list(range(min(n_2019, len(ds)))))
    loader = DataLoader(sub, batch_size=16, shuffle=False, num_workers=0)

    DEPTH_LABELS_ALL = [f"d{d}m" for d in STANDARD_DEPTHS]
    target_stats     = stats["targets"]

    pred_sum  = np.zeros((len(STANDARD_DEPTHS), len(tlats), len(tlons)), dtype=np.float64)
    tgt_sum   = np.zeros_like(pred_sum)
    n_samples = 0

    with torch.no_grad():
        for x, y, mask in loader:
            x   = x.to(device)
            out = model(x).cpu().numpy()   # (B, 15, H, W) normalised
            y_  = y.numpy()               # (B, 15, H, W) normalised

            # Denormalize both
            B = out.shape[0]
            for di, lbl in enumerate(DEPTH_LABELS_ALL):
                mean = float(target_stats[lbl]["mean"])
                std  = max(float(target_stats[lbl]["std"]), 1e-10)
                out[:, di] = out[:, di] * std + mean
                y_[:, di]  = y_[:, di]  * std + mean

            # Mask land (y was 0-filled for land)
            mask_np = mask.numpy().astype(bool)  # (B, H, W)
            for b in range(B):
                for di in range(len(STANDARD_DEPTHS)):
                    o = out[b, di]
                    t = y_[b, di]
                    o[~mask_np[b]] = float("nan")
                    t[~mask_np[b]] = float("nan")

            pred_sum  += np.nansum(out, axis=0)
            tgt_sum   += np.nansum(y_,  axis=0)
            n_samples += B

    mean_bias = (pred_sum - tgt_sum) / n_samples   # (15, H, W)

    # Plot depths 0m (idx 0), 100m (idx 7), 300m (idx 11)
    plot_depths  = [0, 100, 300]
    plot_indices = [STANDARD_DEPTHS.index(d) for d in plot_depths]
    vmag = 1.5   # ±1.5°C colour scale

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, di, depth in zip(axes, plot_indices, plot_depths):
        bias = mean_bias[di]
        im = ax.pcolormesh(tlons, tlats, bias,
                           cmap="RdBu_r", vmin=-vmag, vmax=vmag)
        plt.colorbar(im, ax=ax, label="Bias (°C)", shrink=0.85)
        ax.set_title(f"Mean Bias at {depth} m (2019)", fontweight="bold")
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_aspect("equal")

    fig.suptitle(
        "OceanEmbed PoC — Spatial Mean Bias Maps\n"
        "Model prediction − GLORYS target, averaged over 2019",
        fontweight="bold"
    )
    plt.tight_layout()
    out = OUT_DIR / "spatial_bias_maps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("OceanEmbed PoC — Stage 6: Generating figures ...")
    results = load_results()

    print("\n[1/3] Depth profile plots ...")
    plot_depth_profiles(results)

    print("\n[2/3] Training history ...")
    plot_training_history()

    print("\n[3/3] Spatial bias maps ...")
    plot_spatial_bias()

    print(f"\n✅ All figures saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
