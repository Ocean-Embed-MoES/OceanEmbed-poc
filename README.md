# OceanEmbed — PoC

> **SIH 2026 | INCOIS Problem Statement #SIH26066**  
> *Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature*

---

## Overview

**OceanEmbed** reconstructs 3D subsurface ocean temperature in the **North Indian Ocean (5°N–30°N, 45°E–105°E)** at **daily, 0.25° resolution** using only surface satellite observations — no in-situ data required at inference time.

The system uses a **CBAM-enhanced U-Net** with four tightly integrated components:

```
[1] Preprocessing Pipeline → [2] Satellite Embedding Engine → [3] Reconstruction Model
                                                                        ↕
                                              [4] Validation Framework
```

---

## Architecture at a Glance

| Component | Implementation |
|---|---|
| **Preprocessing Pipeline** | 12-channel feature set · 7-day rolling window · Z-score normalization |
| **Satellite Embedding Engine** | U-Net Encoder with CBAM attention · 512-ch bottleneck at 12×30 |
| **Reconstruction Model** | U-Net Decoder with skip connections · 15 depth-level output head |
| **Validation Framework** | RMSE/R²/Bias profiles · ARGO comparison · SHAP explainability |

### Input Features (12 channels)
| # | Feature | Source |
|---|---|---|
| 1 | SST | OSTIA (0.05° → 0.25°) |
| 2 | ∇SST (log-magnitude) | Derived |
| 3 | SSH | DUACS (0.25°) |
| 4 | ∇SSH (log-magnitude) | Derived |
| 5 | SSS | SMAP/SMOS (0.125° → 0.25°) |
| 6 | WSC (Wind Stress Curl) | Derived from U/V wind |
| 7 | U_curr | OSCAR (0.25°) |
| 8 | V_curr | OSCAR (0.25°) |
| 9 | Bathymetry | ETOPO (static) |
| 10 | Latitude grid | Static |
| 11 | Longitude grid | Static |
| 12 | Day-of-Year | Calendar |

### Output
- **Temperature at 15 standard depth levels**: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000 m
- **Shape**: `(B, 15, 100, 240)` — daily, 0.25°, North Indian Ocean

---

## Accuracy Targets

| Zone | Depths | RMSE Target | R² Target |
|---|---|---|---|
| Mixed Layer | 0–30 m | < 0.5°C | > 0.97 |
| Thermocline | 50–200 m | < 1.0°C | > 0.95 |
| Deep Ocean | 300–1000 m | < 0.2°C | > 0.99 |
| **Overall** | **All 15 depths** | **< 0.7°C** | **> 0.96** |

---

## Repository Structure

```
OceanEmbed-poc/
├── architecture_design_v3_final.md   # Full architecture design doc
├── problem_statement.md              # INCOIS SIH 2026 problem statement
├── README.md
├── .gitignore
├── src/
│   ├── preprocessing/                # Component 1 — Preprocessing Pipeline
│   ├── model/                        # Components 2 & 3 — Embedding Engine + Reconstruction
│   └── validation/                   # Component 4 — Validation Framework
├── configs/                          # Training configs
├── scripts/                          # Data download & preprocessing scripts
└── notebooks/                        # EDA and result visualization
```

---

## Tech Stack

| Component | Tool |
|---|---|
| Deep Learning | PyTorch + PyTorch Lightning |
| Data handling | xarray, netCDF4, numpy |
| Regridding | pyinterp, scipy |
| Explainability | SHAP (GradientSHAP) |
| Visualization | matplotlib, cartopy |
| Experiment tracking | Weights & Biases |

---

## PoC Scope

Demonstration over:
- **Bay of Bengal** — Monsoon-driven halocline, river outflow dynamics
- **Arabian Sea** — Summer upwelling, SST-thermocline coupling

---

## Team

OceanEmbed — SIH 2026 Submission  
*Organization: INCOIS, Ministry of Earth Sciences, Government of India*
