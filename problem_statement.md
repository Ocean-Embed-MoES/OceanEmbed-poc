# INCOIS Problem Statement for Smart India Hackathon (SIH) 2026

**The INCOIS Problem Statements (PS) for the 9th edition of the Smart India Hackathon (SIH) 2026, organized by the Ministry of Education's Innovation Cell in collaboration with the All India Council for Technical Education (AICTE), are presented below in the prescribed format.**

---

## About INCOIS

The Indian National Centre for Ocean Information Services (INCOIS), under the Ministry of Earth Sciences, delivers critical ocean information and advisory services to support coastal communities, maritime operations, and disaster risk reduction. Its services can be broadly categorized into two major domains: disaster-related services and ecosystem-based services.

The disaster-related services focus on early warning and risk mitigation for ocean hazards such as tsunamis, storm surges, high waves, swell surges, and coastal currents. These timely advisories help authorities, disaster management agencies, and coastal populations make informed decisions, thereby enhancing maritime safety and reducing the impacts of ocean-related emergencies.

The ecosystem-based services emphasize sustainable management and monitoring of marine resources and ocean ecosystems. These services support fisheries, coastal zone management, and environmental conservation by providing scientific ocean data and analyses that contribute to the sustainable use and protection of marine ecosystems.

---

## Problem Statement#01

### Title of the Problem Statement

**OceanEmbed - Satellite Embedding-Based Deep Learning Framework for Reconstruction of Subsurface Ocean Temperature from Surface Satellite Observations**

---

## Description (Background, Detailed Description and Expected Solution)

### Background

Subsurface ocean temperature is a fundamental variable for understanding ocean circulation, upper-ocean heat content, stratification, climate variability, air-sea interaction and marine ecosystems. Accurate representation of the vertical ocean temperature is essential for applications such as marine heatwave monitoring, fisheries, and data assimilation, etc.

However, direct measurements of subsurface temperature remain sparse because they rely primarily on in-situ observing systems such as ARGO profiling floats, moored buoys, gliders, and ship observations. While these observations provide valuable vertical information, their spatial and temporal coverage is insufficient for generating continuous, basin-scale subsurface fields.

In contrast, satellite observations provide continuous, large-scale monitoring of surface ocean conditions at relatively high spatial and temporal resolution. Surface variables such as Sea Surface Temperature (SST), Sea Surface Salinity (SSS), Sea Surface Height (SSH)/Sea Level Anomaly (SLA), surface currents, and surface winds contain indirect signatures of subsurface ocean processes through physical mechanisms including thermocline displacement, mesoscale eddies, vertical mixing, transport, and ocean-atmosphere coupling.

Recent advances in Artificial Intelligence (AI), Deep Learning (DL), and representation learning enable the generation of satellite embeddings, where multidimensional surface observations are transformed into compact latent representations that capture hidden ocean dynamics. Such embeddings offer the potential to learn nonlinear relationships between surface observations and subsurface ocean structure more effectively than conventional machine learning approaches.

### Detailed Description

The current problem statement proposes the development of a Satellite Embedding-Based Deep Learning Framework to reconstruct depth-wise subsurface temperature from daily surface satellite observations at 0.25° spatial resolution for North Indian Ocean (5°N to 30°N and 45°E to 105°E). The objective is to estimate the three-dimensional ocean temperature using only surface satellite observations.

The proposed system shall:

1. Develop a preprocessing and harmonization pipeline for multi-source satellite and ocean datasets.
2. Standardize all datasets to:
   - a. Spatial Resolution: 0.25° × 0.25°
   - b. Temporal Resolution: Daily
3. Use surface observations as input variables:
   - a. Sea Surface Temperature (SST)
   - b. Sea Surface Salinity (SSS)
   - c. Sea Surface Height (SSH) / Sea Level Anomaly (SLA)
   - d. Surface ocean currents (U, V)
   - e. Surface Winds (U, V)
4. Generate compact satellite embeddings using DL architectures such as:
   - a. Convolutional Neural Networks (CNN)
   - b. Vision Transformers (ViT)
   - c. Autoencoders
   - d. Graph Neural Networks (GNN)
   - e. Attention-based hybrid architectures
5. Train reconstruction models that learn the relationship between surface ocean state to temperature profiles.
6. Reconstruct:
   - a. Temperature at standard depth levels. Standard depths in meters: (0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000)
7. Evaluate the reconstruction using independent observations and standard skill metrics like correlation, RMSE, Bias etc.

(If a dataset is not available at required resolution, the team may select the openly available product and perform appropriate spatial and temporal interpolation/regridding)

---

## Training Input Datasets

The following datasets are recommended for building the training and evaluation pipeline.

| Variable  | Product & Resolution     | Details                                                                                              |
|-----------|--------------------------|------------------------------------------------------------------------------------------------------|
| SST       | OSTIA: 0.05°, daily     | https://data.marine.copernicus.eu/product/SST_GLO_SST_L4_REP_OBSERVATIONS_010_011                                                                   |
| SSS       | SMAP, SMOS: 0.125°, daily | https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_S_SURFACE_MYNRT_015_013                                                                 |
| SSH       | DUACS: 0.25°, daily     | https://data.marine.copernicus.eu/product/SEALEVEL_GLO_PHY_CLIMATE_L4_MY_008_057                                                                   |
| Currents  | 0.25°, daily             | https://podaac.jpl.nasa.gov/dataset/OSCAR_L4_OC_FINAL_V2.0                                          |
| Winds     | 0.25°, daily             | https://podaac.jpl.nasa.gov/dataset/ASCATC-L2-Coastal https://podaac.jpl.nasa.gov/dataset/CCMP_WINDS_10M6HR_L4_V3.1 |

---

## Training Target Dataset (Subsurface Temperature)

**GLORYS Global Ocean Reanalysis** https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030
Variables: Temperature

---

## In-situ Observations Dataset

**Gridded ARGO**
INCOIS Live Access Server (LAS) – Gridded ARGO

---

## Expected Solution

- End-to-end preprocessing pipeline for satellite and ocean datasets.
- Satellite embedding engine capable of learning latent ocean representations from surface observations.
- Deep learning reconstruction model for estimating subsurface temperature.
- Standardized output at daily temporal resolution and 0.25° spatial resolution.
- Validation framework using independent ARGO observations.
- Demonstration of a working Proof-of-Concept (PoC) over the Bay of Bengal / Arabian Sea.

---

## Organization (Ministry/Department Name)

Indian National Centre for Ocean Information Services (INCOIS), Ministry of Earth Sciences, Government of India

## Category (Hardware/Software)

Software

## Theme (Available at www.sih.gov.in)

Disaster Management


