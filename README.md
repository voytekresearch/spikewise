# spikewise

Spike waveform parameterization and analysis for intracellular and extracellular recordings.

---

## What this is


1. **`spikewise/` — Python package** for fitting and extracting features from action potential waveforms. Extracts: peak amplitude, peak sharpness, peak width, repolarization rate (`exp_lambda`), repolarization constant (`exp_const`), inflection time, and log inter-spike interval.

2. **`AP_empirical_paper1/` — Empirical paper analyses** across two datasets testing whether and how spike waveform features vary with neural input and network state. See [`AP_empirical_paper1/README.md`](AP_empirical_paper1/README.md) for the full dataset and analysis breakdown.

---

## Repository structure

```
spikewise/
├── spikewise/                     # Core Python package
│   ├── patch/                     # Patch-clamp spike parameterization
│   │   ├── features/              # intra.py (waveform features), inter.py (ISI)
│   │   ├── fit/                   # Spike class, SpikeGroup batch fitting
│   │   ├── points/                # Peak / inflection / decay detection
│   │   └── plts/                  # Package-level plotting
│   ├── gaussian/                  # Gaussian mixture model alternative
│   └── tests/                     # Unit tests (mirrors package structure)
│
├── AP_empirical_paper1/           # All paper analyses → see AP_empirical_paper1/README.md
│
├── scripts/                       # Batch runners and utilities
│   ├── run_spe1_batch.py          # Headless batch executor for spe-1 pipeline
│   ├── run_ridge_psd_cell.py      # Per-cell spike-to-LFP ridge regression
│   ├── run_allen_ct_batch.py      # Batch runner for Allen Cell Types dataset
│   ├── save_nocluster_feats.py    # Build cluster_df.pkl for cells without clusters
│   └── benchmark_spe1_notebooks.py
│
├── docs/tutorials/                # Usage tutorials for the spikewise package
├── params/                        # Pre-computed parameter files
└── requirements.txt
```

---

## Install

```bash
git clone https://github.com/voytekresearch/spikewise
cd spikewise
pip install -e .
```

**Dependencies:** Python ≥ 3.9, numpy, scipy, matplotlib, pandas, bycycle, neurodsp, specparam, mne, statsmodels, seaborn, papermill

---

## Running analyses headlessly

The batch script runs per-cell Jupyter notebooks via papermill — all per-cell logic, manual clustering thresholds, and plots are preserved.

```bash
# Cluster all 43 cells (loads from cache where pickles exist)
python scripts/run_spe1_batch.py

# Cluster all cells + run LFP analysis for priority cells
python scripts/run_spe1_batch.py --lfp

# Force-redo clustering for all cells
python scripts/run_spe1_batch.py --force-cluster

# Priority cells only, force-redo everything
python scripts/run_spe1_batch.py --priority --lfp --force-all
```

See [`scripts/README.md`](scripts/README.md) for full flag reference and path override instructions.

---
