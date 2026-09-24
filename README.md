# spikewise

Spike waveform parameterization and analysis for intracellular and extracellular recordings.

---

## What this is


1. **`spikewise/` — Python package** for fitting and extracting features from action potential waveforms. Extracts: peak amplitude, peak sharpness, peak width, repolarization rate (`exp_lambda`), repolarization constant (`exp_const`), inflection time, and log inter-spike interval.

2. **`Action potential waveforms are state-dependent paper/` — Empirical paper analyses** across two datasets testing whether and how spike waveform features vary with neural input and network state. The detailed analysis code lives in the `AP_empirical_paper_all_analyses/` subfolder; see [`Action potential waveforms are state-dependent paper/AP_empirical_paper_all_analyses/README.md`](Action%20potential%20waveforms%20are%20state-dependent%20paper/AP_empirical_paper_all_analyses/README.md) for the full dataset and analysis breakdown.

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
├── Action potential waveforms are state-dependent paper/
│   ├── AP_empirical_paper_all_analyses/        # All paper analyses → see its own README.md
│   ├── 1_parameterization_captures_intra_spike_correlations.ipynb
│   ├── 2_electrical_stimulation_causally_influences_spike_waveform.ipynb
│   ├── 3_neurons_show_multimodal_spontaneous_variability.ipynb
│   ├── 4_within_neuron_variability_exceeds_between_neuron_differences.ipynb
│   └── 5_ap_waveform_predicts_peri_spike_lfp_state.ipynb
│       # One notebook per main finding, each calling the exact real function(s) and
│       # cached data that generate that finding's actual published figure
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
