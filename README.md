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
└── requirements.txt
```

---

## System requirements

- **Operating system:** any platform that runs Python (macOS, Linux, Windows). Tested on macOS 26.3 (Apple M1 Pro, 16 GB RAM).
- **Python:** 3.8 or newer. Tested on Python 3.11.13.
- **Hardware:** no special hardware is required; a standard desktop or laptop is enough.
- **Dependencies of the `spikewise` package** (installed automatically, see `requirements.txt`): numpy ≥ 1.21.4, scipy ≥ 1.7.3, pandas ≥ 1.4.3, matplotlib ≥ 3.5.0, statsmodels ≥ 0.13.2, ruptures ≥ 1.1.9, neurodsp, bycycle.
- **Additional packages for the paper notebooks** are listed in [`Action potential waveforms are state-dependent paper/requirements.txt`](Action%20potential%20waveforms%20are%20state-dependent%20paper/requirements.txt) (e.g. scikit-learn, seaborn, mne, specparam). The batch scripts below also use papermill.
- **Versions tested:** numpy 1.26.4, scipy 1.10.1, pandas 1.5.3, matplotlib 3.10.8, statsmodels 0.14.6, ruptures 1.1.10, neurodsp 2.4.0, bycycle 1.2.0, scikit-learn 1.8.0, seaborn 0.13.2, mne 1.11.0, specparam 2.0.0rc3.

---

## Install

```bash
git clone https://github.com/voytekresearch/spikewise
cd spikewise
pip install -e .
pip install jupyter scikit-learn          # to run the tutorials
pip install -r "Action potential waveforms are state-dependent paper/requirements.txt"   # to run the paper notebooks
```

Typical install time on a normal desktop computer: about 2 minutes (measured in a clean environment: about 45 s for `spikewise` and about 50 s for jupyter and scikit-learn).

---

## Demo

The tutorials in [`docs/tutorials/`](docs/tutorials/) run on a small bundled dataset (`docs/tutorials/data/example_recordings.npz`, 2.6 MB), so nothing needs to be downloaded:

```bash
cd docs/tutorials
jupyter notebook 01_quickstart.ipynb
```

- **Expected output:** each notebook is saved with its outputs (spike fits, feature tables and figures), so a run should reproduce what is shown.
- **Expected run time:** about 10 to 15 seconds per tutorial notebook on a normal desktop computer.
- **Running on your own data:** tutorials 1 to 3 show how to fit one recording (`Spike`), many recordings (`SpikeGroup`), and relate spike shape to another signal (`.alt()`).
- **Reproducing the paper:** the five finding notebooks in `Action potential waveforms are state-dependent paper/` reproduce each main result from small cached data that ships with the repository. The full analyses in `AP_empirical_paper_all_analyses/` require the raw pvc-6 and spe-1 recordings from CRCNS.

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

## License

`spikewise` is released under the Apache License 2.0; see [`LICENSE`](LICENSE).
