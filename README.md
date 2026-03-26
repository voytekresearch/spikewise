# spikeparam

Spike waveform parameterization and analysis for juxtacellular and patch-clamp recordings.

---

## What this is

This repo has two parts:

1. **`spikeparam/` — Python package** for fitting and extracting features from action potential waveforms. Given a matrix of aligned spike waveforms, it extracts: peak amplitude, peak sharpness, peak width, repolarization rate (`exp_lambda`), repolarization constant (`exp_const`), inflection time, and inter-spike interval (log ISI).

2. **`datasets/` — Analysis notebooks** for two complementary datasets that use the package to ask whether and how spike waveform features vary with neural input.

---

## Datasets

### `datasets/spe-1/` — Spontaneous variability, rat cortex (primary dataset)

Juxtacellular + LFP recordings from rat somatosensory cortex. Spikes are parameterized and clustered by waveform feature (Low / Mid / High groups). We then ask: do spikes from different waveform clusters occur during different LFP network states?

**Key analyses (`spe1_patch_LFP_analysis/cluster_analyses/`):**
| Notebook | What it does |
|---|---|
| `spk_waveform_cluster_comparisons.ipynb` | Population-level clustering QC — prevalence, metadata confounds, temporal drift |
| `lfp_spk_cluster_comparisons_target_cells.ipynb` | LFP × spike cluster sliding-window analysis — effect sizes, bootstrap, permutation tests, validated directions |
| `lfp_spk_cluster_metadata_full_comparisons_target_cells.ipynb` | Cross-dataset integration — does clustering quality predict LFP effect strength? |

**Helper modules (`spe1_helper_modules/`):**
| Module | Purpose |
|---|---|
| `spk_feat_cluster_comp_analysis.py` | Cluster QC, metadata correlations, temporal drift |
| `spk_lfp_cluster_comp_analysis.py` | LFP sliding-window stats, bootstrap, permutation, direction/timing plots |
| `data_loader.py` | Load spe-1 ephys + LFP |
| `signal_utils.py` | LFP preprocessing utilities |
| `lfp_spike_window_analysis.py` | Peri-spike LFP windowing |
| `spk_feat_cluster_analysis.py` | Per-cell spike clustering |

---

### `datasets/pvc-6/` — Controlled stimulation, cat V1

Patch-clamp recordings from cat primary visual cortex ([CRCNS PVC-6](http://crcns.org/data-sets/vc/pvc-6)). Cells were driven with constant current, ramp, and pink (1/f) noise. We test whether spike waveform features vary systematically with input drive and can decode or predict stimulus properties.

**Notebooks:**
| Notebook | What it does |
|---|---|
| `spk_waveform_stim_predictors_cell1.ipynb` | Full analysis pipeline for cell 1 |
| `spk_waveform_stim_predictors_cell2.ipynb` | Same pipeline for cell 2 |
| `spk_waveform_pink_noise_analysis_cell2.ipynb` | Exploratory pink noise analysis for cell 2 |

---

## Repository structure

```
spikeparam/
├── spikeparam/                    # Core Python package
│   ├── patch/                     # Patch-clamp spike parameterization
│   │   ├── features/              # intra.py, inter.py
│   │   ├── fit/                   # Waveform fitting
│   │   ├── points/                # Peak/trough/inflection detection
│   │   └── plts/                  # Package-level plotting
│   ├── gaussian/                  # Gaussian mixture model fitting
│   └── tests/                     # Unit tests (mirrors package structure)
│
├── AP_empirical_paper1/           # Everything for the empirical paper
│   ├── AP_empirical_paper1_fig1.ipynb
│   ├── AP_empirical_paper1_fig2_pvc6.ipynb
│   │
│   └── datasets/                  # All dataset analyses
│       ├── spe-1/                 # Primary dataset (juxtacellular + LFP, rat cortex)
│       │   ├── spe1_helper_modules/
│       │   ├── spe1_iapeap_tests/
│       │   └── spe1_patch_LFP_analysis/
│       │       └── cluster_analyses/   # Main notebooks + per-cell notebooks
│       │
│       ├── pvc-6/                 # Controlled stimulation (patch-clamp, cat V1)
│       │   ├── pvc6_helper_modules/
│       │   └── *.ipynb
│       │
│       └── old_work/              # Archived (not actively maintained)
│           ├── primate_dataset/
│           ├── spe1_kilosort/
│           ├── spe1_onlypatch_analysis/
│           └── foof_fit_plots/
│
├── docs/tutorials/                # Usage tutorials for the spikeparam package
├── shared_helper_modules/         # Cross-dataset utilities
└── params/                        # Pre-computed parameter files
```

---

## Install

```bash
git clone https://github.com/voytekresearch/spikeparam
cd spikeparam
pip install -e .
```

**Dependencies:** Python ≥ 3.6, numpy ≥ 1.18, scipy ≥ 1.4, matplotlib ≥ 3.0, bycycle ≥ 1.0, neurodsp ≥ 2.1

Optional: `tqdm` (progress bars), `pytest` (tests), `seaborn` (plotting)

---

## Plotting conventions

See [`AP_empirical_paper1/PLOTTING_GUIDELINES.md`](AP_empirical_paper1/PLOTTING_GUIDELINES.md) for color schemes, figure sizing, and style conventions used across all analysis notebooks.

---

## Funding

Supported by NIH award R01 GM134363 (NIGMS).
