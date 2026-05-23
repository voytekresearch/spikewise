# AP_empirical_paper1

Empirical analyses for the spike waveform variability paper. Two datasets, one question: do spike waveform features vary systematically with neural input and network state?

---

## Scientific question

Do action potential waveform features vary with the state of the local network, and can that variability be predicted from or used to decode the input drive?

- **Controlled stimulation (pvc-6):** Waveform features vary with the type and statistics of the injected current — establishing that input drive directly shapes waveform morphology.
- **Spontaneous variability (spe-1):** In naturalistic recordings, spikes can be clustered by waveform feature. Do spikes from different waveform clusters occur during different LFP network states?

---

## Folder structure

```
AP_empirical_paper1/
├── README.md                          ← this file
├── PLOTTING_GUIDELINES.md             ← colors, figure style, stats conventions
│
├── paper_aux_figs/
│   ├── AP_empirical_paper1_fig1.ipynb
│   └── AP_empirical_paper1_fig2_pvc6.ipynb
│
└── datasets/
    ├── spe-1/                         ← primary dataset (see below)
    ├── pvc-6/                         ← controlled stimulation dataset (see below)
    ├── shared_helper_modules/         ← cross-dataset utilities
    └── old_work/                      ← archived, not actively maintained
```

---

## spe-1 — Spontaneous variability, rat somatosensory cortex

Juxtacellular + Neuropixels LFP recordings (n = 44 cells). Spikes are parameterized and clustered into Low / Mid / High groups by waveform feature. Main question: do spikes from different waveform clusters occur during different LFP network states?

### Notebooks

**Per-cell clustering** (`spe1_patch_LFP_analysis/cluster_analyses/spe-1_c{N}_clusters.ipynb`):
- Spike fitting → feature extraction → clustering → cluster quality report → saves pickles
- `FORCE_CLUSTER = False` at top — set `True` to rerun even if cluster pickle exists

**Per-cell LFP analysis** (priority cells only, `spe-1_c{N}_LFP_analysis.ipynb`):
- Loads cluster pickle → extracts peri-spike LFP windows → loads pre-computed specparam → runs sliding-window stats
- `FORCE_LFP`, `FORCE_SIMPLE`, `FORCE_STATS` flags at top

**Population-level** (`spe1_patch_LFP_analysis/cluster_analyses/population_analyses/`):

| Notebook | What it does |
|---|---|
| `pop_spk_waveform_cluster_comparisons.ipynb` | Clustering prevalence across all cells; metadata confounds (recording type, depth, cell type); temporal drift |
| `pop_lfp_spk_cluster_comparisons_target_cells.ipynb` | LFP × spike cluster sliding-window analysis — effect sizes (Cohen's d), bootstrap CIs, within-cell permutation, direction and timing |
| `pop_lfp_spk_cluster_metadata_full_comparisons_target_cells.ipynb` | Does clustering quality predict LFP effect strength? Cross-cell integration |

**Cell groups** (defined in `config.py`, derived in `pop_spk_waveform_cluster_comparisons.ipynb` § G):
- `PRIORITY_CELLS = [8, 10, 19, 21, 24, 42, 45]` — top 7 cells by mean nRMSE (largest waveform differences across cluster groups)
- `HIGH_DIFF_LOW_DRIFT_CELLS = [5, 8, 14]` — top 3 by nRMSE among cells with low temporal drift (|mean ρ| < 0.2), providing a contrast group where cluster differences exist but are not confounded by time-in-recording

### Helper modules (`spe1_helper_modules/`)

| Module | Purpose |
|---|---|
| `config.py` | Per-cell metadata, sampling rates, `SPE1_DATA_ROOT`, `SPE1_PICKLE_ROOT`, `LFP_FS`, `NPX_FS` |
| `spk_feat_cluster_analysis.py` | Per-cell clustering, LFP windowing, specparam, sliding-window stats, `load_or_compute` |
| `spk_feat_cluster_comp_analysis.py` | Population clustering QC — prevalence, metadata associations, temporal drift |
| `spk_lfp_cluster_comp_analysis.py` | Population LFP-spike comparison — Cohen's d, bootstrap, within-cell permutation |
| `data_loader.py` | Load spe-1 binary recordings → `.npy` |
| `signal_utils.py` | Butterworth LFP filtering |
| `lfp_spike_window_analysis.py` | Peri-spike LFP windowing, sensitivity analysis |
| `spe1_plotting.py` | Dataset-specific plotting utilities |

### Pickle layout (under `SPE1_PICKLE_ROOT`)

```
spe1_pickles/
├── spike_fit_pickles/     c{N}_spike_fit.pkl
├── cluster_pickles/       c{N}_cluster_df.pkl        ← df_features_clust (LFP pipeline)
│                          c{N}_cluster_report.pkl    ← master_report_df (population QC)
├── lfp_window_pickles/    c{N}_lfp_windows.pkl
├── simple_lfp_pickles/    c{N}_simple_lfp.pkl
├── multitaper_pickles/    c{N}/                      ← specparam chunks (pre-computed)
└── lfp_spk_group_pickles/ c{N}_sliding_stats.pkl
                           c{N}_per_spike_data.pkl
```

---

## pvc-6 — Controlled stimulation, mouse visual cortex (in vitro)

Whole-cell current clamp slice recordings from mouse visual cortex ([CRCNS PVC-6](http://crcns.org/data-sets/vc/pvc-6)). Constant current, ramp, and pink (1/f) noise stimuli. Tests whether spike waveform features vary with input drive and can decode stimulus type.

| Notebook | What it does |
|---|---|
| `spk_waveform_stim_predictors_cell1.ipynb` | Feature extraction, Ridge regression, stimulus decoding — cell 1 |
| `spk_waveform_stim_predictors_cell2.ipynb` | Same for cell 2 |
| `spk_waveform_pink_noise_analysis_cell2.ipynb` | Exploratory pink noise analysis — cell 2 |

**Helper modules** (`pvc6_helper_modules/`): `pvc6_load_data.py`, `pvc6_stim_analysis.py`, `pvc6_plotting.py`

---

## Statistical approach

| Analysis | Test | Effect size | Correction |
|---|---|---|---|
| Cluster group differences (per cell) | Mann-Whitney U / Kruskal-Wallis | η² | — |
| Temporal drift (per cell) | Spearman ρ (spike time × cluster label) | ρ | — |
| Sliding-window LFP differences | Welch's t / F-test (vectorized) | Cohen's d | — |
| Population LFP robustness | Bootstrap CIs (n=1000, resample cells) | Cohen's d CI | — |
| Within-cell LFP specificity | Permutation test (n=500 label shuffles) | % cells significant | — |
| Metadata × cluster difference | Mann-Whitney U / Kruskal-Wallis / Spearman | rank-biserial r / η² / ρ | BH-FDR |
| Classifier accuracy (pvc-6) | Bootstrap (n=1000 train/test splits) | accuracy | — |

---

## Plotting conventions

See [`PLOTTING_GUIDELINES.md`](PLOTTING_GUIDELINES.md) — Wong (2011) colorblind-safe palette for cluster groups (blue = low, green = mid, orange = high), figure sizing, axis style.
