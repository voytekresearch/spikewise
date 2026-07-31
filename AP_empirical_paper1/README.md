# AP_empirical_paper1

Empirical analyses for the spike waveform variability paper. Two datasets, one question: do spike waveform features vary with neural input and network state?

---

## Scientific question

Do action potential waveform features vary with the state of the local network, and can that variability be predicted from or used to decode the input/network state?

- **Controlled stimulation (pvc-6):** Waveform features vary with the type and statistics of the injected current — establishing that stimulation input directly shapes waveform morphology.
- **Spontaneous variability, simultaneous patch and npx (spe-1):** In spontaneous recordings, spikes can be clustered by waveform feature. Do spikes from different waveform clusters occur during different LFP network states? Do AP waveform and LFP covary?


---

## Folder structure

```
AP_empirical_paper1/
├── README.md                          ← this file
├── PLOTTING_GUIDELINES.md             ← colors, figure style, stats conventions
│
├── paper_aux_figs/
│   ├── algorithm/                     ← algorithm schematic, pink noise schematic, patch quality supp
│   ├── pvc6/                          ← pvc-6 dataset summary, Allen CT summary
│   └── spe1/                          ← spe-1 dataset summary, schematics, all supp figures
│
└── datasets/
    ├── spe-1/                         ← simultaneous npx and patch (see below)
    ├── pvc-6/                         ← controlled stimulation dataset (see below)
    ├── allen-cell-types/              ← ground-truth cell type validation (spiny/aspiny labels) TBD
    └── shared_helper_modules/         ← cross-dataset utilities
```

---

## spe-1 — Spontaneous variability, rat somatosensory cortex

Simultaneous juxtacellular patch-clamp and Neuropixels LFP recordings (n = 43 cells total). Three cells (c17, c18, c43) had insufficient spike counts for waveform clustering and are excluded from all analyses (n = 40 for clustering and LFP analyses; n = 39 for ridge regression, with c39 additionally excluded due to missing raw LFP data). Spikes are parameterized and clustered into Low / Mid / High groups by waveform feature. Main question: do spikes from different waveform clusters occur during different LFP network states?

### Notebooks

**Per-cell cluster + LFP analyses** (`spe1_spike_lfp_analysis/cluster_analyses/cell_analyses/cluster_feature_analyses/`):

| Notebook pattern | What it does |
|---|---|
| `spe-1_c{N}_clusters.ipynb` | Spike fitting → feature extraction → clustering → cluster quality report → saves pickles |
| `spe-1_c{N}_LFP_analysis.ipynb` | Priority cells only: loads cluster pickle → peri-spike LFP windows → specparam → sliding-window stats |
| `spe-1_c{N}_np_LFP_analysis.ipynb` | All cells: same pipeline via Neuropixels LFP channel |

`FORCE_CLUSTER / FORCE_LFP / FORCE_SIMPLE / FORCE_STATS` flags at top of each notebook — set `True` to rerun even if pickle exists.

**Per-cell ridge regression** (`cell_analyses/ridge_regression_analyses/spe-1_c{N}_spk_to_lfp_ridge.ipynb`):
- Loads cluster pickle → extracts LFP features in pre/post windows → 5-fold CV ridge regression → saves per-cell pickle

**Population-level** (`spe1_spike_lfp_analysis/cluster_analyses/population_analyses/`):

| Notebook | What it does |
|---|---|
| `pop_spk_waveform_clusters.ipynb` | Clustering prevalence across all cells; metadata confounds (recording type, depth, cell type); temporal drift; select priority cells |
| `pop_temporal_transitions.ipynb` | Detect within-cell temporal transitions in cluster membership; logistic sigmoid fitting; AIC model selection |
| `pop_around_transition.ipynb` | Peri-transition LFP analysis: compare LFP state before vs. after cluster transitions |
| `pop_within_vs_between_waveform.ipynb` | Within-cell vs. between-cell waveform variability; nRMSE distributions by cell type and recording method |
| `pop_transition_metadata.ipynb` | Which cells show transitions and in which features; metadata associations |
| `pop_ridge_regression.ipynb` | Pool per-cell ridge regression results; population R² tests; beta consistency; fraction-significant binomial tests |
| `pop_ridge_metadata.ipynb` | Which cells drive waveform→LFP predictability? Cell identity × R² and beta-direction analyses |

**Cell groups** (defined in `config.py`, derived in `pop_spk_waveform_clusters.ipynb` § G):
- `PRIORITY_CELLS = [3, 21, 22, 24, 26, 28, 45]` — top 7 cells by mean nRMSE (largest waveform differences across cluster groups)
- `HIGH_DIFF_LOW_DRIFT_CELLS = [15, 26, 46]` — top 3 by nRMSE among cells with low temporal drift (|mean ρ| < 0.2), providing a contrast group where cluster differences exist but are not confounded by time-in-recording

### Helper modules (`spe1_helper_modules/`)

| Module | Purpose |
|---|---|
| `config.py` | Per-cell metadata, sampling rates, `SPE1_DATA_ROOT`, `SPE1_PICKLE_ROOT`, `LFP_FS`, `NPX_FS` |
| `data_loader.py` | Load spe-1 binary recordings → `.npy` |
| `signal_utils.py` | Butterworth LFP filtering |
| `spk_feat_cluster_analysis.py` | Per-cell clustering, LFP windowing, specparam, sliding-window stats, `load_or_compute` |
| `spk_feat_cluster_comp_analysis.py` | Population clustering QC — prevalence, metadata associations, temporal drift |
| `spk_lfp_cluster_comp_analysis.py` | Population LFP-spike comparison utilities — peri-spike window analysis, effect sizes, population plots |
| `lfp_spike_window_analysis.py` | Peri-spike LFP windowing, sensitivity analysis |
| `spe1_plotting.py` | Dataset-specific plotting utilities |
| `ridge_regression_utils.py` | Per-cell 5-fold ridge regression, feature extraction, LFP target construction |
| `spe1_ridge_utils.py` | spe-1-specific ridge helpers (window constants, `FEAT_LABELS`) |
| `pop_ridge_utils.py` | Load/aggregate population ridge results, population tests, save population pickle |
| `pop_metadata_utils.py` | Build metadata DataFrame; plots for cell-level R², alpha, and beta×metadata analyses |

### Pickle layout (under `SPE1_PICKLE_ROOT`)

```
spe1_pickles/
├── spike_fit_pickles/       c{N}_spike_fit.pkl
├── cluster_pickles/         c{N}_cluster_df.pkl        ← df_features_clust (LFP pipeline)
│                            c{N}_cluster_report.pkl    ← master_report_df (population QC)
├── lfp_window_pickles/      c{N}_lfp_windows.pkl
├── simple_lfp_pickles/      c{N}_simple_lfp.pkl
├── multitaper_pickles/      c{N}/                      ← specparam chunks (pre-computed)
├── lfp_spk_group_pickles/   c{N}_sliding_stats.pkl
│                            c{N}_per_spike_data.pkl
└── ridge_regression_pickles/
    ├── c{N}_ridge_results.pkl        ← per-cell CV R², betas, significance
    └── population_ridge_results.pkl  ← aggregated r2_pop, sig_pop, beta_pop,
                                         df_tests, df_frac, df_r2, mean_beta_mat
```

## pvc-6 — Controlled stimulation, mouse visual cortex (in vitro)

Whole-cell current clamp slice recordings from mouse visual cortex ([CRCNS PVC-6](http://crcns.org/data-sets/vc/pvc-6)). Constant current, ramp, and pink (1/f) noise stimuli. Tests whether spike waveform features vary with input drive and can decode stimulus type.

**Main analyses** (`cell_analyses/`):

| Notebook | What it does |
|---|---|
| `spk_waveform_stim_predictors_cell1_main.ipynb` | Feature extraction, ridge regression, stimulus decoding — cell 1 |

**Supplementary** (`supplementary/`):

| Notebook | What it does |
|---|---|
| `spk_waveform_stim_predictors_cell1_supp.ipynb` | Extended analyses and supplementary figures — cell 1 |
| `spk_waveform_stim_predictors_cell2.ipynb` | Same pipeline for cell 2 (note: class imbalance limits stimulus classification) |
| `spk_waveform_stim_predictors_cell2_supp.ipynb` | Supplementary figures — cell 2 |
| `pvc6_supp_final_figures.ipynb` | Final supplementary figure assembly for pvc-6 |

**Helper modules** (`pvc6_helper_modules/`): `pvc6_load_data.py`, `pvc6_stim_analysis.py`, `pvc6_plotting.py`

---

## paper_aux_figs — Final figure notebooks

### algorithm/

| Notebook | What it does |
|---|---|
| `AP_empirical_algorithm_sch.ipynb` | Algorithm schematic figure — spike fitting and feature extraction pipeline |
| `AP_empirical_paper_pinknoise_sch.ipynb` | Pink noise stimulus schematic |
| `supp_patch_quality.ipynb` | Supplementary figure: patch-clamp recording quality metrics |

### pvc6/

| Notebook | What it does |
|---|---|
| `pvc6_dataset_summary.ipynb` | pvc-6 dataset summary figure |
| `allen_ct_dataset_summary.ipynb` | Allen Cell Types dataset summary figure |

### spe1/

| Notebook | What it does |
|---|---|
| `spe1_dataset_summary.ipynb` | spe-1 dataset summary figure |
| `spe1_schematics.ipynb` | spe-1 analysis schematics |
| `supp_cluster_characterization.ipynb` | Supp: cluster quality metrics, metadata associations, temporal drift |
| `supp_cluster_distributions.ipynb` | Supp: per-cell cluster feature distributions |
| `supp_temporal_trajectories.ipynb` | Supp: within-cell temporal transitions in cluster membership |
| `supp_peri_transition_lfp.ipynb` | Supp: LFP state before vs. after cluster transitions |
| `supp_lfp_by_feature_group_transitions.ipynb` | Supp: LFP environment grouped by transitioning waveform feature |
| `supp_within_vs_between_waveform.ipynb` | Supp: within-cell vs. between-cell waveform variability |
| `supp_waveform_isi_independence.ipynb` | Supp: waveform clusters are independent of ISI clusters — 2-row solid-fill KDE grid for c32 (row 1: colored by peak amplitude cluster; row 2: colored by ISI cluster) + population η² histogram (Kruskal-Wallis H/(n−1); log ISI ~ waveform cluster assignment across all cells with any waveform cluster and an ISI cluster; n = 33 cells) |
| `supp_intra_spike_correlations.ipynb` | Supp: within-cell Pearson r heatmaps (pvc-6 c1 & c2; spe-1 IN / PC / all / best individual cells); decay correlation direction analysis — per-cell r(exp_lambda, peak_amp) ranked strip plot (colored by cell type, shaped by clear EAP); c14 vs pvc-6 c1 side-by-side heatmap (best spe-1 match: 7/7 sign matches); clear EAP metadata analysis (raw p = 0.057, FDR p = 0.286) |
| `supp_ridge_regression.ipynb` | Supp: spike waveform → LFP ridge regression population results |
| `supp_r2_distributions.ipynb` | Supp: per-cell CV R² distributions across LFP targets |

---

## Statistical approach

| Analysis | Test | Effect size | Correction |
|---|---|---|---|
| Cluster group differences (per cell) | Mann-Whitney U / Kruskal-Wallis | η² | — |
| Temporal drift (per cell) | Spearman ρ (spike time × cluster label) | ρ | — |
| Metadata × cluster difference | Mann-Whitney U / Kruskal-Wallis / Spearman | rank-biserial r / η² / ρ | BH-FDR |
| Classifier accuracy (pvc-6) | Bootstrap (n=1000 train/test splits) | accuracy | — |
| Waveform → LFP ridge regression (spe-1) | Per-cell: permutation test (n=1,000); population R²: one-sided Wilcoxon (H₁: CV R²>0) | CV R² | None (raw α=0.05 per target) |
| Beta consistency across cells (spe-1) | One-sample t-test (H₁: mean β≠0); restricted to R²-significant targets | — | None (raw α=0.05) |
| Cell metadata × R² / beta direction | Kruskal-Wallis (categorical) / Spearman (continuous) | η² / ρ | BH-FDR |

---

## Plotting conventions

See [`PLOTTING_GUIDELINES.md`](PLOTTING_GUIDELINES.md) — Wong (2011) colorblind-safe palette for cluster groups (blue = low, green = mid, orange = high), figure sizing, axis style.
