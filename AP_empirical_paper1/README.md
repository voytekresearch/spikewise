# AP Empirical Paper 1 — Spike Waveform Variability and Network State

This folder contains everything for the first empirical paper: datasets, analysis notebooks, paper figure notebooks, and plotting conventions.

---

## Scientific question

Do spike action potential (AP) waveform features vary systematically with the state of the local network, and can that variability be predicted from or used to decode the input drive? We address this from two complementary angles:

- **Controlled stimulation (pvc-6):** Waveform features vary with the type and statistics of the injected current — establishing that input drive directly shapes waveform morphology.
- **Spontaneous variability (spe-1):** In naturalistic recordings, spikes can be clustered by waveform feature. Do spikes from different waveform clusters occur during different LFP network states? Pre-spike LFP effects suggest network state *predicts* upcoming waveform; post-spike effects suggest the spike shapes the local field.

---

## Folder structure

```
AP_empirical_paper1/
├── README.md                     ← this file
├── PLOTTING_GUIDELINES.md        ← colors, figure style, stats conventions
│
├── paper_aux_figs/               ← paper figure notebooks
│   ├── AP_empirical_paper1_fig1.ipynb
│   └── AP_empirical_paper1_fig2_pvc6.ipynb
│
└── datasets/
    ├── spe-1/                    ← primary dataset
    │   ├── spe1_helper_modules/  ← analysis modules
    │   ├── spe1_iapeap_tests/
    │   └── spe1_patch_LFP_analysis/
    │       └── cluster_analyses/ ← main population-level notebooks
    │
    ├── pvc-6/                    ← controlled stimulation dataset
    │   ├── pvc6_helper_modules/
    │   └── *.ipynb
    │
    └── old_work/                 ← archived (not actively maintained)
```

---

## Analysis overview

### spe-1 — spontaneous variability, rat cortex

| Step | Notebook | Description |
|---|---|---|
| 1 | `spe-1_load_data.ipynb` | Load raw electrophysiology and LFP |
| 2 | `spe-1_c*.ipynb` (per-cell) | Parameterize waveforms, cluster by feature |
| 3 | `spk_waveform_cluster_comparisons.ipynb` | Population-level cluster QC — which features cluster, metadata confounds, temporal drift |
| 4 | `lfp_spk_cluster_comparisons_target_cells.ipynb` | LFP sliding-window analysis — do waveform clusters track LFP state? Bootstrap + permutation validation, direction + timing |
| 5 | `lfp_spk_cluster_metadata_full_comparisons_target_cells.ipynb` | Does clustering quality predict LFP effect strength? Cross-dataset integration |

**Helper modules** (`spe1_helper_modules/`):
- `spk_feat_cluster_comp_analysis.py` — cluster QC, metadata correlations, temporal drift
- `spk_lfp_cluster_comp_analysis.py` — LFP windowing, bootstrap, permutation, direction/timing plots

---

### pvc-6 — controlled stimulation, cat V1

| Notebook | Description |
|---|---|
| `spk_waveform_stim_predictors_cell1/2.ipynb` | Full pipeline: parameterize → correlate → classify stim type → predict stim stats + log ISI |
| `spk_waveform_pink_noise_analysis_cell2.ipynb` | Exploratory pink noise analysis for cell 2 |

---

## Key analysis decisions

- **Waveform clusters**: Low / Mid / High based on per-feature k-means; assessed with nRMSE (mean waveform distance) and cos_sim (shape similarity).
- **LFP sliding window**: ±500 ms peri-spike, 100 ms windows, 10 ms steps. Kruskal-Wallis across cluster groups per window, FDR corrected.
- **Effect size**: Cohen's d (pooled SD).
- **Population validation**: bootstrap (resample cells, n=1000) + within-cell permutation (shuffle cluster labels, n=500) + binomial yield test.
- **Direction**: signed as High cluster vs. Low cluster mean in significant windows. Positive = high waveform feature → higher LFP value.
- **Timing**: categorized as pre-spike (< −50 ms), peri-spike (±50 ms), or post-spike (> +50 ms) based on peak Cohen's d window.
