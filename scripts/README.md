# scripts/

## run_spe1_batch.py

Headless batch runner for the spe-1 LFP-spike cluster analysis pipeline.
Replaces manual per-cell notebook execution with a single command that caches
each step independently (spike fit → clustering → LFP windowing → sliding-window stats).

The script is executable, so you can call it either way:

```bash
python scripts/run_spe1_batch.py --no-plot
# or
./scripts/run_spe1_batch.py --no-plot
```

### Quick start

```bash
# All cells, skip steps whose pickles already exist, no plots
python scripts/run_spe1_batch.py --no-plot

# Priority cells only, 4 parallel workers, force-redo LFP stats
python scripts/run_spe1_batch.py --priority --workers 4 --force-lfp --no-plot

# Specific cells
python scripts/run_spe1_batch.py --cells 21 24 42 --no-plot

# Full rerun from scratch
python scripts/run_spe1_batch.py --force-all --no-plot
```

### Flags

| Flag | Effect |
|---|---|
| `--cells 21 24 42` | Run specific cell numbers only |
| `--priority` | Run only `config.PRIORITY_CELLS` = [3, 21, 22, 24, 26, 28, 45] |
| `--workers N` | Parallel processes (default 1 = sequential) |
| `--force-fit` | Recompute spike fitting even if pickle exists |
| `--force-cluster` | Recompute clustering even if pickle exists |
| `--force-lfp` | Recompute LFP windowing + sliding stats |
| `--force-all` | All of the above |
| `--no-plot` | Skip all matplotlib rendering (much faster; stats still saved) |

### Path overrides

Data and pickle roots default to the paths in `config.py` but can be overridden
without editing any files:

```bash
export SPE1_DATA_ROOT="/path/to/Neuropixel Paired Recordings/Recordings"
export SPE1_PICKLE_ROOT="/path/to/spe1_pickles"
python scripts/run_spe1_batch.py --no-plot
```

### Pickle layout (under `SPE1_PICKLE_ROOT`)

```
spe1_pickles/
├── spike_fit_pickles/
│   └── {cnum}_spike_fit.pkl
├── cluster_pickles/
│   └── {cnum}_cluster_df.pkl
├── lfp_window_pickles/
│   └── {cnum}_lfp_windows.pkl
├── simple_lfp_pickles/
│   └── {cnum}_simple_lfp.pkl
├── multitaper_pickles/
│   └── c{cnum}/               ← specparam chunks (pre-computed, read-only)
└── lfp_spk_group_pickles/
    ├── {cnum}_sliding_stats.pkl
    └── {cnum}_per_spike_data.pkl
```

Each step only reruns if its pickle is missing or the matching `--force-*` flag is set.

---

## run_ridge_psd_cell.py

Runs the per-cell spike-to-LFP ridge regression pipeline for a single cell. Called by the batch runner or directly for a specific cell. Loads the cluster pickle, extracts pre/post-spike LFP features, runs 5-fold CV ridge regression with permutation testing, and saves a per-cell ridge results pickle.

```bash
python scripts/run_ridge_psd_cell.py --cell 21
```

---

## run_allen_ct_batch.py

Batch runner for the Allen Cell Types dataset (ground-truth cell type validation). Analogous to `run_spe1_batch.py` but for the allen-cell-types dataset.

---

## save_nocluster_feats.py

Creates `cluster_df.pkl` for cells that have spike fit pickles but no waveform clusters (c17, c18, c43). Saves waveform features + log ISI without cluster columns so these cells can participate in non-clustering analyses (e.g. intra-spike correlations, within-vs-between waveform variability). These cells are excluded from ridge regression and all cluster-based analyses.

```bash
python scripts/save_nocluster_feats.py
```

---

## benchmark_spe1_notebooks.py

Performance profiling script for the spe-1 analysis pipeline.
