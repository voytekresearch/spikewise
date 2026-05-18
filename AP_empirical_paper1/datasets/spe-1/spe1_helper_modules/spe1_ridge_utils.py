# =============================================================================
# spe1_ridge_utils.py
#
# Utility functions for the IAP spike feature → LFP feature ridge regression.
#
# Two-level analysis:
#   (1) Per-cell  : RidgeCV betas + permutation-test p-values (FDR within cell)
#   (2) Population: one-sample t-test on betas across cells (FDR across all tests)
#
# Data model
# ----------
# LFP features are already computed per-spike as sliding-window time series and
# stored in pickles (same format used by all other spe-1 analyses):
#   - cluster_pickles/cXX_cluster_df.pkl      → spike waveform features + log_isi
#   - simple_lfp_pickles/cXX_simple_lfp.pkl  → per-spike LFP: lfp_mean, lfp_std
#   - multitaper_pickles/cXX/                 → per-spike specparam: exponent, band_aucs
#
# Pre/post LFP features are extracted by averaging the sliding-window values
# over a time window relative to the spike (win_avg), matching the pattern
# used in the existing per-cell ridge notebooks.
#
# Workflow
# --------
#   load_cell_data()         → loads the three pickle sources for one cell
#   build_cell_df()          → one DataFrame per cell (spike features + LFP features)
#   run_ridge_cell()         → betas + permutation p-values for one cell
#   run_population_analysis()→ t-stats + FDR p-values across cells
#   plot_*()                 → visualizations
#
# Dependencies
# ------------
#   numpy, pandas, scipy, sklearn, statsmodels, matplotlib, seaborn, tqdm
# =============================================================================

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats          import ttest_1samp
from sklearn.linear_model import RidgeCV
from statsmodels.stats.multitest import fdrcorrection
from tqdm.notebook        import tqdm

# ── Seaborn / matplotlib defaults (matching existing notebooks) ───────────────
sns.set_style("whitegrid", {"axes.grid": False})
sns.set(rc={"figure.figsize": (14, 8)})


# =============================================================================
# CONSTANTS
# =============================================================================

# IAP spike features — order matches sp.df_features / cluster_df
SPIKE_FEATURES = [
    "ramp_amp",
    "inflection_time",
    "inflection_amp",
    "peak_amp",
    "peak_width",
    "peak_sharpness",
    "exp_lambda",
    "exp_const",
]

# Optional: add log_isi to SPIKE_FEATURES when include_log_isi=True
SPIKE_FEATURES_WITH_ISI = SPIKE_FEATURES + ["log_isi"]

# LFP target names (pre and post epochs, 5 features each → 10 total)
LFP_TARGETS = [
    "pre_lfp_amp",   "pre_lfp_std",   "pre_exponent",   "pre_theta_auc",   "pre_gamma_auc",
    "post_lfp_amp",  "post_lfp_std",  "post_exponent",  "post_theta_auc",  "post_gamma_auc",
]

# Time windows for LFP feature averaging (seconds, relative to spike at t=0)
# 50 ms window on each side, excluding the ±5 ms immediately around the spike
# to avoid spike artifact contaminating the LFP estimate.
PRE_WIN  = (-0.055, -0.005)   # pre-spike:  55 ms → 5 ms before spike
POST_WIN = ( 0.005,  0.055)   # post-spike: 5 ms → 55 ms after spike

# Ridge regularization grid (same across all cells for fair comparison)
ALPHA_GRID = np.logspace(-3, 3, 100)

# Minimum number of spikes required to include a cell
MIN_SPIKES = 50

# Number of permutations for the per-cell null distribution
N_PERMS = 1000

# ── Readable labels for plots ─────────────────────────────────────────────────
FEATURE_LABELS = {
    "ramp_amp":        "Ramp slope",
    "inflection_time": "Inflection\ntime",
    "inflection_amp":  "Inflection\namp",
    "peak_amp":        "Peak amp",
    "peak_width":      "Peak width",
    "peak_sharpness":  "Peak\nsharpness",
    "exp_lambda":      "Decay λ",
    "exp_const":       "Decay const",
    "log_isi":         "log ISI",
}

TARGET_LABELS = {
    "pre_lfp_amp":   "Pre\nLFP amp",
    "pre_lfp_std":   "Pre\nLFP std",
    "pre_exponent":  "Pre\nexponent",
    "pre_theta_auc": "Pre\nθ AUC",
    "pre_gamma_auc": "Pre\nγ AUC",
    "post_lfp_amp":  "Post\nLFP amp",
    "post_lfp_std":  "Post\nLFP std",
    "post_exponent": "Post\nexponent",
    "post_theta_auc":"Post\nθ AUC",
    "post_gamma_auc":"Post\nγ AUC",
}

# Colorblind-friendly palette
COLORS = {
    "pre":  "#4C72B0",   # blue   — pre epoch
    "post": "#DD8452",   # orange — post epoch
    "sig":  "#C44E52",   # red    — significant
    "ns":   "#AAAAAA",   # grey   — non-significant
    "zero": "#333333",   # near-black — zero reference lines
}


# =============================================================================
# SECTION 1 — DATA LOADING & LFP FEATURE EXTRACTION
# =============================================================================

def win_avg(t_bins, arr, window):
    """
    Average a 1-D array over a time window.

    Parameters
    ----------
    t_bins : array-like
        Time axis in seconds (relative to spike at t=0).
    arr : array-like
        Values aligned to t_bins.
    window : tuple (t_start, t_end)
        Inclusive time window in seconds.

    Returns
    -------
    float — NaN if no samples fall in the window or all values are non-finite.
    """
    if t_bins is None or arr is None:
        return np.nan
    t = np.asarray(t_bins, float)
    a = np.asarray(arr,    float)
    mask = (t >= window[0]) & (t <= window[1])
    if not np.any(mask):
        return np.nan
    vals = a[mask]
    return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan


def load_cell_data(cell_num, pickle_root,
                   load_chunked_specparam_results, trim_edges,
                   edge_sec=0.5):
    """
    Load the three pickle sources for one cell.

    Parameters
    ----------
    cell_num : int
        Cell number (e.g. 14 for c14).
    pickle_root : str
        Path to spe1_pickles root directory (from config.SPE1_PICKLE_ROOT).
    load_chunked_specparam_results : callable
        Function from spk_feat_cluster_analysis.
    trim_edges : callable
        Function from spk_feat_cluster_analysis.
    edge_sec : float
        Edge duration (seconds) to trim from specparam results.

    Returns
    -------
    cluster_df : pd.DataFrame
        Spike waveform features + log_isi.
    lfp_windows_by_spike : list of dict
        Per-spike LFP sliding-window data with keys: t_bins_s, lfp_mean, lfp_std.
    specparam_by_spike : list of dict
        Per-spike specparam data with keys: t_bins_s, exponent, band_aucs.
    n_spikes : int
        Number of spikes (minimum of all three sources, after trimming).
    """
    # cluster_df — spike waveform features + log ISI
    cluster_path = os.path.join(pickle_root, 'cluster_pickles',
                                f'c{cell_num}_cluster_df.pkl')
    with open(cluster_path, 'rb') as f:
        cluster_df = pickle.load(f)

    # simple LFP pickles — lfp_mean, lfp_std per spike
    simple_path = os.path.join(pickle_root, 'simple_lfp_pickles',
                               f'c{cell_num}_simple_lfp.pkl')
    with open(simple_path, 'rb') as f:
        lfp_windows_by_spike = pickle.load(f)

    # specparam chunks — exponent, band_aucs per spike
    chunk_dir    = os.path.join(pickle_root, 'multitaper_pickles', f'c{cell_num}')
    chunk_prefix = f'c{cell_num}_specparam'
    specparam_by_spike = load_chunked_specparam_results(
        save_dir=chunk_dir, prefix=chunk_prefix
    )
    specparam_by_spike = trim_edges(specparam_by_spike, edge_sec=edge_sec)

    # align all three sources to the shortest
    n_spikes = min(len(cluster_df), len(lfp_windows_by_spike), len(specparam_by_spike))
    cluster_df           = cluster_df.iloc[:n_spikes].reset_index(drop=True)
    lfp_windows_by_spike = lfp_windows_by_spike[:n_spikes]
    specparam_by_spike   = specparam_by_spike[:n_spikes]

    return cluster_df, lfp_windows_by_spike, specparam_by_spike, n_spikes


def build_cell_df(cluster_df, lfp_windows_by_spike, specparam_by_spike,
                  pre_win=PRE_WIN, post_win=POST_WIN,
                  include_log_isi=True):
    """
    Build one merged DataFrame for a single cell containing:
      - IAP spike features (from cluster_df)
      - 10 LFP features: pre/post × {lfp_amp, lfp_std, exponent, theta_auc, gamma_auc}

    LFP features are extracted by averaging per-spike sliding-window values
    over the pre/post time windows — same approach as the existing per-cell
    ridge regression notebooks.

    Parameters
    ----------
    cluster_df : pd.DataFrame
        Output of loading the cluster pickle (spike features + log_isi).
    lfp_windows_by_spike : list of dict
        Per-spike LFP time series. Each dict has keys:
            't_bins_s' : 1-d array  (time in seconds relative to spike)
            'lfp_mean' : 1-d array  (mean LFP in the sliding window)
            'lfp_std'  : 1-d array  (std  LFP in the sliding window)
    specparam_by_spike : list of dict
        Per-spike specparam time series. Each dict has keys:
            't_bins_s'  : 1-d array
            'exponent'  : 1-d array  (aperiodic exponent)
            'band_aucs' : dict with keys 'theta', 'gamma'
    pre_win : tuple (t_start, t_end)
        Pre-spike averaging window in seconds (e.g. (-0.055, -0.005)).
    post_win : tuple (t_start, t_end)
        Post-spike averaging window in seconds (e.g. (0.005, 0.055)).
    include_log_isi : bool
        Include log_isi as a spike feature.

    Returns
    -------
    df : pd.DataFrame
        Shape (n_clean_spikes, n_spike_features + 10_lfp_features).
        Rows with any NaN are dropped.
    """
    spike_cols = SPIKE_FEATURES_WITH_ISI if include_log_isi else SPIKE_FEATURES
    spike_cols = [c for c in spike_cols if c in cluster_df.columns]

    n = len(cluster_df)

    # Pre-allocate LFP target arrays
    arrays = {name: np.full(n, np.nan) for name in LFP_TARGETS}

    for i in range(n):
        sw = lfp_windows_by_spike[i]
        sp = specparam_by_spike[i]

        t_sw      = sw['t_bins_s']    if sw is not None else None
        t_sp      = sp.get('t_bins_s') if sp else None
        band_aucs = sp.get('band_aucs', {}) if sp else {}

        for col_offset, win in [(0, pre_win), (5, post_win)]:
            # LFP amp + std from simple_lfp pickle
            arrays[LFP_TARGETS[col_offset + 0]][i] = win_avg(t_sw, sw['lfp_mean'] if sw else None, win)
            arrays[LFP_TARGETS[col_offset + 1]][i] = win_avg(t_sw, sw['lfp_std']  if sw else None, win)
            # Aperiodic exponent + band AUCs from specparam pickle
            arrays[LFP_TARGETS[col_offset + 2]][i] = win_avg(t_sp, sp.get('exponent')         if sp else None, win)
            arrays[LFP_TARGETS[col_offset + 3]][i] = win_avg(t_sp, band_aucs.get('theta'),                     win)
            arrays[LFP_TARGETS[col_offset + 4]][i] = win_avg(t_sp, band_aucs.get('gamma'),                     win)

    df = cluster_df[spike_cols].copy()
    for name, arr in arrays.items():
        df[name] = arr

    return df.dropna().reset_index(drop=True)


# =============================================================================
# SECTION 2 — RIDGE REGRESSION
# =============================================================================

def zscore_arr(x):
    """
    Z-score an array column-wise (if 2d) or element-wise (if 1d).
    Columns with zero variance are returned as zeros.
    """
    x  = np.asarray(x, dtype=float)
    mu = np.nanmean(x, axis=0)
    sd = np.nanstd(x,  axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd


def _sig_stars(p):
    """Return significance star string for a p-value."""
    if   np.isnan(p): return ""
    elif p < 0.001:   return "***"
    elif p < 0.01:    return "**"
    elif p < 0.05:    return "*"
    else:             return "ns"


def run_ridge_cell(df_cell, spike_features, lfp_targets=LFP_TARGETS,
                   alpha_grid=ALPHA_GRID, n_perms=N_PERMS,
                   random_state=42, show_progress=False):
    """
    Fit ridge regression for one cell and compute permutation-test p-values.

    Steps per LFP target
    --------------------
    1. Z-score X (spike features) and y (LFP target) within the cell.
       This makes all beta weights comparable — each β is in units of
       "SDs of y per SD of x", regardless of original units.
    2. Fit RidgeCV (5-fold CV over alpha_grid) → real betas.
    3. Permute y 1000× and refit → null beta distribution.
    4. Two-tailed p-value: fraction of |null betas| ≥ |real beta| per feature.
    5. FDR correct (Benjamini-Hochberg) across all feature × target combos
       within this cell.

    Parameters
    ----------
    df_cell : pd.DataFrame
        Output of build_cell_df() for one cell.
    spike_features : list of str
        X column names (IAP features).
    lfp_targets : list of str
        y column names (LFP features to predict).
    alpha_grid : 1d array
        Ridge regularization candidates (same grid for all cells).
    n_perms : int
        Number of permutations for null distribution.
    random_state : int
        Reproducibility seed.
    show_progress : bool
        Show tqdm bar over LFP targets.

    Returns
    -------
    betas_df : pd.DataFrame, shape (n_targets × n_features)
        Real ridge beta weights (z-scored units).
    pvals_fdr_df : pd.DataFrame, shape (n_targets × n_features)
        FDR-corrected permutation p-values.
    best_alphas : dict
        Best regularization alpha per LFP target.
    """
    rng = np.random.default_rng(random_state)
    X_z = zscore_arr(df_cell[spike_features].values)   # (n_spikes, n_features)

    betas_raw   = {}
    pvals_raw   = {}
    best_alphas = {}

    targets_iter = (
        tqdm(lfp_targets, desc="  LFP targets", leave=False)
        if show_progress else lfp_targets
    )

    for target in targets_iter:

        y_z = zscore_arr(df_cell[target].values)

        # ── real fit ──────────────────────────────────────────────────────────
        ridge = RidgeCV(alphas=alpha_grid, cv=5, fit_intercept=True)
        ridge.fit(X_z, y_z)

        real_betas          = ridge.coef_
        best_alphas[target] = ridge.alpha_
        betas_raw[target]   = real_betas

        # ── permutation null ─────────────────────────────────────────────────
        null_betas = np.zeros((n_perms, len(spike_features)))
        for p in range(n_perms):
            y_perm  = rng.permutation(y_z)
            r_perm  = RidgeCV(alphas=alpha_grid, cv=5, fit_intercept=True)
            r_perm.fit(X_z, y_perm)
            null_betas[p] = r_perm.coef_

        # two-tailed: fraction of |null β| ≥ |real β|
        pvals_raw[target] = np.mean(np.abs(null_betas) >= np.abs(real_betas), axis=0)

    # ── FDR correction within cell (all feature × target combinations) ────────
    betas_df = pd.DataFrame(betas_raw, index=spike_features).T
    pvals_df = pd.DataFrame(pvals_raw, index=spike_features).T

    flat_p          = pvals_df.values.ravel()
    _, flat_p_fdr   = fdrcorrection(flat_p)
    pvals_fdr_df    = pd.DataFrame(
        flat_p_fdr.reshape(pvals_df.shape),
        index=pvals_df.index,
        columns=pvals_df.columns,
    )

    return betas_df, pvals_fdr_df, best_alphas


def run_population_analysis(all_betas, lfp_targets=LFP_TARGETS, spike_features=None):
    """
    Population-level analysis: one-sample t-test on betas across cells,
    then FDR correction across all feature × target combinations.

    The t-test asks: "is the distribution of betas across cells displaced
    from zero?" — i.e., is this spike feature a consistent predictor of
    this LFP feature across the population?

    Parameters
    ----------
    all_betas : list of pd.DataFrame
        Each element is a betas_df from run_ridge_cell() for one cell.
    lfp_targets : list of str
    spike_features : list of str or None

    Returns
    -------
    mean_betas : pd.DataFrame  (n_targets × n_features)
    sem_betas  : pd.DataFrame
    tstats     : pd.DataFrame
    pvals_fdr  : pd.DataFrame
    n_cells_per_test : pd.DataFrame
    """
    if spike_features is None:
        spike_features = list(all_betas[0].columns)

    stacked = np.stack(
        [df.loc[lfp_targets, spike_features].values for df in all_betas], axis=0
    )   # (n_cells, n_targets, n_features)

    n_cells = stacked.shape[0]
    mean_b  = np.nanmean(stacked, axis=0)
    sem_b   = np.nanstd(stacked,  axis=0) / np.sqrt(n_cells)
    t_arr   = np.full_like(mean_b, np.nan)
    p_arr   = np.full_like(mean_b, np.nan)
    n_arr   = np.zeros_like(mean_b, dtype=int)

    for i, target in enumerate(lfp_targets):
        for j, feat in enumerate(spike_features):
            vals = stacked[:, i, j]
            vals = vals[~np.isnan(vals)]
            n_arr[i, j] = len(vals)
            if len(vals) >= 3:
                t, p        = ttest_1samp(vals, popmean=0.0)
                t_arr[i, j] = t
                p_arr[i, j] = p

    flat_p   = p_arr.ravel()
    mask     = ~np.isnan(flat_p)
    flat_fdr = np.full_like(flat_p, np.nan)
    _, flat_fdr[mask] = fdrcorrection(flat_p[mask])

    def _df(arr):
        return pd.DataFrame(arr, index=lfp_targets, columns=spike_features)

    return _df(mean_b), _df(sem_b), _df(t_arr), _df(flat_fdr.reshape(p_arr.shape)), _df(n_arr)


# =============================================================================
# SECTION 3 — VISUALIZATION
# =============================================================================

def plot_cell_betas(betas_df, pvals_fdr_df, cell_id,
                    spike_features=None, figsize=(16, 7)):
    """
    Bar chart of ridge beta weights for a single cell.
    Bars are colored red (FDR significant) or grey (ns).
    Layout: 2 rows (pre/post) × 5 columns (LFP features).
    """
    if spike_features is None:
        spike_features = list(betas_df.columns)

    pre_targets  = [t for t in LFP_TARGETS if t.startswith("pre")]
    post_targets = [t for t in LFP_TARGETS if t.startswith("post")]
    x = np.arange(len(spike_features))

    fig, axes = plt.subplots(2, 5, figsize=figsize, sharey=False)
    fig.suptitle(f"Ridge β weights — {cell_id}  (FDR-corrected)", fontsize=13, y=1.01)

    for row, targets in enumerate([pre_targets, post_targets]):
        epoch_col = "pre" if row == 0 else "post"
        for col, target in enumerate(targets):
            ax    = axes[row, col]
            betas = betas_df.loc[target, spike_features].values.astype(float)
            pvals = pvals_fdr_df.loc[target, spike_features].values.astype(float)
            colors = [COLORS["sig"] if p < 0.05 else COLORS["ns"] for p in pvals]

            ax.bar(x, betas, color=colors, edgecolor="white", linewidth=0.4, width=0.7)
            ax.axhline(0, color=COLORS["zero"], linewidth=0.9, linestyle="--", alpha=0.6)

            for xi, (b, p) in enumerate(zip(betas, pvals)):
                stars = _sig_stars(p)
                if stars not in ("ns", ""):
                    offset = np.sign(b) * (abs(b) * 0.15 + 0.03)
                    ax.text(xi, b + offset, stars, ha="center", va="bottom",
                            fontsize=8, color=COLORS["sig"], fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(
                [FEATURE_LABELS.get(f, f) for f in spike_features],
                rotation=45, ha="right", fontsize=7,
            )
            ax.set_title(TARGET_LABELS.get(target, target), fontsize=9, pad=4)
            ax.set_ylabel("β (z-score units)", fontsize=7)
            ax.tick_params(axis="y", labelsize=7)

            for spine in ax.spines.values():
                spine.set_edgecolor(COLORS[epoch_col])
                spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.show()


def plot_population_betas(all_betas, pvals_fdr, lfp_targets=LFP_TARGETS,
                           spike_features=None, figsize=(18, 9)):
    """
    Raincloud-style population summary:
    violin + individual cell dots + mean±SEM, for each spike feature × LFP target.
    Significant features (FDR p<0.05) get a star annotation.
    Layout: 2 rows (pre/post) × 5 columns (LFP features).
    """
    if spike_features is None:
        spike_features = list(all_betas[0].columns)

    stacked = np.stack(
        [df.loc[lfp_targets, spike_features].values for df in all_betas], axis=0
    )

    pre_targets  = [t for t in lfp_targets if t.startswith("pre")]
    post_targets = [t for t in lfp_targets if t.startswith("post")]
    x   = np.arange(len(spike_features))
    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(2, 5, figsize=figsize, sharey=False)
    fig.suptitle("Population ridge β weights — IAP → LFP  (FDR-corrected)", fontsize=14, y=1.01)

    for row, targets in enumerate([pre_targets, post_targets]):
        epoch_col = "pre" if row == 0 else "post"

        for col, target in enumerate(targets):
            ax    = axes[row, col]
            t_idx = lfp_targets.index(target)
            betas = stacked[:, t_idx, :]
            color = COLORS[epoch_col]

            parts = ax.violinplot(
                [betas[:, j] for j in range(len(spike_features))],
                positions=x, showmeans=False, showextrema=False, widths=0.6,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.3)

            for j in range(len(spike_features)):
                jitter = rng.uniform(-0.1, 0.1, size=betas.shape[0])
                ax.scatter(j + jitter, betas[:, j], color=color, alpha=0.65, s=18, zorder=3)

            means = np.nanmean(betas, axis=0)
            sems  = np.nanstd(betas, axis=0) / np.sqrt(betas.shape[0])
            ax.errorbar(x, means, yerr=sems, fmt="o", color="k",
                        markersize=5, linewidth=1.5, capsize=3, zorder=4)

            ax.axhline(0, color=COLORS["zero"], linewidth=0.9, linestyle="--", alpha=0.6)

            for j, feat in enumerate(spike_features):
                p     = pvals_fdr.loc[target, feat]
                stars = _sig_stars(p)
                if stars not in ("ns", ""):
                    y_top = np.nanmax(betas[:, j]) + 0.06
                    ax.text(j, y_top, stars, ha="center", fontsize=10,
                            color=COLORS["sig"], fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(
                [FEATURE_LABELS.get(f, f) for f in spike_features],
                rotation=45, ha="right", fontsize=7,
            )
            ax.set_title(TARGET_LABELS.get(target, target), fontsize=9, pad=4)
            ax.set_ylabel("β (z-score units)", fontsize=7)
            ax.tick_params(axis="y", labelsize=7)

            for spine in ax.spines.values():
                spine.set_edgecolor(COLORS[epoch_col])
                spine.set_linewidth(1.5)

    plt.tight_layout()
    plt.show()


def plot_beta_heatmap(mean_betas, pvals_fdr, spike_features=None,
                      title="Mean β weights across cells (population)",
                      figsize=(13, 5)):
    """
    Heatmap of mean beta weights (rows = LFP targets, cols = spike features).
    Each cell is annotated with the numeric value + significance stars.
    """
    if spike_features is None:
        spike_features = list(mean_betas.columns)

    annot = np.empty(mean_betas.shape, dtype=object)
    for i, t in enumerate(mean_betas.index):
        for j, f in enumerate(spike_features):
            val   = mean_betas.loc[t, f]
            stars = _sig_stars(pvals_fdr.loc[t, f])
            annot[i, j] = f"{val:.2f}" if stars in ("ns", "") else f"{val:.2f}\n{stars}"

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        mean_betas[spike_features].astype(float),
        annot=annot, fmt="",
        cmap="coolwarm", center=0,
        linewidths=0.5,
        xticklabels=[FEATURE_LABELS.get(f, f) for f in spike_features],
        yticklabels=[TARGET_LABELS.get(t, t)  for t in mean_betas.index],
        ax=ax,
        cbar_kws={"label": "mean β (z-score units)", "shrink": 0.7},
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.tick_params(axis="x", labelsize=9, rotation=30)
    ax.tick_params(axis="y", labelsize=9, rotation=0)
    plt.tight_layout()
    plt.show()


def plot_significant_scatters(cell_dfs, cell_ids, pvals_fdr, mean_betas,
                               spike_features=None, lfp_targets=LFP_TARGETS,
                               fdr_thresh=0.05, figsize_per_panel=(3.5, 3.5)):
    """
    For every feature × LFP target pair that survives population FDR correction,
    plot the raw relationship pooled across cells (z-scored within cell), with
    per-cell regression lines and pooled Pearson r annotated.

    Parameters
    ----------
    cell_dfs : dict {cell_id: pd.DataFrame}
        One df per cell from build_cell_df().
    cell_ids : list of str
        Cell IDs in the order they appear in cell_dfs.
    pvals_fdr : pd.DataFrame (n_targets × n_features)
        FDR-corrected population p-values.
    mean_betas : pd.DataFrame (n_targets × n_features)
        Mean beta weights — annotated on each panel.
    spike_features : list of str or None
    lfp_targets : list of str
    fdr_thresh : float
    figsize_per_panel : tuple
    """
    from scipy.stats import pearsonr

    if spike_features is None:
        spike_features = [c for c in SPIKE_FEATURES_WITH_ISI
                          if c in list(cell_dfs.values())[0].columns]

    sig_pairs = [
        (target, feat)
        for target in lfp_targets
        for feat   in spike_features
        if not np.isnan(pvals_fdr.loc[target, feat])
        and pvals_fdr.loc[target, feat] < fdr_thresh
    ]

    if len(sig_pairs) == 0:
        print('No significant feature × target pairs to plot.')
        return

    n_panels = len(sig_pairs)
    ncols    = min(n_panels, 4)
    nrows    = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(figsize_per_panel[0] * ncols,
                                       figsize_per_panel[1] * nrows))
    axes = np.array(axes).ravel()

    cell_palette = sns.color_palette('tab10', n_colors=len(cell_ids))

    for ax_idx, (target, feat) in enumerate(sig_pairs):
        ax = axes[ax_idx]
        all_x, all_y = [], []

        for ci, cell_id in enumerate(cell_ids):
            if cell_id not in cell_dfs:
                continue
            df = cell_dfs[cell_id]
            if feat not in df.columns or target not in df.columns:
                continue

            x_z = zscore_arr(df[feat].values)
            y_z = zscore_arr(df[target].values)
            color = cell_palette[ci]

            ax.scatter(x_z, y_z, color=color, alpha=0.25, s=6, zorder=2)

            m, b_val = np.polyfit(x_z, y_z, 1)
            x_line   = np.linspace(x_z.min(), x_z.max(), 50)
            ax.plot(x_line, m * x_line + b_val, color=color,
                    linewidth=1.5, alpha=0.85, zorder=3)

            all_x.extend(x_z.tolist())
            all_y.extend(y_z.tolist())

        if len(all_x) > 2:
            r, _ = pearsonr(all_x, all_y)
            p_fdr = pvals_fdr.loc[target, feat]
            b_pop = mean_betas.loc[target, feat]
            ax.set_title(
                f'{FEATURE_LABELS.get(feat, feat)}\n→ {TARGET_LABELS.get(target, target)}',
                fontsize=8, pad=3
            )
            ax.text(0.97, 0.05,
                    f'r = {r:.2f}\nβ = {b_pop:.2f} {_sig_stars(p_fdr)}',
                    transform=ax.transAxes, ha='right', va='bottom', fontsize=7.5,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              alpha=0.8, lw=0.5))

        ax.axhline(0, color=COLORS['zero'], linewidth=0.6, linestyle='--', alpha=0.4)
        ax.axvline(0, color=COLORS['zero'], linewidth=0.6, linestyle='--', alpha=0.4)
        ax.set_xlabel(f'{FEATURE_LABELS.get(feat, feat)} (z)', fontsize=8)
        ax.set_ylabel(f'{TARGET_LABELS.get(target, target)} (z)', fontsize=8)
        ax.tick_params(labelsize=7)

        ec = 'pre' if target.startswith('pre') else 'post'
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS[ec])
            spine.set_linewidth(1.2)

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=cell_palette[i], markersize=7, label=cid)
        for i, cid in enumerate(cell_ids) if cid in cell_dfs
    ]
    fig.legend(handles=handles, title='Cell', loc='lower right',
               fontsize=7, title_fontsize=8,
               bbox_to_anchor=(1.0, 0.0), framealpha=0.9)

    fig.suptitle(
        f'Significant IAP → LFP pairs  (population FDR p < {fdr_thresh})\n'
        f'Each color = one cell  |  lines = per-cell regression  |  z-scored within cell',
        fontsize=10, y=1.02
    )
    plt.tight_layout()
    plt.show()


def plot_cell_summary_scatter(df_cell, spike_features=None,
                               lfp_targets=LFP_TARGETS, figsize=(16, 10)):
    """
    Quick sanity check: scatter of each IAP feature (x) vs each LFP target (y)
    for a single cell, with Pearson r annotated.
    """
    from scipy.stats import pearsonr

    if spike_features is None:
        spike_features = [c for c in SPIKE_FEATURES_WITH_ISI if c in df_cell.columns]

    n_feats   = len(spike_features)
    n_targets = len(lfp_targets)

    fig, axes = plt.subplots(n_targets, n_feats, figsize=figsize)
    fig.suptitle("Sanity check: IAP features vs LFP targets", fontsize=12, y=1.005)

    for row, target in enumerate(lfp_targets):
        for col, feat in enumerate(spike_features):
            ax = axes[row, col]
            x  = df_cell[feat].values
            y  = df_cell[target].values

            ax.scatter(x, y, alpha=0.3, s=4,
                       color=COLORS["pre"] if "pre" in target else COLORS["post"])

            r, p = pearsonr(x, y)
            ax.set_title(f"r={r:.2f}", fontsize=7, pad=2,
                         color=COLORS["sig"] if p < 0.05 else COLORS["ns"])

            if row == n_targets - 1:
                ax.set_xlabel(FEATURE_LABELS.get(feat, feat), fontsize=7)
            if col == 0:
                ax.set_ylabel(TARGET_LABELS.get(target, target), fontsize=7)

            ax.tick_params(labelsize=6)

    plt.tight_layout()
    plt.show()
