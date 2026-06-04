"""
spk_feat_cluster_analysis.py
-----------------------------
Analysis utilities for identifying and visualizing spike waveform feature clustering,
temporal dynamics of cluster membership, and LFP-spike group comparisons.

Main pipeline:
  1. Cluster spikes by waveform features (cluster_multimodal_features)
  2. Report on cluster quality and temporal structure (plot_full_cluster_report)
  3. Extract LFP windows per spike group (extract_lfp_windows, build_lfp_groups_from_clusters)
  4. Run time-resolved specparam + simple LFP features per group (run_master_LFP_spk_analysis)
  5. Identify sliding windows with significant group differences (lfp_sliding_stats)
"""

import sys
import numpy as np

# numpy 2.0 removed trapz → trapezoid; patch for older specparam versions
if not hasattr(np, 'trapz'):
    np.trapz = np.trapezoid
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm import tqdm
import mne
from specparam import SpectralTimeModel
from typing import Optional, List, Tuple, Dict, Any, Literal, Callable, Union, Sequence
import scipy.stats as stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, probplot, f_oneway, kruskal, zscore, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import os
import re
import pickle
import gc


from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.ndimage import uniform_filter1d


# ------------------------------------------------------------------------------------------- #
# --------------------------------- Cache / IO utilities ------------------------------------ #
# ------------------------------------------------------------------------------------------- #

def load_or_compute(path: str, compute_fn, force: bool = False, verbose: bool = True):
    """
    Load a pickle if it exists and force=False; otherwise call compute_fn(), save, and return.

    Parameters
    ----------
    path : str
        Full path to the pickle file.
    compute_fn : callable
        Zero-argument callable whose return value is saved when the cache is missing or stale.
    force : bool
        If True, always recompute and overwrite the existing pickle.
    verbose : bool
        Print whether the result is loaded from cache or freshly computed.

    Returns
    -------
    The cached or freshly-computed result.

    Examples
    --------
    df_clust = load_or_compute(
        path=os.path.join(PICKLE_ROOT, f"{cell_num}_cluster_df.pkl"),
        compute_fn=lambda: cluster_multimodal_features(df_features, ...),
        force=FORCE_CLUSTER,
    )
    """
    if not force and os.path.exists(path):
        if verbose:
            print(f"  [cache] {os.path.basename(path)}")
        with open(path, "rb") as f:
            return pickle.load(f)
    if verbose:
        print(f"  [compute] {os.path.basename(path)}")
    result = compute_fn()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(result, f)
    return result


def _spike_half_width(w):
    """Number of samples at or above 50% of peak amplitude (half-width proxy)."""
    w = np.asarray(w, float)
    peak = np.max(np.abs(w))
    if peak == 0:
        return 0
    return int(np.sum(np.abs(w) >= 0.5 * peak))


def _iqr_filter_waveforms(wfs, iqr_thresh=1.5):
    """
    Remove waveforms that are outliers in peak amplitude OR half-width.
    Biologically implausible spikes (e.g. extremely narrow artifacts) are
    caught by the width filter even when their amplitude looks normal.
    Both filters use Q1 ± iqr_thresh * IQR per group.
    """
    if not wfs or iqr_thresh is None:
        return wfs

    peaks  = np.array([np.max(np.abs(w)) for w in wfs])
    widths = np.array([_spike_half_width(w) for w in wfs])

    def _bounds(arr):
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        return q1 - iqr_thresh * iqr, q3 + iqr_thresh * iqr

    p_lo, p_hi = _bounds(peaks)
    w_lo, w_hi = _bounds(widths)

    return [w for w, p, wd in zip(wfs, peaks, widths)
            if p_lo <= p <= p_hi and w_lo <= wd <= w_hi]


def compute_cluster_waveforms(sp, df, cluster_cols, outlier_thresh=1.5):
    """
    Compute peak-aligned average waveforms for every cluster group in every
    cluster column.  Results are saved to a lightweight pickle so the population
    notebook can visualize them without needing sp.spikes.

    Spikes whose peak amplitude falls outside Q1 ± outlier_thresh*IQR of their
    group are excluded from the average (plotting only — clustering unchanged).

    Parameters
    ----------
    sp : Spike
        Fitted Spike object containing the waveforms.
    df : pd.DataFrame
        Clustered spike DataFrame (must have 'spk_id' and the cluster columns).
    cluster_cols : list of str
        Columns ending in '_cluster' to process.
    outlier_thresh : float
        IQR multiplier for outlier removal (default 3.0). Pass None to disable.

    Returns
    -------
    dict  keyed by cluster_col, each value is a dict:
        {
          label_str: {'mean': ndarray, 'std': ndarray, 'n': int},
          ...
          't_axis': ndarray   # samples from peak (0 = peak)
        }
    """
    result = {}
    n_spikes = len(sp.spikes)

    for col in cluster_cols:
        groups = sorted(df[col].dropna().unique())

        # Gather waveforms per group with outlier removal, then pool for
        # shared peak alignment so all groups share the same t=0 reference.
        all_wfs, group_raw = [], {}
        for lab in groups:
            inds = df.loc[df[col] == lab, "spk_id"].astype(int).tolist()
            wfs  = [sp.spikes[i] for i in inds if i < n_spikes]
            wfs  = _iqr_filter_waveforms(wfs, outlier_thresh)
            group_raw[lab] = wfs
            all_wfs.extend(wfs)

        if not all_wfs:
            continue

        aligned = peak_align_waveforms(all_wfs)
        pre     = max(int(np.argmax(np.abs(w))) for w in all_wfs)
        total   = aligned.shape[1]
        t_axis  = np.arange(total) - pre

        col_result = {"t_axis": t_axis}
        offset = 0
        for lab in groups:
            n = len(group_raw[lab])
            if n == 0:
                continue
            arr = aligned[offset:offset + n]
            col_result[str(lab)] = {
                "mean": np.nanmean(arr, axis=0),
                "std":  np.nanstd(arr,  axis=0),
                "n":    n,
            }
            offset += n

        result[col] = col_result

    return result


# ------------------------------------------------------------------------------------------- #
# ------------------------------ Global cluster color registry ------------------------------ #
# ------------------------------------------------------------------------------------------- #

CLUSTER_COLORS = {
    "low":  "#1f77b4",   # blue
    "mid":  "#2ca02c",   # green
    "high": "#ff7f0e",   # orange
}

def get_cluster_color(label: str, fallback: str = "black") -> str:
    """Return consistent color for a cluster label."""
    return CLUSTER_COLORS.get(str(label), fallback)


def peak_align_waveforms(waveforms) -> np.ndarray:
    """
    Align each waveform so its own peak (max |amplitude|) is at t=0.

    Creates a common grid spanning the widest pre-peak and post-peak window
    across all waveforms. Shorter waveforms are NaN-padded.

    Parameters
    ----------
    waveforms : list of 1-D array-like
        Raw spike waveforms.

    Returns
    -------
    np.ndarray, shape (n_spikes, output_len)
        Peak-aligned waveforms; the peak of every spike is at column index `pre`
        where pre = max(peak_indices).
    """
    if not waveforms:
        return np.empty((0, 0))

    wfs       = [np.asarray(w, dtype=float) for w in waveforms]
    peak_idxs = [int(np.argmax(np.abs(w))) for w in wfs]

    pre   = max(peak_idxs)
    post  = max(len(w) - pk - 1 for w, pk in zip(wfs, peak_idxs))
    total = pre + post + 1

    aligned = np.full((len(wfs), total), np.nan)
    for k, (w, pk) in enumerate(zip(wfs, peak_idxs)):
        s = pre - pk
        aligned[k, s:s + len(w)] = w

    return aligned

# ------------------------------------------------------------------------------------------- #
# ------------------------------ Cluster features that show grouped data --------------------- #
# ------------------------------------------------------------------------------------------- #
def plot_spike_feature_distributions(
    df_features: pd.DataFrame,
    figsize_per_plot: tuple = (5, 4),
    bins: int = 50,
    color: str = "steelblue",
    kde: bool = True,
    hist: bool = True,
    remove_outliers: bool = True,  
    iqr_multiplier: float = 3,   
):
    """
    Plot distributions of all spike waveform features in a grid.
    Removes extreme outliers using IQR method.
    
    Parameters:
    -----------
    df_features : pd.DataFrame
        DataFrame containing spike features
    figsize_per_plot : tuple
        Size of each subplot (width, height)
    bins : int
        Number of bins for histograms
    color : str
        Color for the distributions
    kde : bool
        Whether to show KDE curve
    hist : bool
        Whether to show histogram bars
    remove_outliers : bool
        Whether to remove outliers using IQR method
    iqr_multiplier : float
        Multiplier for IQR outlier detection (higher = more inclusive)
    """
    
    SPIKE_FEATURES = [
        'ramp_amp', 
        'inflection_time', 
        'inflection_amp', 
        'peak_amp',
        'peak_width', 
        'peak_sharpness', 
        'exp_lambda', 
        'exp_const', 
        'log_isi'
    ]
    
    # Filter to only include features that exist in the dataframe
    features_to_plot = [f for f in SPIKE_FEATURES if f in df_features.columns]
    
    if not features_to_plot:
        print("No spike features found in dataframe!")
        return
    
    print(f"Plotting spike features...")
    if remove_outliers:
        print(f"Removing outliers using IQR method (multiplier={iqr_multiplier})")
    
    # Calculate grid dimensions
    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure
    fig, axes = plt.subplots(
        n_rows, 
        n_cols, 
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows)
    )
    
    # Flatten axes for easier indexing
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Store summary statistics for printing
    stats_summary = []
    
    # Plot each feature
    for idx, feat in enumerate(features_to_plot[:len(axes)]):
        ax = axes[idx]
        
        # Get raw data, convert to numeric and drop NaN
        data_raw = pd.to_numeric(df_features[feat], errors='coerce').dropna()
        
        if len(data_raw) == 0:
            ax.text(0.5, 0.5, "No valid data", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(feat)
            continue
        
        # Remove outliers if requested
        if remove_outliers:
            Q1 = data_raw.quantile(0.25)
            Q3 = data_raw.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            
            # Filter data
            data_clean = data_raw[(data_raw >= lower_bound) & (data_raw <= upper_bound)]
            
            # Calculate how many outliers removed
            n_outliers = len(data_raw) - len(data_clean)
            percent_removed = 100 * n_outliers / len(data_raw)
            
            # Store bounds for plotting
            bounds = (lower_bound, upper_bound)
        else:
            data_clean = data_raw
            n_outliers = 0
            percent_removed = 0
            bounds = None
        
        # Plot histogram with optional KDE
        sns.histplot(
            data_clean,
            ax=ax,
            bins=bins,
            color=color,
            kde=kde,
            stat="density",
            alpha=0.7 if hist else 0,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Add summary statistics
        mean_val = data_clean.mean()
        median_val = data_clean.median()
        std_val = data_clean.std()
        
        # Add vertical lines for mean and median
        ax.axvline(mean_val, color='red', linestyle='-', linewidth=1.5, alpha=0.7, label='Mean')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Median')
        
        # Add outlier bounds if outliers were removed
        if remove_outliers and bounds:
            ax.axvline(bounds[0], color='orange', linestyle=':', 
                      linewidth=1, alpha=0.5, label='IQR bound')
            ax.axvline(bounds[1], color='orange', linestyle=':', 
                      linewidth=1, alpha=0.5)
        
        # Create title with statistics
        if remove_outliers:
            title_lines = [
                f"{feat}",
                f"n={len(data_clean):,}/{len(data_raw):,} ({n_outliers} outliers removed)",
                f"μ={mean_val:.3f} | σ={std_val:.3f}"
            ]
        else:
            title_lines = [
                f"{feat}",
                f"n={len(data_clean):,}",
                f"μ={mean_val:.3f} | σ={std_val:.3f}"
            ]
        
        ax.set_title("\n".join(title_lines), fontsize=10)
        
        # Only add legend to first plot
        if idx == 0:
            legend_items = ['Mean', 'Median']
            if remove_outliers and bounds:
                legend_items.append('IQR bound')
            ax.legend(legend_items, fontsize=8, loc='upper right')
        
        # Add grid for readability
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Store statistics for summary
        stats_summary.append({
            'feature': feat,
            'n_total': len(data_raw),
            'n_clean': len(data_clean),
            'n_outliers': n_outliers,
            'percent_outliers': percent_removed,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'bounds': bounds
        })
    
    # Hide unused subplots
    for idx in range(len(features_to_plot), len(axes)):
        axes[idx].set_visible(False)
    
    # Add overall title
    if remove_outliers:
        plt.suptitle(f"Spike Feature Distributions (IQR×{iqr_multiplier} outlier removal)", 
                    fontsize=14, y=1.02)
    else:
        plt.suptitle("Spike Feature Distributions", fontsize=14, y=1.02)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics table
    print("\n" + "="*90)
    print("FEATURE DISTRIBUTION SUMMARY")
    print("="*90)
    
    if remove_outliers:
        print(f"{'Feature':<20} {'Total N':<10} {'Clean N':<10} {'Outliers':<10} "
              f"{'% Removed':<10} {'Mean':<10} {'Std':<10}")
        print("-"*90)
        
        for stats in stats_summary:
            print(f"{stats['feature']:<20} "
                  f"{stats['n_total']:<10} "
                  f"{stats['n_clean']:<10} "
                  f"{stats['n_outliers']:<10} "
                  f"{stats['percent_outliers']:<10.1f} "
                  f"{stats['mean']:<10.3f} "
                  f"{stats['std']:<10.3f}")
    else:
        print(f"{'Feature':<20} {'N':<10} {'Mean':<12} {'Std':<12} {'Median':<12}")
        print("-"*66)
        
        for stats in stats_summary:
            print(f"{stats['feature']:<20} "
                  f"{stats['n_total']:<10} "
                  f"{stats['mean']:<12.3f} "
                  f"{stats['std']:<12.3f} "
                  f"{stats['median']:<12.3f}")
    
    print("="*90)
    


def load_chunked_specparam_results(save_dir: str, prefix: str = "c21_specparam"):
    """
    Sequentially loads chunked pickle files from a directory and assembles them 
    into a single master list, managing memory carefully to prevent kernel crashes.
    
    Parameters:
    -----------
    save_dir : str
        The path to the folder containing the chunked .pkl files.
    prefix : str
        The string that the chunk files start with (e.g., 'c21_specparam').
        
    Returns:
    --------
    specparam_by_spike : list
        The fully reassembled list of specparam results across all spikes.
    """
    if not os.path.exists(save_dir):
        raise FileNotFoundError(f"The directory {save_dir} does not exist.")

    # 1. Find all matching files
    all_files = [f for f in os.listdir(save_dir) if f.startswith(prefix) and f.endswith(".pkl")]
    
    if len(all_files) == 0:
        print(f"No files found in {save_dir} starting with '{prefix}'.")
        return []

    # 2. Sort them numerically so spikes stay in temporal order
    def extract_chunk_number(filename):
        match = re.search(r'_(\d+)\.pkl$', filename)
        return int(match.group(1)) if match else -1

    all_files.sort(key=extract_chunk_number)
    print(f"Found {len(all_files)} chunk files. Assembling master list...")

    # 3. Memory-safe loading — drop SpectralTimeModel objects (large), keep powers
    specparam_by_spike = []

    for file_name in tqdm(all_files, desc="Loading Chunks", file=sys.stdout, dynamic_ncols=False):
        file_path = os.path.join(save_dir, file_name)

        with open(file_path, 'rb') as f:
            chunk_data = pickle.load(f)

        for spike in chunk_data:
            spike.pop("model", None)  # drop fitted model object, keep powers/freqs

        specparam_by_spike.extend(chunk_data)
        del chunk_data
        gc.collect()

    print(f"Success! Master list assembled with {len(specparam_by_spike)} total spikes.")
    return specparam_by_spike
# Post-process: remove first/last 0.5s epochs
def trim_edges(results, edge_sec=0.5):
    """
    Trims temporal edges in-place to prevent memory duplication and kernel crashes 
    when processing massive datasets (e.g., 10,000+ spikes).
    """
    # Iterate directly over the original list
    for out_w in results:
        t_bins = out_w["t_bins_s"]
        
        # Calculate the mask
        mask = (t_bins >= (t_bins.min() + edge_sec)) & (t_bins <= (t_bins.max() - edge_sec))
        
        # Overwrite the arrays IN-PLACE
        out_w["t_bins_s"] = out_w["t_bins_s"][mask]
        out_w["epoch_idx"] = out_w["epoch_idx"][mask]
        if "powers" in out_w:
            out_w["powers"] = out_w["powers"][mask, :]
        out_w["offset"] = out_w["offset"][mask]
        out_w["exponent"] = out_w["exponent"][mask]
        out_w["r_squared"] = out_w["r_squared"][mask]
        
        # Update dictionaries in-place
        for b in list(out_w["band_aucs"].keys()):
            out_w["band_aucs"][b] = out_w["band_aucs"][b][mask]
            
        # Knee may be None or array
        if out_w.get("knee") is not None:
            out_w["knee"] = out_w["knee"][mask]
            

    return results



def get_cluster_colors_and_labels(cluster_col, unique_clusters, df=None, clustered_feature=None):
    """
    Get consistent colors and labels for clusters.
    
    Returns: (groups, group_names, color_map)
    """
    # Check if clusters are already labeled as 'low', 'mid', 'high'
    has_meaningful_labels = all(str(cluster) in ['low', 'mid', 'high'] for cluster in unique_clusters)
    n_clusters = len(unique_clusters)
    
    if has_meaningful_labels:
        print(f"  Using existing cluster labels: {sorted(unique_clusters)}")
        
        if n_clusters == 2:
            groups = tuple(sorted(unique_clusters, key=lambda x: 0 if x == 'low' else 1))
            group_names = ("Low", "High")
            color_map = {
                'low': "#1f77b4",   # Blue for Low
                'high': "#ff7f0e",  # Orange for High
                "default": "#9467bd"
            }
            
        elif n_clusters == 3:
            groups = tuple(sorted(unique_clusters, key=lambda x: {'low': 0, 'mid': 1, 'high': 2}[x]))
            group_names = ("Low", "Mid", "High")
            color_map = {
                'low': "#1f77b4",   # Blue for Low
                'mid': "#2ca02c",   # Green for Mid
                'high': "#ff7f0e",  # Orange for High
                "default": "#9467bd"
            }
        
        return groups, group_names, color_map
        
    else:
        # Generic labels - sort by clustered feature value if possible
        if clustered_feature and df is not None and clustered_feature in df.columns:
            cluster_means = {}
            for cluster in unique_clusters:
                cluster_data = df.loc[df[cluster_col] == cluster, clustered_feature].dropna()
                if len(cluster_data) > 0:
                    cluster_means[cluster] = cluster_data.mean()
            
            if len(cluster_means) == n_clusters:
                sorted_clusters = sorted(cluster_means.items(), key=lambda x: x[1])
                
                if n_clusters == 2:
                    low_cluster = sorted_clusters[0][0]
                    high_cluster = sorted_clusters[1][0]
                    
                    groups = (low_cluster, high_cluster)
                    group_names = ("Low", "High")
                    color_map = {
                        low_cluster: "#1f77b4",
                        high_cluster: "#ff7f0e",
                        "default": "#9467bd"
                    }
                    
                    print(f"  Assigning by {clustered_feature}: {low_cluster}→'Low' (blue), {high_cluster}→'High' (orange)")
                    
                elif n_clusters == 3:
                    low_cluster = sorted_clusters[0][0]
                    mid_cluster = sorted_clusters[1][0]
                    high_cluster = sorted_clusters[2][0]
                    
                    groups = (low_cluster, mid_cluster, high_cluster)
                    group_names = ("Low", "Mid", "High")
                    color_map = {
                        low_cluster: "#1f77b4",
                        mid_cluster: "#2ca02c",
                        high_cluster: "#ff7f0e",
                        "default": "#9467bd"
                    }
                    
                    print(f"  Assigning by {clustered_feature}: {low_cluster}→'Low' (blue), {mid_cluster}→'Mid' (green), {high_cluster}→'High' (orange)")
                    
                return groups, group_names, color_map
        
        # Fallback to alphabetical order
        groups = tuple(sorted(unique_clusters))
        group_names = tuple([f"Group {i+1}" for i in range(n_clusters)])
        import matplotlib.pyplot as plt
        cmap = plt.cm.get_cmap('tab20', n_clusters)
        color_map = {groups[i]: cmap(i) for i in range(n_clusters)}
        color_map["default"] = "#9467bd"
        
        return groups, group_names, color_map




def plot_full_cluster_report(
    df: pd.DataFrame,
    sp,
    cluster_col: str,
    cluster_group_id: int,
    time_col: str = "spk_times_ms",
    time_unit: str = "ms",
    bin_size_ms: int = 1000,
    sigma_bins: int = 2,
    heatmap_cmap: str = "magma",
    stats_alpha: float = 0.05,
):
    """
    Full diagnostic plotting report for a given cluster column.
    Includes statistical comparisons for the CLUSTERED FEATURE ONLY.
    """
 

    SPIKE_WAVEFORM_FEATURES = [
        "ramp_amp",
        "inflection_time",
        "inflection_amp",
        "peak_amp",
        "peak_width",
        "peak_sharpness",
        "exp_lambda",
        "exp_const",
        "log_isi",
    ]

    if cluster_col not in df.columns:
        raise KeyError(f"{cluster_col} not found in dataframe")
    
    # Get unique clusters and count
    unique_clusters = df[cluster_col].dropna().unique()
    n_clusters = len(unique_clusters)
    
    print(f"\n===== CLUSTER REPORT: {cluster_col} =====")
    print(f"Found {n_clusters} clusters: {sorted(unique_clusters)}")
    
    # --------------------------------------------------
    # COLOR AND LABEL ASSIGNMENT (AT THE TOP)
    # --------------------------------------------------
    print("\n→ Determining cluster colors and labels...")
    
    # Get the clustered feature name
    if cluster_col.endswith('_cluster'):
        clustered_feature = cluster_col.replace('_cluster', '')
    else:
        clustered_feature = None
        for feat in SPIKE_WAVEFORM_FEATURES:
            if cluster_col == f"{feat}_cluster":
                clustered_feature = feat
                break
    
    # Get consistent colors and labels
    groups, group_names, color_map = get_cluster_colors_and_labels(
        cluster_col, unique_clusters, df, clustered_feature
    )
    
    print(f"  Groups: {groups}")
    print(f"  Group names: {group_names}")
    
    # --------------------------------------------------
    # A) Spike waveforms
    # --------------------------------------------------
    print("\n→ 1. Plotting spike waveforms by cluster")
    _, ax_wf = plot_spike_clusters_from_df(df, sp, cluster_col,
                                           plot_average=True, plot_average_std=True,
                                           peak_align=True)
    ax_wf.set_xlim(-4, 4)
    ax_wf.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.5)
    plt.tight_layout(); plt.show()
    
    # --------------------------------------------------
    # A.5) Average waveforms with visual RMSE
    # --------------------------------------------------
    print("\n→ 1.5 Average waveforms with visual RMSE")
    
    # Now we have color_map, groups, group_names defined
    wf_metrics = avg_waveforms_rmse(sp, df, cluster_col, groups, group_names, color_map)
    
    # --------------------------------------------------
    # B) Cluster proportions over time
    # --------------------------------------------------
    print("\n→ 2. Plotting cluster proportions over time")
    plot_clusters_over_time_min(
        df,
        time_col=time_col,
        label_col=cluster_col,
        time_unit=time_unit,
        bin_size_ms=bin_size_ms,
        sigma_bins=sigma_bins,
    )

    # --------------------------------------------------
    # C) Transition matrix
    # --------------------------------------------------
    print("\n→ 3. Computing cluster transition matrix")
    trans_mat = cluster_transition_matrix(df, cluster_col)

    plt.figure(figsize=(5.5, 4.5))
    order = list(trans_mat.index)
    sns.heatmap(
        trans_mat.loc[order, order],
        annot=True,
        cmap=heatmap_cmap,
        cbar=False,
    )
    plt.title(f"{cluster_col} transition probabilities")
    plt.xlabel("Next spike cluster")
    plt.ylabel("Current spike cluster")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # D) Feature distribution diagnostics
    # --------------------------------------------------
    print("\n→ 4. Visualizing feature distributions by cluster")
    
    # Just use the already-defined groups, group_names, color_map
    visualize_feature_groups_hist(
        df,
        features=SPIKE_WAVEFORM_FEATURES,
        group_col=cluster_col,
        groups=groups,
        group_names=group_names,
        color_map=color_map,
        common_norm=True,
        alpha=0.5
    )

    # --------------------------------------------------
    # E) Statistical comparisons for the CLUSTERED FEATURE 
    # --------------------------------------------------
    print(f"\n→ 5. Statistical analysis for: {clustered_feature}")
    print("-" * 40)
    
    # Get group data
    group_data = []
    for group in groups:
        data = df.loc[df[cluster_col] == group, clustered_feature].dropna().values
        group_data.append(data)
    
    # Compute statistics
    stats_results = compute_cluster_statistics(group_data, alpha=stats_alpha)
    
    if stats_results and stats_results['p_value'] is not None:
        # Print results
        print(f"\nTest: {stats_results['test_name']} ({stats_results['test_type']})")
        print(f"Number of groups: {stats_results['n_groups']}")
        print(f"P-value: {stats_results['p_value']:.6f}")
        print(f"Significance: {stats_results['significance']}")
        print(f"η² effect size: {stats_results['eta_squared']:.3f}")
        print(f"Interpretation: {stats_results['interpretation']}")
        
        # Simple visualization
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Box plot
        positions = range(len(group_names))
        ax.boxplot(group_data, labels=group_names)
        
        # Add effect size annotation
        ax.text(0.5, 0.95, 
                f"η² = {stats_results['eta_squared']:.3f} ({stats_results['interpretation']})",
                transform=ax.transAxes, ha='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add p-value annotation
        ax.text(0.5, 0.88, 
                f"p = {stats_results['p_value']:.4f} {stats_results['significance']}",
                transform=ax.transAxes, ha='center',
                color='red' if stats_results['significance'] != 'ns' else 'black')
        
        ax.set_ylabel(clustered_feature)
        ax.set_title(f'{clustered_feature} by Cluster')
        plt.tight_layout()
        plt.show()
        
        stats_summary = stats_results
    else:
        print(" Could not compute statistics")
        stats_summary = None

    print("\n===== REPORT COMPLETE =====\n")
    # --------------------------------------------------
    # F) Construct the rows with nRMSE info for the Master DF
    # --------------------------------------------------
    
    #Generate Pairwise Rows for nRMSE Ranking

    # --------------------------------------------------
    # G) Temporal structure: Spearman rho between spike time and cluster label
    # --------------------------------------------------
    temporal_rho, temporal_p = np.nan, np.nan
    if time_col in df.columns and len(groups) >= 2:
        ordinal_map = {g: i for i, g in enumerate(groups)}
        df_valid = df[[time_col, cluster_col]].dropna()
        df_valid = df_valid[df_valid[cluster_col].isin(groups)]
        ordinal_labels = df_valid[cluster_col].map(ordinal_map)
        if len(ordinal_labels) > 2 and ordinal_labels.nunique() > 1:
            result = spearmanr(df_valid[time_col].values, ordinal_labels.values)
            temporal_rho = result.statistic
            temporal_p   = result.pvalue

    group_indices = range(len(groups))
    pairs = list(combinations(group_indices, 2))

    rows = []
    for idx1, idx2 in pairs:
        g1, g2 = group_names[idx1], group_names[idx2]

        # Match the key in wf_metrics (checking both orders)
        pair_key = f"{g1}_vs_{g2}"
        alt_key = f"{g2}_vs_{g1}"
        metrics = wf_metrics.get(pair_key) or wf_metrics.get(alt_key) or {}

        row = {
            "cluster_group_id": cluster_group_id,
            "feature_clustered": clustered_feature,
            "groups": f"{g1}-{g2}",
            "nRMSE": metrics.get('nrmse', np.nan),
            "cos_sim": metrics.get('cos_sim', np.nan),
            "temporal_rho": temporal_rho,
            "temporal_p": temporal_p,
        }
        rows.append(row)
    return pd.DataFrame(rows)
    


def plot_spike_clusters_from_df(
    df,
    sp,
    cluster_col,
    mode="full",
    plot_average=True,
    plot_average_std=True,
    colors=None,
    title=None,
    figsize=(14, 4),
    peak_align=True,
):
    """
    Plot spike waveforms grouped by clusters using df[‘spk_id’] to index spikes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain cluster_col and ‘spk_id’ (index into sp.spikes).
    sp : Spike
        Spike object containing the waveforms.
    cluster_col : str
        Column name for cluster labels.
    peak_align : bool
        If True (default), align all waveforms by peak before plotting so
        shape differences across clusters are not confounded by timing offsets.
    """
    labels      = sorted(df[cluster_col].dropna().unique().tolist())
    color_map   = {lab: get_cluster_color(lab) for lab in labels}
    if colors:
        color_map.update(colors)

    ind_groups  = []
    group_names = []
    for lab in labels:
        inds = df.loc[df[cluster_col] == lab, "spk_id"].astype(int).tolist()
        if inds:
            ind_groups.append(inds)
            group_names.append(str(lab))

    fig, ax = plt.subplots(figsize=figsize)

    sp.plot(
        inds=None,
        mode=mode,
        in_ms=True,
        show_points=False,
        ax=ax,
        groups=True,
        ind_groups=ind_groups,
        group_names=group_names,
        plot_average=plot_average,
        plot_average_std=plot_average_std,
        peak_align=peak_align,
    )

    # Apply cluster color scheme to the plotted lines
    for line in ax.lines:
        lab = line.get_label()
        for cluster_label in labels:
            if str(cluster_label) in lab:
                line.set_color(color_map[cluster_label])

    suffix = " (peak-aligned)" if peak_align else ""
    ax.set_title(title or f"Spike waveforms grouped by {cluster_col}{suffix}")
    plt.tight_layout()

    return fig, ax



def avg_waveforms_rmse(sp, df, cluster_col, groups, group_names, color_map, peak_align=True):
    """
    Plot average waveforms with 3 distinct panels per pair.
    
    1. Raw (Absolute Amplitude) -> RMSE
    2. Scaled (Relative Amplitude) -> NRMSE
    3. Z-Scored (Pure Shape) -> Cosine Similarity
    """
    
    # --- 1. Data Extraction ---
    avg_waveforms = {}
    waveform_counts = {}
    
    # Collect all waveforms across all groups first so peak alignment uses a
    # shared reference — guarantees all groups end up with the same length.
    group_raw_wfs = {}
    all_wfs_flat  = []
    for group in groups:
        spike_indices = df.loc[df[cluster_col] == group, "spk_id"].astype(int).tolist()
        wfs = [sp.spikes[idx] for idx in spike_indices if idx < len(sp.spikes)]
        group_raw_wfs[group] = wfs
        all_wfs_flat.extend(wfs)

    if peak_align and all_wfs_flat:
        all_aligned = peak_align_waveforms(all_wfs_flat)
        offset = 0
        for group in groups:
            n = len(group_raw_wfs[group])
            if n == 0:
                continue
            arr = all_aligned[offset:offset + n]
            avg_waveforms[group]    = np.nanmean(arr, axis=0)
            waveform_counts[group]  = n
            offset += n
    else:
        for group in groups:
            wfs = group_raw_wfs[group]
            if not wfs:
                continue
            avg_waveforms[group]   = np.nanmean(np.array(wfs), axis=0)
            waveform_counts[group] = len(wfs)

    n_groups = len(avg_waveforms)
    if n_groups < 2: return {}

    # --- 2. Helper to calculate metrics ---
    def get_metrics(wf1, wf2):
        # A. Raw RMSE
        rmse = np.sqrt(np.mean((wf1 - wf2)**2))
        
        # B. Max-Scaled (Preserves relative amplitude ratio)
        global_max = max(np.abs(wf1).max(), np.abs(wf2).max())
        if global_max == 0: global_max = 1e-9
        n_wf1, n_wf2 = wf1 / global_max, wf2 / global_max
        nrmse = np.sqrt(np.mean((n_wf1 - n_wf2)**2))
        
        # C. Z-Scored (Pure Shape / Cosine Proxy)
        z_wf1, z_wf2 = zscore(wf1), zscore(wf2)
        
        # Cosine Similarity
        cos_sim = cosine_similarity(wf1.reshape(1, -1), wf2.reshape(1, -1))[0][0]
        
        return rmse, nrmse, cos_sim, n_wf1, n_wf2, z_wf1, z_wf2

    # Trim all average waveforms to the common non-NaN region so metrics are NaN-free
    avg_arrays = np.array([avg_waveforms[g] for g in avg_waveforms])
    valid      = np.all(np.isfinite(avg_arrays), axis=0)
    if valid.any():
        lo, hi = int(np.argmax(valid)), int(len(valid) - np.argmax(valid[::-1]))
        for g in avg_waveforms:
            avg_waveforms[g] = avg_waveforms[g][lo:hi]

    groups_list = list(avg_waveforms.keys())
    rmse_results = {}
    n_pairs = n_groups * (n_groups - 1) // 2

    # --- 3. Plotting (3 Columns per Pair) ---
    # We increase width to accommodate the 3rd panel
    fig, axes = plt.subplots(n_pairs, 3, figsize=(18, 5 * n_pairs))
    if n_pairs == 1: axes = axes.reshape(1, -1)
        
    pair_idx = 0
    for i in range(n_groups):
        for j in range(i+1, n_groups):
            g1, g2 = groups_list[i], groups_list[j]
            wf1, wf2 = avg_waveforms[g1], avg_waveforms[g2]
            name1, name2 = group_names[i], group_names[j]
            c1, c2 = color_map.get(str(g1), '#9467bd'), color_map.get(str(g2), '#2ca02c')
            
            # Convert sample axis to ms using spike time resolution
            dt_ms = float(sp.times[1] - sp.times[0]) * 1000.0 if len(sp.times) > 1 else 1.0
            time_axis = (np.arange(len(wf1)) - len(wf1) // 2) * dt_ms

            # Compute metrics only from ramp start → exp end (the fitted model region)
            # Use mean indices across spikes in each group for the trimming window
            def _group_mean_idx(group_label, col):
                grp_inds = df.loc[df[cluster_col] == group_label, 'spk_id'].astype(int).tolist()
                valid = [i for i in grp_inds if i < len(sp.indices) and sp.indices[i][col] >= 0]
                return int(np.round(np.mean([sp.indices[i][col] for i in valid]))) if valid else None

            pk1 = _group_mean_idx(groups_list[i], 3)   # peak index col=3
            pk2 = _group_mean_idx(groups_list[j], 3)
            rs1 = _group_mean_idx(groups_list[i], 0)   # ramp_start col=0
            rs2 = _group_mean_idx(groups_list[j], 0)
            ee1 = _group_mean_idx(groups_list[i], 6)   # exp_end col=6
            ee2 = _group_mean_idx(groups_list[j], 6)

            # Convert to indices in the peak-aligned average waveform
            half = len(wf1) // 2
            if pk1 and rs1 and ee1 and pk2 and rs2 and ee2:
                rel_rs = int(np.round(np.mean([rs1 - pk1, rs2 - pk2])))
                rel_ee = int(np.round(np.mean([ee1 - pk1, ee2 - pk2])))
                trim_start = max(0, half + rel_rs)
                trim_end   = min(len(wf1), half + rel_ee + 1)
                wf1_m, wf2_m = wf1[trim_start:trim_end], wf2[trim_start:trim_end]
            else:
                wf1_m, wf2_m = wf1, wf2

            rmse, nrmse, cos_sim, n_wf1, n_wf2, z_wf1, z_wf2 = get_metrics(wf1_m, wf2_m)
            # Rebuild time axis for the trimmed region (for panel plots)
            time_axis_trim = (np.arange(len(wf1_m)) - len(wf1_m) // 2) * dt_ms
            # Full time axis still used for plotting (xlim handles display)
            time_axis_full = time_axis.copy()
            # Recompute normalised/z-scored on full waveform for visual consistency
            _, _, _, n_wf1_full, n_wf2_full, z_wf1_full, z_wf2_full = get_metrics(wf1, wf2)
            n_wf1, n_wf2 = n_wf1_full, n_wf2_full
            z_wf1, z_wf2 = z_wf1_full, z_wf2_full
            
            # --- PANEL 1: Raw Amplitude (RMSE) ---
            ax_raw = axes[pair_idx, 0]
            ax_raw.plot(time_axis, wf1, color=c1, lw=2, label=name1)
            ax_raw.plot(time_axis, wf2, color=c2, lw=2, label=name2)
            ax_raw.fill_between(time_axis, wf1, wf2, color='gray', alpha=0.2)
            
            ax_raw.text(0.05, 0.95, f"RMSE: {rmse:.2f}", transform=ax_raw.transAxes,
                        va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            ax_raw.set_xlim(-4, 4)
            ax_raw.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
            ax_raw.set_title(f"1. Absolute Amp (Raw)\n{name1} vs {name2}")
            ax_raw.set_ylabel("Amplitude (uV)")
            ax_raw.grid(True, alpha=0.3)

            # --- PANEL 2: Relative Amplitude (NRMSE) ---
            ax_scale = axes[pair_idx, 1]
            ax_scale.plot(time_axis, n_wf1, color=c1, lw=2)
            ax_scale.plot(time_axis, n_wf2, color=c2, lw=2)
            ax_scale.fill_between(time_axis, n_wf1, n_wf2, color='orange', alpha=0.1)

            ax_scale.text(0.05, 0.95, f"NRMSE: {nrmse:.3f}", transform=ax_scale.transAxes,
                         va='top', bbox=dict(boxstyle='round', facecolor='floralwhite', alpha=0.9))
            ax_scale.set_xlim(-4, 4)
            ax_scale.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
            ax_scale.set_title(f"2. Relative Amp (Max-Scaled)")
            ax_scale.set_ylabel("Norm. Amp (a.u.)")
            ax_scale.grid(True, alpha=0.3)
            
            # --- PANEL 3: Pure Shape (Cosine Sim) ---
            ax_shape = axes[pair_idx, 2]
            ax_shape.plot(time_axis, z_wf1, color=c1, lw=2, ls='--')
            ax_shape.plot(time_axis, z_wf2, color=c2, lw=2, ls='--')
            
            # Highlight pure shape difference
            ax_shape.fill_between(time_axis, z_wf1, z_wf2, color='red', alpha=0.1)
            
            stats_shape = f"Cos Sim: {cos_sim:.4f}"
            ax_shape.text(0.05, 0.95, stats_shape, transform=ax_shape.transAxes, 
                          va='top', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.9))
            
            ax_shape.set_xlim(-4, 4)
            ax_shape.axvline(0, color='gray', lw=0.8, ls='--', alpha=0.6)
            ax_shape.set_title(f"3. Pure Shape (Z-Scored)")
            ax_shape.set_ylabel("Z-Score (SD)")
            ax_shape.grid(True, alpha=0.3)
            
            # Only label x-axis on the very last row
            if pair_idx == n_pairs - 1:
                ax_raw.set_xlabel("Time (ms)")
                ax_scale.set_xlabel("Time (ms)")
                ax_shape.set_xlabel("Samples")
            
            rmse_results[f"{name1}_vs_{name2}"] = {
                "rmse": rmse, 
                "nrmse": nrmse, 
                "cos_sim": cos_sim
            }
            pair_idx += 1

    plt.tight_layout()
    plt.show()
    return rmse_results



def compute_cluster_statistics(group_data, alpha=0.05):
    """
    Compute statistics for cluster comparisons.
    
    Returns:
    - significance (p-value and significance stars)
    - test_type (string: "parametric" or "nonparametric")
    - n_groups (2 or 3+)
    - η² effect size (comparable across 2 and 3+ groups)
    """

    
    n_groups = len(group_data)
    
    # Quick sanity check
    if any(len(g) < 3 for g in group_data):
        return {
            'p_value': None,
            'significance': None,
            'test_type': None,
            'n_groups': n_groups,
            'eta_squared': None,
            'error': 'Insufficient data'
        }
    
    # Auto choose test based on number of groups
    # For simplicity, always use nonparametric (more robust)
    test_type = "nonparametric"
    
    if n_groups == 2:
        # Mann-Whitney U test
        g1, g2 = group_data[0], group_data[1]
        n1, n2 = len(g1), len(g2)
        
        # Handle edge cases
        if n1 == 0 or n2 == 0:
            return {
                'p_value': None,
                'significance': None,
                'test_type': test_type,
                'n_groups': n_groups,
                'eta_squared': None,
                'error': 'One group has no data'
            }
        
        stat, p = mannwhitneyu(g1, g2)
        test_name = "Mann-Whitney U"
        
        # Direct calculation of η² from data
        # This avoids the problematic prob_sup → d → η² conversion
        all_data = np.concatenate([g1, g2])
        grand_mean = np.mean(all_data)
        
        # Sum of squares between
        ss_between = n1 * (np.mean(g1) - grand_mean) ** 2 + n2 * (np.mean(g2) - grand_mean) ** 2
        
        # Sum of squares total
        ss_total = np.sum((all_data - grand_mean) ** 2)
        
        # η² = SS_between / SS_total
        if ss_total > 0:
            eta_squared = ss_between / ss_total
        else:
            eta_squared = 0
        
    else:  # 3+ groups
        # Kruskal-Wallis test
        stat, p = kruskal(*group_data)
        test_name = "Kruskal-Wallis"
        
        # Direct ε² calculation (analogous to η²)
        total_n = sum(len(g) for g in group_data)
        if total_n > 1:
            eta_squared = stat / (total_n - 1)
        else:
            eta_squared = 0
    
    # Handle NaN/Inf values
    if np.isnan(eta_squared) or np.isinf(eta_squared):
        eta_squared = 0
    
    # Clip to valid range [0, 1]
    eta_squared = np.clip(eta_squared, 0, 1)
    
    # Significance stars
    if p < 0.001:
        sig_stars = '***'
    elif p < 0.01:
        sig_stars = '**'
    elif p < 0.05:
        sig_stars = '*'
    else:
        sig_stars = 'ns'
    
    # Interpretation
    if eta_squared < 0.01:
        interpretation = "Negligible"
    elif eta_squared < 0.06:
        interpretation = "Small"
    elif eta_squared < 0.14:
        interpretation = "Medium"
    elif eta_squared < 0.26:
        interpretation = "Large"
    else:
        interpretation = "Very Large"
    
    return {
        'p_value': p,
        'significance': sig_stars,
        'test_type': test_type,
        'n_groups': n_groups,
        'eta_squared': eta_squared,
        'interpretation': interpretation,
        'test_name': test_name
    }

def visualize_feature_groups_hist(
    df: pd.DataFrame,
    features: list,
    group_col: str,
    groups: tuple = ("low", "mid", "high"),
    group_names: Optional[Tuple] = None,
    bins: int = 50,
    color_map: Optional[Dict] = None,
    remove_outliers: bool = True,
    iqr_multiplier: float = 3,
    kde: bool = True,
    hist: bool = True,
    common_norm: bool = True,
    alpha: float = 0.5,
    plot_order: tuple = ("low", "mid", "high"),
):
    """
    Plot distributions of spike features for multiple groups.
    Includes safety checks for zero-variance data to prevent LinAlgErrors during KDE.
    """
   
    n_groups = len(groups)
    if n_groups < 2 or n_groups > 3:
        raise ValueError(f"Function supports 2 or 3 groups, got {n_groups}")
    
    if group_names is None:
        group_names = groups

    # Default color map
    if color_map is None:
        color_map = {
            "low": "#1f77b4", "mid": "#2ca02c", "high": "#ff7f0e", "default": "#9467bd"
        }
    
    color_map = color_map.copy()
    if "default" not in color_map:
        color_map["default"] = "#9467bd"
    
    group_colors = [color_map.get(g, color_map["default"]) for g in groups]
    
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    # Helper function for IQR outlier removal
    def clean_data_with_iqr(data):
        if len(data) < 2: return data.copy(), 0, None
        Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - iqr_multiplier * IQR, Q3 + iqr_multiplier * IQR
        
        if remove_outliers:
            data_clean = data[(data >= lower_bound) & (data <= upper_bound)].copy()
            return data_clean, len(data) - len(data_clean), (lower_bound, upper_bound)
        return data.copy(), 0, None

    for idx, feat in enumerate(features):
        if idx >= len(axes): break
        ax = axes[idx]
        
        # Prepare storage for this feature's group data
        feat_data_clean = {}
        feat_stats = {}
        feat_outliers = {}
        
        # 1. Collect and Clean Data
        valid_feature = True
        for g in groups:
            raw = pd.to_numeric(df.loc[df[group_col] == g, feat], errors="coerce").dropna()
            if len(raw) < 2: 
                valid_feature = False; break
            
            clean, n_out, bounds = clean_data_with_iqr(raw)
            feat_data_clean[g] = clean
            feat_outliers[g] = n_out
            feat_stats[g] = {'mean': clean.mean(), 'std': clean.std(), 'n': len(clean)}

        if not valid_feature:
            ax.text(0.5, 0.5, f"Insufficient data: {feat}", ha='center', transform=ax.transAxes)
            continue

        # 2. Setup Bins
        all_vals = np.concatenate([feat_data_clean[g].values for g in groups])
        edges = np.histogram_bin_edges(all_vals, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        widths = np.diff(edges)

        # 3. Plot Histograms (Respecting plot_order)
        for g in plot_order:
            if g not in feat_data_clean: continue
            data = feat_data_clean[g]
            color = color_map.get(g, color_map["default"])
            
            if hist:
                norm = len(all_vals) if common_norm else len(data)
                counts, _ = np.histogram(data, bins=edges)
                density = counts / (norm * widths)
                ax.bar(centers, density, width=widths, color=color, alpha=alpha, 
                       edgecolor='black', linewidth=0.5, label=f"{g} (n={len(data)})")

            # 4. KDE Logic with LinAlgError Prevention
            if kde:
                try:
                    # Check for zero variance to prevent singular matrix error
                    if len(data) > 1 and np.var(data) > 0:
                        kde_func = stats.gaussian_kde(data)
                        x_plot = np.linspace(data.min(), data.max(), 200)
                        y_plot = kde_func(x_plot)
                        if common_norm:
                            y_plot *= (len(data) / len(all_vals))
                        ax.plot(x_plot, y_plot, color=color, lw=2, alpha=0.8)
                    else:
                        print(f"Skipping KDE for {feat} ({g}): Zero variance.")
                except Exception as e:
                    print(f"KDE Error on {feat} ({g}): {e}")

        # 5. Styling
        ax.set_title(f"{feat}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()


def kmeans_1d_cluster(
    df: pd.DataFrame,
    feature: str,
    k: int = 2,
    labels: Optional[List[str]] = None,
    new_col: Optional[str] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    plot: bool = False,
    bins: int = 40,
):
    """
    Perform simple 1D K-means clustering on a given numeric feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing the feature to cluster.
    feature : str
        Column name of the feature to cluster.
    k : int, default=2
        Number of clusters (>= 2).
    labels : list of str, optional
        Custom cluster labels (length must equal k). Defaults to ["c0", ..., f"c{k-1}"].
    new_col : str, optional
        Name for the new cluster label column. Defaults to f"{feature}_cluster".
    max_iter : int, default=100
        Maximum number of Lloyd iterations.
    tol : float, default=1e-6
        Convergence threshold for center updates.
    plot : bool, default=False
        If True, plots a histogram colored by cluster with vertical cutoff lines.
    bins : int, default=40
        Number of histogram bins when `plot=True`.

    Returns
    -------
    df_out : pd.DataFrame
        Copy of df with the new cluster label column added.
    centers : np.ndarray
        Cluster centers (sorted ascending).
    cutoffs : np.ndarray
        Midpoints between adjacent sorted centers (length k-1).

    Notes
    -----
    • Non-finite values (NaN/±inf) are ignored for fitting and remain NaN in the label column.
    • Initialization uses quantiles for stability; falls back to linear spacing if needed.
    """
    if feature not in df.columns:
        raise KeyError(f"Column '{feature}' not found in DataFrame.")
    if k < 2:
        raise ValueError("k must be >= 2")

    df_out = df.copy()
    x = pd.to_numeric(df_out[feature], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x)
    idx_valid = np.where(valid)[0]
    data = x[valid]

    if data.size < 2 or np.allclose(data, data[0], rtol=0, atol=0):
        out_col = new_col if new_col else f"{feature}_cluster"
        df_out[out_col] = np.nan
        return df_out, np.array([]), np.array([])

    if labels is None:
        labels = [f"c{i}" for i in range(k)]
    if len(labels) != k:
        raise ValueError(f"`labels` length ({len(labels)}) must equal k ({k}).")
    out_col = new_col if new_col else f"{feature}_cluster"

    # Quantile init (robust)
    qs = np.linspace(0, 1, k + 2)[1:-1]
    centers = np.quantile(data, qs)
    if np.unique(centers).size < k:
        mn, mx = np.min(data), np.max(data)
        if np.isclose(mn, mx):
            df_out[out_col] = np.nan
            return df_out, np.array([]), np.array([])
        centers = np.linspace(mn, mx, k)

    # Lloyd’s algorithm
    for _ in range(max_iter):
        dists = np.abs(data[:, None] - centers[None, :])
        assign = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        changed = False
        for i in range(k):
            mask_i = (assign == i)
            if np.any(mask_i):
                c_new = data[mask_i].mean()
                if not np.isclose(c_new, centers[i], atol=tol, rtol=0):
                    new_centers[i] = c_new
                    changed = True
        centers = new_centers
        if not changed:
            break

    # Sort centers + remap labels
    order = np.argsort(centers)
    centers = centers[order]
    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(k)
    assign_sorted = inverse_order[assign]
    cutoffs = (centers[:-1] + centers[1:]) / 2.0

    # Write labels back at original indices
    label_array = np.full(x.shape, np.nan, dtype=object)
    label_array[idx_valid] = np.asarray(labels, dtype=object)[assign_sorted]
    df_out[out_col] = label_array
    colors = [get_cluster_color(lab) for lab in labels]
    if plot:
        plt.figure(figsize=(7.5, 4.2))
        for i in range(k):
            plt.hist(data[assign_sorted == i], bins=bins, alpha=0.6, label=labels[i],color = colors[i])
        for c in cutoffs:
            plt.axvline(c, linestyle="--", linewidth=2)
        plt.xlabel(feature)
        plt.ylabel("count")
        plt.title(f"{feature} clusters (k={k})")
        if k <= 10:
            plt.legend()
        plt.tight_layout()
        plt.show()

    return df_out, centers, cutoffs


def _fd_bins(data: np.ndarray) -> int:
    """
    Freedman–Diaconis rule for choosing a reasonable histogram bin count.

    Parameters
    ----------
    data : np.ndarray
        1D numeric array.

    Returns
    -------
    int
        Number of bins (clipped to [10, 80]).
    """
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return min(50, max(10, int(np.sqrt(data.size))))
    bin_width = 2 * iqr * (data.size ** (-1/3))
    if bin_width <= 0:
        return min(50, max(10, int(np.sqrt(data.size))))
    bins = int(np.ceil((data.max() - data.min()) / bin_width))
    return int(np.clip(bins, 10, 80))


def _count_peaks_smooth(hist: np.ndarray) -> Tuple[int, np.ndarray]:
    """
    Count local peaks in a smoothed histogram.

    Smoothing kernel reduces spurious wiggles before peak counting.

    Parameters
    ----------
    hist : np.ndarray
        Histogram counts.

    Returns
    -------
    n_peaks : int
        Number of peaks.
    peak_indices : np.ndarray
        Indices of detected peaks in the smoothed histogram.
    """
    kernel = np.array([1, 2, 3, 2, 1], dtype=float)
    kernel /= kernel.sum()
    sm = np.convolve(hist, kernel, mode='same')
    peaks = np.where((sm[1:-1] > sm[:-2]) & (sm[1:-1] > sm[2:]))[0] + 1
    return len(peaks), peaks


def _looks_clustered_1d(
    data: np.ndarray,
    min_peak_ratio: float = 0.10,
    max_valley_ratio: float = 0.70,
    max_k: int = 3
) -> Tuple[int, Dict[str, Any]]:
    """
    Detect potential multimodality (e.g., 2–3 distinct modes) in 1D data
    using a smoothed histogram peak analysis.

    Parameters
    ----------
    data : np.ndarray
        1D numeric data.
    min_peak_ratio : float, default=0.10
        Minimum relative height for the secondary peak (vs. tallest peak).
    max_valley_ratio : float, default=0.70
        Maximum permitted valley depth ratio between the two tallest peaks.
        Lower values indicate better separation.
    max_k : int, default=3
        Maximum clusters suggested (1..3).

    Returns
    -------
    k_suggest : int
        Suggested number of clusters (1, 2, or 3).
    diagnostics : dict
        Histogram/peaks diagnostics, including `valley_ratio` and approximate peak locations.

   
    """
    if data.size < 50:
        # Small samples can be noisy; leave thresholds as-is but keep this in mind.
        pass
    bins = _fd_bins(data)
    hist, edges = np.histogram(data, bins=bins)
    if hist.max() == 0:
        return 1, {"reason": "empty hist"}

    h = hist.astype(float) / hist.max()
    n_peaks, peak_idx = _count_peaks_smooth(h)
    if n_peaks < 2:
        return 1, {"peaks": n_peaks}

    # Two tallest peaks
    peak_heights = h[peak_idx]
    top2 = np.argsort(peak_heights)[-2:]
    p_idx = peak_idx[top2]
    p_idx.sort()
    p1, p2 = p_idx[0], p_idx[1]
    h1, h2 = h[p1], h[p2]

    if min(h1, h2) < min_peak_ratio:
        return 1, {"reason": "secondary peak too small", "h1": h1, "h2": h2}

    valley = h[(p1+1):p2].min() if p2 > p1 + 1 else min(h1, h2)
    valley_ratio = valley / min(h1, h2) if min(h1, h2) > 0 else 1.0
    if valley_ratio >= max_valley_ratio:
        return 1, {"reason": "valley too shallow", "valley_ratio": valley_ratio}

    # At least bimodal; allow tri-modal if a third peak looks meaningful
    k_suggest = 2
    if max_k >= 3 and n_peaks >= 3:
        third_idx = [i for i in peak_idx if i not in (p1, p2)]
        third_heights = [h[i] for i in third_idx]
        if len(third_heights) > 0 and max(third_heights) >= min_peak_ratio:
            k_suggest = 3

    centers_approx = (edges[:-1] + edges[1:]) / 2.0
    approx_peaks_vals = centers_approx[peak_idx]
    return k_suggest, {
        "peaks": int(n_peaks),
        "peak_bins": peak_idx,
        "peak_vals_approx": approx_peaks_vals.tolist(),
        "valley_ratio": float(valley_ratio)
    }

def cluster_multimodal_features(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    ignore_features: Optional[List[str]] = None,  # NEW
    max_k: int = 3,
    labels_map: Optional[Dict[str, List[str]]] = None,
    suffix: str = "_cluster",
    plot_each: bool = False,
    peak_ratio: float = 0.10,
    valley_ratio: float = 0.70,
    unique_min: int = 8,
    manual_thresholds: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None,
    bins: int = 50,
    kde: bool = True,
    assign_labels: bool = True,
    hist_alpha: float = 0.7, 
):
    """
    Cluster multimodal features with consistent histogram style.
    Supports 2 or 3 clusters with manual thresholds.
    """
    df_out = df.copy() if assign_labels else df
    if manual_thresholds is None:
        manual_thresholds = {}
    
    if ignore_features is None: # NEW
        ignore_features = []

    if features is None:
        features = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    report: Dict[str, Dict[str, Any]] = {}
    
    colors = {
        "low": "#1f77b4",
        "mid": "#2ca02c", 
        "high": "#ff7f0e", 
        "default": "steelblue"
    }

    for feat in features:
        # ==========================================================
        #  NEW: IGNORE FEATURE
        # ==========================================================
        if feat in ignore_features:
            print(f"→ {feat}: Skipping (feature ignored).")
            continue

        x = pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(x)
        data = x[valid]
        new_col = f"{feat}{suffix}"

        # ==========================================================
        #  MANUAL THRESHOLD — VISUALIZATION ONLY
        # ==========================================================
        if feat in manual_thresholds:
            thr_value = manual_thresholds[feat]
            
            # Determine if it's 2 clusters (single threshold) or 3 clusters (dual thresholds)
            if isinstance(thr_value, (int, float)):
                # Single threshold -> 2 clusters
                thr = float(thr_value)
                labels = ["low", "high"]
                
                if assign_labels:
                    cluster_assign = np.where(data <= thr, labels[0], labels[1])
                    df_out.loc[valid, new_col] = cluster_assign
                    df_out.loc[~valid, new_col] = np.nan
                
                thresholds = [thr]
                n_clusters = 2
                
            elif isinstance(thr_value, (tuple, list)) and len(thr_value) == 2:
                # Dual thresholds -> 3 clusters
                thr1, thr2 = sorted([float(thr_value[0]), float(thr_value[1])])  # Ensure thr1 < thr2
                labels = ["low", "mid", "high"]
                
                if assign_labels:
                    cluster_assign = np.where(data <= thr1, labels[0], 
                                             np.where(data <= thr2, labels[1], labels[2]))
                    df_out.loc[valid, new_col] = cluster_assign
                    df_out.loc[~valid, new_col] = np.nan
                
                thresholds = [thr1, thr2]
                n_clusters = 3
            else:
                raise ValueError(f"Invalid threshold format for {feat}. Use float for 2 clusters or tuple for 3 clusters.")
            
            report[feat] = {
                "k": n_clusters,
                "centers": [np.nan] * n_clusters,
                "cutoffs": thresholds,
                "diagnostics": {"manual_threshold": thr_value},
                "column": new_col if assign_labels else None,
                "labels": labels,
            }

            if plot_each:
                fig, ax = plt.subplots(figsize=(8, 5))
                
                # Determine global bin edges
                bin_edges = np.histogram_bin_edges(data, bins=bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_widths = np.diff(bin_edges)
                
                # Calculate histogram counts for each segment
                if n_clusters == 2:
                    # Split data for 2 clusters
                    data_low = data[data <= thr]
                    data_high = data[data > thr]
                    
                    # Calculate densities
                    total_count = len(data)
                    counts_low, _ = np.histogram(data_low, bins=bin_edges)
                    counts_high, _ = np.histogram(data_high, bins=bin_edges)
                    
                    density_low = counts_low / (total_count * bin_widths)
                    density_high = counts_high / (total_count * bin_widths)
                    
                    # Plot overlapping histograms
                    ax.bar(
                        bin_centers, 
                        density_low, 
                        width=bin_widths,
                        color=colors["low"],
                        alpha=hist_alpha,
                        edgecolor='black',
                        linewidth=0.5,
                        align='center',
                        label=f"Low (≤{thr:.3f}, n={len(data_low):,})"
                    )
                    
                    ax.bar(
                        bin_centers, 
                        density_high, 
                        width=bin_widths,
                        color=colors["high"],
                        alpha=hist_alpha,
                        edgecolor='black',
                        linewidth=0.5,
                        align='center',
                        label=f"High (> {thr:.3f}, n={len(data_high):,})"
                    )
                    
                    # Add KDE for each segment
                    if kde:
                        from scipy import stats
                        
                        if len(data_low) > 1:
                            kde_low = stats.gaussian_kde(data_low)
                            x_kde_low = np.linspace(data_low.min(), data_low.max(), 500)
                            y_kde_low = kde_low(x_kde_low)
                            scale_factor = len(data_low) / total_count
                            ax.plot(x_kde_low, y_kde_low * scale_factor, 
                                   color=colors["low"], 
                                   linewidth=3, 
                                   alpha=0.8,
                                   label='KDE (low)')
                        
                        if len(data_high) > 1:
                            kde_high = stats.gaussian_kde(data_high)
                            x_kde_high = np.linspace(data_high.min(), data_high.max(), 500)
                            y_kde_high = kde_high(x_kde_high)
                            scale_factor = len(data_high) / total_count
                            ax.plot(x_kde_high, y_kde_high * scale_factor,
                                   color=colors["high"],
                                   linewidth=3,
                                   alpha=0.8,
                                   label='KDE (high)')
                    
                    # Add threshold line
                    ax.axvline(thr, color='k', linestyle='--', linewidth=2.5, 
                              alpha=0.9, label=f'Threshold = {thr:.3f}')
                    
                    # Statistics
                    stats_text = (f"Threshold: {thr:.3f}\n"
                                 f"Total n: {len(data):,}\n"
                                 f"Low (≤{thr:.3f}): {len(data_low):,} points\n"
                                 f"High (> {thr:.3f}): {len(data_high):,} points")
                
                else:  # n_clusters == 3
                    # Split data for 3 clusters
                    data_low = data[data <= thr1]
                    data_mid = data[(data > thr1) & (data <= thr2)]
                    data_high = data[data > thr2]
                    
                    # Calculate densities
                    total_count = len(data)
                    counts_low, _ = np.histogram(data_low, bins=bin_edges)
                    counts_mid, _ = np.histogram(data_mid, bins=bin_edges)
                    counts_high, _ = np.histogram(data_high, bins=bin_edges)
                    
                    density_low = counts_low / (total_count * bin_widths)
                    density_mid = counts_mid / (total_count * bin_widths)
                    density_high = counts_high / (total_count * bin_widths)
                    
                    # Plot overlapping histograms for 3 clusters
                    ax.bar(
                        bin_centers, 
                        density_low, 
                        width=bin_widths,
                        color=colors["low"],
                        alpha=hist_alpha,
                        edgecolor='black',
                        linewidth=0.5,
                        align='center',
                        label=f"Low (≤{thr1:.3f}, n={len(data_low):,})"
                    )
                    
                    ax.bar(
                        bin_centers, 
                        density_mid, 
                        width=bin_widths,
                        color=colors["mid"],
                        alpha=hist_alpha,
                        edgecolor='black',
                        linewidth=0.5,
                        align='center',
                        label=f"Mid ({thr1:.3f}<x≤{thr2:.3f}, n={len(data_mid):,})"
                    )
                    
                    ax.bar(
                        bin_centers, 
                        density_high, 
                        width=bin_widths,
                        color=colors["high"],
                        alpha=hist_alpha,
                        edgecolor='black',
                        linewidth=0.5,
                        align='center',
                        label=f"High (> {thr2:.3f}, n={len(data_high):,})"
                    )
                    
                    # Add KDE for each segment
                    if kde:
                        from scipy import stats
                        
                        for segment_data, label, color_key in [
                            (data_low, "low", "low"),
                            (data_mid, "mid", "mid"), 
                            (data_high, "high", "high")
                        ]:
                            if len(segment_data) > 1:
                                kde_obj = stats.gaussian_kde(segment_data)
                                x_kde = np.linspace(segment_data.min(), segment_data.max(), 500)
                                y_kde = kde_obj(x_kde)
                                scale_factor = len(segment_data) / total_count
                                ax.plot(x_kde, y_kde * scale_factor,
                                       color=colors[color_key],
                                       linewidth=3,
                                       alpha=0.8,
                                       label=f'KDE ({label})')
                    
                    # Add threshold lines
                    ax.axvline(thr1, color='k', linestyle='--', linewidth=2.5, 
                              alpha=0.9, label=f'Threshold1 = {thr1:.3f}')
                    ax.axvline(thr2, color='k', linestyle='--', linewidth=2.5, 
                              alpha=0.9, label=f'Threshold2 = {thr2:.3f}')
                    
                    # Statistics
                    stats_text = (f"Thresholds: {thr1:.3f}, {thr2:.3f}\n"
                                 f"Total n: {len(data):,}\n"
                                 f"Low (≤{thr1:.3f}): {len(data_low):,} points\n"
                                 f"Mid ({thr1:.3f}<x≤{thr2:.3f}): {len(data_mid):,} points\n"
                                 f"High (> {thr2:.3f}): {len(data_high):,} points")
                
                # Add statistics text box
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                        fontsize=9)
                
                ax.set_title(f"{feat} (manual threshold, {n_clusters} clusters)")
                ax.set_xlabel(feat)
                ax.set_ylabel("Density")
                ax.legend(fontsize=9, loc='upper right')
                ax.grid(True, alpha=0.3, linestyle='--')
                plt.tight_layout()
                plt.show()

            continue  # Skip automatic clustering for manual threshold features

        # ==========================================================
        #  HEURISTIC GATES (ONLY FOR AUTO MODE)
        # =========================================================
        if np.unique(data).size < unique_min:
            print(f"{feat}: Skipped - insufficient unique values")
            continue

        k_suggest, diag = _looks_clustered_1d(
            data,
            min_peak_ratio=peak_ratio,
            max_valley_ratio=valley_ratio,
            max_k=max_k,
        )
        if k_suggest < 2:
            print(f"{feat}: Skipped - k_suggest = {k_suggest}")
            continue

        if labels_map and feat in labels_map:
            labels = labels_map[feat]
            if len(labels) != k_suggest:
                # Use default labels based on k_suggest
                if k_suggest == 2:
                    labels = ["low", "high"]
                elif k_suggest == 3:
                    labels = ["low", "mid", "high"]
                else:
                    labels = [f"cluster_{i}" for i in range(k_suggest)]
        else:
            # Use default labels based on k_suggest
            if k_suggest == 2:
                labels = ["low", "high"]
            elif k_suggest == 3:
                labels = ["low", "mid", "high"]
            else:
                labels = [f"cluster_{i}" for i in range(k_suggest)]

        # Use existing kmeans_1d_cluster function
        if assign_labels:
            df_out, centers, cutoffs = kmeans_1d_cluster(
                df_out,
                feature=feat,
                k=k_suggest,
                labels=labels,
                new_col=new_col,
                plot=False,
            )
        else:
            temp_df, centers, cutoffs = kmeans_1d_cluster(
                df.copy(),
                feature=feat,
                k=k_suggest,
                labels=labels,
                new_col="temp_col",
                plot=False,
            )

        report[feat] = {
            "k": int(k_suggest),
            "centers": centers,
            "cutoffs": cutoffs,
            "diagnostics": diag,
            "column": new_col if assign_labels else None,
            "labels": labels,
        }

        if plot_each:
            fig, ax = plt.subplots(figsize=(8, 5))
            
            # Get cluster assignments
            if assign_labels:
                cluster_data = pd.to_numeric(df_out.loc[df_out[new_col].notna(), feat], errors='coerce').dropna()
                cluster_labels = df_out.loc[df_out[new_col].notna(), new_col]
            else:
                temp_df, centers, cutoffs = kmeans_1d_cluster(
                    df.copy(),
                    feature=feat,
                    k=k_suggest,
                    labels=labels,
                    new_col="temp_viz",
                    plot=False,
                )
                cluster_data = pd.to_numeric(temp_df.loc[temp_df["temp_viz"].notna(), feat], errors='coerce').dropna()
                cluster_labels = temp_df.loc[temp_df["temp_viz"].notna(), "temp_viz"]
            
            # Determine global bin edges
            bin_edges = np.histogram_bin_edges(cluster_data, bins=bins)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_widths = np.diff(bin_edges)
            total_count = len(cluster_data)
            
            # Calculate density for each cluster
            all_densities = []
            for i, label in enumerate(labels):
                cluster_points = cluster_data[cluster_labels == label]
                if len(cluster_points) == 0:
                    all_densities.append(np.zeros(len(bin_edges)-1))
                    continue
                    
                counts, _ = np.histogram(cluster_points, bins=bin_edges)
                density = counts / (total_count * bin_widths)
                all_densities.append(density)
            
            # Plot overlapping histograms for all clusters
            for i, (label, density) in enumerate(zip(labels, all_densities)):
                if density.sum() == 0:
                    continue
                    
                color = colors.get(label, colors["default"])
                ax.bar(
                    bin_centers, 
                    density, 
                    width=bin_widths,
                    color=color,
                    alpha=hist_alpha,
                    edgecolor='black',
                    linewidth=0.5,
                    align='center',
                    label=f"{label} (n={len(cluster_data[cluster_labels == label]):,})"
                )
            
            # Add KDE for each cluster
            if kde:
                from scipy import stats
                
                for i, label in enumerate(labels):
                    cluster_points = cluster_data[cluster_labels == label]
                    if len(cluster_points) > 1:
                        kde_obj = stats.gaussian_kde(cluster_points)
                        x_kde = np.linspace(cluster_points.min(), cluster_points.max(), 500)
                        y_kde = kde_obj(x_kde)
                        scale_factor = len(cluster_points) / total_count
                        color = colors.get(label, colors["default"])
                        ax.plot(x_kde, y_kde * scale_factor, 
                               color=color, 
                               linewidth=3, 
                               alpha=0.8,
                               label=f'KDE ({label})')
            
            # Add cutoff lines
            for i, cutoff in enumerate(cutoffs):
                ax.axvline(cutoff, color='k', linestyle=':', linewidth=2, alpha=0.7,
                          label='Cutoff' if i == 0 else None)
            
            # Add cluster centers
            for i, (center, label) in enumerate(zip(centers, labels)):
                color = colors.get(label, 'gray')
                ax.axvline(center, color=color, linestyle='-', linewidth=1.5, alpha=0.6)
            
            # Add statistics text box
            stats_lines = [f"Total n: {len(cluster_data):,}", f"Clusters: {k_suggest}"]
            for label in labels:
                cluster_points = cluster_data[cluster_labels == label]
                if len(cluster_points) > 0:
                    stats_lines.append(f"{label}: μ={cluster_points.mean():.3f}, n={len(cluster_points):,}")
            
            stats_text = "\n".join(stats_lines)
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    verticalalignment='top', horizontalalignment='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                    fontsize=9)
            
            title = f"{feat} (k={k_suggest}, auto-clustered)"
            ax.set_title(title)
            ax.set_xlabel(feat)
            ax.set_ylabel("Density")
            ax.legend(fontsize=9, loc='upper right')
            ax.grid(True, alpha=0.3, linestyle='--')
            plt.tight_layout()
            plt.show()

    return df_out, report

def plot_clusters_over_time_min(
    df: pd.DataFrame,
    time_col: str,
    label_col: str,
    time_unit: str = "ms",
    fs: Optional[float] = None,
    bin_size_ms: float = 1000,
    sigma_bins: Optional[float] = None
):
    """
    Plot temporal evolution of cluster membership across the recording.

    Top panel: raster of spike times colored by cluster.
    Bottom panel: cluster-specific firing rates (binned, optionally smoothed).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns for time and cluster labels.
    time_col : str
        Spike time column; units depend on `time_unit`.
    label_col : str
        Cluster label column (e.g., "low", "high").
    time_unit : {'ms', 'samples'}, default='ms'
        Units of `time_col`. If 'samples', provide `fs`.
    fs : float, optional
        Sampling frequency in Hz when `time_unit='samples'`.
    bin_size_ms : float, default=1000
        Bin width for rate calculation (milliseconds).
    sigma_bins : float, optional
        Gaussian smoothing sigma in units of bins (None = no smoothing).

    Notes
    -----
    • Spike times are converted to seconds for plotting/rate computation.
    • Non-finite times or NaN labels are ignored.
    """
    # ---- extract
    t = pd.to_numeric(df[time_col], errors="coerce").to_numpy(float)
    lab = df[label_col].astype(object).to_numpy()
    valid = np.isfinite(t) & pd.notna(lab)
    if not valid.any():
        print("No valid points.")
        return

    # seconds
    if time_unit == "ms":
        t_sec = t[valid] / 1000.0
    elif time_unit == "samples":
        if not fs or fs <= 0:
            raise ValueError("Provide fs when time_unit='samples'.")
        t_sec = t[valid] / fs
    else:
        raise ValueError("time_unit must be 'ms' or 'samples'.")

    labels = lab[valid].astype(object)
    clusters = [str(x) for x in pd.unique(labels)]

    # bins
    bin_w = bin_size_ms / 1000.0
    t0, t1 = float(np.min(t_sec)), float(np.max(t_sec))
    if np.isclose(t0, t1):
        t1 = t0 + bin_w
    edges = np.arange(t0, t1 + bin_w, bin_w)
    centers = 0.5 * (edges[:-1] + edges[1:])

    def _smooth(x, s):
        if not s or s <= 0:
            return x
        half = int(np.ceil(4 * s))
        g = np.arange(-half, half + 1, dtype=float)
        k = np.exp(-0.5 * (g / s) ** 2)
        k /= k.sum()
        return np.convolve(x, k, mode="same")

    fig, (ax_raster, ax_rate) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [1, 1.2]}
    )

    # raster
    for i, cl in enumerate(clusters):
        m = (labels == cl)
        if m.sum() == 0:
            continue
        y = np.full(m.sum(), i + 1, float) + (np.random.rand(m.sum()) - 0.5) * 0.03
        ax_raster.scatter(
    t_sec[m], y,
    s=4,
    alpha=0.8,
    color=get_cluster_color(cl),
    label=str(cl)
)

    ax_raster.set_yticks(range(1, len(clusters) + 1))
    ax_raster.set_yticklabels(clusters)
    ax_raster.set_ylabel("cluster (raster)")
    ax_raster.legend(loc="upper right", fontsize=8)

    # rates
    for cl in clusters:
        m = (labels == cl)
        counts, _ = np.histogram(t_sec[m], bins=edges)
        rate = _smooth(counts / bin_w, sigma_bins)
        ax_rate.plot(
    centers, rate,
    label=str(cl),
    color=get_cluster_color(cl)
)

    ax_rate.set_ylabel("rate (spikes/s)")
    ax_rate.set_xlabel("time (s)")
    ax_rate.legend(loc="upper right", fontsize=9)
    ax_rate.set_xlim(edges[0], edges[-1])

   
    plt.tight_layout()
    plt.show()


def cluster_transition_matrix(df: pd.DataFrame, cluster_col: str = "log_isi_cluster") -> pd.DataFrame:
    """
    Compute the one-step transition probability matrix between ISI cluster labels.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the sequential ISI cluster labels.
    cluster_col : str, default='log_isi_cluster'
        Column with categorical labels (e.g., 'low', 'high').

    Returns
    -------
    pd.DataFrame
        Square transition probability matrix:
        rows = current label, columns = next label.
    """
    seq = df[cluster_col].dropna().astype(str).to_numpy()
    clusters = sorted(np.unique(seq))
    n = len(clusters)
    mat = np.zeros((n, n))
    for a, b in zip(seq[:-1], seq[1:]):
        i, j = clusters.index(a), clusters.index(b)
        mat[i, j] += 1
    mat = mat / mat.sum(axis=1, keepdims=True)
    return pd.DataFrame(mat, index=clusters, columns=clusters)




# ------------------------------------------------------------------------------------------- #
# --------------------  Extract LFP windows --------------------- #
# ------------------------------------------------------------------------------------------- #

def extract_lfp_windows(
    spk_df: pd.DataFrame,
    lfp_times_ms: np.ndarray,
    lfp_signal: np.ndarray,
    pre_s: float = 1.0,
    post_s: float = 1.5,
    condition: Optional[str] = None
):
    """
    Extract LFP windows around spikes, optionally filtered by a condition.

    Returns dict with:
      windows: list of arrays (lfp µV)
      times_rel_ms: list of arrays (ms, spike = 0)
      spike_times_ms: list of spike times (ms)
      next_spike_times_ms: list of next spike times (ms or NaN)
      meta_df: Dataframe describing each window
    """

    df = spk_df.copy()

    # Detect whether next-spike is required (ISI-based analyses only)
    requires_next_spike = "log_isi_cluster" in spk_df.columns

    # Apply condition if provided
    if condition is not None:
        df = df.query(condition)

    if df.empty:
        print("No spikes matched the condition — returning empty output.")
        return {
            "windows": [],
            "times_rel_ms": [],
            "spike_times_ms": [],
            "next_spike_times_ms": [],
            "meta_df": pd.DataFrame()
        }

    # Convert pre/post from s → ms
    pre_ms = pre_s * 1000.0
    post_ms = post_s * 1000.0

    # Build helper for spike→next spike lookup while preserving the historical
    # "sid + 1 must exist" rule used by the original implementation.
    next_spike_lookup = spk_df.sort_values("spk_id").set_index("spk_id")["spk_times_ms"].copy()
    next_spike_lookup.index = next_spike_lookup.index - 1

    sids = df["spk_id"].values
    t0s  = df["spk_times_ms"].values.astype(float)

    # Vectorised next-spike lookup
    t_next_arr   = next_spike_lookup.reindex(sids).values.astype(float)  # NaN where missing
    has_next_vec = np.isfinite(t_next_arr)

    # Vectorised bounds + validity filter
    t_starts = t0s - pre_ms
    t_ends   = t0s + post_ms
    in_bounds = (t_starts >= lfp_times_ms[0]) & (t_ends <= lfp_times_ms[-1])
    row_valid = in_bounds & (has_next_vec if requires_next_spike else np.ones(len(sids), bool))

    valid_idxs = np.where(row_valid)[0]
    if len(valid_idxs) == 0:
        return {"windows": [], "times_rel_ms": [], "spike_times_ms": [],
                "next_spike_times_ms": [], "meta_df": pd.DataFrame()}

    # Batch searchsorted on the valid subset only
    idx0s = np.searchsorted(lfp_times_ms, t_starts[valid_idxs])
    idx1s = np.searchsorted(lfp_times_ms, t_ends[valid_idxs])

    windows          = []
    times_rel_ms_out = []
    spike_times      = []
    next_spike_times = []
    meta_rows        = []

    for k, i in enumerate(valid_idxs):
        sid    = int(sids[i])
        t0     = float(t0s[i])
        t_next = float(t_next_arr[i])
        idx0, idx1 = int(idx0s[k]), int(idx1s[k])

        seg   = lfp_signal[idx0:idx1]
        t_seg = lfp_times_ms[idx0:idx1]
        t_rel = t_seg - t0

        windows.append(seg)
        times_rel_ms_out.append(t_rel)
        spike_times.append(t0)
        next_spike_times.append(t_next)
        meta_rows.append({
            "spk_id":             sid,
            "spike_time_ms":      t0,
            "next_spike_time_ms": t_next,
            "t_start_ms":         t_starts[i],
            "t_end_ms":           t_ends[i],
        })

    return {
        "windows":              windows,
        "times_rel_ms":         times_rel_ms_out,
        "spike_times_ms":       spike_times,
        "next_spike_times_ms":  next_spike_times,
        "meta_df":              pd.DataFrame(meta_rows),
    }





# ------------------------------------------------------------------------------------------- #
# --------------------  LFP time resolved analysis --------------------- #
# ------------------------------------------------------------------------------------------- #

def compute_lfp_windows(
    lfp_signal: np.ndarray,
    fs: float,
    # Multitaper only
    window_length_sec: float = 2.0,
    freq_range: Tuple[float, float] = (1, 90),
    n_freqs: int = 256,
    time_bandwidth: float = 2.0,
    decim_factor: int = 10,
    # Specparam fit controls:
    progress: bool = True,          # show SpectralTimeModel tqdm if True
    n_jobs: int = 1,
    # Specparam modes & peak settings:
    aperiodic_mode: Literal["fixed", "knee"] = "fixed",
    periodic_mode: Literal["gaussian", "skewed_gaussian", "cauchy"] = "gaussian",
    peak_width_limits: Tuple[float, float] = (4.0, 8.0),
    max_n_peaks: int = 4,
    min_peak_height: float = 0.0,
    peak_threshold: float = 2.0,
    verbose: bool = False,
    # optionally return the spectrogram 
    return_powers: bool = True,
):
    """
    Build multitaper spectra across time for a 1D LFP trace and fit Specparam across bins.

    Returns:
      (model, freqs, window_times)                  if return_powers=False
      (model, freqs, window_times, powers)          if return_powers=True
        - model: SpectralTimeModel (already fit)
        - freqs: (n_freqs,) linear Hz
        - window_times: list[(start_idx, end_idx)] per time bin (samples)
        - powers: (n_bins, n_freqs) linear power
    """
    lfp = np.asarray(lfp_signal, float)
    win_samps  = int(round(window_length_sec * fs))
    step_samps = max(int(decim_factor), 1)

    # Frequency grid & multitaper TFR → spectrogram
    freqs = np.linspace(freq_range[0], freq_range[1], int(n_freqs))
    n_cycles = freqs * float(window_length_sec)
    tb = max(float(time_bandwidth), 2.0)  # keep ≥ ~3 tapers

    epochs = lfp[np.newaxis, np.newaxis, :]  # (1, 1, n_times)
    tfr = mne.time_frequency.tfr_array_multitaper(
        epochs, sfreq=fs, freqs=freqs, n_cycles=n_cycles,
        time_bandwidth=tb, output="power", decim=decim_factor, verbose=False,
        n_jobs=1,
    )  # (1,1,n_freqs,n_bins)
    spec = np.squeeze(tfr, axis=(0, 1))      # (n_freqs, n_bins)
    powers = spec.T                           # (n_bins, n_freqs)

    # Map each TFR bin back to a (start,end) sample window of length win_samps
    n_bins = powers.shape[0]
    window_times = [(i * step_samps, i * step_samps + win_samps) for i in range(n_bins)]

    # Fit Specparam across time
    model = SpectralTimeModel(
        aperiodic_mode=aperiodic_mode,
        periodic_mode=periodic_mode,
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        peak_threshold=peak_threshold,
        verbose=verbose,
    )
    sp_progress = "tqdm" if progress else None
    model.fit(
        freqs=freqs,
        spectrogram=powers.T,   # (n_freqs, n_bins)
        freq_range=freq_range,
        n_jobs=n_jobs,
        progress=sp_progress,
    )

    if return_powers:
        return model, freqs, window_times, powers
    else:
        return model, freqs, window_times


def run_time_resolved_specparam_on_window(
    lfp_window,
    times_rel,
    fs,
    inner_window_sec=0.5,
    freq_range=(1, 90),
    n_freqs=256,
    time_bandwidth=2.0,
    decim_factor=10,
    band_dict=None,
    aperiodic_mode="fixed",
    periodic_mode="gaussian",
    peak_width_limits=(4.0, 8.0),
    max_n_peaks=4,
    min_peak_height=0.0,
    peak_threshold=2.0,
    verbose=False,
    next_spike_rel=None,
):
    """
    Time-resolved Specparam on a single spike-centered LFP window.

    Parameters
    ----------
    lfp_window : 1D array
        LFP samples for one spike-centered window.
    times_rel : 1D array
        Time axis for lfp_window, typically spike-centered.
        Can be in ms or s; this function auto-detects and converts to seconds.
    fs : float
        Sampling rate of the LFP (Hz).
    next_spike_rel : float, optional
        Relative time to the next spike for this window (from your groups dict).
        Returned as-is under key 'next_spike_rel'.

    Returns
    -------
    out : dict
        {
            "t_bins_s": epoch_times_s,      # (n_epochs,)
            "epoch_idx": epoch_idx,         # (n_epochs,)
            "freqs": freqs,                 # (n_freqs,)
            "powers": powers,               # (n_epochs, n_freqs), linear space
            "offset": offset,               # (n_epochs,)
            "exponent": exponent,           # (n_epochs,)
            "knee": knee or None,           # (n_epochs,) or None
            "r_squared": r_squared,         # (n_epochs,)
            "band_aucs": band_aucs,         # dict[band] -> (n_epochs,)
            "next_spike_rel": next_spike_rel,
            "model": time_model,            # SpectralTimeModel
        }
    """

    if band_dict is None:
        band_dict = {
            "delta": (1, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "gamma": (30, 55),
        }

    # ---------------------------------------------------------
    # 1) Run multitaper + Specparam time model
    # ---------------------------------------------------------
    time_model, freqs, window_times, powers = compute_lfp_windows(
        lfp_signal=np.asarray(lfp_window, float),
        fs=fs,
        window_length_sec=inner_window_sec,
        freq_range=freq_range,
        n_freqs=n_freqs,
        time_bandwidth=time_bandwidth,
        decim_factor=decim_factor,
        progress=False,               # no Specparam tqdm
        aperiodic_mode=aperiodic_mode,
        periodic_mode=periodic_mode,
        peak_width_limits=peak_width_limits,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        peak_threshold=peak_threshold,
        verbose=verbose,
        return_powers=True,
    )

    powers = np.asarray(powers, float)          # (n_epochs, n_freqs)
    n_epochs, n_freqs_actual = powers.shape
    freqs = np.asarray(freqs, float)

    # ---------------------------------------------------------
    # 2) Build epoch time axis in *seconds*, using your times_rel
    #    - times_rel is sample-wise time for the raw window (ms or s)
    #    - window_times are (start_idx, end_idx) in samples
    # ---------------------------------------------------------
    times_rel = np.asarray(times_rel, float)

    # crude but robust: if range is big, assume ms and convert
    if np.nanmax(np.abs(times_rel)) > 20.0:
        times_rel_s = times_rel / 1000.0
    else:
        times_rel_s = times_rel

    sample_idx = np.arange(times_rel_s.size)
    epoch_centers = np.array([0.5 * (s + e) for (s, e) in window_times])
    epoch_times_s = np.interp(epoch_centers, sample_idx, times_rel_s)

    epoch_idx = np.arange(n_epochs, dtype=int)

    # ---------------------------------------------------------
    # 3) Extract aperiodic params & r_squared from the time model
    # ---------------------------------------------------------
    aperiodic_params = np.asarray(time_model.get_params("aperiodic_params"))

    if aperiodic_params.shape[1] == 2:
        offset = aperiodic_params[:, 0]
        exponent = aperiodic_params[:, 1]
        knee = None
    else:
        offset = aperiodic_params[:, 0]
        knee = aperiodic_params[:, 1]
        exponent = aperiodic_params[:, 2]

    try:
        r_squared = np.asarray(time_model.get_params("r_squared")).ravel()
    except Exception:
        r_squared = np.full(n_epochs, np.nan, dtype=float)

    # ---------------------------------------------------------
    # 4) Band AUCs: area between full & aperiodic (log10 space)
    #    For each epoch:
    #       - get child SpectralModel via get_model(ind=epoch_i)
    #       - get_data('full', 'log') & get_data('aperiodic', 'log')
    #       - AUC = ∫(full_log - ape_log) df over the band
    # ---------------------------------------------------------
    band_aucs = {bname: np.full(n_epochs, np.nan, dtype=float)
                 for bname in band_dict.keys()}
   

    for ei in range(n_epochs):
        
        try:
            full_log = time_model.get_model(ind = ei).get_model(component="full",      space="log")
            ape_log  = time_model.get_model(ind = ei).get_model(component="aperiodic", space="log")
            
           
        except Exception:
            continue

        if full_log is None or ape_log is None:
            continue

        full_log = np.asarray(full_log, float)
        ape_log  = np.asarray(ape_log, float)
        

        # Specparam SpectralModel get_data should return 1D arrays
        if full_log.shape != freqs.shape or ape_log.shape != freqs.shape:
            # Something is inconsistent; skip this epoch
            continue

        for bname, (f_lo, f_hi) in band_dict.items():
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            if not np.any(mask):
                continue

            diff = full_log[mask] - ape_log[mask]
            auc = np.trapz(diff, freqs[mask])
            band_aucs[bname][ei] = float(auc)

    # ---------------------------------------------------------
    # 5) Pack and return
    # ---------------------------------------------------------
    return {
        "t_bins_s": epoch_times_s,          # (n_epochs,)
        "epoch_idx": epoch_idx,            # (n_epochs,)
        "freqs": freqs,                    # (n_freqs,)
        "powers": powers,                  # (n_epochs, n_freqs), linear
        "offset": offset,                  # (n_epochs,)
        "exponent": exponent,              # (n_epochs,)
        "knee": knee,                      # (n_epochs,) or None
        "r_squared": r_squared,            # (n_epochs,)
        "band_aucs": band_aucs,            # dict[band] -> (n_epochs,)
        "next_spike_rel": next_spike_rel,  # scalar from groups (ms or s, your choice)
        "model": time_model,               # SpectralTimeModel
    }




import os
import pickle
from tqdm import tqdm
from typing import Optional

def run_time_resolved_specparam_per_spike(
    lfp_windows,        # list of windows, one per spike
    times_rel_list,     # same length
    next_rel_list,      # same length
    fs,
    # --- NEW CHUNKING KWARGS ---
    chunk_size: Optional[int] = None,
    save_dir: Optional[str] = None,
    save_prefix: str = "spk_chunk",
    resume_start_chunk: int = 0,
    # ---------------------------
    **specparam_kwargs
):
    """
    Returns a list where index == spike index
    """
    all_results = []
    n_spikes = len(lfp_windows)

    # ------------------------------------------------------------------- #
    # PATH A: ORIGINAL BEHAVIOR (No Chunking)                             #
    # ------------------------------------------------------------------- #
    if chunk_size is None:
        for i, (win, t_rel, next_rel) in enumerate(
            tqdm(zip(lfp_windows, times_rel_list, next_rel_list),
                 total=n_spikes,
                 desc="Specparam per spike",
                 file=sys.stdout, dynamic_ncols=False)
        ):
            out = run_time_resolved_specparam_on_window(
                lfp_window=win,
                times_rel=t_rel,
                fs=fs,
                next_spike_rel=next_rel,
                **specparam_kwargs
            )
            all_results.append(out)

        return all_results

    # ------------------------------------------------------------------- #
    # PATH B: SAFE CHUNKED BEHAVIOR (For massive cells like C21)          #
    # ------------------------------------------------------------------- #
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Create chunk boundaries
    chunk_indices = list(range(0, n_spikes, chunk_size))

    for chunk_idx, start_idx in enumerate(chunk_indices):
        end_idx = min(start_idx + chunk_size, n_spikes)
        
        # 1. RESUME LOGIC: If skipping this chunk, just load it from disk
        if chunk_idx < resume_start_chunk:
            if save_dir is not None:
                save_path = os.path.join(save_dir, f"{save_prefix}_{chunk_idx}.pkl")
                if os.path.exists(save_path):
                    print(f"Skipping and Loading Chunk {chunk_idx} from disk...")
                    with open(save_path, 'rb') as f:
                        chunk_results = pickle.load(f)
                    all_results.extend(chunk_results)
                else:
                    print(f"WARNING: Chunk {chunk_idx} not found at {save_path}. Data missing!")
            continue

        # 2. PROCESSING LOGIC
        print(f"\nProcessing Chunk {chunk_idx} (Spikes {start_idx} to {end_idx - 1})...")
        chunk_results = []
        
        for i in tqdm(range(start_idx, end_idx), desc=f"Chunk {chunk_idx}/{len(chunk_indices)-1}",
                      file=sys.stdout, dynamic_ncols=False):
            out = run_time_resolved_specparam_on_window(
                lfp_window=lfp_windows[i],
                times_rel=times_rel_list[i],
                fs=fs,
                next_spike_rel=next_rel_list[i],
                **specparam_kwargs
            )
            chunk_results.append(out)

        # 3. SAFE SAVE LOGIC: Save this chunk immediately
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{save_prefix}_{chunk_idx}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(chunk_results, f)
                
        all_results.extend(chunk_results)

    return all_results
    
def build_specparam_groups_from_spike_indices(
    specparam_by_spike,
    spike_indices_by_group,
):
    """
    spike_indices_by_group:
        {
          "Short ISI": [1, 5, 20, ...],
          "Long ISI":  [2, 7, 13, ...],
          ...
        }
    """
    grouped = {}

    for gname, inds in spike_indices_by_group.items():
        grouped[gname] = [specparam_by_spike[i] for i in inds]

    return grouped


# ------------------------------------------------------------------------------------------- #
# --------------------  Time-resolved and window visualizations  --------------------- #
# ------------------------------------------------------------------------------------------- #



def plot_aperiodic_fit_with_band_auc(
    freqs,
    powers,
    time_model,
    epoch_i: int = 0,
    band_range=(30.0, 80.0),
    title: str = None,
    log_freq: bool = False,
    ax=None,
):
    """
    Plot raw PSD, Specparam full fit, aperiodic fit, and shade the band AUC
    (area between full and aperiodic in log10 power) for a single epoch.

    Parameters
    ----------
    freqs : 1D array
        Frequency grid (Hz), in linear space. Shape (n_freqs,).
    powers : 2D array
        Linear power spectrogram, shape (n_epochs, n_freqs).
    time_model : SpectralTimeModel
        The fitted SpectralTimeModel from compute_lfp_windows.
    epoch_i : int
        Index of the time bin / epoch to plot.
    band_range : (float, float)
        Frequency band over which to compute & shade AUC (Hz).
    title : str or None
        Figure title.
    log_freq : bool
        If True, plot x-axis in log scale.
    ax : matplotlib Axes or None
        Optional axis to draw on. If None, creates a new figure & axis.

    Returns
    -------
    fig, ax, auc
        fig : Figure
        ax  : Axes
        auc : float, band AUC (log10 space) between full & aperiodic in band_range
    """

    freqs = np.asarray(freqs, float)
    P_lin = np.asarray(powers[epoch_i], float)

    # ---------- grab the child SpectralModel for this epoch ----------
    # This gives you a 'group' object containing just this one time point
    child_time = time_model.get_group([epoch_i], output_type="group")
    child_model = child_time.get_model(ind=0, regenerate=False)  # SpectralModel

    # full & aperiodic components in log10 power
    full_log = time_model.get_model(epoch_i).get_model(component="full",      space="log")
    ape_log  = time_model.get_model(epoch_i).get_model(component="aperiodic", space="log")

    if full_log is None or ape_log is None:
        raise RuntimeError(
            f"No full / aperiodic data available for epoch {epoch_i}. "
            "Check that the model fit completed successfully."
        )

    full_log = np.asarray(full_log, float)
    ape_log  = np.asarray(ape_log,  float)

    if full_log.shape != freqs.shape or ape_log.shape != freqs.shape:
        raise ValueError(
            f"Shape mismatch:\n"
            f"  freqs:    {freqs.shape}\n"
            f"  full_log: {full_log.shape}\n"
            f"  ape_log:  {ape_log.shape}"
        )

    # ---------- compute band AUC in log space ----------
    f_lo, f_hi = band_range
    band_mask = (freqs >= f_lo) & (freqs <= f_hi)

    if not np.any(band_mask):
        raise ValueError(f"No frequencies within band {band_range} Hz")

    diff = full_log[band_mask] - ape_log[band_mask]
    auc = np.trapz(diff, freqs[band_mask])   # log10-power area

    # ---------- plotting ----------
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

   

    # specparam full & aperiodic
    ax.plot(freqs, full_log, lw=2.0, label="Specparam full")
    ax.plot(freqs, ape_log,  lw=2.0, ls="--", label="Aperiodic fit")

    # shaded band area
    ax.fill_between(
        freqs[band_mask],
        full_log[band_mask],
        ape_log[band_mask],
        alpha=0.3,
        label=f"{f_lo:.0f}–{f_hi:.0f} Hz AUC\n= {auc:.3f}",
    )

    if log_freq:
        ax.set_xscale("log")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (log10)")
    if title is None:
        title = f"Epoch {epoch_i} | AUC {f_lo:.0f}–{f_hi:.0f} Hz = {auc:.3f}"
    ax.set_title(title)
    ax.legend(frameon=True, fontsize=8)
    plt.tight_layout()

    return fig, ax, auc


# ------------------------------------------------------------------------------------------- #
# --------------------  Time-resolved post specparam analysis  --------------------- #
# ------------------------------------------------------------------------------------------- #

def make_feature_groups(time_res_results, feature, band=None):
    """
    Convert time_res_results (from run_time_resolved_specparam_for_groups)
    into a groups-style dict for heatmap / trace plotting.

    Parameters
    ----------
    time_res_results : dict
        {
            group_name: [
                {
                    "t_bins_s": 1D array,            # epoch center times in *seconds*
                    "offset": 1D array,
                    "exponent": 1D array,
                    "r_squared": 1D array,
                    "knee": 1D array or None,
                    "band_aucs": dict[band] -> 1D array,
                    "next_spike_rel": float or None, # usually relative time to next spike
                    "epoch_idx": 1D array (optional)
                },
                ...
            ]
        }
    feature : {"offset","exponent","r_squared","knee","band"}
        Which feature to extract.
    band : str, optional
        If feature == "band", which band key to pull from "band_aucs".

    Returns
    -------
    feat_groups : dict
        {
          group_name: {
             "windows":   list of 1D arrays of feature values,
             "times_rel": list of 1D arrays of times (in **seconds**),
             "next_rel":  1D array of next-spike times (in **seconds**, or NaN),
             "epoch_idx": list of epoch_idx arrays or None,
          }
        }
    """

    feat_groups = {}

    for group_name, win_list in time_res_results.items():

        windows_feat      = []
        windows_times     = []
        windows_next      = []
        windows_epoch_idx = []

        if not isinstance(win_list, (list, tuple)) or len(win_list) == 0:
            continue

        for w in win_list:

            # t_bins_s is already in SECONDS from run_time_resolved_specparam_on_window
            t_bins_s = np.asarray(w["t_bins_s"], float)

            # Handle next_spike_rel: try to keep it in seconds as well
            next_raw = w.get("next_spike_rel", None)
            if next_raw is None:
                next_rel_s = np.nan
            else:
                next_raw = float(next_raw)
                # Heuristic: if it's large, assume it was in ms and convert
                if np.abs(next_raw) > 20.0:
                    next_rel_s = next_raw / 1000.0   # ms -> s
                else:
                    next_rel_s = next_raw           # already seconds

            # ---------- PICK FEATURE ----------
            if feature in ["offset", "exponent", "r_squared", "knee"]:
                arr = w.get(feature, None)
                if arr is None:
                    continue

            elif feature == "band":
                if band is None:
                    raise ValueError("If feature=='band', you must pass band='gamma'/'theta'/etc.")
                band_aucs = w.get("band_aucs", {})
                arr = band_aucs.get(band, None)
                if arr is None:
                    continue

            else:
                raise ValueError(f"Unknown feature '{feature}'.")

            arr = np.asarray(arr, float)

            # safety: ensure array and time lengths match
            if arr.shape[0] != t_bins_s.shape[0]:
                # skip weird cases
                continue

            windows_feat.append(arr)
            windows_times.append(t_bins_s)          # *** seconds ***
            windows_next.append(next_rel_s)

            # optional: keep epoch indices if present
            if "epoch_idx" in w:
                windows_epoch_idx.append(np.asarray(w["epoch_idx"], int))
            else:
                windows_epoch_idx.append(None)

        if len(windows_feat) == 0:
            # no valid windows for this group
            continue

        feat_groups[group_name] = {
            "windows":   windows_feat,
            "times_rel": windows_times,                 # seconds
            "next_rel":  np.asarray(windows_next, float),  # seconds (or NaN)
            "epoch_idx": windows_epoch_idx,
        }

    return feat_groups








def plot_window_feature_groups_heatmap(
    groups: Dict[str, Dict[str, Any]],
    feature_label: str = "value",
    time_unit: str = "s",          
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
    sort_by: str = "next_rel",     
    cmap: str = "viridis",
    titles: Optional[Dict[str, str]] = None,
    transition_time: float = 0.0,
    n_cols: int = 4, # NEW: Grid layout control
) -> Dict[str, Dict[str, Any]]:
    
    results: Dict[str, Dict[str, Any]] = {}
    
    n_plots = len(groups)
    if n_plots == 0:
        return results

    # --- Setup Grid ---
    n_cols_actual = min(n_cols, n_plots)
    n_rows = (n_plots + n_cols_actual - 1) // n_cols_actual
    fig, axes = plt.subplots(n_rows, n_cols_actual, figsize=(6 * n_cols_actual, 4 * n_rows))
    
    if n_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (gname, g) in enumerate(groups.items()):
        ax = axes[idx]

        if sort_by == "next_rel" and "next_rel" in g:
            sort_vec   = np.asarray(g["next_rel"], float)
            next_times = sort_vec
        else:
            sort_vec   = None
            next_times = g.get("next_rel", None)

        n_events = len(g["windows"])
        trans_spike_times = np.full(n_events, transition_time, dtype=float)
        title = titles[gname] if (titles is not None and gname in titles) else gname

        # We need a slight modification to the `_single` function to pass the specific `ax`.
        # Because we don't want to break the existing function, we'll extract its logic to fit the grid:
        
        # --- Extracted Heatmap Logic ---
        windows = g["windows"]
        times_rel = g["times_rel"]
        
        base_T = next((np.asarray(t, float) for t in times_rel if len(t) > 1), None)
        
        if tmin is not None or tmax is not None:
            mask = (base_T >= (tmin if tmin is not None else base_T.min())) & \
                   (base_T <= (tmax if tmax is not None else base_T.max()))
            base_T = base_T[mask]
        else:
            mask = slice(None)

        Tgrid = base_T / 1000.0 if time_unit == "ms" else base_T.copy()
        trans_time_sec = transition_time / 1000.0 if time_unit == "ms" else float(transition_time)

        A_list = []
        for w, t in zip(windows, times_rel):
            w = np.asarray(w, float)
            t = np.asarray(t, float)
            if w.size != t.size or w.size < 2:
                A_list.append(np.full(base_T.shape, np.nan))
                continue

            t_crop, w_crop = t[mask], w[mask]
            if t_crop.size != base_T.size:
                yi = np.full(base_T.shape, np.nan, dtype=float)
                inside = (base_T >= t[0]) & (base_T <= t[-1])
                if inside.any():
                    yi[inside] = np.interp(base_T[inside], t, w)
                A_list.append(yi)
            else:
                A_list.append(w_crop)
                
        A = np.vstack(A_list)
        sort_idx = np.argsort(sort_vec) if sort_vec is not None else np.arange(n_events)
        A = A[sort_idx]

        im = ax.imshow(A, aspect="auto", origin="lower", extent=[Tgrid[0], Tgrid[-1], 0, n_events], cmap=cmap)
        ax.axvline(trans_time_sec, color="white", linewidth=2, alpha=0.9, label="Transition")
        
        y_rows = np.arange(n_events) + 0.5
        if trans_spike_times is not None:
            ax.scatter((np.asarray(trans_spike_times, float)[sort_idx] / 1000.0 if time_unit == "ms" else np.asarray(trans_spike_times, float)[sort_idx]), y_rows, s=20, facecolors="none", edgecolors="white", linewidths=1.0, zorder=3)
            
        if next_times is not None:
            ax.scatter((np.asarray(next_times, float)[sort_idx] / 1000.0 if time_unit == "ms" else np.asarray(next_times, float)[sort_idx]), y_rows, s=20, facecolors="orange", edgecolors="black", linewidths=0.5, zorder=3)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Events (sorted)" if sort_vec is not None else "Events")
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(feature_label)
        
        results[gname] = {"fig": fig, "ax": ax, "out": {"Tgrid": Tgrid, "A": A, "sort_idx": sort_idx}}

    for i in range(len(groups), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()
    return results

def plot_window_feature_group_heatmap_single(
    group: Dict[str, Any],
    feature_label: str = "value",
    time_unit: str = "s",          # "s" or "ms" for group['times_rel']
    tmin: Optional[float] = None,  # in same units as times_rel
    tmax: Optional[float] = None,  # in same units as times_rel
    sort_by: Optional[Sequence[float]] = None,  # len = n_events, e.g. durations or next_rel
    title: Optional[str] = None,
    cmap: str = "viridis",
    # markers
    transition_time: float = 0.0,                  # vertical line position (same units as times_rel)
    trans_spike_times: Optional[Sequence[float]] = None,  # per-event, same units as times_rel
    next_spike_times: Optional[Sequence[float]] = None,   # per-event, same units as times_rel
    trans_label: str = "Transition",
    trans_spike_label: str = "Transition spike",
    next_spike_label: str = "Next spike",
) -> Tuple[plt.Figure, plt.Axes, Dict[str, Any]]:
    """
    Plot a single group's windows as an event-by-time heatmap with optional spike markers.

    Parameters
    ----------
    group : dict
        Must have:
          - "windows"   : list of 1D arrays (feature values over time)
          - "times_rel" : list of 1D arrays (time for each window, same length as windows[i])
        Can have:
          - "next_rel"  : 1D array of next spike times (same units as times_rel)
    feature_label : str
        Label for the colorbar (e.g. "aperiodic_exponent", "gamma AUC", "LFP (µV)").
    time_unit : {"s","ms"}
        Unit of the values in times_rel (and marker times).
        X-axis is always seconds.
    tmin, tmax : float or None
        Time range to keep (same units as times_rel). If None, use full range.
    sort_by : array-like or None
        If given, used to sort events (rows). Must be length n_events.
        Common choice: the same as `next_spike_times` or event duration.
    title : str or None
        Figure title.
    cmap : str
        Matplotlib colormap.
    transition_time : float
        Time of the “transition” (e.g. 0), in same units as times_rel.
    trans_spike_times, next_spike_times : array-like or None
        Per-event times for the transition spike and next spike (same units as times_rel).

    Returns
    -------
    fig, ax, out
        out contains:
          - "Tgrid"   : 1D array of time in seconds
          - "A"       : 2D array (n_events, n_time)
          - "sort_idx": 1D array of row indices used
    """

    windows   = group["windows"]
    times_rel = group["times_rel"]

    if len(windows) == 0:
        raise ValueError("Group has no windows.")
    if len(windows) != len(times_rel):
        raise ValueError("len(windows) != len(times_rel).")

    n_events = len(windows)

    # ---------- base time grid ----------
    base_T = None
    for t in times_rel:
        t = np.asarray(t)
        if t.size > 1:
            base_T = t.astype(float)
            break
    if base_T is None:
        raise RuntimeError("No non-empty times_rel arrays found.")

    # crop to [tmin, tmax] in original units
    if tmin is not None or tmax is not None:
        if tmin is None:
            tmin = base_T.min()
        if tmax is None:
            tmax = base_T.max()
        mask = (base_T >= tmin) & (base_T <= tmax)
        base_T = base_T[mask]
    else:
        mask = slice(None)

    # convert grid to seconds
    if time_unit == "ms":
        Tgrid = base_T / 1000.0
        trans_time_sec = transition_time / 1000.0
    elif time_unit == "s":
        Tgrid = base_T.copy()
        trans_time_sec = float(transition_time)
    else:
        raise ValueError("time_unit must be 's' or 'ms'.")

    # ---------- build matrix A (events × time) ----------
    A_list = []
    for w, t in zip(windows, times_rel):
        w = np.asarray(w, float)
        t = np.asarray(t, float)
        if w.size != t.size or w.size < 2:
            A_list.append(np.full(base_T.shape, np.nan))
            continue

        t_crop = t[mask]
        w_crop = w[mask]
        if t_crop.size != base_T.size:
            yi = np.full(base_T.shape, np.nan, dtype=float)
            inside = (base_T >= t[0]) & (base_T <= t[-1])
            if inside.any():
                yi[inside] = np.interp(base_T[inside], t, w)
            A_list.append(yi)
        else:
            A_list.append(w_crop)
    A = np.vstack(A_list)  # (n_events, n_time)

    # ---------- sorting ----------
    sort_idx = np.arange(n_events)
    if sort_by is not None:
        sort_by = np.asarray(sort_by)
        if sort_by.shape[0] != n_events:
            raise ValueError("sort_by must have length n_events.")
        sort_idx = np.argsort(sort_by)
        A = A[sort_idx]
    
    # ---------- markers in seconds ----------
    def _prep_marker(times_arr, label: str):
        if times_arr is None:
            return None
        arr = np.asarray(times_arr, float)
        if arr.shape[0] != n_events:
            raise ValueError(f"{label} must have length n_events.")
        arr = arr[sort_idx]
        if time_unit == "ms":
            arr = arr / 1000.0
        return arr
    
    trans_spike_sec = _prep_marker(trans_spike_times, "trans_spike_times")
    next_spike_sec  = _prep_marker(next_spike_times, "next_spike_times")
    
    # ---------- plotting ----------
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(
        A,
        aspect="auto",
        origin="lower",
        extent=[Tgrid[0], Tgrid[-1], 0, n_events],
        cmap=cmap,
    )
    
    # vertical line at transition
    ax.axvline(trans_time_sec, color="white", linewidth=2, alpha=0.9, label=trans_label)
    
    # scatter markers: one per row
    y_rows = np.arange(n_events) + 0.5
    
    if trans_spike_sec is not None:
        ax.scatter(
            trans_spike_sec, y_rows,
            s=20, facecolors="none", edgecolors="white",
            linewidths=1.0, label=trans_spike_label, zorder=3,
        )
    
    if next_spike_sec is not None:
        ax.scatter(
            next_spike_sec, y_rows,
            s=20, facecolors="orange", edgecolors="black",
            linewidths=0.5, label=next_spike_label, zorder=3,
        )
    
    ax.set_xlabel("Time (s, relative to transition)")
    ax.set_ylabel("Events (sorted)" if sort_by is not None else "Events")
    if title is not None:
        ax.set_title(title)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(feature_label)
    
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(
            frameon=True, facecolor="black", edgecolor="none",
            framealpha=0.7, loc="upper right",
        )
    
    fig.tight_layout()
    
    out = {"Tgrid": Tgrid, "A": A, "sort_idx": sort_idx}
    return fig, ax, out
    

# ------------------------------------------------------------------------------------------- #
# ------------------------------ Functions and wrapper functions for post specparam cluster group analysis --------------------- #
# ------------------------------------------------------------------------------------------- #
def build_lfp_groups_from_clusters(
    df_features,
    all_spike_extractor,
    cluster_cols=None,
):
    """
    Build LFP groups from cluster columns in df_features.

    Returns:
        groups : dict
            groups[group_name] = {
                "windows":   list[np.ndarray],
                "times_rel": list[np.ndarray],
                "next_rel":  np.ndarray,
                "spk_inds":  np.ndarray,
            }
    """

    windows_all     = all_spike_extractor["windows"]
    t_rel_all       = all_spike_extractor["times_rel_ms"]
    spike_times_all = np.asarray(all_spike_extractor["spike_times_ms"], float)
    next_spk_all    = np.asarray(all_spike_extractor["next_spike_times_ms"], float)
    meta_df_all     = all_spike_extractor["meta_df"]

    # Map spk_id → index in windows_all
    spk_id_to_win = {
        sid: i for i, sid in enumerate(meta_df_all["spk_id"].values)
    }

    if cluster_cols is None:
        cluster_cols = [c for c in df_features.columns if c.endswith("_cluster")]

    groups = {}

    for col in cluster_cols:
        for label in df_features[col].dropna().unique():

            # spike IDs belonging to this cluster
            spk_ids = df_features.loc[df_features[col] == label, "spk_id"].values

            # map to window indices (only keep ones that exist)
            win_inds = [
                spk_id_to_win[sid]
                for sid in spk_ids
                if sid in spk_id_to_win
            ]

            if len(win_inds) == 0:
                continue

            group_name = f"{col}: {label}"

            groups[group_name] = {
                "windows":   [windows_all[i] for i in win_inds],
                "times_rel": [t_rel_all[i]   for i in win_inds],
                "next_rel":  next_spk_all[win_inds] - spike_times_all[win_inds],
                "spk_inds":  np.asarray(win_inds, dtype=int),
            }

    return groups


def make_specparam_feature_groups(
    specparam_by_spike,
    groups,
    feature: str,
    band: str = None,
    next_rel_unit: str = "ms",   # groups['next_rel'] is in ms in your current code
):
    """
    Build a groups-style dict (windows/times_rel/next_rel) for specparam features,
    using specparam_by_spike (computed once per spike) and groups (which store spk_inds).

    Returns:
        feat_groups[gname] = {
            "windows":   list of 1D arrays (feature over epochs),
            "times_rel": list of 1D arrays (epoch times, seconds),
            "next_rel":  1D array (seconds),
            "spk_inds":  np.ndarray,
        }
    """
    out = {}

    for gname, gdict in groups.items():
        spk_inds = np.asarray(gdict.get("spk_inds", []), dtype=int)
        if spk_inds.size == 0:
            continue

        win_list = []
        t_list   = []
        next_list = []

        # next_rel for these events (convert to seconds)
        next_rel = np.asarray(gdict.get("next_rel", np.full(spk_inds.size, np.nan)), float)
        if next_rel_unit == "ms":
            next_rel_s = next_rel / 1000.0
        else:
            next_rel_s = next_rel

        # collect per-spike traces
        for k, ind in enumerate(spk_inds):
            if ind < 0 or ind >= len(specparam_by_spike):
                continue

            res = specparam_by_spike[ind]
            if res is None:
                continue

            t_bins_s = np.asarray(res.get("t_bins_s", []), float)
            if t_bins_s.size == 0:
                continue

            if feature in ["offset", "exponent", "r_squared", "knee"]:
                arr = res.get(feature, None)
            elif feature == "band":
                if band is None:
                    raise ValueError("feature='band' requires band='gamma'/'alpha'/etc.")
                arr = res.get("band_aucs", {}).get(band, None)
            else:
                raise ValueError(f"Unknown feature: {feature}")

            if arr is None:
                continue

            arr = np.asarray(arr, float)
            if arr.size != t_bins_s.size:
                continue

            win_list.append(arr)
            t_list.append(t_bins_s)
            next_list.append(next_rel_s[k])

        if len(win_list) == 0:
            continue

        out[gname] = {
            "windows":   win_list,
            "times_rel": t_list,                       # seconds
            "next_rel":  np.asarray(next_list, float), # seconds
            "spk_inds":  spk_inds,
        }

    return out




#Function to analyze windows is specparam feature traces, and compare across cluster groups 

def lfp_sliding_stats(
    feat_groups,
    time_unit="s",
    ylabel="Δ Value",
    window_width=0.05,
    step_size=0.01,
    p_threshold=0.05,
    alpha_ci=0.25,
    plot_mode="both",
    figsize=(14, 6),
    plot=True,
    baseline_window=(-0.75, -0.5),
):


    def cohens_d(d1, d2):
        n1, n2 = len(d1), len(d2)
        if n1 < 2 or n2 < 2: return 0.0
        var1, var2 = np.var(d1, ddof=1), np.var(d2, ddof=1)
        pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        return abs(np.mean(d1) - np.mean(d2)) / (pooled_se + 1e-8)

    def get_stars(p):
        if p < 0.0001: return "****"
        elif p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        return "ns"

    def sort_clusters(names):
        rank = {"low": 0, "mid": 1, "high": 2}
        def get_rank(n):
            ln = n.lower()
            for k, v in rank.items():
                if k in ln: return v
            return 99
        return sorted(names, key=get_rank)

    # Organizes groups by spike feature!
    cluster_families = {}
    for gname in feat_groups.keys():
        family = gname.split(":")[0] if ":" in gname else "all"
        cluster_families.setdefault(family, []).append(gname)

    plot_sets = {}
    if plot_mode in ["all", "both"]: plot_sets["All groups"] = list(feat_groups.keys())
    if plot_mode in ["per_cluster", "both"]:
        for k, v in cluster_families.items(): plot_sets[k] = v

    default_palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    stats_report = [] 
    merged_regions = []
    trace_data = {} 

    for set_name, unsorted_names in plot_sets.items():
        if len(unsorted_names) < 2: continue
        
        group_names = sort_clusters(unsorted_names)
        current_mode = "all" if set_name == "All groups" else "per_cluster"
        
        A_matrices = {}
        colors_dict = {}
        Tgrid = None

        for i, gname in enumerate(group_names):
            if current_mode == "per_cluster":
                lname = gname.lower()
                colors_dict[gname] = "#1f77b4" if "low" in lname else "#ff7f0e" if "high" in lname else "#2ca02c"
            else:
                colors_dict[gname] = default_palette[i % len(default_palette)]
            
            g = feat_groups[gname]
            if Tgrid is None:
                base_T = next((np.asarray(t, float) for t in g["times_rel"] if len(t) > 1), None)
                Tgrid = base_T / 1000.0 if time_unit == "ms" else base_T.copy()
            
            # Collect valid (window, time) pairs
            valid_wt = [
                (np.asarray(w, float), np.asarray(t, float) / 1000.0 if time_unit == "ms" else np.asarray(t, float))
                for w, t in zip(g["windows"], g["times_rel"])
                if len(w) == len(t) and len(w) >= 2
            ]
            if len(valid_wt) < 2:
                continue

            # Fast path: all windows same length → baseline correct + vectorised linear interp
            lengths = [len(w) for w, _ in valid_wt]
            if len(set(lengths)) == 1:
                W     = np.array([w for w, _ in valid_wt])        # (n_spikes, n_samples)
                T_ref = valid_wt[0][1]                             # reference time axis (~identical for all)
                bl_mask = (T_ref >= baseline_window[0]) & (T_ref <= baseline_window[1])
                if not bl_mask.any():
                    # Specified window not available — use all pre-spike data up to −50 ms
                    bl_mask = (T_ref >= T_ref.min()) & (T_ref <= -0.05)
                if not bl_mask.any():
                    # Last resort: earliest single time point
                    bl_mask = T_ref == T_ref.min()
                W -= np.nanmean(W[:, bl_mask], axis=1, keepdims=True)
                # Vectorised linear interpolation: build interp coefficients once, apply to all rows
                idxs  = np.searchsorted(T_ref, Tgrid, side="left").clip(1, len(T_ref) - 1)
                lo, hi = idxs - 1, idxs
                dt     = T_ref[hi] - T_ref[lo]
                dt     = np.where(dt == 0, 1.0, dt)
                alpha  = (Tgrid - T_ref[lo]) / dt                 # (n_grid,)
                A_matrices[gname] = W[:, lo] * (1.0 - alpha) + W[:, hi] * alpha
            else:
                # Fallback for variable-length windows (rare)
                def _bl_correct(w, t):
                    mask = (t >= baseline_window[0]) & (t <= baseline_window[1])
                    if not mask.any():
                        mask = (t >= t.min()) & (t <= -0.05)
                    if not mask.any():
                        mask = t == t.min()
                    return w - np.nanmean(w[mask])
                mats = [np.interp(Tgrid, t, _bl_correct(w, t)) for w, t in valid_wt]
                A_matrices[gname] = np.vstack(mats)

        # --- FIX: Save Trace Data organized by `set_name` (the spike feature) ---
        trace_data[set_name] = {"Tgrid": Tgrid, "groups": {}}
        for gname in group_names:
            if gname in A_matrices:
                A = A_matrices[gname]
                trace_data[set_name]["groups"][gname] = {
                    "mean": np.nanmean(A, axis=0),
                    "std": np.nanstd(A, axis=0),
                    "n": np.sum(np.isfinite(A), axis=0)
                }

        # Sliding Window Math — vectorised across all windows at once
        sig_windows = []
        valid_gns = [gn for gn in group_names if gn in A_matrices]
        if len(valid_gns) >= 2 and Tgrid is not None and len(Tgrid) > 1:
            dt_grid     = float(np.mean(np.diff(Tgrid)))
            n_win_samp  = max(1, round(window_width / dt_grid))

            # uniform_filter1d gives the centered box mean at every Tgrid point — one C call per group
            A_smooth = {gn: uniform_filter1d(A_matrices[gn], size=n_win_samp, axis=1, mode="nearest")
                        for gn in valid_gns}

            # Window start positions → the index of the window centre in Tgrid
            w_starts = np.arange(Tgrid[0], Tgrid[-1] - window_width + dt_grid * 0.5, step_size)
            c_idx    = np.searchsorted(Tgrid, w_starts + window_width / 2).clip(0, len(Tgrid) - 1)

            # Extract per-window per-spike values for each group: (n_spikes, n_wins)
            G = [A_smooth[gn][:, c_idx] for gn in valid_gns]

            if len(G) == 2:
                # Vectorised Welch's t-test
                g1, g2  = G
                n1  = np.maximum(np.sum(np.isfinite(g1), axis=0), 2)
                n2  = np.maximum(np.sum(np.isfinite(g2), axis=0), 2)
                mu1 = np.nanmean(g1, axis=0);  mu2 = np.nanmean(g2, axis=0)
                v1  = np.nanvar(g1, axis=0, ddof=1); v2 = np.nanvar(g2, axis=0, ddof=1)
                se2  = v1 / n1 + v2 / n2
                t_st = (mu1 - mu2) / np.sqrt(se2 + 1e-15)
                df   = se2 ** 2 / ((v1 / n1) ** 2 / np.maximum(n1 - 1, 1)
                                   + (v2 / n2) ** 2 / np.maximum(n2 - 1, 1) + 1e-15)
                from scipy.stats import t as _t_dist
                p_vals = 2.0 * _t_dist.sf(np.abs(t_st), df)
            else:
                # Vectorised one-way F-test
                k     = len(G)
                ns    = np.array([np.maximum(np.sum(np.isfinite(g), axis=0), 1) for g in G])  # (k, n_wins)
                means = np.array([np.nanmean(g, axis=0) for g in G])                           # (k, n_wins)
                N     = ns.sum(axis=0)
                grand = np.nansum(means * ns, axis=0) / np.maximum(N, 1)
                ss_b  = np.nansum(ns * (means - grand) ** 2, axis=0)
                ss_w  = np.nansum(
                    [np.nanvar(g, axis=0, ddof=1) * np.maximum(ns[i] - 1, 0) for i, g in enumerate(G)],
                    axis=0,
                )
                f_st  = (ss_b / (k - 1)) / (ss_w / np.maximum(N - k, 1) + 1e-15)
                from scipy.stats import f as _f_dist
                p_vals = _f_dist.sf(f_st, k - 1, np.maximum(N - k, 1))

            p_vals = np.where(np.isfinite(p_vals), p_vals, 1.0)
            sig_windows = [
                (float(w_starts[i]), float(w_starts[i] + window_width), float(p_vals[i]))
                for i in np.where(p_vals < p_threshold)[0]
            ]

        merged_regions = []
        if sig_windows:
            cur_s, cur_e, cur_p = sig_windows[0]
            for w in sig_windows[1:]:
                if w[0] <= cur_e + 1e-5:
                    cur_e, cur_p = max(cur_e, w[1]), min(cur_p, w[2])
                else:
                    merged_regions.append((cur_s, cur_e, cur_p)); cur_s, cur_e, cur_p = w
            merged_regions.append((cur_s, cur_e, cur_p))

        # --- ALWAYS: pairwise post-hoc stats over significant windows (no matplotlib) ---
        # Pre-compute box_data per merged window so we can reuse in plotting below
        window_box_data = {}
        for s, e, _ in merged_regions:
            amask = (Tgrid >= s) & (Tgrid <= e)
            box_data = [np.nanmean(A_matrices[gn][:, amask], axis=1) for gn in group_names if gn in A_matrices]
            box_data = [d[np.isfinite(d)] for d in box_data]
            window_box_data[(s, e)] = box_data
            for i, j in combinations(range(len(box_data)), 2):
                d1, d2 = box_data[i], box_data[j]
                if len(d1) < 2 or len(d2) < 2: continue
                _, p_pair = ttest_ind(d1, d2, equal_var=False)
                if p_pair < p_threshold:
                    stats_report.append({
                        "spike_feature": set_name,
                        "window_start":  s,
                        "window_end":    e,
                        "group_1":       group_names[i],
                        "group_2":       group_names[j],
                        "p_value":       p_pair,
                        "cohens_d":      cohens_d(d1, d2),
                    })

        if not plot:
            continue

        # --- PLOTTING (skipped when plot=False) ---
        fig = plt.figure(figsize=figsize)
        has_regions = len(merged_regions) > 0
        master_gs = GridSpec(1, 2, width_ratios=[1.5, 1] if has_regions else [1, 0.01], wspace=0.3)

        ax_trace = fig.add_subplot(master_gs[0])
        for gname in group_names:
            if gname not in A_matrices: continue
            mean, std = np.nanmean(A_matrices[gname], axis=0), np.nanstd(A_matrices[gname], axis=0)
            ci = 1.96 * std / np.sqrt(np.sum(np.isfinite(A_matrices[gname]), axis=0))
            ax_trace.plot(Tgrid, mean, label=gname, color=colors_dict[gname], lw=2.5)
            ax_trace.fill_between(Tgrid, mean-ci, mean+ci, color=colors_dict[gname], alpha=alpha_ci, lw=0)

        for s, e, _ in merged_regions: ax_trace.axvspan(s, e, color='gold', alpha=0.15)

        # Shade the actual baseline window used (light blue) and draw y=0 reference
        _bl_test = (Tgrid >= baseline_window[0]) & (Tgrid <= baseline_window[1])
        if _bl_test.any():
            _bl_start, _bl_end = baseline_window[0], baseline_window[1]
        else:
            _bl_start = float(Tgrid.min())
            _bl_end   = min(-0.05, float(Tgrid[Tgrid <= -0.05].max()) if np.any(Tgrid <= -0.05) else float(Tgrid.min()))
        ax_trace.axvspan(_bl_start, _bl_end, color='steelblue', alpha=0.15, zorder=0, label='baseline')
        ax_trace.axhline(0, color='gray', ls=':', lw=1, zorder=0)

        ax_trace.axvline(0, color="k", ls="--", lw=1.5)
        ax_trace.set_title(f"Temporal Dynamics: {set_name}", fontweight='bold')
        ax_trace.set_ylabel(ylabel)
        ax_trace.legend(loc="upper right", frameon=True, fontsize=9)
        ax_trace.grid(False)

        if has_regions:
            n_boxes = len(merged_regions)
            n_cols  = 2 if n_boxes > 2 else 1
            n_rows  = (n_boxes + n_cols - 1) // n_cols
            sub_gs  = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=master_gs[1], wspace=0.4, hspace=0.6)

            for idx, (s, e, _) in enumerate(merged_regions):
                ax_box   = fig.add_subplot(sub_gs[idx])
                box_data = window_box_data[(s, e)]

                bp = ax_box.boxplot(box_data, labels=[l.split(': ')[-1] for l in group_names],
                                    patch_artist=True, medianprops=dict(color="black"))
                for i, box in enumerate(bp['boxes']):
                    box.set_facecolor(colors_dict[group_names[i]])
                    box.set_alpha(0.6)

                nonempty = [d for d in box_data if len(d) > 0]
                if not nonempty: continue
                y_max, y_min = max(np.max(d) for d in nonempty), min(np.min(d) for d in nonempty)
                step      = (y_max - y_min) * 0.15 or 0.1
                current_y = y_max + step

                for i, j in combinations(range(len(box_data)), 2):
                    d1, d2 = box_data[i], box_data[j]
                    if len(d1) < 2 or len(d2) < 2: continue
                    _, p_pair = ttest_ind(d1, d2, equal_var=False)
                    if p_pair < p_threshold:
                        d_eff = cohens_d(d1, d2)
                        stars = get_stars(p_pair)
                        p_str = f"p={p_pair:.4f}" if p_pair > 0.0001 else "p<0.0001"
                        ax_box.plot([i+1, i+1, j+1, j+1],
                                    [current_y, current_y+step*0.2, current_y+step*0.2, current_y],
                                    lw=1.2, c='k')
                        ax_box.text((i+j+2)/2, current_y+step*0.2,
                                    f"{stars}\n{p_str}\n(d={d_eff:.1f})",
                                    ha='center', va='bottom', fontsize=7)
                        current_y += step * 2.5

                ax_box.set_ylim(bottom=y_min - step, top=current_y + step)
                ax_box.set_title(f"Window: {s:+.2f} to {e:+.2f}s", fontsize=9, pad=5)
                ax_box.set_ylabel(ylabel)
                plt.setp(ax_box.get_xticklabels(), rotation=30, ha="right", fontsize=8)
                ax_box.grid(False)

        plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.15)
        plt.show()

    return {
        "omnibus_windows": merged_regions,
        "pairwise_stats": stats_report,
        "trace_data": trace_data
    }

def p_to_stars(p):
    """Convert a p-value to a significance star string ('***', '**', '*', or 'n.s.')."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


def compute_simple_lfp_by_spike(
    windows_all,
    times_rel_list,
    fs,
    inner_window_s=0.5,
    step_s=0.025,
    freq_range=(4, 90),
):
    """
    For each spike's raw LFP window, slide through with inner sub-windows and compute
    mean amplitude, std, and spectral exponent (log-log FFT slope) per bin.

    Parameters
    ----------
    windows_all : list of np.ndarray
        Raw LFP windows, one per spike.
    times_rel_list : list of np.ndarray
        Time arrays in ms relative to spike, one per spike.
    fs : float
        LFP sampling rate in Hz.
    inner_window_s : float
        Sub-window width in seconds.
    step_s : float
        Step size in seconds.
    freq_range : tuple
        (low, high) Hz range used for the exponent fit.

    Returns
    -------
    list of dict (or None if window too short), parallel to windows_all:
        {
            "t_bins_s":     np.ndarray,  # center time of each sub-window (s)
            "lfp_mean":     np.ndarray,
            "lfp_std":      np.ndarray,
            "lfp_exponent": np.ndarray,
        }
    """
    inner_samples = int(inner_window_s * fs)
    step_samples  = max(1, int(step_s * fs))

    # Precompute fixed frequency quantities shared across all spikes/sub-windows
    freqs     = np.fft.rfftfreq(inner_samples, d=1.0 / fs)
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    has_freq  = np.sum(freq_mask) > 2
    if has_freq:
        log_freqs = np.log10(freqs[freq_mask])
        # Design matrix for batch least-squares: [log_freq, 1]
        A_fit = np.column_stack([log_freqs, np.ones(len(log_freqs))])

    results = []

    for raw_win, t_rel_ms in zip(windows_all, times_rel_list):
        raw_win = np.asarray(raw_win, float)
        t_rel_s = np.asarray(t_rel_ms, float) / 1000.0

        n = len(raw_win)
        if n < inner_samples:
            results.append(None)
            continue

        # Build 2D view of all sub-windows at once — zero-copy stride trick
        views   = np.lib.stride_tricks.sliding_window_view(raw_win, inner_samples)[::step_samples]
        n_steps = len(views)  # (n_steps, inner_samples)

        # Batch mean and std across the sample axis
        means = np.nanmean(views, axis=1)
        stds  = np.nanstd(views, axis=1)

        # Batch spectral exponent via single rfft call on all sub-windows
        exps = np.full(n_steps, np.nan)
        if has_freq:
            fft_vals   = np.fft.rfft(views, axis=1)                    # (n_steps, n_rfft)
            power      = (np.abs(fft_vals) ** 2) / inner_samples       # (n_steps, n_rfft)
            pwr_masked = power[:, freq_mask]                            # (n_steps, n_freq)
            valid_pwr  = pwr_masked > 0
            log_pwr    = np.where(valid_pwr, np.log10(pwr_masked), np.nan)

            # Batch lstsq for rows where all freq bins are positive (the common case)
            all_valid = np.all(valid_pwr, axis=1)
            if np.any(all_valid):
                coeffs, _, _, _ = np.linalg.lstsq(A_fit, log_pwr[all_valid].T, rcond=None)
                exps[all_valid] = -coeffs[0]

            # Fall back to per-row polyfit only for the rare partially-invalid rows
            partial = ~all_valid & (np.sum(valid_pwr, axis=1) > 2)
            for idx in np.where(partial)[0]:
                row = log_pwr[idx]
                vm  = np.isfinite(row)
                if vm.sum() > 2:
                    c = np.polyfit(log_freqs[vm], row[vm], 1)
                    exps[idx] = -c[0]

        # Center time for each sub-window
        starts  = np.arange(n_steps) * step_samples
        centers = np.minimum(starts + inner_samples // 2, len(t_rel_s) - 1)
        t_bins  = t_rel_s[centers]

        results.append({
            "t_bins_s":     t_bins,
            "lfp_mean":     means,
            "lfp_std":      stds,
            "lfp_exponent": exps,
        })

    return results


def make_simple_lfp_feature_groups(
    lfp_windows_by_spike,
    groups,
    feature: str,
):
    """
    Build a feat_groups dict for simple LFP features (lfp_mean, lfp_std, lfp_exponent),
    mirroring make_specparam_feature_groups so lfp_sliding_stats works unchanged.

    Parameters
    ----------
    lfp_windows_by_spike : list of dict
        Output of compute_simple_lfp_by_spike.
    groups : dict
        Output of build_lfp_groups_from_clusters.
    feature : str
        One of "lfp_mean", "lfp_std", "lfp_exponent".

    Returns
    -------
    dict with same structure as make_specparam_feature_groups output.
    """
    out = {}

    for gname, gdict in groups.items():
        spk_inds = np.asarray(gdict.get("spk_inds", []), dtype=int)
        if spk_inds.size == 0:
            continue

        next_rel   = np.asarray(gdict.get("next_rel", np.full(spk_inds.size, np.nan)), float)
        next_rel_s = next_rel / 1000.0  # groups stores next_rel in ms

        win_list, t_list, next_list = [], [], []

        for k, ind in enumerate(spk_inds):
            if ind < 0 or ind >= len(lfp_windows_by_spike):
                continue
            res = lfp_windows_by_spike[ind]
            if res is None:
                continue

            t_bins = np.asarray(res.get("t_bins_s", []), float)
            arr    = np.asarray(res.get(feature, []), float)

            if arr.size == 0 or arr.size != t_bins.size:
                continue

            win_list.append(arr)
            t_list.append(t_bins)
            next_list.append(next_rel_s[k])

        if len(win_list) == 0:
            continue

        out[gname] = {
            "windows":   win_list,
            "times_rel": t_list,
            "next_rel":  np.asarray(next_list, float),
            "spk_inds":  spk_inds,
        }

    return out


# Feature names routed to make_simple_lfp_feature_groups instead of make_specparam_feature_groups
_SIMPLE_LFP_FEATURES = {"lfp_mean", "lfp_std", "lfp_exponent"}


def run_master_LFP_spk_analysis(
    cell_id,
    specparam_by_spike,
    groups,
    features_to_analyze,
    lfp_windows_by_spike=None,
    save_dir=None,
    window_width=0.05,
    step_size=0.025,
    p_threshold=0.05,
    force_recompute=False,
    baseline_window=(-0.75, -0.5),
    plot=True,
):
    """
    Master pipeline: run sliding-window LFP-spike group analysis for all requested features
    and save results to a pickle.

    For each feature in features_to_analyze, this function:
      1. Builds per-group feature arrays (specparam or simple LFP)
      2. Optionally plots a heatmap of the feature over time per group (plot=True)
      3. Runs sliding-window stats to find windows with significant group differences
      4. Saves all results to {save_dir}/{cell_id}_sliding_stats.pkl

    Parameters
    ----------
    cell_id : str
        Cell identifier, used as the pickle filename prefix.
    specparam_by_spike : list of dict
        Output of load_chunked_specparam_results — specparam results per spike.
    groups : dict
        Output of build_lfp_groups_from_clusters — spike indices and LFP windows per cluster group.
    features_to_analyze : list of dict
        Each dict specifies a feature, e.g.:
          {"feature": "exponent", "label": "Aperiodic Exponent"}
          {"feature": "band", "band": "gamma", "label": "Gamma AUC"}
          {"feature": "lfp_mean", "label": "LFP Mean Amplitude"}
        Simple LFP features ("lfp_mean", "lfp_std", "lfp_exponent") require lfp_windows_by_spike.
    lfp_windows_by_spike : list of dict, optional
        Output of compute_simple_lfp_by_spike. Required for simple LFP features.
    save_dir : str or None
        Directory to save the output pickle.  Defaults to
        config.SPE1_PICKLE_ROOT / "lfp_spk_group_pickles".
    window_width : float
        Width of each sliding analysis window in seconds.
    step_size : float
        Step size between consecutive windows in seconds.
    p_threshold : float
        Significance threshold for pairwise stats.
    force_recompute : bool
        If True, always rerun the full pipeline even if the output pickle already exists.
        If False, load and return the cached pickle when available.
    plot : bool
        If True (default), render heatmap and sliding-stats figures inline.
        Set to False for headless / batch runs — statistics are still computed and saved.

    Returns
    -------
    dict
        Master results dict keyed by feature label, saved to disk as a pickle.
    """
    # Resolve save directory
    if save_dir is None:
        try:
            from config import SPE1_PICKLE_ROOT
        except ImportError:
            SPE1_PICKLE_ROOT = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                os.pardir, os.pardir, os.pardir, "spe1_pickles",
            )
        save_dir = os.path.join(SPE1_PICKLE_ROOT, "lfp_spk_group_pickles")

    os.makedirs(save_dir, exist_ok=True)
    cell_id = f"c{cell_id}" if not str(cell_id).startswith("c") else str(cell_id)
    save_path      = os.path.join(save_dir, f"{cell_id}_sliding_stats.pkl")
    per_spike_path = os.path.join(save_dir, f"{cell_id}_per_spike_data.pkl")

    # Short-circuit: return cached result if available and not forced
    if not force_recompute and os.path.exists(save_path):
        print(f"  [cache] {os.path.basename(save_path)} — skipping recompute (force_recompute=False)")
        with open(save_path, "rb") as f:
            cached = pickle.load(f)
        if plot:
            for feat_info in features_to_analyze:
                feature_type = feat_info.get("feature")
                band  = feat_info.get("band", None)
                label = feat_info.get("label", feature_type.capitalize())
                if feature_type in _SIMPLE_LFP_FEATURES:
                    if lfp_windows_by_spike is None:
                        continue
                    feat_groups = make_simple_lfp_feature_groups(lfp_windows_by_spike, groups, feature=feature_type)
                elif feature_type == "band" and band is not None:
                    feat_groups = make_specparam_feature_groups(specparam_by_spike, groups, feature=feature_type, band=band)
                else:
                    feat_groups = make_specparam_feature_groups(specparam_by_spike, groups, feature=feature_type)
                _ = plot_window_feature_groups_heatmap(feat_groups, feature_label=label, cmap="viridis", time_unit="s", sort_by="next_rel")
                lfp_sliding_stats(
                    feat_groups, ylabel=f"Δ {label}",
                    window_width=window_width, step_size=step_size,
                    p_threshold=p_threshold, plot_mode="per_cluster",
                    plot=True, baseline_window=baseline_window,
                )
        return cached

    cell_master_results = {"cell_id": cell_id}
    per_spike_data      = {"cell_id": cell_id}

    try:
        from tqdm import tqdm as _tqdm
        _feat_iter = _tqdm(features_to_analyze, desc=f"{cell_id} features", unit="feat",
                           file=sys.stdout, dynamic_ncols=False)
    except ImportError:
        _feat_iter = features_to_analyze

    for feat_info in _feat_iter:
        feature_type = feat_info.get("feature")
        band  = feat_info.get("band", None)
        label = feat_info.get("label", feature_type.capitalize())

        print(f"\n{'='*60}\n  RUNNING PIPELINE FOR: {label.upper()}\n{'='*60}\n")

        # 1. Generate Groups
        if feature_type in _SIMPLE_LFP_FEATURES:
            if lfp_windows_by_spike is None:
                print(f"  Skipping {label}: lfp_windows_by_spike not provided.")
                continue
            feat_groups = make_simple_lfp_feature_groups(lfp_windows_by_spike, groups, feature=feature_type)
        elif feature_type == "band" and band is not None:
            feat_groups = make_specparam_feature_groups(specparam_by_spike, groups, feature=feature_type, band=band)
        else:
            feat_groups = make_specparam_feature_groups(specparam_by_spike, groups, feature=feature_type)

        # 2. Heatmap (skipped when plot=False)
        if plot:
            _ = plot_window_feature_groups_heatmap(feat_groups, feature_label=label, cmap="viridis", time_unit="s", sort_by="next_rel")

        # 3. Run Sliding Window Stats
        sig_report = lfp_sliding_stats(
            feat_groups, ylabel=f"Δ {label}",
            window_width=window_width, step_size=step_size,
            p_threshold=p_threshold, plot_mode="per_cluster",
            plot=plot, baseline_window=baseline_window,
        )

        # Save to master dict
        cell_master_results[label] = sig_report

        # --- Build per-spike data for permutation testing ---
        # For each spike_feature (cluster family), pool all spikes across cluster groups,
        # storing the feature-over-time array and the cluster label per spike.
        # This allows permutation tests later without re-running specparam.
        per_spike_data[label] = {}
        cluster_families = {}
        for gname in feat_groups:
            family = gname.split(":")[0].strip() if ":" in gname else "all"
            cluster_families.setdefault(family, []).append(gname)

        for family, gnames in cluster_families.items():
            pooled_windows = []
            pooled_labels  = []
            time_grid      = None

            for gname in gnames:
                g = feat_groups[gname]
                cluster_label = gname.split(":")[-1].strip() if ":" in gname else gname
                for w, t in zip(g["windows"], g["times_rel"]):
                    if len(w) < 2:
                        continue
                    t_s = t / 1000.0 if np.max(np.abs(t)) > 100 else t
                    if time_grid is None:
                        time_grid = t_s.copy()
                    # interpolate onto common grid
                    pooled_windows.append(np.interp(time_grid, t_s, np.asarray(w, float)))
                    pooled_labels.append(cluster_label)

            if len(pooled_windows) > 0:
                per_spike_data[label][family] = {
                    "feature_matrix":  np.array(pooled_windows),   # (n_spikes, n_time_bins)
                    "cluster_labels":  np.array(pooled_labels),     # (n_spikes,)
                    "time_grid":       time_grid,                   # (n_time_bins,)
                }

    # 4. Save sliding stats pickle
    with open(save_path, "wb") as f:
        pickle.dump(cell_master_results, f)

    # 5. Save per-spike data pickle (used for within-cell permutation testing)
    with open(per_spike_path, "wb") as f:
        pickle.dump(per_spike_data, f)

    print(f"\n  Saved: {save_path}")
    print(f"  Saved: {per_spike_path}")
    return cell_master_results





# -----------------------------------------------------------------
# Plotting Function for avg specparam
# -----------------------------------------------------------------
def plot_cluster_spectra_with_aucs(freqs, spectra_matrix, df_aucs, feature_name="Spike Feature", title_suffix=""):
    """Plots the spectral ribbons and a text box with the pre-calculated AUCs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    color_map = {'low': '#1f77b4', 'mid': '#2ca02c', 'high': '#ff7f0e'}
    default_colors = plt.cm.tab10.colors
    
    unique_labels = [l for l in df_aucs['cluster'].unique() if pd.notna(l)]
    sort_order = {"low": 0, "mid": 1, "high": 2}
    unique_labels = sorted(unique_labels, key=lambda x: sort_order.get(str(x).lower(), 99))
    
    stats_text_lines = [f"Average Band AUCs:", "-"*35]
    
    for i, label in enumerate(unique_labels):
        cluster_mask = (df_aucs['cluster'] == label).values
        cluster_spectra = spectra_matrix[cluster_mask]
        
        valid_mask = ~np.isnan(cluster_spectra).all(axis=1)
        valid_spectra = cluster_spectra[valid_mask]
        
        if len(valid_spectra) == 0: continue
            
        mean_spectrum = np.mean(valid_spectra, axis=0)
        std_spectrum = np.std(valid_spectra, axis=0)
        color = color_map.get(str(label).lower(), default_colors[i % len(default_colors)])
        
        ax.plot(freqs, mean_spectrum, label=f'{label} (n={len(valid_spectra)})', color=color, linewidth=2.5)
        ax.fill_between(freqs, mean_spectrum - std_spectrum, mean_spectrum + std_spectrum, color=color, alpha=0.15)
        
        cluster_mean_aucs = df_aucs[cluster_mask][valid_mask].drop(columns=['cluster']).mean()
        
        auc_strings = []
        for band_name, auc_val in cluster_mean_aucs.items():
            if pd.notna(auc_val):
                auc_strings.append(f"{band_name.capitalize()}: {auc_val:.3f}")
                
        stats_text_lines.append(f"{label.upper()} | " + ", ".join(auc_strings))
                         
    ax.set_title(f'Group Average LFP Spectra by {feature_name}\n{title_suffix}', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Frequency (Hz)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Power (log10)', fontsize=12, fontweight='bold')
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
    ax.text(0.97, 0.95, "\n".join(stats_text_lines), transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')
    
    sns.despine()
    ax.legend(frameon=False, loc='lower left', fontsize=11)
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------
# Extraction Function (Accepts any target_t)
# -----------------------------------------------------------------
def extract_target_time_spectra_and_aucs(df_clust, specparam_list, cluster_col, target_t=0.0):
    """
    Extract log10 power spectra and band AUCs at a target time for each spike.

    Works with or without the SpectralTimeModel object — uses stored `powers`
    and `freqs` arrays directly (model is dropped during loading to save memory).
    """
    # Find a valid spike to get freqs and t_bins reference
    valid_res = next((res for res in specparam_list
                      if res is not None and res.get("freqs") is not None
                      and res.get("t_bins_s") is not None), None)
    if valid_res is None:
        return None, None, None, None

    freqs  = np.asarray(valid_res["freqs"])
    t_bins = np.asarray(valid_res["t_bins_s"])
    target_index = int(np.argmin(np.abs(t_bins - target_t)))
    exact_time   = float(t_bins[target_index])

    spectra_list = []
    auc_list     = []

    df_valid = df_clust.loc[df_clust[cluster_col].notna(), ["spk_id", cluster_col]]
    for row in df_valid.itertuples(index=False):
        spk_idx       = int(row.spk_id)
        cluster_label = getattr(row, cluster_col)

        if spk_idx >= len(specparam_list):
            continue
        res = specparam_list[spk_idx]
        if res is None:
            continue

        try:
            t_sp = np.asarray(res["t_bins_s"])
            idx  = int(np.argmin(np.abs(t_sp - target_t)))

            # Log10 power at this time bin (linear → log10)
            if "powers" in res and res["powers"] is not None:
                pows = np.asarray(res["powers"])   # (n_bins, n_freqs)
                full_log = np.log10(pows[idx] + 1e-30)
            else:
                continue

            spike_aucs = {"cluster": cluster_label}
            for band_name, band_array in res.get("band_aucs", {}).items():
                spike_aucs[band_name] = np.asarray(band_array)[idx]

            spectra_list.append(full_log)
            auc_list.append(spike_aucs)

        except Exception:
            continue

    if len(spectra_list) == 0:
        return None, None, None, None

    return freqs, np.array(spectra_list), pd.DataFrame(auc_list), exact_time

# -----------------------------------------------------------------
#Master Loop Function (Reads the Stats Dictionary)
# -----------------------------------------------------------------
def plot_significant_windows_spectra(df_clust, specparam_list, pop_stats_dict, top_n=20):
    """
    Finds significant time windows in the stats dictionary and plots the full
    SpecParam spectra at the midpoint of each window.

    Parameters
    ----------
    top_n : int
        Maximum windows to plot per spike feature, ranked by Cohen's d descending.
        Set to None to plot all.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display
        _has_widgets = True
    except ImportError:
        _has_widgets = False

    cluster_cols = [c for c in df_clust.columns if c.endswith('_cluster')]
    if not cluster_cols:
        print("No cluster columns found!")
        return

    for col in cluster_cols:
        # 1. Gather all significant windows with their best Cohen's d
        sig_windows = []
        for lfp_feat, stats in pop_stats_dict.items():
            if 'pairwise_stats' in stats:
                for stat in stats['pairwise_stats']:
                    if stat['spike_feature'] == col:
                        sig_windows.append((
                            stat['window_start'], stat['window_end'],
                            lfp_feat,
                            stat.get('cohens_d', 0.0),
                        ))

        if not sig_windows:
            print(f"No significant windows found for {col}. Skipping.")
            continue

        # 2. Group by window, keeping max Cohen's d per window
        unique_windows = {}
        for start, end, feat, d in sig_windows:
            w = (start, end)
            if w not in unique_windows:
                unique_windows[w] = {"feats": [], "max_d": 0.0}
            unique_windows[w]["feats"].append(feat)
            unique_windows[w]["max_d"] = max(unique_windows[w]["max_d"], d)

        # 3. Rank by Cohen's d and take top_n
        ranked = sorted(unique_windows.items(), key=lambda x: x[1]["max_d"], reverse=True)
        if top_n is not None:
            ranked = ranked[:top_n]
        print(f"\n{col}: plotting {len(ranked)} most significant windows (of {len(unique_windows)} total)")

        # 4. Render into a scrollable widget
        out = widgets.Output() if _has_widgets else None

        for (start, end), info in ranked:
            mid_t = (start + end) / 2.0
            print(f"  Extracting window {start:.3f}s – {end:.3f}s  (d={info['max_d']:.3f})")

            freqs, spectra_t, df_aucs, exact_time = extract_target_time_spectra_and_aucs(
                df_clust=df_clust,
                specparam_list=specparam_list,
                cluster_col=col,
                target_t=mid_t,
            )

            if spectra_t is not None:
                feature_name_clean = col.replace('_cluster', '').replace('_', ' ').title()
                sig_feats_str = ", ".join(sorted(set(info["feats"])))
                title_suffix = (
                    f"(Sig. Window: {start:.2f}s to {end:.2f}s | "
                    f"Midpoint: {exact_time:.2f}s | d={info['max_d']:.3f})\n"
                    f"[Significant for: {sig_feats_str}]"
                )
                ctx = out if _has_widgets else None
                if ctx is not None:
                    with ctx:
                        plot_cluster_spectra_with_aucs(
                            freqs=freqs, spectra_matrix=spectra_t, df_aucs=df_aucs,
                            feature_name=feature_name_clean, title_suffix=title_suffix,
                        )
                        plt.show()
                else:
                    plot_cluster_spectra_with_aucs(
                        freqs=freqs, spectra_matrix=spectra_t, df_aucs=df_aucs,
                        feature_name=feature_name_clean, title_suffix=title_suffix,
                    )
            else:
                print(f"    -> Failed to extract spectra at t={mid_t:.3f}s")

        if _has_widgets and out is not None:
            display(widgets.Box(
                [out],
                layout=widgets.Layout(
                    height="600px",
                    overflow_y="auto",
                    display="flex",
                    flex_flow="column",
                    border="1px solid #ccc",
                ),
            ))


# =============================================================================
# PRE / POST SPIKE SPECPARAM COMPARISON
# =============================================================================

_PREPOST_DEFAULT_FEATURES = [
    {"key": "exponent",        "label": "Aperiodic Exponent"},
    {"key": "offset",          "label": "Aperiodic Offset"},
    {"key": "r_squared",       "label": "R²"},
    {"key": "band_aucs.theta", "label": "Theta AUC"},
    {"key": "band_aucs.beta",  "label": "Beta AUC"},
    {"key": "band_aucs.gamma", "label": "Gamma AUC"},
]


def _pp_get_feature(spike_data, feat_key, window):
    """Mean of a specparam feature within a time window for one spike."""
    t = spike_data.get("t_bins_s")
    if t is None or len(t) == 0:
        return np.nan
    mask = (t >= window[0]) & (t <= window[1])
    if not np.any(mask):
        return np.nan
    if "." in feat_key:
        top, sub = feat_key.split(".", 1)
        arr = spike_data.get(top, {}).get(sub)
    else:
        arr = spike_data.get(feat_key)
    if arr is None or len(arr) != len(t):
        return np.nan
    vals = np.asarray(arr, float)[mask]
    return np.nanmean(vals) if np.any(np.isfinite(vals)) else np.nan


_PP_MIN_N = 5  # minimum spikes per group to run any test


def _hedges_g_correction(n1, n2):
    """Small-sample correction factor J for Hedges' g from Cohen's d."""
    df = n1 + n2 - 2
    # Use exact gamma-based factor; approximate with 1 - 3/(4*df - 1) for df >= 2
    return 1.0 - 3.0 / (4.0 * df - 1.0) if df >= 2 else np.nan


def _rank_biserial_between(g1, g2):
    """
    Rank-biserial correlation r = 1 - 2U/(n1*n2) for Mann-Whitney U.
    Bounded [-1, 1]; equivalent to common language effect size (P(g1 > g2) * 2 - 1).
    Sign: positive when g1 > g2 on average.
    """
    from scipy.stats import mannwhitneyu
    if len(g1) < 1 or len(g2) < 1:
        return np.nan
    U, _ = mannwhitneyu(g1, g2, alternative="greater")
    return 1.0 - (2.0 * U) / (len(g1) * len(g2))


def _rank_biserial_within(diff):
    """
    Rank-biserial correlation r for Wilcoxon signed-rank test.
    r = W+ / (n*(n+1)/2) where W+ is the sum of positive ranks.
    Bounded [0, 1]; sign given by sign of mean(diff).
    """
    n = len(diff)
    if n < 1:
        return np.nan
    ranks = np.argsort(np.argsort(np.abs(diff))) + 1.0   # rank of |diff|
    W_plus = np.sum(ranks[diff > 0])
    r = W_plus / (n * (n + 1) / 2.0)
    return r * np.sign(np.mean(diff))


def _pp_bootstrap_between(g1, g2, n_bootstrap=1000):
    """
    Between-group effect sizes + permutation p-value for independent samples.

    Effect sizes computed
    ---------------------
    cohens_d  : (mean1 - mean2) / pooled SD  (Cohen 1988)
    hedges_g  : cohens_d * J  — small-sample corrected (Hedges 1981)
    rank_biserial_r : 1 - 2U/(n1*n2)  — effect size for Mann-Whitney U,
                     bounded [-1,1], positive when g1 > g2 stochastically

    Statistics
    ----------
    p-value from permutation test (n_bootstrap permutations).
    When groups are imbalanced (>3:1 ratio) the permutation null is built
    from subsampled equal-size groups to avoid power inflation.
    Bootstrap 95% CI on Cohen's d and Hedges' g.
    """
    g1 = np.asarray(g1, float); g1 = g1[np.isfinite(g1)]
    g2 = np.asarray(g2, float); g2 = g2[np.isfinite(g2)]
    n1, n2 = len(g1), len(g2)
    nan_result = dict(p=np.nan, cohens_d=np.nan, hedges_g=np.nan,
                      rank_biserial_r=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                      n1=n1, n2=n2, resampled=False, skipped=True)
    if n1 < _PP_MIN_N or n2 < _PP_MIN_N:
        return nan_result

    def _d(a, b):
        na, nb = len(a), len(b)
        pooled = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
        return (np.mean(a) - np.mean(b)) / (pooled + 1e-10)

    obs_d = _d(g1, g2)
    J     = _hedges_g_correction(n1, n2)
    obs_g = obs_d * J
    obs_r = _rank_biserial_between(g1, g2)

    # Permutation null (with subsampling if imbalanced)
    resampled = max(n1, n2) / min(n1, n2) > 3
    n_min = min(n1, n2)
    if resampled:
        null = []
        for _ in range(n_bootstrap):
            s1 = np.random.choice(g1, n_min, replace=False)
            s2 = np.random.choice(g2, n_min, replace=False)
            perm = np.random.permutation(np.concatenate([s1, s2]))
            null.append(_d(perm[:n_min], perm[n_min:]))
    else:
        combined = np.concatenate([g1, g2])
        null = [_d(np.random.permutation(combined)[:n1],
                   np.random.permutation(combined)[n1:])
                for _ in range(n_bootstrap)]
    p = np.mean(np.abs(null) >= np.abs(obs_d))

    # Bootstrap CI on Cohen's d (and Hedges' g = d * J)
    boot_d = [_d(np.random.choice(g1, n1, replace=True),
                 np.random.choice(g2, n2, replace=True))
              for _ in range(n_bootstrap)]
    ci_lo, ci_hi = np.percentile(boot_d, [2.5, 97.5])

    return dict(p=p,
                cohens_d=obs_d, hedges_g=obs_g,
                rank_biserial_r=obs_r,
                ci_lo=ci_lo, ci_hi=ci_hi,
                g_ci_lo=ci_lo * J, g_ci_hi=ci_hi * J,
                n1=n1, n2=n2, resampled=resampled, skipped=False)


def _pp_bootstrap_within(pre, post, n_bootstrap=1000):
    """
    Within-group (paired) effect sizes + Wilcoxon signed-rank p-value.

    Effect sizes computed
    ---------------------
    cohens_dz : mean(post-pre) / SD(post-pre)  — paired Cohen's d
                (equivalent to the d computed on difference scores; Lakens 2013)
    hedges_gz : cohens_dz * J  — small-sample corrected paired effect size
    rank_biserial_r : W+ / (n*(n+1)/2) * sign(mean_diff)
                      effect size for Wilcoxon signed-rank, bounded [-1,1]
    mean_diff : raw mean of (post - pre) for interpretability

    Statistics
    ----------
    Wilcoxon signed-rank p-value (two-sided).
    Bootstrap 95% CI on cohens_dz and mean_diff.
    """
    from scipy.stats import wilcoxon
    pre  = np.asarray(pre,  float)
    post = np.asarray(post, float)
    valid = np.isfinite(pre) & np.isfinite(post)
    pre, post = pre[valid], post[valid]
    n = len(pre)
    nan_result = dict(p=np.nan, cohens_dz=np.nan, hedges_gz=np.nan,
                      rank_biserial_r=np.nan, mean_diff=np.nan,
                      ci_lo=np.nan, ci_hi=np.nan, n=n, skipped=True)
    if n < _PP_MIN_N:
        return nan_result

    diff      = post - pre
    mean_diff = np.mean(diff)
    sd_diff   = np.std(diff, ddof=1)

    if np.allclose(diff, 0) or sd_diff < 1e-10:
        return dict(p=1.0, cohens_dz=0.0, hedges_gz=0.0,
                    rank_biserial_r=0.0, mean_diff=0.0,
                    ci_lo=0.0, ci_hi=0.0, n=n, skipped=False)

    # Cohen's d_z and Hedges' g_z (paired correction uses df = n-1)
    dz = mean_diff / sd_diff
    J  = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0) if n > 2 else np.nan
    gz = dz * J

    # Rank-biserial r for Wilcoxon
    rb_r = _rank_biserial_within(diff)

    # Wilcoxon p-value
    try:
        _, p = wilcoxon(diff, alternative="two-sided")
    except Exception:
        p = np.nan

    # Bootstrap CI on d_z and mean_diff
    idx = np.arange(n)
    boot_dz   = []
    boot_diff = []
    for _ in range(n_bootstrap):
        b = np.random.choice(idx, n, replace=True)
        d_ = diff[b]
        md_ = np.mean(d_); sd_ = np.std(d_, ddof=1)
        boot_diff.append(md_)
        boot_dz.append(md_ / sd_ if sd_ > 1e-10 else 0.0)

    ci_lo,    ci_hi    = np.percentile(boot_diff, [2.5, 97.5])
    dz_ci_lo, dz_ci_hi = np.percentile(boot_dz,  [2.5, 97.5])

    return dict(p=p,
                cohens_dz=dz, hedges_gz=gz,
                rank_biserial_r=rb_r,
                mean_diff=mean_diff,
                ci_lo=ci_lo, ci_hi=ci_hi,
                dz_ci_lo=dz_ci_lo, dz_ci_hi=dz_ci_hi,
                n=n, skipped=False)


def compute_pre_post_specparam_comparison(
    specparam_by_spike,
    df_features_clust,
    pre_window=(-0.5, -0.05),
    post_window=(0.05, 1.0),
    features=None,
    n_bootstrap=1000,
    p_thresh=0.05,
    plot=True,
    cell_id="",
    save_path=None,
    force=False,
):
    """
    Compare specparam features between cluster groups in pre- and post-spike windows.

    For each cluster column and each feature:
      - Bootstrap permutation test between cluster groups (pre window, post window)
      - Bootstrap Wilcoxon test: pre vs post within each group
      - BH-FDR correction across all tests per cluster column
      - Violin plots (if plot=True)

    Parameters
    ----------
    specparam_by_spike : list of dict
        Output of load_chunked_specparam_results / run_time_resolved_specparam_per_spike.
    df_features_clust : pd.DataFrame
        Spike feature DataFrame with *_cluster columns. Row i corresponds to spike i.
    pre_window : (float, float)
        (start_s, end_s) of the pre-spike averaging window relative to spike time.
    post_window : (float, float)
        (start_s, end_s) of the post-spike averaging window.
    features : list of dict, optional
        Each dict has 'key' (e.g. 'exponent' or 'band_aucs.theta') and 'label'.
        Defaults to exponent, offset, R², theta/beta/gamma AUC.
    n_bootstrap : int
        Bootstrap / permutation iterations.
    p_thresh : float
        Significance threshold (after FDR correction) for plot annotations.
    plot : bool
        Whether to produce violin plots.
    cell_id : str
        Used in plot titles and printed output.
    save_path : str or None
        If given, saves the results dict as a pickle.

    Returns
    -------
    results : dict
        Keyed by cluster column. Each value contains 'tests' (list of dicts with
        p, p_fdr, cohens_d / mean_diff, CIs) and summary arrays for population use.
    """
    from itertools import combinations
    from statsmodels.stats.multitest import multipletests

    if not force and save_path and os.path.exists(save_path):
        print(f"  [cache] {os.path.basename(save_path)}")
        with open(save_path, "rb") as fh:
            results = pickle.load(fh)
        if plot:
            if features is None:
                features = _PREPOST_DEFAULT_FEATURES
            for col, col_res in results.items():
                _plot_pre_post_specparam(col_res, features, col, cell_id, p_thresh)
        return results

    if features is None:
        features = _PREPOST_DEFAULT_FEATURES

    cluster_cols = [c for c in df_features_clust.columns if c.endswith("_cluster")]
    n_spikes = len(specparam_by_spike)

    # ── 1. Extract per-spike window means ─────────────────────────────────────
    pre_vals  = {f["key"]: np.full(n_spikes, np.nan) for f in features}
    post_vals = {f["key"]: np.full(n_spikes, np.nan) for f in features}

    # Detect freq axis from first spike that has powers
    freqs = next((np.asarray(s["freqs"]) for s in specparam_by_spike if "freqs" in s), None)
    n_freqs = len(freqs) if freqs is not None else 0
    pre_spectra  = np.full((n_spikes, n_freqs), np.nan) if n_freqs else None
    post_spectra = np.full((n_spikes, n_freqs), np.nan) if n_freqs else None

    for i, spike in enumerate(tqdm(specparam_by_spike, desc="Extracting pre/post means",
                                    file=sys.stdout, dynamic_ncols=False)):
        for f in features:
            pre_vals[f["key"]][i]  = _pp_get_feature(spike, f["key"], pre_window)
            post_vals[f["key"]][i] = _pp_get_feature(spike, f["key"], post_window)

        # Mean power spectrum within each window
        if n_freqs and "powers" in spike and "t_bins_s" in spike:
            t    = np.asarray(spike["t_bins_s"])
            pows = np.asarray(spike["powers"])   # (n_bins, n_freqs)
            pre_mask  = (t >= pre_window[0])  & (t <= pre_window[1])
            post_mask = (t >= post_window[0]) & (t <= post_window[1])
            if np.any(pre_mask):
                pre_spectra[i]  = np.nanmean(pows[pre_mask],  axis=0)
            if np.any(post_mask):
                post_spectra[i] = np.nanmean(pows[post_mask], axis=0)

    results = {}

    # Align df to specparam_by_spike — edge spikes may have been dropped during
    # window extraction, so df can be slightly longer than specparam_by_spike.
    df_aligned = df_features_clust.iloc[:n_spikes].reset_index(drop=True)
    if len(df_features_clust) != n_spikes:
        print(f"  Note: aligning df ({len(df_features_clust)} spikes) → "
              f"specparam ({n_spikes} spikes); {len(df_features_clust) - n_spikes} edge spikes dropped.")

    for col in cluster_cols:
        labels = df_aligned[col].values
        unique  = [l for l in ["low", "mid", "high"] if l in labels]
        if len(unique) < 2:
            continue

        all_tests = []

        for feat in features:
            key, label = feat["key"], feat["label"]
            g_pre  = {l: pre_vals[key][labels == l]  for l in unique}
            g_post = {l: post_vals[key][labels == l] for l in unique}

            # Between-group: pre window  (A vs B before spike)
            for l1, l2 in combinations(unique, 2):
                res = _pp_bootstrap_between(g_pre[l1], g_pre[l2], n_bootstrap)
                all_tests.append({**res, "feature": label, "feat_key": key,
                                   "comparison": f"{l1} vs {l2}", "window": "pre"})

            # Between-group: post window  (A vs B after spike)
            for l1, l2 in combinations(unique, 2):
                res = _pp_bootstrap_between(g_post[l1], g_post[l2], n_bootstrap)
                all_tests.append({**res, "feature": label, "feat_key": key,
                                   "comparison": f"{l1} vs {l2}", "window": "post"})

            # Within-group: pre vs post  (modulation within each cluster)
            for l in unique:
                idx = labels == l
                res = _pp_bootstrap_within(pre_vals[key][idx], post_vals[key][idx], n_bootstrap)
                all_tests.append({**res, "feature": label, "feat_key": key,
                                   "comparison": f"pre→post ({l})", "window": "within"})

            # Interaction: does the pre→post change differ between groups?
            # Tests whether spike waveform clusters have different LFP modulation
            # H0: (A_post - A_pre) == (B_post - B_pre)
            for l1, l2 in combinations(unique, 2):
                idx1 = labels == l1; idx2 = labels == l2
                change1 = post_vals[key][idx1] - pre_vals[key][idx1]
                change2 = post_vals[key][idx2] - pre_vals[key][idx2]
                res = _pp_bootstrap_between(change1, change2, n_bootstrap)
                all_tests.append({**res, "feature": label, "feat_key": key,
                                   "comparison": f"{l1} vs {l2} Δ(post-pre)",
                                   "window": "interaction"})

        # BH-FDR correction
        pvals = [t["p"] for t in all_tests]
        finite_mask = np.isfinite(pvals)
        p_fdr = np.full(len(pvals), np.nan)
        if finite_mask.any():
            _, p_adj, _, _ = multipletests(
                np.where(finite_mask, pvals, 1.0), method="fdr_bh")
            p_fdr[finite_mask] = p_adj[finite_mask]
        for t, pf in zip(all_tests, p_fdr):
            t["p_fdr"] = pf
            t["sig"] = ("***" if pf < 0.001 else "**" if pf < 0.01
                        else "*" if pf < p_thresh else "ns")

        results[col] = {
            "tests":          all_tests,
            "pre_vals":       {f["key"]: pre_vals[f["key"]] for f in features},
            "post_vals":      {f["key"]: post_vals[f["key"]] for f in features},
            "pre_spectra":    pre_spectra,
            "post_spectra":   post_spectra,
            "freqs":          freqs,
            "cluster_labels": labels,
            "unique_labels":  unique,
            "pre_window":     pre_window,
            "post_window":    post_window,
            "cell_id":        cell_id,
        }

    # Save before plotting so a plot crash never prevents the pickle being written
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as fh:
            pickle.dump(results, fh)
        print(f"  Saved: {os.path.basename(save_path)}")

    if plot:
        for col, col_res in results.items():
            try:
                _plot_pre_post_specparam(col_res, features, col, cell_id, p_thresh)
            except Exception as e:
                print(f"  [plot warning] {col}: {e}")

    return results


def _plot_pre_post_specparam(col_results, features, col_name, cell_id, p_thresh):
    """
    Two plots per cluster column, matching the style of plot_cluster_spectra_with_aucs:
      1. Mean log10 power spectrum (pre solid, post dashed) per group + specparam fit
      2. Violin plots of derived features with significance annotations
    """
    from specparam import SpectralModel

    unique     = col_results["unique_labels"]
    labels     = col_results["cluster_labels"]
    pre_vals   = col_results["pre_vals"]
    post_vals  = col_results["post_vals"]
    tests      = col_results["tests"]
    pre_specs  = col_results.get("pre_spectra")   # (n_spikes, n_freqs) linear power
    post_specs = col_results.get("post_spectra")
    freqs      = col_results.get("freqs")

    COLOR_MAP     = {"low": "#1f77b4", "mid": "#2ca02c", "high": "#ff7f0e"}
    default_colors = plt.cm.tab10.colors
    feat_name     = col_name.replace("_cluster", "").replace("_", " ").title()

    # ── Plot 1: average spectra + specparam fit ───────────────────────────────
    if pre_specs is not None and freqs is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(
            f"Group Average LFP Spectra by {feat_name} — Pre vs Post Spike\n{cell_id}",
            fontsize=14, fontweight="bold", pad=15)

        stats_lines = ["Average Band AUCs (pre | post):", "-" * 40]

        for i, lbl in enumerate(unique):
            idx   = labels == lbl
            color = COLOR_MAP.get(str(lbl).lower(), default_colors[i % len(default_colors)])
            n     = int(idx.sum())

            for specs, tag, ls in [(pre_specs, "pre", "-"), (post_specs, "post", "--")]:
                grp = np.log10(specs[idx] + 1e-30)          # linear → log10
                valid = ~np.isnan(grp).all(axis=1)
                grp   = grp[valid]
                if len(grp) == 0:
                    continue
                mean_s = np.nanmean(grp, axis=0)
                std_s  = np.nanstd(grp,  axis=0)
                ax.plot(freqs, mean_s, color=color, lw=2.5, ls=ls,
                        label=f"{lbl} {tag} (n={n})")
                ax.fill_between(freqs, mean_s - std_s, mean_s + std_s,
                                color=color, alpha=0.10 if tag == "pre" else 0.18)

                # Fit SpectralModel to group-mean spectrum and overlay
                try:
                    sm = SpectralModel(verbose=False)
                    sm.fit(freqs, 10 ** mean_s)          # fit in linear, log internally
                    fit_log = sm.get_model(component="full", space="log")
                    rsq     = sm.r_squared_
                    if fit_log is not None:
                        ax.plot(freqs, fit_log, color=color, lw=1.2, ls=ls,
                                alpha=0.6, zorder=3,
                                label=f"{lbl} {tag} fit (R²={rsq:.3f})")
                except Exception:
                    pass

            # AUC + R² text for pre and post
            auc_keys = [f["key"] for f in features if f["key"].startswith("band_aucs.")]
            auc_parts = []
            for fk in auc_keys:
                band   = fk.split(".", 1)[1]
                pre_m  = np.nanmean(pre_vals[fk][idx])
                post_m = np.nanmean(post_vals[fk][idx])
                auc_parts.append(f"{band.capitalize()}: {pre_m:.3f} | {post_m:.3f}")
            rsq_pre  = np.nanmean(pre_vals["r_squared"][idx])  if "r_squared" in pre_vals  else np.nan
            rsq_post = np.nanmean(post_vals["r_squared"][idx]) if "r_squared" in post_vals else np.nan
            auc_parts.append(f"R²: {rsq_pre:.3f} | {rsq_post:.3f}")
            stats_lines.append(f"{lbl.upper()} | " + ", ".join(auc_parts))

        props = dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray")
        ax.text(0.97, 0.95, "\n".join(stats_lines), transform=ax.transAxes, fontsize=9,
                verticalalignment="top", horizontalalignment="right",
                bbox=props, family="monospace")
        ax.set_xlabel("Frequency (Hz)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Power (log10)", fontsize=12, fontweight="bold")
        ax.legend(frameon=False, loc="lower left", fontsize=10)
        sns.despine()
        plt.tight_layout()
        plt.show()

    # ── Helper: draw a significance bracket ──────────────────────────────────
    def _bracket(ax, x1, x2, y, text, dy=0.03):
        yrange = ax.get_ylim()
        h = (yrange[1] - yrange[0]) * dy
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=0.8, color="black", clip_on=False)
        ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom", fontsize=7)

    def _es_str(t):
        """Short effect-size string for a test result."""
        if "cohens_dz" in t and not np.isnan(t.get("cohens_dz", np.nan)):
            return f"d_z={t['cohens_dz']:.2f}, r={t.get('rank_biserial_r', np.nan):.2f}"
        return (f"g={t.get('hedges_g', np.nan):.2f}, "
                f"r={t.get('rank_biserial_r', np.nan):.2f}")

    def _lookup(tests, feat_key, window, comparison_contains):
        return next((t for t in tests
                     if t["feat_key"] == feat_key
                     and t["window"] == window
                     and comparison_contains in t["comparison"]), None)

    # ── Plot 2: violin plots per feature ──────────────────────────────────────
    n_feat = len(features)
    fig, axes = plt.subplots(1, n_feat, figsize=(3.8 * n_feat, 5), sharey=False)
    if n_feat == 1:
        axes = [axes]
    fig.suptitle(
        f"Group LFP Features by {feat_name} — Pre vs Post Spike  |  {cell_id}",
        fontsize=11, fontweight="bold")

    for ax, feat in zip(axes, features):
        key, label = feat["key"], feat["label"]

        # Layout: [A_pre, B_pre, (C_pre,)]  |gap|  [A_post, B_post, (C_post,)]
        # Positions keyed by (window, label)
        pos_map = {}
        pos = 0
        for win_tag, vals_dict, alpha in [("pre", pre_vals, 0.5), ("post", post_vals, 0.85)]:
            for i, lbl in enumerate(unique):
                pos_map[(win_tag, lbl)] = pos
                idx   = labels == lbl
                color = COLOR_MAP.get(str(lbl).lower(), default_colors[i % len(default_colors)])
                arr   = vals_dict[key][idx]; arr = arr[np.isfinite(arr)]
                if len(arr):
                    vp = ax.violinplot([arr], [pos], widths=0.65,
                                       showmedians=True, showextrema=False)
                    for pc in vp["bodies"]:
                        pc.set_facecolor(color); pc.set_alpha(alpha)
                    vp["cmedians"].set_color(color)
                pos += 1
            pos += 0.6  # gap between windows

        xtick_positions = list(pos_map.values())
        xtick_labels    = [f"{lbl}\n{win}" for (win, lbl) in pos_map.keys()]
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, fontsize=7)

        # Dotted separator between pre and post groups
        sep_x = (pos_map[("pre", unique[-1])] + pos_map[("post", unique[0])]) / 2
        ax.axvline(sep_x, color="gray", lw=0.8, ls=":")

        ax.set_title(label, fontsize=9, fontweight="bold")
        sns.despine(ax=ax)

        # ── Significance brackets ──────────────────────────────────────────
        ax.autoscale(enable=True, axis="y")
        plt.draw()  # needed so get_ylim() is current
        ymax = ax.get_ylim()[1]
        step = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.07

        bracket_y = ymax
        for l1, l2 in combinations(unique, 2):
            for win in ["pre", "post"]:
                t = _lookup(tests, key, win, f"{l1} vs {l2}")
                if t is None:
                    continue
                x1, x2 = pos_map[(win, l1)], pos_map[(win, l2)]
                text = f"{t['sig']} [{win}]\n{_es_str(t)}"
                _bracket(ax, x1, x2, bracket_y, text)
                bracket_y += step * 2.2

            # pre→post within each group
            for lbl in [l1, l2]:
                t = _lookup(tests, key, "within", f"pre→post ({lbl})")
                if t is None:
                    continue
                x1, x2 = pos_map[("pre", lbl)], pos_map[("post", lbl)]
                text = f"{lbl}: pre→post {t['sig']}\n{_es_str(t)}"
                _bracket(ax, x1, x2, bracket_y, text)
                bracket_y += step * 2.2

            # Interaction
            t = _lookup(tests, key, "interaction", f"{l1} vs {l2}")
            if t is not None:
                x1 = (pos_map[("pre", l1)] + pos_map[("pre", l2)]) / 2
                x2 = (pos_map[("post", l1)] + pos_map[("post", l2)]) / 2
                text = f"Interaction {t['sig']}\n{_es_str(t)}"
                _bracket(ax, x1, x2, bracket_y, text)
                bracket_y += step * 2.2

    plt.tight_layout()
    plt.show()
