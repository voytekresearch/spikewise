"""
skp_feat_cluster_analysis.py
----------------------------
Analysis utilities for identifying and visualizing spike feature clustering patterns
and their temporal dynamics.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from tqdm.auto import tqdm
import mne
from specparam import SpectralTimeModel
from typing import Optional, List, Tuple, Dict, Any, Literal, Callable, Union, Sequence
import scipy.stats as stats
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu, probplot, f_oneway, kruskal, zscore
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
import os
import re
import pickle
import gc


from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from itertools import combinations




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

    # 3. Memory-safe loading
    specparam_by_spike = []
    
    for file_name in tqdm(all_files, desc="Loading Chunks"):
        file_path = os.path.join(save_dir, file_name)
        
        with open(file_path, 'rb') as f:
            chunk_data = pickle.load(f)
        
        # Extend the master list
        specparam_by_spike.extend(chunk_data)
        
        # Free up memory immediately
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




def compute_cluster_statistics(group_data, alpha=0.05):
    """
    Compute statistics for cluster comparisons.
    """
    # ADD THESE IMPORTS INSIDE THE FUNCTION
    from scipy import stats
    import numpy as np
    
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
    
    test_type = "nonparametric"
    
    if n_groups == 2:
        # Mann-Whitney U test
        g1, g2 = group_data[0], group_data[1]
        
        # Handle edge cases
        if len(g1) == 0 or len(g2) == 0:
            return {
                'p_value': None,
                'significance': None,
                'test_type': test_type,
                'n_groups': n_groups,
                'eta_squared': None,
                'error': 'One group has no data'
            }
        
        stat, p = stats.mannwhitneyu(g1, g2)
        test_name = "Mann-Whitney U"
        
        # Direct calculation of η² from data
        all_data = np.concatenate([g1, g2])
        grand_mean = np.mean(all_data)
        
        # Sum of squares between
        ss_between = len(g1) * (np.mean(g1) - grand_mean) ** 2 + len(g2) * (np.mean(g2) - grand_mean) ** 2
        
        # Sum of squares total
        ss_total = np.sum((all_data - grand_mean) ** 2)
        
        # η² = SS_between / SS_total
        if ss_total > 0:
            eta_squared = ss_between / ss_total
        else:
            eta_squared = 0
        
    else:  # 3+ groups
        # Kruskal-Wallis test
        stat, p = stats.kruskal(*group_data)
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
    plot_spike_clusters_from_df(df, sp, cluster_col)
    
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
    
    group_indices = range(len(groups))
    pairs = list(combinations(group_indices, 2))
    
    rows = []
    for idx1, idx2 in pairs:
        g1, g2 = group_names[idx1], group_names[idx2]
        
        # Match the key in wf_metrics (checking both orders)
        pair_key = f"{g1}_vs_{g2}"
        alt_key = f"{g2}_vs_{g1}"
        metrics = wf_metrics.get(pair_key) or wf_metrics.get(alt_key) or {}

        # The specific minimalist structure
        row = {
            "cluster_group_id": cluster_group_id,
            "feature_clustered": clustered_feature,
            "groups": f"{g1}-{g2}",
            "nRMSE": metrics.get('nrmse', np.nan),
            "cos_sim": metrics.get('cos_sim', np.nan)
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
):
    """
    Plot spike waveforms grouped by clusters using df['spk_id'] to index spikes.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
            cluster_col : cluster labels (e.g. 'peak_width_cluster')
            'spk_id'    : the true spike index into sp.spikes
    sp : Spike
        The Spike object containing the actual waveforms
    cluster_col : str
        Cluster feature column name
    """

    # Extract cluster labels
    labels = sorted(df[cluster_col].dropna().unique().tolist())


    # Default color scheme
    colors = {lab: get_cluster_color(lab) for lab in labels}


    # Build index groups based on df['spk_id']
    ind_groups = []
    group_names = []

    for lab in labels:
        inds = df.loc[df[cluster_col] == lab, "spk_id"].astype(int).tolist()
        if len(inds) > 0:
            ind_groups.append(inds)
            group_names.append(str(lab))

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    # Use the Spike class’ existing plotting engine
    sp.plot(
        inds=None,                   # we supply groups below
        mode=mode,
        in_ms=True,
        show_points=False,
        ax=ax,
        groups=True,                 # activates group overlays
        ind_groups=ind_groups,       # list of lists of spike indices
        group_names=group_names,
        plot_average=plot_average,
        plot_average_std=plot_average_std,
    )

    # Update average line colors based on cluster colors
    for line in ax.lines:
        lab = line.get_label()
        for cluster_label in labels:
            if cluster_label in lab:
                line.set_color(colors[cluster_label])

    ax.set_title(title or f"Spike waveforms grouped by {cluster_col}")
    plt.tight_layout()

    return fig, ax



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




def avg_waveforms_rmse(sp, df, cluster_col, groups, group_names, color_map):
    """
    Plot average waveforms with 3 distinct panels per pair.
    
    1. Raw (Absolute Amplitude) -> RMSE
    2. Scaled (Relative Amplitude) -> NRMSE
    3. Z-Scored (Pure Shape) -> Cosine Similarity
    """
    
    # --- 1. Data Extraction ---
    avg_waveforms = {}
    waveform_counts = {}
    
    for i, group in enumerate(groups):
        spike_indices = df.loc[df[cluster_col] == group, "spk_id"].astype(int).tolist()
        if not spike_indices: continue
        
        wfs = []
        for idx in spike_indices:
            if idx < len(sp.spikes): wfs.append(sp.spikes[idx])
        if not wfs: continue
        
        wfs = np.array(wfs)
        avg_waveforms[group] = np.mean(wfs, axis=0)
        waveform_counts[group] = len(wfs)
    
    n_groups = len(avg_waveforms)
    if n_groups < 2: return {}

    # --- 2. Helper to calculate metrics ---
    def get_metrics(wf1, wf2):
        # A. Raw RMSE
        rmse = np.sqrt(np.mean((wf1 - wf2)**2))
        
        # B. Max-Scaled (Preserves relative amplitude ratio)
        global_max = np.max([np.abs(wf1), np.abs(wf2)])
        if global_max == 0: global_max = 1e-9
        n_wf1, n_wf2 = wf1 / global_max, wf2 / global_max
        nrmse = np.sqrt(np.mean((n_wf1 - n_wf2)**2))
        
        # C. Z-Scored (Pure Shape / Cosine Proxy)
        z_wf1, z_wf2 = zscore(wf1), zscore(wf2)
        
        # Cosine Similarity
        cos_sim = cosine_similarity(wf1.reshape(1, -1), wf2.reshape(1, -1))[0][0]
        
        return rmse, nrmse, cos_sim, n_wf1, n_wf2, z_wf1, z_wf2

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
            
            rmse, nrmse, cos_sim, n_wf1, n_wf2, z_wf1, z_wf2 = get_metrics(wf1, wf2)
            time_axis = np.arange(len(wf1)) - len(wf1) // 2
            
            # --- PANEL 1: Raw Amplitude (RMSE) ---
            ax_raw = axes[pair_idx, 0]
            ax_raw.plot(time_axis, wf1, color=c1, lw=2, label=name1)
            ax_raw.plot(time_axis, wf2, color=c2, lw=2, label=name2)
            ax_raw.fill_between(time_axis, wf1, wf2, color='gray', alpha=0.2)
            
            ax_raw.text(0.05, 0.95, f"RMSE: {rmse:.2f}", transform=ax_raw.transAxes, 
                        va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
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
            
            ax_shape.set_title(f"3. Pure Shape (Z-Scored)")
            ax_shape.set_ylabel("Z-Score (SD)")
            ax_shape.grid(True, alpha=0.3)
            
            # Only label x-axis on the very last row
            if pair_idx == n_pairs - 1:
                ax_raw.set_xlabel("Samples")
                ax_scale.set_xlabel("Samples")
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
    remap = {old_i: new_i for new_i, old_i in enumerate(order)}
    assign_sorted = np.vectorize(remap.get)(assign)
    cutoffs = (centers[:-1] + centers[1:]) / 2.0

    # Write labels back at original indices
    label_array = np.full(x.shape, np.nan, dtype=object)
    for i in range(k):
        full_idx = idx_valid[assign_sorted == i]
        label_array[full_idx] = labels[i]
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

    # Build helper for spike→next spike lookup
    df_sorted = spk_df.sort_values("spk_id").set_index("spk_id")

    windows = []
    times_rel_ms = []
    spike_times = []
    next_spike_times = []
    meta_rows = []

    for _, row in df.iterrows():

        sid = int(row["spk_id"])
        t0 = float(row["spk_times_ms"])

        # Lookup next spike
        if sid + 1 in df_sorted.index:
            t_next = float(df_sorted.loc[sid + 1, "spk_times_ms"])
        else:
            if requires_next_spike:
                continue          # ONLY drop for ISI-based analyses
            else:
                t_next = np.nan   # keep spike

        # Window boundaries
        t_start = t0 - pre_ms
        t_end = t0 + post_ms

        # Clip windows to LFP range
        if t_start < lfp_times_ms[0] or t_end > lfp_times_ms[-1]:
            continue

        # Index into LFP vector
        idx0 = np.searchsorted(lfp_times_ms, t_start)
        idx1 = np.searchsorted(lfp_times_ms, t_end)

        seg = lfp_signal[idx0:idx1]
        t_seg = lfp_times_ms[idx0:idx1]

        # build relative time (spike = 0)
        t_rel = t_seg - t0

        windows.append(seg)
        times_rel_ms.append(t_rel)
        spike_times.append(t0)
        next_spike_times.append(t_next)

        meta_rows.append({
            "spk_id": sid,
            "spike_time_ms": t0,
            "next_spike_time_ms": t_next,
            "t_start_ms": t_start,
            "t_end_ms": t_end
        })

    return {
        "windows": windows,
        "times_rel_ms": times_rel_ms,
        "spike_times_ms": spike_times,
        "next_spike_times_ms": next_spike_times,
        "meta_df": pd.DataFrame(meta_rows)
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
        time_bandwidth=tb, output="power", decim=decim_factor, verbose=False
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
from tqdm.auto import tqdm
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
                 desc="Specparam per spike")
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
        
        for i in tqdm(range(start_idx, end_idx), desc=f"Chunk {chunk_idx}/{len(chunk_indices)-1}"):
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
# --------------------  Time-resolved and window visualziations  --------------------- #
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
    figsize=(18, 7) # Wider figure for side-by-side layout
):
    """
    Slides a window to find significant regions. 
    Demeans each individual trace by its own WHOLE-TRACE average.
    Layout: Trace on LEFT, Boxplots on RIGHT.
    """


    # --- HELPERS ---
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

    # Organize groups
    cluster_families = {}
    for gname in feat_groups.keys():
        family = gname.split(":")[0] if ":" in gname else "all"
        cluster_families.setdefault(family, []).append(gname)

    plot_sets = {}
    if plot_mode in ["all", "both"]: plot_sets["All groups"] = list(feat_groups.keys())
    if plot_mode in ["per_cluster", "both"]:
        for k, v in cluster_families.items(): plot_sets[k] = v

    default_palette = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for set_name, unsorted_names in plot_sets.items():
        if len(unsorted_names) < 2: continue
        
        group_names = sort_clusters(unsorted_names)
        current_mode = "all" if set_name == "All groups" else "per_cluster"
        
        # --- DATA PREP ---
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
            
            mats = []
            for w, t in zip(g["windows"], g["times_rel"]):
                if len(w) != len(t) or len(w) < 2: continue
                w_demeaned = np.asarray(w, float) - np.nanmean(w)
                mats.append(np.interp(Tgrid, (t/1000.0 if time_unit=="ms" else t), w_demeaned))
            if len(mats) >= 2: A_matrices[gname] = np.vstack(mats)

        # Sliding Window Math
        sig_windows = []
        for w_start in np.arange(Tgrid[0], Tgrid[-1] - window_width, step_size):
            amask = (Tgrid >= w_start) & (Tgrid <= w_start + window_width)
            data_for_test = [np.nanmean(A_matrices[gn][:, amask], axis=1) for gn in group_names if gn in A_matrices]
            data_for_test = [d[np.isfinite(d)] for d in data_for_test if len(d) > 2]
            
            if len(data_for_test) >= 2:
                p = f_oneway(*data_for_test)[1] if len(data_for_test) > 2 else ttest_ind(data_for_test[0], data_for_test[1], equal_var=False)[1]
                if p < p_threshold:
                    sig_windows.append((w_start, w_start + window_width, p))

        merged_regions = []
        if sig_windows:
            cur_s, cur_e, cur_p = sig_windows[0]
            for w in sig_windows[1:]:
                if w[0] <= cur_e + 1e-5:
                    cur_e, cur_p = max(cur_e, w[1]), min(cur_p, w[2])
                else:
                    merged_regions.append((cur_s, cur_e, cur_p)); cur_s, cur_e, cur_p = w
            merged_regions.append((cur_s, cur_e, cur_p))

        # --- NEW SIDE-BY-SIDE LAYOUT ---
        fig = plt.figure(figsize=figsize)
        has_regions = len(merged_regions) > 0
        
        # 1 Row, 2 Columns (Trace gets more space)
        master_gs = GridSpec(1, 2, width_ratios=[1.5, 1] if has_regions else [1, 0.01], wspace=0.3)
        
        # LEFT SIDE: TRACE
        ax_trace = fig.add_subplot(master_gs[0])
        for gname in group_names:
            if gname not in A_matrices: continue
            mean, std = np.nanmean(A_matrices[gname], axis=0), np.nanstd(A_matrices[gname], axis=0)
            ci = 1.96 * std / np.sqrt(np.sum(np.isfinite(A_matrices[gname]), axis=0))
            ax_trace.plot(Tgrid, mean, label=gname, color=colors_dict[gname], lw=2.5)
            ax_trace.fill_between(Tgrid, mean-ci, mean+ci, color=colors_dict[gname], alpha=alpha_ci, lw=0)

        for s, e, _ in merged_regions:
            ax_trace.axvspan(s, e, color='gold', alpha=0.15)
        ax_trace.axvline(0, color="k", ls="--", lw=1.5)
        ax_trace.set_title(f"Temporal Dynamics: {set_name}", fontweight='bold')
        ax_trace.set_ylabel(ylabel)
        ax_trace.legend(loc="upper right", frameon=True, fontsize=9)
        ax_trace.grid(False)

        # RIGHT SIDE: BOXPLOTS (Stacked or Grid)
        if has_regions:
            n_boxes = len(merged_regions)
            # If many regions, make a 2-col grid on the right, otherwise 1-col
            n_cols = 2 if n_boxes > 2 else 1
            n_rows = (n_boxes + n_cols - 1) // n_cols
            sub_gs = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=master_gs[1], wspace=0.4, hspace=0.6)
            
            for idx, (s, e, _) in enumerate(merged_regions):
                ax_box = fig.add_subplot(sub_gs[idx])
                amask = (Tgrid >= s) & (Tgrid <= e)
                box_data = [np.nanmean(A_matrices[gn][:, amask], axis=1) for gn in group_names if gn in A_matrices]
                box_data = [d[np.isfinite(d)] for d in box_data]
                
                bp = ax_box.boxplot(box_data, labels=[l.split(': ')[-1] for l in group_names], patch_artist=True, medianprops=dict(color="black"))
                for i, box in enumerate(bp['boxes']):
                    box.set_facecolor(colors_dict[group_names[i]])
                    box.set_alpha(0.6)

                # Pairwise Annotations
                y_max, y_min = max([np.max(d) for d in box_data if len(d)>0]), min([np.min(d) for d in box_data if len(d)>0])
                y_range = y_max - y_min
                step = y_range * 0.15
                current_y = y_max + step

                for i, j in combinations(range(len(box_data)), 2):
                    d1, d2 = box_data[i], box_data[j]
                    if len(d1) < 2 or len(d2) < 2: continue
                    _, p_pair = ttest_ind(d1, d2, equal_var=False)
                    if p_pair < p_threshold:
                        stars = get_stars(p_pair)
                        p_str = f"p={p_pair:.4f}" if p_pair > 0.0001 else "p<0.0001"
                        ax_box.plot([i+1, i+1, j+1, j+1], [current_y, current_y+step*0.2, current_y+step*0.2, current_y], lw=1.2, c='k')
                        ax_box.text((i+j+2)/2, current_y+step*0.2, f"{stars}\n{p_str}\n(d={cohens_d(d1, d2):.1f})", ha='center', va='bottom', fontsize=7)
                        current_y += step * 2.5

                ax_box.set_ylim(bottom=y_min - step, top=current_y + step)
                ax_box.set_title(f"Window: {s:+.2f} to {e:+.2f}s", fontsize=9, pad=5)
                ax_box.set_ylabel(ylabel)
                plt.setp(ax_box.get_xticklabels(), rotation=30, ha="right", fontsize=8)
                ax_box.grid(False)

        plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.15)
        plt.show()


def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."
