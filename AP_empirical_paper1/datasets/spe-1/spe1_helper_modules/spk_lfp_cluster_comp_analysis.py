import os
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import warnings
from matplotlib.lines import Line2D

# ============================================================
# Publication style constants
# ============================================================
_FS_SM    = 11   # small annotations, tick labels
_FS_AX    = 13   # axis labels
_FS_SUB   = 14   # subplot titles
_FS_TTL   = 16   # figure suptitles

# Spike feature palette (consistent across all notebooks/modules)
_SPIKE_FEAT_COLORS = {
    'peak_amp':        '#8c564b',
    'peak_sharpness':  '#a06d62',
    'peak_width':      '#b38479',
    'exp_lambda':      '#c561a8',
    'inflection_time': '#9b59b6',
    'exp_const':       '#d7aee0',
    'log_isi':         '#7f7f7f',
    'spk_times_ms':    '#b0b0b0',
    'peak_amp_cluster':        '#8c564b',
    'peak_sharpness_cluster':  '#a06d62',
    'peak_width_cluster':      '#b38479',
    'exp_lambda_cluster':      '#c561a8',
    'inflection_time_cluster': '#9b59b6',
    'exp_const_cluster':       '#d7aee0',
    'log_isi_cluster':         '#7f7f7f',
    'spk_times_ms_cluster':    '#b0b0b0',
}

# Colorblind-friendly palette (Wong 2011)
_CB_PALETTE = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
               '#56B4E9', '#E69F00', '#F0E442', '#000000']

# Cluster group colors (low/mid/high)
_CLUST_LOW  = '#0072B2'
_CLUST_MID  = '#009E73'
_CLUST_HIGH = '#D55E00'

# Significance colors
_SIG_COL   = '#D55E00'
_INSIG_COL = '#56B4E9'
# ============================================================




# ==========================================
# 1. DATA COMPILER
# ==========================================

def _index_master_traces(master_traces):
    """Build a lookup for (cell_id, lfp_feature, spike_feature) trace records."""

    indexed = {}
    for trace in master_traces:
        key = (trace.get('cell_id'), trace.get('lfp_feature'), trace.get('spike_feature'))
        indexed.setdefault(key, trace)
    return indexed

def compile_lfp_stats(pickle_dir="/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/spe1_pickles/lfp_spk_group_pickles"):
    """
    Scans a directory for sliding window statistical results (pickle files) and 
    compiles them into a unified format for population-level analysis.

    Parameters
    ----------
    pickle_dir : str, optional
        The absolute path to the directory containing the saved '_sliding_stats.pkl' files.

    Returns
    -------
    df_stats : pandas.DataFrame
        A master DataFrame containing all pairwise statistical comparisons, 
        p-values, effect sizes (Cohen's d), and window timings across all cells.
    master_traces : list of dict
        A structured list containing the raw time-series trace arrays (mean, std, n) 
        for every cell and feature, used for plotting population grand averages.
    """
    all_files = sorted(glob.glob(os.path.join(pickle_dir, "*_sliding_stats.pkl")))
    
    master_stats_rows = []
    master_traces = []
    
    for file_path in all_files:
        with open(file_path, 'rb') as f:
            cell_data = pickle.load(f)
            
        cell_id = cell_data.get("cell_id", "Unknown")
        
        for feature_name, data in cell_data.items():
            if feature_name == "cell_id": 
                continue 
                
            # --- 1. EXTRACT STATS ---
            for pair in data.get("pairwise_stats", []):
                g1 = pair['group_1'].split(': ')[-1] if ':' in pair['group_1'] else pair['group_1']
                g2 = pair['group_2'].split(': ')[-1] if ':' in pair['group_2'] else pair['group_2']
                
                master_stats_rows.append({
                    "cell_id": cell_id,
                    "spike_feature": pair.get("spike_feature", "Unknown"), # Pulled from inside the stats
                    "lfp_feature": feature_name,
                    "window_start": pair["window_start"],
                    "window_end": pair["window_end"],
                    "group_1": g1,
                    "group_2": g2,
                    "comparison": f"{g1} vs {g2}",
                    "p_value": pair["p_value"],
                    "cohens_d": pair["cohens_d"]
                })
                
            # --- 2. EXTRACT TRACES ---
            for spike_feature, t_data in data.get("trace_data", {}).items():
                master_traces.append({
                    "cell_id": cell_id,
                    "spike_feature": spike_feature,
                    "lfp_feature": feature_name,
                    "trace_data": t_data
                })
                
    return pd.DataFrame(master_stats_rows), master_traces


# ==========================================
# 2. COMPARE EFFECT SIZES
# ==========================================



def plot_population_effect_sizes(df_stats, feature_shades):
    """
    Generates a statistical summary table and plots the population distribution 
    of effect sizes (Cohen's d) for each LFP feature.
    Includes vertical dividers to cleanly separate LFP features.
    """
    warnings.filterwarnings('ignore', category=FutureWarning)

    print("=== SIGNIFICANT HITS SUMMARY ===")
    summary_table = (
        df_stats.groupby(['lfp_feature', 'spike_feature'])
        .agg(
            sig_windows_count=('cohens_d', 'size'),      
            median_cohens_d=('cohens_d', 'median')       
        )
        .reset_index()
        .sort_values(by='median_cohens_d', ascending=False) 
        .reset_index(drop=True)
    )
    display(summary_table)

    plt.figure(figsize=(15, 7))

    # Force the exact order so the dodging aligns perfectly
    master_order = list(feature_shades.keys())
    lfp_order = sorted(df_stats['lfp_feature'].unique())

    # 1. Plot the boxes
    ax = sns.boxplot(
        data=df_stats, 
        x="lfp_feature", 
        y="cohens_d", 
        hue="spike_feature", 
        order=lfp_order,
        hue_order=master_order,
        palette=feature_shades, 
        width=0.75,
        fliersize=0,
        boxprops={'alpha': 0.4},
        zorder=1
    )

    # 2. Plot the raw data DOTS
    sns.stripplot(
        data=df_stats, 
        x="lfp_feature", 
        y="cohens_d", 
        hue="spike_feature",
        order=lfp_order,
        hue_order=master_order,
        palette=feature_shades,   
        dodge=True,             
        alpha=0.9, 
        jitter=0.2,             
        size=6,                   
        linewidth=1,              
        edgecolor='gray',
        legend=False,
        ax=ax,
        zorder=2                  
    )

    # 3. ADD VISUAL ANCHORS (Vertical Fences Only)
    # Add vertical dotted lines exactly halfway between each categorical tick
    for i in range(len(lfp_order) - 1):
        plt.axvline(i + 0.5, color='grey', linestyle=':', linewidth=1.5, alpha=0.6, zorder=0)

    plt.title("Population Effect Sizes", fontweight="bold", fontsize=16, pad=15)
    plt.ylabel("Effect Size (Cohen's d)", fontsize=13, fontweight="bold")
    plt.xlabel("LFP Feature", fontsize=13, fontweight="bold")
    plt.xticks(rotation=15, ha='right', fontsize=13)
    
    # Clean up the legend
    handles, labels = ax.get_legend_handles_labels()
    num_features = len(master_order)
    clean_labels = [l.replace('_cluster', '') for l in labels[:num_features]]
    plt.legend(handles[:num_features], clean_labels, title="Spike Feature",
               bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, shadow=True)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. ANALYZE TEMPORAL WINDOWS
# ==========================================

def plot_temporal_significance_density(df_stats, feature_shades, bin_size=0.01):
    """
    Creates a multi-panel grid of density plots showing the temporal distribution 
    of statistically significant windows across the cell population.

    For each LFP feature, the time axis is divided into discrete bins. A count is 
    added to a bin for every cell that shows a significant difference during that time.
    This reveals if the LFP divergences are tightly time-locked to the spike event (t=0).

    Parameters
    ----------
    df_stats : pandas.DataFrame
        The master statistics DataFrame generated by `compile_lfp_stats`.
    feature_shades : dict
        Dictionary mapping spike feature names to specific hex color codes.
    bin_size : float, optional
        The width of the time bins (in seconds) used to construct the histogram. 
        Default is 0.01s (10 ms).
    """
    unique_lfp_features = df_stats["lfp_feature"].unique()
    n_features = len(unique_lfp_features)

    if n_features == 0:
        print("No data found in df_stats!")
        return

    # Dynamically set up a grid (2 columns, as many rows as needed)
    cols = 2
    rows = math.ceil(n_features / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Define a master time grid spanning the entire dataset's window range
    t_min = df_stats["window_start"].min()
    t_max = df_stats["window_end"].max()
    time_bins = np.arange(t_min, t_max, bin_size) 
    
    for idx, target_lfp in enumerate(unique_lfp_features):
        ax = axes[idx]
        
        df_timing = df_stats[df_stats["lfp_feature"] == target_lfp].copy()
        unique_spike_features = df_timing["spike_feature"].unique()
        
        for spk_feat in unique_spike_features:
            df_sub = df_timing[df_timing["spike_feature"] == spk_feat]
            sig_counts = np.zeros_like(time_bins, dtype=float)
            
            # Count the overlapping significant windows per time bin
            if not df_sub.empty:
                diff = np.zeros(len(time_bins) + 1, dtype=float)
                starts = df_sub["window_start"].to_numpy(dtype=float)
                ends = df_sub["window_end"].to_numpy(dtype=float)
                start_idx = np.searchsorted(time_bins, starts, side='left')
                end_idx = np.searchsorted(time_bins, ends, side='right')
                np.add.at(diff, start_idx, 1)
                np.add.at(diff, end_idx, -1)
                sig_counts = np.cumsum(diff[:-1])
                
            # Look up the hex color directly from the provided dictionary
            color = feature_shades.get(spk_feat, '#cccccc')
            
            # Plot the density curve
            ax.plot(time_bins, sig_counts, color=color, lw=2.5)
            ax.fill_between(time_bins, 0, sig_counts, color=color, alpha=0.15)

        # Subplot Aesthetics
        ax.axvline(0, color="k", ls="--", lw=2)
        ax.set_title(f"{target_lfp}", fontweight="bold", fontsize=15)
        ax.set_ylabel("Count of Significant Windows")
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Keep legend in the first subplot, but force it to show ALL active features
        if idx == 0:
            all_active_spk_feats = df_stats["spike_feature"].unique()
            custom_lines = [Line2D([0], [0], color=feature_shades.get(feat, '#cccccc'), lw=2.5)
                            for feat in all_active_spk_feats]
            clean_feat_labels = [f.replace('_cluster', '') for f in all_active_spk_feats]
            ax.legend(custom_lines, clean_feat_labels, title="Clustered By", fontsize=_FS_SM, loc="upper right")

    # Clean up any empty subplots in the grid
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    # Main Titles and layout adjustments
    plt.suptitle("Temporal Distribution of Significance Across All Features", fontweight="bold", fontsize=18, y=1.02)
    fig.text(0.5, -0.01, 'Time relative to spike (s)', ha='center', fontsize=15)
    plt.tight_layout()
    plt.show()



# ==========================================
# 4. PLOT ALL GRAND AVERAGE TRACES (GRID)
# ==========================================

def plot_all_grand_average_traces(master_traces, df_stats):
    """
    Generates a master grid of population Grand Average traces for EVERY 
    LFP feature and spike feature combination.
    FILTERS out any spike features that did not yield significant results.
    """
    # 1. Identify which features actually survived
    valid_spike_features = set(df_stats["spike_feature"].unique())
    
    unique_lfp_features = sorted(list(set(t["lfp_feature"] for t in master_traces)))
    # 2. Only keep spike features that are in the valid list
    unique_spike_features = sorted(list(set(t["spike_feature"] for t in master_traces if t["spike_feature"] in valid_spike_features)))
    
    if not unique_lfp_features or not unique_spike_features:
        print("No valid traces found in the dataset to plot!")
        return
        
    master_Tgrid = None
    for t in master_traces:
        if "trace_data" in t and "Tgrid" in t["trace_data"]:
            master_Tgrid = t["trace_data"]["Tgrid"]
            break
            
    if master_Tgrid is None:
        print("Could not find 'Tgrid' in any trace data!")
        return

    for lfp_feat in unique_lfp_features:
        n_spk = len(unique_spike_features)
        cols = 3
        rows = math.ceil(n_spk / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        if n_spk == 1: axes = [axes]
        else: axes = axes.flatten()
        
        for idx, spk_feat in enumerate(unique_spike_features):
            ax = axes[idx]
            
            filtered_traces = [t for t in master_traces if t["lfp_feature"] == lfp_feat and t["spike_feature"] == spk_feat]
            
            cluster_names = set()
            for t in filtered_traces:
                if "groups" in t["trace_data"]:
                    cluster_names.update(t["trace_data"]["groups"].keys())
            
            cluster_names = sorted(list(cluster_names))
            stacked_data = {c: [] for c in cluster_names}
            
            for cell_record in filtered_traces:
                td = cell_record["trace_data"]
                if "groups" not in td or "Tgrid" not in td: continue
                
                cell_Tgrid = td["Tgrid"]
                    
                for c_name in cluster_names:
                    if c_name in td["groups"]:
                        val = td["groups"][c_name]
                        cell_mean = None
                        
                        if isinstance(val, dict):
                            cell_mean = val.get("mean", val.get("avg", None))
                        else:
                            val_arr = np.array(val)
                            if val_arr.ndim == 1: cell_mean = val_arr
                            elif val_arr.ndim == 2: cell_mean = np.nanmean(val_arr, axis=0)
                        
                        if cell_mean is not None:
                            aligned_mean = np.interp(master_Tgrid, cell_Tgrid, cell_mean)
                            stacked_data[c_name].append(aligned_mean)
            
            lines_plotted = 0
            for c_name in cluster_names:
                if len(stacked_data[c_name]) > 0:
                    matrix = np.vstack(stacked_data[c_name])
                    grand_mean = np.nanmean(matrix, axis=0)
                    grand_sem = np.nanstd(matrix, axis=0) / np.sqrt(matrix.shape[0])
                    
                    c_lower = c_name.lower()
                    if 'low' in c_lower: color = '#0072B2'
                    elif 'high' in c_lower: color = '#D55E00'
                    elif 'mid' in c_lower: color = '#009E73'
                    else: color = None
                    
                    label_clean = c_name.split(': ')[-1].capitalize() if ':' in c_name else c_name
                    ax.plot(master_Tgrid, grand_mean, label=f"{label_clean} (n={matrix.shape[0]})", color=color, lw=2.5)
                    ax.fill_between(master_Tgrid, grand_mean - grand_sem, grand_mean + grand_sem, color=color, alpha=0.2, lw=0)
                    lines_plotted += 1
            
            ax.axvline(0, color="k", ls="--", lw=1.5)
            ax.set_title(spk_feat, fontweight="bold", fontsize=13)
            
            if lines_plotted > 0:
                ax.legend(fontsize=11, loc="upper right")
            else:
                ax.text(0.5, 0.5, "No valid arrays found", ha='center', va='center', transform=ax.transAxes, color='gray')

        for i in range(n_spk, len(axes)):
            fig.delaxes(axes[i])
            
        plt.suptitle(f"Population Grand Average: {lfp_feat}", fontweight="bold", fontsize=18, y=1.02)
        fig.text(0.5, -0.01, 'Time relative to spike (s)', ha='center', fontsize=15)
        fig.text(-0.01, 0.5, f'Δ {lfp_feat}', va='center', rotation='vertical', fontsize=15)
        
        plt.tight_layout()
        plt.show()


# ==========================================
# 5. CLUSTER RELATIONSHIP HEATMAP
# ==========================================

def plot_cluster_relationship_heatmap(df_stats, master_traces):
    """
    Maps out the relationship between spike clusters and LFP features.
    Features an upgraded, publication-ready aesthetic palette.
    """
    if df_stats.empty:
        print("No data found in df_stats!")
        return

    print("Extracting relationship signs from raw traces...")
    df_signed = df_stats.copy()
    signs = []
    trace_lookup = _index_master_traces(master_traces)
    
    # 1. Recover the sign for every single significant window
    for row in df_signed.itertuples(index=False):
        cell_id = row.cell_id
        lfp_feat = row.lfp_feature
        spk_feat = row.spike_feature
        w_start = row.window_start
        w_end = row.window_end
        
        trace_record = trace_lookup.get((cell_id, lfp_feat, spk_feat))
        
        sign = 1 # Default to Positive (High cluster > Low cluster)
        
        if trace_record and 'groups' in trace_record['trace_data']:
            groups = trace_record['trace_data']['groups']
            tgrid = trace_record['trace_data'].get('Tgrid', [])
            
            high_key = next((k for k in groups.keys() if 'high' in k.lower()), None)
            low_key = next((k for k in groups.keys() if 'low' in k.lower()), None)
            
            if high_key and low_key and len(tgrid) > 0:
                high_val = groups[high_key]
                low_val = groups[low_key]
                
                high_arr = high_val.get('mean', high_val.get('avg')) if isinstance(high_val, dict) else np.array(high_val)
                low_arr = low_val.get('mean', low_val.get('avg')) if isinstance(low_val, dict) else np.array(low_val)
                
                if high_arr is not None and low_arr is not None:
                    if high_arr.ndim == 2: high_arr = np.nanmean(high_arr, axis=0)
                    if low_arr.ndim == 2: low_arr = np.nanmean(low_arr, axis=0)
                    
                    mask = (tgrid >= w_start) & (tgrid <= w_end)
                    if np.any(mask) and len(high_arr) == len(tgrid) and len(low_arr) == len(tgrid):
                        high_mean = np.nanmean(high_arr[mask])
                        low_mean = np.nanmean(low_arr[mask])
                        
                        if low_mean > high_mean:
                            sign = -1
                            
        signs.append(sign)
        
    df_signed['signed_cohens_d'] = df_signed['cohens_d'] * signs

    # 2. Pivot the data
    heatmap_data = df_signed.groupby(['lfp_feature', 'spike_feature'])['signed_cohens_d'].median().reset_index()
    pivot_table = heatmap_data.pivot(index='lfp_feature', columns='spike_feature', values='signed_cohens_d')
    pivot_table.columns = [c.replace('_cluster', '') for c in pivot_table.columns]

    # 3. Create clean text labels (+ and -)
    annot_text = []
    for row in pivot_table.values:
        text_row = []
        for val in row:
            if np.isnan(val):
                text_row.append("")
            elif val > 0:
                text_row.append(f"+{val:.2f}")  
            else:
                text_row.append(f"{val:.2f}")   
        annot_text.append(text_row)

    # 4. Plotting (AESTHETIC UPGRADES HERE)
    # Tighter figure size so boxes aren't too stretched
    plt.figure(figsize=(11, 7))

    max_val = np.nanmax(np.abs(pivot_table.values))
    if np.isnan(max_val) or max_val == 0: max_val = 1.0 

    sns.heatmap(
        pivot_table,
        annot=np.array(annot_text),  
        fmt="",                      
        cmap="RdBu_r",               # UPGRADE: Rich Red-Blue diverging palette
        center=0,            
        vmin=-max_val,       
        vmax=max_val,
        linewidths=1.0,              # Thicker, crisper grid lines
        linecolor='white',
        annot_kws={"size": 12, "weight": "bold"}, # Make the numbers bolder and larger
        cbar_kws={
            'label': "Median Effect Size (Signed Cohen's d)", 
            'shrink': 0.8 # Shrinks the colorbar slightly so it aligns nicely with the plot
        }
    )

    # Aesthetics
    plt.title("Spike-LFP Relationship Matrix", fontweight="bold", fontsize=16, pad=20)
    plt.xlabel("Spike Feature (Clustering Metric)", fontsize=14, fontweight="bold")
    plt.ylabel("LFP Feature", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha='right', fontsize=13)
    plt.yticks(fontsize=13)
    
    # Text legend without emojis!
    legend_text = (
        "Positive (+ / Red): High cluster is associated with HIGHER LFP values.\n"
        "Negative (- / Blue): High cluster is associated with LOWER LFP values."
    )
    plt.figtext(0.5, -0.05, legend_text, ha="center", fontsize=13, 
                bbox={"facecolor":"#f8f9fa", "edgecolor":"#dee2e6", "pad":8, "boxstyle":"round,pad=0.5"})
    
    plt.tight_layout()
    plt.show()



# ==========================================
# 6. RESPONDER YIELD (PERCENTAGE) HEATMAP
# ==========================================

def plot_significant_yield_heatmap(df_stats, master_traces):
    """
    Calculates and plots the percentage of 'valid' cells (cells that actually 
    formed High/Low clusters) that exhibited at least one significant LFP 
    difference for each feature combination.
    FILTERS out features that yielded 0 significant windows across the board.
    """
    print("Calculating yields across all cells...")
    yield_data = []
    
    # 1. Get unique features, but ONLY pull spike features that exist in df_stats
    lfp_feats = sorted(list(set(t["lfp_feature"] for t in master_traces)))
    spk_feats = sorted(list(df_stats["spike_feature"].unique()))  # <-- The strict filter
    
    if not spk_feats:
        print("No significant spike features available to plot.")
        return
    
    for lfp in lfp_feats:
        for spk in spk_feats:
            combo_traces = [t for t in master_traces if t["lfp_feature"] == lfp and t["spike_feature"] == spk]
            
            valid_cells = 0
            for t in combo_traces:
                if "groups" in t.get("trace_data", {}):
                    groups = t["trace_data"]["groups"]
                    has_high = any('high' in k.lower() for k in groups.keys())
                    has_low = any('low' in k.lower() for k in groups.keys())
                    if has_high and has_low:
                        valid_cells += 1
                        
            if valid_cells > 0:
                sig_cells = df_stats[
                    (df_stats["lfp_feature"] == lfp) & 
                    (df_stats["spike_feature"] == spk)
                ]["cell_id"].nunique()
                
                yield_pct = (sig_cells / valid_cells) * 100
            else:
                yield_pct = np.nan 
                
            yield_data.append({
                "lfp_feature": lfp, 
                "spike_feature": spk, 
                "yield_pct": yield_pct,
                "text_label": f"{yield_pct:.0f}%\n({sig_cells}/{valid_cells})" if valid_cells > 0 else ""
            })
            
    df_yield = pd.DataFrame(yield_data)

    pivot_color = df_yield.pivot(index='lfp_feature', columns='spike_feature', values='yield_pct')
    pivot_text  = df_yield.pivot(index='lfp_feature', columns='spike_feature', values='text_label')
    pivot_color.columns = [c.replace('_cluster', '') for c in pivot_color.columns]
    pivot_text.columns  = [c.replace('_cluster', '') for c in pivot_text.columns]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        pivot_color,
        annot=pivot_text,    
        fmt="",              
        cmap="YlGnBu",       
        vmin=0,
        vmax=100,
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': '% of Cells with Significant Effect'}
    )
    
    plt.title("How Consistent is the Effect Across the Population?", fontweight="bold", fontsize=16, pad=15)
    plt.xlabel("Spike Feature (Clustering Metric)", fontsize=13, fontweight="bold")
    plt.ylabel("LFP Feature", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()


# ==========================================
# 7. RELATIONSHIP REDUNDANCY (EFFECT SIZE CORRELATION)
# ==========================================

def plot_relationship_redundancy(df_stats, master_traces):
    """
    Correlates the Effect Size (Signed Cohen's d) profiles to see if 
    different spike features have redundant relationships with the LFP, 
    and vice versa.
    """
    if df_stats.empty:
        print("No data found to correlate!")
        return

    print("Extracting relationship profiles...")
    df_signed = df_stats.copy()
    signs = []
    
    # 1. Recover the sign (+ or -) for every significant window
    for _, row in df_signed.iterrows():
        cell_id = row['cell_id']
        lfp_feat = row['lfp_feature']
        spk_feat = row['spike_feature']
        w_start = row['window_start']
        w_end = row['window_end']
        
        trace_record = next((t for t in master_traces 
                             if t['cell_id'] == cell_id and 
                             t['lfp_feature'] == lfp_feat and 
                             t['spike_feature'] == spk_feat), None)
        
        sign = 1 # Default to Positive
        
        if trace_record and 'groups' in trace_record['trace_data']:
            groups = trace_record['trace_data']['groups']
            tgrid = trace_record['trace_data'].get('Tgrid', [])
            
            high_key = next((k for k in groups.keys() if 'high' in k.lower()), None)
            low_key = next((k for k in groups.keys() if 'low' in k.lower()), None)
            
            if high_key and low_key and len(tgrid) > 0:
                high_val = groups[high_key]
                low_val = groups[low_key]
                
                high_arr = high_val.get('mean', high_val.get('avg')) if isinstance(high_val, dict) else np.array(high_val)
                low_arr = low_val.get('mean', low_val.get('avg')) if isinstance(low_val, dict) else np.array(low_val)
                
                if high_arr is not None and low_arr is not None:
                    if high_arr.ndim == 2: high_arr = np.nanmean(high_arr, axis=0)
                    if low_arr.ndim == 2: low_arr = np.nanmean(low_arr, axis=0)
                    
                    mask = (tgrid >= w_start) & (tgrid <= w_end)
                    if np.any(mask) and len(high_arr) == len(tgrid) and len(low_arr) == len(tgrid):
                        if np.nanmean(low_arr[mask]) > np.nanmean(high_arr[mask]):
                            sign = -1
                            
        signs.append(sign)
        
    df_signed['signed_cohens_d'] = df_signed['cohens_d'] * signs

    # 2. Pivot into a Matrix: Rows = LFP Features, Columns = Spike Features
    heatmap_data = df_signed.groupby(['lfp_feature', 'spike_feature'])['signed_cohens_d'].median().reset_index()
    pivot_table = heatmap_data.pivot(index='lfp_feature', columns='spike_feature', values='signed_cohens_d')
    pivot_table.columns = [c.replace('_cluster', '') for c in pivot_table.columns]

    # Fill missing values with 0 (No effect) so the correlation math works perfectly
    pivot_table = pivot_table.fillna(0)

    # 3. Correlate Spike Features (Columns) and LFP Features (Rows)
    spike_corr = pivot_table.corr(method='spearman')
    lfp_corr = pivot_table.T.corr(method='spearman')

    # 4. Plot them side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot A: Spike Feature Redundancy
    sns.heatmap(
        spike_corr, annot=True, fmt=".2f", cmap="mako", 
        vmin=-1, vmax=1, linewidths=1, linecolor='white', ax=axes[0],
        cbar_kws={'shrink': 0.8}
    )
    axes[0].set_title("Spike Feature Redundancy\n(Do any spike features have the same relationship with the LFP features?)", fontweight='bold', pad=15)
    axes[0].set_xlabel("Spike Feature", fontweight='bold')
    axes[0].set_ylabel("Spike Feature", fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)

    # Plot B: LFP Feature Redundancy
    sns.heatmap(
        lfp_corr, annot=True, fmt=".2f", cmap="rocket", 
        vmin=-1, vmax=1, linewidths=1, linecolor='white', ax=axes[1],
        cbar_kws={'shrink': 0.8}
    )
    axes[1].set_title("LFP Feature Redundancy\n(Do any LFP features have the same relationship with the spike features?)", fontweight='bold', pad=15)
    axes[1].set_xlabel("LFP Feature", fontweight='bold')
    axes[1].set_ylabel("LFP Feature", fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()
    return spike_corr, lfp_corr


def get_nonredundant_lfp_features(df_stats, lfp_corr, threshold=0.9):
    """
    Uses the LFP redundancy correlation matrix from plot_relationship_redundancy
    to identify and drop redundant LFP features.

    Greedy strategy: rank LFP features by mean Cohen's d (higher = more informative).
    For each redundant pair (|r| >= threshold), drop the lower-ranked feature.

    Returns the list of LFP features to keep, and prints what was dropped and why.
    """
    lfp_feats = list(lfp_corr.columns)
    mean_d = df_stats.groupby('lfp_feature')['cohens_d'].mean()
    # Sort by descending effect size — keep the most informative first
    feats_ranked = (mean_d.reindex(lfp_feats)
                    .sort_values(ascending=False)
                    .index.tolist())

    keep, dropped = [], {}
    for feat in feats_ranked:
        redundant_with = None
        for kept in keep:
            if feat in lfp_corr.index and kept in lfp_corr.columns:
                if abs(lfp_corr.loc[feat, kept]) >= threshold:
                    redundant_with = kept
                    break
        if redundant_with is None:
            keep.append(feat)
        else:
            dropped[feat] = (redundant_with, abs(lfp_corr.loc[feat, redundant_with]))

    print(f"\nLFP Feature Redundancy Filter  (threshold |r| >= {threshold})")
    print(f"  Keeping  ({len(keep)}): {keep}")
    if dropped:
        for feat, (reason, r) in dropped.items():
            print(f"  Dropping '{feat}'  —  |r| = {r:.2f} with '{reason}'")
    else:
        print("  No features dropped at this threshold.")
    return keep


# ==========================================
# 8. TEMPORAL LANDSCAPE (TIMING)
# ==========================================

def plot_relationship_timing(df_stats, feature_shades):
    """
    Plots the exact timing of significant windows relative to the spike.
    Only reserves colors/legend space for features that actually exist in the data.
    """
    if df_stats.empty: 
        print("No stats data available to plot timing.")
        return
    
    df_plot = df_stats.copy()
    df_plot['midpoint'] = (df_plot['window_start'] + df_plot['window_end']) / 2
    
    plt.figure(figsize=(12, 7))
    
    # FILTER: Only keep the colors for features that actually have significant data
    valid_features = set(df_plot['spike_feature'].unique())
    master_order = [f for f in feature_shades.keys() if f in valid_features]
    
    ax = sns.stripplot(
        data=df_plot, 
        x='midpoint', 
        y='lfp_feature', 
        hue='spike_feature', 
        hue_order=master_order,
        palette=feature_shades,
        dodge=True, 
        alpha=0.7, 
        size=7,
        jitter=0.25
    )
    
    plt.axvline(0, color='black', ls='--', lw=2, label='Spike (t=0)')
    
    plt.title("Temporal Landscape: When do these LFP relationships occur?", fontweight='bold', fontsize=16, pad=15)
    plt.xlabel("Time relative to spike (seconds)", fontsize=13, fontweight='bold')
    plt.ylabel("LFP Feature", fontsize=13, fontweight='bold')
    
    handles, labels = ax.get_legend_handles_labels()
    num_features = len(master_order)
    handles = handles[:num_features] + [Line2D([0], [0], color='black', ls='--', lw=2)]
    labels = labels[:num_features] + ['Spike (t=0)']
    
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', title="Spike Feature", frameon=True, shadow=True)
    plt.grid(axis='x', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 9. BOOTSTRAP POPULATION STATS
# ==========================================

def bootstrap_population_stats(df_stats, master_traces=None, n_bootstrap=1000, alpha=0.05):
    from matplotlib.patches import Patch

    cells = df_stats['cell_id'].unique()
    n_cells = len(cells)
    combos = list(df_stats.groupby(['lfp_feature', 'spike_feature']).groups.keys())

    # Per-combo valid cell counts: cells that were actually tested (have trace data),
    # not just those that had significant windows. Fixes yield denominator when not
    # every cell was tested for every spike feature.
    if master_traces is not None:
        valid_cells_map = {}
        for t in master_traces:
            key = (t['lfp_feature'], t['spike_feature'])
            valid_cells_map.setdefault(key, set()).add(t['cell_id'])
    else:
        valid_cells_map = {k: set(cells) for k in combos}

    real_med_d, real_med_p, real_yield = {}, {}, {}
    for (lfp, spk), grp in df_stats.groupby(['lfp_feature', 'spike_feature']):
        n_valid = max(len(valid_cells_map.get((lfp, spk), cells)), 1)
        real_med_d[(lfp, spk)] = grp['cohens_d'].abs().median()
        real_med_p[(lfp, spk)] = grp['p_value'].median()
        real_yield[(lfp, spk)] = grp['cell_id'].nunique() / n_valid * 100

    rng = np.random.default_rng(42)
    boot_d, boot_p, boot_yield = {k: [] for k in combos}, {k: [] for k in combos}, {k: [] for k in combos}

    for _ in range(n_bootstrap):
        sampled_cells = rng.choice(cells, size=n_cells, replace=True)
        boot_df = pd.concat([df_stats[df_stats['cell_id'] == c] for c in sampled_cells], ignore_index=True)
        for (lfp, spk), grp in boot_df.groupby(['lfp_feature', 'spike_feature']):
            if (lfp, spk) in boot_d:
                n_valid = max(len(valid_cells_map.get((lfp, spk), cells)), 1)
                boot_d[(lfp, spk)].append(grp['cohens_d'].abs().median())
                boot_p[(lfp, spk)].append(grp['p_value'].median())
                boot_yield[(lfp, spk)].append(grp['cell_id'].nunique() / n_valid * 100)

    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100
    rows = []
    for (lfp, spk) in combos:
        bd, bp, by = np.array(boot_d[(lfp, spk)]), np.array(boot_p[(lfp, spk)]), np.array(boot_yield[(lfp, spk)])
        d_ci_lo = np.percentile(bd, lo) if len(bd) else np.nan
        d_ci_hi = np.percentile(bd, hi) if len(bd) else np.nan
        p_ci_lo = np.percentile(bp, lo) if len(bp) else np.nan
        p_ci_hi = np.percentile(bp, hi) if len(bp) else np.nan
        y_ci_lo = np.percentile(by, lo) if len(by) else np.nan
        y_ci_hi = np.percentile(by, hi) if len(by) else np.nan
        rows.append({
            'lfp_feature': lfp, 'spike_feature': spk,
            'real_median_d': real_med_d.get((lfp, spk), np.nan),
            'median_d_ci_lo': d_ci_lo, 'median_d_ci_hi': d_ci_hi,
            'd_robust': bool(d_ci_lo > 0),
            'real_median_p': real_med_p.get((lfp, spk), np.nan),
            'median_p_ci_lo': p_ci_lo, 'median_p_ci_hi': p_ci_hi,
            'p_robust': bool(p_ci_hi < alpha),
            'real_yield_pct': real_yield.get((lfp, spk), np.nan),
            'yield_ci_lo': y_ci_lo, 'yield_ci_hi': y_ci_hi,
            'yield_robust': bool(y_ci_lo > 0),
        })

    df_results = pd.DataFrame(rows).sort_values('real_median_d', ascending=False).reset_index(drop=True)
    df_results['all_robust'] = df_results['d_robust'] & df_results['p_robust'] & df_results['yield_robust']

    lfp_feats = sorted(df_results['lfp_feature'].unique())
    n_lfp = len(lfp_feats)
    grid_cols = 2
    grid_rows = math.ceil(n_lfp / grid_cols)
    robust_colors = {True: '#D55E00', False: '#0072B2'}
    legend_els_d = [Patch(facecolor='#D55E00', alpha=0.75, label='CI excludes 0 (robust)'),
                    Patch(facecolor='#0072B2', alpha=0.75, label='CI includes 0 (not robust)')]
    legend_els_p = [Patch(facecolor='#D55E00', alpha=0.75, label=f'CI upper bound < {alpha} (robust)'),
                    Patch(facecolor='#0072B2', alpha=0.75, label=f'CI upper bound >= {alpha} (not robust)')]

    # Figure 1: Cohen's d
    fig1, axes1 = plt.subplots(grid_rows, grid_cols, figsize=(14, 4 * grid_rows))
    axes1 = axes1.flatten()
    for idx, lfp in enumerate(lfp_feats):
        ax = axes1[idx]
        sub = df_results[df_results['lfp_feature'] == lfp].sort_values('real_median_d', ascending=True).reset_index(drop=True)
        y = range(len(sub))
        ax.barh(list(y), sub['real_median_d'], color=[robust_colors[r] for r in sub['d_robust']], alpha=0.75)
        ax.errorbar(sub['real_median_d'], list(y),
                    xerr=[sub['real_median_d'] - sub['median_d_ci_lo'], sub['median_d_ci_hi'] - sub['real_median_d']],
                    fmt='none', color='black', capsize=4, lw=1.5)
        ax.set_yticks(list(y))
        ax.set_yticklabels([s.replace('_cluster', '') for s in sub['spike_feature']], fontsize=11)
        ax.set_xlabel("Median |Cohen's d|", fontsize=13)
        ax.set_title(f'{lfp}\n({int((1-alpha)*100)}% bootstrap CI)', fontsize=14, fontweight='bold')
        ax.axvline(0, color='gray', lw=1, linestyle='--')
        sns.despine(ax=ax)
    for j in range(n_lfp, len(axes1)): fig1.delaxes(axes1[j])
    fig1.legend(handles=legend_els_d, loc='lower center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(f"Bootstrap: Effect Size (Cohen's d)\n(n={n_bootstrap} resamples, {n_cells} cells)", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Figure 2: Median p-value
    fig2, axes2 = plt.subplots(grid_rows, grid_cols, figsize=(14, 4 * grid_rows))
    axes2 = axes2.flatten()
    for idx, lfp in enumerate(lfp_feats):
        ax = axes2[idx]
        sub = df_results[df_results['lfp_feature'] == lfp].sort_values('real_median_p', ascending=True).reset_index(drop=True)
        y = range(len(sub))
        ax.barh(list(y), sub['real_median_p'], color=[robust_colors[r] for r in sub['p_robust']], alpha=0.75)
        ax.errorbar(sub['real_median_p'], list(y),
                    xerr=[sub['real_median_p'] - sub['median_p_ci_lo'], sub['median_p_ci_hi'] - sub['real_median_p']],
                    fmt='none', color='black', capsize=4, lw=1.5)
        ax.set_yticks(list(y))
        ax.set_yticklabels([s.replace('_cluster', '') for s in sub['spike_feature']], fontsize=11)
        ax.set_xlabel('Median p-value', fontsize=13)
        ax.set_title(f'{lfp}\n({int((1-alpha)*100)}% bootstrap CI)', fontsize=14, fontweight='bold')
        ax.axvline(alpha, color='red', lw=1, linestyle='--', alpha=0.6, label=f'α={alpha}')
        ax.legend(fontsize=11)
        sns.despine(ax=ax)
    for j in range(n_lfp, len(axes2)): fig2.delaxes(axes2[j])
    fig2.legend(handles=legend_els_p, loc='lower center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, -0.02))
    plt.suptitle(f'Bootstrap: Median p-value\n(n={n_bootstrap} resamples, {n_cells} cells)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Figure 3: Summary heatmap
    metrics = ['d_robust', 'p_robust', 'yield_robust', 'all_robust']
    labels  = ["Cohen's d CI\nexcludes 0", f"Median p CI\n< {alpha}", "Yield CI\nexcludes 0", "All three\nrobust"]
    fig3, axes3 = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), max(4, len(df_results['spike_feature'].unique()) * 0.5 + 2)))
    for ax, metric, label in zip(axes3, metrics, labels):
        pivot = df_results.pivot(index='spike_feature', columns='lfp_feature', values=metric).astype(float)
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', vmin=0, vmax=1,
                    linewidths=0.5, ax=ax, cbar=False)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.set_xlabel(''); ax.set_ylabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha='right', fontsize=11)
        ax.set_yticklabels([s.get_text().replace('_cluster', '') for s in ax.get_yticklabels()], fontsize=11)
    plt.suptitle('Bootstrap Robustness Summary\n(1 = robust, 0 = not robust)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    n_d = df_results['d_robust'].sum(); n_p = df_results['p_robust'].sum()
    n_y = df_results['yield_robust'].sum(); n_all = df_results['all_robust'].sum()
    print(f"\nBootstrap summary ({int((1-alpha)*100)}% CI, n={n_bootstrap}):")
    print(f"  Effect size robust:  {n_d} / {len(df_results)}")
    print(f"  p-value robust:      {n_p} / {len(df_results)}")
    print(f"  Yield robust:        {n_y} / {len(df_results)}")
    print(f"  All three robust:    {n_all} / {len(df_results)}")
    print()
    print(df_results[['lfp_feature', 'spike_feature', 'real_median_d', 'd_robust',
                       'real_median_p', 'p_robust', 'real_yield_pct', 'yield_robust', 'all_robust']].to_string(index=False))
    return df_results


# ==========================================
# 10. PERMUTATION YIELD TEST
# ==========================================

def binomial_yield_test(df_stats, master_traces=None, alpha=0.05):
    """
    Tests whether the observed cell yield for each (LFP × spike feature) combo
    exceeds what would be expected by chance using a one-sided binomial test.

    H0: each cell independently has probability alpha of being significant by chance.
    Uses per-combo valid cell counts (from master_traces) as the denominator so that
    spike features not present in all cells are handled correctly.
    """
    from scipy.stats import binom
    from matplotlib.patches import Patch

    all_cells = df_stats['cell_id'].unique()

    if master_traces is not None:
        valid_cells_map = {}
        for t in master_traces:
            key = (t['lfp_feature'], t['spike_feature'])
            valid_cells_map.setdefault(key, set()).add(t['cell_id'])
    else:
        valid_cells_map = None

    rows = []
    for (lfp, spk), grp in df_stats.groupby(['lfp_feature', 'spike_feature']):
        k = grp['cell_id'].nunique()
        if valid_cells_map and (lfp, spk) in valid_cells_map:
            n = len(valid_cells_map[(lfp, spk)])
        else:
            n = len(all_cells)
        # P(X >= k) under Binomial(n, alpha)
        binom_p = float(binom.sf(k - 1, n, alpha))
        rows.append({
            'lfp_feature': lfp,
            'spike_feature': spk,
            'n_sig_cells': k,
            'n_valid_cells': n,
            'yield_pct': round(100 * k / max(n, 1), 1),
            'expected_pct': round(alpha * 100, 1),
            'binom_p': round(binom_p, 4),
            'sig': binom_p < alpha,
        })

    df_binom = pd.DataFrame(rows).sort_values('binom_p').reset_index(drop=True)

    sig_df   = df_binom[df_binom['sig']].sort_values('yield_pct', ascending=True)
    insig_df = df_binom[~df_binom['sig']].sort_values('yield_pct', ascending=True)
    plot_df  = pd.concat([insig_df, sig_df]).reset_index(drop=True)
    plot_df['label'] = (plot_df['lfp_feature'] + '\n' +
                        plot_df['spike_feature'].str.replace('_cluster', '', regex=False))

    fig, ax = plt.subplots(figsize=(8, max(5, len(plot_df) * 0.45 + 1)))
    colors = [_SIG_COL if s else _INSIG_COL for s in plot_df['sig']]
    ax.barh(plot_df['label'], plot_df['yield_pct'], color=colors, alpha=0.8)
    ax.axvline(alpha * 100, color='black', lw=1.5, linestyle='--',
               label=f'Expected by chance ({alpha*100:.0f}%)')

    for i, (_, row) in enumerate(plot_df.iterrows()):
        p_str = f"p={row['binom_p']:.4f}" if row['binom_p'] >= 0.0001 else "p<0.0001"
        ax.text(row['yield_pct'] + 1, i, p_str, va='center', fontsize=_FS_SM,
                color='black' if row['sig'] else 'gray')

    ax.set_xlabel('Cell Yield (%)', fontsize=_FS_AX)
    ax.set_title(
        f'Binomial Yield Test\n(H\u2080: yield \u2264 {alpha*100:.0f}% by chance)',
        fontsize=_FS_TTL, fontweight='bold'
    )
    legend_els = [
        Patch(facecolor=_SIG_COL,   alpha=0.8, label=f'Sig (p < {alpha})'),
        Patch(facecolor=_INSIG_COL, alpha=0.8, label='Not sig'),
    ]
    ax.legend(handles=legend_els, fontsize=_FS_SM, loc='lower right')
    ax.set_xlim(0, 110)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()

    n_sig = df_binom['sig'].sum()
    print(f"\nBinomial yield test summary (alpha={alpha}):")
    print(f"  Significant combos: {n_sig} / {len(df_binom)}")
    print()
    print(df_binom.to_string(index=False))
    return df_binom


# ==========================================
# 11. LFP FEATURE CO-OCCURRENCE
# ==========================================

def plot_lfp_feature_cooccurrence(df_stats):
    df_binary = df_stats.groupby(['cell_id', 'spike_feature', 'lfp_feature']).size().reset_index(name='n_sig_windows')
    df_binary['present'] = 1
    pivot = df_binary.pivot_table(index=['cell_id', 'spike_feature'], columns='lfp_feature', values='present', fill_value=0)
    lfp_feats = list(pivot.columns)
    n = len(lfp_feats)
    cooccur = np.zeros((n, n))
    for i, f1 in enumerate(lfp_feats):
        for j, f2 in enumerate(lfp_feats):
            cooccur[i, j] = ((pivot[f1] == 1) & (pivot[f2] == 1)).sum()
    cooccur_df = pd.DataFrame(cooccur, index=lfp_feats, columns=lfp_feats)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(cooccur_df, annot=True, fmt='.0f', cmap='YlOrRd', linewidths=0.5, ax=ax1,
                cbar_kws={'label': 'n cell x spike feature combos'})
    ax1.set_title('LFP Feature Co-occurrence\n(# combos where both features are significant)', fontweight='bold', fontsize=14)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=40, ha='right', fontsize=11)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=11)
    n_cells_total = df_stats['cell_id'].nunique()
    marginal = df_stats.groupby('lfp_feature')['cell_id'].nunique().sort_values(ascending=True)
    ax2.barh(marginal.index, marginal.values, color='#0072B2', alpha=0.8)
    ax2.set_xlabel('# Cells with >= 1 Significant Window', fontsize=13)
    ax2.set_title('LFP Feature Prevalence\n(across cells)', fontweight='bold', fontsize=14)
    ax2.axvline(n_cells_total, color='gray', lw=1, linestyle='--', label=f'total cells = {n_cells_total}')
    ax2.legend(fontsize=11)
    sns.despine(ax=ax2)
    plt.tight_layout()
    plt.show()
    return cooccur_df


# ==========================================
# 12. DIRECTION CONSISTENCY
# ==========================================

def plot_direction_consistency(df_stats, master_traces):
    """
    Checks how consistently the direction of the LFP effect aligns with spike
    cluster rank (Low < High) across cells.

    Works for both 2-cluster (Low/High) and 3-cluster (Low/Mid/High) cells:
    - 2-cluster: direction is 'up' (High > Low) or 'down' (High < Low)
    - 3-cluster: additionally checks monotonicity; 'non-monotonic' if Mid breaks rank
    """
    df3 = df_stats.copy()
    df3['comp_clean'] = df3['comparison'].str.lower().str.replace(' ', '')
    results = []

    for (cell, lfp, spk), grp in df3.groupby(['cell_id', 'lfp_feature', 'spike_feature']):
        comps_present = set(grp['comp_clean'].values)
        has_low_high  = 'lowvshigh' in comps_present
        has_3cluster  = {'lowvsmid', 'midvshigh'}.issubset(comps_present)
        if not has_low_high:
            continue

        trace_record = next((t for t in master_traces if t['cell_id'] == cell and
                             t['lfp_feature'] == lfp and t['spike_feature'] == spk), None)
        if not trace_record:
            continue
        groups = trace_record['trace_data'].get('groups', {})
        tgrid  = np.array(trace_record['trace_data'].get('Tgrid', []))
        if len(tgrid) == 0:
            continue

        def get_arr(key):
            val = groups.get(key)
            if val is None: return None
            arr = val.get('mean', val.get('avg')) if isinstance(val, dict) else np.array(val)
            return np.array(arr) if arr is not None else None

        low_key  = next((k for k in groups if 'low'  in k.lower()), None)
        high_key = next((k for k in groups if 'high' in k.lower()), None)
        if not (low_key and high_key): continue

        low_arr  = get_arr(low_key)
        high_arr = get_arr(high_key)
        if low_arr is None or high_arr is None: continue

        min_len  = min(len(low_arr), len(high_arr))
        low_mean  = np.nanmean(low_arr[:min_len])
        high_mean = np.nanmean(high_arr[:min_len])

        row = {'cell_id': cell, 'lfp_feature': lfp, 'spike_feature': spk,
               'n_clusters': 2, 'low_mean': round(low_mean, 4),
               'mid_mean': np.nan, 'high_mean': round(high_mean, 4)}

        if has_3cluster:
            mid_key = next((k for k in groups if 'mid' in k.lower()), None)
            if mid_key:
                mid_arr = get_arr(mid_key)
                if mid_arr is not None:
                    min_len3 = min(len(low_arr), len(mid_arr), len(high_arr))
                    mid_mean = np.nanmean(mid_arr[:min_len3])
                    low_mean3  = np.nanmean(low_arr[:min_len3])
                    high_mean3 = np.nanmean(high_arr[:min_len3])
                    mono_up   = low_mean3 <= mid_mean <= high_mean3
                    mono_down = low_mean3 >= mid_mean >= high_mean3
                    row['n_clusters'] = 3
                    row['mid_mean']   = round(mid_mean, 4)
                    row['low_mean']   = round(low_mean3, 4)
                    row['high_mean']  = round(high_mean3, 4)
                    if mono_up:   row['direction'] = 'up'
                    elif mono_down: row['direction'] = 'down'
                    else:           row['direction'] = 'non-monotonic'
                    results.append(row)
                    continue

        # 2-cluster direction
        row['direction'] = 'up' if high_mean > low_mean else 'down'
        results.append(row)

    if not results:
        print("No cells with Low/High comparison found.")
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    df_res['spike_feature_clean'] = df_res['spike_feature'].str.replace('_cluster', '', regex=False)

    summary = df_res.groupby(['lfp_feature', 'spike_feature_clean', 'direction']).size().reset_index(name='count')
    total   = df_res.groupby(['lfp_feature', 'spike_feature_clean']).size().reset_index(name='total')
    summary = summary.merge(total, on=['lfp_feature', 'spike_feature_clean'])
    summary['pct'] = (summary['count'] / summary['total'] * 100).round(1)

    dir_colors = {'up': '#009E73', 'down': '#0072B2', 'non-monotonic': '#CC79A7'}
    summary_pivot = summary.pivot_table(
        index=['lfp_feature', 'spike_feature_clean'], columns='direction',
        values='pct', fill_value=0)

    fig, ax = plt.subplots(figsize=(12, max(5, len(summary_pivot) * 0.55 + 1)))
    summary_pivot.plot(kind='barh', stacked=True,
                       color=[dir_colors.get(c, 'gray') for c in summary_pivot.columns],
                       ax=ax, alpha=0.85)
    ax.set_xlabel('% of cells', fontsize=_FS_AX)
    ax.set_title('Direction Consistency: High Cluster vs Low Cluster\n'
                 '(2-cluster: up/down only  |  3-cluster: also checks monotonicity)',
                 fontsize=_FS_TTL, fontweight='bold')
    ax.legend(title='Direction', fontsize=_FS_SM, bbox_to_anchor=(1.02, 1), loc='upper left')
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()

    print("\nDirection consistency summary:")
    print(summary.rename(columns={'spike_feature_clean': 'spike_feature'}).to_string(index=False))
    return df_res


# ==========================================
# 13. WITHIN-CELL PERMUTATION TEST
# ==========================================

def run_within_cell_permutation(per_spike_pickle_dir, df_stats, n_permutations=500, alpha=0.05):
    from matplotlib.patches import Patch

    def _max_d_vectorized(matrix, group_sizes):
        """
        Compute max Cohen's d across ALL time bins and ALL group pairs in one pass.
        matrix : (n_spikes, n_bins) float32 — rows already ordered by group
        group_sizes : list of ints summing to n_spikes
        Returns a scalar (the max d across bins and pairs).
        """
        groups_data = []
        start = 0
        for sz in group_sizes:
            groups_data.append(matrix[start:start + sz])
            start += sz
        max_d = 0.0
        for i in range(len(groups_data)):
            for j in range(i + 1, len(groups_data)):
                a, b = groups_data[i], groups_data[j]
                na, nb = len(a), len(b)
                if na < 2 or nb < 2:
                    continue
                pooled = np.sqrt(
                    ((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2)
                )
                d = np.abs(a.mean(0) - b.mean(0)) / (pooled + 1e-8)
                cur = float(d.max())
                if cur > max_d:
                    max_d = cur
        return max_d

    pkl_files = glob.glob(os.path.join(per_spike_pickle_dir, "*_per_spike_data.pkl"))
    if not pkl_files:
        print(f"No per-spike pickles found in {per_spike_pickle_dir}")
        return None, None

    rng = np.random.default_rng(42)
    rows = []

    for pkl_path in sorted(pkl_files):
        with open(pkl_path, 'rb') as f:
            per_spike = pickle.load(f)
        cell_id = per_spike['cell_id']
        for lfp_label, spike_families in per_spike.items():
            if lfp_label == 'cell_id': continue
            for spike_family, data in spike_families.items():
                # float32 halves memory and speeds up numpy ops
                mat    = np.array(data['feature_matrix'], dtype=np.float32)
                labels = np.array(data['cluster_labels'])
                groups = np.unique(labels)
                if len(groups) < 2: continue

                # group sizes in label order; real data — rows already grouped
                group_sizes = [int((labels == g).sum()) for g in groups]
                # sort rows by group so _max_d_vectorized can slice cleanly
                sort_idx = np.argsort(labels, kind='stable')
                mat_sorted = mat[sort_idx]

                real_max_d = _max_d_vectorized(mat_sorted, group_sizes)

                # permutation loop — only shuffle row order, vectorized over bins
                null_max_ds = np.empty(n_permutations, dtype=np.float32)
                for p in range(n_permutations):
                    perm_idx = rng.permutation(len(labels))
                    null_max_ds[p] = _max_d_vectorized(mat[perm_idx], group_sizes)

                null_95th = float(np.percentile(null_max_ds, 95))
                perm_p    = float(np.mean(null_max_ds >= real_max_d))
                rows.append({'cell_id': cell_id, 'lfp_feature': lfp_label,
                             'spike_feature': spike_family,
                             'real_max_d': round(real_max_d, 4),
                             'null_95th':  round(null_95th, 4),
                             'perm_p':     round(perm_p, 4),
                             'sig':        perm_p < alpha})
        print(f"  Done: {cell_id}")

    df_perm = pd.DataFrame(rows)
    summary_rows = []
    for (lfp, spk), grp in df_perm.groupby(['lfp_feature', 'spike_feature']):
        n_cells_tested = grp['cell_id'].nunique()
        n_sig = grp['sig'].sum()
        summary_rows.append({'lfp_feature': lfp, 'spike_feature': spk,
                             'n_cells_tested': n_cells_tested, 'n_perm_sig': n_sig,
                             'perm_yield_pct': round(100 * n_sig / n_cells_tested, 1) if n_cells_tested > 0 else 0})
    df_summary = pd.DataFrame(summary_rows).sort_values('perm_yield_pct', ascending=False).reset_index(drop=True)

    # Plot 1: per-cell heatmap, one panel per LFP feature
    # rows = cells, columns = spike features, value = max Cohen's d
    # significant cells get a thick black border
    df_perm['spike_feature_clean'] = df_perm['spike_feature'].str.replace('_cluster', '', regex=False)
    lfp_feats_sorted  = sorted(df_perm['lfp_feature'].unique())
    spk_feats_sorted  = sorted(df_perm['spike_feature_clean'].unique())
    cells_sorted      = sorted(df_perm['cell_id'].unique())
    n_lfp   = len(lfp_feats_sorted)
    n_cols  = min(3, n_lfp)
    n_rows  = math.ceil(n_lfp / n_cols)
    panel_w = max(4, len(spk_feats_sorted) * 1.3)
    panel_h = max(3, len(cells_sorted) * 0.7 + 1)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(panel_w * n_cols, panel_h * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()
    for idx, lfp in enumerate(lfp_feats_sorted):
        ax = axes_flat[idx]
        sub = df_perm[df_perm['lfp_feature'] == lfp]
        pivot_d   = sub.pivot(index='cell_id', columns='spike_feature_clean', values='real_max_d')
        pivot_sig = sub.pivot(index='cell_id', columns='spike_feature_clean', values='sig').fillna(False)
        pivot_d   = pivot_d.reindex(index=cells_sorted, columns=spk_feats_sorted)
        pivot_sig = pivot_sig.reindex(index=cells_sorted, columns=spk_feats_sorted).fillna(False)
        sns.heatmap(pivot_d, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                    linewidths=0.5, linecolor='white', vmin=0,
                    annot_kws={'size': _FS_SM},
                    cbar_kws={'label': "Max Cohen's d", 'shrink': 0.8})
        # Thick border on significant cells
        for ri, cell in enumerate(cells_sorted):
            for ci, spk in enumerate(spk_feats_sorted):
                try:
                    is_sig = bool(pivot_sig.loc[cell, spk])
                except KeyError:
                    is_sig = False
                if is_sig:
                    ax.add_patch(plt.Rectangle((ci, ri), 1, 1, fill=False,
                                               edgecolor='black', lw=2.5, zorder=3))
        ax.set_title(lfp, fontsize=_FS_SUB, fontweight='bold')
        ax.set_xlabel('Spike Feature', fontsize=_FS_AX)
        ax.set_ylabel('Cell', fontsize=_FS_AX)
        ax.tick_params(axis='x', labelsize=_FS_SM, rotation=40)
        ax.tick_params(axis='y', labelsize=_FS_SM, rotation=0)
    for j in range(n_lfp, len(axes_flat)):
        fig.delaxes(axes_flat[j])
    plt.suptitle(
        f"Within-Cell Permutation: Max Cohen's d per Cell\n"
        f"(n={n_permutations} shuffles  |  black border = perm p < {alpha})",
        fontsize=_FS_TTL, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()

    # Plot 2: population summary heatmap
    if not df_summary.empty:
        pivot = df_summary.pivot(index='spike_feature', columns='lfp_feature', values='perm_yield_pct')
        pivot.index = [i.replace('_cluster', '') for i in pivot.index]
        fig2, ax2 = plt.subplots(figsize=(max(6, len(pivot.columns) * 1.2), max(4, len(pivot) * 0.6 + 1)))
        sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlOrRd', vmin=0, vmax=100,
                    linewidths=0.5, ax=ax2, cbar_kws={'label': '% cells perm-sig'})
        ax2.set_title('Within-Cell Permutation: % Cells Significant per Combo', fontsize=_FS_SUB, fontweight='bold')
        ax2.set_xticklabels(ax2.get_xticklabels(), rotation=40, ha='right', fontsize=_FS_SM)
        ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=_FS_SM)
        plt.tight_layout()
        plt.show()

    n_sig_total = df_perm['sig'].sum()
    print(f"\nWithin-cell permutation summary (alpha={alpha}, n={n_permutations}):")
    print(f"  Significant cell-combos: {n_sig_total} / {len(df_perm)}")
    print("\nPopulation summary:")
    print(df_summary.to_string(index=False))
    return df_perm, df_summary


# ==========================================
# 14. COMBINED VALIDATION SUMMARY
# ==========================================

def plot_combined_validation_summary(boot_results, perm_summary, binom_results=None):
    """
    Unified two-panel figure tying together bootstrap and within-cell permutation.

    Left panel  — Bootstrap robustness heatmap (all three criteria: d + p + yield)
    Right panel — Within-cell permutation yield heatmap (% cells significant)

    Optionally overlays binomial yield test significance as hatching on the right panel.
    """
    boot_clean = boot_results[['lfp_feature', 'spike_feature',
                                'd_robust', 'p_robust', 'yield_robust', 'all_robust']].copy()
    boot_clean['spike_feature'] = boot_clean['spike_feature'].str.replace('_cluster', '', regex=False)

    perm_clean = perm_summary[['lfp_feature', 'spike_feature', 'perm_yield_pct']].copy()
    perm_clean['spike_feature'] = perm_clean['spike_feature'].str.replace('_cluster', '', regex=False)

    merged = boot_clean.merge(perm_clean, on=['lfp_feature', 'spike_feature'], how='outer')

    if binom_results is not None:
        binom_clean = binom_results[['lfp_feature', 'spike_feature', 'sig']].copy()
        binom_clean = binom_clean.rename(columns={'sig': 'binom_sig'})
        binom_clean['spike_feature'] = binom_clean['spike_feature'].str.replace('_cluster', '', regex=False)
        merged = merged.merge(binom_clean, on=['lfp_feature', 'spike_feature'], how='left')

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, max(4, merged['spike_feature'].nunique() * 0.65 + 2))
    )

    # Panel 1: Bootstrap "all robust"
    pivot1 = merged.pivot(index='spike_feature', columns='lfp_feature', values='all_robust').astype(float)
    sns.heatmap(pivot1, annot=True, fmt='.0f', cmap='RdYlGn', vmin=0, vmax=1,
                linewidths=0.5, ax=axes[0], cbar=False)
    axes[0].set_title('Bootstrap: All Robust\n(effect size + p-value + yield)', fontsize=_FS_SUB, fontweight='bold')
    axes[0].set_xlabel('LFP Feature', fontsize=_FS_AX)
    axes[0].set_ylabel('Spike Feature', fontsize=_FS_AX)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=40, ha='right', fontsize=_FS_SM)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, fontsize=_FS_SM)

    # Panel 2: Within-cell permutation yield
    pivot2 = merged.pivot(index='spike_feature', columns='lfp_feature', values='perm_yield_pct')
    sns.heatmap(pivot2, annot=True, fmt='.0f', cmap='YlOrRd', vmin=0, vmax=100,
                linewidths=0.5, ax=axes[1], cbar_kws={'label': '% cells sig (within-cell perm)'})

    # Hatch cells that are binomial-yield significant
    if binom_results is not None and 'binom_sig' in merged.columns:
        pivot_binom = merged.pivot(index='spike_feature', columns='lfp_feature', values='binom_sig').fillna(False)
        # Align to same row/col order as pivot2
        pivot_binom = pivot_binom.reindex(index=pivot2.index, columns=pivot2.columns).fillna(False)
        for ri, row_lbl in enumerate(pivot2.index):
            for ci, col_lbl in enumerate(pivot2.columns):
                if pivot_binom.loc[row_lbl, col_lbl]:
                    axes[1].add_patch(
                        plt.Rectangle((ci, ri), 1, 1, fill=False,
                                      edgecolor='black', lw=2, hatch='///')
                    )

    axes[1].set_title('Within-Cell Permutation\n% Cells Significant', fontsize=_FS_SUB, fontweight='bold')
    axes[1].set_xlabel('LFP Feature', fontsize=_FS_AX)
    axes[1].set_ylabel('')
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=40, ha='right', fontsize=_FS_SM)
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=_FS_SM)

    hatch_note = ' (/// = binomial yield sig)' if binom_results is not None else ''
    plt.suptitle(f'Validation Summary: Bootstrap + Within-Cell Permutation{hatch_note}',
                 fontsize=_FS_TTL, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ==========================================
# 15. CONCLUSION SUMMARY
# ==========================================

def summarize_significant_combos(df_stats, master_traces, boot_results, perm_summary,
                                  binom_results=None, perm_yield_threshold=50, alpha=0.05):
    """
    Identifies validated LFP × spike feature associations and summarises:
      - Direction: does the High spike cluster show higher or lower LFP values?
      - Peak timing: which window (in ms) relative to spike has the largest effect?
      - Timing category: pre-spike (< -50 ms), peri-spike (±50 ms), post-spike (> +50 ms)
      - Validation status: bootstrap robust AND/OR within-cell perm yield >= threshold

    A combo is considered validated if:
      all_robust (bootstrap) OR perm_yield_pct >= perm_yield_threshold
    """
    # --- 1. Compute signed Cohen's d per row ---
    df_s = df_stats.copy()
    signs = []
    for _, row in df_s.iterrows():
        tr = next((t for t in master_traces
                   if t['cell_id']    == row['cell_id'] and
                      t['lfp_feature'] == row['lfp_feature'] and
                      t['spike_feature'] == row['spike_feature']), None)
        sign = 1
        if tr and 'groups' in tr['trace_data']:
            groups = tr['trace_data']['groups']
            tgrid  = np.array(tr['trace_data'].get('Tgrid', []))
            hk = next((k for k in groups if 'high' in k.lower()), None)
            lk = next((k for k in groups if 'low'  in k.lower()), None)
            if hk and lk and len(tgrid):
                hv = groups[hk]; lv = groups[lk]
                ha = hv.get('mean', hv.get('avg')) if isinstance(hv, dict) else np.array(hv)
                la = lv.get('mean', lv.get('avg')) if isinstance(lv, dict) else np.array(lv)
                if ha is not None and la is not None:
                    if np.ndim(ha) == 2: ha = np.nanmean(ha, axis=0)
                    if np.ndim(la) == 2: la = np.nanmean(la, axis=0)
                    mask = (tgrid >= row['window_start']) & (tgrid <= row['window_end'])
                    if np.any(mask) and len(ha) == len(tgrid) and len(la) == len(tgrid):
                        if np.nanmean(la[mask]) > np.nanmean(ha[mask]):
                            sign = -1
        signs.append(sign)
    df_s['signed_d'] = df_s['cohens_d'] * signs

    # --- 2. Aggregate per combo: direction + peak window ---
    combo_rows = []
    for (lfp, spk), grp in df_s.groupby(['lfp_feature', 'spike_feature']):
        win_med = grp.groupby(['window_start', 'window_end'])['cohens_d'].median()
        peak_ws, peak_we = win_med.idxmax()
        peak_d  = win_med.max()
        peak_mid_ms = (peak_ws + peak_we) / 2 * 1000
        median_signed = grp['signed_d'].median()
        if peak_mid_ms < -50:
            timing_cat = 'pre-spike'
        elif peak_mid_ms > 50:
            timing_cat = 'post-spike'
        else:
            timing_cat = 'peri-spike'
        combo_rows.append({
            'lfp_feature':        lfp,
            'spike_feature':      spk,
            'spike_feature_clean': spk.replace('_cluster', ''),
            'median_signed_d':    round(median_signed, 3),
            'direction':          'positive' if median_signed >= 0 else 'negative',
            'peak_window':        f"{peak_ws*1000:.0f} to {peak_we*1000:.0f} ms",
            'peak_mid_ms':        round(peak_mid_ms, 1),
            'timing_cat':         timing_cat,
            'peak_d':             round(peak_d, 3),
        })
    df_combos = pd.DataFrame(combo_rows)

    # --- 3. Join validation results ---
    def _clean(df, extra_cols):
        d = df[['lfp_feature', 'spike_feature'] + extra_cols].copy()
        d['spike_feature_clean'] = d['spike_feature'].str.replace('_cluster', '', regex=False)
        return d.drop(columns='spike_feature')

    df_combos = df_combos.merge(
        _clean(boot_results, ['all_robust', 'd_robust']),
        on=['lfp_feature', 'spike_feature_clean'], how='left')
    df_combos = df_combos.merge(
        _clean(perm_summary, ['perm_yield_pct', 'n_perm_sig', 'n_cells_tested']),
        on=['lfp_feature', 'spike_feature_clean'], how='left')
    if binom_results is not None:
        df_combos = df_combos.merge(
            _clean(binom_results, ['sig']).rename(columns={'sig': 'binom_sig'}),
            on=['lfp_feature', 'spike_feature_clean'], how='left')

    df_combos['perm_sig']  = df_combos['perm_yield_pct'].fillna(0) >= perm_yield_threshold
    df_combos['validated'] = df_combos['all_robust'].fillna(False) | df_combos['perm_sig']
    df_sig = (df_combos[df_combos['validated']]
              .sort_values('peak_d', ascending=False)
              .reset_index(drop=True))

    # --- 4. Text summary ---
    print("=" * 72)
    print("CONCLUSION: Validated LFP × Spike Feature Associations")
    print("=" * 72)
    print(f"  Criterion: bootstrap all-robust OR within-cell perm yield >= {perm_yield_threshold}%")
    print(f"  Validated: {len(df_sig)} / {len(df_combos)} combos\n")
    for _, r in df_sig.iterrows():
        hi_lo = 'HIGHER' if r['direction'] == 'positive' else 'LOWER'
        n_str = (f"{int(r['n_perm_sig'])}/{int(r['n_cells_tested'])} cells"
                 if pd.notna(r.get('n_perm_sig')) else 'N/A')
        boot_str = 'robust' if r['all_robust'] else 'not robust'
        print(f"  [{r['spike_feature_clean']}]  x  [{r['lfp_feature']}]")
        print(f"    Direction : spikes with higher {r['spike_feature_clean']} → {hi_lo} {r['lfp_feature']}")
        print(f"    Peak timing: {r['peak_window']}  ({r['timing_cat']})  |  peak Cohen's d = {r['peak_d']}")
        print(f"    Bootstrap  : {boot_str}")
        print(f"    Perm test  : {n_str} significant  ({r['perm_yield_pct']:.0f}%)")
        if 'binom_sig' in r and pd.notna(r['binom_sig']):
            print(f"    Binom test : {'sig' if r['binom_sig'] else 'not sig'}")
        print()

    # --- 5. Summary table figure ---
    if df_sig.empty:
        print("No validated combos to display.")
        return df_sig

    col_headers = ['Spike Feature', 'LFP Feature', 'Direction',
                   'Peak Window (ms)', 'Timing', "Peak d",
                   'Boot\nRobust', 'Perm\nYield %']
    has_binom = 'binom_sig' in df_sig.columns
    if has_binom:
        col_headers.append('Binom\nSig')

    table_data = []
    row_colors = []
    for _, r in df_sig.iterrows():
        arrow = '↑ Higher' if r['direction'] == 'positive' else '↓ Lower'
        row = [r['spike_feature_clean'], r['lfp_feature'], arrow,
               r['peak_window'], r['timing_cat'], f"{r['peak_d']:.2f}",
               '✓' if r['all_robust'] else '✗',
               f"{r['perm_yield_pct']:.0f}%"]
        if has_binom:
            row.append('✓' if r.get('binom_sig') else '✗')
        table_data.append(row)
        row_colors.append('#fff0e6' if r['direction'] == 'positive' else '#e6f0ff')

    fig, ax = plt.subplots(figsize=(16, max(2.5, len(df_sig) * 0.6 + 1.5)))
    ax.axis('off')
    tbl = ax.table(cellText=table_data, colLabels=col_headers,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(_FS_SM)
    tbl.scale(1, 2.0)
    for ci in range(len(col_headers)):
        tbl[0, ci].set_facecolor('#d0d0d0')
        tbl[0, ci].set_text_props(fontweight='bold')
    for ri, color in enumerate(row_colors):
        for ci in range(len(col_headers)):
            tbl[ri + 1, ci].set_facecolor(color)
    ax.set_title('Validated LFP × Spike Feature Associations\n'
                 '(orange = High cluster → higher LFP  |  blue = High cluster → lower LFP)',
                 fontsize=_FS_SUB, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.show()

    return df_sig


# ==========================================
# 16. DIRECTION + TIMING TRACE PLOTS
# ==========================================

def plot_validated_directions(df_stats, master_traces, df_conclusions=None):
    """
    For each validated LFP × spike feature combo, plots the grand average LFP
    trace for Low / Mid / High spike clusters with significant time windows
    shaded. The subplot title states the direction and peak timing directly.

    One figure per LFP feature; columns = spike features (validated only).
    If df_conclusions is None, plots all combos in df_stats.
    """
    # --- decide which combos to plot ---
    if df_conclusions is not None and not df_conclusions.empty:
        combos_to_plot = list(zip(df_conclusions['lfp_feature'],
                                  df_conclusions['spike_feature']))
        direction_map  = {(r['lfp_feature'], r['spike_feature']): r
                          for _, r in df_conclusions.iterrows()}
    else:
        combos_to_plot = list(df_stats.groupby(['lfp_feature','spike_feature']).groups.keys())
        direction_map  = {}

    # group by LFP feature
    lfp_groups = {}
    for lfp, spk in combos_to_plot:
        lfp_groups.setdefault(lfp, []).append(spk)

    for lfp, spk_list in sorted(lfp_groups.items()):
        spk_list = sorted(spk_list)
        n_spk = len(spk_list)
        cols  = min(3, n_spk)
        rows  = math.ceil(n_spk / cols)

        fig, axes = plt.subplots(rows, cols,
                                 figsize=(5.5 * cols, 4.5 * rows),
                                 squeeze=False)
        axes_flat = axes.flatten()

        for idx, spk in enumerate(spk_list):
            ax = axes_flat[idx]
            spk_clean = spk.replace('_cluster', '')

            # collect traces for this combo
            cell_traces = [t for t in master_traces
                           if t['lfp_feature'] == lfp and t['spike_feature'] == spk]
            if not cell_traces:
                fig.delaxes(ax); continue

            # master time grid (seconds)
            ref = next((t for t in cell_traces
                        if 'Tgrid' in t.get('trace_data', {})), None)
            if ref is None:
                fig.delaxes(ax); continue
            Tgrid = np.array(ref['trace_data']['Tgrid'])

            # stack grand averages per cluster group
            stacked = {'low': [], 'mid': [], 'high': []}
            for t in cell_traces:
                td = t['trace_data']
                if 'groups' not in td: continue
                cell_T = np.array(td.get('Tgrid', Tgrid))
                for grp_key in ['low', 'mid', 'high']:
                    gk = next((k for k in td['groups'] if grp_key in k.lower()), None)
                    if not gk: continue
                    val = td['groups'][gk]
                    arr = val.get('mean', val.get('avg')) if isinstance(val, dict) else np.array(val)
                    if arr is None: continue
                    arr = np.array(arr)
                    if arr.ndim == 2: arr = np.nanmean(arr, axis=0)
                    stacked[grp_key].append(np.interp(Tgrid, cell_T, arr))

            # plot traces
            for grp_key, color, label in [('low',  _CLUST_LOW,  'Low'),
                                           ('mid',  _CLUST_MID,  'Mid'),
                                           ('high', _CLUST_HIGH, 'High')]:
                if not stacked[grp_key]: continue
                mat = np.vstack(stacked[grp_key])
                gm  = np.nanmean(mat, axis=0)
                ge  = np.nanstd(mat, axis=0) / np.sqrt(mat.shape[0])
                ax.plot(Tgrid * 1000, gm, color=color, lw=2.5,
                        label=f'{label} (n={mat.shape[0]})')
                ax.fill_between(Tgrid * 1000, gm - ge, gm + ge,
                                color=color, alpha=0.2, lw=0)

            # shade significant windows
            sig_wins = (df_stats[(df_stats['lfp_feature'] == lfp) &
                                 (df_stats['spike_feature'] == spk)]
                        [['window_start','window_end']].drop_duplicates())
            for _, win in sig_wins.iterrows():
                ax.axvspan(win['window_start'] * 1000, win['window_end'] * 1000,
                           alpha=0.12, color='gray', zorder=0)

            ax.axvline(0, color='black', ls='--', lw=1.5, alpha=0.7, label='spike (t=0)')

            # direction + timing annotation in title
            info = direction_map.get((lfp, spk))
            if info is not None:
                arrow = '\u2191' if info['direction'] == 'positive' else '\u2193'
                title = (f"{spk_clean}\n"
                         f"{arrow} Higher {spk_clean}  \u2192  {arrow} {lfp}\n"
                         f"Peak: {info['peak_window']}  ({info['timing_cat']})")
            else:
                title = spk_clean

            ax.set_title(title, fontsize=_FS_SM, fontweight='bold', pad=6)
            ax.set_xlabel('Time relative to spike (ms)', fontsize=_FS_AX)
            ax.set_ylabel(lfp, fontsize=_FS_AX)
            ax.tick_params(labelsize=_FS_SM)
            ax.legend(fontsize=_FS_SM - 1, loc='upper right')
            sns.despine(ax=ax)

        for j in range(n_spk, len(axes_flat)):
            fig.delaxes(axes_flat[j])

        plt.suptitle(
            f'{lfp}  —  Grand Average Traces by Cluster\n'
            f'(grey shading = significant windows  |  arrow = direction of effect)',
            fontsize=_FS_TTL, fontweight='bold'
        )
        plt.tight_layout()
        plt.show()


# ============================================================
# CELL SUBSET UTILITIES
# ============================================================

def get_cell_subsets(priority_cells, all_cell_ids):
    """
    Return three lists of cell ID strings for population analysis.

    Parameters
    ----------
    priority_cells : list of int   e.g. [3, 4, 21, 24, 26, 27, 42]
    all_cell_ids   : list of str   e.g. ['c1', 'c2', ...]

    Returns
    -------
    subsets : dict with keys 'priority', 'np', 'all'
    """
    priority_str = {f"c{n}" for n in priority_cells}
    all_str      = set(all_cell_ids)
    return {
        "priority": sorted(priority_str & all_str),
        "np":       sorted(all_str - priority_str),
        "all":      sorted(all_str),
    }


def filter_df_stats(df_stats, cell_ids):
    """Filter a compiled stats DataFrame to the given cell IDs."""
    return df_stats[df_stats["cell_id"].isin(cell_ids)].copy()


def filter_traces(master_traces, cell_ids):
    """Filter a master_traces list to the given cell IDs."""
    cell_set = set(cell_ids)
    return [t for t in master_traces if t.get("cell_id") in cell_set]


# ============================================================
# PRE/POST SPECPARAM DATA COMPILER
# ============================================================

def compile_prepost_stats(
    pickle_dir=None,
):
    """
    Load all per-cell pre/post specparam pickles and compile into a
    unified DataFrame for population-level analysis.

    Parameters
    ----------
    pickle_dir : str, optional
        Directory containing c{N}_prepost.pkl files.
        Defaults to SPE1_PICKLE_ROOT/prepost_specparam_pickles.

    Returns
    -------
    df : pd.DataFrame
        One row per statistical test. Columns include:
        cell_id, spike_feature, lfp_feature, window,
        comparison, p, p_fdr, sig,
        cohens_d / cohens_dz, hedges_g / hedges_gz,
        rank_biserial_r, ci_lo, ci_hi, n1, n2 / n,
        resampled, skipped, pre_window, post_window.
    """
    if pickle_dir is None:
        try:
            import sys, os
            sys.path.insert(0, os.path.dirname(__file__))
            from config import SPE1_PICKLE_ROOT
            pickle_dir = os.path.join(SPE1_PICKLE_ROOT, "prepost_specparam_pickles")
        except Exception:
            raise ValueError("pickle_dir is required if SPE1_PICKLE_ROOT is not available.")

    files = sorted(glob.glob(os.path.join(pickle_dir, "*_prepost.pkl")))
    rows  = []

    for fpath in files:
        with open(fpath, "rb") as f:
            cell_data = pickle.load(f)

        for spike_feat, col_res in cell_data.items():
            cell_id     = col_res.get("cell_id", "unknown")
            pre_window  = col_res.get("pre_window")
            post_window = col_res.get("post_window")

            for test in col_res.get("tests", []):
                if test.get("skipped"):
                    continue
                row = {
                    "cell_id":     cell_id,
                    "spike_feature": spike_feat,
                    "lfp_feature":   test.get("feat_key"),
                    "lfp_label":     test.get("feature"),
                    "window":        test.get("window"),
                    "comparison":    test.get("comparison"),
                    "p":             test.get("p"),
                    "p_fdr":         test.get("p_fdr"),
                    "sig":           test.get("sig", "ns"),
                    # between-group effect sizes
                    "cohens_d":       test.get("cohens_d"),
                    "hedges_g":       test.get("hedges_g"),
                    # within-group (paired) effect sizes
                    "cohens_dz":      test.get("cohens_dz"),
                    "hedges_gz":      test.get("hedges_gz"),
                    # shared
                    "rank_biserial_r": test.get("rank_biserial_r"),
                    "mean_diff":       test.get("mean_diff"),
                    "ci_lo":           test.get("ci_lo"),
                    "ci_hi":           test.get("ci_hi"),
                    "n1":              test.get("n1"),
                    "n2":              test.get("n2"),
                    "n":               test.get("n"),
                    "resampled":       test.get("resampled", False),
                    "pre_window":      str(pre_window),
                    "post_window":     str(post_window),
                }
                rows.append(row)

    return pd.DataFrame(rows)


def plot_prepost_effect_sizes(df_prepost, feature_shades=None, subset_label=""):
    """
    Population effect sizes for pre/post specparam analysis.

    Matches the style of plot_population_effect_sizes:
      - Boxplot + stripplot, x = LFP feature, hue = spike feature
      - Vertical dotted dividers between LFP features
      - Separate figure per window (pre, post, within, interaction)
      - Effect size: Hedges' g for between-group; Cohen's d_z for within-group
    """
    warnings.filterwarnings("ignore", category=FutureWarning)

    if feature_shades is None:
        feature_shades = _SPIKE_FEAT_COLORS

    master_order = list(feature_shades.keys())

    for win in ["pre", "post", "within", "interaction"]:
        df_w = df_prepost[df_prepost["window"] == win].copy()
        if df_w.empty:
            continue

        es_col, es_label = (
            ("cohens_dz", "Effect Size (Cohen's d_z)")
            if win == "within"
            else ("hedges_g", "Effect Size (Hedges' g)")
        )
        df_w = df_w.dropna(subset=[es_col])
        if df_w.empty:
            continue

        lfp_order    = sorted(df_w["lfp_feature"].unique())
        hue_order    = [h for h in master_order if h in df_w["spike_feature"].values]
        palette_used = {k: v for k, v in feature_shades.items() if k in hue_order}

        title = f"Population Pre/Post Effect Sizes  [{win}]"
        if subset_label:
            title += f"  —  {subset_label}"

        plt.figure(figsize=(15, 7))
        ax = sns.boxplot(
            data=df_w,
            x="lfp_feature", y=es_col,
            hue="spike_feature",
            order=lfp_order, hue_order=hue_order,
            palette=palette_used,
            width=0.75, fliersize=0,
            boxprops={"alpha": 0.4}, zorder=1,
        )
        sns.stripplot(
            data=df_w,
            x="lfp_feature", y=es_col,
            hue="spike_feature",
            order=lfp_order, hue_order=hue_order,
            palette=palette_used,
            dodge=True, alpha=0.85, jitter=0.2,
            size=5, linewidth=0.8, edgecolor="gray",
            legend=False, ax=ax, zorder=2,
        )
        for i in range(len(lfp_order) - 1):
            plt.axvline(i + 0.5, color="grey", linestyle=":", linewidth=1.5, alpha=0.6, zorder=0)
        plt.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

        plt.title(title, fontweight="bold", fontsize=_FS_TTL, pad=15)
        plt.ylabel(es_label, fontsize=_FS_AX, fontweight="bold")
        plt.xlabel("LFP Feature", fontsize=_FS_AX, fontweight="bold")
        plt.xticks(rotation=15, ha="right", fontsize=_FS_AX)

        handles, labels = ax.get_legend_handles_labels()
        n = len(hue_order)
        clean_labels = [l.replace("_cluster", "") for l in labels[:n]]
        plt.legend(handles[:n], clean_labels, title="Spike Feature",
                   bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, shadow=True)
        plt.tight_layout()
        plt.show()


def plot_prepost_yield(df_prepost, subset_label=""):
    """
    Yield heatmap for pre/post specparam analysis.

    Matches the style of plot_significant_yield_heatmap:
      - YlGnBu colormap, 0–100% scale
      - Annotated with '% (sig/total cells)'
      - Separate heatmap per window (pre, post, within, interaction)
    """
    for win in ["pre", "post", "within", "interaction"]:
        df_w = df_prepost[df_prepost["window"] == win]
        if df_w.empty:
            continue

        total_cells = df_w["cell_id"].nunique()
        if total_cells == 0:
            continue

        sig_counts = (
            df_w[df_w["sig"] != "ns"]
            .groupby(["lfp_feature", "spike_feature"])["cell_id"]
            .nunique()
            .reset_index(name="sig_cells")
        )
        sig_counts["yield_pct"]   = sig_counts["sig_cells"] / total_cells * 100
        sig_counts["text_label"]  = (sig_counts["sig_cells"].astype(str)
                                     + "/" + str(total_cells)
                                     + "\n(" + sig_counts["yield_pct"].round(0).astype(int).astype(str) + "%)")

        pivot_color = sig_counts.pivot(index="lfp_feature", columns="spike_feature",
                                       values="yield_pct").fillna(0)
        pivot_text  = sig_counts.pivot(index="lfp_feature", columns="spike_feature",
                                       values="text_label").fillna("")
        pivot_color.columns = [c.replace("_cluster", "") for c in pivot_color.columns]
        pivot_text.columns  = [c.replace("_cluster", "") for c in pivot_text.columns]

        title = f"Pre/Post: % Cells Significant  [{win}]"
        if subset_label:
            title += f"  —  {subset_label}"

        plt.figure(figsize=(max(10, pivot_color.shape[1] * 1.4),
                            max(5,  pivot_color.shape[0] * 0.8)))
        sns.heatmap(
            pivot_color,
            annot=pivot_text, fmt="",
            cmap="YlGnBu", vmin=0, vmax=100,
            linewidths=0.5, linecolor="white",
            cbar_kws={"label": "% of Cells with Significant Effect"},
        )
        plt.title(title, fontweight="bold", fontsize=_FS_TTL, pad=15)
        plt.xlabel("Spike Feature (Clustering Metric)", fontsize=_FS_AX, fontweight="bold")
        plt.ylabel("LFP Feature", fontsize=_FS_AX, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
