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


# ==========================================
# 1. DATA COMPILER
# ==========================================

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
    all_files = glob.glob(os.path.join(pickle_dir, "*_sliding_stats.pkl"))
    
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

    plt.title("Population Effect Sizes", fontweight="bold", fontsize=15, pad=15)
    plt.ylabel("Effect Size (Cohen's d)", fontsize=12, fontweight="bold")
    plt.xlabel("LFP Feature", fontsize=12, fontweight="bold")
    plt.xticks(rotation=15, ha='right', fontsize=11)
    
    # Clean up the legend
    handles, labels = ax.get_legend_handles_labels()
    num_features = len(master_order)
    plt.legend(handles[:num_features], labels[:num_features], title="Spike Feature", 
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
            sig_counts = np.zeros_like(time_bins)
            
            # Count the overlapping significant windows per time bin
            for _, row in df_sub.iterrows():
                start = row["window_start"]
                end = row["window_end"]
                mask = (time_bins >= start) & (time_bins <= end)
                sig_counts[mask] += 1
                
            # Look up the hex color directly from the provided dictionary
            color = feature_shades.get(spk_feat, '#cccccc')
            
            # Plot the density curve
            ax.plot(time_bins, sig_counts, color=color, lw=2.5)
            ax.fill_between(time_bins, 0, sig_counts, color=color, alpha=0.15)

        # Subplot Aesthetics
        ax.axvline(0, color="k", ls="--", lw=2)
        ax.set_title(f"{target_lfp}", fontweight="bold", fontsize=14)
        ax.set_ylabel("Count of Significant Windows")
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # Keep legend in the first subplot, but force it to show ALL active features
        if idx == 0:
            all_active_spk_feats = df_stats["spike_feature"].unique()
            custom_lines = [Line2D([0], [0], color=feature_shades.get(feat, '#cccccc'), lw=2.5) 
                            for feat in all_active_spk_feats]
            ax.legend(custom_lines, all_active_spk_feats, title="Clustered By", fontsize=9, loc="upper right")

    # Clean up any empty subplots in the grid
    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    # Main Titles and layout adjustments
    plt.suptitle("Temporal Distribution of Significance Across All Features", fontweight="bold", fontsize=18, y=1.02)
    fig.text(0.5, -0.01, 'Time relative to spike (s)', ha='center', fontsize=14)
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
                    if 'low' in c_lower: color = '#1f77b4'
                    elif 'high' in c_lower: color = '#ff7f0e'
                    elif 'mid' in c_lower: color = '#2ca02c'
                    else: color = None
                    
                    label_clean = c_name.split(': ')[-1].capitalize() if ':' in c_name else c_name
                    ax.plot(master_Tgrid, grand_mean, label=f"{label_clean} (n={matrix.shape[0]})", color=color, lw=2.5)
                    ax.fill_between(master_Tgrid, grand_mean - grand_sem, grand_mean + grand_sem, color=color, alpha=0.2, lw=0)
                    lines_plotted += 1
            
            ax.axvline(0, color="k", ls="--", lw=1.5)
            ax.set_title(spk_feat, fontweight="bold", fontsize=12)
            
            if lines_plotted > 0:
                ax.legend(fontsize=8, loc="upper right")
            else:
                ax.text(0.5, 0.5, "No valid arrays found", ha='center', va='center', transform=ax.transAxes, color='gray')

        for i in range(n_spk, len(axes)):
            fig.delaxes(axes[i])
            
        plt.suptitle(f"Population Grand Average: {lfp_feat}", fontweight="bold", fontsize=18, y=1.02)
        fig.text(0.5, -0.01, 'Time relative to spike (s)', ha='center', fontsize=14)
        fig.text(-0.01, 0.5, f'Δ {lfp_feat}', va='center', rotation='vertical', fontsize=14)
        
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
    
    # 1. Recover the sign for every single significant window
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
    plt.xlabel("Spike Feature (Clustering Metric)", fontsize=13, fontweight="bold")
    plt.ylabel("LFP Feature", fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha='right', fontsize=11)
    plt.yticks(fontsize=11)
    
    # Text legend without emojis!
    legend_text = (
        "Positive (+ / Red): High cluster is associated with HIGHER LFP values.\n"
        "Negative (- / Blue): High cluster is associated with LOWER LFP values."
    )
    plt.figtext(0.5, -0.05, legend_text, ha="center", fontsize=11, 
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
    pivot_text = df_yield.pivot(index='lfp_feature', columns='spike_feature', values='text_label')
    
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
    
    plt.title("How Consistent is the Effect Across the Population?", fontweight="bold", fontsize=15, pad=15)
    plt.xlabel("Spike Feature (Clustering Metric)", fontsize=12, fontweight="bold")
    plt.ylabel("LFP Feature", fontsize=12, fontweight="bold")
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()


# ==========================================
# 7. SPIKE FEATURE REDUNDANCY (CORRELATION)
# ==========================================

def plot_feature_redundancy(pop_traces, df_stats):
    """
    Checks if spike features are redundant across the population.
    Filters out any spike features that did not yield significant results.
    """
    # 1. Get the list of features that ACTUALLY survived the stats test
    valid_features = set(df_stats['spike_feature'].unique())
    
    data = []
    
    for t in pop_traces:
        # 2. SKIP this trace entirely if it's a "dead" feature
        if t['spike_feature'] not in valid_features:
            continue
            
        groups = t['trace_data'].get('groups', {})
        cell_vals = []
        for v in groups.values():
            arr = v.get('mean', v) if isinstance(v, dict) else v
            if arr is not None:
                cell_vals.append(np.nanmean(arr))
        
        if cell_vals:
            data.append({
                'cell': t['cell_id'], 
                'feat': t['spike_feature'], 
                'val': np.mean(cell_vals)
            })
    
    if not data:
        print("Could not extract feature values for redundancy matrix.")
        return

    df = pd.DataFrame(data).groupby(['cell', 'feat'])['val'].mean().unstack()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        df.corr(method='spearman'), 
        annot=True, 
        fmt=".2f", 
        cmap="mako", 
        vmin=-1, 
        vmax=1,
        linewidths=1,
        linecolor='white'
    )
    
    plt.title("Spike Feature Redundancy (Significant Features Only)", fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()



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
    
    plt.title("Temporal Landscape: When do these LFP relationships occur?", fontweight='bold', fontsize=15, pad=15)
    plt.xlabel("Time relative to spike (seconds)", fontsize=12, fontweight='bold')
    plt.ylabel("LFP Feature", fontsize=12, fontweight='bold')
    
    handles, labels = ax.get_legend_handles_labels()
    num_features = len(master_order)
    handles = handles[:num_features] + [Line2D([0], [0], color='black', ls='--', lw=2)]
    labels = labels[:num_features] + ['Spike (t=0)']
    
    plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', title="Spike Feature", frameon=True, shadow=True)
    plt.grid(axis='x', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.show()