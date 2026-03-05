import os
import sys
import glob
import math
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.patches import Circle
from scipy.stats import pearsonr
from scipy.stats import chi2_contingency
from scipy.stats import kruskal

# ------------------------------------------------------------------------------------------- #
#                                     Environment Setup                                       #
# ------------------------------------------------------------------------------------------- #

# Import metadata file
config_dir = "/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/spikeparam/datasets/spe-1/spe1_helper_modules/"
if config_dir not in sys.path:
    sys.path.append(config_dir)
import config

# ------------------------------------------------------------------------------------------- #
#                                     Aggregate Results                                       #
# ------------------------------------------------------------------------------------------- #

def compile_experiment_results(folder_path):
    """
    Iterates through config Cell IDs. Populates num_clusters (0 if missing).
    """
    all_cell_ids = list(config.DICT_CELL_TYPE.keys())
    master_list = []

    for cell_num in all_cell_ids:
        cell_id_str = f"c{cell_num}"
        search_pattern = os.path.join(folder_path, f"{cell_id_str}_*.pkl")
        matching_files = glob.glob(search_pattern)
        
        if matching_files:
            df = pd.read_pickle(matching_files[0])
            # Count unique clusters in this specific experiment
            # We assume 'groups' contains the cluster IDs
            n_clusters = df['groups'].nunique()
            if n_clusters == 1:
                n_clusters = 2

            # convert to str to convert to categorical variable
            n_clusters = str(n_clusters)
        else:
            # Placeholder for no-pickle cells
            df = pd.DataFrame({
                'feature_clustered': [np.nan],
                'groups': [np.nan],
                'nRMSE': [np.nan],
                'cos_sim': [np.nan]
            })
            n_clusters = 0 # No pickle = 0 clusters

        # Map Metadata
        df['cell_id'] = cell_id_str
        df['num_clusters'] = n_clusters # Add the new count column
        
        patch_info = config.DICT_PATCH_TYPE.get(cell_num)
        if patch_info and ", " in patch_info:
            df['patch_type'], df['current_type'] = patch_info.split(', ')
        else:
            df['patch_type'], df['current_type'] = np.nan, np.nan

        df['cell_type'] = config.DICT_CELL_TYPE.get(cell_num)
        df['cortical_depth'] = config.DICT_CORT_DEPTH.get(cell_num)
        df['dark_neuron'] = config.DICT_DARK_NEURONS.get(cell_num)
        df['clear_EAP_waveform'] = config.DICT_CLEAR_EAP_WAV.get(cell_num)
        
        master_list.append(df)

    final_table = pd.concat(master_list, ignore_index=True)
    
    final_table = final_table.rename(columns={
        'feature_clustered': 'spike_feature',
        'groups': 'cluster'
    })

    # Updated column order including num_clusters
    cols = [
        'cell_id', 'patch_type', 'current_type', 'cell_type', 'cortical_depth', 
        'dark_neuron', 'clear_EAP_waveform', 'spike_feature', 'num_clusters', 
        'cluster', 'nRMSE', 'cos_sim'
    ]

    for c in cols:
        if c not in final_table.columns:
            final_table[c] = np.nan

    return final_table[cols]

def gen_table_fig(df, filename='clust_table_report.png', save_fig=True):
    # 1. Internal Global Stats Calculation
    raw_depth_all = pd.to_numeric(df['cortical_depth'], errors='coerce')
    raw_nrmse_all = pd.to_numeric(df['nRMSE'], errors='coerce')
    raw_cossim_all = pd.to_numeric(df['cos_sim'], errors='coerce')
    
    g_min_d, g_max_d = raw_depth_all.min(), raw_depth_all.max()
    g_min_n, g_max_n = raw_nrmse_all.min(), raw_nrmse_all.max()
    g_min_c, g_max_c = raw_cossim_all.min(), raw_cossim_all.max()

    # 2. Formatting and Numerical Sorting (c1, c2, c3... c46)
    cols_order = [
        'cell_id', 'patch_type', 'current_type', 'cell_type', 'cortical_depth',
        'dark_neuron', 'clear_EAP_waveform', 'spike_feature','num_clusters', 'cluster', 'nRMSE', 'cos_sim'
    ]
    df_copy = df.copy()
    # Sort numerically (c1, c2, c10...)
    df_copy['sort_idx'] = df_copy['cell_id'].str.extract('(\d+)').astype(int)
    plot_data = df_copy.sort_values(by=['sort_idx', 'spike_feature']).drop(columns=['sort_idx'])[cols_order].copy()
    
    # Prep display strings for the table
    plot_data['nRMSE'] = pd.to_numeric(plot_data['nRMSE'], errors='coerce').map(lambda x: f'{x:.3f}' if pd.notnull(x) else '')
    plot_data['cos_sim'] = pd.to_numeric(plot_data['cos_sim'], errors='coerce').map(lambda x: f'{x:.3f}' if pd.notnull(x) else '')
    plot_data['cortical_depth'] = pd.to_numeric(plot_data['cortical_depth'], errors='coerce').map(lambda x: f'{x:.1f}' if pd.notnull(x) else '')

    # 3. Fixed Family Color Map
    feature_shades = {
        'peak_amp': '#8c564b', 'peak_sharpness': '#a06d62', 'peak_width': '#b38479',
        'exp_const': '#e377c2', 'exp_lambda': '#c561a8',
        'inflection_amp': '#d62728', 'inflection_time': '#e05354',
        'ramp_amp': '#ff7f0e', 'log_isi': '#7f7f7f'
    }

    # 4. Setup Figure
    headers = [c.replace('_', ' ').title() for c in plot_data.columns]
    # FIX: Shifted header indices to 10 and 11
    headers[6], headers[10], headers[11] = "Clear EAP\nWaveform", "nRMSE", "Cos Sim"
    
    fig_height = len(plot_data) * 0.6 + 2
    fig, ax = plt.subplots(figsize=(22, fig_height))
    ax.axis('off')
    table = ax.table(cellText=plot_data.values, colLabels=headers, cellLoc='center', loc='center')

    # 5. Merging Logic & Selective Coloring
    start_row = 1
    for i in range(1, len(plot_data) + 1):
        is_cell_end = (i == len(plot_data) or plot_data.iloc[i]['cell_id'] != plot_data.iloc[start_row-1]['cell_id'])
        
        if is_cell_end:
            end_row = i
            # Seamless Metadata merge (Cols 0-6)
            for c in range(7):
                for r in range(start_row, end_row + 1):
                    cell = table[r, c]
                    if r != start_row: cell.get_text().set_text("")
                    
                    # Remove horizontal lines within merged blocks
                    if start_row == end_row: cell.visible_edges = 'closed'
                    elif r == start_row: cell.visible_edges = 'LRT'
                    elif r == end_row: cell.visible_edges = 'LRB'
                    else: cell.visible_edges = 'LR'
                    
                    cell.get_text().set_verticalalignment('center')
                    if c == 0: cell.get_text().set_weight('bold')
                    
                    # SHADE DEPTH: First row only
                    if c == 4 and r == start_row:
                        val = raw_depth_all.iloc[start_row-1]
                        if pd.notnull(val) and g_max_d != g_min_d:
                            norm = (val - g_min_d) / (g_max_d - g_min_d)
                            cell.set_facecolor(mcolors.to_hex(plt.cm.YlGn(0.1 + norm * 0.4)))

            # Seamless Feature merge (Col 7)
            feat_start = start_row
            for j in range(start_row, end_row + 1):
                curr_feat = plot_data.iloc[j-1]['spike_feature']
                if j == end_row or plot_data.iloc[j]['spike_feature'] != curr_feat:
                    shade = feature_shades.get(curr_feat, 'white')
                    brightness = sum(mcolors.to_rgb(shade)) / 3
                    t_color = 'white' if brightness < 0.55 else 'black'
                    
                    for r_f in range(feat_start, j + 1):
                        cell_f = table[r_f, 7]
                        if r_f != feat_start: cell_f.get_text().set_text("") 
                        
                        # SHADE FEATURE: Only first row of block to avoid artifacts
                        if r_f == feat_start:
                            cell_f.set_facecolor(shade)
                            cell_f.get_text().set_color(t_color)
                        
                        cell_f.get_text().set_weight('bold')
                        cell_f.get_text().set_verticalalignment('center')
                        
                        # Remove horizontal lines within feature block
                        if feat_start == j: cell_f.visible_edges = 'closed'
                        elif r_f == feat_start: cell_f.visible_edges = 'LRT'
                        elif r_f == j: cell_f.visible_edges = 'LRB'
                        else: cell_f.visible_edges = 'LR'
                    feat_start = j + 1

            # Metric Gradients (Cols 9-11)
            for r in range(start_row, end_row + 1):
                n_val = raw_nrmse_all.iloc[r-1]
                c_val = raw_cossim_all.iloc[r-1]
                
                # nRMSE (Col 10): Darker = Larger (Global Bad)
                # FIX: Shifted to index 10
                if pd.notnull(n_val) and g_max_n != g_min_n:
                    n_norm = (n_val - g_min_n) / (g_max_n - g_min_n)
                    table[r, 10].set_facecolor(mcolors.to_hex(plt.cm.Oranges(0.05 + n_norm * 0.4)))
                
                # Cos Sim (Col 11): Darker = Smaller (Global Bad)
                # FIX: Shifted to index 11
                if pd.notnull(c_val) and g_max_c != g_min_c:
                    c_norm = (g_max_c - c_val) / (g_max_c - g_min_c)
                    table[r, 11].set_facecolor(mcolors.to_hex(plt.cm.Blues(0.05 + c_norm * 0.4)))
                
                # Cluster background shading
                # FIX: Shifted to index 9
                table[r, 9].set_facecolor('#F8F9FA') 
            
            start_row = i + 1

    # 6. Global Polish
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3.5)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor('#40466e')
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')

    if save_fig: plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()
# ------------------------------------------------------------------------------------------- #
# ------------------------------ Analyze metadata results ------------------------------ #
# ------------------------------------------------------------------------------------------- #

def analyze_waveform_variance(df, N=20):
    """
    Ranks cells by waveform change, plots the selection logic, 
    and returns the intersection of top-tier results.
    """
    # 1. Selection Logic
    df_ranked_comb = df.sort_values(by=['nRMSE', 'cos_sim'], ascending=[False, True])
    df_ranked_nrmse = df.sort_values(by=['nRMSE'], ascending=[False])
    df_ranked_cosim = df.sort_values(by=['cos_sim'], ascending=[True])

    top_n_comb = df_ranked_comb.head(N)
    top_n_nrmse = df_ranked_nrmse.head(N)
    top_n_cosim = df_ranked_cosim.head(N)

    # Intersection: In Both (nRMSE & CosSim) AND in the Combined Sort
    both_indices = df.index[df.index.isin(top_n_nrmse.index) & df.index.isin(top_n_cosim.index)]
    final_targets = df.loc[both_indices[both_indices.isin(top_n_comb.index)]].copy()
    final_targets = final_targets.sort_values(['nRMSE', 'cos_sim'], ascending=[False, True])

    # 2. Setup Figure
    fig, (ax_plot, ax_list) = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [2, 1]})

    # --- LEFT: THE PLOT ---
    sns.scatterplot(data=df, x='nRMSE', y='cos_sim', color='darkgrey', alpha=0.3, s=40, ax=ax_plot, label='other cell-feature groups')
    
    # Selection Markers
    sns.scatterplot(data=df.loc[both_indices], x='nRMSE', y='cos_sim', 
                    color='purple', s=120, marker='X', label=f'Top {N} in Both', ax=ax_plot)
    sns.scatterplot(data=top_n_comb, x='nRMSE', y='cos_sim', 
                    facecolor='none', edgecolor='gold', s=200, linewidth=1.5, label=f'Top {N} Combined', ax=ax_plot)

    ax_plot.set_title(f"Waveform Variance Landscape (N={N})", fontsize=14)
    ax_plot.set_xlabel("nRMSE (Amplitude Variance)")
    ax_plot.set_ylabel("Cos Sim (Shape Similarity)")
    ax_plot.grid(True, linestyle='--', alpha=0.2)
    ax_plot.legend(loc='upper right')

    # --- RIGHT: THE IDENTITY LIST ---
    ax_list.axis('off')
    title_text = f"INTERSECTION IDENTITIES (N={N})\n"
    header = f"{'Cell ID':<15} | {'Spike Feature':<15}\n"
    separator = "-" * 35 + "\n"

    list_content = ""
    for _, row in final_targets.iterrows():
        list_content += f"{str(row['cell_id']):<15} | {str(row['spike_feature']):<15}\n"

    ax_list.text(0, 1, title_text + header + separator + list_content, 
                 family='monospace', fontsize=10, verticalalignment='top')

    plt.tight_layout()
    plt.show()

    return final_targets

def analyze_cross_correlations(df, alpha=0.05):
    # 1. Define Groups
    metadata_cols = ['patch_type', 'current_type', 'cell_type', 'cortical_depth', 'dark_neuron', 'clear_EAP_waveform']
    feature_cols = ['num_clusters', 'nRMSE', 'cos_sim']
    all_cols = metadata_cols + feature_cols
    
    # 2. Encode and Clean
    df_sub = df[all_cols].copy()
    cat_feats = ['patch_type', 'current_type', 'cell_type', 'dark_neuron', 'clear_EAP_waveform', 'num_clusters']
    for col in cat_feats:
        df_sub[col] = df_sub[col].astype('category').cat.codes
    df_clean = df_sub.dropna()
    
    # 3. Calculate Correlation and P-values
    corr_matrix = df_clean.corr()
    n_total = len(all_cols)
    p_values = np.ones((n_total, n_total))
    significant_cross_pairs = []
    
    for i in range(n_total):
        for j in range(i + 1, n_total):
            col_a, col_b = all_cols[i], all_cols[j]
            r, p = pearsonr(df_clean[col_a], df_clean[col_b])
            p_values[i, j] = p
            p_values[j, i] = p
            
            is_cross = (col_a in metadata_cols and col_b in feature_cols) or \
                       (col_b in metadata_cols and col_a in feature_cols)
            
            if is_cross and p < alpha:
                significant_cross_pairs.append({
                    'Metadata': col_a if col_a in metadata_cols else col_b,
                    'Feature': col_b if col_b in feature_cols else col_a,
                    'Pearson $r$': round(r, 3),
                    'p-value': f"{p:.2e}",
                    'Significance': '***' if p < 0.001 else '**' if p < 0.01 else '*'
                })

    # 4. Slicing for Plotting
    corr_sliced = corr_matrix.iloc[1:, :-1]
    p_sliced = p_values[1:, :-1]
    mask = np.triu(np.ones_like(corr_sliced, dtype=bool), k=1)

    # 5. Plotting
    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(
        corr_sliced, mask=mask, cmap='PRGn', center=0, 
        square=True, linewidths=.5, annot=False,
        cbar_kws={"label": "Pearson Correlation ($r$)"}
    )

    row_names = corr_sliced.index.tolist()
    col_names = corr_sliced.columns.tolist()

    for i in range(len(row_names)):
        for j in range(len(col_names)):
            if not mask[i, j]:
                r_val = corr_sliced.iloc[i, j]
                p_val = p_sliced[i, j]
                row_feat = row_names[i]
                col_feat = col_names[j]
                
                # Dynamic Text Color: White for dark colors, Black for light colors
                text_color = "white" if abs(r_val) > 0.45 else "black"
                
                is_cross = (row_feat in metadata_cols and col_feat in feature_cols) or \
                           (col_feat in metadata_cols and row_feat in feature_cols)
                
                stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                
                ax.text(j + 0.5, i + 0.35, stars, ha='center', va='center', 
                        color=text_color, fontsize=14, fontweight='bold')
                ax.text(j + 0.5, i + 0.65, f"{r_val:.2f}", ha='center', va='center', 
                        color=text_color, fontsize=11)
                
                if is_cross and p_val < alpha:
                    # Circle significant cross-pairs
                    circle_color = "white" if abs(r_val) > 0.7 else "black"
                    circle = Circle((j + 0.5, i + 0.5), 0.44, color=circle_color, fill=False, linewidth=2.5)
                    ax.add_patch(circle)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return pd.DataFrame(significant_cross_pairs).sort_values('Pearson $r$', key=abs, ascending=False).reset_index(drop=True)

def plot_sig_feat_pairs(df, sig_pairs_df):
    # Nuke the warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)
    
    sns.set_theme(style="ticks")
    
    n_plots = len(sig_pairs_df)
    cols = 3
    rows = math.ceil(n_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten() 

    for i, (_, row) in enumerate(sig_pairs_df.iterrows()):
        m, f = row['Metadata'], row['Feature']
        r, p = float(row['Pearson $r$']), float(row['p-value'])
        stars = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
        ax = axes[i]

        plot_data = df[[m, f]].replace([np.inf, -np.inf], np.nan).dropna()
        if plot_data.empty: continue
        
        # 1. HORIZONTAL BOXPLOTS for Num Clusters
        if 'num_clusters' in [m, f]:
            cat_col = 'num_clusters'
            val_col = f if m == 'num_clusters' else m
            plot_data[cat_col] = plot_data[cat_col].astype(float).astype(str)
            order = ['0.0', '2.0', '3.0']
            
            sns.boxplot(data=plot_data, y=cat_col, x=val_col, palette="Paired", 
                        order=order, showfliers=False, orient='h', ax=ax)
            sns.stripplot(data=plot_data, y=cat_col, x=val_col, color=".3", 
                          alpha=.3, order=order, orient='h', ax=ax)
            
            x_min, x_max = plot_data[val_col].min(), plot_data[val_col].max()
            ax.set_xlim(x_min - (x_max - x_min) * 0.1, x_max + (x_max - x_min) * 0.1)
            
            # Explicit Labels
            ax.set_xlabel(val_col.replace('_', ' ').title())
            ax.set_ylabel(cat_col.replace('_', ' ').title())

        # 2. CONTINUOUS REGRESSION
        elif plot_data[m].nunique() > 5:
            x_num = pd.to_numeric(plot_data[m], errors='coerce')
            y_num = pd.to_numeric(plot_data[f], errors='coerce')
            ax.scatter(x_num, y_num, alpha=0.3, color='purple')
            
            idx = np.isfinite(x_num) & np.isfinite(y_num)
            m_slope, b_int = np.polyfit(x_num[idx], y_num[idx], 1)
            ax.plot(x_num, m_slope*x_num + b_int, color='darkblue', lw=2)
            
            y_min, y_max = y_num.min(), y_num.max()
            ax.set_ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)
            
            # Explicit Labels
            ax.set_xlabel(m.replace('_', ' ').title())
            ax.set_ylabel(f.replace('_', ' ').title())

        # 3. OTHER CATEGORICAL
        else:
            sns.boxplot(data=plot_data, x=m, y=f, palette="Paired", showfliers=False, ax=ax)
            sns.stripplot(data=plot_data, x=m, y=f, color=".3", alpha=.3, ax=ax)
            
            y_min, y_max = plot_data[f].min(), plot_data[f].max()
            ax.set_ylim(y_min - (y_max - y_min) * 0.1, y_max + (y_max - y_min) * 0.1)
            
            # Explicit Labels
            ax.set_xlabel(m.replace('_', ' ').title())
            ax.set_ylabel(f.replace('_', ' ').title())

        ax.set_title(f"{stars} (r={r:.2f})", fontweight='bold', fontsize=12)
        sns.despine(ax=ax)

    # Cleanup unused axes
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------------------- #
# ------------------------------ Analyze spk feat results ------------------------------ #
# ------------------------------------------------------------------------------------------- #

def quantify_spk_feature_prevalence(df):
    cols_to_fix = ['num_clusters', 'cos_sim', 'nRMSE']
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    # Calculate unique cells per feature
    feature_counts = df.groupby('spike_feature')['cell_id'].nunique().reset_index()
    feature_counts.columns = ['spike_feature', 'n_cells_with_feature']
    
    # Calculate prevalence based on the total unique cells in your current master df
    total_cells = df['cell_id'].nunique()
    feature_counts['prevalence_pct'] = (feature_counts['n_cells_with_feature'] / total_cells) * 100
    
    # Merge with cluster metrics
    metrics = df.groupby('spike_feature').agg({
        'num_clusters': 'mean',
        'cos_sim': 'mean',
        'nRMSE': 'mean'
    }).reset_index()
    
    return feature_counts.merge(metrics, on='spike_feature').sort_values('prevalence_pct', ascending=False)

def plot_aggregated_spike_feat(raw_df):
    # 1. Internal Aggregation: Calculate Means and 95% CI
    stats = raw_df.groupby('spike_feature').agg({
        'nRMSE': ['mean', 'std', 'count'],
        'cos_sim': ['mean', 'std', 'count'],
        'num_clusters': 'mean'
    })
    
    # Flatten columns
    stats.columns = ['nRMSE', 'nRMSE_std', 'nRMSE_n', 'cos_sim', 'cos_sim_std', 'cos_sim_n', 'num_clusters']
    stats = stats.reset_index()
    
    # Calculate the 95% Confidence Interval arms
    stats['nRMSE_ci'] = 1.96 * (stats['nRMSE_std'] / np.sqrt(stats['nRMSE_n'])).fillna(0)
    stats['cos_sim_ci'] = 1.96 * (stats['cos_sim_std'] / np.sqrt(stats['cos_sim_n'])).fillna(0)

    # Prevalence calculation
    total_cells = raw_df['cell_id'].nunique()
    stats['prevalence_pct'] = (stats['nRMSE_n'] / total_cells) * 100

    # 2. Setup Plot
    plt.figure(figsize=(11, 7))
    sns.set_theme(style="ticks")

    # 3. Draw the 95% CI Crosses
    plt.errorbar(
        x=stats.nRMSE, 
        y=stats.cos_sim, 
        xerr=stats.nRMSE_ci, 
        yerr=stats.cos_sim_ci,
        fmt='none', 
        ecolor='#5e5e5e', 
        elinewidth=1.2, 
        capsize=3, 
        alpha=0.6, 
        zorder=1
    )

    # 4. Draw Main Bubbles
    scatter = sns.scatterplot(
        data=stats,
        x='nRMSE', 
        y='cos_sim',
        size='prevalence_pct',
        hue='num_clusters',
        sizes=(40, 400),
        palette='viridis',
        alpha=0.9,
        edgecolor='black',
        linewidth=1,
        zorder=2
    )

    # 5. Annotations
    for i, row in stats.iterrows():
        plt.text(
            row['nRMSE'] + 0.003, 
            row['cos_sim'] + 0.001, 
            row['spike_feature'], 
            fontsize=9, fontweight='semibold', va='bottom'
        )

    # 6. Final Polish
    plt.title('Spike Feature Aggregated Analysis (Mean ± 95% CI)', fontsize=15, fontweight='bold', pad=20)
    plt.xlabel('$\longrightarrow$ Higher difference in waveforms (nRMSE)', fontsize=11)
    plt.ylabel('$\longleftarrow$ Higher difference in shape morphology (Cosine Similarity)', fontsize=11)
    
    plt.ylim(stats['cos_sim'].min() - 0.03, 1.01)
    plt.xlim(-0.005, stats['nRMSE'].max() + 0.03)
    
    plt.legend(title='Clusters & % Prevalence', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine()
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_depth_distribution(df):
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="ticks")

    # 1. Sort features by median depth so the Y-axis follows the anatomy
    sorted_features = df.groupby('spike_feature')['cortical_depth'].median().sort_values().index
    
    # 2. Create a "Deeper is Darker" palette
    palette = sns.color_palette("YlGnBu", n_colors=len(sorted_features))

    # 3. Plot Boxplot (The Range)
    sns.boxplot(
        data=df, 
        x='cortical_depth', 
        y='spike_feature', 
        order=sorted_features,
        whis=[5, 95], 
        palette=palette,
        showfliers=False,
        boxprops=dict(alpha=0.2, edgecolor='none'), 
        whiskerprops=dict(color='gray', alpha=0.5),
        capprops=dict(color='gray', alpha=0.5),
        medianprops=dict(color='red', alpha=0.8, linewidth=2), 
        width=0.4
    )

    # 4. Plot Stripplot (The Raw Data)
    sns.stripplot(
        data=df, 
        x='cortical_depth', 
        y='spike_feature', 
        order=sorted_features,
        hue='spike_feature',
        palette=palette,
        jitter=0.25, 
        alpha=0.5, 
        s=7, 
        legend=False
    )

    # 5. Anatomical Formatting
    plt.title('Cortical Depth Distribution of clustered spike features', fontsize=16, fontweight='bold', pad=25)
    plt.xlabel('Cortical Depth ($\mu m$)', fontsize=13, fontweight='semibold')
    plt.ylabel('Clustered Spike Features', fontsize=13, fontweight='semibold')
    
    # 6. Clean up
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()

def plot_meta_spk_feature_dependency(df):
    # 1. Identify Categorical Metadata (excluding metrics and keys)
    exclude = ['spike_feature', 'num_clusters', 'cos_sim', 'nRMSE', 'cell_id', 'cortical_depth', 'nRMSE_std', 'cos_sim_std', 'cluster']
    meta_cols = [c for c in df.columns if c not in exclude]
    
    # 2. Calculate Prevalence (%) for Categorical Metadata
    cat_data = []
    group_sizes = [] # Track sizes for the borders
    for col in meta_cols:
        counts = df.groupby(['spike_feature', col]).size().unstack(fill_value=0)
        prevalence = (counts / counts.sum()) * 100
        prevalence.columns = [f"{col}: {c}" for c in prevalence.columns]
        cat_data.append(prevalence)
        group_sizes.append(len(prevalence.columns)) 
    
    # 3. Calculate Mean Depth for Sorting and the Depth Column
    depth_stats = df.groupby('spike_feature')['cortical_depth'].mean().to_frame()
    depth_stats.columns = ['Avg_Depth_um']

    # 4. Merge and Sort by Depth
    master_matrix = pd.concat(cat_data + [depth_stats], axis=1).fillna(0)
    master_matrix = master_matrix.sort_values(by='Avg_Depth_um')

    # 5. Plotting 
    fig_height = max(8, len(master_matrix) * 0.6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, fig_height), 
                                   gridspec_kw={'width_ratios': [8, 1]})
    
    # Categorical Prevalence Heatmap (Purples color scheme)
    sns.heatmap(master_matrix.drop(columns=['Avg_Depth_um']), 
                annot=True, fmt=".1f", cmap="Purples", ax=ax1, 
                cbar_kws={'label': 'Group Prevalence (%)'},
                linewidths=0.5)
    
    # Depth Heatmap (Deeper = Darker using "YlGnBu")
    sns.heatmap(master_matrix[['Avg_Depth_um']], 
                annot=True, fmt=".0f", cmap="YlGnBu", ax=ax2, 
                cbar_kws={'label': 'Depth (um)'},
                linewidths=0.5)

    # --- THE BORDER ADDITION ---
    current_col = 0
    for size in group_sizes:
        # Draw a thick rectangle around the group
        rect = patches.Rectangle(
            (current_col, 0), size, len(master_matrix), 
            linewidth=4, edgecolor='black', facecolor='none', zorder=10
        )
        ax1.add_patch(rect)
        current_col += size

    ax1.set_title('Spike Feature Prevalence by Metadata (%)', fontweight='bold', fontsize=14, pad=15)
    ax2.set_title('Mean Depth', fontweight='bold', fontsize=14, pad=15)
    ax1.set_ylabel('Spike Features (Sorted by Depth)', fontsize=12)
    ax1.set_xlabel('Metadata Categories', fontsize=12)
    
    plt.tight_layout()
    plt.show()



def stat_test_depth_stratification(df):
    """
    Tests if spike features live at different depths without external post-hoc libs.
    """
    # Group depths by feature, dropping NaNs
    groups = {name: group['cortical_depth'].dropna().values 
              for name, group in df.groupby('spike_feature')}
    
    # 1. Global Kruskal-Wallis Test
    stat, p = kruskal(*groups.values())
    
    print("-" * 40)
    print(f"DEPTH STRATIFICATION ANALYSIS")
    print("-" * 40)
    print(f"Kruskal-Wallis H-stat: {stat:.3f}")
    print(f"p-value: {p:.3e}")
    
    if p < 0.05:
        print("\nSignificant differences found across cortical layers.")
    else:
        print("\nNo significant depth stratification found.")
    print("-" * 40)
    
    return p



def stat_test_metadata_dependency(df):
    """
    Runs Chi-Square tests for all metadata categories to see if they 
    influence which spike features appear.
    """
    exclude = ['spike_feature', 'num_clusters', 'cos_sim', 'nRMSE', 'cell_id', 
               'cortical_depth', 'nRMSE_std', 'cos_sim_std', 'cluster']
    meta_cols = [c for c in df.columns if c not in exclude]
    
    print("-" * 40)
    print(f"METADATA DEPENDENCY ANALYSIS (Chi-Square)")
    print("-" * 40)
    
    results = []
    for col in meta_cols:
        # Create the contingency table (Counts of Feature vs Metadata Category)
        contingency = pd.crosstab(df['spike_feature'], df[col])
        
        # Test if the rows (features) and columns (metadata) are independent
        chi2, p, dof, expected = chi2_contingency(contingency)
        
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{col:<20} | p = {p:.3e} ({sig})")
        
        results.append({'metadata': col, 'p_value': p, 'sig': sig})
        
    print("-" * 40)
    return pd.DataFrame(results)
