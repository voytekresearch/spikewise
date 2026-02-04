import os
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import pandas as pd
import numpy as np
import seaborn as sns
import glob
from scipy.stats import pearsonr


#import metadata file
config_dir = "/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/spikeparam/datasets/spe-1/spe1_helper_modules/"
if config_dir not in sys.path:
    sys.path.append(config_dir)
import config


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
            if n_clusters ==1:
                n_clusters =2
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
        'dark_neuron', 'clear_EAP_waveform', 'spike_feature', 'cluster', 'nRMSE', 'cos_sim'
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
    headers[6], headers[9], headers[10] = "Clear EAP\nWaveform", "nRMSE", "Cos Sim"
    
    fig_height = len(plot_data) * 0.6 + 2
    fig, ax = plt.subplots(figsize=(22, fig_height))
    ax.axis('off')
    table = ax.table(cellText=plot_data.values, colLabels=headers, cellLoc='center', loc='center')

    # 5. Merging Logic & Selective Coloring (Cols 0-7)
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

            # Metric Gradients (Cols 9-10)
            for r in range(start_row, end_row + 1):
                n_val = raw_nrmse_all.iloc[r-1]
                c_val = raw_cossim_all.iloc[r-1]
                
                # nRMSE: Darker = Larger (Global Bad)
                if pd.notnull(n_val) and g_max_n != g_min_n:
                    n_norm = (n_val - g_min_n) / (g_max_n - g_min_n)
                    table[r, 9].set_facecolor(mcolors.to_hex(plt.cm.Oranges(0.05 + n_norm * 0.4)))
                
                # Cos Sim: Darker = Smaller (Global Bad)
                if pd.notnull(c_val) and g_max_c != g_min_c:
                    c_norm = (g_max_c - c_val) / (g_max_c - g_min_c)
                    table[r, 10].set_facecolor(mcolors.to_hex(plt.cm.Blues(0.05 + c_norm * 0.4)))
                
                table[r, 8].set_facecolor('#F8F9FA') 
            
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
    sns.scatterplot(data=df, x='nRMSE', y='cos_sim', color='#e0e0e0', alpha=0.3, s=40, ax=ax_plot, label='other cell-feature groups')
    
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
    cat_feats = ['patch_type', 'current_type', 'cell_type', 'dark_neuron', 'clear_EAP_waveform']
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
            
            # Identify Cross-Group Logic: One is Metadata, one is Feature
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

    # 4. Slicing for Plotting (Matches your layout: removes row 0 and last col)
    # Heatmap Rows: current_type ... cos_sim (indices 1 to 8)
    # Heatmap Cols: patch_type ... nRMSE (indices 0 to 7)
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
                
                # Check for Cross-Group Highlight
                is_cross = (row_feat in metadata_cols and col_feat in feature_cols) or \
                           (col_feat in metadata_cols and row_feat in feature_cols)
                
                # Add Stars and R-values
                stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                ax.text(j + 0.5, i + 0.35, stars, ha='center', va='center', color='black', fontsize=14, fontweight='bold')
                ax.text(j + 0.5, i + 0.65, f"{r_val:.2f}", ha='center', va='center', color='black', fontsize=11)
                
                # Circle only SIGNIFICANT CROSS-GROUP pairs
                if is_cross and p_val < alpha:
                    circle = Circle((j + 0.5, i + 0.5), 0.42, color='black', fill=False, linewidth=2.5)
                    ax.add_patch(circle)

    plt.xticks(rotation=45, ha='right')
    plt.title(f"Cross-Correlation Matrix: Exp Metadata vs. Clustering Waveform Features ($p < {alpha}$)", fontsize=16)
    plt.tight_layout()
    plt.show()

    # 6. Return Table
    return pd.DataFrame(significant_cross_pairs).sort_values('Pearson $r$', key=abs, ascending=False).reset_index(drop=True)

