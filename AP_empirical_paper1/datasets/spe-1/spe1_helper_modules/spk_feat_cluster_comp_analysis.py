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
from scipy.stats import pearsonr, spearmanr, chi2_contingency, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from pop_ridge_utils import _stars

# ------------------------------------------------------------------------------------------- #
#                                     Environment Setup                                       #
# ------------------------------------------------------------------------------------------- #

# Import metadata file
config_dir = "/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/spikeparam/AP_empirical_paper1/datasets/spe-1/spe1_helper_modules/"
if config_dir not in sys.path:
    sys.path.append(config_dir)
import config

# ============================================================
# Global plot style — cartoony / presentation-ready
# ============================================================
sns.set_theme(style='ticks', font_scale=1.4, rc={
    'axes.linewidth':    2.0,
    'xtick.major.width': 2.0,
    'ytick.major.width': 2.0,
    'xtick.major.size':  6,
    'ytick.major.size':  6,
    'patch.linewidth':   2.0,
    'lines.linewidth':   2.0,
    'figure.titlesize':  22,
})

# ============================================================
# Publication style constants
# ============================================================
_FS_SM    = 14   # small annotations, tick labels
_FS_AX    = 16   # axis labels
_FS_SUB   = 18   # subplot titles
_FS_TTL   = 22   # figure suptitles

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



# ------------------------------------------------------------------------------------------- #
#                                     Aggregate Results                                       #
# ------------------------------------------------------------------------------------------- #

def _index_experiment_pickles(folder_path):
    """Map each cell id to its cluster report pickle (c{N}_cluster_report.pkl)."""
    indexed = {}
    for path in sorted(glob.glob(os.path.join(folder_path, "c*_cluster_report.pkl"))):
        cell_id = os.path.basename(path).split("_", 1)[0]
        indexed[cell_id] = path
    return indexed

def compile_experiment_results(folder_path):
    """
    Iterates through config Cell IDs. Populates num_clusters (0 if missing).
    """
    all_cell_ids = list(config.DICT_CELL_TYPE.keys())
    master_list = []
    indexed_pickles = _index_experiment_pickles(folder_path)

    for cell_num in all_cell_ids:
        cell_id_str = f"c{cell_num}"
        pickle_path = indexed_pickles.get(cell_id_str)
        
        if pickle_path:
            df = pd.read_pickle(pickle_path)

        if pickle_path and 'groups' in df.columns and not df.empty:
            n_clusters = df['groups'].nunique()
            if n_clusters == 1:
                n_clusters = 2
            n_clusters = str(n_clusters)
        else:
            # Placeholder for no-pickle cells
            df = pd.DataFrame({
                'feature_clustered': [np.nan],
                'groups': [np.nan],
                'nRMSE': [np.nan],
                'cos_sim': [np.nan],
                'temporal_rho': [np.nan],
                'temporal_p': [np.nan]
            })
            n_clusters = '0'  # No pickle = 0 clusters; string to match real cells

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

    # Updated column order including num_clusters and temporal metrics
    cols = [
        'cell_id', 'patch_type', 'current_type', 'cell_type', 'cortical_depth',
        'dark_neuron', 'clear_EAP_waveform', 'spike_feature', 'num_clusters',
        'cluster', 'nRMSE', 'cos_sim', 'temporal_rho', 'temporal_p', 'temporal_component'
    ]

    for c in cols:
        if c not in final_table.columns:
            final_table[c] = np.nan

    # Ensure numeric columns have correct dtype (mixed-type concat can produce object columns)
    for num_col in ['nRMSE', 'cos_sim', 'temporal_rho', 'temporal_p', 'cortical_depth']:
        final_table[num_col] = pd.to_numeric(final_table[num_col], errors='coerce')

    # Binary flag: 1 if time dependence is moderate-or-stronger (|ρ| ≥ 0.3; Cohen 1988)
    final_table['temporal_component'] = (
        pd.to_numeric(final_table['temporal_rho'], errors='coerce').abs() >= 0.3
    ).astype(float)

    return final_table[cols]

def gen_table_fig(df, filename='clust_table_report.png', save_fig=True):
    # 1. Internal Global Stats Calculation
    raw_depth_all = pd.to_numeric(df['cortical_depth'], errors='coerce')
    raw_nrmse_all = pd.to_numeric(df['nRMSE'], errors='coerce')
    raw_cossim_all = pd.to_numeric(df['cos_sim'], errors='coerce')
    raw_trho_all  = pd.to_numeric(df['temporal_rho'], errors='coerce')

    g_min_d, g_max_d = raw_depth_all.min(), raw_depth_all.max()
    g_min_n, g_max_n = raw_nrmse_all.min(), raw_nrmse_all.max()
    g_min_c, g_max_c = raw_cossim_all.min(), raw_cossim_all.max()
    g_min_t, g_max_t = raw_trho_all.min(), raw_trho_all.max()

    # 2. Formatting and Numerical Sorting (c1, c2, c3... c46)
    cols_order = [
        'cell_id', 'patch_type', 'current_type', 'cell_type', 'cortical_depth',
        'dark_neuron', 'clear_EAP_waveform', 'spike_feature', 'num_clusters', 'cluster', 'nRMSE', 'cos_sim', 'temporal_rho'
    ]
    df_copy = df.copy()
    # Sort numerically (c1, c2, c10...)
    df_copy['sort_idx'] = df_copy['cell_id'].str.extract('(\d+)').astype(int)
    plot_data = df_copy.sort_values(by=['sort_idx', 'spike_feature']).drop(columns=['sort_idx'])[cols_order].copy()
    
    # Prep display strings for the table
    plot_data['nRMSE'] = pd.to_numeric(plot_data['nRMSE'], errors='coerce').map(lambda x: f'{x:.3f}' if pd.notnull(x) else '')
    plot_data['cos_sim'] = pd.to_numeric(plot_data['cos_sim'], errors='coerce').map(lambda x: f'{x:.3f}' if pd.notnull(x) else '')
    plot_data['cortical_depth'] = pd.to_numeric(plot_data['cortical_depth'], errors='coerce').map(lambda x: f'{x:.1f}' if pd.notnull(x) else '')
    plot_data['temporal_rho'] = pd.to_numeric(plot_data['temporal_rho'], errors='coerce').map(lambda x: f'{x:.3f}' if pd.notnull(x) else '')

    # 3. Fixed Family Color Map
    feature_shades = {
        'peak_amp': '#8c564b', 'peak_sharpness': '#a06d62', 'peak_width': '#b38479',
        'exp_const': '#e377c2', 'exp_lambda': '#c561a8',
        'inflection_amp': '#D55E00', 'inflection_time': '#D55E00',
        'ramp_amp': '#D55E00', 'log_isi': '#7f7f7f'
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

            # Metric Gradients (Cols 9-12)
            for r in range(start_row, end_row + 1):
                n_val = raw_nrmse_all.iloc[r-1]
                c_val = raw_cossim_all.iloc[r-1]
                t_val = raw_trho_all.iloc[r-1]

                # nRMSE (Col 10): Darker = Larger
                if pd.notnull(n_val) and g_max_n != g_min_n:
                    n_norm = (n_val - g_min_n) / (g_max_n - g_min_n)
                    table[r, 10].set_facecolor(mcolors.to_hex(plt.cm.Oranges(0.05 + n_norm * 0.4)))

                # Cos Sim (Col 11): Darker = Smaller
                if pd.notnull(c_val) and g_max_c != g_min_c:
                    c_norm = (g_max_c - c_val) / (g_max_c - g_min_c)
                    table[r, 11].set_facecolor(mcolors.to_hex(plt.cm.Blues(0.05 + c_norm * 0.4)))

                # Temporal Rho (Col 12): diverging — negative=cool, positive=warm
                if pd.notnull(t_val) and g_max_t != g_min_t:
                    t_norm = (t_val - g_min_t) / (g_max_t - g_min_t)
                    table[r, 12].set_facecolor(mcolors.to_hex(plt.cm.RdBu_r(t_norm)))

                # Cluster background shading
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

    ax_plot.set_title(f"Waveform Variance Landscape (N={N})", fontsize=15)
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
                 family='monospace', fontsize=12, verticalalignment='top')

    plt.tight_layout()
    plt.show()

    return final_targets

def analyze_cross_correlations(df, alpha=0.05, n_bootstrap=1000):
    """
    Test associations between recording metadata and spike cluster difference metrics.

    Variable types determine the test:
      Binary categorical (patch_type, current_type, dark_neuron, clear_EAP_waveform)
          → Mann-Whitney U,  effect = rank-biserial r
      Multi-level categorical (cell_type)
          → Kruskal-Wallis,  effect = η²
      Continuous (cortical_depth)
          → Spearman ρ

    FDR correction: Benjamini-Hochberg across all cross-pairs.
    Bootstrap CIs: n_bootstrap resamples (rows = cells) for each effect size.

    Tests run per spike_feature — one row per (cell, spike_feature) pair so n
    reflects cells where that specific feature clustered, preserving
    feature-specific biological interpretation without averaging across features.
    """
    binary_cat  = ['patch_type', 'current_type', 'dark_neuron', 'clear_EAP_waveform']
    multi_cat   = ['cell_type']
    cont_meta   = ['cortical_depth']
    metadata_cols = binary_cat + multi_cat + cont_meta
    feature_cols  = ['num_clusters', 'nRMSE', 'cos_sim', 'temporal_rho']

    # Convert feature cols to numeric (num_clusters stored as str)
    df_work = df.copy()
    for c in feature_cols:
        df_work[c] = pd.to_numeric(df_work[c], errors='coerce')

    # One row per (cell_id, spike_feature): metadata is constant per cell,
    # metrics reflect that specific spike feature's clustering.
    agg = {c: 'first' for c in metadata_cols}
    agg.update({c: 'first' for c in feature_cols})  # already one row per (cell, feature)
    if 'spike_feature' in df_work.columns:
        df_cell = (df_work.groupby(['cell_id', 'spike_feature'])[metadata_cols + feature_cols]
                          .agg(agg)
                          .reset_index(drop=True))
    else:
        # Fallback: aggregate to one row per cell
        df_cell = (df_work.groupby('cell_id')[metadata_cols + feature_cols]
                          .agg(agg)
                          .reset_index(drop=True))

    rng     = np.random.default_rng(42)
    results = []

    for meta in metadata_cols:
        for feat in feature_cols:
            sub = df_cell[[meta, feat]].dropna()
            if len(sub) < 5:
                continue
            x = sub[meta].values
            y = sub[feat].values

            if meta in binary_cat:
                groups = np.unique(x)
                if len(groups) != 2:
                    continue
                g1, g2 = y[x == groups[0]], y[x == groups[1]]
                if len(g1) < 2 or len(g2) < 2:
                    continue
                stat, p = mannwhitneyu(g1, g2, alternative='two-sided')
                effect = 1.0 - 2.0 * stat / (len(g1) * len(g2))  # rank-biserial r
                effect_label = 'rank-biserial r'
                test = 'Mann-Whitney U'

                boot = []
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, len(sub), len(sub))
                    bs  = sub.iloc[idx]
                    bx, by = bs[meta].values, bs[feat].values
                    bg1, bg2 = by[bx == groups[0]], by[bx == groups[1]]
                    if len(bg1) < 2 or len(bg2) < 2:
                        continue
                    bs_stat, _ = mannwhitneyu(bg1, bg2, alternative='two-sided')
                    boot.append(1.0 - 2.0 * bs_stat / (len(bg1) * len(bg2)))

            elif meta in multi_cat:
                group_vals = {g: y[x == g] for g in np.unique(x)}
                group_vals = {g: v for g, v in group_vals.items() if len(v) >= 2}
                if len(group_vals) < 2:
                    continue
                stat, p = kruskal(*group_vals.values())
                n, k = len(sub), len(group_vals)
                effect = max(0.0, (stat - k + 1) / (n - k))
                effect_label = 'η²'
                test = 'Kruskal-Wallis'

                boot = []
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, len(sub), len(sub))
                    bs  = sub.iloc[idx]
                    bx, by = bs[meta].values, bs[feat].values
                    bg = {g: by[bx == g] for g in np.unique(bx)}
                    bg = {g: v for g, v in bg.items() if len(v) >= 2}
                    if len(bg) < 2:
                        continue
                    bs_stat, _ = kruskal(*bg.values())
                    nb, kb = len(bs), len(bg)
                    boot.append(max(0.0, (bs_stat - kb + 1) / (nb - kb)))

            else:  # continuous metadata → Spearman
                rho, p = spearmanr(x, y)
                effect = rho
                effect_label = 'ρ'
                test = 'Spearman'

                boot = []
                for _ in range(n_bootstrap):
                    idx = rng.integers(0, len(sub), len(sub))
                    bs  = sub.iloc[idx]
                    br, _ = spearmanr(bs[meta].values, bs[feat].values)
                    boot.append(br)

            ci_lo = float(np.nanpercentile(boot, 2.5))  if boot else np.nan
            ci_hi = float(np.nanpercentile(boot, 97.5)) if boot else np.nan

            results.append({
                'Metadata':     meta,
                'Feature':      feat,
                'test':         test,
                'effect_label': effect_label,
                'Effect_Size':  round(float(effect), 3),
                'ci_lo':        round(ci_lo, 3),
                'ci_hi':        round(ci_hi, 3),
                'p_raw':        float(p),
            })

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)

    # Benjamini-Hochberg FDR across all cross-pairs
    reject, p_fdr, _, _ = multipletests(df_res['p_raw'], method='fdr_bh', alpha=alpha)
    df_res['p_fdr']       = p_fdr
    df_res['significant'] = reject
    df_res['Significance'] = df_res['p_fdr'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    )

    # Forest plot: effect sizes with 95% bootstrap CIs
    df_plot = df_res.sort_values('Effect_Size', key=abs, ascending=True).reset_index(drop=True)
    df_plot['label'] = df_plot['Metadata'] + ' × ' + df_plot['Feature']
    n = len(df_plot)

    TEST_COLORS = {
        'Mann-Whitney U': '#0072B2',
        'Kruskal-Wallis': '#D55E00',
        'Spearman':       '#009E73',
    }
    x_max = max(abs(df_plot['ci_lo'].min()), abs(df_plot['ci_hi'].max())) * 1.1

    fig, ax = plt.subplots(figsize=(9, max(4, n * 0.55)))
    for i, row in enumerate(df_plot.itertuples()):
        color = TEST_COLORS.get(row.test, 'gray')
        alpha_pt = 1.0 if row.significant else 0.35
        ax.plot([row.ci_lo, row.ci_hi], [i, i], color=color, lw=2.5, alpha=alpha_pt)
        ax.scatter(row.Effect_Size, i, color=color, s=90, zorder=5, alpha=alpha_pt,
                   edgecolors='black' if row.significant else color, linewidths=1.5)
        stars = row.Significance if row.Significance != 'ns' else ''
        ax.text(x_max + 0.02, i,
                f"{row.Effect_Size:+.3f}  {stars}", va='center')

    ax.axvline(0, color='black', lw=1, ls='--', alpha=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df_plot['label'].tolist())
    ax.set_xlim(-x_max * 1.05, x_max * 1.4)
    ax.set_xlabel('Effect Size  (95% bootstrap CI)')
    ax.set_title(
        'Metadata × Cluster Difference\n(BH-FDR corrected; filled marker = significant)',
        fontweight='bold'
    )
    from matplotlib.lines import Line2D
    legend_els = [Line2D([0], [0], color=c, lw=3, label=t) for t, c in TEST_COLORS.items()]
    ax.legend(handles=legend_els, frameon=False, loc='lower right')
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()

    # Print summary
    sig = df_res[df_res['significant']].sort_values('p_fdr')
    print(f"\n{len(sig)}/{len(df_res)} pairs significant after BH-FDR (α={alpha}, n_bootstrap={n_bootstrap}):")
    for _, r in sig.iterrows():
        print(f"  {r['Metadata']:22s} × {r['Feature']:14s} | "
              f"{r['effect_label']} = {r['Effect_Size']:+.3f}  "
              f"95% CI [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]  "
              f"p_fdr={('< 0.0001' if r['p_fdr'] < 0.0001 else str(round(r['p_fdr'], 4)))} {r['Significance']}")

    sig['p-value'] = sig['p_fdr'].apply(lambda p: '< 0.0001' if p < 0.0001 else f'{p:.4f}')
    return sig.reset_index(drop=True)

def plot_sig_feat_pairs(df, sig_pairs_df):
    # Nuke the warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    if sig_pairs_df is None or len(sig_pairs_df) == 0:
        print("No significant pairs to plot.")
        return

    n_plots = len(sig_pairs_df)
    cols = 3
    rows = math.ceil(n_plots / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows))
    axes = axes.flatten() 

    for i, (_, row) in enumerate(sig_pairs_df.iterrows()):
        m, f = row['Metadata'], row['Feature']
        r, p = float(row['Effect_Size']), float(row['p_fdr'] if 'p_fdr' in row.index else row['p-value'])
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
            
            plot_data[val_col] = pd.to_numeric(plot_data[val_col], errors='coerce')
            plot_data = plot_data.dropna(subset=[val_col])
            if plot_data.empty:
                ax.set_visible(False)
                continue

            sns.boxplot(data=plot_data, y=cat_col, x=val_col, palette="Paired",
                        order=order, showfliers=False, orient='h', ax=ax)
            sns.stripplot(data=plot_data, y=cat_col, x=val_col, color=".3",
                          alpha=.3, order=order, orient='h', ax=ax)

            x_min, x_max = plot_data[val_col].min(), plot_data[val_col].max()
            pad = (x_max - x_min) * 0.1 if x_max != x_min else 0.1
            ax.set_xlim(x_min - pad, x_max + pad)
            
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

        ax.set_title(f"{stars} (r={r:.2f})", fontweight='bold')
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
        'nRMSE': 'mean',
        'temporal_rho': 'mean'
    }).reset_index()
    
    return feature_counts.merge(metrics, on='spike_feature').sort_values('prevalence_pct', ascending=False)


def plot_feature_cluster_grid(df_master):
    """
    Cell × feature grid showing number of clusters per (cell, feature).
    White = no clustering, blue = 2 clusters (bimodal), orange = 3+ clusters.
    ISI/timing features (log_isi, spk_times_ms, spk_times_idx) are separated
    by a dashed vertical line and labelled in red-italic to distinguish them
    from waveform shape features.
    """
    import matplotlib.colors as mcolors
    from matplotlib.patches import Patch

    ISI_FEATS = {'log_isi', 'spk_times_ms', 'spk_times_idx'}

    # Max num_clusters per (cell, feature) — collapse cluster-pair rows
    pivot = (df_master.groupby(['cell_id', 'spike_feature'])['num_clusters']
             .max()
             .unstack(fill_value=0))

    # Cell order by numeric ID
    cell_order = sorted(pivot.index, key=lambda x: int(x.lstrip('c')))
    pivot = pivot.loc[cell_order]

    # Feature order: waveform first, ISI/timing last
    wf_feats  = [f for f in pivot.columns if f not in ISI_FEATS]
    isi_feats = [f for f in pivot.columns if f in ISI_FEATS]
    feat_order = wf_feats + isi_feats
    pivot = pivot.reindex(columns=feat_order, fill_value=0)

    n_cells = len(pivot)
    n_feats = len(feat_order)

    # Recode: 0 → no clustering, 1 → bimodal (2 clusters), 2 → multimodal (3+)
    mat = pivot.values.copy().astype(float)
    coded = np.zeros_like(mat)
    coded[mat == 2] = 1.0
    coded[mat >= 3] = 2.0

    cmap  = mcolors.ListedColormap(['#f2f2f2', '#4393C3', '#D6604D'])
    norm  = mcolors.BoundaryNorm([0, 0.5, 1.5, 3], cmap.N)

    fig_w = max(9, n_feats * 0.75 + 2)
    fig_h = max(5, n_cells * 0.28 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(coded, aspect='auto', cmap=cmap, norm=norm, interpolation='nearest')

    # Axis ticks
    ax.set_xticks(np.arange(n_feats))
    ax.set_yticks(np.arange(n_cells))
    ax.set_yticklabels(cell_order, fontsize=7)

    xlabels = ax.set_xticklabels(feat_order, rotation=40, ha='right', fontsize=8)
    for lbl, feat in zip(xlabels, feat_order):
        if feat in ISI_FEATS:
            lbl.set_color('#B22222')
            lbl.set_fontstyle('italic')

    # Separator between waveform and ISI groups
    if isi_feats:
        ax.axvline(len(wf_feats) - 0.5, color='#444', lw=1.5, ls='--', zorder=5)
        ax.text(len(wf_feats) + len(isi_feats) / 2 - 0.5, n_cells + 0.8,
                'ISI / timing', ha='center', va='bottom',
                fontsize=8, color='#B22222', style='italic',
                transform=ax.transData, clip_on=False)

    # Minor grid lines as cell borders
    ax.set_xticks(np.arange(-0.5, n_feats), minor=True)
    ax.set_yticks(np.arange(-0.5, n_cells), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.8)
    ax.tick_params(which='minor', length=0)

    # Legend
    legend_els = [
        Patch(facecolor='#f2f2f2', edgecolor='#aaa', label='No clustering'),
        Patch(facecolor='#4393C3', label='2 clusters (bimodal)'),
        Patch(facecolor='#D6604D', label='3+ clusters (multimodal)'),
    ]
    ax.legend(handles=legend_els, loc='upper left', bbox_to_anchor=(0, 1.08),
              fontsize=8, frameon=False, ncol=3)

    ax.set_title('Feature cluster presence — all cells', fontsize=11,
                 fontweight='bold', loc='left', pad=28)
    ax.set_xlabel('Spike feature', fontsize=9, labelpad=6)
    ax.set_ylabel('Cell', fontsize=9)

    sns.despine(ax=ax, left=True, bottom=True)
    plt.tight_layout()
    return fig, ax


def plot_feature_distribution_grid(df_master, cluster_pickle_dir,
                                   min_cells=2, n_cols_per_row=14):
    """
    Small-multiples grid of smooth KDE distributions split by cluster, all cells.
    Features with more than n_cols_per_row cells wrap onto multiple rows.
    Waveform features first; log_isi last (red-italic, pink background).
    spk_times_ms / spk_times_idx excluded.
    Colors: low=blue, mid=green, high=orange. Solid filled KDE curves.
    """
    from scipy.stats import gaussian_kde as _kde
    from matplotlib.patches import Patch

    ISI_FEATS    = {'log_isi'}
    SKIP_FEATS   = {'spk_times_ms', 'spk_times_idx'}
    CLUST_COLORS = {'low': '#0072B2', 'mid': '#E69F00', 'high': '#CC79A7'}

    df_num = df_master.copy()
    df_num['num_clusters'] = pd.to_numeric(df_num['num_clusters'], errors='coerce')

    all_pkl = {os.path.basename(p).replace('_cluster_df.pkl', ''): p
               for p in glob.glob(os.path.join(cluster_pickle_dir, 'c*_cluster_df.pkl'))}

    # All cells per feature (from df_master)
    clustered = {}
    for feat, grp in df_num[df_num['num_clusters'] >= 2].groupby('spike_feature'):
        if feat in SKIP_FEATS:
            continue
        cells = sorted(grp['cell_id'].unique(), key=lambda c: int(c.lstrip('c')))
        if len(cells) >= min_cells:
            clustered[feat] = cells

    wf_feats  = sorted([f for f in clustered if f not in ISI_FEATS],
                       key=lambda f: -len(clustered[f]))
    isi_feats = [f for f in clustered if f in ISI_FEATS]
    feat_order = wf_feats + isi_feats

    # Build row list with wrapping
    row_groups = []   # (feat, cell_chunk, is_first_chunk, is_isi)
    n_wf_rows  = 0
    for feat in feat_order:
        is_isi = feat in ISI_FEATS
        cells  = clustered[feat]
        chunks = [cells[i:i + n_cols_per_row]
                  for i in range(0, len(cells), n_cols_per_row)]
        for ci, chunk in enumerate(chunks):
            row_groups.append((feat, chunk, ci == 0, is_isi))
            if not is_isi:
                n_wf_rows += 1

    n_rows  = len(row_groups)
    label_w = 1.8
    cell_w  = 1.1
    cell_h  = 1.1
    fig_w   = label_w + n_cols_per_row * cell_w + 0.3
    fig_h   = n_rows * cell_h + 0.7

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = fig.add_gridspec(
        n_rows, n_cols_per_row + 1,
        left=label_w / fig_w,
        right=0.99, top=0.94, bottom=0.02,
        hspace=0.4, wspace=0.08,
        width_ratios=[0.001] + [1] * n_cols_per_row,
    )

    for ri, (feat, chunk, is_first, is_isi) in enumerate(row_groups):
        clust_col  = feat + '_cluster'
        row_center = 1 - (ri + 0.5) / n_rows

        if is_first:
            fig.text(
                (label_w * 0.90) / fig_w, row_center,
                feat, ha='right', va='center',
                fontsize=20, fontweight='bold',
                color='#B22222' if is_isi else 'black',
                style='italic' if is_isi else 'normal',
            )

        for ci, cell_id in enumerate(chunk):
            ax = fig.add_subplot(gs[ri, ci + 1])
            pkl_path = all_pkl.get(cell_id)
            if pkl_path is not None:
                try:
                    df_cell = pd.read_pickle(pkl_path)
                    if clust_col in df_cell.columns and feat in df_cell.columns:
                        all_vals = df_cell[feat].dropna()
                        pad    = (all_vals.max() - all_vals.min()) * 0.08
                        x_grid = np.linspace(all_vals.min() - pad,
                                             all_vals.max() + pad, 300)
                        for grp in sorted(df_cell[clust_col].dropna().unique()):
                            vals = df_cell.loc[df_cell[clust_col] == grp, feat].dropna()
                            if len(vals) < 5:
                                continue
                            y   = _kde(vals, bw_method=0.3)(x_grid)
                            col = CLUST_COLORS.get(grp, '#888')
                            ax.fill_between(x_grid, y, color=col, alpha=1.0)
                            ax.plot(x_grid, y, color=col, lw=0.6)
                except Exception:
                    pass

            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            ax.set_facecolor('#f5f5f5' if not is_isi else '#fff0f0')
            ax.set_title(cell_id, fontsize=13, pad=3, color='#444')

        for ci in range(len(chunk), n_cols_per_row):
            fig.add_subplot(gs[ri, ci + 1]).set_axis_off()

    # Separator between waveform and ISI sections
    if wf_feats and isi_feats:
        sep_y = 1 - n_wf_rows / n_rows
        fig.add_artist(plt.Line2D(
            [label_w / fig_w, 0.99], [sep_y, sep_y],
            color='#888', lw=1.0, ls='--', transform=fig.transFigure,
        ))

    legend_els = [Patch(facecolor='#0072B2', label='low cluster'),
                  Patch(facecolor='#E69F00', label='mid cluster'),
                  Patch(facecolor='#CC79A7', label='high cluster')]
    fig.legend(handles=legend_els, loc='upper right',
               bbox_to_anchor=(0.99, 1.0), fontsize=16, frameon=False, ncol=3)
    return fig


def plot_metadata_effect_heatmap(df_res):
    """
    Heatmap of effect sizes (metadata × cluster metric).
    Cells show effect size; asterisks mark BH-FDR significant pairs.
    """
    METRIC_LABELS = {'nRMSE': 'nRMSE', 'cos_sim': 'Cos Sim',
                     'num_clusters': 'N Clusters', 'temporal_rho': 'Temporal ρ'}
    META_LABELS = {
        'patch_type':         'Patch type',
        'current_type':       'Current type',
        'cell_type':          'Cell type',
        'dark_neuron':        'Dark neuron',
        'clear_EAP_waveform': 'Clear EAP',
        'cortical_depth':     'Cortical depth',
    }

    metrics  = [m for m in ['nRMSE', 'cos_sim', 'num_clusters', 'temporal_rho']
                if m in df_res['Feature'].values]
    metadata = [m for m in META_LABELS if m in df_res['Metadata'].values]

    effect_mat = pd.DataFrame(index=metadata, columns=metrics, dtype=float)
    sig_mat    = pd.DataFrame(index=metadata, columns=metrics, data='')

    for _, row in df_res.iterrows():
        m, f = row['Metadata'], row['Feature']
        if m in metadata and f in metrics:
            effect_mat.loc[m, f] = row['Effect_Size']
            stars = row['Significance'] if row['Significance'] != 'ns' else ''
            sig_mat.loc[m, f] = stars

    effect_mat = effect_mat.astype(float)

    vmax = np.nanmax(np.abs(effect_mat.values))
    fig, ax = plt.subplots(figsize=(len(metrics) * 1.6 + 1.5, len(metadata) * 0.9 + 1.2))

    im = ax.imshow(effect_mat.values, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics], fontsize=13)
    ax.set_yticks(range(len(metadata)))
    ax.set_yticklabels([META_LABELS.get(m, m) for m in metadata], fontsize=13)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    for i, meta in enumerate(metadata):
        for j, feat in enumerate(metrics):
            val  = effect_mat.loc[meta, feat]
            star = sig_mat.loc[meta, feat]
            if pd.notna(val):
                txt = f"{val:+.2f}"
                if star:
                    txt += f"\n{star}"
                text_col = 'white' if abs(val) > vmax * 0.6 else '#222222'
                ax.text(j, i, txt, ha='center', va='center',
                        fontsize=10, fontweight='bold', color=text_col)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label('Effect size', fontsize=11)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    ax.set_title('Recording metadata × cluster waveform metrics\n(BH-FDR corrected; *p<.05  **p<.01  ***p<.001)',
                 fontsize=13, fontweight='bold', pad=40)
    plt.tight_layout()
    plt.show()


def plot_feature_distribution_r2(df_master, cluster_pickle_dir, spike_fit_dir,
                                  r2_type='exp', min_cells=2, n_cols=12):
    """
    For each (cell × spike feature) pair with ≥2 clusters, show a KDE distribution
    of the feature values — same layout as the population cluster histograms — but
    each cluster's KDE is colored by the mean R² of spikes in that cluster (viridis).

    x-axis : spike feature value
    y-axis : KDE density
    color  : mean per-spike R² for each cluster on viridis (dark=low, yellow=high)
    shared colorbar on the right

    spk_id in the cluster pickle indexes into the spike-fit R² array; spikes
    whose spk_id exceeds the fit array length are silently skipped.
    """
    import pickle
    import matplotlib.cm as _cm
    import matplotlib.colors as _mc
    from scipy.stats import gaussian_kde as _kde

    r2_attr  = 'r_squared_exp'  if r2_type == 'exp'  else 'r_squared_ramp'
    r2_label = 'Exp decay R²'   if r2_type == 'exp'  else 'Ramp fit R²'
    ORDINAL    = {'low': 0, 'mid': 1, 'high': 2}
    SKIP_FEATS = {'spk_times_ms', 'spk_times_idx'}

    cmap = _cm.get_cmap('viridis')
    norm = _mc.Normalize(vmin=0, vmax=1)

    # ── Load per-spike R² arrays keyed by cell ───────────────────────────
    r2_arr_by_cell = {}
    for _pkl_f in glob.glob(os.path.join(spike_fit_dir, 'c*_spike_fit.pkl')):
        _cid = os.path.basename(_pkl_f).replace('_spike_fit.pkl', '')
        try:
            with open(_pkl_f, 'rb') as _fh:
                _spk = pickle.load(_fh)
            r2_arr_by_cell[_cid] = np.asarray(getattr(_spk, r2_attr))
        except Exception:
            pass

    if not r2_arr_by_cell:
        print('No spike fit pickles found in', spike_fit_dir)
        return

    # ── Build (cell, feature) pairs with ≥2 clusters ─────────────────────
    df_num = df_master.copy()
    df_num['num_clusters'] = pd.to_numeric(df_num['num_clusters'], errors='coerce')
    pairs = (df_num[df_num['num_clusters'] >= 2][['cell_id', 'spike_feature']]
             .drop_duplicates())
    pairs = pairs[~pairs['spike_feature'].isin(SKIP_FEATS)]
    pairs = pairs[pairs['cell_id'].isin(r2_arr_by_cell)]

    # ── One figure per feature ────────────────────────────────────────────
    for feat in sorted(pairs['spike_feature'].unique()):
        cells = sorted(
            pairs[pairs['spike_feature'] == feat]['cell_id'].tolist(),
            key=lambda c: int(c.lstrip('c'))
        )
        if len(cells) < min_cells:
            continue

        ncols_f = min(n_cols, len(cells))
        nrows_f = int(np.ceil(len(cells) / ncols_f))

        fig = plt.figure(figsize=(ncols_f * 2.6 + 0.7, nrows_f * 2.6))
        gs  = fig.add_gridspec(nrows_f, ncols_f + 1,
                               width_ratios=[1] * ncols_f + [0.05],
                               hspace=0.55, wspace=0.35)
        axes    = np.array([[fig.add_subplot(gs[r, c])
                             for c in range(ncols_f)]
                            for r in range(nrows_f)])
        cbar_ax = fig.add_subplot(gs[:, ncols_f])
        axs = axes.flat

        for ax, cid in zip(axs, cells):
            pkl_path = os.path.join(cluster_pickle_dir, f'{cid}_cluster_df.pkl')
            try:
                cl_df = pd.read_pickle(pkl_path)
            except Exception:
                ax.set_visible(False)
                continue

            col_name = f'{feat}_cluster'
            if feat not in cl_df.columns or 'spk_id' not in cl_df.columns or col_name not in cl_df.columns:
                ax.set_visible(False)
                continue

            r2_arr = r2_arr_by_cell[cid]
            n_fit  = len(r2_arr)

            _sub = cl_df[[feat, 'spk_id', col_name]].dropna()
            _sub = _sub[_sub['spk_id'].astype(int) < n_fit]
            _sub = _sub[_sub[col_name].astype(str).str.lower().isin(ORDINAL)]
            if _sub.empty:
                ax.set_visible(False)
                continue

            _sub = _sub.copy()
            _sub['r2']    = r2_arr[_sub['spk_id'].astype(int).values]
            _sub['clust'] = _sub[col_name].astype(str).str.lower()

            any_plotted = False
            for lbl in ['low', 'mid', 'high']:
                grp = _sub[_sub['clust'] == lbl]
                if len(grp) < 5:
                    continue
                feat_vals = grp[feat].values
                mean_r2   = float(np.nanmean(grp['r2'].values))
                fill_c    = cmap(norm(mean_r2))
                try:
                    kde_fn = _kde(feat_vals)
                    x_grid = np.linspace(feat_vals.min(), feat_vals.max(), 300)
                    y_grid = kde_fn(x_grid)
                    ax.fill_between(x_grid, y_grid, alpha=0.55, color=fill_c)
                    ax.plot(x_grid, y_grid, color=fill_c, lw=1.0)
                    any_plotted = True
                except Exception:
                    pass

            if not any_plotted:
                ax.set_visible(False)
                continue

            ax.set_title(cid, fontsize=8, fontweight='bold')
            ax.set_xlabel(feat.replace('_', ' '), fontsize=7)
            ax.set_ylabel('density', fontsize=6)
            ax.tick_params(labelsize=6)
            sns.despine(ax=ax)

        for ax in list(axs)[len(cells):]:
            ax.set_visible(False)

        sm = _cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label(r2_label, fontsize=9, fontweight='bold')
        cbar.ax.tick_params(labelsize=7)

        fig.suptitle(
            f'{feat.replace("_", " ")}  —  KDE per cluster, color = mean {r2_label} of that cluster',
            fontsize=11, fontweight='bold'
        )
        plt.show()


def plot_aggregated_spike_feat(raw_df):
    # 1. Internal Aggregation: Calculate Means and 95% CI
    stats = raw_df.groupby('spike_feature').agg({
        'nRMSE': ['mean', 'std', 'count'],
        'cos_sim': ['mean', 'std', 'count'],
        'num_clusters': 'mean',
        'temporal_rho': 'mean'
    })

    # Flatten columns
    stats.columns = ['nRMSE', 'nRMSE_std', 'nRMSE_n', 'cos_sim', 'cos_sim_std', 'cos_sim_n', 'num_clusters', 'temporal_rho']
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
            fontsize=11, fontweight='semibold', va='bottom'
        )

    # 6. Final Polish
    plt.title('Spike Feature Aggregated Analysis (Mean ± 95% CI)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('$\longrightarrow$ Higher difference in waveforms (nRMSE)', fontsize=13)
    plt.ylabel('$\longleftarrow$ Higher difference in shape morphology (Cosine Similarity)', fontsize=13)
    
    plt.ylim(stats['cos_sim'].min() - 0.03, 1.01)
    plt.xlim(-0.005, stats['nRMSE'].max() + 0.03)
    
    plt.legend(title='Clusters & % Prevalence', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine()
    plt.grid(True, linestyle=':', alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_feature_depth_distribution(df):
    plt.figure(figsize=(12, 8))

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
    plt.title('Cortical Depth Distribution of clustered spike features', fontweight='bold', pad=25)
    plt.xlabel('Cortical Depth ($\mu m$)', fontweight='semibold')
    plt.ylabel('Clustered Spike Features', fontweight='semibold')
    
    # 6. Clean up
    plt.grid(axis='x', linestyle='--', alpha=0.4)
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()

def plot_meta_spk_feature_dependency(df):
    # 1. Identify Categorical Metadata (excluding metrics and keys)
    exclude = ['spike_feature', 'num_clusters', 'cos_sim', 'nRMSE', 'temporal_rho', 'temporal_p', 'cell_id', 'cortical_depth', 'nRMSE_std', 'cos_sim_std', 'cluster']
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

    ax1.set_title('Spike Feature Prevalence by Metadata (%)', fontweight='bold', fontsize=15, pad=15)
    ax2.set_title('Mean Depth', fontweight='bold', fontsize=15, pad=15)
    ax1.set_ylabel('Spike Features (Sorted by Depth)', fontsize=13)
    ax1.set_xlabel('Metadata Categories', fontsize=13)
    
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
    print(f"p-value: {'< 0.0001' if p < 0.0001 else f'{p:.4f}'}")
    
    if p < 0.05:
        print("\nSignificant differences found across cortical layers.")
    else:
        print("\nNo significant depth stratification found.")
    print("-" * 40)
    
    return p



# ------------------------------------------------------------------------------------------- #
# ------------------------------ Temporal Structure Analysis ------------------------------ #
# ------------------------------------------------------------------------------------------- #

def plot_temporal_structure(df, alpha=0.05):
    """
    Visualize temporal_rho across all cell-feature groups.

    Shows whether spike cluster identity shows time dependence over recording time
    (Spearman rho between spike time and ordinal cluster label).

    Panel A: histogram of temporal_rho across all cell-feature groups,
             with significant groups highlighted.
    Panel B: box/strip of temporal_rho distribution per spike feature.
    """
    df_plot = df[['cell_id', 'spike_feature', 'temporal_rho', 'temporal_p']].dropna().copy()
    df_plot['temporal_rho'] = pd.to_numeric(df_plot['temporal_rho'], errors='coerce')
    df_plot['temporal_p']   = pd.to_numeric(df_plot['temporal_p'],   errors='coerce')
    # Deduplicate to one row per cell-feature group
    df_plot = df_plot.groupby(['cell_id', 'spike_feature'], as_index=False).first()
    df_plot = df_plot.dropna(subset=['temporal_rho'])
    df_plot['significant'] = df_plot['temporal_p'] < alpha

    # Consistent color per spike feature
    features = sorted(df_plot['spike_feature'].unique())
    palette = dict(zip(features, sns.color_palette('tab10', n_colors=len(features))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Panel A: Histogram ---
    ax1.hist(df_plot['temporal_rho'], bins=20, color='#0072B2', alpha=0.6, edgecolor='white', label='all')
    sig_vals = df_plot.loc[df_plot['significant'], 'temporal_rho']
    ax1.hist(sig_vals, bins=20, color='#D55E00', alpha=0.8, edgecolor='white', label=f'p < {alpha}')
    ax1.axvline(0, color='black', lw=1.5, linestyle='--', alpha=0.6)
    ax1.set_xlabel('Temporal Rho (Spearman)')
    ax1.set_ylabel('Count')
    n_sig = df_plot['significant'].sum()
    n_total = len(df_plot)
    ax1.set_title(f'Time Dependence Distribution\n{n_sig}/{n_total} significant (p < {alpha})',
                  fontweight='bold')
    ax1.legend(frameon=False)
    ax1.set_xlim(-1.1, 1.1)
    sns.despine(ax=ax1)

    # --- Panel B: Distribution per spike feature ---
    feat_order = df_plot.groupby('spike_feature')['temporal_rho'].median().sort_values().index.tolist()
    feat_palette = [palette[f] for f in feat_order]

    sns.boxplot(data=df_plot, x='temporal_rho', y='spike_feature', order=feat_order,
                palette=feat_palette, showfliers=False, width=0.5, linewidth=2.5, ax=ax2)
    sns.stripplot(data=df_plot, x='temporal_rho', y='spike_feature', order=feat_order,
                  palette=feat_palette, alpha=0.5, size=5, ax=ax2)
    ax2.axvline(0, color='black', lw=1, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Temporal Rho')
    ax2.set_ylabel('')
    ax2.set_title('Distribution by Spike Feature', fontweight='bold')
    ax2.set_xlim(-1.1, 1.1)
    sns.despine(ax=ax2)

    plt.tight_layout()
    plt.show()

    # Print summary
    n_sig = df_plot['significant'].sum()
    n_total = len(df_plot)
    print(f"\nTime dependence summary (p < {alpha}):")
    print(f"  Significant: {n_sig} / {n_total} cell-feature groups ({100*n_sig/n_total:.1f}%)")
    print(f"  Mean |rho|: {df_plot['temporal_rho'].abs().mean():.3f}")
    sig_df = df_plot[df_plot['significant']][['cell_id', 'spike_feature', 'temporal_rho', 'temporal_p']].copy()
    sig_df['temporal_rho'] = sig_df['temporal_rho'].round(3)
    sig_df['temporal_p']   = sig_df['temporal_p'].map(lambda x: f"{x:.4f}" if x >= 0.0001 else "<0.0001")
    if not sig_df.empty:
        print("\nSignificant groups:")
        print(sig_df.sort_values('temporal_rho').to_string(index=False))

    return df_plot


def analyze_temporal_metadata_dependency(df, alpha=0.05):
    """
    Tests whether cell-level metadata predicts temporal_rho (time dependence of cluster identity
    over recording time).

    Categorical metadata (cell_type, patch_type, etc.): Kruskal-Wallis + box/strip plot.
    Continuous metadata (cortical_depth): Spearman correlation + scatter plot.

    Returns a DataFrame of test results sorted by p-value.
    """
    meta_cols = ['patch_type', 'current_type', 'cell_type', 'dark_neuron', 'clear_EAP_waveform', 'cortical_depth']
    cont_cols = {'cortical_depth'}

    df_plot = df[['cell_id', 'spike_feature', 'temporal_rho'] + meta_cols].copy()
    df_plot['temporal_rho'] = pd.to_numeric(df_plot['temporal_rho'], errors='coerce')
    df_plot = df_plot.dropna(subset=['temporal_rho'])

    n_plots = len(meta_cols)
    ncols = 3
    nrows = math.ceil(n_plots / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
    axes = axes.flatten()

    results = []
    for ax_idx, col in enumerate(meta_cols):
        ax = axes[ax_idx]
        sub = df_plot[['temporal_rho', col]].dropna()

        if col in cont_cols:
            x = pd.to_numeric(sub[col], errors='coerce')
            y = sub['temporal_rho']
            valid = np.isfinite(x) & np.isfinite(y)
            rho, p = spearmanr(x[valid], y[valid])
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

            ax.scatter(x[valid], y[valid], alpha=0.4, color='#0072B2', s=30)
            m_s, b_s = np.polyfit(x[valid], y[valid], 1)
            x_line = np.linspace(x[valid].min(), x[valid].max(), 100)
            ax.plot(x_line, m_s * x_line + b_s, color='darkblue', lw=2)
            ax.axhline(0, color='black', lw=1, linestyle='--', alpha=0.4)
            ax.set_xlabel(col.replace('_', ' ').title())
            ax.set_ylabel('Temporal Rho')
            p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            ax.set_title(f'{col}\nSpearman ρ={rho:.2f}, {sig} (p={p_str})', fontweight='bold')
            results.append({'variable': col, 'test': 'Spearman', 'statistic': round(rho, 3), 'p': p, 'sig': sig})
        else:
            groups_list = [g['temporal_rho'].values for _, g in sub.groupby(col)]
            if len(groups_list) >= 2:
                stat, p = kruskal(*groups_list)
            else:
                stat, p = np.nan, np.nan
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

            sns.boxplot(data=sub, x=col, y='temporal_rho', showfliers=False,
                        palette='Paired', linewidth=2.5, ax=ax)
            sns.stripplot(data=sub, x=col, y='temporal_rho', color='.3', alpha=0.4, ax=ax)
            ax.axhline(0, color='black', lw=1, linestyle='--', alpha=0.4)
            ax.set_xlabel(col.replace('_', ' ').title())
            ax.set_ylabel('Temporal Rho')
            p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            ax.set_title(f'{col}\nKruskal-Wallis {sig} (p={p_str})', fontweight='bold')
            results.append({'variable': col, 'test': 'Kruskal-Wallis', 'statistic': round(stat, 3) if pd.notnull(stat) else np.nan, 'p': p, 'sig': sig})

        sns.despine(ax=ax)

    for j in range(len(meta_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Metadata Predictors of Time Dependence (temporal_rho)', fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()

    res_df = pd.DataFrame(results).sort_values('p').reset_index(drop=True)
    print("\nMetadata → temporal_rho test results:")
    print(res_df.to_string(index=False))
    return res_df


def analyze_temporal_clustering_relationship(df, alpha=0.05):
    """
    Tests whether time dependence (temporal_rho / temporal_component) is related to
    clustering quality metrics: nRMSE, cos_sim, num_clusters.

    Panel 1: scatter of temporal_rho vs each metric, colored by spike_feature.
    Panel 2: box/strip of nRMSE and cos_sim split by temporal_component (0 vs 1).

    Returns a DataFrame of Spearman correlation results.
    """
    df_plot = df[['cell_id', 'spike_feature', 'temporal_rho', 'temporal_component',
                  'nRMSE', 'cos_sim', 'num_clusters']].copy()
    for col in ['temporal_rho', 'nRMSE', 'cos_sim', 'num_clusters', 'temporal_component']:
        df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')
    df_plot = df_plot.dropna(subset=['temporal_rho', 'nRMSE', 'cos_sim'])

    # Aggregate to one row per cell-feature group (temporal_rho is constant within a group;
    # nRMSE/cos_sim vary per cluster so we average across clusters)
    df_plot = df_plot.groupby(['cell_id', 'spike_feature'], as_index=False).agg({
        'temporal_rho': 'first',
        'temporal_component': 'first',
        'nRMSE': 'mean',
        'cos_sim': 'mean',
        'num_clusters': 'first',
    })

    features = sorted(df_plot['spike_feature'].dropna().unique())
    palette = dict(zip(features, sns.color_palette('tab10', n_colors=len(features))))

    # --- Panel 1: temporal_rho vs clustering metrics ---
    metrics = ['nRMSE', 'cos_sim', 'num_clusters']
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    results = []
    for ax, metric in zip(axes, metrics):
        sub = df_plot[['temporal_rho', metric, 'spike_feature']].dropna()
        colors = [palette.get(f, 'gray') for f in sub['spike_feature']]
        ax.scatter(sub['temporal_rho'], sub[metric], c=colors, alpha=0.5, s=40)

        rho, p = spearmanr(sub['temporal_rho'], sub[metric])
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        valid = np.isfinite(sub['temporal_rho']) & np.isfinite(sub[metric])
        m_s, b_s = np.polyfit(sub['temporal_rho'][valid], sub[metric][valid], 1)
        x_line = np.linspace(sub['temporal_rho'].min(), sub['temporal_rho'].max(), 100)
        ax.plot(x_line, m_s * x_line + b_s, color='black', lw=2, alpha=0.8)

        ax.axvline(0, color='gray', lw=1, linestyle='--', alpha=0.4)
        ax.set_xlabel('Temporal Rho')
        ax.set_ylabel(metric)
        ax.set_title(f'temporal_rho vs {metric}\nSpearman ρ={rho:.2f}, {sig}', fontweight='bold')
        sns.despine(ax=ax)
        results.append({'metric': metric, 'spearman_rho': round(rho, 3), 'p': p, 'sig': sig})

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[f],
                          markersize=8, label=f) for f in features]
    axes[-1].legend(handles=handles, title='Spike Feature', bbox_to_anchor=(1.05, 1),
                    loc='upper left', frameon=True)

    plt.suptitle('Time Dependence vs Clustering Quality', fontweight='bold')
    plt.tight_layout()
    plt.show()

    # --- Panel 2: temporal_rho distribution by spike_feature ---
    SKIP_FEATS_TEMPORAL = {'spk_times_idx', 'spk_times_ms'}
    df_feat_plot = df_plot[~df_plot['spike_feature'].isin(SKIP_FEATS_TEMPORAL)]
    feat_order = df_feat_plot.groupby('spike_feature')['temporal_rho'].median().sort_values().index.tolist()
    feat_palette_list = [palette.get(f, 'gray') for f in feat_order]

    fig_feat, ax_feat = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_feat_plot, x='spike_feature', y='temporal_rho', order=feat_order,
                palette=feat_palette_list, showfliers=False, width=0.5, linewidth=2.5, ax=ax_feat)
    sns.stripplot(data=df_feat_plot, x='spike_feature', y='temporal_rho', order=feat_order,
                  palette=feat_palette_list, alpha=0.5, size=6, ax=ax_feat)
    ax_feat.axhline(0, color='black', lw=1, linestyle='--', alpha=0.4)
    ax_feat.text(len(feat_order) - 0.5, 0.03, 'no time\ndependence',
                 ha='right', va='bottom', fontsize=_FS_SM, color='#666666', style='italic',
                 linespacing=1.2)

    feat_groups = [df_feat_plot.loc[df_feat_plot['spike_feature'] == f, 'temporal_rho'].dropna().values for f in feat_order]
    feat_groups = [g for g in feat_groups if len(g) > 0]
    if len(feat_groups) >= 2:
        kw_stat, kw_p = kruskal(*feat_groups)
    else:
        kw_stat, kw_p = np.nan, np.nan
    kw_sig = '***' if kw_p < 0.001 else '**' if kw_p < 0.01 else '*' if kw_p < 0.05 else 'ns'
    kw_p_str = f"{kw_p:.4f}" if pd.notnull(kw_p) and kw_p >= 0.0001 else ("<0.0001" if pd.notnull(kw_p) else "n/a")

    ax_feat.set_title('Temporal Rho by Spike Feature', fontweight='bold', pad=28)
    ax_feat.text(0.5, 1.01, f'Kruskal-Wallis {kw_sig} (p={kw_p_str})',
                 transform=ax_feat.transAxes, ha='center', va='bottom', fontsize=_FS_SM, color='#555555')
    ax_feat.set_xlabel('Spike Feature')
    ax_feat.set_ylabel('Temporal Rho')
    ax_feat.set_xticklabels(ax_feat.get_xticklabels(), rotation=30, ha='right')
    sns.despine(ax=ax_feat)
    plt.tight_layout()
    plt.show()

    # --- Panel 3: temporal_component (0 vs 1) split on nRMSE / cos_sim ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    for ax2, metric in zip(axes2, ['nRMSE', 'cos_sim']):
        sub2 = df_plot[['temporal_component', metric]].dropna()
        sub2['temporal_component'] = sub2['temporal_component'].astype(float).map({0.0: '|ρ| < 0.3', 1.0: '|ρ| ≥ 0.3'})
        sns.boxplot(data=sub2, x='temporal_component', y=metric, showfliers=False,
                    palette=['#56B4E9', '#D55E00'], order=['|ρ| < 0.3', '|ρ| ≥ 0.3'],
                    linewidth=2.5, ax=ax2)
        sns.stripplot(data=sub2, x='temporal_component', y=metric, color='.3', alpha=0.4,
                      order=['|ρ| < 0.3', '|ρ| ≥ 0.3'], ax=ax2)

        groups_list = [g[metric].values for _, g in sub2.groupby('temporal_component') if len(g) > 0]
        if len(groups_list) >= 2:
            stat2, p2 = kruskal(*groups_list)
        else:
            p2 = np.nan
        sig2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else 'ns'
        p2_str = f"{p2:.4f}" if pd.notnull(p2) and p2 >= 0.0001 else ("<0.0001" if pd.notnull(p2) else "n/a")
        ax2.set_title(f'temporal_component vs {metric}\n{sig2} (p={p2_str})', fontweight='bold')
        ax2.set_xlabel('Temporal Component')
        sns.despine(ax=ax2)

    plt.suptitle('Does Significant Time Dependence Affect Cluster Waveform Differences?', fontweight='bold')
    plt.tight_layout()
    plt.show()

    res_df = pd.DataFrame(results)
    print("\ntemporal_rho vs clustering metrics (Spearman):")
    print(res_df.to_string(index=False))
    return res_df


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
        p_str = '< 0.0001' if p < 0.0001 else f'{p:.4f}'
        print(f"{col:<20} | p = {p_str} ({sig})")
        
        results.append({'metadata': col, 'p_value': p, 'sig': sig})
        
    print("-" * 40)
    return pd.DataFrame(results)


# ------------------------------------------------------------------------------------------- #
# --------------------------------- Cell selection ------------------------------------------ #
# ------------------------------------------------------------------------------------------- #

def select_target_cells(
    df,
    n_priority=7,
    rho_thresh=0.2,
    n_nodrift=3,
    lfp_nb_dir=None,
    plot=True,
):
    """
    Derive priority cells and high-difference / time-independent cells from df_master,
    report which have LFP analysis notebooks, and optionally create a summary plot.

    Parameters
    ----------
    df : pd.DataFrame
        df_master from compile_experiment_results.
    n_priority : int
        Number of top-nRMSE cells to designate as priority.
    rho_thresh : float
        Max |mean temporal rho| for the time-independent group.
    n_nodrift : int
        Number of top-nRMSE time-independent cells to select.
    lfp_nb_dir : str or Path, optional
        Directory containing spe-1_c{N}_LFP_analysis.ipynb notebooks.
        If None, notebook status is not checked.
    plot : bool
        If True, produce a bar chart showing nRMSE per cell with group colours.

    Returns
    -------
    priority_nums : list of int
    nodrift_nums  : list of int
    df_cell       : pd.DataFrame  (one row per cell, sorted by nRMSE)
    """
    from pathlib import Path

    df_cell = (
        df.groupby('cell_id')[['nRMSE', 'cos_sim', 'temporal_rho', 'temporal_p']]
        .agg(
            mean_nRMSE    =('nRMSE',        'mean'),
            mean_cos_sim  =('cos_sim',       'mean'),
            mean_abs_rho  =('temporal_rho',  lambda x: x.abs().mean()),
            any_sig_drift =('temporal_p',    lambda x: (x < 0.05).any()),
        )
        .reset_index()
        .sort_values('mean_nRMSE', ascending=False)
    )

    priority_cells = df_cell.head(n_priority)['cell_id'].tolist()
    priority_nums  = sorted([int(c.lstrip('c')) for c in priority_cells])

    no_drift      = df_cell[df_cell['mean_abs_rho'] < rho_thresh].sort_values('mean_nRMSE', ascending=False)
    nodrift_cells = no_drift.head(n_nodrift)['cell_id'].tolist()
    nodrift_nums  = sorted([int(c.lstrip('c')) for c in nodrift_cells])

    print("=== All cells ranked by mean nRMSE ===")
    print(df_cell.to_string(index=False))

    print(f"\n=== Priority cells (top {n_priority} by nRMSE) ===")
    print(f"  {priority_cells}")
    print(f"  → config.PRIORITY_CELLS = {priority_nums}")

    print(f"\n=== High-diff / time-independent (|rho| < {rho_thresh}, top {n_nodrift}) ===")
    print(no_drift.head(n_nodrift).to_string(index=False))
    print(f"  → config.HIGH_DIFF_LOW_DRIFT_CELLS = {nodrift_nums}")

    if lfp_nb_dir is not None:
        nb_dir = Path(lfp_nb_dir)
        all_target = sorted(set(priority_nums) | set(nodrift_nums))
        print("\n=== LFP analysis notebook status ===")
        for n in all_target:
            nb = nb_dir / f'spe-1_c{n}_LFP_analysis.ipynb'
            status = 'EXISTS' if nb.exists() else 'MISSING — needs to be created'
            print(f'  c{n}: {status}')

    if plot:
        priority_set = set(priority_cells)
        nodrift_set  = set(nodrift_cells)
        colors = []
        for _, row in df_cell.iterrows():
            if row['cell_id'] in priority_set:
                colors.append('#D55E00')
            elif row['cell_id'] in nodrift_set:
                colors.append('#009E73')
            else:
                colors.append('#BBBBBB')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(df_cell)), df_cell['mean_nRMSE'], color=colors,
               edgecolor='white', linewidth=0.5)
        ax.set_xticks(range(len(df_cell)))
        ax.set_xticklabels(df_cell['cell_id'].tolist(), rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Mean nRMSE across clustering features', fontsize=11)
        ax.set_title(
            'Waveform cluster difference by cell\n'
            '(orange = priority, green = high-diff/time-independent)',
            fontsize=12, fontweight='bold'
        )
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color='#D55E00', label=f'Priority cells (top {n_priority})'),
            Patch(color='#009E73', label=f'High-diff / time-independent (|ρ| < {rho_thresh}, top {n_nodrift})'),
            Patch(color='#BBBBBB', label='Other cells'),
        ], fontsize=9, frameon=False)
        sns.despine(ax=ax)
        plt.tight_layout()
        plt.show()

    # Auto-update config.py so PRIORITY_CELLS and HIGH_DIFF_LOW_DRIFT_CELLS
    # stay in sync with the notebook results — no manual editing needed.
    import re
    config_path = Path(__file__).parent / 'config.py'
    config_src  = config_path.read_text()

    def _replace_list(src, var, new_list):
        pattern = rf'({re.escape(var)}\s*=\s*)\[.*?\]'
        replacement = rf'\g<1>{new_list}'
        return re.sub(pattern, replacement, src, flags=re.DOTALL)

    config_src = _replace_list(config_src, 'PRIORITY_CELLS',            priority_nums)
    config_src = _replace_list(config_src, 'HIGH_DIFF_LOW_DRIFT_CELLS', nodrift_nums)
    config_path.write_text(config_src)
    print(f"\nconfig.py updated automatically:")
    print(f"  PRIORITY_CELLS           = {priority_nums}")
    print(f"  HIGH_DIFF_LOW_DRIFT_CELLS = {nodrift_nums}")

    return priority_nums, nodrift_nums, df_cell



# ------------------------------------------------------------------------------------------- #
# ----------------------- Population waveform cluster visualization ------------------------- #
# ------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------- #
# ----------------------- Population waveform cluster visualization ------------------------- #
# ------------------------------------------------------------------------------------------- #

def plot_population_waveform_grid(
    wf_dir,
    priority_cells=None,
    nodrift_cells=None,
    cells_to_plot=None,
    xlim=(-200, 200),
    cols=6,
    figsize_per_panel=(2.6, 2.0),
):
    """
    Grid of peak-aligned average waveforms by cluster group for all (or selected) cells.

    - Orange border  = priority cell
    - Green border   = high-diff / time-independent cell
    - Gray border    = other cell
    - X-axis label only on bottom row panels
    - No figure title (avoids overlap with waveforms)

    Parameters
    ----------
    wf_dir : str or Path
        Directory containing c{N}_cluster_waveforms.pkl files.
    priority_cells : list of int, optional
        Cell numbers to highlight in orange. Defaults to config.PRIORITY_CELLS.
    nodrift_cells : list of int, optional
        Cell numbers to highlight in green. Defaults to config.HIGH_DIFF_LOW_DRIFT_CELLS.
    cells_to_plot : list of int, optional
        Subset of cell numbers to show. None = all available.
    xlim : tuple
        x-axis limits in samples from peak.
    cols : int
        Grid columns.
    figsize_per_panel : tuple
        (width, height) per panel in inches.
    """
    import pickle
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import importlib
    import config as _cfg
    # Force reload from disk every call so changes to config.py are always picked up
    importlib.reload(_cfg)

    CLUSTER_COLORS = {"low": "#0072B2", "mid": "#E69F00", "high": "#CC79A7"}
    if priority_cells is None:
        priority_cells = set(_cfg.PRIORITY_CELLS)
    else:
        priority_cells = set(priority_cells)
    if nodrift_cells is None:
        nodrift_cells = set(_cfg.HIGH_DIFF_LOW_DRIFT_CELLS)
    else:
        nodrift_cells = set(nodrift_cells)

    wf_dir = Path(wf_dir)
    pkl_files = sorted(wf_dir.glob("c*_cluster_waveforms.pkl"),
                       key=lambda p: int(p.stem.split("_")[0].lstrip("c")))

    # Build panels: (cnum, col, col_data)
    panels = []
    for pkl in pkl_files:
        cnum = int(pkl.stem.split("_")[0].lstrip("c"))
        if cells_to_plot is not None and cnum not in cells_to_plot:
            continue
        SKIP_WF_FEATS = {"log_isi", "spk_times_ms", "spk_times_idx"}
        wf_data = pickle.load(open(pkl, "rb"))
        for col, col_data in wf_data.items():
            feat_name = col.replace("_cluster", "")
            if feat_name in SKIP_WF_FEATS:
                continue
            if len([k for k in col_data if k != "t_axis"]) >= 2:
                panels.append((cnum, col, col_data))

    if not panels:
        print("No waveform panels found — run cluster notebooks first.")
        return

    rows     = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_panel[0] * cols, figsize_per_panel[1] * rows),
    )
    axes = np.array(axes).flatten()
    bottom_row_start = (rows - 1) * cols

    for ax_idx, (cnum, col, col_data) in enumerate(panels):
        ax      = axes[ax_idx]
        t_axis  = col_data["t_axis"]
        is_prio = cnum in priority_cells
        is_nd   = cnum in nodrift_cells and not is_prio

        cluster_items = [(lab, v) for lab, v in col_data.items() if lab != "t_axis"]

        # Compute valid peak amplitudes (NaN-safe) for outlier detection
        peak_amps = []
        for _, v in cluster_items:
            p = np.nanmax(np.abs(v["mean"])) if np.any(np.isfinite(v["mean"])) else np.nan
            peak_amps.append(p)
        valid_peaks = [p for p in peak_amps if np.isfinite(p)]
        amp_lo = np.median(valid_peaks) * 0.1 if len(valid_peaks) > 1 else 0.0

        feat_name  = col.replace("_cluster", "")
        show_ribbon = feat_name not in {"log_isi", "spk_times_ms", "spk_times_idx"}

        for (lab, vals), peak in zip(cluster_items, peak_amps):
            mean = vals["mean"]
            if not np.isfinite(peak) or peak < amp_lo:
                continue
            color = CLUSTER_COLORS.get(str(lab), "gray")
            t     = t_axis[:len(mean)]
            ax.plot(t, mean, color=color, lw=2.5)
            if show_ribbon and "std" in vals:
                std = vals["std"]
                ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.15)

        ax.set_xlim(xlim)
        ax.set_xticks([])
        ax.set_yticks([])

        feat = col.replace("_cluster", "")
        title_color = "#D55E00" if is_prio else ("#009E73" if is_nd else "#444444")
        ax.set_title(f"c{cnum} | {feat}", fontsize=13, fontweight="bold",
                     color=title_color, pad=4)

        ax.set_facecolor("#f5f5f5")
        for spine in ax.spines.values():
            spine.set_visible(False)

    for ax in axes[len(panels):]:
        ax.set_visible(False)

    from matplotlib.lines import Line2D
    legend_els = [Line2D([0], [0], color=c, lw=2.5, label=lab)
                  for lab, c in CLUSTER_COLORS.items()]
    fig.legend(handles=legend_els, loc="lower right", fontsize=13,
               frameon=False, ncol=3)

    plt.tight_layout(h_pad=0.4, w_pad=0.3)


# ------------------------------------------------------------------------------------------- #
#                          Within-cell vs Between-cell Waveform Distances                     #
# ------------------------------------------------------------------------------------------- #

_GROUP_CATEGORY = {
    "Within\n(all)":        ("",             ""),
    "Between\n(all)":       ("",             ""),
    "Between\nPC–PC":       ("cell type",    "#6633AA"),
    "Between\nIN–IN":       ("cell type",    "#6633AA"),
    "Between\nJuxta–Juxta": ("patch type",   "#1155AA"),
    "Between\nWC–WC":       ("patch type",   "#1155AA"),
    "Between\nIC–IC":       ("current type", "#AA3322"),
    "Between\nVC–VC":       ("current type", "#AA3322"),
}

def _add_group_category_labels(ax, group_order):
    """Draw category-row labels and underline brackets below x-tick labels."""
    trans = ax.get_xaxis_transform()  # x: data, y: axes fraction

    # Collect runs of the same non-empty category
    runs = []
    for xi, gname in enumerate(group_order):
        cat, col = _GROUP_CATEGORY.get(gname, ("", ""))
        if not cat:
            continue
        if runs and runs[-1][0] == cat:
            runs[-1] = (cat, col, runs[-1][2], xi)
        else:
            runs.append((cat, col, xi, xi))

    y_text  = -0.25   # axes-fraction below bottom of plot
    y_line  = -0.20

    for cat, col, x_lo, x_hi in runs:
        x_mid = (x_lo + x_hi) / 2
        # bracket line
        ax.annotate("", xy=(x_hi + 0.3, y_line), xytext=(x_lo - 0.3, y_line),
                    xycoords=trans, textcoords=trans,
                    arrowprops=dict(arrowstyle="-", color=col, lw=1.8))
        # label
        ax.text(x_mid, y_text, cat, ha='center', va='top',
                fontsize=12, color=col, style='italic',
                transform=trans, clip_on=False)


def _peak_align_and_trim(mean_wf, t_axis, half_win=75):
    """
    Find the sample with the largest absolute value (the peak), return the
    waveform trimmed to [peak-half_win : peak+half_win].

    Returns (trimmed_wf, peak_idx_in_t_axis).  If the peak is too close to
    the edge to fit the full window, None is returned.
    """
    peak_idx = int(np.argmax(np.abs(mean_wf)))
    lo = peak_idx - half_win
    hi = peak_idx + half_win
    if lo < 0 or hi > len(mean_wf):
        return None, None
    trimmed = mean_wf[lo:hi].copy()
    return trimmed, peak_idx


def _nrmse_cossim(a, b):
    """Normalised RMSE and cosine similarity between two equal-length 1-D arrays."""
    denom = max(np.max(np.abs(a)), np.max(np.abs(b)))
    nrmse = np.sqrt(np.mean((a - b) ** 2)) / denom if denom > 0 else np.nan
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    cos_sim = float(np.dot(a, b) / (norm_a * norm_b)) if (norm_a > 0 and norm_b > 0) else np.nan
    return nrmse, cos_sim


def compute_between_cell_waveform_distances(wf_dir, half_win=75):
    """
    For each cell compute an overall mean waveform (weighted average of cluster
    means), peak-align it, then compute all pairwise nRMSE / cos_sim between
    cells.

    Parameters
    ----------
    wf_dir : str or Path
        Directory containing c{N}_cluster_waveforms.pkl files.
    half_win : int
        Half-window (in samples) around the peak for alignment / trimming.

    Returns
    -------
    df_between : pd.DataFrame
        One row per (cell_i, cell_j) pair.  Columns: cell_i, cell_j, nRMSE, cos_sim.
    cell_mean_wfs : dict {cell_id -> trimmed mean wf array}
    """
    import pickle
    from pathlib import Path

    wf_dir = Path(wf_dir)
    pkl_files = sorted(wf_dir.glob("c*_cluster_waveforms.pkl"),
                       key=lambda p: int(p.stem.split("_")[0].lstrip("c")))

    cell_mean_wfs = {}
    for pkl in pkl_files:
        cnum = int(pkl.stem.split("_")[0].lstrip("c"))
        cell_id = f"c{cnum}"
        wf_data = pickle.load(open(pkl, "rb"))

        # Pool cluster means across all features to get one overall mean per cell.
        # Strategy: take the first available feature and compute n-weighted mean.
        pooled_sum = None
        pooled_n   = 0
        for feat, col_data in wf_data.items():
            t_axis = col_data.get("t_axis", None)
            for grp, vals in col_data.items():
                if grp == "t_axis" or not isinstance(vals, dict):
                    continue
                mean_wf = vals.get("mean")
                n       = vals.get("n", 1)
                if mean_wf is None or not np.all(np.isfinite(mean_wf)):
                    continue
                if pooled_sum is None:
                    pooled_sum = np.zeros_like(mean_wf, dtype=float)
                if len(mean_wf) == len(pooled_sum):
                    pooled_sum += mean_wf * n
                    pooled_n   += n

        if pooled_sum is None or pooled_n == 0:
            continue
        overall_mean = pooled_sum / pooled_n

        trimmed, _ = _peak_align_and_trim(overall_mean, None, half_win=half_win)
        if trimmed is None:
            continue
        cell_mean_wfs[cell_id] = trimmed

    # All pairwise distances
    cell_ids = sorted(cell_mean_wfs.keys(), key=lambda c: int(c.lstrip("c")))
    rows = []
    for i, ci in enumerate(cell_ids):
        for j, cj in enumerate(cell_ids):
            if j <= i:
                continue
            nrmse, cos_sim = _nrmse_cossim(cell_mean_wfs[ci], cell_mean_wfs[cj])
            rows.append({"cell_i": ci, "cell_j": cj, "nRMSE": nrmse, "cos_sim": cos_sim})

    return pd.DataFrame(rows), cell_mean_wfs


def plot_within_vs_between_neuron_distances(df_master, wf_dir, half_win=75,
                                            n_bootstrap=2000, alpha=0.05):
    """
    Compare within-neuron waveform cluster differences to between-neuron distances.

    Groups:
      Within-cell (all) | Between-cell (all) | Between PC–PC | Between IN–IN |
      Between Juxta–Juxta | Between WC–WC (if n >= 3 pairs)

    Lets you see whether the between-cell distribution is inflated by mixing
    cell types or recording methods.
    """
    from scipy.stats import mannwhitneyu as _mwu

    df_between, _ = compute_between_cell_waveform_distances(wf_dir, half_win=half_win)
    if df_between.empty:
        print("No between-cell distances computed — check wf_dir.")
        return

    # Per-cell metadata lookup
    meta_cols = ["cell_id", "cell_type", "patch_type", "current_type"]
    cell_meta = (df_master[meta_cols].drop_duplicates("cell_id")
                 .set_index("cell_id"))
    ct_map  = cell_meta["cell_type"].to_dict()
    pt_map  = cell_meta["patch_type"].to_dict()
    cur_map = cell_meta["current_type"].to_dict()

    def _method(pt):
        return "WC" if isinstance(pt, str) and "WC" in pt else "Juxta"

    # Within-cell (all)
    within_nrmse = pd.to_numeric(df_master["nRMSE"],   errors="coerce").dropna().values
    within_cos   = pd.to_numeric(df_master["cos_sim"], errors="coerce").dropna().values

    # Between-cell — annotate pairs
    df_b = df_between.copy()
    df_b["ct_i"]  = df_b["cell_i"].map(ct_map)
    df_b["ct_j"]  = df_b["cell_j"].map(ct_map)
    df_b["pt_i"]  = df_b["cell_i"].map(pt_map).apply(_method)
    df_b["pt_j"]  = df_b["cell_j"].map(pt_map).apply(_method)
    df_b["cur_i"] = df_b["cell_i"].map(cur_map)
    df_b["cur_j"] = df_b["cell_j"].map(cur_map)

    def _vals(mask, col):
        return df_b.loc[mask, col].dropna().values

    all_mask  = pd.Series([True] * len(df_b), index=df_b.index)
    pc_mask   = (df_b["ct_i"]  == "PC")    & (df_b["ct_j"]  == "PC")
    in_mask   = (df_b["ct_i"]  == "IN")    & (df_b["ct_j"]  == "IN")
    jux_mask  = (df_b["pt_i"]  == "Juxta") & (df_b["pt_j"]  == "Juxta")
    wc_mask   = (df_b["pt_i"]  == "WC")    & (df_b["pt_j"]  == "WC")
    ic_mask   = (df_b["cur_i"] == "IC")    & (df_b["cur_j"]  == "IC")
    vc_mask   = (df_b["cur_i"] == "VC")    & (df_b["cur_j"]  == "VC")

    wc_nrmse = _vals(wc_mask, "nRMSE")
    wc_entry = [("Between\nWC–WC", wc_nrmse, _vals(wc_mask, "cos_sim"), "#0072B2")] \
               if len(wc_nrmse) >= 3 else []

    groups = [
        ("Within\n(all)",        within_nrmse,             within_cos,                        "#555555"),
        ("Between\n(all)",       _vals(all_mask, "nRMSE"), _vals(all_mask,  "cos_sim"),        "#AAAAAA"),
        ("Between\nPC–PC",       _vals(pc_mask,  "nRMSE"), _vals(pc_mask,   "cos_sim"),        "#CC44CC"),
        ("Between\nIN–IN",       _vals(in_mask,  "nRMSE"), _vals(in_mask,   "cos_sim"),        "#00CCCC"),
        ("Between\nJuxta–Juxta", _vals(jux_mask, "nRMSE"), _vals(jux_mask,  "cos_sim"),        "#E69F00"),
        *wc_entry,
        ("Between\nIC–IC",       _vals(ic_mask,  "nRMSE"), _vals(ic_mask,   "cos_sim"),        "#009E73"),
        ("Between\nVC–VC",       _vals(vc_mask,  "nRMSE"), _vals(vc_mask,   "cos_sim"),        "#D55E00"),
    ]

    GROUP_ORDER = [g[0] for g in groups]
    PALETTE     = {g[0]: g[3] for g in groups}

    fw = max(10, len(groups) * 1.8)

    for metric_idx, metric_label in [(1, "nRMSE"), (2, "Cos Sim")]:
        fig, ax = plt.subplots(figsize=(fw, 6))
        fig.suptitle(f"Within-cell cluster vs Between-cell waveform distances — {metric_label}",
                     fontsize=22, fontweight="bold")

        rows = []
        for g in groups:
            for v in g[metric_idx]:
                rows.append({"group": g[0], "value": v})
        plot_df = pd.DataFrame(rows)

        sns.boxplot(data=plot_df, x="group", y="value", order=GROUP_ORDER,
                    palette=PALETTE, showfliers=False, width=0.55,
                    linewidth=2.5, ax=ax)
        sns.stripplot(data=plot_df, x="group", y="value", order=GROUP_ORDER,
                      palette=PALETTE, size=4, alpha=0.45, jitter=True, ax=ax)

        ax.axvline(1.5, color="#888888", lw=2.0, ls="--", alpha=0.7)
        ax.set_xlabel("")
        ax.set_ylabel(metric_label, fontsize=18)
        ax.tick_params(axis="both", labelsize=15, width=2.0, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        sns.despine(ax=ax)
        _add_group_category_labels(ax, GROUP_ORDER)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.28)
        plt.show()

        print(f"\n── {metric_label} ──")
        for g in groups:
            vals = g[metric_idx]
            if len(vals):
                print(f"  {g[0].replace(chr(10),' '):22s}: "
                      f"median={np.median(vals):.3f}  "
                      f"IQR=[{np.percentile(vals,25):.3f}, {np.percentile(vals,75):.3f}]  "
                      f"n={len(vals)}")


def plot_spike_to_avg_distances(df_master, wf_dir, spike_fit_dir, half_win=75):
    """
    Fully spike-level within vs between comparison.

    Within  : each spike vs its own cell's mean waveform.
    Between : each spike from cell A vs the mean waveform of cell B
              (both directions per pair), grouped by cell/patch/current type.

    This makes within and between directly comparable — both are
    spike-to-average distances, not average-to-average.
    """
    import pickle
    from pathlib import Path

    spike_fit_dir = Path(spike_fit_dir)
    spike_pkls = sorted(spike_fit_dir.glob("c*_spike_fit.pkl"),
                        key=lambda p: int(p.stem.split("_")[0].lstrip("c")))

    # Load all cells: peak-align and trim both mean and spike matrix
    cell_data = {}
    for pkl in spike_pkls:
        cnum    = int(pkl.stem.split("_")[0].lstrip("c"))
        cell_id = f"c{cnum}"
        try:
            sp = pickle.load(open(pkl, "rb"))
            W  = np.asarray(sp.spikes, float)
        except Exception:
            continue
        avg      = W.mean(axis=0)
        peak_idx = int(np.argmax(np.abs(avg)))
        lo, hi   = peak_idx - half_win, peak_idx + half_win
        if lo < 0 or hi > W.shape[1]:
            continue
        cell_data[cell_id] = (avg[lo:hi], W[:, lo:hi])

    n_spikes_total = sum(v[1].shape[0] for v in cell_data.values())
    print(f"Loaded {len(cell_data)} cells, {n_spikes_total:,} total spikes")

    # ── Within: each spike vs own cell mean (vectorized per cell) ──────────────
    within_nrmse_l, within_cos_l = [], []
    for mean_wf, spikes in cell_data.values():
        denom   = np.max(np.abs(mean_wf)) + 1e-12
        diff    = spikes - mean_wf
        nrmse   = np.sqrt(np.mean(diff ** 2, axis=1)) / denom
        norm_s  = np.linalg.norm(spikes, axis=1)
        norm_m  = np.linalg.norm(mean_wf) + 1e-12
        cos     = (spikes @ mean_wf) / (norm_s * norm_m + 1e-12)
        within_nrmse_l.append(nrmse)
        within_cos_l.append(cos)
    within_nrmse = np.concatenate(within_nrmse_l)
    within_cos   = np.concatenate(within_cos_l)

    # ── Between: spikes of A vs mean of B, and spikes of B vs mean of A ────────
    cell_ids = sorted(cell_data.keys(), key=lambda c: int(c.lstrip("c")))
    btw_nrmse_l, btw_cos_l = [], []
    btw_ci_l,    btw_cj_l  = [], []

    for idx_i, ci in enumerate(cell_ids):
        for idx_j, cj in enumerate(cell_ids):
            if idx_j <= idx_i:
                continue
            mean_i, spikes_i = cell_data[ci]
            mean_j, spikes_j = cell_data[cj]
            denom = max(np.max(np.abs(mean_i)), np.max(np.abs(mean_j))) + 1e-12

            # spikes of i vs mean of j
            diff_ij  = spikes_i - mean_j
            nrmse_ij = np.sqrt(np.mean(diff_ij ** 2, axis=1)) / denom
            norm_si  = np.linalg.norm(spikes_i, axis=1)
            cos_ij   = (spikes_i @ mean_j) / (norm_si * (np.linalg.norm(mean_j) + 1e-12) + 1e-12)

            # spikes of j vs mean of i
            diff_ji  = spikes_j - mean_i
            nrmse_ji = np.sqrt(np.mean(diff_ji ** 2, axis=1)) / denom
            norm_sj  = np.linalg.norm(spikes_j, axis=1)
            cos_ji   = (spikes_j @ mean_i) / (norm_sj * (np.linalg.norm(mean_i) + 1e-12) + 1e-12)

            n_tot = len(nrmse_ij) + len(nrmse_ji)
            btw_nrmse_l.append(np.concatenate([nrmse_ij, nrmse_ji]))
            btw_cos_l.append(np.concatenate([cos_ij, cos_ji]))
            btw_ci_l.extend([ci] * n_tot)
            btw_cj_l.extend([cj] * n_tot)

    df_b = pd.DataFrame({
        "cell_i":  btw_ci_l,
        "cell_j":  btw_cj_l,
        "nRMSE":   np.concatenate(btw_nrmse_l),
        "cos_sim": np.concatenate(btw_cos_l),
    })
    print(f"Between: {len(df_b):,} spike-to-avg comparisons across {len(cell_ids)} cells")

    # ── Metadata masks ──────────────────────────────────────────────────────────
    meta_cols = ["cell_id", "cell_type", "patch_type", "current_type"]
    cell_meta = df_master[meta_cols].drop_duplicates("cell_id").set_index("cell_id")
    ct_map  = cell_meta["cell_type"].to_dict()
    pt_map  = cell_meta["patch_type"].to_dict()
    cur_map = cell_meta["current_type"].to_dict()

    def _method(pt):
        return "WC" if isinstance(pt, str) and "WC" in pt else "Juxta"

    df_b["ct_i"]  = df_b["cell_i"].map(ct_map)
    df_b["ct_j"]  = df_b["cell_j"].map(ct_map)
    df_b["pt_i"]  = df_b["cell_i"].map(pt_map).apply(_method)
    df_b["pt_j"]  = df_b["cell_j"].map(pt_map).apply(_method)
    df_b["cur_i"] = df_b["cell_i"].map(cur_map)
    df_b["cur_j"] = df_b["cell_j"].map(cur_map)

    def _vals(mask, col):
        return df_b.loc[mask, col].dropna().values

    all_mask = pd.Series([True] * len(df_b), index=df_b.index)
    pc_mask  = (df_b["ct_i"] == "PC")    & (df_b["ct_j"] == "PC")
    in_mask  = (df_b["ct_i"] == "IN")    & (df_b["ct_j"] == "IN")
    jux_mask = (df_b["pt_i"] == "Juxta") & (df_b["pt_j"] == "Juxta")
    wc_mask  = (df_b["pt_i"] == "WC")    & (df_b["pt_j"] == "WC")
    ic_mask  = (df_b["cur_i"] == "IC")   & (df_b["cur_j"] == "IC")
    vc_mask  = (df_b["cur_i"] == "VC")   & (df_b["cur_j"] == "VC")

    wc_nrmse = _vals(wc_mask, "nRMSE")
    wc_entry = [("Between\nWC–WC", wc_nrmse, _vals(wc_mask, "cos_sim"), "#0072B2")] \
               if len(wc_nrmse) >= 3 else []

    groups = [
        ("Within\n(all)",        within_nrmse,             within_cos,                       "#555555"),
        ("Between\n(all)",       _vals(all_mask, "nRMSE"), _vals(all_mask,  "cos_sim"),       "#AAAAAA"),
        ("Between\nPC–PC",       _vals(pc_mask,  "nRMSE"), _vals(pc_mask,   "cos_sim"),       "#CC44CC"),
        ("Between\nIN–IN",       _vals(in_mask,  "nRMSE"), _vals(in_mask,   "cos_sim"),       "#00CCCC"),
        ("Between\nJuxta–Juxta", _vals(jux_mask, "nRMSE"), _vals(jux_mask,  "cos_sim"),       "#E69F00"),
        *wc_entry,
        ("Between\nIC–IC",       _vals(ic_mask,  "nRMSE"), _vals(ic_mask,   "cos_sim"),       "#009E73"),
        ("Between\nVC–VC",       _vals(vc_mask,  "nRMSE"), _vals(vc_mask,   "cos_sim"),       "#D55E00"),
    ]

    GROUP_ORDER = [g[0] for g in groups]
    PALETTE     = {g[0]: g[3] for g in groups}
    fw = max(10, len(groups) * 1.8)

    for metric_idx, metric_label in [(1, "nRMSE"), (2, "Cos Sim")]:
        fig, ax = plt.subplots(figsize=(fw, 6))
        fig.suptitle(
            f"Spike-to-average vs Between-cell waveform distances — {metric_label}",
            fontsize=22, fontweight="bold")

        rows = []
        for g in groups:
            for v in g[metric_idx]:
                rows.append({"group": g[0], "value": v})
        plot_df = pd.DataFrame(rows)

        sns.boxplot(data=plot_df, x="group", y="value", order=GROUP_ORDER,
                    palette=PALETTE, showfliers=False, width=0.55,
                    linewidth=2.5, ax=ax)
        sns.stripplot(data=plot_df, x="group", y="value", order=GROUP_ORDER,
                      palette=PALETTE, size=4, alpha=0.45, jitter=True, ax=ax)

        ax.axvline(1.5, color="#888888", lw=2.0, ls="--", alpha=0.7)
        ax.set_xlabel("")
        ax.set_ylabel(metric_label, fontsize=18)
        ax.tick_params(axis="both", labelsize=15, width=2.0, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(2.0)
        sns.despine(ax=ax)
        _add_group_category_labels(ax, GROUP_ORDER)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.28)
        plt.show()

        print(f"\n── {metric_label} (spike-to-avg) ──")
        for g in groups:
            vals = g[metric_idx]
            if len(vals):
                print(f"  {g[0].replace(chr(10),' '):22s}: "
                      f"median={np.median(vals):.3f}  "
                      f"IQR=[{np.percentile(vals,25):.3f}, {np.percentile(vals,75):.3f}]  "
                      f"n={len(vals)}")


# ── Temporal transition detection ─────────────────────────────────────────────

def _fit_multi_sigmoid(times_sec, labels_ord, rolling_n=50, min_frac=0.1,
                       max_sigs=3, r2_early_stop=0.95):
    """
    Fit 1–max_sigs logistic sigmoids to the rolling mean of ordinal cluster
    labels.  Model selection by AIC.

    Model: y = b + Σ_i  L_i / (1 + exp(-k_i·(t − t0_i)))

    Tries alternating-sign k initializations to catch on-off-on patterns.

    Parameters
    ----------
    times_sec  : 1-D array of recording times (seconds, sorted ascending)
    labels_ord : 1-D array of ordinal-coded cluster labels (same order)
    rolling_n  : int   rolling-mean window size in spikes (default 50)
    min_frac   : float minimum segment fraction used for t0 bounding
    max_sigs   : int   maximum sigmoids to try (default 3)

    Returns
    -------
    transitions : list of (t0_sec, k) tuples, sorted by time
    r2          : float  R² of winning model  (NaN on total failure)
    popt        : list   [b, L1, k1, t01, ...]  for winning model
    n_sigs      : int    number of sigmoids selected
    """
    from scipy.optimize import curve_fit

    t  = np.asarray(times_sec, float)
    y  = np.asarray(labels_ord, float)
    n  = len(t)
    rm = pd.Series(y).rolling(rolling_n, center=True, min_periods=1).mean().values

    # Early exit: if rolling mean has almost no variance, nothing to fit
    rm_range = float(rm.max() - rm.min())
    if rm_range < 0.15:
        lo = int(np.ceil(n * min_frac))
        fb = float(t[max(lo, n // 2)])
        return [(fb, float('nan'))], float('nan'), [float(rm.min()), rm_range, float('nan'), fb], 1

    # Downsample to at most 300 points so curve_fit stays fast on long recordings
    _MAX_PTS = 300
    if n > _MAX_PTS:
        idx = np.round(np.linspace(0, n - 1, _MAX_PTS)).astype(int)
        t_fit_arr = t[idx]
        rm_fit    = rm[idx]
    else:
        t_fit_arr = t
        rm_fit    = rm

    ss_tot = float(np.sum((rm_fit - rm_fit.mean()) ** 2))

    # Allow multi-sigma only when the rolling mean has a genuine interior
    # peak or dip — i.e. the extremum is in the middle of the recording,
    # not just at an endpoint (which would be a plain monotone transition).
    # Threshold: interior must exceed endpoints by ≥15% of total range.
    if max_sigs > 1:
        q   = max(1, len(rm_fit) // 5)          # outer-20% endpoint bands
        end_hi = max(float(rm_fit[:q].mean()), float(rm_fit[-q:].mean()))
        end_lo = min(float(rm_fit[:q].mean()), float(rm_fit[-q:].mean()))
        mid    = rm_fit[q:-q] if len(rm_fit) > 2 * q else rm_fit
        has_bump = float(mid.max()) > end_hi + 0.15 * rm_range
        has_dip  = float(mid.min()) < end_lo - 0.15 * rm_range
        if not (has_bump or has_dip):
            max_sigs = 1

    lo_t = float(t[int(np.ceil(n * min_frac))])
    hi_t = float(t[max(int(n * (1.0 - min_frac)) - 1, int(np.ceil(n * min_frac)))])

    def _make_model(ns_):
        def _m(t_, b, *lkt):
            val = np.full_like(t_, float(b))
            for i in range(ns_):
                L_, k_, t0_ = lkt[3 * i], lkt[3 * i + 1], lkt[3 * i + 2]
                val = val + L_ / (1.0 + np.exp(np.clip(-k_ * (t_ - t0_), -500, 500)))
            return val
        return _m

    best_aic         = np.inf
    best_transitions = None
    best_r2          = float('nan')
    best_popt        = None
    best_n           = 1

    k_scale = 2.0 / max(hi_t - lo_t, 1e-9)
    L_range = max(rm.max() - rm.min(), 1e-3)
    b0      = float(rm.min())

    # Only two sign patterns per ns — enough to catch monotone and alternating cases
    _SIGN_PATS = {
        1: [(1,), (-1,)],
        2: [(1, -1), (-1, 1)],
        3: [(1, -1, 1), (-1, 1, -1)],
    }
    AIC_MIN_IMPROVEMENT = 30.0  # extra sigmoid needs substantial improvement to justify complexity

    for ns in range(1, max_sigs + 1):
        n_params  = 1 + 3 * ns
        model     = _make_model(ns)
        t_cands   = np.linspace(lo_t, hi_t, ns + 2)[1:-1]
        sign_pats = _SIGN_PATS.get(ns, [(1,) * ns])

        lower = [-np.inf] + [0.0,    -np.inf, float(t[0])   ] * ns
        upper = [ np.inf] + [np.inf,  np.inf, float(t[-1])  ] * ns

        ns_best_aic = np.inf
        for signs in sign_pats:
            p0 = [b0]
            for i in range(ns):
                p0.extend([L_range / ns, signs[i] * k_scale, float(t_cands[i])])
            try:
                popt, _ = curve_fit(model, t_fit_arr, rm_fit, p0=p0,
                                    bounds=(lower, upper), maxfev=5000)
                resid = rm_fit - model(t_fit_arr, *popt)
                rss   = float(np.sum(resid ** 2))
                nf    = len(rm_fit)
                aic   = nf * np.log(max(rss / nf, 1e-300)) + 2 * n_params
                r2    = 1.0 - rss / ss_tot if ss_tot > 0 else 0.0

                if aic < ns_best_aic:
                    ns_best_aic = aic

                if aic < best_aic:
                    best_aic  = aic
                    best_r2   = float(r2)
                    best_popt = list(popt)
                    best_n    = ns
                    transitions = []
                    for i in range(ns):
                        k_i  = float(popt[2 + 3 * i])
                        t0_i = float(np.clip(popt[3 + 3 * i], lo_t, hi_t))
                        transitions.append((t0_i, k_i))
                    transitions.sort(key=lambda x: x[0])
                    best_transitions = transitions
            except Exception:
                continue

        # If single sigmoid already good enough, don't try more (avoids overfitting noise)
        if ns == 1 and best_r2 >= r2_early_stop:
            break
        # Stop early if going from ns-1 → ns didn't help enough
        if ns > 1 and (np.isinf(ns_best_aic) or best_n < ns or
                       ns_best_aic > best_aic + AIC_MIN_IMPROVEMENT):
            break

    if best_transitions is None:
        lo = int(np.ceil(n * min_frac))
        fb = float(t[max(lo, n // 2)])
        return [(fb, float('nan'))], float('nan'), [b0, L_range, float('nan'), fb], 1

    return best_transitions, best_r2, best_popt, best_n


def find_temporal_transitions(cluster_pickle_dir, r2_thresh=0.80, min_k=0.02,
                               min_spikes=20, min_frac=0.1, rolling_n=50,
                               max_sigs=3, save_path=None, force=False):
    """
    For each (cell × spike_feature) pair with enough spikes, fit 1–max_sigs
    logistic sigmoids (AIC selection) to the rolling mean of ordinal cluster
    labels.  Keep the result only if the overall R² ≥ r2_thresh and the
    sharpest sigmoid has |k| ≥ min_k.  Returns one row per inflection point,
    so a 2-sigmoid fit produces 2 rows for the same (cell, feature).

    If save_path already exists and force=False, the cached DataFrame is loaded
    and returned immediately without recomputing.

    Parameters
    ----------
    cluster_pickle_dir : str    path to cluster_pickles/
    r2_thresh          : float  minimum overall R² to keep (default 0.80)
    min_k              : float  minimum |k| (1/s) of the sharpest sigmoid (default 0.02)
    min_spikes         : int    minimum valid spikes required (default 20)
    min_frac           : float  segment fraction for t0 bounding (default 0.1)
    rolling_n          : int    rolling-mean window in spikes (default 50)
    max_sigs           : int    maximum sigmoids per cell/feature (default 3)
    save_path          : str or None  if given, pickle the result DataFrame
    force              : bool   if True, recompute even when save_path exists (default False)

    Returns
    -------
    df_transitions : pd.DataFrame  one row per inflection point, columns:
        cell_id, spike_feature,
        n_spikes, n_transitions, transition_index,
        transition_time_ms, transition_sharpness_k,
        sigmoid_r2, sigmoid_popt,
        cluster_before, cluster_after,
        frac_dominant_before, frac_dominant_after,
        mean_time_before_ms, mean_time_after_ms
    """
    import glob
    import os

    if save_path and not force and os.path.exists(save_path):
        df_cached = pd.read_pickle(save_path)
        print(f'Loaded cached transitions from {save_path}  '
              f'({len(df_cached)} rows — pass force=True to recompute)')
        return df_cached

    ORDINAL = {'low': 0, 'mid': 1, 'high': 2,
               'Low': 0, 'Mid': 1, 'High': 2,
               'low-mid': 0.5, 'low-high': 1.0, 'mid-high': 1.5}

    def _to_ord(label):
        if isinstance(label, (int, float)):
            return float(label)
        l = str(label).strip().lower()
        for k, v in ORDINAL.items():
            if k.lower() == l:
                return float(v)
        return float('nan')

    def _dominant(arr):
        vals, cnts = np.unique(arr, return_counts=True)
        return vals[np.argmax(cnts)], float(np.max(cnts) / len(arr))

    pkl_files = sorted(glob.glob(f'{cluster_pickle_dir}/c*_cluster_df.pkl'))
    rows = []

    for pkl_path in pkl_files:
        cid = pkl_path.split('/')[-1].replace('_cluster_df.pkl', '')
        df  = pd.read_pickle(pkl_path)

        cluster_cols = [c for c in df.columns if c.endswith('_cluster')
                        and not c.startswith('spk_times')]

        for col in cluster_cols:
            feat = col.replace('_cluster', '')

            labels     = df[col].dropna()
            times      = df.loc[labels.index, 'spk_times_ms']
            labels_ord = labels.map(_to_ord).values
            valid      = np.isfinite(labels_ord)
            if valid.sum() < min_spikes:
                continue

            sort_idx     = np.argsort(times.values[valid])
            times_ms_s   = times.values[valid][sort_idx]
            labels_s     = labels_ord[valid][sort_idx]
            raw_labels_s = labels.values[valid][sort_idx]

            transitions, r2, popt, n_sigs = _fit_multi_sigmoid(
                times_ms_s / 1000.0, labels_s,
                rolling_n=rolling_n, min_frac=min_frac, max_sigs=max_sigs,
                r2_early_stop=0.95,
            )

            # Filter: good overall fit AND at least one sigmoid must be sharp enough
            if not np.isfinite(r2) or r2 < r2_thresh:
                continue
            finite_ks = [abs(k) for _, k in transitions if np.isfinite(k)]
            if not finite_ks or max(finite_ks) < min_k:
                continue

            n_v = len(times_ms_s)
            lo  = int(np.ceil(n_v * min_frac))
            hi  = n_v - lo

            # Segment boundaries for before/after per transition
            t0_ms_list = [t0 * 1000.0 for t0, _ in transitions]
            boundaries_ms = [times_ms_s[0]] + t0_ms_list + [times_ms_s[-1]]

            for ti, (t0_sec, k_val) in enumerate(transitions):
                t0_ms_i  = t0_sec * 1000.0
                cp       = int(np.clip(np.argmin(np.abs(times_ms_s - t0_ms_i)), lo, hi))
                prev_ms  = boundaries_ms[ti]
                next_ms  = boundaries_ms[ti + 2]

                mask_b = (times_ms_s >= prev_ms) & (times_ms_s <  t0_ms_i)
                mask_a = (times_ms_s >= t0_ms_i) & (times_ms_s <= next_ms)

                if mask_b.sum() == 0 or mask_a.sum() == 0:
                    continue

                cl_before, frac_before = _dominant(raw_labels_s[mask_b])
                cl_after,  frac_after  = _dominant(raw_labels_s[mask_a])

                if cl_before == cl_after:
                    continue

                rows.append(dict(
                    cell_id               = cid,
                    spike_feature         = feat,
                    n_spikes              = int(valid.sum()),
                    n_transitions         = n_sigs,
                    transition_index      = ti,
                    transition_time_ms    = float(t0_ms_i),
                    transition_sharpness_k= float(k_val),
                    sigmoid_r2            = float(r2),
                    sigmoid_popt          = popt,
                    cluster_before        = cl_before,
                    cluster_after         = cl_after,
                    frac_dominant_before  = float(frac_before),
                    frac_dominant_after   = float(frac_after),
                    mean_time_before_ms   = float(np.mean(times_ms_s[mask_b])),
                    mean_time_after_ms    = float(np.mean(times_ms_s[mask_a])),
                ))

    df_transitions = pd.DataFrame(rows)
    n_pairs = df_transitions[['cell_id','spike_feature']].drop_duplicates().shape[0] if len(df_transitions) else 0

    print(f'Found {len(df_transitions)} transition events across '
          f'{n_pairs} (cell, feature) pairs '
          f'(R² ≥ {r2_thresh}, |k| ≥ {min_k} /s, max {max_sigs} sigmoids)')

    if save_path and len(df_transitions):
        df_transitions.to_pickle(save_path)
        print(f'Saved: {save_path}')

    return df_transitions


def plot_temporal_transitions(df_transitions, cluster_pickle_dir,
                               n_cols=4, rolling_n=50):
    """
    For each detected transition, plot cluster label vs spike time with the
    changepoint marked. Shows the rolling mean cluster label alongside individual
    spike labels to make the transition visible.

    Parameters
    ----------
    df_transitions   : output of find_temporal_transitions
    cluster_pickle_dir : str  path to cluster_pickles/
    n_cols           : int  subplot columns (default 4)
    rolling_n        : int  window for rolling mean (default 50 spikes)
    """
    if len(df_transitions) == 0:
        print('No transitions to plot.')
        return

    sns.set_theme(style='ticks', font_scale=1.3, rc={
        'axes.linewidth':    2.5,
        'xtick.major.width': 2.5,
        'ytick.major.width': 2.5,
        'xtick.major.size':  6,
        'ytick.major.size':  6,
        'lines.linewidth':   2.5,
    })
    _FS_TICK, _FS_AX, _FS_SUB = 11, 12, 17
    trans_color = '#CC79A7'  # Okabe-Ito reddish purple (colour-blind safe, distinct from cluster colours)

    ORDINAL = {'low': 0, 'mid': 1, 'high': 2,
               'Low': 0, 'Mid': 1, 'High': 2}
    CLR = {'low': '#0072B2', 'mid': '#009E73', 'high': '#D55E00',
           'Low': '#0072B2', 'Mid': '#009E73', 'High': '#D55E00'}

    def _to_ord(l):
        return ORDINAL.get(str(l).strip(), 1)

    # One panel per (cell_id, spike_feature) group — supports multi-sigmoid
    groups   = list(df_transitions.groupby(['cell_id', 'spike_feature'], sort=False))
    n_panels = len(groups)

    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 4.8, n_rows * 3.5),
                             squeeze=False)

    def _multi_sigmoid(t_, popt_):
        b   = popt_[0]
        val = np.full_like(t_, float(b))
        ns  = (len(popt_) - 1) // 3
        for i in range(ns):
            L_, k_, t0_ = popt_[1+3*i], popt_[2+3*i], popt_[3+3*i]
            val = val + L_ / (1.0 + np.exp(np.clip(-k_*(t_-t0_), -500, 500)))
        return val

    for ax_i, ((cid, feat), group) in enumerate(groups):
        ax    = axes[ax_i // n_cols][ax_i % n_cols]
        first = group.iloc[0]
        r2_val = float(first.get('sigmoid_r2', float('nan')))
        popt   = first.get('sigmoid_popt')
        n_sigs = int(first.get('n_transitions', 1))

        pkl = f'{cluster_pickle_dir}/{cid}_cluster_df.pkl'
        try:
            df = pd.read_pickle(pkl)
        except FileNotFoundError:
            ax.set_visible(False)
            continue

        col = f'{feat}_cluster'
        if col not in df.columns:
            ax.set_visible(False)
            continue

        sub    = df[['spk_times_ms', col]].dropna()
        sub    = sub.sort_values('spk_times_ms').reset_index(drop=True)
        times  = sub['spk_times_ms'].values / 1000.0
        labels = sub[col].values
        ord_l  = np.array([_to_ord(l) for l in labels], dtype=float)

        # Scatter
        for lbl in np.unique(labels):
            mask = labels == lbl
            ax.scatter(times[mask], ord_l[mask],
                       color=CLR.get(str(lbl), 'gray'),
                       s=14, alpha=0.45, linewidths=0, zorder=2)

        # Rolling mean
        rm = pd.Series(ord_l).rolling(rolling_n, center=True, min_periods=1).mean()
        ax.plot(times, rm.values, color='black', lw=3.5, zorder=5, alpha=0.6)

        # Multi-sigmoid curve from stored popt
        if popt is not None:
            try:
                t_fit = np.linspace(times[0], times[-1], 400)
                ax.plot(t_fit, _multi_sigmoid(t_fit, popt),
                        color=trans_color, lw=3.5, zorder=6)
            except Exception:
                pass

        # One dashed line + label per transition in the group
        title_parts = []
        for _, tr_row in group.sort_values('transition_index').iterrows():
            t_tr  = float(tr_row['transition_time_ms'])
            k_val = float(tr_row['transition_sharpness_k'])
            cl_b  = tr_row['cluster_before']
            cl_a  = tr_row['cluster_after']
            ax.axvline(t_tr / 1000.0, color=trans_color, lw=2.5, ls='--',
                       zorder=7, alpha=0.85)
            ax.text(t_tr / 1000.0, 2.15, f'  {t_tr/1000:.1f}s',
                    color=trans_color, fontsize=_FS_TICK - 1,
                    fontweight='bold', va='top', clip_on=True)
            k_str = f'k={k_val:.3f}/s' if np.isfinite(k_val) else 'k=?'
            title_parts.append(f'{cl_b}→{cl_a} {k_str}')

        r2_str  = f'R²={r2_val:.2f}' if np.isfinite(r2_val) else 'R²=?'
        sig_tag = f'({n_sigs}σ)' if n_sigs > 1 else ''
        ax.set_title(f'{cid}  ·  {feat}  {sig_tag}\n{r2_str}  ' + '   '.join(title_parts),
                     fontsize=_FS_AX, fontweight='bold', color='black')

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['low', 'mid', 'high'], fontsize=_FS_TICK, color='black')
        ax.set_xlabel('Time (s)', fontsize=_FS_AX, color='black')
        ax.tick_params(axis='both', labelsize=_FS_TICK, colors='black')
        sns.despine(ax=ax)

    # Hide unused axes
    for ax_i in range(n_panels, n_rows * n_cols):
        axes[ax_i // n_cols][ax_i % n_cols].set_visible(False)

    fig.suptitle('Temporal transitions in cluster membership\n'
                 'Black = rolling mean   |   Purple curve = sigmoid fit   |   '
                 'Purple dashed = sigmoid inflection (t₀)   |   Colour = cluster label',
                 fontsize=_FS_SUB, fontweight='bold', color='black', y=1.01)
    fig.tight_layout()
    plt.show()


def plot_temporal_transitions_highlights(df_transitions, cluster_pickle_dir,
                                           selections, n_cols=3, rolling_n=50):
    """
    Cartoony highlight-reel version of plot_temporal_transitions: shows only a
    hand-picked subset of (cell_id, spike_feature) transitions, laid out in a
    small grid with much bigger fonts, dots, and lines — for slides.

    Parameters
    ----------
    df_transitions : output of find_temporal_transitions
    cluster_pickle_dir : str  path to cluster_pickles/
    selections     : list[tuple[str, str]]
        (cell_id, spike_feature) pairs to plot, e.g.
        [('c42', 'peak_width'), ('c20', 'peak_amp'), ...]
    n_cols         : int  subplot columns (default 3)
    rolling_n      : int  window for rolling mean (default 50 spikes)
    """
    sns.set_theme(style='ticks', font_scale=2.2, rc={
        'axes.linewidth':    4.5,
        'xtick.major.width': 4.5,
        'ytick.major.width': 4.5,
        'xtick.major.size':  10,
        'ytick.major.size':  10,
        'lines.linewidth':   4.5,
    })
    _FS_TICK, _FS_AX, _FS_SUB = 22, 24, 30
    trans_color = '#CC79A7'  # Okabe-Ito reddish purple (colour-blind safe, distinct from cluster colours)

    ORDINAL = {'low': 0, 'mid': 1, 'high': 2,
               'Low': 0, 'Mid': 1, 'High': 2}
    CLR = {'low': '#0072B2', 'mid': '#009E73', 'high': '#D55E00',
           'Low': '#0072B2', 'Mid': '#009E73', 'High': '#D55E00'}

    def _to_ord(l):
        return ORDINAL.get(str(l).strip(), 1)

    rows = []
    for cid, feat in selections:
        match = df_transitions[(df_transitions['cell_id'] == cid) &
                               (df_transitions['spike_feature'] == feat)]
        if match.empty:
            print(f'No transition found for ({cid}, {feat}) — skipping.')
            continue
        rows.append(match.iloc[0])

    if not rows:
        print('No matching transitions found.')
        return

    n_rows = int(np.ceil(len(rows) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 7.5, n_rows * 5.5),
                             squeeze=False)

    for ax_i, row in enumerate(rows):
        ax = axes[ax_i // n_cols][ax_i % n_cols]
        cid   = row['cell_id']
        feat  = row['spike_feature']
        t_tr  = row['transition_time_ms']
        cl_b  = row['cluster_before']
        cl_a  = row['cluster_after']
        k_val = row.get('transition_sharpness_k', float('nan'))
        r2_val= row.get('sigmoid_r2', float('nan'))

        pkl = f'{cluster_pickle_dir}/{cid}_cluster_df.pkl'
        try:
            df = pd.read_pickle(pkl)
        except FileNotFoundError:
            ax.set_visible(False)
            continue

        col = f'{feat}_cluster'
        if col not in df.columns:
            ax.set_visible(False)
            continue

        sub = df[['spk_times_ms', col]].dropna()
        sub = sub.sort_values('spk_times_ms').reset_index(drop=True)
        times  = sub['spk_times_ms'].values / 1000.0   # → seconds
        labels = sub[col].values
        ord_l  = np.array([_to_ord(l) for l in labels], dtype=float)

        # Scatter: individual spike cluster labels — big, bold dots
        for lbl in np.unique(labels):
            mask = labels == lbl
            ax.scatter(times[mask], ord_l[mask],
                       color=CLR.get(str(lbl), 'gray'),
                       s=60, alpha=0.55, linewidths=0, zorder=2)

        # Rolling mean
        rm = pd.Series(ord_l).rolling(rolling_n, center=True, min_periods=1).mean()
        ax.plot(times, rm.values, color='black', lw=6.5, zorder=5)

        # Transition line
        t_tr_s = t_tr / 1000.0
        ax.axvline(t_tr_s, color=trans_color, lw=6.0, ls='--', zorder=6)

        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['low', 'mid', 'high'], fontsize=_FS_TICK, color='black')
        ax.set_xlabel('Time (s)', fontsize=_FS_AX, color='black')
        k_str  = f'k={k_val:.3f}/s' if np.isfinite(k_val) else 'k=?'
        r2_str = f'R²={r2_val:.2f}' if np.isfinite(r2_val) else ''
        ax.set_title(f'{cid}  ·  {feat}\n{r2_str}  {k_str}  {cl_b}→{cl_a}',
                     fontsize=_FS_AX, fontweight='bold', color='black', pad=14)
        ax.tick_params(axis='both', labelsize=_FS_TICK, colors='black')
        sns.despine(ax=ax)

    # Hide unused axes
    for ax_i in range(len(rows), n_rows * n_cols):
        axes[ax_i // n_cols][ax_i % n_cols].set_visible(False)

    fig.suptitle('Temporal transitions in cluster membership',
                 fontsize=_FS_SUB, fontweight='bold', color='black', y=1.06)

    handles = [
        plt.Line2D([0], [0], color='black', lw=6.5,
                   label='Rolling mean of cluster label'),
        plt.Line2D([0], [0], marker='s', linestyle='', color=trans_color,
                   markeredgecolor='black', markeredgewidth=2.0, markersize=18,
                   label='Detected changepoint'),
        plt.Line2D([0], [0], marker='o', linestyle='', color=CLR['low'],
                   markersize=18, label='Low'),
        plt.Line2D([0], [0], marker='o', linestyle='', color=CLR['mid'],
                   markersize=18, label='Mid'),
        plt.Line2D([0], [0], marker='o', linestyle='', color=CLR['high'],
                   markersize=18, label='High'),
    ]
    leg = fig.legend(handles=handles, fontsize=_FS_TICK, frameon=False,
                     loc='upper center', bbox_to_anchor=(0.5, 1.0),
                     ncol=5, columnspacing=2.0, handlelength=2.5)
    for text in leg.get_texts():
        text.set_color('black')

    fig.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()


# ── LFP block analysis at transition times ───────────────────────────────────

def analyze_lfp_at_transitions(
    df_transitions,
    lfp_npy_dir,
    fs=2500,
    freq_range=(1, 90),
    save_path=None,
    force=False,
):
    """
    For each (cell, feature) in df_transitions, cut the continuous LFP recording
    at the transition time(s) to create temporal blocks.  For each block compute:

        mean_amp  : mean LFP amplitude (µV, raw)
        std_amp   : std  LFP amplitude (µV, raw)
        freqs     : frequency axis (Hz)
        psd       : Welch PSD of the block (µV²/Hz)
        exponent  : specparam aperiodic exponent
        offset    : specparam aperiodic offset
        theta_auc : specparam theta peak AUC (4–10 Hz)
        slow_gamma_auc / high_gamma_auc / total_gamma_auc : specparam gamma AUC (30–60 / 60–80 / 30–80 Hz)

    Parameters
    ----------
    lfp_npy_dir : str  path to filt_lfp_recordings/ containing c{N}_lfp.npy
    """
    import os
    import pickle
    from scipy.signal import welch
    from specparam import SpectralModel
    try:
        from specparam.utils import interpolate_spectrum
    except ImportError:
        from fooof.utils import interpolate_spectrum

    if save_path and not force and os.path.exists(save_path):
        with open(save_path, 'rb') as _f:
            cached = pickle.load(_f)
        print(f'Loaded cached LFP block results from {save_path}  '
              f'(pass force=True to recompute)')
        return cached

    BLOCK_COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7']  # Okabe-Ito
    results = {}

    for cid, cell_grp in df_transitions.groupby('cell_id'):
        npy_path = os.path.join(lfp_npy_dir, f'{cid}_lfp.npy')
        if not os.path.exists(npy_path):
            print(f'  skip {cid}: no LFP file at {npy_path}')
            continue

        lfp_raw = np.load(npy_path).astype(float)
        n_samps = len(lfp_raw)

        for feat, feat_grp in cell_grp.groupby('spike_feature'):
            t0s_ms   = sorted(feat_grp['transition_time_ms'].dropna().values)
            t0_samps = [int(round(t * fs / 1000.0)) for t in t0s_ms]
            boundaries = [0] + t0_samps + [n_samps]

            block_results = []
            for b_idx in range(len(boundaries) - 1):
                i0, i1 = boundaries[b_idx], boundaries[b_idx + 1]
                if i1 - i0 < int(fs * 2.0):
                    continue

                block = lfp_raw[i0:i1]
                mean_amp = float(np.mean(block))
                std_amp  = float(np.std(block))

                nperseg = min(len(block), int(fs * 4.0))
                freqs_w, psd_w = welch(block, fs=fs, nperseg=nperseg,
                                       noverlap=nperseg // 2, scaling='density')
                fmask     = (freqs_w >= freq_range[0]) & (freqs_w <= freq_range[1])
                freqs_out = freqs_w[fmask]
                psd_out   = psd_w[fmask]

                sm = SpectralModel(
                    aperiodic_mode='fixed', peak_width_limits=(4.0, 8.0),
                    max_n_peaks=4, min_peak_height=0.0,
                    peak_threshold=1.5, verbose=False,
                )
                try:
                    freqs_sm, psd_sm = interpolate_spectrum(freqs_out, psd_out, [58, 62])
                    sm.fit(freqs_sm, psd_sm, freq_range=freq_range)
                    exponent  = float(sm.get_params('aperiodic_params', 'exponent'))
                    offset    = float(sm.get_params('aperiodic_params', 'offset'))
                    full_log  = np.asarray(sm.get_model(component='full',      space='log'))
                    ape_log   = np.asarray(sm.get_model(component='aperiodic', space='log'))
                    freqs_fit = np.asarray(sm.freqs)
                    # AUC = specparam peak model above aperiodic (NaN-safe: missing → 0 contribution)
                    def _band_auc(fl, al, ff, flo, fhi):
                        m = (ff >= flo) & (ff <= fhi)
                        d = np.where(np.isfinite(fl[m] - al[m]), fl[m] - al[m], 0.0)
                        return float(np.trapz(np.clip(d, 0, None), ff[m]))
                    theta_auc      = _band_auc(full_log, ape_log, freqs_fit,  4,  15)
                    slow_gamma_auc = _band_auc(full_log, ape_log, freqs_fit, 30,  60)
                    high_gamma_auc = _band_auc(full_log, ape_log, freqs_fit, 60,  80)
                    total_gamma_auc= _band_auc(full_log, ape_log, freqs_fit, 30,  80)
                    _peaks = np.atleast_2d(sm.peak_params_) if sm.n_peaks_ > 0 else np.empty((0, 3))
                    _pk_str = ', '.join(f'CF={p[0]:.1f}Hz PW={p[1]:.3f} BW={p[2]:.1f}' for p in _peaks) or 'none'
                    print(f'  [{cid}|{feat}|blk{b_idx}] exp={exponent:.2f}  '
                          f'θ={theta_auc:.4f}  sγ={slow_gamma_auc:.4f}  '
                          f'hγ={high_gamma_auc:.4f}  totγ={total_gamma_auc:.4f}  '
                          f'n_peaks={sm.n_peaks_}  peaks=[{_pk_str}]', flush=True)
                except Exception as _e:
                    print(f"  specparam warn [{cid}|{feat}|blk{b_idx}]: {_e}", flush=True)
                    exponent = offset = float('nan')
                    theta_auc = slow_gamma_auc = high_gamma_auc = total_gamma_auc = float('nan')
                    freqs_fit = full_log = ape_log = None

                block_results.append({
                    'block':      b_idx,
                    'color':      BLOCK_COLORS[b_idx % len(BLOCK_COLORS)],
                    'label':      f'Block {b_idx + 1}',
                    't_start_ms': i0 / fs * 1000.0,
                    't_end_ms':   i1 / fs * 1000.0,
                    'n_samples':  i1 - i0,
                    'mean_amp':   mean_amp,
                    'std_amp':    std_amp,
                    'freqs':      freqs_out,
                    'psd':        psd_out,
                    'freqs_fit':  freqs_fit,
                    'full_log':   full_log,
                    'ape_log':    ape_log,
                    'exponent':   exponent,
                    'offset':     offset,
                    'theta_auc':       theta_auc,
                    'slow_gamma_auc':  slow_gamma_auc,
                    'high_gamma_auc':  high_gamma_auc,
                    'total_gamma_auc': total_gamma_auc,
                })

            if block_results:
                results[(cid, feat)] = block_results
                dur = [f"{b['n_samples']/fs:.0f}s" for b in block_results]
                print(f'  {cid:6s}  {feat:25s}  blocks: {dur}')

    if save_path:
        with open(save_path, 'wb') as _f:
            pickle.dump(results, _f)
        print(f'Saved: {save_path}')

    return results


def plot_lfp_transition_validation(
    df_transitions,
    lfp_npy_dir,
    cluster_pickle_dir,
    fs=2500,
    rolling_n=50,
    n_cols=4,
):
    """
    Grid validation plot: all (cell, feature) pairs in one figure.
    Each column = one (cell, feature); 2 rows per column:
      Row 0: full spike cluster rolling mean with transition line(s)
      Row 1: full raw LFP recording with transition line(s) and block shading

    Confirms that spike-derived transition times map to the correct LFP samples.
    """
    import matplotlib.gridspec as gridspec
    import seaborn as sns

    ORDINAL   = {'low': 0, 'mid': 1, 'high': 2, 'Low': 0, 'Mid': 1, 'High': 2}
    CLR       = {'low': '#0072B2', 'mid': '#009E73', 'high': '#D55E00',
                 'Low': '#0072B2', 'Mid': '#009E73', 'High': '#D55E00'}
    BLK_CLRS  = ['#2980b9', '#e74c3c', '#27ae60', '#8e44ad']
    TRANS_CLR = 'crimson'
    _DS       = 10  # downsample factor for LFP trace

    pairs = list(df_transitions.groupby(['cell_id', 'spike_feature']))
    n_pairs = len(pairs)
    n_cols  = min(n_cols, n_pairs)
    n_rows  = int(np.ceil(n_pairs / n_cols))

    # Cache loaded LFP per cell to avoid re-loading
    _lfp_cache = {}

    fig = plt.figure(figsize=(n_cols * 5.5, n_rows * 4.5),
                     constrained_layout=False)
    outer_gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                                 hspace=0.15, wspace=0.35)

    for p_idx, ((cid, feat), grp) in enumerate(pairs):
        row, col = divmod(p_idx, n_cols)
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[row, col],
            height_ratios=[1, 1.5], hspace=0.25,
        )
        ax_spk = fig.add_subplot(inner_gs[0])
        ax_lfp = fig.add_subplot(inner_gs[1])

        t0s_ms = sorted(grp['transition_time_ms'].dropna().values)

        # ── spike rolling mean ──────────────────────────────────────────
        pkl = os.path.join(cluster_pickle_dir, f'{cid}_cluster_df.pkl')
        try:
            df_cl = pd.read_pickle(pkl)
        except FileNotFoundError:
            ax_spk.set_visible(False); ax_lfp.set_visible(False); continue
        cl_col = f'{feat}_cluster'
        if cl_col not in df_cl.columns:
            ax_spk.set_visible(False); ax_lfp.set_visible(False); continue
        sub   = df_cl[['spk_times_ms', cl_col]].dropna().sort_values('spk_times_ms')
        spk_t = sub['spk_times_ms'].values / 1000.0
        lbls  = sub[cl_col].values
        ord_l = np.array([ORDINAL.get(str(l).strip(), 1) for l in lbls], dtype=float)
        rm    = pd.Series(ord_l).rolling(rolling_n, center=True, min_periods=1).mean().values

        for lbl in np.unique(lbls):
            m = lbls == lbl
            ax_spk.scatter(spk_t[m], ord_l[m], color=CLR.get(str(lbl), 'gray'),
                           s=5, alpha=0.2, linewidths=0)
        ax_spk.plot(spk_t, rm, color='black', lw=1.8)
        for t0_ms in t0s_ms:
            ax_spk.axvline(t0_ms / 1000.0, color=TRANS_CLR, lw=1.5, ls='--')
        ax_spk.set_yticks([0, 1, 2])
        ax_spk.set_yticklabels(['low', 'mid', 'high'], fontsize=7)
        ax_spk.set_title(f'{cid}  ·  {feat}', fontsize=9, fontweight='bold')
        ax_spk.tick_params(labelbottom=False)
        sns.despine(ax=ax_spk)

        # ── full LFP recording ──────────────────────────────────────────
        if cid not in _lfp_cache:
            npy_path = os.path.join(lfp_npy_dir, f'{cid}_lfp.npy')
            if not os.path.exists(npy_path):
                ax_spk.set_visible(False); ax_lfp.set_visible(False); continue
            _lfp_cache[cid] = np.load(npy_path).astype(float)
        lfp_raw = _lfp_cache[cid]
        n_samps = len(lfp_raw)
        lfp_t_s = np.arange(n_samps) / fs
        lfp_dur_s = n_samps / fs

        # Add recording length to spike panel title
        ax_spk.set_title(f'{cid}  ·  {feat}\nrec: {lfp_dur_s:.0f} s', fontsize=9, fontweight='bold')

        boundaries_s = [0.0] + [t / 1000.0 for t in t0s_ms] + [lfp_t_s[-1]]
        for b_i in range(len(boundaries_s) - 1):
            ax_lfp.axvspan(boundaries_s[b_i], boundaries_s[b_i + 1],
                           alpha=0.08, color=BLK_CLRS[b_i % len(BLK_CLRS)])
        ax_lfp.plot(lfp_t_s[::_DS], lfp_raw[::_DS], color='black', lw=0.35, alpha=0.8)
        for t0_ms in t0s_ms:
            ax_lfp.axvline(t0_ms / 1000.0, color=TRANS_CLR, lw=1.5, ls='--',
                           label=f't₀={t0_ms/1000:.1f}s')
        # Annotate recording duration on the LFP panel
        ax_lfp.text(0.98, 0.97, f'{lfp_dur_s:.0f} s', transform=ax_lfp.transAxes,
                    fontsize=8, ha='right', va='top', color='dimgray')
        ax_lfp.legend(fontsize=7, frameon=False, loc='upper left')
        ax_lfp.set_xlabel('Time (s)', fontsize=8)
        ax_lfp.set_ylabel('LFP (µV)', fontsize=8)
        sns.despine(ax=ax_lfp)

    fig.suptitle('LFP transition validation — spike t₀ mapped to continuous LFP',
                 fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.show()


def plot_lfp_block_comparison(
    lfp_block_results,
    df_transitions,
    cluster_pickle_dir,
    rolling_n=50,
):
    """
    For each (cell, feature) in lfp_block_results, plot a 3-panel figure:
      Left  : rolling mean of cluster labels with transition time(s) marked
      Middle: overlaid PSDs (raw + specparam fit) per block
      Right : grouped bar chart of exponent, theta AUC, gamma AUC, std per block
    """
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    from specparam import SpectralModel
    try:
        from specparam.utils import interpolate_spectrum
    except ImportError:
        from fooof.utils import interpolate_spectrum

    ORDINAL = {'low': 0, 'mid': 1, 'high': 2, 'Low': 0, 'Mid': 1, 'High': 2}
    CLR_CLUSTER = {'low': '#0072B2', 'mid': '#009E73', 'high': '#D55E00',
                   'Low': '#0072B2', 'Mid': '#009E73', 'High': '#D55E00'}

    _sm_kwargs = dict(aperiodic_mode='fixed', peak_width_limits=(4.0, 8.0),
                      max_n_peaks=4, min_peak_height=0.0, peak_threshold=1.5, verbose=False)

    bar_specs = [
        ('mean_amp',        'Mean amp\n(µV)'),
        ('std_amp',         'Std amp\n(µV)'),
        ('exponent',        'Exponent'),
        ('offset',          'Offset'),
        ('theta_auc',       'θ AUC\n4–15 Hz'),
        ('total_gamma_auc', 'Total γ\n30–80 Hz'),
        ('slow_gamma_auc',  'Slow γ\n30–60 Hz'),
        ('high_gamma_auc',  'High γ\n60–80 Hz'),
    ]
    _auc_keys = {'theta_auc', 'slow_gamma_auc', 'high_gamma_auc', 'total_gamma_auc'}

    # Pre-scan all blocks for global y-limits (bar features + PSD)
    _gvals = {key: [] for key, _ in bar_specs}
    _psd_mins, _psd_maxs = [], []
    for _blks in lfp_block_results.values():
        for _blk in _blks:
            for key, _ in bar_specs:
                v = _blk.get(key, float('nan'))
                if np.isfinite(v):
                    _gvals[key].append(v)
            _p = _blk.get('psd', _blk.get('mean_psd'))
            if _p is not None:
                _p = np.asarray(_p)
                _p = _p[_p > 0]
                if len(_p):
                    _psd_mins.append(float(np.min(_p)))
                    _psd_maxs.append(float(np.max(_p)))
    _global_ylims = {}
    for key, vals in _gvals.items():
        if not vals:
            _global_ylims[key] = None
            continue
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, abs(hi) * 0.05, 1e-6)
        top  = max(hi, 0) + span * 0.45  # headroom above 0 for bracket even when all vals negative
        bot  = 0.0 if key in _auc_keys else min(lo, 0) - span * 0.1
        _global_ylims[key] = (bot, top)
    _psd_ylim = (min(_psd_mins) * 0.5, max(_psd_maxs) * 2.0) if _psd_mins else None

    for (cid, feat), blocks in lfp_block_results.items():
        if not blocks:
            continue

        # Ensure every block has valid specparam fit + AUC
        for blk in blocks:
            if blk.get('freqs_fit') is not None:
                continue  # cached fit present — trust stored AUC values
            # No fit — run specparam now
            psd_b = blk.get('psd', blk.get('mean_psd'))
            if psd_b is None or blk.get('freqs') is None:
                continue
            try:
                _sm = SpectralModel(**_sm_kwargs)
                _freqs_sm, _psd_sm = interpolate_spectrum(blk['freqs'], psd_b, [58, 62])
                _sm.fit(_freqs_sm, _psd_sm, freq_range=(1, 90))
                ff = np.asarray(_sm.freqs)
                fl = np.asarray(_sm.get_model(component='full',      space='log'))
                al = np.asarray(_sm.get_model(component='aperiodic', space='log'))
                tm = (ff >= 4)  & (ff <= 10)
                gm = (ff >= 30) & (ff <= 55)
                blk['freqs_fit'] = ff
                blk['full_log']  = fl
                blk['ape_log']   = al
                blk['exponent']  = float(_sm.get_params('aperiodic_params', 'exponent'))
                blk['offset']    = float(_sm.get_params('aperiodic_params', 'offset'))
                def _bauc(fl, al, ff, flo, fhi):
                    m = (ff >= flo) & (ff <= fhi)
                    d = np.where(np.isfinite(fl[m] - al[m]), fl[m] - al[m], 0.0)
                    return float(np.trapz(np.clip(d, 0, None), ff[m]))
                blk['theta_auc']       = _bauc(fl, al, ff,  4,  15)
                blk['slow_gamma_auc']  = _bauc(fl, al, ff, 30,  60)
                blk['high_gamma_auc']  = _bauc(fl, al, ff, 60,  80)
                blk['total_gamma_auc'] = _bauc(fl, al, ff, 30,  80)
                blk['r_squared']       = float(_sm.r_squared_)
                _pks_r = np.atleast_2d(_sm.peak_params_) if _sm.n_peaks_ > 0 else np.empty((0, 3))
                _pk_str_r = ', '.join(f'CF={p[0]:.1f}Hz PW={p[1]:.3f} BW={p[2]:.1f}' for p in _pks_r) or 'none'
                print(f'  [refit {cid}|{feat}|blk{blk["block"]}] R²={blk["r_squared"]:.3f}  '
                      f'exp={blk["exponent"]:.2f}  '
                      f'θ={blk["theta_auc"]:.4f}  sγ={blk["slow_gamma_auc"]:.4f}  '
                      f'hγ={blk["high_gamma_auc"]:.4f}  totγ={blk["total_gamma_auc"]:.4f}  '
                      f'n_peaks={_sm.n_peaks_}  peaks=[{_pk_str_r}]', flush=True)
            except Exception as _e:
                print(f"  specparam refit failed [{cid}|{feat}]: {_e}")

        # Repair / backfill all AUC keys using specparam peak model (NaN → 0 contribution)
        _AUC_BANDS = [
            ('theta_auc',       4,  15),
            ('slow_gamma_auc',  30, 60),
            ('high_gamma_auc',  60, 80),
            ('total_gamma_auc', 30, 80),
        ]
        for blk in blocks:
            _ff2 = blk.get('freqs_fit')
            _fl2 = blk.get('full_log')
            _al2 = blk.get('ape_log')
            if _ff2 is None or _fl2 is None or _al2 is None:
                continue
            _ff2 = np.asarray(_ff2); _fl2 = np.asarray(_fl2); _al2 = np.asarray(_al2)
            for _key, _flo, _fhi in _AUC_BANDS:
                if not np.isfinite(blk.get(_key, float('nan'))):
                    _m2 = (_ff2 >= _flo) & (_ff2 <= _fhi)
                    _d2 = np.where(np.isfinite(_fl2[_m2] - _al2[_m2]), _fl2[_m2] - _al2[_m2], 0.0)
                    blk[_key] = float(np.trapz(np.clip(_d2, 0, None), _ff2[_m2]))

        # Print specparam summary for every block (from cache or refit)
        print(f'\n=== {cid} | {feat} ===', flush=True)
        for blk in blocks:
            ff_p = blk.get('freqs_fit')
            fl_p = blk.get('full_log')
            al_p = blk.get('ape_log')
            exp_p  = blk.get('exponent', float('nan'))
            th_p   = blk.get('theta_auc', float('nan'))
            sg_p   = blk.get('slow_gamma_auc', float('nan'))
            hg_p   = blk.get('high_gamma_auc', float('nan'))
            tg_p   = blk.get('total_gamma_auc', float('nan'))
            # Compute R² from stored model vs raw PSD if not already stored
            r2_p = blk.get('r_squared', float('nan'))
            if not np.isfinite(r2_p) and ff_p is not None and fl_p is not None:
                _raw_psd = blk.get('psd', blk.get('mean_psd'))
                if _raw_psd is not None:
                    try:
                        _ff_arr = np.asarray(ff_p)
                        _fl_arr = np.asarray(fl_p)
                        _raw_log = np.log10(np.interp(_ff_arr, blk['freqs'], np.asarray(_raw_psd)))
                        _ss_res = np.sum((_raw_log - _fl_arr) ** 2)
                        _ss_tot = np.sum((_raw_log - np.mean(_raw_log)) ** 2)
                        r2_p = float(1.0 - _ss_res / _ss_tot) if _ss_tot > 0 else float('nan')
                        blk['r_squared'] = r2_p
                    except Exception:
                        pass
            if ff_p is not None and fl_p is not None and al_p is not None:
                ff_p = np.asarray(ff_p); fl_p = np.asarray(fl_p); al_p = np.asarray(al_p)
                # Reconstruct peaks: contiguous regions where full_log > ape_log
                _diff_p = fl_p - al_p
                _diff_p = np.where(np.isfinite(_diff_p), _diff_p, 0.0)
                _above  = _diff_p > 0.01
                _pk_info = []
                in_peak = False
                for i_p, val in enumerate(_above):
                    if val and not in_peak:
                        pk_start = i_p; in_peak = True
                    elif not val and in_peak:
                        seg = _diff_p[pk_start:i_p]
                        cf_idx = pk_start + int(np.argmax(seg))
                        _pk_info.append(f'CF={ff_p[cf_idx]:.1f}Hz PW={_diff_p[cf_idx]:.3f}')
                        in_peak = False
                if in_peak:
                    seg = _diff_p[pk_start:]
                    cf_idx = pk_start + int(np.argmax(seg))
                    _pk_info.append(f'CF={ff_p[cf_idx]:.1f}Hz PW={_diff_p[cf_idx]:.3f}')
                pk_str = ', '.join(_pk_info) or 'none'
            else:
                pk_str = 'no fit'
            print(f'  {blk["label"]:10s}  R²={r2_p:.3f}  exp={exp_p:.3f}  '
                  f'θ={th_p:.4f}  sγ={sg_p:.4f}  hγ={hg_p:.4f}  totγ={tg_p:.4f}  '
                  f'peaks=[{pk_str}]', flush=True)

        # Load cluster data for spike rolling mean
        pkl = f'{cluster_pickle_dir}/{cid}_cluster_df.pkl'
        try:
            df_cl = pd.read_pickle(pkl)
        except FileNotFoundError:
            continue
        col = f'{feat}_cluster'
        if col not in df_cl.columns:
            continue
        sub = df_cl[['spk_times_ms', col]].dropna().sort_values('spk_times_ms')
        times_s = sub['spk_times_ms'].values / 1000.0
        labels  = sub[col].values
        ord_l   = np.array([ORDINAL.get(str(l).strip(), 1) for l in labels], dtype=float)
        rm      = pd.Series(ord_l).rolling(rolling_n, center=True, min_periods=1).mean().values

        # Transition times
        grp = df_transitions[(df_transitions['cell_id'] == cid) &
                             (df_transitions['spike_feature'] == feat)]
        t0s_s = sorted(grp['transition_time_ms'].dropna().values / 1000.0)

        _rc = {
            'axes.linewidth':      3.5,
            'xtick.major.width':   3.0,  'ytick.major.width':   3.0,
            'xtick.major.size':    9,    'ytick.major.size':    9,
            'xtick.minor.visible': False, 'ytick.minor.visible': False,
            'font.size':           22,   'font.family': 'sans-serif',
            'font.weight':         'bold',
            'axes.titlesize':      26,   'axes.labelsize':      24,
            'xtick.labelsize':     20,   'ytick.labelsize':     20,
            'legend.fontsize':     18,
            'lines.linewidth':     3.5,
        }
        with plt.rc_context(_rc):
            # ── Flat 4-row × 3-col layout (tight_layout works correctly with flat GridSpec) ──
            # Col 0: spike (rows 0-1) + PSD (rows 2-3); Cols 1-2: bar panels
            fig = plt.figure(figsize=(26, 15))
            gs  = gridspec.GridSpec(4, 3, figure=fig,
                                    width_ratios=[1.3, 1.0, 1.0],
                                    wspace=0.5, hspace=0.7)
            ax_spike = fig.add_subplot(gs[0:2, 0])
            ax_psd   = fig.add_subplot(gs[2:4, 0])

            # ── Top-left: spike rolling mean ──────────────────────────────
            for lbl in np.unique(labels):
                m = labels == lbl
                ax_spike.scatter(times_s[m], ord_l[m],
                                 color=CLR_CLUSTER.get(str(lbl), 'gray'),
                                 s=12, alpha=0.3, linewidths=0, zorder=2)
            ax_spike.plot(times_s, rm, color='black', lw=5.0, zorder=4)
            for t0 in t0s_s:
                ax_spike.axvline(t0, color='crimson', lw=4.0, ls='--', zorder=5)
            boundaries_s = [-np.inf] + t0s_s + [np.inf]
            x_lo = float(times_s[0]); x_hi = float(times_s[-1])
            for b_i, blk in enumerate(blocks):
                lo = max(boundaries_s[b_i],  x_lo)
                hi = min(boundaries_s[b_i+1], x_hi)
                ax_spike.axvspan(lo, hi, alpha=0.12, color=blk['color'], zorder=1)
            ax_spike.set_yticks([0, 1, 2])
            ax_spike.set_yticklabels(['low', 'mid', 'high'], fontsize=22)
            ax_spike.set_xlabel('Time (s)', fontsize=24)
            ax_spike.set_ylabel('Cluster', fontsize=24)
            ax_spike.set_title(f'{cid}  ·  {feat}', fontsize=26, fontweight='bold')
            sns.despine(ax=ax_spike, offset=10)

            # ── Bottom-left: PSDs + specparam ────────────────────────────
            _added_band_labels = {'theta': False, 'slow_gamma': False, 'high_gamma': False}
            for blk in blocks:
                freqs, psd = blk['freqs'], blk.get('psd', blk.get('mean_psd'))
                dur_s = blk.get('n_samples', blk.get('n_spikes', 0))
                dur_label = f"{dur_s/2500:.0f}s" if 'n_samples' in blk else f"n={dur_s}"
                ax_psd.semilogy(freqs, psd, color=blk['color'], lw=3.0, alpha=0.45,
                                label=f"{blk['label']} ({dur_label})")
                if blk.get('freqs_fit') is not None:
                    ff = np.asarray(blk['freqs_fit'])
                    fl = np.asarray(blk['full_log']) if blk.get('full_log') is not None else None
                    al = np.asarray(blk['ape_log'])
                    if fl is not None:
                        _v = np.isfinite(fl)
                        if _v.any():
                            ax_psd.semilogy(ff[_v], 10**fl[_v], color=blk['color'],
                                            lw=5.5, zorder=5)
                    ax_psd.semilogy(ff, 10**al, color=blk['color'],
                                    lw=3.0, ls='--', alpha=0.85)
                    if fl is not None:
                        _shade_bands = [
                            ('theta',      4,  15, 'mediumpurple', 'θ (4–15 Hz)'),
                            ('slow_gamma', 30, 60, 'goldenrod',    'slow γ (30–60 Hz)'),
                            ('high_gamma', 60, 80, 'tomato',       'high γ (60–80 Hz)'),
                        ]
                        for _bkey, _blo, _bhi, _bcol, _blbl_str in _shade_bands:
                            _bm = (ff >= _blo) & (ff <= _bhi)
                            if _bm.any():
                                _fl_s = np.where(np.isfinite(fl[_bm]), fl[_bm], al[_bm])
                                _fl_c = np.clip(_fl_s, al[_bm], None)
                                _lbl  = _blbl_str if not _added_band_labels.get(_bkey) else None
                                ax_psd.fill_between(ff[_bm], 10**al[_bm], 10**_fl_c,
                                                    alpha=0.40, color=_bcol, zorder=4, label=_lbl)
                                _added_band_labels[_bkey] = True
            ax_psd.set_xlabel('Frequency (Hz)', fontsize=24)
            ax_psd.set_ylabel('PSD (µV²/Hz)',   fontsize=24)
            ax_psd.legend(fontsize=18, frameon=False)
            ax_psd.set_title('PSD + specparam', fontsize=26)
            if _psd_ylim is not None:
                ax_psd.set_ylim(_psd_ylim)
            ax_psd.set_xlim(0, 90)
            ax_psd.set_xticks([0, 20, 40, 60, 80])
            ax_psd.yaxis.set_major_locator(plt.LogLocator(numticks=3))
            ax_psd.yaxis.set_minor_locator(plt.NullLocator())
            sns.despine(ax=ax_psd)

            # ── Bar panels directly in the flat GridSpec ──────────────────
            _bar_axes = [
                fig.add_subplot(gs[0, 1]),  # mean_amp
                fig.add_subplot(gs[0, 2]),  # std_amp
                fig.add_subplot(gs[1, 1]),  # exponent
                fig.add_subplot(gs[1, 2]),  # offset
                fig.add_subplot(gs[2, 1]),  # theta_auc
                fig.add_subplot(gs[2, 2]),  # total_gamma
                fig.add_subplot(gs[3, 1]),  # slow_gamma
                fig.add_subplot(gs[3, 2]),  # high_gamma
            ]
            for p_i, (ax_b, (key, ylabel)) in enumerate(zip(_bar_axes, bar_specs)):
                vals_all = [blk.get(key, float('nan')) for blk in blocks]
                for b_i, (blk, val) in enumerate(zip(blocks, vals_all)):
                    ax_b.bar(b_i, val, color=blk['color'], zorder=3, width=0.4,
                             label=blk['label'] if p_i == 0 else None)
                    if np.isfinite(val):
                        _txt_y  = val if val >= 0 else val * 0.97
                        _txt_va = 'bottom' if val >= 0 else 'top'
                        ax_b.text(b_i, _txt_y, f'{val:.3g}', ha='center',
                                  va=_txt_va, fontsize=16, color='black', fontweight='bold')

                ylim = _global_ylims.get(key)
                if ylim is not None:
                    ax_b.set_ylim(ylim)
                elif key in _auc_keys:
                    ax_b.set_ylim(bottom=0)

                y0, y1 = ax_b.get_ylim()
                yspan   = y1 - y0
                for pair_i in range(len(blocks) - 1):
                    v1, v2 = vals_all[pair_i], vals_all[pair_i + 1]
                    if not (np.isfinite(v1) and np.isfinite(v2)):
                        continue
                    delta = v2 - v1
                    lbl = f'{"↑" if delta >= 0 else "↓"}Δ={abs(delta):.3g}'
                    bh  = y1 - yspan * (0.06 + pair_i * 0.14)
                    ax_b.text((pair_i + pair_i + 1) / 2, bh,
                              lbl, ha='center', va='top', fontsize=16,
                              color='#555555', clip_on=False)

                ax_b.set_ylabel(ylabel, fontsize=16, fontweight='bold')
                ax_b.axhline(0, color='gray', lw=1.5, ls='--')
                ax_b.set_xticks(range(len(blocks)))
                if p_i >= 6:  # slow/high γ row shows x-tick labels
                    ax_b.set_xticklabels([blk['label'] for blk in blocks],
                                         fontsize=13, rotation=20, ha='right')
                else:
                    ax_b.set_xticklabels([])
                sns.despine(ax=ax_b, offset=5)
            _bar_axes[1].legend(fontsize=13, frameon=False, loc='upper right')

            fig.suptitle(f'{cid}  ·  {feat}  —  LFP blocks at transition',
                         fontsize=26, fontweight='bold', y=1.01)
            plt.tight_layout(pad=2.5, w_pad=2.0, h_pad=2.0)
            plt.show()


def plot_pop_lfp_block_summary(lfp_block_results):
    """
    Population-level summary of LFP features across blocks at transition.

    For each LFP feature: bars show mean ± SEM across cells (block 0 vs block 1),
    individual cell values shown as connected dots, Wilcoxon signed-rank p-value annotated.
    """
    from scipy.stats import wilcoxon as _wilcoxon

    bar_specs = [
        ('mean_amp',        'Mean amp\n(µV)'),
        ('std_amp',         'Std amp\n(µV)'),
        ('exponent',        'Exponent'),
        ('offset',          'Offset'),
        ('theta_auc',       'θ AUC\n4–15 Hz'),
        ('total_gamma_auc', 'Total γ\n30–80 Hz'),
        ('slow_gamma_auc',  'Slow γ\n30–60 Hz'),
        ('high_gamma_auc',  'High γ\n60–80 Hz'),
    ]

    _sm_kwargs = dict(aperiodic_mode='fixed', peak_width_limits=(4.0, 8.0),
                      max_n_peaks=4, min_peak_height=0.0, peak_threshold=1.5, verbose=False)

    # Pre-fit specparam for any blocks missing it
    for _pblocks in lfp_block_results.values():
        for _pblk in _pblocks:
            if _pblk.get('freqs_fit') is not None:
                continue
            _ppsd = _pblk.get('psd', _pblk.get('mean_psd'))
            if _ppsd is None or _pblk.get('freqs') is None:
                continue
            try:
                _psm = SpectralModel(**_sm_kwargs)
                _pff2, _ppsd2 = interpolate_spectrum(_pblk['freqs'], _ppsd, [58, 62])
                _psm.fit(_pff2, _ppsd2, freq_range=(1, 90))
                _pff2 = np.asarray(_psm.freqs)
                _pfl  = np.asarray(_psm.get_model(component='full',      space='log'))
                _pal  = np.asarray(_psm.get_model(component='aperiodic', space='log'))
                def _pbauc(_fl, _al, _ff, _flo, _fhi):
                    _m = (_ff >= _flo) & (_ff <= _fhi)
                    _d = np.where(np.isfinite(_fl[_m] - _al[_m]), _fl[_m] - _al[_m], 0.0)
                    return float(np.trapz(np.clip(_d, 0, None), _ff[_m]))
                _pblk['freqs_fit']       = _pff2
                _pblk['full_log']        = _pfl
                _pblk['ape_log']         = _pal
                _pblk['exponent']        = float(_psm.get_params('aperiodic_params', 'exponent'))
                _pblk['offset']          = float(_psm.get_params('aperiodic_params', 'offset'))
                _pblk['theta_auc']       = _pbauc(_pfl, _pal, _pff2,  4,  15)
                _pblk['slow_gamma_auc']  = _pbauc(_pfl, _pal, _pff2, 30,  60)
                _pblk['high_gamma_auc']  = _pbauc(_pfl, _pal, _pff2, 60,  80)
                _pblk['total_gamma_auc'] = _pbauc(_pfl, _pal, _pff2, 30,  80)
            except Exception:
                pass

    # Collect per-cell (v0, v1) pairs for each feature
    _pairs = {key: [] for key, _ in bar_specs}
    for blocks in lfp_block_results.values():
        if len(blocks) < 2:
            continue
        b0, b1 = blocks[0], blocks[1]
        for key, _ in bar_specs:
            v0 = b0.get(key, float('nan'))
            v1 = b1.get(key, float('nan'))
            if np.isfinite(v0) and np.isfinite(v1):
                _pairs[key].append((v0, v1))

    # Block colors: reuse first two CLR_CLUSTER colors as generic block colors
    _blk_cols = ['#0072B2', '#D55E00']
    _blk_labels = ['Pre', 'Post']
    _SPINE_LW   = 2.5
    _LBL_FS     = 22
    _TICK_FS    = 18
    _STAR_FS    = 24
    _NS_FS      = 16

    nrows, ncols = 2, 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 11), squeeze=False)

    for p_i, (ax, (key, ylabel)) in enumerate(zip(axes.flat, bar_specs)):
        pairs = _pairs[key]
        if not pairs:
            ax.set_visible(False)
            continue

        v0s = np.array([p[0] for p in pairs])
        v1s = np.array([p[1] for p in pairs])
        n   = len(pairs)

        # Bars: mean ± SEM with pvc-6 style outlines
        for b_i, (vals, col) in enumerate(zip([v0s, v1s], _blk_cols)):
            mu  = np.mean(vals)
            sem = np.std(vals, ddof=1) / np.sqrt(n)
            ax.bar(b_i, mu, color=col, alpha=0.75, width=0.55, zorder=2,
                   edgecolor='#1a1a1a', linewidth=1.8)
            ax.errorbar(b_i, mu, yerr=sem, fmt='none', color='#111111',
                        elinewidth=3, capsize=8, capthick=3, zorder=3)

        # Individual cells as connected dots
        jitter = (np.random.default_rng(p_i).random(n) - 0.5) * 0.16
        for j, (vv0, vv1) in zip(jitter, pairs):
            ax.plot([0 + j, 1 + j], [vv0, vv1],
                    color='#888', lw=1.0, alpha=0.45, zorder=1)
            ax.scatter([0 + j, 1 + j], [vv0, vv1],
                       color=[_blk_cols[0], _blk_cols[1]], s=45, alpha=0.8,
                       zorder=2, edgecolors='none')

        # Wilcoxon signed-rank test
        try:
            _, p_val = _wilcoxon(v0s, v1s)
            star = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            print(f'  [pop Wilcoxon] {key}: n={n} p={p_val:.4f} {star}', flush=True)
        except Exception as _e:
            p_val, star = np.nan, 'n/a'
            print(f'  [pop Wilcoxon] {key}: n={n} FAILED {_e}', flush=True)

        # Expand ylim to leave room for bracket + star
        y0, y1 = ax.get_ylim()
        ax.set_ylim(y0, y1 + (y1 - y0) * 0.28)
        y0, y1 = ax.get_ylim()
        span = y1 - y0

        # Significance bracket
        bh = y1 - span * 0.09
        tk = span * 0.025
        ax.plot([0, 0, 1, 1], [bh - tk, bh, bh, bh - tk],
                color='#1a1a1a', lw=2.0)
        star_fs = _STAR_FS if star not in ('ns', 'n/a') else _NS_FS
        ax.text(0.5, bh + span * 0.008, star,
                ha='center', va='bottom', fontsize=star_fs,
                fontweight='bold', color='#1a1a1a', clip_on=False)

        # n count below bracket ticks
        ax.text(0.5, bh - tk - span * 0.005, f'n={n}',
                ha='center', va='top', fontsize=14, color='#555', clip_on=False)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([], fontsize=_TICK_FS)
        ax.set_ylabel(ylabel, fontsize=_LBL_FS, fontweight='bold')
        ax.axhline(0, color='gray', lw=1.5, ls='--')
        ax.tick_params(axis='both', which='major', labelsize=_TICK_FS, width=2.5)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(_SPINE_LW)
        sns.despine(ax=ax, offset=8)

    # Shared legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(color=_blk_cols[0], alpha=0.75, label='Pre-transition'),
                      Patch(color=_blk_cols[1], alpha=0.75, label='Post-transition')]
    fig.legend(handles=legend_handles, fontsize=16, frameon=False,
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('Population LFP: pre vs post transition  (mean ± SEM, Wilcoxon signed-rank)',
                 fontsize=18, fontweight='bold')
    plt.tight_layout(pad=2.5, w_pad=3.0, h_pad=4.0)
    plt.show()
    return fig, axes


# ─────────────────────────────────────────────────────────────────────────────
# Population transition delta plot
# ─────────────────────────────────────────────────────────────────────────────

_DELTA_LFP_KEYS = ['theta_auc', 'slow_gamma_auc', 'high_gamma_auc', 'total_gamma_auc',
                   'exponent', 'offset', 'mean_amp', 'std_amp']
_DELTA_LFP_LABELS = {
    'theta_auc':       'θ AUC',
    'slow_gamma_auc':  'Slow γ\n(30–60)',
    'high_gamma_auc':  'High γ\n(60–80)',
    'total_gamma_auc': 'Total γ\n(30–80)',
    'exponent':        'Exponent',
    'offset':          'Offset',
    'mean_amp':        'Amplitude',
    'std_amp':         'Std',
}
_DELTA_WF_LABELS = {
    'exp_lambda':      'Exp λ',
    'log_isi':         'Log ISI',
    'peak_amp':        'Peak Amp',
    'peak_sharpness':  'Sharpness',
    'peak_width':      'Peak Width',
    'inflection_time': 'Infl. Time',
}
# direction colours: blue = LFP increases, red = LFP decreases
_DIR_COL_POS = '#2166AC'   # blue  – LFP goes UP   when waveform low→high
_DIR_COL_NEG = '#D6604D'   # red   – LFP goes DOWN when waveform low→high


def compute_transition_deltas(lfp_block_results, df_transitions, lfp_keys=None):
    """
    Extract sign-corrected, normalised LFP Δ values for every
    (cell, waveform feature, LFP feature) combination and return a tidy DataFrame.

    Each row is one (cell_id × wf_feat × lfp_feat) observation.
    delta_norm is normalised by the cross-cell SD of raw Δ for that LFP feature,
    so values are in comparable units across features.

    Columns
    -------
    cell_id, wf_feat, lfp_feat, delta_raw, delta_norm, direction_sign
    """
    from collections import defaultdict

    lfp_keys = lfp_keys or _DELTA_LFP_KEYS
    _ORD     = {'low': 0, 'mid': 1, 'high': 2}

    primary = (df_transitions
               .sort_values('transition_index')
               .drop_duplicates(subset=['cell_id', 'spike_feature'], keep='first'))
    direction_map = {}
    for _, row in primary.iterrows():
        before = _ORD.get(str(row['cluster_before']).lower(), 1)
        after  = _ORD.get(str(row['cluster_after']).lower(),  1)
        direction_map[(row['cell_id'], row['spike_feature'])] = 1 if after >= before else -1

    raw_by_key = defaultdict(list)   # (cell_id, wf_feat) -> [{lfp_key: raw_delta, ...}]
    for (cid, wf_feat), blocks in lfp_block_results.items():
        if len(blocks) < 2:
            continue
        b0, b1   = blocks[0], blocks[1]
        sign     = direction_map.get((cid, wf_feat), 1)
        row_data = {'cell_id': cid, 'wf_feat': wf_feat, 'direction_sign': sign}
        for lk in lfp_keys:
            if lk in b0 and lk in b1:
                row_data[lk] = sign * (float(b1[lk]) - float(b0[lk]))
        raw_by_key[(cid, wf_feat)].append(row_data)

    all_rows = [r for rows in raw_by_key.values() for r in rows]
    norm_sd  = {}
    for lk in lfp_keys:
        vals = [r[lk] for r in all_rows if lk in r]
        norm_sd[lk] = float(np.std(vals)) if len(vals) > 1 else 1.0

    records = []
    for row_data in all_rows:
        for lk in lfp_keys:
            if lk in row_data:
                records.append(dict(
                    cell_id        = row_data['cell_id'],
                    wf_feat        = row_data['wf_feat'],
                    lfp_feat       = lk,
                    delta_raw      = row_data[lk],
                    delta_norm     = row_data[lk] / (norm_sd[lk] + 1e-12),
                    direction_sign = row_data['direction_sign'],
                ))

    return pd.DataFrame(records)


def plot_transition_deltas(lfp_block_results, df_transitions, lfp_keys=None):
    """
    For each (cell, waveform feature) pair compute Δ = post-transition minus
    pre-transition for each LFP feature, sign-corrected so that positive Δ
    always means the LFP feature increased when the waveform cluster went from
    a lower to a higher ordinal group (low→mid, low→high, mid→high).
    Cells where the waveform went high→low have their Δ flipped.

    Layout: one panel per waveform feature, x = LFP feature, y = Δ.
    Each dot is one cell.  Horizontal line at zero marks no change.

    Parameters
    ----------
    lfp_block_results : dict keyed by (cell_id, wf_feat), values = [block0, block1]
    df_transitions    : DataFrame from find_temporal_transitions (needs
                        cell_id, spike_feature, cluster_before, cluster_after)
    lfp_keys          : list[str] LFP feature keys to include (default: 5 core)
    """
    lfp_keys = lfp_keys or _DELTA_LFP_KEYS

    _ORD = {'low': 0, 'mid': 1, 'high': 2}

    # build a direction lookup: (cell_id, wf_feat) -> +1 or -1
    # use the primary transition (first row per cell/feature pair)
    direction_map = {}
    primary = (df_transitions
               .sort_values('transition_index')
               .drop_duplicates(subset=['cell_id', 'spike_feature'], keep='first'))
    for _, row in primary.iterrows():
        before = _ORD.get(str(row['cluster_before']).lower(), 1)
        after  = _ORD.get(str(row['cluster_after']).lower(),  1)
        sign   = 1 if after >= before else -1
        direction_map[(row['cell_id'], row['spike_feature'])] = sign

    # collect per-waveform-feature rows
    from collections import defaultdict
    rows_by_wf = defaultdict(list)

    # pass 1: collect raw deltas, sign-corrected for waveform direction
    for (cid, wf_feat), blocks in lfp_block_results.items():
        if len(blocks) < 2:
            continue
        b0, b1 = blocks[0], blocks[1]
        sign = direction_map.get((cid, wf_feat), 1)
        row = {'cell': cid, '_sign': sign}
        for lk in lfp_keys:
            if lk in b0 and lk in b1:
                row[lk] = sign * (float(b1[lk]) - float(b0[lk]))
        rows_by_wf[wf_feat].append(row)

    # pass 2: compute cross-cell std per LFP feature (pooled across all wf groups)
    all_rows = [r for rows in rows_by_wf.values() for r in rows]
    norm_sd = {}
    for lk in lfp_keys:
        vals = [r[lk] for r in all_rows if lk in r]
        norm_sd[lk] = float(np.std(vals)) if len(vals) > 1 else 1.0

    # pass 3: normalise in place
    for rows in rows_by_wf.values():
        for row in rows:
            for lk in lfp_keys:
                if lk in row:
                    row[lk] = row[lk] / (norm_sd[lk] + 1e-12)

    wf_feats = sorted(rows_by_wf.keys(),
                      key=lambda w: -len(rows_by_wf[w]))  # most cells first

    # global y range across all panels
    all_vals = [r[lk] for rows in rows_by_wf.values()
                for r in rows for lk in lfp_keys if lk in r]
    global_ymax = max(abs(v) for v in all_vals) * 1.15 if all_vals else 1.0
    global_ylim = (-global_ymax, global_ymax)

    n_wf  = len(wf_feats)
    ncols = min(3, n_wf)
    nrows = int(np.ceil(n_wf / ncols))
    n_lk  = len(lfp_keys)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=((0.8 + 0.9 * n_lk) * ncols, 4.0 * nrows),
                             squeeze=False)

    from matplotlib.lines import Line2D

    for idx, wf in enumerate(wf_feats):
        ax      = axes[idx // ncols][idx % ncols]
        rows    = rows_by_wf[wf]
        n_cells = len(rows)

        ax.set_ylim(*global_ylim)
        ax.axhline(0, color='#888', lw=1.0, ls='--', zorder=1)

        x_ticks, x_labels = [], []
        for xi, lk in enumerate(lfp_keys):
            deltas = [r[lk] for r in rows if lk in r]
            if not deltas:
                continue

            jitter = (np.random.default_rng(xi).random(len(deltas)) - 0.5) * 0.3

            for j, d in zip(jitter, deltas):
                col = _DIR_COL_POS if d >= 0 else _DIR_COL_NEG
                ax.scatter(xi + j, d, color=col, edgecolors='white',
                           linewidths=0.6, s=50, alpha=0.8, zorder=3)

            mu  = float(np.mean(deltas))
            sem = float(np.std(deltas) / np.sqrt(len(deltas)))
            muc = _DIR_COL_POS if mu >= 0 else _DIR_COL_NEG
            ax.plot([xi - 0.3, xi + 0.3], [mu, mu],
                    color='k', lw=2.5, solid_capstyle='round', zorder=4)
            ax.plot([xi, xi], [mu - sem, mu + sem],
                    color='k', lw=1.5, zorder=4)

            x_ticks.append(xi)
            x_labels.append(_DELTA_LFP_LABELS.get(lk, lk))

        ax.set_xlim(-0.6, n_lk - 0.4)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(f'{_DELTA_WF_LABELS.get(wf, wf)}  (n={n_cells})',
                     fontsize=11, fontweight='bold')
        if idx % ncols == 0:
            ax.set_ylabel('Normalised Δ  (post − pre)', fontsize=9)
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax)

    for idx in range(n_wf, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_DIR_COL_POS,
               markersize=9, label='LFP ↑  when spike-feature cluster: low → high'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_DIR_COL_NEG,
               markersize=9, label='LFP ↓  when spike-feature cluster: low → high'),
        Line2D([0], [0], color='k', lw=2.5, label='mean ± SEM'),
    ]
    fig.legend(handles=legend_handles, fontsize=9, frameon=False,
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        'LFP change at spike-feature cluster transition  (post − pre, normalised)\n'
        'Each dot = one cell  ·  all panels share the same y-axis',
        fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()
    return fig, axes


def _collect_delta_rows(lfp_block_results, df_transitions, lfp_keys):
    """Shared data-prep for the transition-delta family of plots.

    Returns (rows_by_wf, norm_sd, wf_feats, global_ylim).
    Each row dict has keys: 'cell', '_sign', and one float per lk (normalised).
    """
    _ORD = {'low': 0, 'mid': 1, 'high': 2}

    direction_map = {}
    primary = (df_transitions
               .sort_values('transition_index')
               .drop_duplicates(subset=['cell_id', 'spike_feature'], keep='first'))
    for _, row in primary.iterrows():
        before = _ORD.get(str(row['cluster_before']).lower(), 1)
        after  = _ORD.get(str(row['cluster_after']).lower(),  1)
        sign   = 1 if after >= before else -1
        direction_map[(row['cell_id'], row['spike_feature'])] = sign

    from collections import defaultdict
    rows_by_wf = defaultdict(list)
    for (cid, wf_feat), blocks in lfp_block_results.items():
        if len(blocks) < 2:
            continue
        b0, b1 = blocks[0], blocks[1]
        sign = direction_map.get((cid, wf_feat), 1)
        row = {'cell': cid, '_sign': sign}
        for lk in lfp_keys:
            if lk in b0 and lk in b1:
                row[lk] = sign * (float(b1[lk]) - float(b0[lk]))
        rows_by_wf[wf_feat].append(row)

    all_rows = [r for rows in rows_by_wf.values() for r in rows]
    norm_sd = {}
    for lk in lfp_keys:
        vals = [r[lk] for r in all_rows if lk in r]
        norm_sd[lk] = float(np.std(vals)) if len(vals) > 1 else 1.0

    for rows in rows_by_wf.values():
        for row in rows:
            for lk in lfp_keys:
                if lk in row:
                    row[lk] = row[lk] / (norm_sd[lk] + 1e-12)

    wf_feats = sorted(rows_by_wf.keys(), key=lambda w: -len(rows_by_wf[w]))

    all_vals = [r[lk] for rows in rows_by_wf.values()
                for r in rows for lk in lfp_keys if lk in r]
    global_ymax = max(abs(v) for v in all_vals) * 1.15 if all_vals else 1.0

    return rows_by_wf, norm_sd, wf_feats, (-global_ymax, global_ymax)


def plot_transition_deltas_signed_mean(lfp_block_results, df_transitions, lfp_keys=None):
    """Same as plot_transition_deltas but the mean ± SEM crosshair is coloured
    by the sign of the population mean: blue if mean > 0, red if mean < 0.
    One-sample Wilcoxon signed-rank test (vs 0) annotated below each LFP feature.
    """
    from scipy.stats import wilcoxon as _wilcoxon

    lfp_keys = lfp_keys or _DELTA_LFP_KEYS
    rows_by_wf, _, wf_feats, global_ylim = _collect_delta_rows(
        lfp_block_results, df_transitions, lfp_keys)

    _LBL_FS   = 34
    _TICK_FS  = 28
    _TTL_FS   = 32
    _STAR_FS  = 44
    _SPINE_LW = 2.5

    n_wf  = len(wf_feats)
    ncols = 2
    nrows = int(np.ceil(n_wf / ncols))
    n_lk  = len(lfp_keys)

    panel_w = 2.5 + 2.0 * n_lk
    panel_h = 9.0
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols, panel_h * nrows),
                             squeeze=False)
    from matplotlib.lines import Line2D

    for idx, wf in enumerate(wf_feats):
        ax      = axes[idx // ncols][idx % ncols]
        rows    = rows_by_wf[wf]
        n_cells = len(rows)

        ax.set_ylim(*global_ylim)
        ax.axhline(0, color='#aaa', lw=2.0, ls='--', zorder=1)

        x_ticks, x_labels = [], []
        for xi, lk in enumerate(lfp_keys):
            deltas = [r[lk] for r in rows if lk in r]
            if not deltas:
                continue

            jitter = (np.random.default_rng(xi).random(len(deltas)) - 0.5) * 0.3
            for j, d in zip(jitter, deltas):
                col = _DIR_COL_POS if d >= 0 else _DIR_COL_NEG
                ax.scatter(xi + j, d, color=col, edgecolors='white',
                           linewidths=0.8, s=100, alpha=0.85, zorder=3)

            mu  = float(np.mean(deltas))
            sem = float(np.std(deltas) / np.sqrt(len(deltas)))
            muc = _DIR_COL_POS if mu >= 0 else _DIR_COL_NEG
            ax.plot([xi - 0.35, xi + 0.35], [mu, mu],
                    color=muc, lw=5.0, solid_capstyle='round', zorder=4)
            ax.plot([xi, xi], [mu - sem, mu + sem],
                    color=muc, lw=3.0, zorder=4)

            # One-sample Wilcoxon vs 0 — star placed just below x-axis using axes transform
            star = ''
            if len(deltas) >= 5:
                try:
                    _, p = _wilcoxon(deltas)
                    star = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                    print(f'  [{wf}|{lk}] n={len(deltas)} p={p:.4f} {star or "ns"}', flush=True)
                except Exception as _e:
                    print(f'  [{wf}|{lk}] FAILED: {_e}', flush=True)
            if star:
                # get_xaxis_transform: x in data coords, y in axes fraction (0=bottom, -0.x=below)
                ax.text(xi, -0.10, star, ha='center', va='top',
                        fontsize=_STAR_FS, fontweight='bold', color='#1a1a1a',
                        transform=ax.get_xaxis_transform(), clip_on=False)

            x_ticks.append(xi)
            x_labels.append(_DELTA_LFP_LABELS.get(lk, lk))

        ax.set_xlim(-0.6, n_lk - 0.4)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=_TICK_FS, fontweight='bold')
        ax.set_title(f'{_DELTA_WF_LABELS.get(wf, wf)}  (n={n_cells})',
                     fontsize=_TTL_FS, fontweight='bold', pad=14)
        if idx % ncols == 0:
            ax.set_ylabel('Normalised Δ  (post − pre)', fontsize=_LBL_FS, fontweight='bold')
        ax.tick_params(axis='y', labelsize=_TICK_FS, width=2.5, length=6)
        ax.tick_params(axis='x', length=0)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(_SPINE_LW)
        sns.despine(ax=ax, offset=10)

    for idx in range(n_wf, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_DIR_COL_POS,
               markersize=18, label='LFP ↑  (cluster: low → high)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_DIR_COL_NEG,
               markersize=18, label='LFP ↓  (cluster: low → high)'),
        Line2D([0], [0], color=_DIR_COL_POS, lw=5.0, label='mean ± SEM  (positive)'),
        Line2D([0], [0], color=_DIR_COL_NEG, lw=5.0, label='mean ± SEM  (negative)'),
    ]
    fig.legend(handles=legend_handles, fontsize=20, frameon=False,
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.03))

    fig.suptitle(
        'LFP change at spike-feature cluster transition  (post − pre, normalised)\n'
        'Stars = one-sample Wilcoxon vs 0  ·  all panels share y-axis',
        fontsize=26, fontweight='bold')
    fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.12,
                        wspace=0.35, hspace=0.55)
    plt.show()
    return fig, axes


def plot_transition_deltas_by_celltype(lfp_block_results, df_transitions,
                                        cell_type_dict, lfp_keys=None):
    """Same panel layout as plot_transition_deltas but dots coloured by putative
    cell type (PC vs IN) rather than by LFP direction.

    Parameters
    ----------
    cell_type_dict : dict
        Maps cell number (int) → cell type string, e.g. {1: 'PC', 2: 'IN'}.
        Sourced from config.DICT_CELL_TYPE.
    """
    lfp_keys = lfp_keys or _DELTA_LFP_KEYS
    rows_by_wf, _, wf_feats, global_ylim = _collect_delta_rows(
        lfp_block_results, df_transitions, lfp_keys)

    _CT_COL = {'PC': _DIR_COL_POS, 'IN': _DIR_COL_NEG}

    def _cell_type(cid):
        try:
            return cell_type_dict.get(int(cid.lstrip('c')), 'unknown')
        except (ValueError, AttributeError):
            return 'unknown'

    n_wf  = len(wf_feats)
    ncols = min(3, n_wf)
    nrows = int(np.ceil(n_wf / ncols))
    n_lk  = len(lfp_keys)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=((0.8 + 0.9 * n_lk) * ncols, 4.0 * nrows),
                             squeeze=False)
    from matplotlib.lines import Line2D

    for idx, wf in enumerate(wf_feats):
        ax      = axes[idx // ncols][idx % ncols]
        rows    = rows_by_wf[wf]
        n_cells = len(rows)

        ax.set_ylim(*global_ylim)
        ax.axhline(0, color='#888', lw=1.0, ls='--', zorder=1)

        x_ticks, x_labels = [], []
        for xi, lk in enumerate(lfp_keys):
            deltas_by_ct = {'PC': [], 'IN': [], 'unknown': []}
            for r in rows:
                if lk not in r:
                    continue
                ct = _cell_type(r['cell'])
                deltas_by_ct.get(ct, deltas_by_ct['unknown']).append(r[lk])

            all_deltas = [d for vals in deltas_by_ct.values() for d in vals]
            if not all_deltas:
                continue

            rng = np.random.default_rng(xi)
            for ct, deltas in deltas_by_ct.items():
                if not deltas:
                    continue
                col    = _CT_COL.get(ct, 'gray')
                jitter = (rng.random(len(deltas)) - 0.5) * 0.3
                for j, d in zip(jitter, deltas):
                    ax.scatter(xi + j, d, color=col, edgecolors='white',
                               linewidths=0.6, s=50, alpha=0.8, zorder=3)

            mu  = float(np.mean(all_deltas))
            sem = float(np.std(all_deltas) / np.sqrt(len(all_deltas)))
            ax.plot([xi - 0.3, xi + 0.3], [mu, mu],
                    color='k', lw=2.5, solid_capstyle='round', zorder=4)
            ax.plot([xi, xi], [mu - sem, mu + sem],
                    color='k', lw=1.5, zorder=4)

            x_ticks.append(xi)
            x_labels.append(_DELTA_LFP_LABELS.get(lk, lk))

        ax.set_xlim(-0.6, n_lk - 0.4)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_title(f'{_DELTA_WF_LABELS.get(wf, wf)}  (n={n_cells})',
                     fontsize=11, fontweight='bold')
        if idx % ncols == 0:
            ax.set_ylabel('Normalised Δ  (post − pre)', fontsize=9)
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax)

    for idx in range(n_wf, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_CT_COL['PC'],
               markersize=9, label='PC (putative pyramidal)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_CT_COL['IN'],
               markersize=9, label='IN (putative interneuron)'),
        Line2D([0], [0], color='k', lw=2.5, label='mean ± SEM  (all cells)'),
    ]
    fig.legend(handles=legend_handles, fontsize=9, frameon=False,
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        'LFP change at spike-feature cluster transition  (post − pre, normalised)\n'
        'Dot colour = putative cell type  ·  all panels share the same y-axis',
        fontsize=11, y=1.01)
    plt.tight_layout()
    plt.show()
    return fig, axes


# ── EAP waveform extraction ───────────────────────────────────────────────────

def extract_eap_waveforms(npx_signal, spike_times_ms, npx_fs, pre_samp, post_samp):
    """Extract EAP windows around each spike time.

    Parameters
    ----------
    npx_signal : 1-D array
        Filtered Neuropixels recording (samples).
    spike_times_ms : array-like
        Spike times in milliseconds.
    npx_fs : float
        Neuropixels sampling rate (Hz).
    pre_samp, post_samp : int
        Samples to include before and after each spike peak.

    Returns
    -------
    np.ndarray, shape (n_spikes, pre_samp + post_samp)
        One row per extracted waveform; spikes too close to the recording
        boundary are silently dropped.
    """
    n = len(npx_signal)
    waveforms = []
    spike_samps = np.round(np.asarray(spike_times_ms) * npx_fs / 1000).astype(int)
    for s in spike_samps:
        lo, hi = s - pre_samp, s + post_samp
        if lo >= 0 and hi <= n:
            waveforms.append(npx_signal[lo:hi])
    return np.array(waveforms) if waveforms else np.empty((0, pre_samp + post_samp))


# ── Transition–metadata correlation plots ────────────────────────────────────

_TRANS_CONT_VARS = {
    'cort_depth':           'Cortical depth (µm)',
    'firing_rate_hz':       'Firing rate (Hz)',
    'rec_duration_min':     'Recording duration (min)',
    'trans_abruptness':     'Transition speed |k|',
    'trans_time_frac':      'Transition time (frac. of rec.)',
    'sigmoid_r2':           'Transition sigmoid R²',
    'frac_dominant_before': 'Dominant cluster fraction (pre)',
    'frac_dominant_after':  'Dominant cluster fraction (post)',
}

_TRANS_CAT_VARS = {
    'cell_type':         'Cell type',
    'patch_type_simple': 'Patch type',
    'dark_neuron':       'Dark neuron',
    'clear_eap':         'Clear EAP',
}

_TYPE_COLORS = {'PC': '#2166AC', 'IN': '#D6604D'}


def plot_lfp_delta_heatmap(df_wide, lfp_cols, cont_vars=None, lfp_labels=None):
    """Spearman ρ heatmap between LFP Δ values and continuous metadata variables."""
    if cont_vars is None:   cont_vars  = _TRANS_CONT_VARS
    if lfp_labels is None:  lfp_labels = _DELTA_LFP_LABELS

    cont_cols = [k for k in cont_vars if k in df_wide.columns]
    rho_mat = np.full((len(lfp_cols), len(cont_cols)), np.nan)
    p_mat   = np.full((len(lfp_cols), len(cont_cols)), np.nan)
    for i, lk in enumerate(lfp_cols):
        for j, mk in enumerate(cont_cols):
            sub = df_wide[[lk, mk]].dropna()
            if len(sub) >= 5:
                r, p = spearmanr(sub[lk], sub[mk])
                rho_mat[i, j] = r
                p_mat[i, j]   = p

    row_labels = [lfp_labels.get(lk, lk) for lk in lfp_cols]
    col_labels = [cont_vars[mk] for mk in cont_cols]

    fig, ax = plt.subplots(figsize=(max(8, len(cont_cols) * 1.1), max(4, len(lfp_cols) * 0.9)))
    im = ax.imshow(rho_mat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')
    for i in range(len(lfp_cols)):
        for j in range(len(cont_cols)):
            if not np.isfinite(rho_mat[i, j]): continue
            tc = 'white' if abs(rho_mat[i, j]) > 0.4 else '#222'
            ax.text(j, i, f'{rho_mat[i, j]:+.2f}\n{_stars(p_mat[i, j])}',
                    ha='center', va='center', fontsize=7.5, color=tc)
    ax.set_xticks(range(len(cont_cols)))
    ax.set_xticklabels(col_labels, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(len(lfp_cols)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title('Spearman ρ: LFP Δ  ×  continuous metadata\n'
                 '* p<0.05  ** p<0.01  *** p<0.001', fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.6).set_label('Spearman ρ', fontsize=9)
    fig.tight_layout()
    plt.show()
    return fig, ax


def plot_lfp_delta_by_category(df_wide, lfp_cols, cat_vars=None, lfp_labels=None):
    """Strip plots of LFP Δ grouped by each categorical metadata variable.

    One figure per variable. All panels share the same y-axis. Mean ± SEM
    shown as a black crosshair. Boolean columns (dark_neuron, clear_eap) are
    cast to strings automatically so seaborn palette lookup works correctly.
    """
    if cat_vars is None:   cat_vars   = _TRANS_CAT_VARS
    if lfp_labels is None: lfp_labels = _DELTA_LFP_LABELS

    all_vals = [v for lk in lfp_cols for v in df_wide[lk].dropna()]
    ymax = max(abs(v) for v in all_vals) * 1.15

    for cat_col, cat_label in cat_vars.items():
        if cat_col not in df_wide.columns:
            continue
        categories = [str(c) for c in sorted(df_wide[cat_col].dropna().unique(), key=str)]
        n_lk  = len(lfp_cols)
        ncols = min(4, n_lk)
        nrows = int(np.ceil(n_lk / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(3.5 * ncols, 3.2 * nrows), squeeze=False)
        palette = sns.color_palette('Set2', len(categories))
        col_map = dict(zip(categories, palette))

        for idx, lk in enumerate(lfp_cols):
            ax  = axes[idx // ncols][idx % ncols]
            sub = df_wide[[cat_col, lk]].dropna().copy()
            sub[cat_col] = sub[cat_col].astype(str)

            sns.stripplot(data=sub, x=cat_col, y=lk, order=categories,
                          palette=col_map, size=6, alpha=0.75, jitter=True,
                          edgecolor='white', linewidth=0.4, ax=ax)

            for xi, cat in enumerate(categories):
                vals = sub.loc[sub[cat_col] == cat, lk].dropna().values
                if len(vals) == 0: continue
                mu  = np.mean(vals)
                sem = np.std(vals) / np.sqrt(len(vals))
                ax.plot([xi - 0.25, xi + 0.25], [mu, mu],
                        color='k', lw=2.5, solid_capstyle='round', zorder=5)
                ax.plot([xi, xi], [mu - sem, mu + sem],
                        color='k', lw=1.5, zorder=5)

            ax.axhline(0, color='#888', lw=0.8, ls='--')
            ax.set_ylim(-ymax, ymax)
            ax.set_xlabel('')
            ax.set_xticklabels(categories, fontsize=9)
            ax.set_ylabel(lfp_labels.get(lk, lk) + ' Δ' if idx % ncols == 0 else '', fontsize=9)
            ax.set_title(lfp_labels.get(lk, lk), fontsize=9, fontweight='bold')
            ax.tick_params(labelsize=8)
            sns.despine(ax=ax)

        for idx in range(n_lk, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.suptitle(f'LFP Δ by  {cat_label}  (mean ± SEM, all panels same y-axis)',
                     fontsize=11, y=1.01)
        plt.tight_layout()
        plt.show()


def plot_lfp_delta_by_continuous(df_wide, lfp_cols, cont_vars=None, lfp_labels=None,
                                  type_colors=None):
    """Scatter plots of LFP Δ vs each continuous metadata variable, coloured by cell type."""
    if cont_vars is None:   cont_vars   = _TRANS_CONT_VARS
    if lfp_labels is None:  lfp_labels  = _DELTA_LFP_LABELS
    if type_colors is None: type_colors = _TYPE_COLORS

    for mk, mlabel in cont_vars.items():
        if mk not in df_wide.columns:
            continue
        n_lk  = len(lfp_cols)
        ncols = min(4, n_lk)
        nrows = int(np.ceil(n_lk / ncols))
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(3.8 * ncols, 3.2 * nrows), squeeze=False)

        for idx, lk in enumerate(lfp_cols):
            ax  = axes[idx // ncols][idx % ncols]
            sub = df_wide[[mk, lk, 'cell_type']].dropna()

            for ct, grp in sub.groupby('cell_type'):
                ax.scatter(grp[mk], grp[lk],
                           color=type_colors.get(ct, 'gray'),
                           edgecolors='white', linewidths=0.4,
                           s=50, alpha=0.8, label=ct, zorder=3)

            xy = sub[[mk, lk]].dropna()
            if len(xy) >= 5:
                rho, p = spearmanr(xy[mk], xy[lk])
                m, b   = np.polyfit(xy[mk], xy[lk], 1)
                xs     = np.linspace(xy[mk].min(), xy[mk].max(), 100)
                ax.plot(xs, m * xs + b, color='#444', lw=1.4, ls='--', zorder=2)
                ax.text(0.97, 0.97, f'ρ={rho:+.2f}{_stars(p)}',
                        transform=ax.transAxes, ha='right', va='top', fontsize=8)

            ax.axhline(0, color='#aaa', lw=0.8, ls=':')
            ax.set_xlabel(mlabel, fontsize=8)
            ax.set_ylabel(lfp_labels.get(lk, lk) + ' Δ' if idx % ncols == 0 else '', fontsize=8)
            ax.set_title(lfp_labels.get(lk, lk), fontsize=9, fontweight='bold')
            ax.tick_params(labelsize=7)
            sns.despine(ax=ax)

        for idx in range(n_lk, nrows * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        handles = [plt.scatter([], [], color=c, label=ct, s=40) for ct, c in type_colors.items()]
        fig.legend(handles=handles, fontsize=9, frameon=False,
                   loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.04))
        fig.suptitle(f'LFP Δ  vs  {mlabel}  (coloured by cell type)', fontsize=10, y=1.01)
        plt.tight_layout()
        plt.show()
