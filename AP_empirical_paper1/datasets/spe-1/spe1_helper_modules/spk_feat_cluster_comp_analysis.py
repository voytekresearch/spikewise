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

# ------------------------------------------------------------------------------------------- #
#                                     Environment Setup                                       #
# ------------------------------------------------------------------------------------------- #

# Import metadata file
config_dir = "/Users/blancamartin/Desktop/Voytek_Lab/spike_waveform/spikeparam/AP_empirical_paper1/datasets/spe-1/spe1_helper_modules/"
if config_dir not in sys.path:
    sys.path.append(config_dir)
import config

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

    # Binary flag: 1 if temporal drift is statistically significant (p < 0.05)
    final_table['temporal_component'] = (
        pd.to_numeric(final_table['temporal_p'], errors='coerce') < 0.05
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
                f"{row.Effect_Size:+.3f}  {stars}", va='center', fontsize=9)

    ax.axvline(0, color='black', lw=1, ls='--', alpha=0.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(df_plot['label'].tolist(), fontsize=9)
    ax.set_xlim(-x_max * 1.05, x_max * 1.4)
    ax.set_xlabel('Effect Size  (95% bootstrap CI)', fontsize=11)
    ax.set_title(
        'Metadata × Cluster Difference\n(BH-FDR corrected; filled marker = significant)',
        fontsize=12, fontweight='bold'
    )
    from matplotlib.lines import Line2D
    legend_els = [Line2D([0], [0], color=c, lw=3, label=t) for t, c in TEST_COLORS.items()]
    ax.legend(handles=legend_els, fontsize=9, frameon=False, loc='lower right')
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
              f"p_fdr={r['p_fdr']:.3e} {r['Significance']}")

    sig['p-value'] = sig['p_fdr'].apply(lambda p: f"{p:.2e}")
    return sig.reset_index(drop=True)

def plot_sig_feat_pairs(df, sig_pairs_df):
    # Nuke the warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    if sig_pairs_df is None or len(sig_pairs_df) == 0:
        print("No significant pairs to plot.")
        return

    sns.set_theme(style="ticks")

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

        ax.set_title(f"{stars} (r={r:.2f})", fontweight='bold', fontsize=13)
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
    plt.xlabel('Cortical Depth ($\mu m$)', fontsize=14, fontweight='semibold')
    plt.ylabel('Clustered Spike Features', fontsize=14, fontweight='semibold')
    
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
    print(f"p-value: {p:.3e}")
    
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

    Shows whether spike cluster identity drifts over recording time
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
    ax1.set_xlabel('Temporal Rho (Spearman)', fontsize=13)
    ax1.set_ylabel('Count', fontsize=13)
    n_sig = df_plot['significant'].sum()
    n_total = len(df_plot)
    ax1.set_title(f'Temporal Drift Distribution\n{n_sig}/{n_total} significant (p < {alpha})',
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, frameon=False)
    ax1.set_xlim(-1.1, 1.1)
    sns.despine(ax=ax1)

    # --- Panel B: Distribution per spike feature ---
    feat_order = df_plot.groupby('spike_feature')['temporal_rho'].median().sort_values().index.tolist()
    feat_palette = [palette[f] for f in feat_order]

    sns.boxplot(data=df_plot, x='temporal_rho', y='spike_feature', order=feat_order,
                palette=feat_palette, showfliers=False, width=0.5, ax=ax2)
    sns.stripplot(data=df_plot, x='temporal_rho', y='spike_feature', order=feat_order,
                  palette=feat_palette, alpha=0.5, size=5, ax=ax2)
    ax2.axvline(0, color='black', lw=1, linestyle='--', alpha=0.5)
    ax2.set_xlabel('Temporal Rho', fontsize=13)
    ax2.set_ylabel('')
    ax2.set_title('Distribution by Spike Feature', fontsize=13, fontweight='bold')
    ax2.set_xlim(-1.1, 1.1)
    sns.despine(ax=ax2)

    plt.tight_layout()
    plt.show()

    # Print summary
    n_sig = df_plot['significant'].sum()
    n_total = len(df_plot)
    print(f"\nTemporal drift summary (p < {alpha}):")
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
    Tests whether cell-level metadata predicts temporal_rho (drift of cluster identity
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
            ax.set_xlabel(col.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel('Temporal Rho', fontsize=12)
            p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            ax.set_title(f'{col}\nSpearman ρ={rho:.2f}, {sig} (p={p_str})', fontsize=12, fontweight='bold')
            results.append({'variable': col, 'test': 'Spearman', 'statistic': round(rho, 3), 'p': p, 'sig': sig})
        else:
            groups_list = [g['temporal_rho'].values for _, g in sub.groupby(col)]
            if len(groups_list) >= 2:
                stat, p = kruskal(*groups_list)
            else:
                stat, p = np.nan, np.nan
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

            sns.boxplot(data=sub, x=col, y='temporal_rho', showfliers=False,
                        palette='Paired', ax=ax)
            sns.stripplot(data=sub, x=col, y='temporal_rho', color='.3', alpha=0.4, ax=ax)
            ax.axhline(0, color='black', lw=1, linestyle='--', alpha=0.4)
            ax.set_xlabel(col.replace('_', ' ').title(), fontsize=12)
            ax.set_ylabel('Temporal Rho', fontsize=12)
            p_str = f"{p:.4f}" if p >= 0.0001 else "<0.0001"
            ax.set_title(f'{col}\nKruskal-Wallis {sig} (p={p_str})', fontsize=12, fontweight='bold')
            results.append({'variable': col, 'test': 'Kruskal-Wallis', 'statistic': round(stat, 3) if pd.notnull(stat) else np.nan, 'p': p, 'sig': sig})

        sns.despine(ax=ax)

    for j in range(len(meta_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Metadata Predictors of Temporal Drift (temporal_rho)', fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()

    res_df = pd.DataFrame(results).sort_values('p').reset_index(drop=True)
    print("\nMetadata → temporal_rho test results:")
    print(res_df.to_string(index=False))
    return res_df


def analyze_temporal_clustering_relationship(df, alpha=0.05):
    """
    Tests whether temporal drift (temporal_rho / temporal_component) is related to
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
        ax.set_xlabel('Temporal Rho', fontsize=13)
        ax.set_ylabel(metric, fontsize=13)
        ax.set_title(f'temporal_rho vs {metric}\nSpearman ρ={rho:.2f}, {sig}', fontsize=13, fontweight='bold')
        sns.despine(ax=ax)
        results.append({'metric': metric, 'spearman_rho': round(rho, 3), 'p': p, 'sig': sig})

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=palette[f],
                          markersize=8, label=f) for f in features]
    axes[-1].legend(handles=handles, title='Spike Feature', bbox_to_anchor=(1.05, 1),
                    loc='upper left', fontsize=11, frameon=True)

    plt.suptitle('Temporal Drift vs Clustering Quality', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # --- Panel 2: temporal_rho distribution by spike_feature ---
    feat_order = df_plot.groupby('spike_feature')['temporal_rho'].median().sort_values().index.tolist()
    feat_palette_list = [palette.get(f, 'gray') for f in feat_order]

    fig_feat, ax_feat = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df_plot, x='spike_feature', y='temporal_rho', order=feat_order,
                palette=feat_palette_list, showfliers=False, width=0.5, ax=ax_feat)
    sns.stripplot(data=df_plot, x='spike_feature', y='temporal_rho', order=feat_order,
                  palette=feat_palette_list, alpha=0.5, size=6, ax=ax_feat)
    ax_feat.axhline(0, color='black', lw=1, linestyle='--', alpha=0.4)

    feat_groups = [df_plot.loc[df_plot['spike_feature'] == f, 'temporal_rho'].dropna().values for f in feat_order]
    feat_groups = [g for g in feat_groups if len(g) > 0]
    if len(feat_groups) >= 2:
        kw_stat, kw_p = kruskal(*feat_groups)
    else:
        kw_stat, kw_p = np.nan, np.nan
    kw_sig = '***' if kw_p < 0.001 else '**' if kw_p < 0.01 else '*' if kw_p < 0.05 else 'ns'
    kw_p_str = f"{kw_p:.4f}" if pd.notnull(kw_p) and kw_p >= 0.0001 else ("<0.0001" if pd.notnull(kw_p) else "n/a")

    ax_feat.set_title(f'Temporal Rho by Spike Feature\nKruskal-Wallis {kw_sig} (p={kw_p_str})',
                      fontsize=13, fontweight='bold')
    ax_feat.set_xlabel('Spike Feature', fontsize=13)
    ax_feat.set_ylabel('Temporal Rho', fontsize=13)
    ax_feat.set_xticklabels(ax_feat.get_xticklabels(), rotation=30, ha='right')
    sns.despine(ax=ax_feat)
    plt.tight_layout()
    plt.show()

    # --- Panel 3: temporal_component (0 vs 1) split on nRMSE / cos_sim ---
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))
    for ax2, metric in zip(axes2, ['nRMSE', 'cos_sim']):
        sub2 = df_plot[['temporal_component', metric]].dropna()
        sub2['temporal_component'] = sub2['temporal_component'].astype(float).map({0.0: 'No drift', 1.0: 'Sig drift'})
        sns.boxplot(data=sub2, x='temporal_component', y=metric, showfliers=False,
                    palette=['#56B4E9', '#D55E00'], order=['No drift', 'Sig drift'], ax=ax2)
        sns.stripplot(data=sub2, x='temporal_component', y=metric, color='.3', alpha=0.4,
                      order=['No drift', 'Sig drift'], ax=ax2)

        groups_list = [g[metric].values for _, g in sub2.groupby('temporal_component') if len(g) > 0]
        if len(groups_list) >= 2:
            stat2, p2 = kruskal(*groups_list)
        else:
            p2 = np.nan
        sig2 = '***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else 'ns'
        p2_str = f"{p2:.4f}" if pd.notnull(p2) and p2 >= 0.0001 else ("<0.0001" if pd.notnull(p2) else "n/a")
        ax2.set_title(f'temporal_component vs {metric}\n{sig2} (p={p2_str})', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Temporal Component')
        sns.despine(ax=ax2)

    plt.suptitle('Does Significant Temporal Drift Affect Cluster Waveform Differences?', fontsize=14, fontweight='bold')
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
        print(f"{col:<20} | p = {p:.3e} ({sig})")
        
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
    Derive priority cells and high-difference / low-drift cells from df_master,
    report which have LFP analysis notebooks, and optionally create a summary plot.

    Parameters
    ----------
    df : pd.DataFrame
        df_master from compile_experiment_results.
    n_priority : int
        Number of top-nRMSE cells to designate as priority.
    rho_thresh : float
        Max |mean temporal rho| for the low-drift group.
    n_nodrift : int
        Number of top-nRMSE low-drift cells to select.
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

    print(f"\n=== High-diff / low-drift (|rho| < {rho_thresh}, top {n_nodrift}) ===")
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
            '(orange = priority, green = high-diff/low-drift)',
            fontsize=12, fontweight='bold'
        )
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color='#D55E00', label=f'Priority cells (top {n_priority})'),
            Patch(color='#009E73', label=f'High-diff / low-drift (|ρ| < {rho_thresh}, top {n_nodrift})'),
            Patch(color='#BBBBBB', label='Other cells'),
        ], fontsize=9, frameon=False)
        sns.despine(ax=ax)
        plt.tight_layout()
        plt.show()

    return priority_nums, nodrift_nums, df_cell



# ------------------------------------------------------------------------------------------- #
# ----------------------- Population waveform cluster visualization ------------------------- #
# ------------------------------------------------------------------------------------------- #

def plot_population_waveform_clusters(
    df,
    spike_pickle_dir,
    cluster_pickle_dir,
    cells_to_plot=None,
    priority_cells=None,
    cols=6,
    figsize_per_cell=(2.8, 2.2),
):
    """
    Grid of peak-aligned average waveforms per cluster group, one panel per cell.
    Matches the overlay style from individual cluster report notebooks.

    Parameters
    ----------
    df : pd.DataFrame
        df_master (used for cell_id list).
    spike_pickle_dir : str
        Directory containing c{N}_spike_fit.pkl files.
    cluster_pickle_dir : str
        Directory containing c{N}_cluster_df.pkl files.
    cells_to_plot : list of int, optional
        Cell numbers to include. None = all cells in df_master.
    priority_cells : list of int, optional
        Cell numbers to highlight with an orange border.
        Defaults to config.PRIORITY_CELLS.
    cols : int
        Columns in the grid.
    figsize_per_cell : tuple
        (width, height) per panel in inches.
    """
    import pickle
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import config as _cfg

    CLUSTER_COLORS = {"low": "#1f77b4", "mid": "#2ca02c", "high": "#ff7f0e"}

    if priority_cells is None:
        priority_cells = set(_cfg.PRIORITY_CELLS)
    else:
        priority_cells = set(priority_cells)

    all_cell_ids = sorted(df["cell_id"].unique(), key=lambda c: int(c.lstrip("c")))
    if cells_to_plot is not None:
        target = {f"c{n}" for n in cells_to_plot}
        cell_ids = [c for c in all_cell_ids if c in target]
    else:
        cell_ids = all_cell_ids

    rows = math.ceil(len(cell_ids) / cols)
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(figsize_per_cell[0] * cols, figsize_per_cell[1] * rows),
    )
    axes = np.array(axes).flatten()

    for ax_idx, cell_id in enumerate(cell_ids):
        ax   = axes[ax_idx]
        cnum = int(cell_id.lstrip("c"))
        is_priority = cnum in priority_cells

        sp_path = Path(spike_pickle_dir)   / f"c{cnum}_spike_fit.pkl"
        cl_path = Path(cluster_pickle_dir) / f"c{cnum}_cluster_df.pkl"

        if not sp_path.exists() or not cl_path.exists():
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=7, color="gray")
            ax.set_xticks([]); ax.set_yticks([])
            _style_ax(ax, cell_id, is_priority)
            continue

        try:
            with open(sp_path,  "rb") as f: sp       = pickle.load(f)
            with open(cl_path,  "rb") as f: df_clust = pickle.load(f)

            cluster_cols = [c for c in df_clust.columns if c.endswith("_cluster")]
            # Pick the first feature with >1 group
            col = next(
                (c for c in cluster_cols if df_clust[c].dropna().nunique() > 1), None
            )
            if col is None:
                raise ValueError("no multi-group cluster column")

            labels = sorted(df_clust[col].dropna().unique())

            # Gather all waveforms across all groups for shared peak alignment
            all_wfs, grp_idx = [], {}
            offset = 0
            for lab in labels:
                inds = df_clust.loc[df_clust[col] == lab, "spk_id"].astype(int).tolist()
                wfs  = [sp.spikes[i] for i in inds if i < len(sp.spikes)]
                grp_idx[lab] = (offset, offset + len(wfs))
                all_wfs.extend(wfs)
                offset += len(wfs)

            if not all_wfs:
                raise ValueError("no waveforms")

            # Peak-align all waveforms to a shared grid (peak = index `pre`)
            peak_idxs = [int(np.argmax(np.abs(w))) for w in all_wfs]
            pre  = max(peak_idxs)
            post = max(len(w) - pk - 1 for w, pk in zip(all_wfs, peak_idxs))
            total = pre + post + 1

            aligned = np.full((len(all_wfs), total), np.nan)
            for k, (w, pk) in enumerate(zip(all_wfs, peak_idxs)):
                s = pre - pk
                aligned[k, s:s + len(w)] = w

            t_axis = np.arange(total) - pre  # samples; 0 = peak

            # Plot mean ± std per cluster group
            for lab in labels:
                s, e = grp_idx[lab]
                arr  = aligned[s:e]
                if arr.shape[0] == 0:
                    continue
                mean = np.nanmean(arr, axis=0)
                std  = np.nanstd(arr, axis=0)
                color = CLUSTER_COLORS.get(str(lab), "gray")
                ax.plot(t_axis, mean, color=color, lw=1.8, label=str(lab))
                ax.fill_between(t_axis, mean - std, mean + std,
                                color=color, alpha=0.15)

            ax.axvline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
            ax.set_xticks([]); ax.set_yticks([])

        except Exception as e:
            ax.text(0.5, 0.5, str(e)[:40], ha="center", va="center",
                    transform=ax.transAxes, fontsize=6, color="red")
            ax.set_xticks([]); ax.set_yticks([])

        _style_ax(ax, cell_id, is_priority)

    for ax in axes[len(cell_ids):]:
        ax.set_visible(False)

    # Shared legend
    from matplotlib.lines import Line2D
    legend_els = [Line2D([0],[0], color=c, lw=2, label=lab)
                  for lab, c in CLUSTER_COLORS.items()]
    fig.legend(handles=legend_els, loc="lower right", fontsize=9,
               frameon=False, ncol=3)

    n_shown = "all" if cells_to_plot is None else str(len(cell_ids))
    fig.suptitle(
        f"Peak-aligned spike waveforms by cluster group  ({n_shown} cells)\n"
        "orange border = priority cell | mean ± SD per group",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()


def _style_ax(ax, cell_id, is_priority):
    """Apply title and border style to a waveform panel."""
    ax.set_title(cell_id, fontsize=8,
                 fontweight="bold" if is_priority else "normal",
                 color="#D55E00" if is_priority else "black")
    lw    = 2.5 if is_priority else 0.5
    color = "#D55E00" if is_priority else "#cccccc"
    for spine in ax.spines.values():
        spine.set_linewidth(lw)
        spine.set_edgecolor(color)
