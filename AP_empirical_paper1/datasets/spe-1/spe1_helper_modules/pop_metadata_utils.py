"""
pop_metadata_utils.py
---------------------
Metadata analysis utilities for the spe-1 population ridge regression.
Called from pop_ridge_metadata.ipynb.

Typical usage
-------------
    from pop_metadata_utils import (
        build_metadata_df,
        plot_cell_sig_heatmap,
        plot_r2_by_group,
        plot_alpha_by_group,
        plot_metadata_vs_mean_r2,
        plot_n_spikes_confound,
        run_beta_metadata_analysis,
        plot_predictor_set_comparison,
    )

    df = build_metadata_df(
        all_results, cell_ids, target_names, target_labels, predictor_sets,
        DICT_CELL_TYPE, DICT_PATCH_TYPE, DICT_CORT_DEPTH,
        DICT_DARK_NEURONS, DICT_EAP_WAV,
    )
    plot_cell_sig_heatmap(df, cell_ids, target_names, target_labels)
    plot_r2_by_group(df, target_names)
    plot_alpha_by_group(df, target_names)
    plot_metadata_vs_mean_r2(df)
    plot_n_spikes_confound(df)
    run_beta_metadata_analysis(beta_pop, df_r2, df, target_names, target_labels)
    plot_predictor_set_comparison(df, predictor_sets)
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

from ridge_regression_utils import FEAT_LABELS, WAVEFORM_LABELS

# ── Colour palette (colour-blind-friendly) ────────────────────────────────────
_CB      = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00']
_SIG_COL = '#D55E00'   # significant (red-orange)
_NOM_COL = '#E69F00'   # nominally significant (p<0.05, not FDR; amber)

# Standard 5-group target structure (Pre / Pre-BL / Post / Post-BL / Δ)
_GRP_COLS  = ['#1976D2', '#43A047', '#9C27B0', '#FF9800', '#E53935']
_GRP_NAMES = ['Pre (abs)', 'Pre − BL', 'Post (abs)', 'Post − BL', 'Δ post−pre']


def _pset_col(pn):
    """Predictor set name → safe DataFrame column suffix."""
    return pn.replace(' + ', '_').replace(' ', '_').lower()


def _feat_short(feat_labels):
    """Return abbreviated labels for a list of spe-1 LFP feature names."""
    return [
        l.replace('LFP ', '').replace(' AUC', '').replace('onent', '')
        for l in feat_labels
    ]


# ── 1. Build metadata DataFrame ───────────────────────────────────────────────

def build_metadata_df(all_results, cell_ids, target_names, target_labels,
                       predictor_sets, dict_cell_type, dict_patch_type,
                       dict_cort_depth, dict_dark_neurons, dict_eap_wav):
    """
    Build a per-cell DataFrame combining cell metadata with ridge regression stats.

    Parameters
    ----------
    all_results    : dict  {cell_id → results_dict}  from load_population_results
    cell_ids       : list[str]
    target_names   : list[str]
    target_labels  : list[str]
    predictor_sets : list[str]
    dict_cell_type, dict_patch_type, dict_cort_depth,
    dict_dark_neurons, dict_eap_wav : dict {cell_num → value}
        Cell property lookup tables (from spe-1 config).

    Returns
    -------
    df : pd.DataFrame
        One row per cell.  Columns:
          cell_id, cell_num, cell_type, patch_type, cort_depth, dark_neuron,
          eap_visible, n_spikes, mean_r2_wv,
          n_sig_<pset> for each predictor set,
          r2_<target>, sig_<target>, alpha_<target> (Waveform only) per target.
    """
    rows = []
    for cid in cell_ids:
        cell_num = int(cid.replace('c', ''))
        res = all_results[cid]

        row = dict(
            cell_id     = cid,
            cell_num    = cell_num,
            cell_type   = dict_cell_type.get(cell_num, 'unknown'),
            patch_type  = dict_patch_type.get(cell_num, 'unknown'),
            cort_depth  = dict_cort_depth.get(cell_num, np.nan),
            dark_neuron = str(dict_dark_neurons.get(cell_num, np.nan)),
            eap_visible = str(dict_eap_wav.get(cell_num, np.nan)),
            n_spikes    = res[target_names[0]]['Waveform only'].get('n_valid', np.nan),
            mean_r2_wv  = float(np.nanmean([
                res[tn]['Waveform only'].get('r2_cv', np.nan)
                for tn in target_names
            ])),
        )
        # n FDR-significant targets per predictor set
        for pn in predictor_sets:
            row[f'n_sig_{_pset_col(pn)}'] = sum(
                res[tn].get(pn, {}).get('sig_fdr', False) for tn in target_names
            )
        # per-target R², significance flag, best alpha (Waveform only)
        for tn in target_names:
            pset = res[tn]['Waveform only']
            row[f'r2_{tn}']    = pset.get('r2_cv',      np.nan)
            row[f'sig_{tn}']   = pset.get('sig_fdr',    False)
            row[f'alpha_{tn}'] = pset.get('best_alpha', np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Replace string 'nan' (from dict lookup misses) with actual NaN
    for col in ['dark_neuron', 'eap_visible']:
        df[col] = df[col].replace('nan', np.nan)
    return df


# ── 2. Cell × target significance heatmap ────────────────────────────────────

def plot_cell_sig_heatmap(df, cell_ids, target_names, target_labels):
    """
    Heatmap: rows = cells sorted by total n significant, columns = LFP targets.
    Shows which cells have FDR-significant Waveform-only models per target.
    Fraction significant per column annotated above x-axis.

    Parameters
    ----------
    df           : output of build_metadata_df
    cell_ids     : list[str]  original ordering (must match df rows)
    target_names : list[str]
    target_labels: list[str]
    """
    import seaborn as sns

    sig_matrix = np.array(
        [[bool(df.loc[df.cell_id == cid, f'sig_{tn}'].values[0])
          for tn in target_names]
         for cid in cell_ids],
        dtype=float
    )
    n_sig_per_cell  = sig_matrix.sum(axis=1)
    sort_idx        = np.argsort(n_sig_per_cell)[::-1]
    cell_ids_sorted = [cell_ids[i] for i in sort_idx]
    n_cells         = len(cell_ids)

    fig, ax = plt.subplots(figsize=(18, max(8, 0.45 * n_cells + 3)))
    im = ax.imshow(sig_matrix[sort_idx], aspect='auto', cmap='Blues', vmin=0, vmax=1,
                   extent=[-0.5, len(target_names) - 0.5, n_cells - 0.5, -0.5])

    # Group dividers + labels for 5-group, 5-feature structure
    for d_pos in [4.5, 9.5, 14.5, 19.5]:
        ax.axvline(d_pos, color='white', lw=2.5)
    for g_idx, (lbl, x) in enumerate(
        [('Pre\n(abs)', 2), ('Pre−BL', 7), ('Post\n(abs)', 12), ('Post−BL', 17), ('Δ\npost−pre', 22)]
    ):
        ax.text(x, n_cells + 0.8, lbl, ha='center', va='top', fontsize=9,
                fontweight='bold', color=_GRP_COLS[g_idx])

    ax.set_xticks(range(len(target_names)))
    ax.set_xticklabels([tl.split(' ', 1)[-1] for tl in target_labels],
                       rotation=40, ha='right', fontsize=8)
    ax.set_yticks(range(n_cells))
    ax.set_yticklabels(
        [f'{cell_ids_sorted[i]}  ({int(n_sig_per_cell[sort_idx[i]])})'
         for i in range(n_cells)],
        fontsize=8.5
    )
    ax.set_xlabel('LFP target', fontsize=11)
    ax.set_ylabel('Cell  (n significant targets)', fontsize=11)
    ax.set_title('Which cells have FDR-significant Waveform-only models?\n'
                 '(sorted by total n significant)', fontsize=12, pad=16)

    cbar = plt.colorbar(im, ax=ax, shrink=0.3, pad=0.01)
    cbar.set_ticks([0, 1]); cbar.set_ticklabels(['Not sig.', 'Sig.'])

    for c in range(len(target_names)):
        ax.text(c, -0.7, f'{sig_matrix[:, c].mean():.0%}',
                ha='center', va='bottom', fontsize=7, color='#555')

    fig.tight_layout()
    plt.show()

    n_cells_any = int((n_sig_per_cell > 0).sum())
    n_targs_any = int((sig_matrix.sum(axis=0) > 0).sum())
    print(f'Cells with ≥1 significant target: {n_cells_any}/{n_cells}')
    print(f'Targets with ≥1 significant cell: {n_targs_any}/{len(target_names)}')


# ── 3 & 4. CV R² and alpha distributions by target group ─────────────────────

def plot_r2_by_group(df, target_names, feat_labels=None):
    """
    Boxplot + strip: per-cell CV R² for each LFP feature split by target group
    (Pre abs / Pre-BL / Post abs / Post-BL / Δ).
    Assumes 25 targets in 5 groups of 5, matching the spe-1 structure.

    Parameters
    ----------
    df           : output of build_metadata_df
    target_names : list[str], length 25
    feat_labels  : list[str]  — x-tick labels (defaults to FEAT_LABELS)
    """
    import seaborn as sns

    if feat_labels is None:
        feat_labels = FEAT_LABELS
    fs = _feat_short(feat_labels)
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 5, figsize=(20, 5), sharey=True)
    fig.suptitle('CV R² across cells — Waveform only model', fontsize=13, y=1.01)

    for g in range(5):
        ax    = axes[g]
        col   = _GRP_COLS[g]
        tns   = target_names[g * 5: (g + 1) * 5]

        for pos, (tn, label) in enumerate(zip(tns, fs)):
            vals = df[f'r2_{tn}'].dropna().values
            ax.boxplot(vals, positions=[pos], widths=0.55, patch_artist=True,
                       medianprops=dict(color='k', lw=2.5),
                       boxprops=dict(facecolor=col, alpha=0.4),
                       whiskerprops=dict(color='#555', lw=1.2),
                       capprops=dict(color='#555', lw=1.2),
                       flierprops=dict(marker='o', markersize=3, alpha=0.3, color=col))
            jitter = rng.uniform(-0.2, 0.2, len(vals))
            ax.scatter(pos + jitter, vals, alpha=0.6, s=18, color=col, zorder=4)
            ax.text(pos, ax.get_ylim()[0] if g else -0.06,
                    f'{np.median(vals):+.3f}', ha='center', va='top',
                    fontsize=7, color='#333')

        ax.axhline(0, color='gray', lw=1.2, ls='--', alpha=0.7)
        ax.set_xticks(range(5))
        ax.set_xticklabels(fs, fontsize=10)
        ax.set_title(_GRP_NAMES[g], fontsize=10, fontweight='bold', color=col, pad=8)
        ax.set_xlabel('LFP feature', fontsize=9)
        if g == 0:
            ax.set_ylabel('5-fold CV R²', fontsize=10)
        sns.despine(ax=ax)

    fig.tight_layout()
    plt.show()


def plot_alpha_by_group(df, target_names, feat_labels=None):
    """
    Boxplot: per-cell best Ridge alpha (log₁₀ scale) per LFP feature by target group.
    High alpha = strong regularisation needed → less signal in that target.

    Parameters
    ----------
    df           : output of build_metadata_df
    target_names : list[str], length 25
    feat_labels  : list[str]  — x-tick labels (defaults to FEAT_LABELS)
    """
    import seaborn as sns

    if feat_labels is None:
        feat_labels = FEAT_LABELS
    fs = _feat_short(feat_labels)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4.5), sharey=True)
    fig.suptitle('Selected Ridge alpha per target (Waveform only)\n'
                 'High alpha = strong regularisation needed (little signal)', fontsize=12)

    for g in range(5):
        ax   = axes[g]
        col  = _GRP_COLS[g]
        tns  = target_names[g * 5: (g + 1) * 5]

        for pos, tn in enumerate(tns):
            vals = np.log10(df[f'alpha_{tn}'].dropna().values.clip(1e-4))
            ax.boxplot(vals, positions=[pos], widths=0.55, patch_artist=True,
                       medianprops=dict(color='k', lw=2.5),
                       boxprops=dict(facecolor=col, alpha=0.4),
                       whiskerprops=dict(color='#555', lw=1.2),
                       capprops=dict(color='#555', lw=1.2),
                       flierprops=dict(marker='o', markersize=3, alpha=0.3, color=col))

        ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.5, label='α=1')
        ax.set_xticks(range(5))
        ax.set_xticklabels(fs, fontsize=10)
        ax.set_title(_GRP_NAMES[g], fontsize=10, fontweight='bold', color=col, pad=8)
        ax.set_xlabel('LFP feature', fontsize=9)
        if g == 0:
            ax.set_ylabel('log₁₀(best alpha)', fontsize=10)
        sns.despine(ax=ax)

    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    plt.show()


# ── 5. Does metadata predict mean R²? ─────────────────────────────────────────

def plot_metadata_vs_mean_r2(df):
    """
    Test whether cell identity predicts a cell's mean CV R² (Waveform only).

    Categorical variables (cell type, patch type, dark neuron, EAP visible):
      Kruskal-Wallis test, box + strip plots, one panel per variable.

    Continuous variable (cortical depth):
      Spearman correlation, scatter + trend line.

    Parameters
    ----------
    df : output of build_metadata_df
    """
    import seaborn as sns

    cat_vars = {
        'Cell type':   'cell_type',
        'Patch type':  'patch_type',
        'Dark neuron': 'dark_neuron',
        'EAP visible': 'eap_visible',
    }

    fig, axes = plt.subplots(1, len(cat_vars), figsize=(4.5 * len(cat_vars), 5))
    fig.suptitle('Does cell metadata predict mean CV R²? (Waveform only)', fontsize=13)

    for ax, (label, col) in zip(axes, cat_vars.items()):
        d = df[['mean_r2_wv', col]].dropna()
        order = sorted(d[col].unique())
        if len(order) < 2:
            ax.set_visible(False)
            continue
        palette = dict(zip(order, _CB[:len(order)]))

        sns.boxplot(data=d, x=col, y='mean_r2_wv', order=order, palette=palette,
                    width=0.5, ax=ax, fliersize=0,
                    boxprops={'alpha': 0.45}, medianprops=dict(color='k', lw=2))
        sns.stripplot(data=d, x=col, y='mean_r2_wv', order=order, palette=palette,
                      alpha=0.75, jitter=0.18, size=7, ax=ax)
        ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.6)

        # Mean line per group
        for i, g in enumerate(order):
            m = d[d[col] == g]['mean_r2_wv'].mean()
            ax.hlines(m, i - 0.3, i + 0.3, color='k', lw=2, zorder=5)

        # Kruskal-Wallis test
        groups      = [d[d[col] == g]['mean_r2_wv'].values for g in order]
        valid_grps  = [g for g in groups if len(g) >= 3]
        if len(valid_grps) >= 2:
            _, p  = stats.kruskal(*valid_grps)
            sig   = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            color = _SIG_COL if sig != 'ns' else '#888'
            ax.text(0.5, 0.97, sig, transform=ax.transAxes, ha='center', va='top',
                    fontsize=14, fontweight='bold', color=color)
            ax.text(0.5, 0.91, f'p={p:.3f}', transform=ax.transAxes,
                    ha='center', va='top', fontsize=8, color='#555')

        for i, (g, grp) in enumerate(zip(order, groups)):
            ax.text(i, ax.get_ylim()[0], f'n={len(grp)}',
                    ha='center', va='bottom', fontsize=8, color='#555')

        ax.set_title(label, fontsize=11, fontweight='bold', pad=10)
        ax.set_xlabel('')
        ax.set_xticklabels(order, fontsize=9)
        ax.set_ylabel('Mean CV R² across targets' if ax is axes[0] else '', fontsize=10)
        sns.despine(ax=ax)

    fig.tight_layout()
    plt.show()

    # Cortical depth: Spearman scatter
    d_depth = df[['mean_r2_wv', 'cort_depth']].dropna()
    if len(d_depth) >= 5:
        rho, p = stats.spearmanr(d_depth['cort_depth'], d_depth['mean_r2_wv'])
        color  = _SIG_COL if p < 0.05 else '#888'
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(d_depth['cort_depth'], d_depth['mean_r2_wv'],
                   color=_CB[0], s=60, alpha=0.8, edgecolors='white', lw=0.5)
        m, b = np.polyfit(d_depth['cort_depth'], d_depth['mean_r2_wv'], 1)
        xs   = np.linspace(d_depth['cort_depth'].min(), d_depth['cort_depth'].max(), 100)
        ax.plot(xs, m * xs + b, color=color, lw=2, ls='--' if p >= 0.05 else '-')
        ax.set_xlabel('Cortical depth (µm)', fontsize=10)
        ax.set_ylabel('Mean CV R²', fontsize=10)
        ax.set_title(f'Cortical depth × mean R²\nSpearman ρ={rho:.3f},  p={p:.3f}', fontsize=11)
        sns.despine(ax=ax)
        plt.tight_layout()
        plt.show()
    else:
        print('Not enough cells with cortical depth data.')


# ── 6. Sample-size confound check ─────────────────────────────────────────────

def plot_n_spikes_confound(df):
    """
    Spearman correlation between n_spikes and mean CV R² (Waveform only).
    Points coloured by n_sig (number of significant targets) to show whether
    more spikes leads to more significant models independently of effect size.

    Parameters
    ----------
    df : output of build_metadata_df
    """
    import seaborn as sns

    # Find the n_sig column for Waveform only
    wv_cols   = [c for c in df.columns if 'waveform_only' in c and c.startswith('n_sig_')]
    n_sig_col = wv_cols[0] if wv_cols else df.columns[df.columns.str.startswith('n_sig_')][0]

    rho, p   = stats.spearmanr(df['n_spikes'].dropna(), df['mean_r2_wv'].dropna())
    line_col = _SIG_COL if p < 0.05 else '#888'

    fig, ax = plt.subplots(figsize=(5, 4))
    sc = ax.scatter(df['n_spikes'], df['mean_r2_wv'],
                    c=df[n_sig_col], cmap='YlOrRd', s=60,
                    alpha=0.85, edgecolors='white', lw=0.5, vmin=0)
    plt.colorbar(sc, ax=ax, label='N sig. targets', shrink=0.7)
    ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.6)

    m, b = np.polyfit(df['n_spikes'].dropna(), df['mean_r2_wv'].dropna(), 1)
    xs   = np.linspace(df['n_spikes'].min(), df['n_spikes'].max(), 100)
    ax.plot(xs, m * xs + b, color=line_col, lw=2, ls='-' if p < 0.05 else '--')

    ax.set_xlabel('N spikes', fontsize=10)
    ax.set_ylabel('Mean CV R² (Waveform only)', fontsize=10)
    ax.set_title(f'Sample size vs model performance\nSpearman ρ={rho:.3f},  p={p:.3f}', fontsize=11)
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
    ax.text(0.97, 0.05, sig, transform=ax.transAxes, ha='right', va='bottom',
            fontsize=14, fontweight='bold', color=line_col)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.show()

    if p < 0.05:
        print('⚠  Sample size significantly predicts R² — potential confound.')
    else:
        print('✓  Sample size does not significantly predict R².')


# ── 7. Beta direction × metadata ──────────────────────────────────────────────

def run_beta_metadata_analysis(beta_pop, df_r2, df, target_names, target_labels,
                                 features=None, n_detail=6):
    """
    For each R²-significant target (Wilcoxon median R²>0, FDR q<0.05, Waveform only):
    test whether cell metadata explains the directionality of beta weights.

    Statistical tests
    -----------------
    Categorical metadata (cell type, patch type, dark neuron, EAP visible):
        Kruskal-Wallis.  Effect size: η² (from H statistic).
        For 2-group variables: Δmedian also reported.
    Continuous metadata (cortical depth, n_spikes):
        Spearman correlation.  Effect size: ρ.
    FDR correction (BH, q<0.05) within each target, across all feature × metadata pairs.

    Outputs per target
    ------------------
    1. P-value + effect-size heatmap (colour = −log₁₀(p); cells annotated with ρ or Δmed or η²).
    2. Printed summary table of top-n_detail pairs.
    3. Detail plots (top-n_detail by raw p, always shown regardless of FDR significance).
       Colour tier: red = FDR q<0.05 | amber = raw p<0.05 | grey = p≥0.05.
       Categorical: boxplot + strip with group medians annotated.
       Continuous: scatter coloured by cell type + bootstrap 95% CI on trend line.

    Parameters
    ----------
    beta_pop      : {target → {feature → ndarray(n_cells)}}  from aggregate_population
    df_r2         : pd.DataFrame  output of run_r2_tests
    df            : pd.DataFrame  output of build_metadata_df
    target_names  : list[str]
    target_labels : list[str]
    features      : list[str]  waveform features to test (default: WAVEFORM_LABELS)
    n_detail      : int  top-N pairs to show in detail plots (default 6)
    """
    import seaborn as sns

    if features is None:
        features = WAVEFORM_LABELS

    meta_cat = {
        'Cell type':   'cell_type',
        'Patch type':  'patch_type',
        'Dark neuron': 'dark_neuron',
        'EAP visible': 'eap_visible',
    }
    meta_cont = {
        'Cort. depth': 'cort_depth',
        'N spikes':    'n_spikes',
    }
    all_meta_labels = list(meta_cat.keys()) + list(meta_cont.keys())

    unique_ctypes = sorted(df['cell_type'].dropna().unique())
    ctype_pal     = {ct: _CB[i % len(_CB)] for i, ct in enumerate(unique_ctypes)}

    sig_r2_df = df_r2[
        (df_r2['sig_r2']) & (df_r2['predictor_set'] == 'Waveform only')
    ].reset_index(drop=True)

    if len(sig_r2_df) == 0:
        print('No R²-significant (Wilcoxon median R²>0) Waveform-only targets.')
        return

    print(f'{len(sig_r2_df)} R²-significant Waveform-only targets:')
    print(sig_r2_df[['target_label', 'median_r2', 'mean_r2', 'p_wilcox_fdr']].to_string(index=False))
    print()

    rng_boot = np.random.default_rng(42)

    for _, trow in sig_r2_df.iterrows():
        tn = trow['target']
        tl = trow['target_label']

        # ── Compute raw p-values + effect sizes ───────────────────────────────
        p_raw_dict = {}
        rho_dict   = {}   # Spearman ρ  (continuous)
        dmed_dict  = {}   # Δmedian     (2-group categorical)
        eta2_dict  = {}   # η²          (all categorical via KW H)

        for f_idx, feat in enumerate(features):
            beta_arr = beta_pop[tn][feat]   # shape (n_cells,), aligned with df

            for m_idx, mlabel in enumerate(all_meta_labels):
                mcol = (meta_cat if mlabel in meta_cat else meta_cont)[mlabel]
                tmp  = df[['cell_id', mcol]].copy()
                tmp['beta'] = beta_arr
                d = tmp[['beta', mcol]].dropna()

                if mlabel in meta_cat:
                    cats  = sorted(d[mcol].unique())
                    grps  = [(c, d[d[mcol] == c]['beta'].values) for c in cats]
                    valid = [(c, g) for c, g in grps if len(g) >= 3]
                    if len(valid) >= 2:
                        valid_vals = [g for _, g in valid]
                        H, p = stats.kruskal(*valid_vals)
                        p_raw_dict[(f_idx, m_idx)] = float(p)
                        k     = len(valid_vals)
                        n_tot = sum(len(g) for g in valid_vals)
                        eta2_dict[(f_idx, m_idx)] = max(
                            0.0, (H - k + 1) / (n_tot - k)
                        ) if n_tot > k else 0.0
                        if len(valid) == 2:   # report Δmedian for 2-group case
                            dmed_dict[(f_idx, m_idx)] = float(
                                np.median(valid[1][1]) - np.median(valid[0][1])
                            )
                else:
                    if len(d) >= 5:
                        rho, p = stats.spearmanr(d[mcol], d['beta'])
                        p_raw_dict[(f_idx, m_idx)] = float(p)
                        rho_dict[(f_idx, m_idx)]   = float(rho)

        # ── FDR correction within this target ─────────────────────────────────
        keys  = sorted(p_raw_dict.keys())
        ps_in = np.array([p_raw_dict[k] for k in keys])
        rej, p_fdr_arr = fdrcorrection(ps_in, alpha=0.05, method='indep')
        fdr_lookup = {
            k: {'p_raw': p_raw_dict[k], 'p_fdr': float(p_fdr_arr[i]), 'sig': bool(rej[i])}
            for i, k in enumerate(keys)
        }

        p_mat   = np.full((len(features), len(all_meta_labels)), np.nan)
        sig_mat = np.zeros((len(features), len(all_meta_labels)), dtype=bool)
        for (fi, mi), entry in fdr_lookup.items():
            p_mat[fi, mi]   = entry['p_raw']
            sig_mat[fi, mi] = entry['sig']

        # ── Heatmap: −log₁₀(p) colour, effect size + p annotated ─────────────
        log_p   = -np.log10(np.clip(p_mat, 1e-10, 1))
        vmax_lp = max(2.0, float(np.nanmax(log_p)))

        fig, ax = plt.subplots(
            figsize=(len(all_meta_labels) * 1.9 + 2.2, len(features) * 0.9 + 2.8)
        )
        im = ax.imshow(log_p, aspect='auto', cmap='YlOrRd',
                       vmin=0, vmax=vmax_lp, origin='upper')
        ax.axvline(len(meta_cat) - 0.5, color='white', lw=2.5, ls='--')  # cat | cont divider

        for fi in range(len(features)):
            for mi in range(len(all_meta_labels)):
                v = p_mat[fi, mi]
                if not np.isfinite(v):
                    continue
                star = '*' if sig_mat[fi, mi] else ''
                tc   = 'white' if log_p[fi, mi] > vmax_lp * 0.6 else 'black'
                mlabel = all_meta_labels[mi]

                if mlabel in meta_cont:
                    rho = rho_dict.get((fi, mi), np.nan)
                    top = f'ρ={rho:+.2f}' if np.isfinite(rho) else ''
                elif (fi, mi) in dmed_dict:
                    top = f'Δmed={dmed_dict[(fi, mi)]:+.2f}'
                else:
                    eta2 = eta2_dict.get((fi, mi), np.nan)
                    top  = f'η²={eta2:.2f}' if np.isfinite(eta2) else ''

                ax.text(mi, fi - 0.2, top, ha='center', va='center',
                        fontsize=7.5, color=tc, fontweight='bold' if star else 'normal')
                ax.text(mi, fi + 0.2, f'p={v:.3f}{star}', ha='center', va='center',
                        fontsize=7, color=tc, alpha=0.85)

        ax.set_xticks(range(len(all_meta_labels)))
        ax.set_xticklabels(all_meta_labels, rotation=30, ha='right', fontsize=10)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('Metadata variable', fontsize=11)
        ax.set_ylabel('Waveform feature (β)', fontsize=11)
        ax.set_title(
            f'{tl}  —  β × metadata\n'
            f'Colour: −log₁₀(p)  |  ρ = Spearman  |  Δmed = 2-group median diff  '
            f'|  η² = Kruskal-Wallis  |  * = FDR q<0.05',
            fontsize=10, pad=14
        )
        cbar = plt.colorbar(im, ax=ax, label='−log₁₀(p)', shrink=0.55, pad=0.02)
        cbar.ax.axhline(-np.log10(0.05) / vmax_lp, color='black', lw=1.5, ls='--')
        cbar.ax.text(2.2, -np.log10(0.05) / vmax_lp, 'p=0.05', va='center',
                     fontsize=7, transform=cbar.ax.transAxes, clip_on=False)
        fig.tight_layout()
        plt.show()

        # ── Summary table + detail plots ──────────────────────────────────────
        sorted_pairs = sorted(
            [(entry['p_raw'], fi, mi) for (fi, mi), entry in fdr_lookup.items()],
            key=lambda x: x[0]
        )[:n_detail]

        n_fdr = int(sig_mat.sum())
        n_nom = int((p_mat < 0.05).sum())
        print(f'  {tl}:  FDR-sig={n_fdr}  |  nominally-sig (p<0.05)={n_nom}  '
              f'|  showing top-{len(sorted_pairs)} trends\n')
        print(f'  {"Feature":<14}  {"Metadata":<14}  {"Effect":>10}  '
              f'{"p-raw":>7}  {"FDR q":>7}')
        print('  ' + '-' * 60)
        for p_v, fi, mi in sorted_pairs:
            mlabel = all_meta_labels[mi]
            feat   = features[fi]
            eff    = (f'ρ={rho_dict.get((fi,mi),float("nan")):+.3f}'    if mlabel in meta_cont else
                      f'Δmed={dmed_dict[(fi,mi)]:+.3f}'                  if (fi,mi) in dmed_dict  else
                      f'η²={eta2_dict.get((fi,mi),float("nan")):.3f}')
            entry   = fdr_lookup[(fi, mi)]
            sig_tag = '***' if p_v < 0.001 else '**' if p_v < 0.01 else '*' if p_v < 0.05 else ''
            fdr_tag = '(FDR*)' if entry['sig'] else ''
            print(f'  {feat:<14}  {mlabel:<14}  {eff:>10}  '
                  f'{p_v:>7.4f}  {entry["p_fdr"]:>7.4f}  {sig_tag} {fdr_tag}')
        print()

        ncols      = min(len(sorted_pairs), 3)
        nrows      = math.ceil(len(sorted_pairs) / ncols)
        fig2, ax2s = plt.subplots(nrows, ncols,
                                   figsize=(ncols * 4.8, nrows * 4.8),
                                   squeeze=False)
        ax2_flat   = ax2s.flatten()
        fig2.suptitle(
            f'{tl}  —  top {n_detail} β × metadata trends\n'
            f'red = FDR q<0.05  |  amber = raw p<0.05  |  grey = p≥0.05',
            fontsize=11, y=1.01
        )

        for ax_i, (p_v, fi, mi) in enumerate(sorted_pairs):
            ax2    = ax2_flat[ax_i]
            feat   = features[fi]
            mlabel = all_meta_labels[mi]
            entry  = fdr_lookup[(fi, mi)]
            mcol   = (meta_cat if mlabel in meta_cat else meta_cont)[mlabel]
            col    = _SIG_COL if entry['sig'] else (_NOM_COL if p_v < 0.05 else '#888888')

            tmp = df[['cell_id', mcol, 'cell_type']].copy()
            tmp['beta'] = beta_pop[tn][feat]
            d   = tmp[['beta', mcol, 'cell_type']].dropna(subset=['beta', mcol])

            if mlabel in meta_cat:
                order = sorted(d[mcol].unique())
                pal   = dict(zip(order, _CB[:len(order)]))
                sns.boxplot(data=d, x=mcol, y='beta', order=order, palette=pal,
                            width=0.5, ax=ax2, fliersize=0,
                            boxprops={'alpha': 0.35}, medianprops=dict(color='k', lw=2.5))
                sns.stripplot(data=d, x=mcol, y='beta', order=order, palette=pal,
                              alpha=0.7, jitter=0.18, size=7, ax=ax2, zorder=4)
                ax2.axhline(0, color='gray', lw=1.2, ls='--', alpha=0.7)
                y_bot = ax2.get_ylim()[0]
                for i_g, g in enumerate(order):
                    grp = d[d[mcol] == g]['beta'].values
                    ax2.text(i_g, y_bot, f'n={len(grp)}\nmed={np.median(grp):+.3f}',
                             ha='center', va='bottom', fontsize=7.5, color='#444')
                eta2    = eta2_dict.get((fi, mi), np.nan)
                dmed    = dmed_dict.get((fi, mi), np.nan)
                eff_str = (f'Δmed={dmed:+.3f}  η²={eta2:.2f}' if np.isfinite(dmed)
                           else f'η²={eta2:.2f}' if np.isfinite(eta2) else '')
                sig_str = '***' if p_v < 0.001 else '**' if p_v < 0.01 else '*' if p_v < 0.05 else 'ns'
                ax2.text(0.5, 0.97, sig_str, transform=ax2.transAxes,
                         ha='center', va='top', fontsize=14, fontweight='bold', color=col)
                ax2.text(0.5, 0.91, f'{eff_str}\np={p_v:.3f}  FDR q={entry["p_fdr"]:.3f}',
                         transform=ax2.transAxes, ha='center', va='top',
                         fontsize=8, color='#444')
                ax2.set_xlabel(mlabel, fontsize=10)
                ax2.set_xticklabels(order, fontsize=9)
            else:
                # Continuous: scatter coloured by cell type + bootstrap CI
                rho    = rho_dict.get((fi, mi), np.nan)
                x_vals = d[mcol].values
                y_vals = d['beta'].values
                for ct in unique_ctypes:
                    mask = d['cell_type'] == ct
                    if mask.sum():
                        ax2.scatter(d.loc[mask, mcol], d.loc[mask, 'beta'],
                                    color=ctype_pal[ct], s=65, alpha=0.82,
                                    edgecolors='white', lw=0.5, zorder=4, label=ct)
                if len(d) >= 4:
                    m_fit, b_fit = np.polyfit(x_vals, y_vals, 1)
                    xs = np.linspace(x_vals.min(), x_vals.max(), 100)
                    ax2.plot(xs, m_fit * xs + b_fit, color=col, lw=2.2,
                             ls='-' if p_v < 0.05 else '--', zorder=5)
                    boot_lines = []
                    for _ in range(500):
                        idx = rng_boot.integers(0, len(x_vals), len(x_vals))
                        xb, yb = x_vals[idx], y_vals[idx]
                        if len(np.unique(xb)) > 1:
                            mb, bb = np.polyfit(xb, yb, 1)
                            boot_lines.append(mb * xs + bb)
                    if len(boot_lines) >= 10:
                        ax2.fill_between(xs,
                                         np.percentile(boot_lines, 2.5, axis=0),
                                         np.percentile(boot_lines, 97.5, axis=0),
                                         color=col, alpha=0.13, zorder=3)
                ax2.axhline(0, color='gray', lw=1, ls='--', alpha=0.6)
                ax2.set_xlabel(mlabel, fontsize=10)
                ax2.legend(fontsize=8, frameon=False, title='Cell type',
                           title_fontsize=8, loc='upper left')
                ax2.text(0.97, 0.97,
                         f'ρ = {rho:+.3f}\np = {p_v:.3f}\nFDR q = {entry["p_fdr"]:.3f}',
                         transform=ax2.transAxes, ha='right', va='top', fontsize=9, color=col,
                         bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.85))

            ax2.set_ylabel(f'β  {feat}  (std units)', fontsize=10)
            ax2.set_title(f'{feat}  ×  {mlabel}', fontsize=10, fontweight='bold')
            sns.despine(ax=ax2)

        for i in range(len(sorted_pairs), len(ax2_flat)):
            ax2_flat[i].set_visible(False)
        fig2.tight_layout()
        plt.show()
        print()


# ── 8. Predictor set comparison ───────────────────────────────────────────────

def plot_predictor_set_comparison(df, predictor_sets):
    """
    Compare n_sig across predictor sets.

    Left panel:  histogram of n_sig per cell for the first predictor set.
    Right panel: mean n_sig per predictor set (bar + jitter),
                 Wilcoxon paired tests vs first predictor set.

    Parameters
    ----------
    df             : output of build_metadata_df
    predictor_sets : list[str]  same list used in build_metadata_df
    """
    import seaborn as sns

    # Map predictor set names → column names created by build_metadata_df
    pset_data = {}
    for pn in predictor_sets:
        col = f'n_sig_{_pset_col(pn)}'
        if col in df.columns:
            pset_data[pn] = df[col].values

    if len(pset_data) < 2:
        print('Need at least 2 predictor sets to compare.')
        return

    first_pn   = predictor_sets[0]
    first_vals = pset_data[first_pn]
    first_col  = f'n_sig_{_pset_col(first_pn)}'
    n_targets  = sum(1 for c in df.columns if c.startswith('sig_'))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('How much does each predictor set explain?', fontsize=12)

    # Left: n_sig distribution for first predictor set
    axes[0].hist(df[first_col], bins=range(0, n_targets + 2),
                 color=_CB[0], alpha=0.85, edgecolor='white', linewidth=0.8)
    axes[0].axvline(df[first_col].mean(), color=_SIG_COL, lw=2, ls='--',
                    label=f'Mean = {df[first_col].mean():.1f}')
    axes[0].set_xlabel('N FDR-significant targets per cell', fontsize=10)
    axes[0].set_ylabel('N cells', fontsize=10)
    axes[0].set_title(f'{first_pn} — distribution across cells', fontsize=10)
    axes[0].legend(fontsize=9, frameon=False)
    axes[0].set_xticks(range(n_targets + 1))
    sns.despine(ax=axes[0])

    # Right: bar + jitter per predictor set
    rng = np.random.default_rng(42)
    for i, (pn, vals) in enumerate(pset_data.items()):
        col   = _CB[i % len(_CB)]
        mean_ = np.mean(vals)
        sem_  = stats.sem(vals)
        axes[1].bar(i, mean_, color=col, alpha=0.6, width=0.5, edgecolor='white')
        axes[1].errorbar(i, mean_, yerr=sem_, fmt='none', color='k', capsize=5, lw=2)
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        axes[1].scatter(i + jitter, vals, color=col, alpha=0.7, s=20, zorder=3)

    y_top = max(v.max() for v in pset_data.values())
    for i, (pn, vals) in enumerate(list(pset_data.items())[1:], start=1):
        try:
            _, p = stats.wilcoxon(first_vals, vals)
        except ValueError:
            continue
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        y   = y_top + 0.5 + (i - 1) * 0.8
        c   = _SIG_COL if sig != 'ns' else '#888'
        axes[1].plot([0, i], [y, y], color=c, lw=1.2)
        axes[1].text((0 + i) / 2, y + 0.1, sig, ha='center',
                     fontsize=11, fontweight='bold', color=c)

    axes[1].set_xticks(range(len(pset_data)))
    axes[1].set_xticklabels(
        [pn.replace(' + ', '\n+ ').replace(' only', '\nonly') for pn in pset_data],
        fontsize=9
    )
    axes[1].set_ylabel('Mean N significant targets per cell', fontsize=10)
    axes[1].set_title(f'Predictor set comparison\n(Wilcoxon vs {first_pn})', fontsize=10)
    sns.despine(ax=axes[1])

    fig.tight_layout()
    plt.show()
