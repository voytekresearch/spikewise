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

# ── Colour palette (colour-blind-friendly, Wong 2011) ─────────────────────────
_CB      = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#56B4E9', '#E69F00']
_SIG_COL = '#D55E00'   # FDR-significant  (red-orange)
_NOM_COL = '#E69F00'   # nominally p<0.05 (amber)
_NSG_COL = '#888888'   # not significant  (grey)

# Target-group colours and display names (Pre / Pre-BL / Post / Post-BL / Δ)
_GRP_COLS  = ['#1976D2', '#43A047', '#9C27B0', '#FF9800', '#E53935']
_GRP_NAMES = ['Pre (abs)', 'Pre − BL', 'Post (abs)', 'Post − BL', 'Δ post−pre']

# Consistent font sizes used throughout
_FS_TITLE  = 12
_FS_LABEL  = 10
_FS_TICK   = 9
_FS_ANNOT  = 8
_FS_SMALL  = 7


def _pset_col(pn):
    """Predictor set name → safe DataFrame column suffix."""
    return pn.replace(' + ', '_').replace(' ', '_').lower()


def _feat_short(feat_labels):
    """Return abbreviated labels for a list of spe-1 LFP feature names."""
    return [
        l.replace('LFP ', '').replace(' AUC', '').replace('onent', '')
        for l in feat_labels
    ]


def _sig_str(p):
    """Return significance star string for a p-value."""
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


# ── 1. Build metadata DataFrame ───────────────────────────────────────────────

def build_metadata_df(all_results, cell_ids, target_names, target_labels,
                       predictor_sets, dict_cell_type, dict_patch_type,
                       dict_cort_depth, dict_dark_neurons, dict_eap_wav,
                       dict_firing_rate=None, dict_rec_duration=None):
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
    dict_firing_rate  : dict {cell_num → float} Hz, optional
    dict_rec_duration : dict {cell_num → float} minutes, optional

    Returns
    -------
    df : pd.DataFrame
        One row per cell.  Columns:
          cell_id, cell_num, cell_type, patch_type, cort_depth, dark_neuron,
          eap_visible, n_spikes, firing_rate_hz, rec_duration_min, mean_r2_wv,
          n_sig_<pset> for each predictor set,
          r2_<target>, sig_<target>, alpha_<target> (Waveform only) per target.
    """
    rows = []
    for cid in cell_ids:
        cell_num = int(cid.replace('c', ''))
        res = all_results[cid]

        row = dict(
            cell_id          = cid,
            cell_num         = cell_num,
            cell_type        = dict_cell_type.get(cell_num, 'unknown'),
            patch_type       = dict_patch_type.get(cell_num, 'unknown'),
            cort_depth       = dict_cort_depth.get(cell_num, np.nan),
            dark_neuron      = str(dict_dark_neurons.get(cell_num, np.nan)),
            eap_visible      = str(dict_eap_wav.get(cell_num, np.nan)),
            n_spikes         = res[target_names[0]]['Waveform only'].get('n_valid', np.nan),
            firing_rate_hz   = dict_firing_rate.get(cell_num, np.nan)  if dict_firing_rate  else np.nan,
            rec_duration_min = dict_rec_duration.get(cell_num, np.nan) if dict_rec_duration else np.nan,
            mean_r2_wv       = float(np.nanmean([
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
    Blue = FDR-significant Waveform-only model; white = not significant.

    Above each column group: group label (colour-coded).
    Below each column:       fraction of cells with a significant model.
    Right y-axis:            cell ID + count of significant targets.

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
    n_targs         = len(target_names)

    # Extra top/bottom margin for labels outside the image
    fig, ax = plt.subplots(figsize=(max(16, n_targs * 0.7), max(9, 0.42 * n_cells + 3)))

    im = ax.imshow(sig_matrix[sort_idx], aspect='auto', cmap='Blues', vmin=0, vmax=1,
                   extent=[-0.5, n_targs - 0.5, n_cells - 0.5, -0.5])

    # Vertical white dividers between the 5 target groups (every 5 columns)
    for d_pos in [4.5, 9.5, 14.5, 19.5]:
        ax.axvline(d_pos, color='white', lw=2.5)

    # Group labels — placed ABOVE the image using the xaxis blend transform
    # (data x coordinates, axes-fraction y — so y=1 is the top edge of the axes)
    trans_x = ax.get_xaxis_transform()
    for g_idx, (lbl, x_centre) in enumerate(
        [('Pre (abs)', 2), ('Pre − BL', 7), ('Post (abs)', 12),
         ('Post − BL', 17), ('Δ post−pre', 22)]
    ):
        ax.text(x_centre, 1.04, lbl,
                ha='center', va='bottom', fontsize=_FS_TICK,
                fontweight='bold', color=_GRP_COLS[g_idx],
                transform=trans_x)

    # Column tick labels (strip the group prefix, keep the feature name)
    ax.set_xticks(range(n_targs))
    ax.set_xticklabels(
        [tl.split(' ', 1)[-1] for tl in target_labels],
        rotation=40, ha='right', fontsize=_FS_SMALL
    )

    # Fraction-significant annotation below x-axis ticks
    for c in range(n_targs):
        ax.text(c, -0.12, f'{sig_matrix[:, c].mean():.0%}',
                ha='center', va='top', fontsize=_FS_SMALL, color='#555',
                transform=trans_x)
    # Label for the fraction row
    ax.text(-0.5, -0.12, 'frac\nsig', ha='right', va='top',
            fontsize=_FS_SMALL - 1, color='#555', transform=trans_x)

    # Row labels: cell ID + n significant
    ax.set_yticks(range(n_cells))
    ax.set_yticklabels(
        [f'{cell_ids_sorted[i]}  ({int(n_sig_per_cell[sort_idx[i]])})'
         for i in range(n_cells)],
        fontsize=_FS_SMALL + 1
    )

    ax.set_xlabel('LFP target  (feature within group)', fontsize=_FS_LABEL, labelpad=28)
    ax.set_ylabel('Cell  (n significant targets in parentheses)', fontsize=_FS_LABEL)
    ax.set_title(
        'FDR-significant Waveform-only models per cell × LFP target\n'
        'Cells sorted top → bottom by total n significant; '
        'group labels colour-coded above; fraction significant shown below',
        fontsize=_FS_TITLE, pad=30
    )

    cbar = plt.colorbar(im, ax=ax, shrink=0.25, pad=0.01, aspect=12)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Not sig.', 'Sig.'], fontsize=_FS_SMALL)

    fig.tight_layout()
    plt.show()

    n_cells_any = int((n_sig_per_cell > 0).sum())
    n_targs_any = int((sig_matrix.sum(axis=0) > 0).sum())
    print(f'Cells with ≥1 significant target: {n_cells_any}/{n_cells}')
    print(f'Targets with ≥1 significant cell: {n_targs_any}/{n_targs}')


# ── 3 & 4. CV R² and alpha distributions by target group ─────────────────────

def plot_r2_by_group(df, target_names, feat_labels=None):
    """
    Boxplot + individual-cell strip: 5-fold CV R² for each LFP feature,
    split across the 5 target groups (Pre abs / Pre-BL / Post abs / Post-BL / Δ).

    Median annotated below each feature on the x-axis (outside data area).
    Reference line at R² = 0.

    Parameters
    ----------
    df           : output of build_metadata_df
    target_names : list[str], length 25 (5 groups × 5 features)
    feat_labels  : list[str]  display labels (defaults to FEAT_LABELS)
    """
    import seaborn as sns

    if feat_labels is None:
        feat_labels = FEAT_LABELS
    fs  = _feat_short(feat_labels)
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5), sharey=True)
    fig.suptitle(
        'CV R² across cells — Waveform-only predictor set\n'
        'Each point = one cell; median annotated below x-axis',
        fontsize=_FS_TITLE, y=1.02
    )

    # Collect all values first so we can set a common ylim before annotating
    all_vals = []
    for g in range(5):
        for tn in target_names[g * 5: (g + 1) * 5]:
            all_vals.extend(df[f'r2_{tn}'].dropna().values)
    y_min = min(min(all_vals) - 0.04, -0.06)
    y_max = max(all_vals) + 0.04

    for g, ax in enumerate(axes):
        col  = _GRP_COLS[g]
        tns  = target_names[g * 5: (g + 1) * 5]
        trans_x = ax.get_xaxis_transform()   # data-x, axes-fraction y

        for pos, tn in enumerate(tns):
            vals = df[f'r2_{tn}'].dropna().values
            ax.boxplot(
                vals, positions=[pos], widths=0.55, patch_artist=True,
                medianprops=dict(color='k', lw=2.5),
                boxprops=dict(facecolor=col, alpha=0.4),
                whiskerprops=dict(color='#555', lw=1.2),
                capprops=dict(color='#555', lw=1.2),
                flierprops=dict(marker='o', markersize=3, alpha=0.3, color=col),
                zorder=3,
            )
            jitter = rng.uniform(-0.2, 0.2, len(vals))
            ax.scatter(pos + jitter, vals, alpha=0.55, s=16, color=col, zorder=4)

            # Median annotation in axes-fraction y — safely outside data area
            ax.text(pos, -0.10, f'{np.median(vals):+.3f}',
                    ha='center', va='top', fontsize=_FS_SMALL, color='#333',
                    transform=trans_x)

        ax.set_ylim(y_min, y_max)
        ax.axhline(0, color='gray', lw=1.2, ls='--', alpha=0.7, zorder=2)
        ax.set_xticks(range(5))
        ax.set_xticklabels(fs, fontsize=_FS_TICK)
        ax.set_title(_GRP_NAMES[g], fontsize=_FS_TICK + 1, fontweight='bold',
                     color=col, pad=8)
        ax.set_xlabel('LFP feature', fontsize=_FS_ANNOT)
        if g == 0:
            ax.set_ylabel('5-fold CV R²', fontsize=_FS_LABEL)
        sns.despine(ax=ax)

    fig.tight_layout()
    plt.show()


def plot_alpha_by_group(df, target_names, feat_labels=None):
    """
    Boxplot: best Ridge regularisation strength (log₁₀ alpha) per cell,
    grouped by LFP feature and target group.

    Higher alpha = stronger regularisation selected = less predictive signal
    in that target.  Reference line at log₁₀(α) = 0 (α = 1).

    Parameters
    ----------
    df           : output of build_metadata_df
    target_names : list[str], length 25
    feat_labels  : list[str]  display labels (defaults to FEAT_LABELS)
    """
    import seaborn as sns

    if feat_labels is None:
        feat_labels = FEAT_LABELS
    fs = _feat_short(feat_labels)

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.5), sharey=True)
    fig.suptitle(
        'Ridge regularisation strength (log₁₀ α) per cell × target group\n'
        'High α = strong regularisation needed → weak signal in that target;  '
        'dashed line = α = 1',
        fontsize=_FS_TITLE, y=1.02
    )

    for g, ax in enumerate(axes):
        col  = _GRP_COLS[g]
        tns  = target_names[g * 5: (g + 1) * 5]

        for pos, tn in enumerate(tns):
            vals = np.log10(df[f'alpha_{tn}'].dropna().values.clip(1e-4))
            ax.boxplot(
                vals, positions=[pos], widths=0.55, patch_artist=True,
                medianprops=dict(color='k', lw=2.5),
                boxprops=dict(facecolor=col, alpha=0.4),
                whiskerprops=dict(color='#555', lw=1.2),
                capprops=dict(color='#555', lw=1.2),
                flierprops=dict(marker='o', markersize=3, alpha=0.3, color=col),
            )

        ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.6)
        ax.set_xticks(range(5))
        ax.set_xticklabels(fs, fontsize=_FS_TICK)
        ax.set_title(_GRP_NAMES[g], fontsize=_FS_TICK + 1, fontweight='bold',
                     color=col, pad=8)
        ax.set_xlabel('LFP feature', fontsize=_FS_ANNOT)
        if g == 0:
            ax.set_ylabel('log₁₀(best α)', fontsize=_FS_LABEL)
        sns.despine(ax=ax)

    fig.tight_layout()
    plt.show()


# ── 5. Does metadata predict mean R²? ─────────────────────────────────────────

def plot_metadata_vs_mean_r2(df):
    """
    Test whether intrinsic cell properties explain how well a cell's waveform
    predicts LFP features (Waveform-only predictor set, mean CV R² across all targets).

    Categorical variables (cell type, patch type, dark neuron, EAP visibility):
      Kruskal-Wallis H-test.  Result shown as significance stars in the panel title.

    Continuous variable (cortical depth):
      Spearman ρ.  Shown as scatter + trend line in a separate panel.

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

    fig, axes = plt.subplots(1, len(cat_vars), figsize=(5.0 * len(cat_vars), 5.5),
                              constrained_layout=True)
    fig.suptitle(
        'Does cell identity predict mean CV R²? (Waveform-only model)\n'
        'Kruskal-Wallis test; significance in title; n per group below x-axis',
        fontsize=_FS_TITLE
    )

    for ax, (label, col) in zip(axes, cat_vars.items()):
        d     = df[['mean_r2_wv', col]].dropna()
        order = sorted(d[col].unique())
        if len(order) < 2:
            ax.set_visible(False)
            continue

        palette = dict(zip(order, _CB[:len(order)]))
        trans_x = ax.get_xaxis_transform()

        sns.boxplot(data=d, x=col, y='mean_r2_wv', order=order, palette=palette,
                    width=0.5, ax=ax, fliersize=0,
                    boxprops={'alpha': 0.45}, medianprops=dict(color='k', lw=2))
        sns.stripplot(data=d, x=col, y='mean_r2_wv', order=order, palette=palette,
                      alpha=0.75, jitter=0.18, size=7, ax=ax, zorder=4)
        ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.6)

        # Kruskal-Wallis test
        groups     = [d[d[col] == g]['mean_r2_wv'].values for g in order]
        valid_grps = [g for g in groups if len(g) >= 3]
        sig = 'ns'
        if len(valid_grps) >= 2:
            _, p = stats.kruskal(*valid_grps)
            sig  = _sig_str(p)
            p_str = f'p = {p:.3f}'
        else:
            p_str = ''

        sig_col = _SIG_COL if sig != 'ns' else _NSG_COL

        # n per group — below x-ticks using axes-fraction y (no data overlap)
        for i, (g, grp) in enumerate(zip(order, groups)):
            ax.text(i, -0.10, f'n={len(grp)}',
                    ha='center', va='top', fontsize=_FS_SMALL, color='#555',
                    transform=trans_x)

        ax.set_title(f'{label}\n{sig}  {p_str}', fontsize=_FS_TICK + 1,
                     fontweight='bold', color=sig_col, pad=6)
        ax.set_xlabel('')
        ax.set_xticklabels(order, fontsize=_FS_TICK)
        ax.set_ylabel('Mean CV R² across all targets' if ax is axes[0] else '',
                      fontsize=_FS_LABEL)
        sns.despine(ax=ax)

    # Cortical depth: Spearman scatter (separate figure)
    d_depth = df[['mean_r2_wv', 'cort_depth']].dropna()
    if len(d_depth) >= 5:
        rho, p = stats.spearmanr(d_depth['cort_depth'], d_depth['mean_r2_wv'])
        sig     = _sig_str(p)
        line_col = _SIG_COL if p < 0.05 else _NSG_COL

        fig2, ax2 = plt.subplots(figsize=(5, 4.5))
        ax2.scatter(d_depth['cort_depth'], d_depth['mean_r2_wv'],
                    color=_CB[0], s=65, alpha=0.82, edgecolors='white', lw=0.5)

        m, b = np.polyfit(d_depth['cort_depth'], d_depth['mean_r2_wv'], 1)
        xs   = np.linspace(d_depth['cort_depth'].min(), d_depth['cort_depth'].max(), 100)
        ax2.plot(xs, m * xs + b, color=line_col, lw=2,
                 ls='-' if p < 0.05 else '--')

        ax2.set_xlabel('Cortical depth (µm)', fontsize=_FS_LABEL)
        ax2.set_ylabel('Mean CV R²', fontsize=_FS_LABEL)
        ax2.set_title(
            f'Cortical depth × mean R²\n'
            f'Spearman ρ = {rho:+.3f}   p = {p:.3f}   {sig}',
            fontsize=_FS_TITLE, color=line_col
        )
        sns.despine(ax=ax2)
        fig2.tight_layout()
        plt.show()
    else:
        print('Not enough cells with cortical depth data for scatter.')


# ── 6. Sample-size confound check ─────────────────────────────────────────────

def plot_n_spikes_confound(df):
    """
    Check whether n_spikes, firing rate, or recording duration confound mean CV R².

    n_spikes = firing_rate × recording_duration, so it conflates cell activity
    with how long the cell was recorded. Three panels show each separately so
    you can tell which (if either) is the real driver.

    Points coloured by number of FDR-significant targets.

    Parameters
    ----------
    df : output of build_metadata_df  (must include firing_rate_hz,
         rec_duration_min if those confounds are to be tested; if absent,
         those panels are skipped)
    """
    import seaborn as sns

    wv_cols   = [c for c in df.columns if 'waveform_only' in c and c.startswith('n_sig_')]
    n_sig_col = wv_cols[0] if wv_cols else df.columns[df.columns.str.startswith('n_sig_')][0]

    panels = [('n_spikes', 'N spikes\n(firing rate × duration)')]
    if 'firing_rate_hz' in df.columns and df['firing_rate_hz'].notna().sum() > 3:
        panels.append(('firing_rate_hz', 'Firing rate (Hz)'))
    if 'rec_duration_min' in df.columns and df['rec_duration_min'].notna().sum() > 3:
        panels.append(('rec_duration_min', 'Recording duration (min)'))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 4.5),
                             constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]

    fig.suptitle(
        'Sample size confound check  (Waveform-only model)\n'
        'n_spikes = firing rate × recording duration — shown separately to\n'
        'distinguish cell activity from recording length as confounds',
        fontsize=_FS_TITLE
    )

    for ax, (xcol, xlabel) in zip(axes, panels):
        d = df[[xcol, 'mean_r2_wv', n_sig_col]].dropna()
        rho, p   = stats.spearmanr(d[xcol], d['mean_r2_wv'])
        sig      = _sig_str(p)
        line_col = _SIG_COL if p < 0.05 else _NSG_COL

        sc = ax.scatter(d[xcol], d['mean_r2_wv'],
                        c=d[n_sig_col], cmap='YlOrRd', s=65,
                        alpha=0.85, edgecolors='white', lw=0.5, vmin=0)
        plt.colorbar(sc, ax=ax, label='N FDR-sig. targets', shrink=0.7)
        ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.6)

        m, b = np.polyfit(d[xcol], d['mean_r2_wv'], 1)
        xs   = np.linspace(d[xcol].min(), d[xcol].max(), 100)
        ax.plot(xs, m * xs + b, color=line_col, lw=2,
                ls='-' if p < 0.05 else '--')

        ax.set_xlabel(xlabel, fontsize=_FS_LABEL)
        ax.set_ylabel('Mean CV R²  (Waveform-only)', fontsize=_FS_LABEL)
        ax.set_title(
            f'Spearman ρ = {rho:+.3f}   p = {p:.3f}   {sig}',
            fontsize=_FS_TICK + 1, color=line_col
        )
        sns.despine(ax=ax)

        flag = '⚠  potential confound' if p < 0.05 else '✓  not a confound'
        print(f'{xcol:20s}  rho={rho:+.3f}  p={p:.3f}  {sig}  {flag}')

    plt.show()


# ── 6b. Recording duration and firing rate × metadata ────────────────────────

def plot_duration_rate_vs_metadata(df):
    """
    Test whether recording duration and firing rate are correlated with the
    same cell metadata variables used in the R² analysis.

    If duration or firing rate correlates with cell type / patch type / etc.,
    those metadata effects on R² could be driven by recording differences
    rather than biology — a methodological confound worth flagging.

    Layout: two rows (duration, firing rate) × N metadata variables.
    Categorical: Kruskal-Wallis, box + strip.  Continuous: Spearman scatter.
    Significant panels (p < 0.05, uncorrected) highlighted in orange.

    Parameters
    ----------
    df : output of build_metadata_df  (must contain firing_rate_hz and
         rec_duration_min columns)
    """
    import seaborn as sns

    outcomes = []
    if 'rec_duration_min' in df.columns and df['rec_duration_min'].notna().sum() > 3:
        outcomes.append(('rec_duration_min', 'Recording duration (min)'))
    if 'firing_rate_hz' in df.columns and df['firing_rate_hz'].notna().sum() > 3:
        outcomes.append(('firing_rate_hz', 'Firing rate (Hz)'))

    if not outcomes:
        print('No firing_rate_hz or rec_duration_min columns — run build_metadata_df '
              'with dict_firing_rate and dict_rec_duration.')
        return

    cat_vars = {
        'Cell type':   'cell_type',
        'Patch type':  'patch_type',
        'Dark neuron': 'dark_neuron',
        'EAP visible': 'eap_visible',
    }
    cont_vars = {'Cort. depth': 'cort_depth'}
    all_vars  = list(cat_vars.items()) + list(cont_vars.items())

    n_rows = len(outcomes)
    n_cols = len(all_vars)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.5 * n_cols, 4.8 * n_rows),
                             constrained_layout=True)
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    fig.suptitle(
        'Are recording duration and firing rate confounded with cell metadata?\n'
        'Significant associations (p<0.05) would mean metadata × R² effects '
        'could be driven by recording differences, not biology.',
        fontsize=_FS_TITLE
    )

    for r, (ycol, ylabel) in enumerate(outcomes):
        for c, (label, xcol) in enumerate(all_vars):
            ax = axes[r, c]
            d  = df[[ycol, xcol]].dropna()

            if label in cat_vars:
                order = sorted(d[xcol].unique())
                if len(order) < 2:
                    ax.set_visible(False)
                    continue
                pal = dict(zip(order, _CB[:len(order)]))
                trans_x = ax.get_xaxis_transform()

                sns.boxplot(data=d, x=xcol, y=ycol, order=order, palette=pal,
                            width=0.5, ax=ax, fliersize=0,
                            boxprops={'alpha': 0.45},
                            medianprops=dict(color='k', lw=2))
                sns.stripplot(data=d, x=xcol, y=ycol, order=order, palette=pal,
                              alpha=0.75, jitter=0.18, size=7, ax=ax, zorder=4)

                groups = [d[d[xcol] == g][ycol].values for g in order]
                valid  = [g for g in groups if len(g) >= 3]
                if len(valid) >= 2:
                    _, p = stats.kruskal(*valid)
                    sig  = _sig_str(p)
                else:
                    p, sig = 1.0, 'ns'

                col = _SIG_COL if p < 0.05 else _NSG_COL
                for i, (g, grp) in enumerate(zip(order, groups)):
                    ax.text(i, -0.10, f'n={len(grp)}', ha='center', va='top',
                            fontsize=_FS_SMALL, color='#555', transform=trans_x)
                ax.set_xticklabels(order, fontsize=_FS_TICK)

            else:
                # continuous: Spearman scatter
                rho, p = stats.spearmanr(d[xcol], d[ycol])
                sig    = _sig_str(p)
                col    = _SIG_COL if p < 0.05 else _NSG_COL

                ax.scatter(d[xcol], d[ycol], color=_CB[0], s=55,
                           alpha=0.8, edgecolors='white', lw=0.5)
                if len(d) >= 4:
                    m_, b_ = np.polyfit(d[xcol], d[ycol], 1)
                    xs = np.linspace(d[xcol].min(), d[xcol].max(), 100)
                    ax.plot(xs, m_ * xs + b_, color=col, lw=2,
                            ls='-' if p < 0.05 else '--')
                ax.set_xlabel(label, fontsize=_FS_ANNOT)
                ax.text(0.97, 0.97, f'ρ={rho:+.2f}\np={p:.3f}',
                        transform=ax.transAxes, ha='right', va='top',
                        fontsize=_FS_SMALL, color=col,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                  ec='#ccc', alpha=0.9))

            ax.set_title(f'{label}  [{sig}]', fontsize=_FS_TICK + 1,
                         fontweight='bold', color=col, pad=6)
            ax.set_xlabel(label if label in cont_vars else '', fontsize=_FS_ANNOT)
            ax.set_ylabel(ylabel if c == 0 else '', fontsize=_FS_LABEL)
            sns.despine(ax=ax)

    plt.show()


# ── 7. Beta direction × metadata ──────────────────────────────────────────────

def run_beta_metadata_analysis(beta_pop, df_r2, df, target_names, target_labels,
                                 features=None, n_detail=6):
    """
    For each R²-significant target (Wilcoxon median R²>0, FDR q<0.05, Waveform only):
    test whether cell metadata explains the *direction* of beta weights.

    Statistical tests
    -----------------
    Categorical metadata (cell type, patch type, dark neuron, EAP visible):
        Kruskal-Wallis H-test.  Effect size: η² = (H − k + 1) / (n − k).
        For 2-group variables: Δmedian also reported.
    Continuous metadata (cortical depth, n_spikes):
        Spearman correlation.  Effect size: ρ.
    FDR correction (BH, q<0.05) within each target across all
    feature × metadata combinations.

    Outputs per R²-significant target
    ----------------------------------
    1. Heatmap: −log₁₀(p) colour;  ρ / Δmed / η² annotated per cell;
       FDR-significant cells marked with *.
    2. Printed summary table of top-n_detail pairs.
    3. Detail plots: top-n_detail by raw p (always shown regardless of FDR).
       Colour: red = FDR q<0.05 | amber = raw p<0.05 | grey = p≥0.05.
       Categorical: box + strip with group n and median below x-axis.
       Continuous: scatter coloured by cell type + bootstrap 95% CI line.

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
                        eta2_dict[(f_idx, m_idx)] = (
                            max(0.0, (H - k + 1) / (n_tot - k))
                            if n_tot > k else 0.0
                        )
                        if len(valid) == 2:
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

        # ── Heatmap: −log₁₀(p) with effect size + p annotated ────────────────
        log_p   = -np.log10(np.clip(p_mat, 1e-10, 1))
        vmax_lp = max(2.0, float(np.nanmax(log_p)))

        n_feat = len(features)
        n_meta = len(all_meta_labels)
        cell_h = 1.05   # inches per row
        cell_w = 1.6    # inches per column
        fig_h  = n_feat * cell_h + 1.8
        fig_w  = n_meta * cell_w + 2.2

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(log_p, aspect='auto', cmap='YlOrRd',
                       vmin=0, vmax=vmax_lp, origin='upper')

        # Divider between categorical and continuous metadata
        ax.axvline(len(meta_cat) - 0.5, color='white', lw=2.5, ls='--')
        ax.text(len(meta_cat) - 0.5, -0.7, '← categorical  |  continuous →',
                ha='center', va='top', fontsize=_FS_SMALL, color='#666',
                transform=ax.get_xaxis_transform())

        # Cell annotations: effect size (top line) + p-value (bottom line)
        for fi in range(n_feat):
            for mi in range(n_meta):
                v = p_mat[fi, mi]
                if not np.isfinite(v):
                    continue
                star   = ' *' if sig_mat[fi, mi] else ''
                bright = log_p[fi, mi] > vmax_lp * 0.62
                tc     = 'white' if bright else '#222'
                mlabel = all_meta_labels[mi]

                # Effect size string
                if mlabel in meta_cont:
                    rho = rho_dict.get((fi, mi), np.nan)
                    eff = f'ρ={rho:+.2f}' if np.isfinite(rho) else ''
                elif (fi, mi) in dmed_dict:
                    eff = f'Δmed={dmed_dict[(fi,mi)]:+.2f}'
                else:
                    eta2 = eta2_dict.get((fi, mi), np.nan)
                    eff  = f'η²={eta2:.2f}' if np.isfinite(eta2) else ''

                # Two lines per cell: effect size above centre, p-value below
                ax.text(mi, fi - 0.22, eff,
                        ha='center', va='center', fontsize=_FS_SMALL,
                        fontweight='bold', color=tc)
                ax.text(mi, fi + 0.22, f'p={v:.3f}{star}',
                        ha='center', va='center', fontsize=_FS_SMALL - 1,
                        color=tc, alpha=0.9)

        ax.set_xticks(range(n_meta))
        ax.set_xticklabels(all_meta_labels, rotation=35, ha='right',
                           fontsize=_FS_TICK)
        ax.set_yticks(range(n_feat))
        ax.set_yticklabels(features, fontsize=_FS_TICK)
        ax.set_xlabel('Metadata variable', fontsize=_FS_LABEL, labelpad=14)
        ax.set_ylabel('Waveform feature  (β weight direction tested)',
                      fontsize=_FS_LABEL)
        ax.set_title(
            f'{tl}  —  β × metadata association\n'
            f'Colour = −log₁₀(p raw);  ρ = Spearman (continuous);  '
            f'Δmed = median difference (2-group);  η² = Kruskal-Wallis;  '
            f'* = FDR q<0.05',
            fontsize=_FS_TICK + 1, pad=12
        )

        cbar = plt.colorbar(im, ax=ax, label='−log₁₀(p raw)',
                            shrink=0.5, pad=0.02)
        # Mark p=0.05 threshold on colorbar
        thresh_frac = -np.log10(0.05) / vmax_lp
        cbar.ax.axhline(thresh_frac, color='black', lw=1.5, ls='--')
        cbar.ax.text(1.15, thresh_frac, 'p=0.05', va='center',
                     fontsize=_FS_SMALL, transform=cbar.ax.transAxes)
        cbar.ax.tick_params(labelsize=_FS_SMALL)

        fig.tight_layout()
        plt.show()

        # ── Summary table ──────────────────────────────────────────────────────
        sorted_pairs = sorted(
            [(entry['p_raw'], fi, mi) for (fi, mi), entry in fdr_lookup.items()],
            key=lambda x: x[0]
        )[:n_detail]

        n_fdr = int(sig_mat.sum())
        n_nom = int((p_mat < 0.05).sum())
        print(f'  {tl}:  FDR-sig = {n_fdr}  |  nominally sig (p<0.05) = {n_nom}  '
              f'|  showing top {len(sorted_pairs)} by raw p\n')
        print(f'  {"Feature":<16}  {"Metadata":<14}  {"Effect":>10}  '
              f'{"p-raw":>7}  {"FDR q":>7}  sig')
        print('  ' + '─' * 65)
        for p_v, fi, mi in sorted_pairs:
            mlabel = all_meta_labels[mi]
            feat   = features[fi]
            eff    = (f'ρ = {rho_dict.get((fi,mi), float("nan")):+.3f}'
                      if mlabel in meta_cont else
                      f'Δmed = {dmed_dict[(fi,mi)]:+.3f}'
                      if (fi, mi) in dmed_dict else
                      f'η² = {eta2_dict.get((fi,mi), float("nan")):.3f}')
            entry   = fdr_lookup[(fi, mi)]
            sig_tag = _sig_str(p_v)
            fdr_tag = ' (FDR*)' if entry['sig'] else ''
            print(f'  {feat:<16}  {mlabel:<14}  {eff:>12}  '
                  f'{p_v:>7.4f}  {entry["p_fdr"]:>7.4f}  {sig_tag}{fdr_tag}')
        print()

        # ── Detail plots ───────────────────────────────────────────────────────
        ncols    = min(len(sorted_pairs), 3)
        nrows    = math.ceil(len(sorted_pairs) / ncols)
        fig2, ax2s = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 4.8, nrows * 5.2),   # extra height for bottom labels
            squeeze=False,
        )
        ax2_flat = ax2s.flatten()
        fig2.suptitle(
            f'{tl}  —  top {n_detail} β × metadata trends\n'
            f'Colour:  red = FDR q<0.05  |  amber = raw p<0.05  |  grey = p≥0.05',
            fontsize=_FS_TITLE, y=1.01
        )

        for ax_i, (p_v, fi, mi) in enumerate(sorted_pairs):
            ax2    = ax2_flat[ax_i]
            feat   = features[fi]
            mlabel = all_meta_labels[mi]
            entry  = fdr_lookup[(fi, mi)]
            mcol   = (meta_cat if mlabel in meta_cat else meta_cont)[mlabel]
            col    = (_SIG_COL if entry['sig'] else
                      _NOM_COL if p_v < 0.05 else _NSG_COL)
            sig_s  = _sig_str(p_v)

            # deduplicate columns in case mcol == 'cell_type'
            sel   = list(dict.fromkeys(['cell_id', mcol, 'cell_type']))
            tmp   = df[sel].copy()
            tmp['beta'] = beta_pop[tn][feat]
            dcols = list(dict.fromkeys(['beta', mcol, 'cell_type']))
            d     = tmp[dcols].dropna(subset=['beta', mcol])

            if mlabel in meta_cat:
                order    = sorted(d[mcol].unique())
                pal      = dict(zip(order, _CB[:len(order)]))
                trans_x  = ax2.get_xaxis_transform()

                sns.boxplot(data=d, x=mcol, y='beta', order=order, palette=pal,
                            width=0.5, ax=ax2, fliersize=0,
                            boxprops={'alpha': 0.35},
                            medianprops=dict(color='k', lw=2.5))
                sns.stripplot(data=d, x=mcol, y='beta', order=order, palette=pal,
                              alpha=0.70, jitter=0.18, size=7, ax=ax2, zorder=4)
                ax2.axhline(0, color='gray', lw=1.2, ls='--', alpha=0.7)

                # n + median annotations below each group — outside data area
                for i_g, g in enumerate(order):
                    grp = d[d[mcol] == g]['beta'].values
                    ax2.text(i_g, -0.10,
                             f'n={len(grp)}\nmed={np.median(grp):+.3f}',
                             ha='center', va='top',
                             fontsize=_FS_SMALL, color='#444',
                             transform=trans_x)

                # Stats box in top-right corner (axes fraction, above data)
                eta2  = eta2_dict.get((fi, mi), np.nan)
                dmed  = dmed_dict.get((fi, mi), np.nan)
                parts = []
                if np.isfinite(dmed):
                    parts.append(f'Δmed = {dmed:+.3f}')
                if np.isfinite(eta2):
                    parts.append(f'η² = {eta2:.3f}')
                parts.append(f'p = {p_v:.3f}')
                parts.append(f'FDR q = {entry["p_fdr"]:.3f}')
                ax2.text(0.97, 0.97, '\n'.join(parts),
                         transform=ax2.transAxes, ha='right', va='top',
                         fontsize=_FS_SMALL, color='#333',
                         bbox=dict(boxstyle='round,pad=0.35', fc='white',
                                   ec='#ccc', alpha=0.9))
                ax2.set_xlabel(mlabel, fontsize=_FS_LABEL)
                ax2.set_xticklabels(order, fontsize=_FS_TICK)

            else:
                # Continuous: scatter coloured by cell type + bootstrap CI
                rho    = rho_dict.get((fi, mi), np.nan)
                x_vals = d[mcol].values.astype(float)
                y_vals = d['beta'].values.astype(float)

                for ct in unique_ctypes:
                    mask = d['cell_type'] == ct
                    if mask.sum():
                        ax2.scatter(
                            d.loc[mask, mcol], d.loc[mask, 'beta'],
                            color=ctype_pal[ct], s=65, alpha=0.82,
                            edgecolors='white', lw=0.5, zorder=4, label=ct,
                        )

                if len(d) >= 4:
                    m_fit, b_fit = np.polyfit(x_vals, y_vals, 1)
                    xs = np.linspace(x_vals.min(), x_vals.max(), 100)
                    ax2.plot(xs, m_fit * xs + b_fit, color=col, lw=2.2,
                             ls='-' if p_v < 0.05 else '--', zorder=5)
                    # Bootstrap 95% CI band
                    boot_lines = []
                    for _ in range(500):
                        idx = rng_boot.integers(0, len(x_vals), len(x_vals))
                        xb, yb = x_vals[idx], y_vals[idx]
                        if len(np.unique(xb)) > 1:
                            mb, bb = np.polyfit(xb, yb, 1)
                            boot_lines.append(mb * xs + bb)
                    if len(boot_lines) >= 10:
                        ax2.fill_between(
                            xs,
                            np.percentile(boot_lines, 2.5,  axis=0),
                            np.percentile(boot_lines, 97.5, axis=0),
                            color=col, alpha=0.13, zorder=3,
                        )

                ax2.axhline(0, color='gray', lw=1, ls='--', alpha=0.6)
                ax2.set_xlabel(mlabel, fontsize=_FS_LABEL)
                ax2.legend(fontsize=_FS_SMALL, frameon=False,
                           title='Cell type', title_fontsize=_FS_SMALL,
                           loc='upper left')

                # Stats box top-right
                parts = [
                    f'ρ = {rho:+.3f}',
                    f'p = {p_v:.3f}',
                    f'FDR q = {entry["p_fdr"]:.3f}',
                ]
                ax2.text(0.97, 0.97, '\n'.join(parts),
                         transform=ax2.transAxes, ha='right', va='top',
                         fontsize=_FS_ANNOT, color=col,
                         bbox=dict(boxstyle='round,pad=0.35', fc='white',
                                   ec='#ccc', alpha=0.9))

            # Panel title carries the significance colour
            ax2.set_title(f'{feat}  ×  {mlabel}  [{sig_s}]',
                          fontsize=_FS_TICK + 1, fontweight='bold', color=col, pad=8)
            ax2.set_ylabel(f'β  ({feat},  standardised)', fontsize=_FS_LABEL)
            sns.despine(ax=ax2)

        for i in range(len(sorted_pairs), len(ax2_flat)):
            ax2_flat[i].set_visible(False)

        fig2.tight_layout()
        plt.show()
        print()


# ── 8. Predictor set comparison ───────────────────────────────────────────────

def plot_predictor_set_comparison(df, predictor_sets):
    """
    Compare the number of FDR-significant targets per cell across predictor sets.

    Left panel:  histogram of n_sig per cell for the Waveform-only set.
    Right panel: mean n_sig per predictor set (bar + jitter + SEM error bars),
                 with Wilcoxon paired tests vs Waveform-only annotated.

    Parameters
    ----------
    df             : output of build_metadata_df
    predictor_sets : list[str]  same list used in build_metadata_df
    """
    import seaborn as sns

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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    fig.suptitle(
        'Does adding log-ISI to waveform features improve prediction?\n'
        'Left: distribution of n significant targets per cell (Waveform only);  '
        'Right: mean ± SEM with Wilcoxon paired tests',
        fontsize=_FS_TITLE
    )

    # Left panel: histogram of n_sig for first predictor set
    axes[0].hist(df[first_col], bins=range(0, n_targets + 2),
                 color=_CB[0], alpha=0.85, edgecolor='white', linewidth=0.8)
    axes[0].axvline(df[first_col].mean(), color=_SIG_COL, lw=2, ls='--',
                    label=f'Mean = {df[first_col].mean():.1f}')
    axes[0].set_xlabel('N FDR-significant targets per cell', fontsize=_FS_LABEL)
    axes[0].set_ylabel('N cells', fontsize=_FS_LABEL)
    axes[0].set_title(f'{first_pn}\n(distribution across cells)',
                      fontsize=_FS_TICK + 1)
    axes[0].legend(fontsize=_FS_ANNOT, frameon=False)
    axes[0].set_xticks(range(n_targets + 1))
    axes[0].tick_params(labelsize=_FS_TICK)
    sns.despine(ax=axes[0])

    # Right panel: bar + jitter per predictor set
    rng    = np.random.default_rng(42)
    y_tops = []
    for i, (pn, vals) in enumerate(pset_data.items()):
        bar_col = _CB[i % len(_CB)]
        mean_   = np.mean(vals)
        sem_    = stats.sem(vals)
        axes[1].bar(i, mean_, color=bar_col, alpha=0.6, width=0.5,
                    edgecolor='white')
        axes[1].errorbar(i, mean_, yerr=sem_, fmt='none',
                         color='k', capsize=5, lw=2)
        jitter = rng.uniform(-0.18, 0.18, len(vals))
        axes[1].scatter(i + jitter, vals, color=bar_col, alpha=0.65,
                        s=22, zorder=3)
        y_tops.append(float(np.max(vals)))

    y_top = max(y_tops)
    for i, (pn, vals) in enumerate(list(pset_data.items())[1:], start=1):
        try:
            _, p = stats.wilcoxon(first_vals, vals)
        except ValueError:
            continue
        sig   = _sig_str(p)
        sig_col = _SIG_COL if sig != 'ns' else _NSG_COL
        y_bar = y_top + 0.6 + (i - 1) * 1.0
        axes[1].plot([0, i], [y_bar, y_bar], color=sig_col, lw=1.2)
        axes[1].text((0 + i) / 2, y_bar + 0.12, sig,
                     ha='center', fontsize=11, fontweight='bold', color=sig_col)

    pn_labels = [
        pn.replace(' + ', '\n+ ').replace(' only', '\nonly')
        for pn in pset_data
    ]
    axes[1].set_xticks(range(len(pset_data)))
    axes[1].set_xticklabels(pn_labels, fontsize=_FS_TICK)
    axes[1].set_ylabel('Mean N significant targets per cell\n(± SEM)',
                       fontsize=_FS_LABEL)
    axes[1].set_title(
        f'Predictor set comparison\n(Wilcoxon paired vs {first_pn})',
        fontsize=_FS_TICK + 1
    )
    axes[1].tick_params(labelsize=_FS_TICK)
    sns.despine(ax=axes[1])

    plt.show()
