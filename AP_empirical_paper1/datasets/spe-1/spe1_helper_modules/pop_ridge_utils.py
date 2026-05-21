"""
pop_ridge_utils.py
------------------
Population-level analysis functions for the spike-waveform → LFP ridge regression.

Typical notebook usage
----------------------
    from pop_ridge_utils import (
        load_population_results, aggregate_population,
        run_population_tests, plot_population_results,
        build_population_scatter, save_population_results,
    )

    all_results, cell_ids, target_names, predictor_sets = load_population_results(RIDGE_PICKLE_DIR)
    r2_pop, sig_pop, beta_pop = aggregate_population(all_results, cell_ids, target_names, predictor_sets)
    df_tests = run_population_tests(beta_pop, target_names, target_labels, MIN_CELLS, FDR_Q)
    plot_population_results(all_results, cell_ids, r2_pop, sig_pop, beta_pop,
                            df_tests, target_names, target_labels, predictor_sets)
    build_population_scatter(all_results, cell_ids, target_names, target_labels,
                             PRE_WIN, POST_WIN, BASELINE_WIN)
    save_population_results(RIDGE_PICKLE_DIR, r2_pop, sig_pop, beta_pop, df_tests,
                            target_names, target_labels, predictor_sets, cell_ids)
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm

from ridge_regression_utils import (
    load_cell_data, build_ridge_matrices, get_predictions,
    WAVEFORM_LABELS, WAVEFORM_COLS, FEAT_LABELS,
)


# ── Load ──────────────────────────────────────────────────────────────────────

def load_population_results(pickle_dir):
    """
    Load all per-cell ridge result pickles.

    Returns
    -------
    all_results  : dict  {cell_id → results_dict}
    cell_ids     : list[str]
    target_names : list[str]
    predictor_sets : list[str]
    """
    files = sorted(f for f in os.listdir(pickle_dir)
                   if f.startswith('c') and f.endswith('_ridge_results.pkl'))
    cell_ids = [f.replace('_ridge_results.pkl', '') for f in files]

    all_results = {}
    for fname, cid in zip(files, cell_ids):
        with open(os.path.join(pickle_dir, fname), 'rb') as f:
            all_results[cid] = pickle.load(f)

    first          = next(iter(all_results.values()))
    target_names   = list(first.keys())
    predictor_sets = list(first[target_names[0]].keys())

    print(f'Loaded {len(all_results)} cells  |  {len(target_names)} targets  |  {len(predictor_sets)} predictor sets')
    return all_results, cell_ids, target_names, predictor_sets


# ── Aggregate ─────────────────────────────────────────────────────────────────

def aggregate_population(all_results, cell_ids, target_names, predictor_sets):
    """
    Collect CV R², significance flags, and waveform beta weights across cells.

    Returns
    -------
    r2_pop   : {target → {pset → ndarray(n_cells)}}
    sig_pop  : {target → {pset → bool ndarray(n_cells)}}
    beta_pop : {target → {feature_label → ndarray(n_cells)}}
    """
    r2_pop   = {tn: {pn: [] for pn in predictor_sets} for tn in target_names}
    sig_pop  = {tn: {pn: [] for pn in predictor_sets} for tn in target_names}
    beta_pop = {tn: {wl: [] for wl in WAVEFORM_LABELS} for tn in target_names}

    for cid in cell_ids:
        res = all_results[cid]
        for tn in target_names:
            for pn in predictor_sets:
                e = res.get(tn, {}).get(pn, {})
                r2_pop[tn][pn].append(e.get('r2_cv', np.nan))
                sig_pop[tn][pn].append(e.get('sig_fdr', False))
            beta = res.get(tn, {}).get('Waveform only', {}).get('beta') or {}
            for wl in WAVEFORM_LABELS:
                beta_pop[tn][wl].append(beta.get(wl, np.nan))

    for tn in target_names:
        for pn in predictor_sets:
            r2_pop[tn][pn]  = np.array(r2_pop[tn][pn],  dtype=float)
            sig_pop[tn][pn] = np.array(sig_pop[tn][pn], dtype=bool)
        for wl in WAVEFORM_LABELS:
            beta_pop[tn][wl] = np.array(beta_pop[tn][wl], dtype=float)

    return r2_pop, sig_pop, beta_pop


# ── Population t-tests ────────────────────────────────────────────────────────

def run_population_tests(beta_pop, target_names, target_labels, min_cells=5, fdr_q=0.05):
    """
    One-sample t-test (beta ≠ 0) for each (target, waveform feature) pair.
    FDR-corrected across all tests.

    Returns
    -------
    df_tests : pd.DataFrame
    """
    rows = []
    for tn, tl in zip(target_names, target_labels):
        for wl in WAVEFORM_LABELS:
            valid = beta_pop[tn][wl]
            valid = valid[np.isfinite(valid)]
            if len(valid) < min_cells:
                continue
            t_stat, p_val = stats.ttest_1samp(valid, popmean=0)
            rows.append(dict(
                target=tn, target_label=tl, feature=wl,
                n=len(valid),
                mean_beta=float(np.mean(valid)),
                sem_beta=float(stats.sem(valid)),
                t_stat=float(t_stat),
                p_val=float(p_val),
            ))

    df = pd.DataFrame(rows)
    rejected, p_fdr = fdrcorrection(df['p_val'].values, alpha=fdr_q, method='indep')
    df['p_val_fdr'] = p_fdr
    df['sig_fdr']   = rejected

    print(f'Population tests: {len(df)} total  |  '
          f'raw p<0.05: {(df.p_val < 0.05).sum()}  |  '
          f'FDR q<{fdr_q}: {rejected.sum()}')
    if rejected.sum():
        print(df[df.sig_fdr][['target_label','feature','n','mean_beta','sem_beta','p_val_fdr']].to_string(index=False))
    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

_CB = ['#1976D2', '#E53935', '#43A047', '#CC79A7', '#56B4E9', '#E69F00']
_GRP_COLS = {'Pre (abs)': '#1976D2', 'Pre − BL': '#43A047', 'Δ post−pre': '#E53935'}


def _heatmap_dividers(ax):
    for d in [4.5, 9.5, 14.5, 19.5]: ax.axhline(d, color='white', lw=2.5, ls='--')
    for y, lbl in [(2.0,  'Pre\n(abs)'),
                   (7.0,  'Pre\n−BL'),
                   (12.0, 'Post\n(abs)'),
                   (17.0, 'Post\n−BL'),
                   (22.0, 'Δ\npost−pre')]:
        ax.text(-0.9, y, lbl, va='center', ha='right', fontsize=9,
                fontweight='bold', color='#555')


def plot_population_results(r2_pop, sig_pop, beta_pop, df_tests,
                             target_names, target_labels, predictor_sets):
    """Mean CV R² heatmap, mean beta heatmap, and beta distribution rainclouds."""
    import seaborn as sns

    n_t = len(target_names)
    n_p = len(predictor_sets)

    mean_r2  = np.full((n_t, n_p), np.nan)
    frac_sig = np.full((n_t, n_p), np.nan)
    for t_idx, tn in enumerate(target_names):
        for p_idx, pn in enumerate(predictor_sets):
            vals = r2_pop[tn][pn]; valid = vals[np.isfinite(vals)]
            if len(valid):
                mean_r2[t_idx, p_idx]  = np.mean(valid)
                frac_sig[t_idx, p_idx] = np.mean(sig_pop[tn][pn])

    mean_beta_mat = np.full((n_t, len(WAVEFORM_LABELS)), np.nan)
    sig_beta_mat  = np.zeros((n_t, len(WAVEFORM_LABELS)), dtype=bool)
    for t_idx, tn in enumerate(target_names):
        for w_idx, wl in enumerate(WAVEFORM_LABELS):
            valid = beta_pop[tn][wl]; valid = valid[np.isfinite(valid)]
            if len(valid): mean_beta_mat[t_idx, w_idx] = np.mean(valid)
            row = df_tests[(df_tests.target == tn) & (df_tests.feature == wl)]
            if not row.empty: sig_beta_mat[t_idx, w_idx] = bool(row.iloc[0]['sig_fdr'])

    # ── Heatmap 1: mean R² ──
    vmax = max(abs(np.nanmax(mean_r2)), abs(np.nanmin(mean_r2)), 0.005)
    fig, ax = plt.subplots(figsize=(9, 10))
    im = ax.imshow(mean_r2, aspect='auto', vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    ax.set_xticks(range(n_p))
    ax.set_xticklabels(predictor_sets, rotation=20, ha='right', fontsize=11)
    ax.set_yticks(range(n_t))
    ax.set_yticklabels(target_labels, fontsize=10)
    for r in range(n_t):
        for c in range(n_p):
            v, fs = mean_r2[r, c], frac_sig[r, c]
            if np.isfinite(v):
                tc = 'white' if abs(v) > vmax * 0.6 else 'black'
                ax.text(c, r - 0.15, f'{v:+.3f}', ha='center', va='center',
                        fontsize=8, color=tc, fontweight='bold')
                ax.text(c, r + 0.22, f'{fs:.0%} sig.', ha='center', va='center',
                        fontsize=7, color=tc, alpha=0.85)
    _heatmap_dividers(ax)
    plt.colorbar(im, ax=ax, label='Mean CV R²', shrink=0.55, pad=0.02)
    ax.set_title('Population – Mean 5-fold CV R²\n'
                 '(bold = mean R²  |  % = fraction of cells FDR-significant)',
                 fontsize=12, pad=12)
    fig.tight_layout(); plt.show()

    # ── Heatmap 2: mean beta weights ──
    vmax2 = max(abs(np.nanmax(mean_beta_mat)), abs(np.nanmin(mean_beta_mat)), 0.01)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(mean_beta_mat, aspect='auto', vmin=-vmax2, vmax=vmax2, cmap='RdBu_r')
    ax.set_xticks(range(len(WAVEFORM_LABELS)))
    ax.set_xticklabels(WAVEFORM_LABELS, rotation=35, ha='right', fontsize=10)
    ax.set_yticks(range(n_t))
    ax.set_yticklabels(target_labels, fontsize=10)
    for r in range(n_t):
        for c in range(len(WAVEFORM_LABELS)):
            v = mean_beta_mat[r, c]
            if np.isfinite(v):
                star = '*' if sig_beta_mat[r, c] else ''
                tc   = 'white' if abs(v) > vmax2 * 0.6 else 'black'
                ax.text(c, r, f'{v:+.3f}{star}', ha='center', va='center',
                        fontsize=8, color=tc, fontweight='bold' if star else 'normal')
    _heatmap_dividers(ax)
    plt.colorbar(im, ax=ax, label='Mean β across cells', shrink=0.55, pad=0.02)
    ax.set_title('Population – Mean Beta Weights (Waveform only)\n'
                 '(* FDR q<0.05, one-sample t-test vs 0)', fontsize=12, pad=12)
    fig.tight_layout(); plt.show()

    # ── Rainclouds: FDR-significant (target, feature) pairs ──
    sig_pairs = df_tests[df_tests.sig_fdr][['target','target_label','feature']].values.tolist()
    if not sig_pairs:
        print('No population-level FDR-significant pairs.')
        return mean_r2, frac_sig, mean_beta_mat, sig_beta_mat

    rng   = np.random.default_rng(42)
    ncols = min(len(sig_pairs), 4)
    nrows = math.ceil(len(sig_pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4.2))
    axes = np.array(axes).flatten() if len(sig_pairs) > 1 else [axes]
    fig.suptitle('Population Beta Distributions — FDR-significant (target × feature)\n'
                 'One-sample t-test vs 0  |  Waveform only model',
                 fontsize=12, y=1.01)

    for ax, (tn, tl, feat) in zip(axes, sig_pairs):
        valid = beta_pop[tn][feat]; valid = valid[np.isfinite(valid)]
        row   = df_tests[(df_tests.target == tn) & (df_tests.feature == feat)].iloc[0]
        col   = _CB[0]

        # Dots above, box at centre
        ax.scatter(valid, rng.uniform(0.05, 0.35, len(valid)),
                   alpha=0.7, s=40, color=col, zorder=4,
                   edgecolors='white', linewidths=0.4)
        ax.boxplot(valid, vert=False, positions=[0], widths=0.22, patch_artist=True,
                   medianprops=dict(color='k', lw=2.5),
                   boxprops=dict(facecolor=col, alpha=0.35),
                   whiskerprops=dict(color='#333', lw=1.5),
                   capprops=dict(color='#333', lw=1.5),
                   flierprops=dict(marker='o', markersize=3, alpha=0.3))
        ax.scatter([np.mean(valid)], [0], marker='D', color='k',
                   s=50, zorder=6, label=f'mean = {np.mean(valid):+.3f}')
        ax.axvline(0, color='gray', lw=1.2, ls='--', alpha=0.7)
        ax.set_ylim(-0.35, 0.55)
        ax.set_yticks([])
        ax.set_xlabel('β (std units)', fontsize=10)
        ax.set_title(f'{tl}\n{feat}', fontsize=10, fontweight='bold')
        p_str = f'p_fdr = {row.p_val_fdr:.3f}' if row.p_val_fdr >= 0.001 else 'p_fdr < 0.001'
        ax.text(0.5, -0.18,
                f'n = {row.n}  |  t = {row.t_stat:+.2f}  |  {p_str}',
                transform=ax.transAxes, ha='center', fontsize=8, color='#444')
        ax.legend(fontsize=8, frameon=False, loc='upper right')
        sns.despine(ax=ax)

    for i in range(len(sig_pairs), len(axes)): axes[i].set_visible(False)
    fig.tight_layout(); plt.show()

    return mean_r2, frac_sig, mean_beta_mat, sig_beta_mat


def build_population_scatter(all_results, cell_ids, target_names, target_labels,
                              pre_win=None, post_win=None, baseline_win=None):
    """
    Pool scatter data (actual vs predicted, z-scored) across cells.
    Reads y_actual/y_pred saved in each cell's pickle — no cell data reloading needed.
    Only includes cells where the Waveform-only model was FDR-significant.
    """
    import seaborn as sns
    pooled = {tn: {'y': [], 'yp': [], 'colors': []} for tn in target_names}
    cmap   = plt.cm.get_cmap('tab20', len(cell_ids))

    for c_idx, cid in enumerate(cell_ids):
        for tn in target_names:
            cell_res = all_results.get(cid, {}).get(tn, {}).get('Waveform only', {})
            if not cell_res.get('sig_fdr', False):
                continue
            y_actual = cell_res.get('y_actual')
            y_pred   = cell_res.get('y_pred')
            if y_actual is None or y_pred is None:
                continue
            pooled[tn]['y'].extend(y_actual.tolist())
            pooled[tn]['yp'].extend(y_pred.tolist())
            pooled[tn]['colors'].extend([cmap(c_idx)] * len(y_actual))

    sig_targets = [(tn, tl) for tn, tl in zip(target_names, target_labels)
                   if len(pooled[tn]['y']) > 0]
    if not sig_targets:
        print('No significant targets found.')
        return

    ncols = min(len(sig_targets), 4)
    nrows = math.ceil(len(sig_targets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.2, nrows * 4.2))
    axes = np.array(axes).flatten() if len(sig_targets) > 1 else [axes]
    fig.suptitle('Population – Actual vs Predicted (Waveform only)\n'
                 'Each colour = one cell  |  z-scored within cell  |  FDR-significant cells only',
                 fontsize=12, y=1.01)

    for ax, (tn, tl) in zip(axes, sig_targets):
        y   = np.array(pooled[tn]['y'])
        yp  = np.array(pooled[tn]['yp'])
        r2  = float(np.corrcoef(y, yp)[0, 1] ** 2)
        ax.scatter(y, yp, c=pooled[tn]['colors'], alpha=0.07, s=2, rasterized=True)
        lo, hi = min(y.min(), yp.min()), max(y.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.2, alpha=0.8)
        ax.set_xlabel('Actual (z-score)', fontsize=10)
        ax.set_ylabel('Predicted (z-score)', fontsize=10)
        ax.set_title(tl, fontsize=11, fontweight='bold')
        ax.text(0.05, 0.93, f'Pooled r² = {r2:.3f}', transform=ax.transAxes,
                fontsize=9, va='top')
        ax.text(0.05, 0.85, f'{len(y):,} spikes', transform=ax.transAxes,
                fontsize=8, va='top', color='#555')
        sns.despine(ax=ax)

    for i in range(len(sig_targets), len(axes)): axes[i].set_visible(False)
    fig.tight_layout(); plt.show()


# ── Save ──────────────────────────────────────────────────────────────────────

def save_population_results(pickle_dir, r2_pop, sig_pop, beta_pop, df_tests,
                             mean_r2, frac_sig, mean_beta_mat, sig_beta_mat,
                             target_names, target_labels, predictor_sets, cell_ids):
    path = os.path.join(pickle_dir, 'population_ridge_results.pkl')
    payload = dict(
        df_tests=df_tests, r2_pop=r2_pop, sig_pop=sig_pop, beta_pop=beta_pop,
        mean_r2=mean_r2, frac_sig=frac_sig,
        mean_beta_mat=mean_beta_mat, sig_beta_mat=sig_beta_mat,
        target_names=target_names, target_labels=target_labels,
        predictor_sets=predictor_sets, cell_ids=cell_ids,
    )
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    print(f'Saved: {os.path.basename(path)}')
    if df_tests['sig_fdr'].any():
        print(df_tests[df_tests.sig_fdr][['target_label','feature','n','mean_beta','p_val_fdr']].to_string(index=False))
