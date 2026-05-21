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

def plot_population_results(r2_pop, sig_pop, beta_pop, df_tests,
                             target_names, target_labels, predictor_sets):
    """Mean CV R² heatmap, raincloud betas, and mean beta heatmap."""

    n_t = len(target_names)
    n_p = len(predictor_sets)

    # Pre-compute matrices
    mean_r2 = np.full((n_t, n_p), np.nan)
    frac_sig = np.full((n_t, n_p), np.nan)
    for t_idx, tn in enumerate(target_names):
        for p_idx, pn in enumerate(predictor_sets):
            vals  = r2_pop[tn][pn]
            valid = vals[np.isfinite(vals)]
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

    def _dividers(ax):
        for d in [4.5, 9.5]: ax.axhline(d, color='white', lw=2, ls='--')
        ax.text(-0.8,  2.0, 'Pre\n(abs)',   va='center', ha='right', fontsize=8, color='gray')
        ax.text(-0.8,  7.0, 'Pre\n−BL',    va='center', ha='right', fontsize=8, color='gray')
        ax.text(-0.8, 12.0, 'Δ\npost−pre', va='center', ha='right', fontsize=8, color='gray')

    # ── Heatmap 1: mean R² ──
    vmax = max(abs(np.nanmax(mean_r2)), abs(np.nanmin(mean_r2)), 0.005)
    fig, ax = plt.subplots(figsize=(8, 9))
    im = ax.imshow(mean_r2, aspect='auto', vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    ax.set_xticks(range(n_p)); ax.set_xticklabels(predictor_sets, rotation=20, ha='right', fontsize=10)
    ax.set_yticks(range(n_t)); ax.set_yticklabels(target_labels, fontsize=9)
    for r in range(n_t):
        for c in range(n_p):
            v, fs = mean_r2[r, c], frac_sig[r, c]
            if np.isfinite(v):
                tc = 'white' if abs(v) > vmax * 0.6 else 'black'
                ax.text(c, r, f'{v:+.3f}\n({fs:.0%})', ha='center', va='center', fontsize=7, color=tc)
    _dividers(ax)
    plt.colorbar(im, ax=ax, label='Mean CV R²', shrink=0.6)
    ax.set_title('Population – Mean CV R²\n(% = fraction of cells with FDR-significant model)', fontsize=11)
    fig.tight_layout(); plt.show()

    # ── Heatmap 2: mean betas ──
    vmax2 = max(abs(np.nanmax(mean_beta_mat)), abs(np.nanmin(mean_beta_mat)), 0.01)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(mean_beta_mat, aspect='auto', vmin=-vmax2, vmax=vmax2, cmap='RdBu_r')
    ax.set_xticks(range(len(WAVEFORM_LABELS))); ax.set_xticklabels(WAVEFORM_LABELS, rotation=35, ha='right', fontsize=9)
    ax.set_yticks(range(n_t)); ax.set_yticklabels(target_labels, fontsize=9)
    for r in range(n_t):
        for c in range(len(WAVEFORM_LABELS)):
            v = mean_beta_mat[r, c]
            if np.isfinite(v):
                star = '*' if sig_beta_mat[r, c] else ''
                tc   = 'white' if abs(v) > vmax2 * 0.6 else 'black'
                ax.text(c, r, f'{v:+.3f}{star}', ha='center', va='center',
                        fontsize=7, color=tc, fontweight='bold' if star else 'normal')
    _dividers(ax)
    plt.colorbar(im, ax=ax, label='Mean β across cells', shrink=0.6)
    ax.set_title('Population – Mean Beta Weights (Waveform only)\n(* FDR q<0.05, one-sample t-test)', fontsize=11)
    fig.tight_layout(); plt.show()

    # ── Raincloud: FDR-significant pairs ──
    sig_pairs = df_tests[df_tests.sig_fdr][['target','target_label','feature']].values.tolist()
    if not sig_pairs:
        print('No population-level FDR-significant pairs.')
        return mean_r2, frac_sig, mean_beta_mat, sig_beta_mat

    ncols = min(len(sig_pairs), 5)
    nrows = math.ceil(len(sig_pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.5))
    axes = np.array(axes).flatten() if len(sig_pairs) > 1 else [axes]
    fig.suptitle('Population Beta Weights — FDR-significant pairs\nOne-sample t-test vs 0 | Waveform only', fontsize=11)
    rng = np.random.default_rng(42)
    for ax, (tn, tl, feat) in zip(axes, sig_pairs):
        valid = beta_pop[tn][feat]; valid = valid[np.isfinite(valid)]
        row   = df_tests[(df_tests.target == tn) & (df_tests.feature == feat)].iloc[0]
        ax.axvline(0, color='gray', lw=1, ls='--')
        ax.scatter(valid, rng.uniform(-0.15, 0.15, len(valid)), alpha=0.6, s=20, color='#1976D2', zorder=3)
        ax.boxplot(valid, vert=False, positions=[0], widths=0.3, patch_artist=True,
                   medianprops=dict(color='k', lw=2),
                   boxprops=dict(facecolor='#1976D2', alpha=0.3),
                   whiskerprops=dict(color='k'), capprops=dict(color='k'),
                   flierprops=dict(marker='o', markersize=3, alpha=0.4))
        ax.set_yticks([])
        ax.set_xlabel('β (std units)', fontsize=8)
        p_str = f'p_fdr={row.p_val_fdr:.3f}' if row.p_val_fdr >= 0.001 else 'p_fdr<0.001'
        ax.set_title(f'{tl}\n{feat}  n={row.n}  {p_str}', fontsize=8)
    for i in range(len(sig_pairs), len(axes)): axes[i].set_visible(False)
    fig.tight_layout(); plt.show()

    return mean_r2, frac_sig, mean_beta_mat, sig_beta_mat


def build_population_scatter(all_results, cell_ids, target_names, target_labels,
                              pre_win, post_win, baseline_win):
    """
    Load each cell once, compute predictions for all significant targets, pool and plot.
    Colors each cell differently.
    """
    pooled = {tn: {'y': [], 'yp': [], 'colors': []} for tn in target_names}
    cmap   = plt.cm.get_cmap('tab20', len(cell_ids))

    for c_idx, cid in tqdm(list(enumerate(cell_ids)), desc='Loading cells', unit='cell'):
        cell_num = int(cid.replace('c', ''))
        try:
            df_reg, sp, lfp_win = load_cell_data(cell_num)
        except Exception as e:
            print(f'  {cid}: skip ({e})')
            continue

        _, _, _, _, Y_cell, tnames_cell, _ = build_ridge_matrices(
            df_reg, sp, lfp_win, pre_win, post_win, baseline_win)
        X_wv = df_reg[WAVEFORM_COLS].values.astype(float)

        for tn in target_names:
            cell_res   = all_results.get(cid, {}).get(tn, {}).get('Waveform only', {})
            best_alpha = cell_res.get('best_alpha')
            if not cell_res.get('sig_fdr', False) or best_alpha is None or not np.isfinite(best_alpha):
                continue
            if tn not in tnames_cell:
                continue
            y_z, yp_z = get_predictions(Y_cell[:, tnames_cell.index(tn)], X_wv, best_alpha)
            if y_z is None:
                continue
            pooled[tn]['y'].extend(y_z.tolist())
            pooled[tn]['yp'].extend(yp_z.tolist())
            pooled[tn]['colors'].extend([cmap(c_idx)] * len(y_z))

    sig_targets = [(tn, tl) for tn, tl in zip(target_names, target_labels)
                   if len(pooled[tn]['y']) > 0]
    if not sig_targets:
        print('No significant targets found.')
        return

    ncols = min(len(sig_targets), 5)
    nrows = math.ceil(len(sig_targets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = np.array(axes).flatten() if len(sig_targets) > 1 else [axes]
    fig.suptitle('Population – Actual vs Predicted (Waveform only, z-scored within cell)\n'
                 'Pooled across cells with FDR-significant model', fontsize=11)

    for ax, (tn, tl) in zip(axes, sig_targets):
        y   = np.array(pooled[tn]['y'])
        yp  = np.array(pooled[tn]['yp'])
        r2  = float(np.corrcoef(y, yp)[0, 1] ** 2)
        ax.scatter(y, yp, c=pooled[tn]['colors'], alpha=0.08, s=2, rasterized=True)
        lo, hi = min(y.min(), yp.min()), max(y.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlabel('Actual (z)', fontsize=8)
        ax.set_ylabel('Predicted (z)', fontsize=8)
        ax.set_title(f'{tl}\nPooled r²={r2:.3f}  ({len(y):,} spikes)', fontsize=8)

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
