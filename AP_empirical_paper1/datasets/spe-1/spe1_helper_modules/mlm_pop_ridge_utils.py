"""
bayes_pop_ridge_utils.py
------------------------
Random-effects meta-regression for population-level ridge results.

For each (target × waveform-feature) pair the 37 cell beta weights are modelled as:
    β_i = μ + X_meta_i @ γ + ε_i

Estimation: OLS with HC3 heteroskedasticity-robust standard errors.
FDR correction (BH) applied separately across all 200 models for:
  - the population intercept μ
  - each metadata slope γ

HC3 robust SEs are preferred over bootstrap here because:
  - n=37 makes bootstrap CIs noisy (high Monte Carlo variance)
  - HC3 directly corrects for heteroskedasticity without resampling

Metadata predictors (standardised / binary):
    is_IN        : 1 = interneuron, 0 = PC
    is_WC        : 1 = whole-cell, 0 = juxta
    cort_depth_z : z-scored cortical depth
    dark_neuron  : 1 = dark neuron
    eap_visible  : 1 = EAP visible on Neuropixels
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.multitest import fdrcorrection

warnings.filterwarnings('ignore')

from config import (
    DICT_CELL_TYPE, DICT_PATCH_TYPE, DICT_CORT_DEPTH,
    DICT_DARK_NEURONS, DICT_EAP_WAV,
)
from ridge_regression_utils import WAVEFORM_LABELS

META_NAMES  = ['is_IN', 'is_WC', 'is_IC', 'cort_depth_z', 'dark_neuron', 'eap_visible']
META_LABELS = ['Cell type\n(IN vs PC)', 'Patch type\n(WC vs Juxta)', 'Current type\n(IC vs VC)',
               'Cort. depth (z)', 'Dark neuron', 'EAP visible']


# ── Metadata ──────────────────────────────────────────────────────────────────

def build_meta_df(cell_ids):
    """Return a DataFrame of standardised metadata for each cell."""
    rows = []
    for cid in cell_ids:
        n  = int(cid.replace('c', ''))
        ct = DICT_CELL_TYPE.get(n, 'PC')
        pt = DICT_PATCH_TYPE.get(n, 'Juxta, VC')
        rows.append(dict(
            cell_id     = cid,
            is_IN       = int(ct == 'IN'),
            is_WC       = int('WC' in pt),
            is_IC       = int('IC' in pt),
            cort_depth  = DICT_CORT_DEPTH.get(n, np.nan),
            dark_neuron = int(DICT_DARK_NEURONS.get(n, False)),
            eap_visible = int(DICT_EAP_WAV.get(n, False)),
        ))
    df = pd.DataFrame(rows)
    mu, sd = df['cort_depth'].mean(), df['cort_depth'].std()
    df['cort_depth_z'] = (df['cort_depth'] - mu) / sd
    return df[['cell_id', 'is_IN', 'is_WC', 'is_IC', 'cort_depth_z', 'dark_neuron', 'eap_visible']]


# ── Single model ──────────────────────────────────────────────────────────────

def run_meta_regression(betas, meta_mat, ci=0.94):
    """
    OLS + HC3 robust SEs for one (target, feature) pair.

    Parameters
    ----------
    betas    : (n_cells,) array of mean beta weights
    meta_mat : (n_cells, n_meta) design matrix
    ci       : confidence/credible interval width (default 0.94)

    Returns
    -------
    dict with intercept and metadata slope estimates, CIs, and p-values
    """
    valid = np.isfinite(betas)
    b     = betas[valid].astype(float)
    Xm    = meta_mat[valid].astype(float)
    n     = len(b)

    if n < 5:
        return None

    X      = sm.add_constant(Xm, has_constant='add')
    result = sm.OLS(b, X).fit(cov_type='HC3')

    alpha  = 1 - ci
    ci_arr = np.asarray(result.conf_int(alpha=alpha))   # (n_params, 2)
    params = np.asarray(result.params)
    pvals  = np.asarray(result.pvalues)

    return dict(
        mu_mean    = float(params[0]),
        mu_ci_lo   = float(ci_arr[0, 0]),
        mu_ci_hi   = float(ci_arr[0, 1]),
        mu_pval    = float(pvals[0]),
        gamma_mean = params[1:].tolist(),
        gamma_ci   = ci_arr[1:].tolist(),    # (n_meta, 2)
        gamma_pval = pvals[1:].tolist(),
        n_cells    = int(n),
    )


# ── Run all pairs + FDR ───────────────────────────────────────────────────────

def run_all_meta_regressions(beta_pop, meta_df, target_names, target_labels,
                              ci=0.94, fdr_q=0.05,
                              save_path=None, force_recompute=False):
    """
    Fit one model per (target × waveform feature) pair, then FDR-correct.

    FDR is applied separately to:
      - all 200 intercept p-values
      - all 200 p-values for each metadata variable (5 × 200 = 1000 tests)

    Returns
    -------
    df : pd.DataFrame, one row per (target, feature)
    """
    from tqdm import tqdm

    if save_path and not force_recompute and os.path.exists(save_path):
        with open(save_path, 'rb') as f:
            df = pickle.load(f)
        print(f'Loaded cached results: {os.path.basename(save_path)}')
        return df

    meta_mat = meta_df[META_NAMES].values.astype(float)
    pairs    = [(tn, tl, wl)
                for tn, tl in zip(target_names, target_labels)
                for wl in WAVEFORM_LABELS]

    rows = []
    for tn, tl, wl in tqdm(pairs, desc='Meta-regression (OLS + HC3)'):
        b   = np.array(beta_pop[tn][wl], dtype=float)
        res = run_meta_regression(b, meta_mat, ci=ci)
        if res is None:
            continue

        row = dict(target=tn, target_label=tl, feature=wl,
                   n_cells=res['n_cells'],
                   mu_mean=res['mu_mean'],
                   mu_ci_lo=res['mu_ci_lo'],
                   mu_ci_hi=res['mu_ci_hi'],
                   mu_pval=res['mu_pval'])

        for k, mn in enumerate(META_NAMES):
            row[f'gamma_{mn}']      = res['gamma_mean'][k]
            row[f'gamma_{mn}_lo']   = res['gamma_ci'][k][0]
            row[f'gamma_{mn}_hi']   = res['gamma_ci'][k][1]
            row[f'gamma_{mn}_pval'] = res['gamma_pval'][k]

        rows.append(row)

    df = pd.DataFrame(rows)

    # FDR correction — intercept
    _, df['mu_pval_fdr'] = fdrcorrection(df['mu_pval'].values, alpha=fdr_q)
    df['mu_sig_fdr']     = df['mu_pval_fdr'] < fdr_q

    # FDR correction — each metadata variable
    for mn in META_NAMES:
        _, df[f'gamma_{mn}_pval_fdr'] = fdrcorrection(
            df[f'gamma_{mn}_pval'].values, alpha=fdr_q)
        df[f'gamma_{mn}_sig_fdr'] = df[f'gamma_{mn}_pval_fdr'] < fdr_q

    # Summary
    n_mu_sig = df['mu_sig_fdr'].sum()
    print(f'\nPopulation intercept: {n_mu_sig} / {len(df)} FDR-significant')
    for mn, ml in zip(META_NAMES, META_LABELS):
        n = df[f'gamma_{mn}_sig_fdr'].sum()
        print(f'  {ml:20s}: {n} / {len(df)} FDR-significant')

    if n_mu_sig:
        print('\nFDR-significant intercepts:')
        sig = df[df['mu_sig_fdr']]
        print(sig[['target_label', 'feature', 'mu_mean',
                   'mu_ci_lo', 'mu_ci_hi', 'mu_pval_fdr']].to_string(index=False))

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(df, f)
        print(f'\nSaved: {os.path.basename(save_path)}')

    return df


# ── Helpers ───────────────────────────────────────────────────────────────────

def _credible(lo, hi):
    return (lo > 0) or (hi < 0)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_population_effects(df_bayes, target_labels):
    """
    Forest plot: population intercept μ for every (target × feature) pair.
    Filled diamond = FDR-significant.  Red = positive, Blue = negative.
    """
    n_feat     = len(WAVEFORM_LABELS)
    targ_order = list(dict.fromkeys(df_bayes['target_label'].tolist()))
    n_targ     = len(targ_order)

    fig, axes = plt.subplots(1, n_feat,
                             figsize=(n_feat * 2.2, n_targ * 0.38 + 1.5),
                             sharey=True)
    fig.suptitle('Population intercept μ  (94% CI, HC3 robust)\n'
                 'Filled diamond = FDR-significant  |  open circle = not significant',
                 fontsize=12, y=1.01)

    for ax, wl in zip(axes, WAVEFORM_LABELS):
        sub = (df_bayes[df_bayes['feature'] == wl]
               .set_index('target_label')
               .reindex(targ_order))

        for yi, tl in enumerate(targ_order):
            if tl not in sub.index or pd.isna(sub.loc[tl, 'mu_mean']):
                continue
            mu     = sub.loc[tl, 'mu_mean']
            lo     = sub.loc[tl, 'mu_ci_lo']
            hi     = sub.loc[tl, 'mu_ci_hi']
            fdr_sig = bool(sub.loc[tl, 'mu_sig_fdr'])
            col    = '#E53935' if (mu > 0 and fdr_sig) else \
                     '#1976D2' if (mu < 0 and fdr_sig) else '#bbb'
            ax.plot([lo, hi], [yi, yi], lw=1.5, color=col, alpha=0.7)
            ax.scatter([mu], [yi], s=22, color=col, zorder=4,
                       marker='D' if fdr_sig else 'o')

        ax.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        ax.set_title(wl, fontsize=8, fontweight='bold')
        ax.set_yticks(range(n_targ))
        if ax is axes[0]:
            ax.set_yticklabels(targ_order, fontsize=7.5)
        ax.tick_params(axis='x', labelsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        for d in [4.5, 9.5, 14.5, 19.5]:
            ax.axhline(d, color='#ddd', lw=1, ls='--')

    fig.tight_layout()
    plt.show()


def plot_metadata_effects(df_bayes, target_labels):
    """
    One heatmap per metadata variable: γ mean, ★ = FDR-significant.
    """
    targ_order = list(dict.fromkeys(df_bayes['target_label'].tolist()))
    n_targ     = len(targ_order)
    n_feat     = len(WAVEFORM_LABELS)

    fig, axes = plt.subplots(1, len(META_NAMES),
                             figsize=(len(META_NAMES) * 2.8, n_targ * 0.38 + 2),
                             sharey=True)
    fig.suptitle('Metadata effects on beta weights γ  (94% CI, HC3 robust)\n'
                 '★ = FDR-significant', fontsize=12, y=1.01)

    for ax, mn, ml in zip(axes, META_NAMES, META_LABELS):
        mat     = np.full((n_targ, n_feat), np.nan)
        sig_mat = np.zeros((n_targ, n_feat), dtype=bool)

        for ti, tl in enumerate(targ_order):
            for fi, wl in enumerate(WAVEFORM_LABELS):
                row = df_bayes[(df_bayes['target_label'] == tl) &
                               (df_bayes['feature'] == wl)]
                if row.empty:
                    continue
                mat[ti, fi]     = row[f'gamma_{mn}'].values[0]
                sig_mat[ti, fi] = bool(row[f'gamma_{mn}_sig_fdr'].values[0])

        vmax = max(abs(np.nanmax(mat)), abs(np.nanmin(mat)), 0.01)
        im   = ax.imshow(mat, aspect='auto', vmin=-vmax, vmax=vmax,
                         cmap='RdBu_r', origin='upper')
        plt.colorbar(im, ax=ax, shrink=0.4, pad=0.02)

        for ti in range(n_targ):
            for fi in range(n_feat):
                if sig_mat[ti, fi]:
                    ax.text(fi, ti, '★', ha='center', va='center',
                            fontsize=7, color='k')

        ax.set_xticks(range(n_feat))
        ax.set_xticklabels(WAVEFORM_LABELS, rotation=40, ha='right', fontsize=7)
        ax.set_yticks(range(n_targ))
        if ax is axes[0]:
            ax.set_yticklabels(targ_order, fontsize=7.5)
        ax.set_title(ml, fontsize=9, fontweight='bold')
        for d in [4.5, 9.5, 14.5, 19.5]:
            ax.axhline(d, color='white', lw=1.5, ls='--')

    fig.tight_layout()
    plt.show()


def plot_credible_pairs(df_bayes):
    """Bar chart for FDR-significant population intercepts."""
    sig = df_bayes[df_bayes['mu_sig_fdr']].copy()

    if sig.empty:
        print('No FDR-significant population effects.')
        # Show top 10 by raw p-value as a guide
        top = df_bayes.nsmallest(10, 'mu_pval')[
            ['target_label', 'feature', 'mu_mean', 'mu_ci_lo', 'mu_ci_hi',
             'mu_pval', 'mu_pval_fdr']]
        print('\nTop 10 by raw p-value (none survived FDR):')
        print(top.to_string(index=False))
        return

    ncols = min(len(sig), 4)
    nrows = int(np.ceil(len(sig) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = np.array(axes).flatten() if len(sig) > 1 else [axes]
    fig.suptitle('FDR-significant population effects', fontsize=12, y=1.01)

    for ax, (_, row) in zip(axes, sig.iterrows()):
        mu, lo, hi = row['mu_mean'], row['mu_ci_lo'], row['mu_ci_hi']
        col = '#E53935' if mu > 0 else '#1976D2'
        ax.barh([0], [mu], color=col, alpha=0.6, height=0.4)
        ax.errorbar([mu], [0], xerr=[[mu - lo], [hi - mu]],
                    fmt='none', color='#333', capsize=5, lw=2)
        ax.axvline(0, color='k', lw=1, ls='--', alpha=0.6)
        ax.set_yticks([])
        ax.set_xlabel('μ (std units)', fontsize=9)
        ax.set_title(f'{row["target_label"]}\n{row["feature"]}',
                     fontsize=9, fontweight='bold')
        ax.text(0.5, -0.25,
                f'p_fdr = {row["mu_pval_fdr"]:.3f}  |  n = {row["n_cells"]}',
                transform=ax.transAxes, ha='center', fontsize=8, color='#555')
        ax.spines[['top', 'right']].set_visible(False)

    for i in range(len(sig), len(axes)):
        axes[i].set_visible(False)

    fig.tight_layout()
    plt.show()
