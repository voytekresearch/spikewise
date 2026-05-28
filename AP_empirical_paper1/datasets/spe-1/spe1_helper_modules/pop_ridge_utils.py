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


def _binom_p(k, n, p, alternative='two-sided'):
    """Binomial test wrapper — handles both old (binom_test) and new (binomtest) scipy API."""
    try:
        from scipy.stats import binomtest
        return float(binomtest(k, n, p, alternative=alternative).pvalue)
    except ImportError:
        return float(stats.binom_test(k, n, p, alternative=alternative))

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


# ── Population tests: betas ───────────────────────────────────────────────────

def run_population_tests(beta_pop, target_names, target_labels, min_cells=5, fdr_q=0.05):
    """
    For each (target, waveform feature) pair test whether betas are consistently
    non-zero across cells using two complementary tests:

      t-test          : one-sample, mean beta ≠ 0 (parametric, sensitive to magnitude)
      Wilcoxon signed-rank : median beta ≠ 0 (non-parametric, robust; primary test)

    Both FDR-corrected (BH) across all pairs.

    Returns
    -------
    df_tests : pd.DataFrame
        Columns include p_val / p_val_fdr / sig_fdr (t-test)
                        p_wilcox / p_wilcox_fdr / sig_wilcox (Wilcoxon — use this)
    """
    rows = []
    for tn, tl in zip(target_names, target_labels):
        for wl in WAVEFORM_LABELS:
            valid = beta_pop[tn][wl]
            valid = valid[np.isfinite(valid)]
            if len(valid) < min_cells:
                continue

            # parametric: one-sample t-test, mean ≠ 0
            t_stat, p_ttest = stats.ttest_1samp(valid, popmean=0)

            # non-parametric: Wilcoxon signed-rank, median ≠ 0
            if len(valid) >= 10 and len(np.unique(valid)) > 1:
                w_stat, p_wilcox = stats.wilcoxon(valid, alternative='two-sided')
            else:
                w_stat, p_wilcox = np.nan, np.nan

            rows.append(dict(
                target=tn, target_label=tl, feature=wl,
                n=len(valid),
                mean_beta=float(np.mean(valid)),
                median_beta=float(np.median(valid)),
                sem_beta=float(stats.sem(valid)),
                t_stat=float(t_stat),
                p_val=float(p_ttest),
                w_stat=float(w_stat) if np.isfinite(w_stat) else np.nan,
                p_wilcox=float(p_wilcox) if np.isfinite(p_wilcox) else np.nan,
            ))

    df = pd.DataFrame(rows)

    # FDR on t-test
    rej_t, p_fdr_t = fdrcorrection(df['p_val'].values, alpha=fdr_q, method='indep')
    df['p_val_fdr'] = p_fdr_t
    df['sig_fdr']   = rej_t

    # FDR on Wilcoxon (NaN rows treated as p=1 for correction, then masked back)
    wilcox_raw   = df['p_wilcox'].values.copy()
    finite_mask  = np.isfinite(wilcox_raw)
    wilcox_in    = np.where(finite_mask, wilcox_raw, 1.0)
    rej_w, p_fdr_w = fdrcorrection(wilcox_in, alpha=fdr_q, method='indep')
    rej_w = rej_w & finite_mask
    df['p_wilcox_fdr'] = np.where(finite_mask, p_fdr_w, np.nan)
    df['sig_wilcox']   = rej_w

    print(f'Beta population tests: {len(df)} (target × feature) pairs')
    print(f'  t-test   : raw p<0.05: {(df.p_val < 0.05).sum():3d}  |  FDR q<{fdr_q}: {rej_t.sum()}')
    n_raw_w = int((df['p_wilcox'].fillna(1) < 0.05).sum())
    print(f'  Wilcoxon : raw p<0.05: {n_raw_w:3d}  |  FDR q<{fdr_q}: {rej_w.sum()}')
    if rej_w.sum():
        print(df[df.sig_wilcox][
            ['target_label','feature','n','median_beta','p_wilcox_fdr']
        ].to_string(index=False))

    return df


# ── Population tests: fraction significant ────────────────────────────────────

def run_fraction_sig_tests(sig_pop, target_names, target_labels, predictor_sets,
                            min_cells=5, fdr_q=0.05, chance_p=None):
    """
    Binomial test: is the fraction of FDR-significant cells greater than expected
    by chance for each (target, predictor_set) pair?

    Under H0, each cell has at most `chance_p` probability of being significant
    (defaults to fdr_q — the cell-level FDR threshold). One-sided (greater).

    Returns
    -------
    df_frac : pd.DataFrame
    """
    if chance_p is None:
        chance_p = fdr_q

    rows = []
    for tn, tl in zip(target_names, target_labels):
        for pn in predictor_sets:
            sig_arr = np.array(sig_pop[tn][pn], dtype=bool)
            n_cells = len(sig_arr)
            n_sig   = int(sig_arr.sum())
            if n_cells < min_cells:
                continue
            p_binom = _binom_p(n_sig, n_cells, p=chance_p, alternative='greater')
            rows.append(dict(
                target=tn, target_label=tl, predictor_set=pn,
                n_cells=n_cells, n_sig=n_sig,
                frac_sig=float(n_sig / n_cells),
                p_binom=float(p_binom),
            ))

    df = pd.DataFrame(rows)
    rejected, p_fdr = fdrcorrection(df['p_binom'].values, alpha=fdr_q, method='indep')
    df['p_binom_fdr'] = p_fdr
    df['sig_binom']   = rejected

    n_raw = int((df.p_binom < 0.05).sum())
    n_fdr = int(rejected.sum())
    print(f'Fraction-sig binomial tests: {len(df)} (target × predictor_set) pairs  |  '
          f'raw p<0.05: {n_raw}  |  FDR q<{fdr_q}: {n_fdr}')
    if n_fdr:
        print(df[df.sig_binom][
            ['target_label','predictor_set','n_sig','n_cells','frac_sig','p_binom_fdr']
        ].to_string(index=False))

    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

_CB = ['#1976D2', '#E53935', '#43A047', '#CC79A7', '#56B4E9', '#E69F00']
_GRP_COLS = {'Pre (abs)': '#1976D2', 'Pre − BL': '#43A047', 'Δ post−pre': '#E53935'}


def _heatmap_dividers(ax):
    for d in [4.5, 9.5, 14.5, 19.5]: ax.axhline(d, color='white', lw=2.5, ls='--')


def plot_population_results(r2_pop, sig_pop, beta_pop, df_tests,
                             target_names, target_labels, predictor_sets,
                             df_frac=None, df_r2=None):
    """Mean CV R² heatmap, mean beta heatmap, and beta distribution rainclouds.

    Parameters
    ----------
    df_frac : pd.DataFrame or None
        Output of run_fraction_sig_tests(). If provided, marks cells in the R²
        heatmap with '*' where the binomial fraction-sig test is FDR-significant.
    df_r2 : pd.DataFrame or None
        Output of run_r2_tests(). If provided, its sig_r2 flag overrides df_frac
        for the R² heatmap '*' marker (preferred — more meaningful test).
    """
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
            # primary significance marker = Wilcoxon FDR; fall back to t-test if unavailable
            if not row.empty:
                sig_beta_mat[t_idx, w_idx] = bool(
                    row.iloc[0].get('sig_wilcox', row.iloc[0]['sig_fdr'])
                )

    # ── Heatmap 1: mean R² ──
    vmax = max(abs(np.nanmax(mean_r2)), abs(np.nanmin(mean_r2)), 0.005)
    fig, ax = plt.subplots(figsize=(9, 10))
    im = ax.imshow(mean_r2, aspect='auto', vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    ax.set_xticks(range(n_p))
    ax.set_xticklabels(predictor_sets, rotation=20, ha='right', fontsize=11)
    ax.set_yticks(range(n_t))
    ax.set_yticklabels(target_labels, fontsize=10)
    # build lookup for R² heatmap significance marker
    # prefer df_r2 (Wilcoxon R²>0) over df_frac (binomial fraction-sig)
    binom_sig = {}
    if df_r2 is not None:
        for _, row in df_r2.iterrows():
            binom_sig[(row['target'], row['predictor_set'])] = bool(row['sig_r2'])
    elif df_frac is not None:
        for _, row in df_frac.iterrows():
            binom_sig[(row['target'], row['predictor_set'])] = bool(row['sig_binom'])

    for r in range(n_t):
        for c in range(n_p):
            v, fs = mean_r2[r, c], frac_sig[r, c]
            if np.isfinite(v):
                tc   = 'white' if abs(v) > vmax * 0.6 else 'black'
                star = '*' if binom_sig.get((target_names[r], predictor_sets[c]), False) else ''
                ax.text(c, r - 0.15, f'{v:+.3f}', ha='center', va='center',
                        fontsize=8, color=tc, fontweight='bold')
                ax.text(c, r + 0.22, f'{fs:.0%} sig.{star}', ha='center', va='center',
                        fontsize=7, color=tc, alpha=0.85)
    _heatmap_dividers(ax)
    plt.colorbar(im, ax=ax, label='Mean CV R²', shrink=0.55, pad=0.02)
    ax.set_title('Population – Mean 5-fold CV R²\n'
                 '(bold = mean R²  |  % = fraction sig  |  * = Wilcoxon median R²>0, FDR q<0.05)',
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
                 '(* FDR q<0.05, Wilcoxon signed-rank vs 0)', fontsize=12, pad=12)
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


# ── Population tests: R² > 0 ────────────────────────────────────────────────

def run_r2_tests(r2_pop, target_names, target_labels, predictor_sets,
                  min_cells=5, fdr_q=0.05):
    """
    One-sided Wilcoxon signed-rank test: is median CV R² > 0 across cells?

    This is the most direct test of whether the model actually works at the
    population level. Unlike the binomial fraction-sig test, it requires the
    effect to have a non-trivial effect size (median R² shifted above zero),
    not just that more cells pass their individual FDR threshold.

    FDR-corrected (BH) across all (target × predictor_set) pairs.

    Returns
    -------
    df_r2 : pd.DataFrame
    """
    rows = []
    for tn, tl in zip(target_names, target_labels):
        for pn in predictor_sets:
            vals  = r2_pop[tn][pn]
            valid = vals[np.isfinite(vals)]
            if len(valid) < min_cells:
                continue

            median_r2 = float(np.median(valid))
            mean_r2   = float(np.mean(valid))
            sem_r2    = float(stats.sem(valid))

            # one-sided Wilcoxon: median > 0
            if len(valid) >= 10 and len(np.unique(valid)) > 1:
                w_stat, p_wilcox = stats.wilcoxon(valid, alternative='greater')
            else:
                w_stat, p_wilcox = np.nan, np.nan

            rows.append(dict(
                target=tn, target_label=tl, predictor_set=pn,
                n_cells=len(valid),
                mean_r2=mean_r2, median_r2=median_r2, sem_r2=sem_r2,
                w_stat=float(w_stat) if np.isfinite(w_stat) else np.nan,
                p_wilcox=float(p_wilcox) if np.isfinite(p_wilcox) else np.nan,
            ))

    df = pd.DataFrame(rows)

    # FDR correction
    wilcox_raw  = df['p_wilcox'].values.copy()
    finite_mask = np.isfinite(wilcox_raw)
    wilcox_in   = np.where(finite_mask, wilcox_raw, 1.0)
    rejected, p_fdr = fdrcorrection(wilcox_in, alpha=fdr_q, method='indep')
    rejected = rejected & finite_mask
    df['p_wilcox_fdr'] = np.where(finite_mask, p_fdr, np.nan)
    df['sig_r2']       = rejected

    n_raw = int((df['p_wilcox'].fillna(1) < 0.05).sum())
    n_fdr = int(rejected.sum())
    print(f'R² > 0 Wilcoxon tests: {len(df)} (target × predictor_set) pairs  |  '
          f'raw p<0.05: {n_raw}  |  FDR q<{fdr_q}: {n_fdr}')
    if n_fdr:
        print(df[df.sig_r2][
            ['target_label','predictor_set','n_cells','median_r2','mean_r2','p_wilcox_fdr']
        ].to_string(index=False))

    return df


# ── Beta distribution strip plot ─────────────────────────────────────────────

def plot_beta_distributions(beta_pop, df_tests, target_names, target_labels,
                              sig_pop=None, sig_pset='Waveform only',
                              features=None, sort_by='mean_r2', r2_pop=None):
    """
    One panel per waveform feature: horizontal strip plot of per-cell beta values
    across all targets.

    Shared x- and y-axes across all panels so beta magnitudes and target positions
    are directly comparable. Target order is fixed (same in every panel).

    Dot colour  : red   = that cell's R² model was FDR-significant (from sig_pop)
                  grey  = not significant
    Median ◆    : black = population Wilcoxon FDR-significant for that target×feature
                  grey  = not significant

    Parameters
    ----------
    sig_pop  : output of aggregate_population; used for per-cell R² significance colouring
    sig_pset : predictor set name to read sig_pop from (default 'Waveform only')
    features : list of feature labels to plot (defaults to WAVEFORM_LABELS)
    sort_by  : 'mean_r2'  — sort targets by mean R² descending (requires r2_pop)
               'label'    — alphabetical by target label
               'original' — keep original target_names order
    r2_pop   : output of aggregate_population; required when sort_by='mean_r2'
    """
    import seaborn as sns

    if features is None:
        features = WAVEFORM_LABELS

    # ── fixed sort order (same for every panel) ──────────────────────────────
    if sort_by == 'mean_r2' and r2_pop is not None:
        pset_keys = list(list(r2_pop.values())[0].keys())
        sort_pn   = 'Waveform only' if 'Waveform only' in pset_keys else pset_keys[0]
        means = np.array([np.nanmean(r2_pop[tn][sort_pn]) for tn in target_names])
        order = np.argsort(means)[::-1]
    elif sort_by == 'label':
        order = np.argsort(target_labels)
    else:
        order = np.arange(len(target_names))

    tnames_s  = [target_names[i]  for i in order]
    tlabels_s = [target_labels[i] for i in order]
    n_t       = len(tnames_s)

    rng    = np.random.default_rng(0)
    n_feat = len(features)
    ncols  = min(n_feat, 4)
    nrows  = math.ceil(n_feat / ncols)

    # minimum 0.5 in per target row so labels never overlap
    row_h  = max(8, n_t * 0.5)

    # sharex + sharey so all panels have identical axes
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 5.5, row_h * nrows),
                              sharex=True, sharey=True)
    axes = np.array(axes).flatten()
    fig.suptitle('Per-cell beta weight distributions (Waveform only model)\n'
                 '(red dot = cell R² FDR-sig  |  open dot = not sig  |  '
                 'black ◆ = pop. Wilcoxon FDR-sig  |  grey ◆ = not sig)',
                 fontsize=11, y=1.02)

    pop_sig_col = 'sig_wilcox' if 'sig_wilcox' in df_tests.columns else 'sig_fdr'

    for ax_idx, (ax, feat) in enumerate(zip(axes, features)):
        is_left_col = (ax_idx % ncols == 0)

        for row_idx, (tn, tl) in enumerate(zip(tnames_s, tlabels_s)):
            vals  = beta_pop[tn][feat]
            finite = np.isfinite(vals)
            valid  = vals[finite]
            if len(valid) == 0:
                continue

            # ── per-cell colour: red = cell R² FDR-significant ──────────────
            if sig_pop is not None and sig_pset in sig_pop.get(tn, {}):
                cell_sig = np.array(sig_pop[tn][sig_pset], dtype=bool)[finite]
            else:
                cell_sig = np.zeros(len(valid), dtype=bool)

            jitter = rng.uniform(-0.28, 0.28, len(valid))
            ax.scatter(valid[cell_sig],  row_idx + jitter[cell_sig],
                       color='#D32F2F', alpha=0.7, s=22, zorder=4, linewidths=0)
            ax.scatter(valid[~cell_sig], row_idx + jitter[~cell_sig],
                       facecolors='none', edgecolors='#888',
                       alpha=0.6, s=22, zorder=3, linewidths=0.8)

            # ── median diamond: black = pop Wilcoxon significant ─────────────
            row_t   = df_tests[(df_tests.target == tn) & (df_tests.feature == feat)]
            pop_sig = bool(row_t.iloc[0][pop_sig_col]) if not row_t.empty else False
            d_color = 'k' if pop_sig else '#aaa'
            ax.scatter([np.median(valid)], [row_idx],
                       marker='D', color=d_color, s=45, zorder=6)

        ax.axvline(0, color='k', lw=1.2, ls='--', alpha=0.6)
        ax.set_yticks(range(n_t))
        ax.set_yticklabels(tlabels_s, fontsize=7)
        if not is_left_col:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_xlabel('β (std units)', fontsize=9)
        ax.set_title(feat, fontsize=11, fontweight='bold')
        sns.despine(ax=ax)

    # invert y once — sharey=True means calling per-panel toggles back
    axes[0].invert_yaxis()

    for ax in axes[n_feat:]:
        ax.set_visible(False)

    fig.tight_layout(h_pad=1.5, w_pad=0.5)
    plt.show()


# ── R² distribution strip plot ───────────────────────────────────────────────

def plot_r2_distributions(r2_pop, sig_pop, target_names, target_labels,
                           predictor_sets, sort_by='Waveform only',
                           min_r2_line=0.01):
    """
    For each predictor set: horizontal strip plot of per-cell CV R² values,
    one row per target, sorted by mean R² (descending).

    Each dot = one cell. Mean shown as a diamond. Vertical line at x=0 and
    at min_r2_line. Cells with sig_fdr=True shown filled; non-sig shown open.

    Parameters
    ----------
    sort_by     : predictor set name used to determine row order (default 'Waveform only')
    min_r2_line : draws a vertical reference line at this R² value (practical significance floor)
    """
    import seaborn as sns

    rng   = np.random.default_rng(0)
    n_t   = len(target_names)
    n_p   = len(predictor_sets)

    # ── sort targets by mean R² of sort_by predictor set ──
    sort_pn   = sort_by if sort_by in predictor_sets else predictor_sets[0]
    mean_sort = np.array([np.nanmean(r2_pop[tn][sort_pn]) for tn in target_names])
    order     = np.argsort(mean_sort)[::-1]      # descending
    tnames_sorted = [target_names[i]  for i in order]
    tlabels_sorted = [target_labels[i] for i in order]

    row_h = max(8, n_t * 0.5)
    fig, axes = plt.subplots(1, n_p,
                              figsize=(n_p * 5.5, row_h),
                              sharex=True, sharey=True)
    axes = np.array(axes).flatten() if n_p > 1 else [axes]
    fig.suptitle('Per-cell CV R² distributions\n'
                 '(red filled = cell FDR-significant  |  ◆ = mean  |  dashed = R²=0 reference)',
                 fontsize=12, y=1.02)

    for p_idx, (ax, pn) in enumerate(zip(axes, predictor_sets)):
        for row_idx, (tn, tl) in enumerate(zip(tnames_sorted, tlabels_sorted)):
            vals = r2_pop[tn][pn]
            sigs = sig_pop[tn][pn]
            finite = np.isfinite(vals)
            v = vals[finite]
            s = sigs[finite]

            if len(v) == 0:
                continue

            jitter = rng.uniform(-0.28, 0.28, len(v))
            y_pos  = row_idx + jitter

            ax.scatter(v[s],  y_pos[s],  color='#D32F2F', alpha=0.7,
                       s=22, zorder=4, linewidths=0)
            ax.scatter(v[~s], y_pos[~s], facecolors='none', edgecolors='#888',
                       alpha=0.6, s=22, zorder=3, linewidths=0.8)
            ax.scatter([np.mean(v)], [row_idx], marker='D', color='k',
                       s=40, zorder=6)

        ax.axvline(0, color='k', lw=1.2, ls='--', alpha=0.6)

        ax.set_yticks(range(n_t))
        ax.set_yticklabels(tlabels_sorted, fontsize=7)
        if p_idx > 0:
            plt.setp(ax.get_yticklabels(), visible=False)

        ax.set_xlabel('CV R²', fontsize=10)
        ax.set_title(pn, fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        sns.despine(ax=ax)

    fig.tight_layout(h_pad=1.5, w_pad=0.5)
    plt.show()


# ── Save ──────────────────────────────────────────────────────────────────────

def save_population_results(pickle_dir, r2_pop, sig_pop, beta_pop, df_tests,
                             mean_r2, frac_sig, mean_beta_mat, sig_beta_mat,
                             target_names, target_labels, predictor_sets, cell_ids,
                             df_frac=None, df_r2=None):
    path = os.path.join(pickle_dir, 'population_ridge_results.pkl')
    payload = dict(
        df_tests=df_tests, df_frac=df_frac, df_r2=df_r2,
        r2_pop=r2_pop, sig_pop=sig_pop, beta_pop=beta_pop,
        mean_r2=mean_r2, frac_sig=frac_sig,
        mean_beta_mat=mean_beta_mat, sig_beta_mat=sig_beta_mat,
        target_names=target_names, target_labels=target_labels,
        predictor_sets=predictor_sets, cell_ids=cell_ids,
    )
    with open(path, 'wb') as f:
        pickle.dump(payload, f)
    print(f'Saved: {os.path.basename(path)}')
    sig_col = 'sig_wilcox' if 'sig_wilcox' in df_tests.columns else 'sig_fdr'
    if df_tests[sig_col].any():
        print(df_tests[df_tests[sig_col]][
            ['target_label','feature','n','median_beta','p_wilcox_fdr']
        ].to_string(index=False))
