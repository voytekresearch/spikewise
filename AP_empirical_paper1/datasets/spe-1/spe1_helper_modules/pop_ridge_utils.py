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
    df_tests = run_population_tests(beta_pop, target_names, target_labels, MIN_CELLS, ALPHA)
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


def _fmt_p(p):
    """Format a p-value as plain decimal (no scientific notation)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return 'n/a'
    if p < 0.0001:
        return '< 0.0001'
    return f'{p:.4f}'


def _stars(p):
    """Return significance star string for a p-value."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ''
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'ns'


def _fmt_p_stars(p):
    """Combined formatted p-value and stars, e.g. '0.0234 *'."""
    return f'{_fmt_p(p)} {_stars(p)}'.strip()


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


def merge_hpf_targets(pickle_dir, feat_labels):
    """
    Load both no-HPF and HPF per-cell results and build a clean merged analysis.

    - Keeps only Pre and Post absolute-window targets (removes Pre-BL, Post-BL, Δ).
    - Adds HPF variants of amp and std targets with '_hpf' suffix.
    - Returns a unified all_results dict and matching target_names / target_labels
      ready to pass directly to aggregate_population and downstream functions.

    Target order (14 total):
        Pre LFP Amp, Pre LFP Amp (HPF),
        Pre LFP Std, Pre LFP Std (HPF),
        Pre Gamma AUC, Pre Exponent, Pre Theta AUC,
        Post LFP Amp, Post LFP Amp (HPF),
        Post LFP Std, Post LFP Std (HPF),
        Post Gamma AUC, Post Exponent, Post Theta AUC

    Parameters
    ----------
    pickle_dir  : str  path to ridge_regression_pickles/
    feat_labels : list[str]  e.g. FEAT_LABELS = ['LFP Amp','LFP Std','Gamma AUC','Exponent','Theta AUC']

    Returns
    -------
    all_results    : dict  {cell_id → merged results dict}
    cell_ids       : list[str]
    target_names   : list[str]  14 names
    target_labels  : list[str]  14 display labels
    predictor_sets : list[str]
    """
    import glob

    HPF_AMP_STD = {'pre_lfp_amp', 'pre_lfp_std', 'post_lfp_amp', 'post_lfp_std'}

    def _load(suffix):
        pkls = sorted(glob.glob(os.path.join(pickle_dir, f'c*_ridge_results{suffix}.pkl')))
        pkls = [p for p in pkls if 'population' not in os.path.basename(p)]
        out = {}
        for p in pkls:
            cid = os.path.basename(p).replace(f'_ridge_results{suffix}.pkl', '')
            with open(p, 'rb') as f:
                out[cid] = pickle.load(f)
        return out

    res_raw = _load('')
    res_hpf = _load('_hpf')

    cell_ids = sorted(set(res_raw.keys()) & set(res_hpf.keys()))
    if len(cell_ids) < len(res_raw):
        missing = sorted(set(res_raw.keys()) - set(res_hpf.keys()))
        print(f'⚠  HPF pickles missing for {missing} — using raw only for those cells')

    first          = next(iter(res_raw.values()))
    predictor_sets = list(first[list(first.keys())[0]].keys())

    # Build target name/label list: Pre window first, then Post
    target_names  = []
    target_labels = []
    for window, win_label in [('pre', 'Pre'), ('post', 'Post')]:
        for feat_key, feat_lbl in zip(
            ['lfp_amp', 'lfp_std', 'gamma_auc', 'exponent', 'theta_auc'],
            feat_labels,
        ):
            tn = f'{window}_{feat_key}'
            target_names.append(tn)
            target_labels.append(f'{win_label} {feat_lbl}')
            if feat_key in ('lfp_amp', 'lfp_std'):
                target_names.append(f'{tn}_hpf')
                target_labels.append(f'{win_label} {feat_lbl} (HPF)')

    # Merge into one results dict per cell
    all_results = {}
    for cid in cell_ids:
        merged = {}
        raw_cell = res_raw.get(cid, {})
        hpf_cell = res_hpf.get(cid, {})
        for tn in target_names:
            if tn.endswith('_hpf'):
                base = tn[:-4]
                merged[tn] = hpf_cell.get(base, raw_cell.get(base, {}))
            else:
                merged[tn] = raw_cell.get(tn, {})
        all_results[cid] = merged

    print(f'Merged: {len(all_results)} cells  |  {len(target_names)} targets '
          f'(Pre+Post abs + HPF amp/std, no BL-corrected or Δ)')
    return all_results, cell_ids, target_names, target_labels, predictor_sets


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
                # Use raw permutation p < 0.05 per target independently.
                # sig_corrected was over-corrected across all targets/predictor sets;
                # each target is a separate scientific question.
                p_raw = e.get('p_val', 1.0)
                sig_pop[tn][pn].append(float(p_raw) < 0.05 if p_raw is not None else False)
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

def run_population_tests(beta_pop, target_names, target_labels, min_cells=5, alpha=0.05,
                          sig_r2_targets=None):
    """
    For each (target, waveform feature) pair test whether betas are consistently
    non-zero across cells using two complementary tests:

      t-test              : one-sample, mean beta ≠ 0 (parametric)
      Wilcoxon signed-rank: median beta ≠ 0 (non-parametric; primary test)

    Correction applied within each target (BH across 8 features per target).

    Parameters
    ----------
    sig_r2_targets : list[str] or None
        If provided, only test (target, feature) pairs where the target is in
        this list (i.e. targets that survived the R² > 0 test). This reduces
        the correction burden from n_targets×8 to n_sig×8 pairs, which is
        the scientifically correct hierarchical approach — only ask "which
        features drive prediction?" for targets where prediction was established.
        If None, all targets are tested (fully exploratory mode).

    Returns
    -------
    df_tests : pd.DataFrame
        Columns include p_val / p_val_adj / sig_corrected (t-test)
                        p_wilcox / p_wilcox_adj / sig_wilcox (Wilcoxon — use this)
    """
    test_targets = sig_r2_targets if sig_r2_targets is not None else target_names
    tl_map       = dict(zip(target_names, target_labels))

    if sig_r2_targets is not None:
        print(f'Beta tests restricted to {len(sig_r2_targets)} R²-significant targets '
              f'({len(sig_r2_targets) * len(WAVEFORM_LABELS)} pairs vs '
              f'{len(target_names) * len(WAVEFORM_LABELS)} if all targets used)')

    rows = []
    for tn in test_targets:
        tl = tl_map[tn]
        for wl in WAVEFORM_LABELS:
            valid = beta_pop[tn][wl]
            valid = valid[np.isfinite(valid)]
            if len(valid) < min_cells:
                continue

            # one-sample t-test: is mean beta ≠ 0?
            t_stat, p_ttest = stats.ttest_1samp(valid, popmean=0)

            # Wilcoxon signed-rank: is median beta ≠ 0?
            if len(valid) >= 10 and len(np.unique(valid)) > 1:
                _, p_wilcox = stats.wilcoxon(valid, alternative='two-sided')
            else:
                p_wilcox = np.nan

            rows.append(dict(
                target=tn, target_label=tl, feature=wl,
                n=len(valid),
                mean_beta=float(np.mean(valid)),
                median_beta=float(np.median(valid)),
                sem_beta=float(stats.sem(valid)),
                t_stat=float(t_stat),
                p_val=float(p_ttest),
                p_wilcox=float(p_wilcox) if np.isfinite(p_wilcox) else np.nan,
            ))

    df = pd.DataFrame(rows)
    # No correction — each test assessed at raw alpha independently
    df['sig_ttest']    = df['p_val'].fillna(1.0) < alpha
    df['sig_wilcox']   = df['p_wilcox'].fillna(1.0) < alpha
    df['stars_ttest']  = df['p_val'].apply(_stars)
    df['stars_wilcox'] = df['p_wilcox'].apply(_stars)
    df['p_str']        = df['p_wilcox'].apply(_fmt_p)

    n_sig_t = int(df['sig_ttest'].sum())
    n_sig_w = int(df['sig_wilcox'].sum())
    print(f'Beta consistency: {len(df)} pairs across {df["target"].nunique()} targets  |  '
          f't-test p<{alpha}: {n_sig_t}  |  Wilcoxon p<{alpha}: {n_sig_w}')
    if n_sig_w:
        print(df[df.sig_wilcox][
            ['target_label', 'feature', 'n', 'median_beta', 'p_str', 'stars_wilcox']
        ].to_string(index=False))

    return df


# ── Population tests: fraction significant ────────────────────────────────────

def run_fraction_sig_tests(sig_pop, target_names, target_labels, predictor_sets,
                            min_cells=5, alpha=0.05, chance_p=None):
    """
    Binomial test: is the fraction of significant cells (p<0.05) greater than expected
    by chance for each (target, predictor_set) pair?

    Under H0, each cell has at most `chance_p` probability of being significant
    (defaults to alpha — the cell-level significance threshold). One-sided (greater).

    Returns
    -------
    df_frac : pd.DataFrame
    """
    if chance_p is None:
        chance_p = alpha

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
    # No cross-target correction — each target is an independent question.
    df['sig_binom'] = df['p_binom'] < alpha
    df['stars']     = df['p_binom'].apply(_stars)
    df['p_str']     = df['p_binom'].apply(_fmt_p)

    n_sig = int(df['sig_binom'].sum())
    print(f'Fraction-sig binomial tests: {len(df)} pairs  |  p<{alpha}: {n_sig} '
          f'(each target tested independently)')
    if n_sig:
        out = df[df.sig_binom][
            ['target_label', 'predictor_set', 'n_sig', 'n_cells', 'frac_sig']
        ].copy()
        print(out.to_string(index=False))

    return df


# ── Plots ─────────────────────────────────────────────────────────────────────

_CB = ['#1976D2', '#E53935', '#43A047', '#CC79A7', '#56B4E9', '#E69F00']
_GRP_COLS = {'Pre (abs)': '#1976D2', 'Pre − BL': '#43A047', 'Δ post−pre': '#E53935'}


def _heatmap_dividers(ax):
    pass  # dividers removed — target list is dynamic, no fixed group boundaries


def plot_population_results(r2_pop, sig_pop, beta_pop, df_tests,
                             target_names, target_labels, predictor_sets,
                             df_frac=None, df_r2=None):
    """Mean CV R² heatmap, mean beta heatmap, and beta distribution rainclouds.

    Parameters
    ----------
    df_frac : pd.DataFrame or None
        Output of run_fraction_sig_tests(). If provided, marks cells in the R²
        heatmap with '*' where the binomial fraction-sig test is significant.
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
            # primary significance marker = Wilcoxon; fall back to t-test if unavailable
            if not row.empty:
                sig_beta_mat[t_idx, w_idx] = bool(
                    row.iloc[0].get('sig_wilcox', row.iloc[0]['sig_ttest'])
                )

    # ── Heatmap 1: mean R² ──
    vmax = max(abs(np.nanmax(mean_r2)), abs(np.nanmin(mean_r2)), 0.005)
    fig, ax = plt.subplots(figsize=(max(7, n_p * 3), max(4, n_t * 0.55 + 2)))
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
                 '(bold = mean R²  |  % = fraction sig  |  * = Wilcoxon median R²>0, p<0.05)',
                 fontsize=12, pad=12)
    fig.tight_layout(); plt.show()

    # ── Heatmap 2: one-sample t-statistic per (target × waveform feature) ──
    # Build t-stat matrix from df_tests
    t_stat_mat = np.full((n_t, len(WAVEFORM_LABELS)), np.nan)
    p_val_mat  = np.full((n_t, len(WAVEFORM_LABELS)), np.nan)
    wl_idx     = {wl: i for i, wl in enumerate(WAVEFORM_LABELS)}
    tn_idx     = {tn: i for i, tn in enumerate(target_names)}
    for _, row in df_tests.iterrows():
        r = tn_idx.get(row['target'])
        c = wl_idx.get(row['feature'])
        if r is not None and c is not None:
            t_stat_mat[r, c] = row.get('t_stat', np.nan)
            p_val_mat[r, c]  = row.get('p_val', np.nan)

    vmax2 = max(np.nanmax(np.abs(t_stat_mat)), 2.0)
    fig, ax = plt.subplots(figsize=(max(9, len(WAVEFORM_LABELS) * 1.2), max(4, n_t * 0.55 + 2)))
    im = ax.imshow(t_stat_mat, aspect='auto', vmin=-vmax2, vmax=vmax2, cmap='RdBu_r')
    ax.set_xticks(range(len(WAVEFORM_LABELS)))
    ax.set_xticklabels(WAVEFORM_LABELS, rotation=35, ha='right', fontsize=10)
    ax.set_yticks(range(n_t))
    ax.set_yticklabels(target_labels, fontsize=10)
    for r in range(n_t):
        for c in range(len(WAVEFORM_LABELS)):
            t = t_stat_mat[r, c]
            p = p_val_mat[r, c]
            if np.isfinite(t):
                star = _stars(p) if np.isfinite(p) else ''
                tc   = 'white' if abs(t) > vmax2 * 0.6 else 'black'
                label = f'{t:+.2f}' + (f'\n{star}' if star and star != 'ns' else '')
                ax.text(c, r, label, ha='center', va='center',
                        fontsize=7.5, color=tc,
                        fontweight='bold' if (star and star != 'ns') else 'normal')
    plt.colorbar(im, ax=ax, label='t-statistic (one-sample vs 0)', shrink=0.55, pad=0.02)
    ax.set_title('Population – Beta weight consistency (one-sample t-test, Waveform only)\n'
                 'Colour = t-statistic direction & magnitude  |  stars = p < 0.05',
                 fontsize=12, pad=12)
    fig.tight_layout(); plt.show()

    # ── Rainclouds: significant (target, feature) pairs ──
    sig_pairs = df_tests[df_tests.sig_wilcox][['target','target_label','feature']].values.tolist()
    if not sig_pairs:
        print('No population-level significant pairs.')
        return mean_r2, frac_sig, mean_beta_mat, sig_beta_mat

    rng   = np.random.default_rng(42)
    ncols = min(len(sig_pairs), 4)
    nrows = math.ceil(len(sig_pairs) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 4.2))
    axes = np.array(axes).flatten() if len(sig_pairs) > 1 else [axes]
    fig.suptitle('Population Beta Distributions — significant (target × feature)\n'
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
        p_str = f'p = {row.p_val:.3f}' if row.p_val >= 0.001 else 'p < 0.001'
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
    Pool scatter data (actual vs predicted, z-scored) across cells — cartoony
    style, core (non-HPF) targets only.

    Organised exactly like plot_r2_summary_boxplot: one row per LFP feature
    (LFP Amp, LFP Std, Gamma AUC, Exponent, Theta AUC), Pre-spike scatters on
    the left of a thick dashed divider and Post-spike scatters on the right.

    Reads y_actual/y_pred saved in each cell's pickle — no cell data reloading needed.
    Only includes cells where the Waveform-only model was significant.
    """
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.8, rc={
        'axes.linewidth':    4.0,
        'xtick.major.width': 4.0,
        'ytick.major.width': 4.0,
        'xtick.major.size':  9,
        'ytick.major.size':  9,
        'lines.linewidth':   4.0,
        'figure.titlesize':  28,
    })
    _FS_SM, _FS_AX, _FS_SUB = 22, 26, 30

    feat_order = ['LFP Amp', 'LFP Std', 'Gamma AUC', 'Exponent', 'Theta AUC']
    win_order  = ['Pre', 'Post']

    # core (non-HPF) target name <-> label lookup
    tl_idx = {tl: i for i, tl in enumerate(target_labels) if '(HPF)' not in tl}
    win_feat_tn = {win: {} for win in win_order}
    for win in win_order:
        for feat in feat_order:
            tl = f'{win} {feat}'
            if tl in tl_idx:
                win_feat_tn[win][feat] = target_names[tl_idx[tl]]
    feats = [f for f in feat_order if any(f in win_feat_tn[w] for w in win_order)]

    cmap = plt.cm.get_cmap('tab20', len(cell_ids))

    def _pool(tn):
        y, yp, colors = [], [], []
        for c_idx, cid in enumerate(cell_ids):
            cell_res = all_results.get(cid, {}).get(tn, {}).get('Waveform only', {})
            # Use raw p < 0.05 — consistent with sig_pop (pickles store sig_fdr, not used here)
            p_raw = cell_res.get('p_val', 1.0)
            if p_raw is None or float(p_raw) >= 0.05:
                continue
            y_actual = cell_res.get('y_actual')
            y_pred   = cell_res.get('y_pred')
            if y_actual is None or y_pred is None:
                continue
            y.extend(y_actual.tolist())
            yp.extend(y_pred.tolist())
            colors.extend([cmap(c_idx)] * len(y_actual))
        return np.array(y), np.array(yp), colors

    # each side gets its own little grid (e.g. 5 feats -> 2 rows x 3 cols)
    ncols_blk = min(len(feats), 3)
    nrows_blk = math.ceil(len(feats) / ncols_blk)
    ncols_tot = 2 * ncols_blk

    # insert a narrow spacer column between the two blocks so the divider has
    # room to sit in open space rather than overlapping axis labels/ticks
    spacer_idx = ncols_blk
    width_ratios = [1.0] * ncols_blk + [0.18] + [1.0] * ncols_blk
    fig = plt.figure(figsize=(ncols_tot * 5.0 + 1.0, nrows_blk * 4.8))
    gs = fig.add_gridspec(nrows_blk, ncols_tot + 1, width_ratios=width_ratios,
                          wspace=0.45, hspace=0.55)
    axes = np.empty((nrows_blk, ncols_tot), dtype=object)
    for r in range(nrows_blk):
        for c in range(ncols_tot):
            gs_col = c if c < ncols_blk else c + 1
            axes[r, c] = fig.add_subplot(gs[r, gs_col])

    fig.text(0.5, 1.0,
             'Each colour = one cell   |   z-scored within cell   |   p < 0.05 cells only',
             ha='center', va='top', fontsize=_FS_SM, style='italic', color='black')

    for wi, win in enumerate(win_order):
        col_off = wi * ncols_blk
        for fi, feat in enumerate(feats):
            r, c = divmod(fi, ncols_blk)
            ax = axes[r, col_off + c]
            tn = win_feat_tn[win].get(feat)
            if tn is None:
                ax.set_visible(False)
                continue
            y, yp, colors = _pool(tn)
            if len(y) == 0:
                ax.set_visible(False)
                continue
            r2 = float(np.corrcoef(y, yp)[0, 1] ** 2)
            ax.scatter(y, yp, c=colors, alpha=0.07, s=3, rasterized=True)
            lo, hi = min(y.min(), yp.min()), max(y.max(), yp.max())
            ax.plot([lo, hi], [lo, hi], 'k--', lw=4.5, alpha=0.8)
            ax.set_title(f'{win} — {feat}', fontsize=_FS_SUB, fontweight='bold',
                         color='black', pad=16)
            box = dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2.0)
            ax.text(0.05, 0.88, f'r² = {r2:.3f}', transform=ax.transAxes,
                    fontsize=_FS_AX, fontweight='bold', va='top', color='black', bbox=box)
            ax.text(0.05, 0.74, f'{len(y):,} spikes', transform=ax.transAxes,
                    fontsize=_FS_SM - 6, va='top', color='black', bbox=box)
            ax.set_xlabel('Actual (z-score)', fontsize=_FS_AX, color='black')
            ax.set_ylabel('Predicted (z-score)', fontsize=_FS_AX, color='black')
            ax.tick_params(axis='both', labelsize=_FS_SM, colors='black')
            sns.despine(ax=ax)

        # hide unused slots in this block (e.g. 5 feats in a 2x3 grid -> 1 empty)
        for fi in range(len(feats), nrows_blk * ncols_blk):
            r, c = divmod(fi, ncols_blk)
            axes[r, col_off + c].set_visible(False)

    fig.subplots_adjust(top=0.88)

    # ── thick dashed divider, centred in the spacer column between blocks ──
    fig.canvas.draw()
    pos_l = next(ax for ax in axes[:, ncols_blk - 1] if ax.get_visible()).get_position()
    pos_r = next(ax for ax in axes[:, ncols_blk] if ax.get_visible()).get_position()
    x_div = (pos_l.x1 + pos_r.x0) / 2
    all_pos = [ax.get_position() for ax in axes.flatten() if ax.get_visible()]
    y0, y1 = min(p.y0 for p in all_pos), max(p.y1 for p in all_pos)
    fig.add_artist(plt.Line2D([x_div, x_div], [y0, y1], transform=fig.transFigure,
                              color='black', linestyle='--', linewidth=5.0, alpha=0.7))

    plt.show()


# ── Population tests: R² > 0 ────────────────────────────────────────────────

def run_r2_tests(r2_pop, target_names, target_labels, predictor_sets,
                  min_cells=5, alpha=0.05):
    """
    One-sided Wilcoxon signed-rank test: is median CV R² > 0 across cells?

    Each LFP target represents a distinct scientific question (amplitude,
    variability, gamma power, aperiodic slope, theta power) and is tested
    independently — no cross-target correction is applied.  Each test is
    assessed against alpha directly.

    If multiple predictor sets are passed, they are shown for context but
    significance is evaluated per (target, predictor_set) independently.

    Parameters
    ----------
    alpha : float  significance threshold (default 0.05, uncorrected per target)

    Returns
    -------
    df_r2 : pd.DataFrame
        Columns: target, target_label, predictor_set, n_cells,
                 median_r2, mean_r2, sem_r2, p_wilcox, sig_r2
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
    df['sig_r2'] = df['p_wilcox'].fillna(1.0) < alpha
    df['stars']  = df['p_wilcox'].apply(_stars)
    df['p_str']  = df['p_wilcox'].apply(_fmt_p)

    n_sig = int(df['sig_r2'].sum())
    print(f'R² > 0 Wilcoxon: {len(df)} (target × predictor_set) pairs  |  '
          f'p<{alpha}: {n_sig}  (each target independent, no cross-target correction)')
    if n_sig:
        out = df[df.sig_r2][['target_label', 'predictor_set', 'n_cells',
                              'median_r2', 'mean_r2', 'p_str', 'stars']].copy()
        print(out.to_string(index=False))

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

    Dot colour  : red   = that cell's R² model was significant (from sig_pop)
                  grey  = not significant
    Median ◆    : black = population Wilcoxon significant for that target×feature
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
                 '(red dot = cell R² sig  |  open dot = not sig  |  '
                 'black ◆ = pop. Wilcoxon sig  |  grey ◆ = not sig)',
                 fontsize=11, y=1.02)

    pop_sig_col = 'sig_wilcox' if 'sig_wilcox' in df_tests.columns else 'sig_ttest'

    for ax_idx, (ax, feat) in enumerate(zip(axes, features)):
        is_left_col = (ax_idx % ncols == 0)

        for row_idx, (tn, tl) in enumerate(zip(tnames_s, tlabels_s)):
            vals  = beta_pop[tn][feat]
            finite = np.isfinite(vals)
            valid  = vals[finite]
            if len(valid) == 0:
                continue

            # ── per-cell colour: red = cell R² significant ──────────────
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
    at min_r2_line. Cells with sig_corrected=True shown filled; non-sig shown open.

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
                 '(red filled = cell significant  |  ◆ = mean  |  dashed = R²=0 reference)',
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
    sig_col = 'sig_wilcox' if 'sig_wilcox' in df_tests.columns else 'sig_ttest'
    if df_tests[sig_col].any():
        show_cols = [c for c in ['target_label','feature','n','median_beta','p_wilcox','p_val']
                     if c in df_tests.columns]
        print(df_tests[df_tests[sig_col]][show_cols].to_string(index=False))


# ── Focused beta table ───────────────────────────────────────────────────────

def show_beta_tables(df_tests, sig_r2_targets, target_names, target_labels,
                      r2_pop=None, predictor_set='Waveform only'):
    """
    Display one styled table per R²-significant target showing per-waveform-feature
    one-sample t-test p-values (uncorrected), sorted by raw p-value.

    Parameters
    ----------
    df_tests         : output of run_population_tests (restricted to sig targets)
    sig_r2_targets   : list[str]  target names that survived R² testing
    target_names     : list[str]
    target_labels    : list[str]
    r2_pop           : dict or None  output of aggregate_population; if provided,
                       adds mean and median CV R² column to the table caption
    predictor_set    : str  used to look up R² from r2_pop
    """
    try:
        from IPython.display import display, HTML
    except ImportError:
        display = print
        HTML = lambda x: x

    tl_map = dict(zip(target_names, target_labels))
    for tn in sig_r2_targets:
        tl = tl_map.get(tn, tn)
        sub = (
            df_tests[df_tests['target'] == tn]
            [['feature', 'n', 'mean_beta', 'sem_beta', 'p_wilcox']]
            .sort_values('p_wilcox')
            .reset_index(drop=True)
        )
        if sub.empty:
            continue
        sub['sig'] = sub['p_wilcox'].apply(_stars)
        sub['p_wilcox'] = sub['p_wilcox'].apply(_fmt_p)
        sub.columns = ['Feature', 'N cells', 'Mean β', 'SEM β', 'p (Wilcoxon)', 'sig']

        # Build caption with CV R² summary if available
        r2_info = ''
        if r2_pop is not None and tn in r2_pop and predictor_set in r2_pop[tn]:
            r2_vals = r2_pop[tn][predictor_set]
            r2_vals = r2_vals[np.isfinite(r2_vals)]
            r2_info = (f'  |  mean CV R² = {np.mean(r2_vals):.4f}  '
                       f'median CV R² = {np.median(r2_vals):.4f}')

        styled = (
            sub.style
            .format({'Mean β': '{:+.4f}', 'SEM β': '{:.4f}', 't': '{:+.3f}'})
            .set_caption(f'{tl} — one-sample t-test on β  (n={sub["N cells"].iloc[0]} cells, uncorrected){r2_info}')
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '13px'), ('font-weight', 'bold'),
                                          ('text-align', 'left')]}])
        )
        display(styled)
        display(HTML('<br/>'))


# ── R² summary bar chart ─────────────────────────────────────────────────────

def plot_r2_summary(r2_pop, df_r2, target_names, target_labels,
                    predictor_set='Waveform only'):
    """
    Bar chart: mean ± SEM CV R² per target with Wilcoxon significance stars.

    Parameters
    ----------
    r2_pop        : output of aggregate_population
    df_r2         : output of run_r2_tests
    target_names  : list[str]
    target_labels : list[str]
    predictor_set : str
    """
    import seaborn as sns

    means, sems, pvals, sig_flags = [], [], [], []
    for tn in target_names:
        vals  = r2_pop[tn][predictor_set]
        valid = vals[np.isfinite(vals)]
        means.append(float(np.mean(valid)) if len(valid) else np.nan)
        sems.append(float(stats.sem(valid)) if len(valid) > 1 else np.nan)
        row = df_r2[(df_r2['target'] == tn) & (df_r2['predictor_set'] == predictor_set)]
        if len(row):
            pvals.append(float(row.iloc[0]['p_wilcox']))
            sig_flags.append(bool(row.iloc[0]['sig_r2']))
        else:
            pvals.append(np.nan)
            sig_flags.append(False)

    means = np.array(means)
    sems  = np.array(sems)
    x     = np.arange(len(target_names))

    feat_cols = {
        'lfp_amp':   '#0072B2',
        'lfp_std':   '#56B4E9',
        'gamma_auc': '#009E73',
        'exponent':  '#E69F00',
        'theta_auc': '#CC79A7',
    }
    bar_cols = []
    for tn in target_names:
        for k, c in feat_cols.items():
            if k in tn:
                bar_cols.append(c)
                break
        else:
            bar_cols.append('#888')

    fig, ax = plt.subplots(figsize=(max(12, len(target_names) * 0.85), 5.5))

    y_top  = float(np.nanmax(means + np.where(np.isfinite(sems), sems, 0)))
    y_star = y_top * 1.12   # fixed height above all bars — no overlap with ticks

    ax.bar(x, means, yerr=sems, color=bar_cols, alpha=0.75, width=0.7,
           error_kw=dict(lw=1.5, capsize=4, capthick=1.5, ecolor='#333'),
           edgecolor='white')
    ax.set_ylim(top=y_star * 1.15)   # make room for stars above

    for i, (p, sig) in enumerate(zip(pvals, sig_flags)):
        s = _stars(p)
        if s != 'ns' and sig:
            ax.text(i, y_star, s, ha='center', va='bottom', fontsize=13,
                    color='#D55E00', fontweight='bold')

    ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(target_labels, rotation=40, ha='right', fontsize=9)
    fig.subplots_adjust(bottom=0.28)   # extra space for rotated labels
    ax.set_ylabel('Mean CV R²  (± SEM,  n=37 cells)', fontsize=11)
    ax.set_title(
        f'Population CV R²  —  {predictor_set}\n'
        f'Stars = Wilcoxon p < 0.05 (median R² > 0);  colours = LFP feature type',
        fontsize=11
    )
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.75,
                              label=k.replace('_', ' ').replace('auc', 'AUC'))
               for k, c in feat_cols.items()]
    ax.legend(handles=handles, fontsize=8, frameon=False,
              loc='upper right', ncol=len(feat_cols))
    sns.despine(ax=ax)
    fig.tight_layout()
    plt.show()


def plot_r2_summary_boxplot(r2_pop, sig_pop, df_r2, target_names, target_labels,
                             predictor_set='Waveform only',
                             control_predictor_set='Log ISI only'):
    """
    Cartoony boxplot of per-cell CV R² for the core LFP-feature targets only
    (Pre/Post × LFP Amp, LFP Std, Gamma AUC, Exponent, Theta AUC — no HPF,
    no baseline-corrected, no Δ variants).

    Single plot, split by a vertical dashed divider: Pre-spike targets on the
    left, Post-spike targets on the right. Within each side, boxes are grouped
    by LFP feature and coloured by model — Waveform model vs. a Log-ISI-only
    control (showing whether waveform shape carries information beyond simple
    spike timing).

    Diamonds mark the population mean CV R² (the boxplot line itself is the
    median — they differ because the per-cell R² distribution is right-skewed).
    Each pair of boxes is annotated above with the fraction of cells
    individually significant and a star marking population-level significance
    (Wilcoxon median R² > 0, from df_r2).

    Parameters
    ----------
    r2_pop, sig_pop        : output of aggregate_population
    df_r2                  : output of run_r2_tests
    target_names, target_labels : list[str]
    predictor_set          : str  the main model
    control_predictor_set  : str  the control comparison
    """
    import seaborn as sns

    sns.set_theme(style='ticks', font_scale=1.8, rc={
        'axes.linewidth':    3.0,
        'xtick.major.width': 3.0,
        'ytick.major.width': 3.0,
        'xtick.major.size':  8,
        'ytick.major.size':  8,
        'patch.linewidth':   3.0,
        'lines.linewidth':   3.0,
        'figure.titlesize':  24,
    })
    _FS_SM, _FS_AX, _FS_SUB = 18, 22, 26

    feat_order   = ['LFP Amp', 'LFP Std', 'Gamma AUC', 'Exponent', 'Theta AUC']
    win_order    = ['Pre', 'Post']
    model_order  = [predictor_set, control_predictor_set]
    model_colors = {predictor_set: '#1976D2', control_predictor_set: '#9E9E9E'}
    model_labels = {predictor_set: 'Waveform model', control_predictor_set: 'Log-ISI control'}

    tl_idx = {tl: i for i, tl in enumerate(target_labels)}

    # feature -> target name, per window (only keep features present in both windows)
    win_feat_tn = {win: {} for win in win_order}
    for win in win_order:
        for feat in feat_order:
            tl = f'{win} {feat}'
            if tl in tl_idx:
                win_feat_tn[win][feat] = target_names[tl_idx[tl]]
    feats = [f for f in feat_order if all(f in win_feat_tn[w] for w in win_order)]

    # ── build one long-form df per side, and a shared whisker-based y-range ──
    side_dfs = []
    whisk_hi, whisk_lo = [], []
    for win in win_order:
        rows = []
        for feat in feats:
            tn = win_feat_tn[win][feat]
            for pn in model_order:
                vals  = np.asarray(r2_pop[tn][pn], dtype=float)
                valid = vals[np.isfinite(vals)]
                rows.extend(dict(feat=feat, model=pn, r2=v) for v in valid)
                if len(valid):
                    q1, q3 = np.percentile(valid, [25, 75])
                    iqr = q3 - q1
                    whisk_hi.append(valid[valid <= q3 + 1.5 * iqr].max())
                    whisk_lo.append(valid[valid >= q1 - 1.5 * iqr].min())
        side_dfs.append(pd.DataFrame(rows))

    y_top = max(whisk_hi)
    y_bot = min(0.0, min(whisk_lo))
    pad   = (y_top - y_bot) * 0.08
    ylim  = (y_bot - pad, y_top + pad)

    box_w  = 0.65
    offset = box_w / 4   # seaborn's dodge offset for 2 hue levels at this width

    fig, axes = plt.subplots(1, 2, figsize=(max(15, len(feats) * 4.4), 7.5),
                             sharey=True, gridspec_kw={'wspace': 0.05})

    for ax, win, df_plot in zip(axes, win_order, side_dfs):
        sns.boxplot(data=df_plot, x='feat', y='r2', hue='model', order=feats,
                    hue_order=model_order,
                    palette=[model_colors[m] for m in model_order],
                    width=box_w, dodge=True, fliersize=0, ax=ax,
                    showmeans=True,
                    meanprops=dict(marker='D', markerfacecolor='white',
                                   markeredgecolor='black', markeredgewidth=2.0,
                                   markersize=9, zorder=5),
                    boxprops=dict(linewidth=3.0, alpha=0.85),
                    medianprops=dict(linewidth=3.0, color='black'),
                    whiskerprops=dict(linewidth=3.0),
                    capprops=dict(linewidth=3.0))
        ax.get_legend().remove()
        ax.axhline(0, color='gray', linestyle='--', linewidth=2.5, alpha=0.6)
        ax.set_ylim(*ylim)

        # annotations at a fixed height *relative to the axes* — never overlap boxes
        trans = ax.get_xaxis_transform()
        for fi, feat in enumerate(feats):
            tn = win_feat_tn[win][feat]
            for hi, pn in enumerate(model_order):
                sig_arr  = np.asarray(sig_pop[tn][pn], dtype=bool)
                frac_sig = float(sig_arr.mean()) if len(sig_arr) else np.nan
                row = df_r2[(df_r2['target'] == tn) & (df_r2['predictor_set'] == pn)]
                star = row.iloc[0]['stars'] if len(row) and row.iloc[0]['sig_r2'] else ''
                x = fi + (-offset if hi == 0 else offset)
                ax.text(x, 1.02, f'{frac_sig:.0%}{star}',
                        ha='center', va='bottom', fontsize=_FS_SM - 4, fontweight='bold',
                        color='black', transform=trans, clip_on=False)

        ax.set_xlabel('')
        ax.set_ylabel('CV R²' if ax is axes[0] else '', fontsize=_FS_AX, color='black')
        ax.tick_params(axis='x', labelsize=_FS_SM, colors='black')
        ax.tick_params(axis='y', labelsize=_FS_SM, colors='black', left=(ax is axes[0]))
        sns.despine(ax=ax, left=(ax is axes[1]), right=(ax is axes[0]))

    fig.text(0.5, 1.04,
             '% = fraction of cells individually significant   |   '
             '* = population-level significant (median R² > 0)   |   '
             '◇ mean,  — median',
             ha='center', fontsize=_FS_SM, style='italic', color='black')

    handles = [plt.Rectangle((0, 0), 1, 1, color=model_colors[m], alpha=0.85,
                             label=model_labels[m]) for m in model_order]
    fig.legend(handles=handles, fontsize=_FS_SM, frameon=False, labelcolor='black',
               loc='upper right', bbox_to_anchor=(0.99, 1.10))

    fig.tight_layout(rect=[0, 0, 0.88, 0.92])

    # ── dashed divider between the two panels, placed using final (post-layout)
    # axes positions ──
    fig.canvas.draw()
    pos_l, pos_r = axes[0].get_position(), axes[1].get_position()
    x_div = (pos_l.x1 + pos_r.x0) / 2
    y0, y1 = min(pos_l.y0, pos_r.y0), max(pos_l.y1, pos_r.y1)
    fig.add_artist(plt.Line2D([x_div, x_div], [y0, y1], transform=fig.transFigure,
                              color='black', linestyle='--', linewidth=4.0, alpha=0.6))

    plt.show()
    return fig, axes


# ── Target correlation check ─────────────────────────────────────────────────

def plot_target_correlations(r2_pop, target_names, target_labels, predictor_set='Waveform only'):
    """
    Pairwise Spearman correlation between per-cell R² vectors for all targets.

    If two targets have highly correlated R² profiles across cells (i.e., the same
    cells tend to succeed or fail on both), they may be capturing the same underlying
    dimension of LFP state and one could be excluded without loss of information.

    Parameters
    ----------
    r2_pop        : output of aggregate_population
    target_names  : list[str]
    target_labels : list[str]
    predictor_set : str  which predictor set to use (default 'Waveform only')
    """
    import seaborn as sns

    # Build matrix: rows = cells, columns = targets
    mat = np.column_stack([
        r2_pop[tn][predictor_set] for tn in target_names
    ])  # shape (n_cells, n_targets)

    n_t = len(target_names)
    rho_mat = np.full((n_t, n_t), np.nan)
    p_mat   = np.full((n_t, n_t), np.nan)

    for i in range(n_t):
        for j in range(n_t):
            xi = mat[:, i]; yi = mat[:, j]
            mask = np.isfinite(xi) & np.isfinite(yi)
            if mask.sum() >= 5:
                rho, p = stats.spearmanr(xi[mask], yi[mask])
                rho_mat[i, j] = rho
                p_mat[i, j]   = p

    # Short labels
    short_labels = [tl.replace('Pre ', 'Pre\n').replace('Post ', 'Post\n') for tl in target_labels]

    fig, ax = plt.subplots(figsize=(max(10, n_t * 0.7), max(8, n_t * 0.65)))
    im = ax.imshow(rho_mat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='auto')

    for i in range(n_t):
        for j in range(n_t):
            if not np.isfinite(rho_mat[i, j]):
                continue
            star = _stars(p_mat[i, j]) if i != j else ''
            bright = abs(rho_mat[i, j]) > 0.5
            tc = 'white' if bright else '#222'
            ax.text(j, i, f'{rho_mat[i,j]:+.2f}\n{star}',
                    ha='center', va='center', fontsize=7, color=tc)

    ax.set_xticks(range(n_t))
    ax.set_yticks(range(n_t))
    ax.set_xticklabels(short_labels, fontsize=7, rotation=40, ha='right')
    ax.set_yticklabels(short_labels, fontsize=7)
    ax.set_title(
        f'Pairwise Spearman ρ between per-cell CV R² vectors  '
        f'({predictor_set})\n'
        f'High |ρ| = same cells succeed/fail on both targets → possible redundancy\n'
        f'* p<0.05  ** p<0.01  *** p<0.001',
        fontsize=10
    )
    cb = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.02)
    cb.set_label('Spearman ρ', fontsize=9)
    fig.tight_layout()
    plt.show()

    # Flag highly correlated pairs
    threshold = 0.7
    high_corr = []
    for i in range(n_t):
        for j in range(i + 1, n_t):
            if np.isfinite(rho_mat[i, j]) and abs(rho_mat[i, j]) >= threshold:
                high_corr.append((target_labels[i], target_labels[j], rho_mat[i, j], p_mat[i, j]))
    if high_corr:
        print(f'\nHighly correlated pairs (|ρ| ≥ {threshold}):')
        for a, b, r, p in sorted(high_corr, key=lambda x: -abs(x[2])):
            print(f'  {a:35s}  ×  {b:35s}  ρ = {r:+.3f}  p = {_fmt_p(p)}')
    else:
        print(f'No pairs with |ρ| ≥ {threshold}.')


# ── HPF vs no-HPF comparison ─────────────────────────────────────────────────

def compare_hpf_versions(pickle_dir, cell_ids, target_names, target_labels,
                          predictor_sets, min_cells=5, alpha=0.05):
    """
    Load both no-HPF (c{N}_ridge_results.pkl) and HPF (c{N}_ridge_results_hpf.pkl)
    per-cell results, run R² population tests on each, and produce a side-by-side
    comparison table + bar chart for the four LFP amplitude/std targets.

    Parameters
    ----------
    pickle_dir    : str  path to ridge_regression_pickles/
    cell_ids      : list[str]
    target_names  : list[str]
    target_labels : list[str]
    predictor_sets: list[str]
    min_cells     : int
    alpha         : float

    Returns
    -------
    df_r2_raw, df_r2_hpf : pd.DataFrame  R² test results for both versions
    """
    import glob
    import seaborn as sns

    def _load(suffix):
        pkls = sorted(glob.glob(os.path.join(pickle_dir, f'c*_ridge_results{suffix}.pkl')))
        pkls = [p for p in pkls if 'population' not in os.path.basename(p)]
        out = {}
        for p in pkls:
            cid = os.path.basename(p).replace(f'_ridge_results{suffix}.pkl', '')
            with open(p, 'rb') as f:
                out[cid] = pickle.load(f)
        return out

    res_raw = _load('')
    res_hpf = _load('_hpf')

    missing_raw = set(cell_ids) - set(res_raw.keys())
    missing_hpf = set(cell_ids) - set(res_hpf.keys())
    if missing_raw or missing_hpf:
        print(f'⚠  Missing no-HPF pickles: {sorted(missing_raw)}')
        print(f'⚠  Missing HPF pickles:    {sorted(missing_hpf)}')
        return None, None

    print(f'No-HPF: {len(res_raw)} cells  |  HPF 0.1 Hz: {len(res_hpf)} cells')

    r2_raw, _, _ = aggregate_population(res_raw, cell_ids, target_names, predictor_sets)
    r2_hpf, _, _ = aggregate_population(res_hpf, cell_ids, target_names, predictor_sets)

    df_r2_raw = run_r2_tests(r2_raw, target_names, target_labels, predictor_sets,
                              min_cells=min_cells, alpha=alpha)
    df_r2_hpf = run_r2_tests(r2_hpf, target_names, target_labels, predictor_sets,
                              min_cells=min_cells, alpha=alpha)

    # ── Comparison table ──────────────────────────────────────────────────────
    AMP_STD = [('pre_lfp_amp',  'Pre LFP Amp'),
               ('pre_lfp_std',  'Pre LFP Std'),
               ('post_lfp_amp', 'Post LFP Amp'),
               ('post_lfp_std', 'Post LFP Std')]

    rows = []
    for tn, tl in AMP_STD:
        raw_row = df_r2_raw[(df_r2_raw['target'] == tn) &
                            (df_r2_raw['predictor_set'] == 'Waveform only')]
        hpf_row = df_r2_hpf[(df_r2_hpf['target'] == tn) &
                            (df_r2_hpf['predictor_set'] == 'Waveform only')]
        if not len(raw_row) or not len(hpf_row):
            continue
        r, h = raw_row.iloc[0], hpf_row.iloc[0]
        ratio = float(h['median_r2'] / r['median_r2']) if r['median_r2'] > 0 else float('nan')
        for label, row in [('No HPF', r), ('HPF 0.1 Hz', h)]:
            rows.append(dict(
                Target=tl, Version=label, N=int(row['n_cells']),
                median_r2=row['median_r2'], mean_r2=row['mean_r2'],
                p=_fmt_p(row['p_wilcox']),
                sig=_stars(row['p_wilcox']),
                **({'HPF/raw': ratio} if label == 'HPF 0.1 Hz' else {}),
            ))

    df_cmp = pd.DataFrame(rows)

    try:
        from IPython.display import display
        styled = (
            df_cmp.style
            .format({'median_r2': '{:.4f}', 'mean_r2': '{:.4f}', 'HPF/raw': '{:.2f}'})
            .set_caption(
                'R² comparison: no HPF vs HPF 0.1 Hz  |  Waveform-only predictor set\n'
                'HPF/raw < 1 = signal attenuated by detrending;  sig: * p<0.05  ** p<0.01  *** p<0.001'
            )
            .set_table_styles([{'selector': 'caption',
                                'props': [('font-size', '12px'), ('font-weight', 'bold'),
                                          ('text-align', 'left'), ('white-space', 'pre-line')]}])
        )
        display(styled)
    except Exception:
        print(df_cmp.to_string(index=False))

    # ── Bar chart ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    x, width = range(len(AMP_STD)), 0.35
    palette = {'No HPF': '#0072B2', 'HPF 0.1 Hz': '#D55E00'}

    for i, (label, col) in enumerate(palette.items()):
        vals = [df_cmp[(df_cmp['Target'] == tl) & (df_cmp['Version'] == label)]['median_r2'].values[0]
                for _, tl in AMP_STD]
        sigs = [df_cmp[(df_cmp['Target'] == tl) & (df_cmp['Version'] == label)]['sig'].values[0]
                for _, tl in AMP_STD]
        bars = ax.bar([xi + i * width for xi in x], vals, width,
                      label=label, color=col, alpha=0.75, edgecolor='white')
        for b, s in zip(bars, sigs):
            if s == '✓':
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.0003,
                        '*', ha='center', va='bottom', fontsize=14, color=col)

    ax.set_xticks([xi + width / 2 for xi in x])
    ax.set_xticklabels([tl for _, tl in AMP_STD], fontsize=10)
    ax.set_ylabel('Median CV R²  (Waveform only)', fontsize=11)
    ax.set_title('Effect of 0.1 Hz HPF detrending on LFP amplitude/std targets\n'
                 '* = p<0.05  |  bars show median R² across 37 cells', fontsize=11)
    ax.legend(fontsize=10, frameon=False)
    ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.5)
    sns.despine(ax=ax)
    fig.tight_layout()
    plt.show()

    # ── Summary ───────────────────────────────────────────────────────────────
    sig_raw = df_r2_raw[df_r2_raw['sig_r2'] & (df_r2_raw['predictor_set'] == 'Waveform only')]['target'].tolist()
    sig_hpf = df_r2_hpf[df_r2_hpf['sig_r2'] & (df_r2_hpf['predictor_set'] == 'Waveform only')]['target'].tolist()
    print(f'\nNo-HPF significant (Waveform only):    {sig_raw}')
    print(f'HPF 0.1 Hz significant (Waveform only): {sig_hpf}')
    print(f'Surviving HPF: {sorted(set(sig_raw) & set(sig_hpf))}')
    print(f'Lost to HPF:   {sorted(set(sig_raw) - set(sig_hpf))}')

    # Also return aggregated HPF data so the notebook can run the full plot suite
    r2_hpf_full, sig_hpf_full, beta_hpf_full = aggregate_population(
        res_hpf, cell_ids, target_names, predictor_sets
    )
    sig_hpf_targets = df_r2_hpf[
        df_r2_hpf['sig_r2'] & (df_r2_hpf['predictor_set'] == 'Waveform only')
    ]['target'].tolist()
    df_tests_hpf = run_population_tests(
        beta_hpf_full, target_names, target_labels,
        min_cells=min_cells, alpha=alpha,
        sig_r2_targets=sig_hpf_targets if sig_hpf_targets else None,
    )

    return dict(
        df_r2_raw=df_r2_raw, df_r2_hpf=df_r2_hpf,
        r2_hpf=r2_hpf_full, sig_hpf=sig_hpf_full, beta_hpf=beta_hpf_full,
        df_tests_hpf=df_tests_hpf, sig_r2_targets_hpf=sig_hpf_targets,
        res_hpf=res_hpf,
    )
