"""
ridge_regression_utils.py
--------------------------
Helper functions for the per-cell spike-waveform → LFP ridge regression analysis.

Typical notebook usage
----------------------
    from ridge_regression_utils import (
        load_cell_data, build_ridge_matrices,
        run_ridge_regression, apply_fdr, plot_ridge_results,
    )

    df_reg, sp, lfp_win = load_cell_data(cell_num)

    X_wv, X_isi, X_both, wv_labels, Y, tnames, tlabels = build_ridge_matrices(
        df_reg, sp, lfp_win, PRE_WIN, POST_WIN, BASELINE_WIN)

    predictor_sets = {
        'Waveform only':      (X_wv,   wv_labels),
        'Log ISI only':       (X_isi,  ['Log ISI']),
        'Waveform + Log ISI': (X_both, wv_labels + ['Log ISI']),
    }

    results = run_ridge_regression(Y, predictor_sets, tnames, N_PERM, RNG_SEED, ALPHAS)
    results = apply_fdr(results, tnames, predictor_sets)
    plot_ridge_results(cell_num, results, tnames, tlabels, predictor_sets, wv_labels, Y, X_wv)
"""

import os
import math
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import permutation_test_score, KFold
from statsmodels.stats.multitest import fdrcorrection
from tqdm import tqdm


# ── Constants ─────────────────────────────────────────────────────────────────

WAVEFORM_COLS = [
    'ramp_amp', 'inflection_time', 'inflection_amp', 'peak_amp',
    'peak_width', 'peak_sharpness', 'exp_lambda', 'exp_const',
]
WAVEFORM_LABELS = [
    'Ramp Amp', 'Infl. Time', 'Infl. Amp', 'Peak Amp',
    'Peak Width', 'Sharpness', 'Decay λ', 'Decay Const',
]
FEAT_KEYS   = ['lfp_amp', 'lfp_std', 'slow_gamma_auc', 'high_gamma_auc', 'total_gamma_auc', 'exponent', 'theta_auc']
FEAT_LABELS = ['LFP Amp', 'LFP Std', 'Slow γ AUC\n30–60 Hz', 'High γ AUC\n60–80 Hz', 'Total γ AUC\n30–80 Hz', 'Exponent', 'θ AUC\n4–15 Hz']


# ── Data loading ──────────────────────────────────────────────────────────────

def load_cell_data(cell_num):
    """
    Load cluster_df, simple_lfp, and trimmed specparam for one cell.

    Returns
    -------
    df_reg : pd.DataFrame  (aligned to specparam length)
    specparam_by_spike : list of dict
    lfp_windows_by_spike : list of dict

    To also load HPF-detrended LFP windows call load_hpf_lfp_windows(cell_num)
    separately and trim to len(specparam_by_spike), then pass as
    hpf_lfp_by_spike to build_ridge_matrices.
    """
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from config import SPE1_PICKLE_ROOT
    from spk_feat_cluster_analysis import load_chunked_specparam_results, trim_edges

    cluster_df = pickle.load(open(
        os.path.join(SPE1_PICKLE_ROOT, 'cluster_pickles', f'c{cell_num}_cluster_df.pkl'), 'rb'))
    simple_lfp = pickle.load(open(
        os.path.join(SPE1_PICKLE_ROOT, 'simple_lfp_pickles', f'c{cell_num}_simple_lfp.pkl'), 'rb'))

    chunk_dir = os.path.join(SPE1_PICKLE_ROOT, 'multitaper_pickles', f'c{cell_num}')
    sp = load_chunked_specparam_results(save_dir=chunk_dir, prefix=f'c{cell_num}_specparam')
    sp = trim_edges(sp, edge_sec=0.5)

    n = len(sp)
    df_reg = cluster_df.iloc[:n].reset_index(drop=True)
    simple_lfp = simple_lfp[:n]

    print(f'c{cell_num}: {n} spikes loaded')
    return df_reg, sp, simple_lfp


# ── Matrix builder ────────────────────────────────────────────────────────────

def win_avg(t_bins, arr, window):
    """Mean of a 1-D array over a time window (seconds). Returns NaN if no data."""
    if t_bins is None or arr is None:
        return np.nan
    t = np.asarray(t_bins, float)
    a = np.asarray(arr, float)
    mask = (t >= window[0]) & (t <= window[1])
    if not np.any(mask):
        return np.nan
    vals = a[mask]
    return float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan


def win_std(t_bins, arr, window):
    """Std of a 1-D array over a time window (seconds). Returns NaN if no data."""
    if t_bins is None or arr is None:
        return np.nan
    t = np.asarray(t_bins, float)
    a = np.asarray(arr, float)
    mask = (t >= window[0]) & (t <= window[1])
    if not np.any(mask):
        return np.nan
    vals = a[mask]
    return float(np.nanstd(vals)) if np.any(np.isfinite(vals)) else np.nan


def load_hpf_lfp_windows(cell_num, hpf_cutoff=0.1):
    """
    Load raw per-spike LFP windows, apply a zero-phase highpass filter, and
    return a list of dicts compatible with lfp_windows_by_spike but at full
    LFP resolution (not the 25ms-binned simple_lfp).

    Used by build_ridge_matrices when hpf_lfp_by_spike is supplied, so that
    lfp_mean and lfp_std targets are computed from HPF-detrended traces rather
    than the baseline-corrected simple_lfp values.

    Parameters
    ----------
    cell_num   : int
    hpf_cutoff : float  highpass cutoff in Hz (default 1.0)

    Returns
    -------
    list of dict, one per spike:
        {'t_bins_s': ndarray (n_samples,), 'lfp_hpf': ndarray (n_samples,)}
    """
    from config import SPE1_PICKLE_ROOT, LFP_FS
    from signal_utils import butter_highpass

    win_path = os.path.join(
        SPE1_PICKLE_ROOT, 'lfp_window_pickles', f'c{cell_num}_lfp_windows.pkl'
    )
    with open(win_path, 'rb') as f:
        d = pickle.load(f)

    windows      = d['windows']        # list of (n_samples,) arrays
    times_rel_ms = d['times_rel_ms']   # (n_spikes, n_samples) in ms

    result = []
    for raw, t_ms in zip(windows, times_rel_ms):
        hpf = butter_highpass(np.asarray(raw, float), LFP_FS, hpf_cutoff)
        result.append({'t_bins_s': np.asarray(t_ms, float) / 1000.0,
                       'lfp_hpf':  hpf})
    return result


def build_ridge_matrices(df_reg, specparam_by_spike, lfp_windows_by_spike,
                         pre_win, post_win, baseline_win,
                         hpf_lfp_by_spike=None):
    """
    Build predictor (X) and target (Y) matrices.

    Targets (25 total, 5 groups × 5 features):
      Group 0 — pre absolute   : LFP state just before spike
      Group 1 — pre−BL         : pre minus baseline (LFP ramp into spike)
      Group 2 — post absolute  : LFP state just after spike
      Group 3 — post−BL        : post minus baseline (spike-triggered response)
      Group 4 — delta (post−pre): net spike-triggered change

    Returns
    -------
    X_waveform, X_log_isi, X_both, waveform_labels,
    Y, target_names, target_labels
    """
    n = len(specparam_by_spike)

    X_waveform = df_reg[WAVEFORM_COLS].values.astype(float)
    X_log_isi  = df_reg[['log_isi']].values.astype(float)
    X_both     = np.hstack([X_waveform, X_log_isi])

    target_names = (
        [f'pre_{k}'    for k in FEAT_KEYS] +
        [f'prebc_{k}'  for k in FEAT_KEYS] +
        [f'post_{k}'   for k in FEAT_KEYS] +
        [f'postbc_{k}' for k in FEAT_KEYS] +
        [f'delta_{k}'  for k in FEAT_KEYS]
    )
    target_labels = (
        [f'Pre {l}'      for l in FEAT_LABELS] +
        [f'Pre−BL {l}'   for l in FEAT_LABELS] +
        [f'Post {l}'     for l in FEAT_LABELS] +
        [f'Post−BL {l}'  for l in FEAT_LABELS] +
        [f'Δ {l}'        for l in FEAT_LABELS]
    )

    Y = np.full((n, 5 * len(FEAT_KEYS)), np.nan)

    for i in range(n):
        sp        = specparam_by_spike[i]
        sw        = lfp_windows_by_spike[i]
        t_sp      = sp.get('t_bins_s')      if sp else None
        t_sw      = sw['t_bins_s']          if sw is not None else None
        band_aucs = sp.get('band_aucs', {}) if sp else {}

        def _sp(k): return sp.get(k) if sp else None
        def _ba(b): return band_aucs.get(b)

        # HPF-detrended amp/std targets (if HPF data supplied) —
        # use full-resolution raw trace; win_avg → mean, win_std → variability
        if hpf_lfp_by_spike is not None:
            hw     = hpf_lfp_by_spike[i]
            t_hpf  = hw['t_bins_s']
            hpf    = hw['lfp_hpf']
            _amp   = lambda win: win_avg(t_hpf, hpf, win)
            _std   = lambda win: win_std(t_hpf, hpf, win)
        else:
            _amp   = lambda win: win_avg(t_sw, sw['lfp_mean'] if sw else None, win)
            _std   = lambda win: win_avg(t_sw, sw['lfp_std']  if sw else None, win)

        pre_vals = [
            _amp(pre_win),
            _std(pre_win),
            win_avg(t_sp, _ba('gamma'),    pre_win),
            win_avg(t_sp, _sp('exponent'), pre_win),
            win_avg(t_sp, _ba('theta'),    pre_win),
        ]
        bl_vals = [
            _amp(baseline_win),
            _std(baseline_win),
            win_avg(t_sp, _ba('gamma'),    baseline_win),
            win_avg(t_sp, _sp('exponent'), baseline_win),
            win_avg(t_sp, _ba('theta'),    baseline_win),
        ]
        post_vals = [
            _amp(post_win),
            _std(post_win),
            win_avg(t_sp, _ba('gamma'),    post_win),
            win_avg(t_sp, _sp('exponent'), post_win),
            win_avg(t_sp, _ba('theta'),    post_win),
        ]

        Y[i,  0: 5] = pre_vals
        Y[i,  5:10] = [p - b if np.isfinite(p) and np.isfinite(b) else np.nan
                       for p, b in zip(pre_vals, bl_vals)]
        Y[i, 10:15] = post_vals
        Y[i, 15:20] = [po - b if np.isfinite(po) and np.isfinite(b) else np.nan
                       for po, b in zip(post_vals, bl_vals)]
        Y[i, 20:25] = [po - pr if np.isfinite(po) and np.isfinite(pr) else np.nan
                       for po, pr in zip(post_vals, pre_vals)]

    nan_pct = np.isnan(Y).mean(axis=0) * 100
    print('Target NaN %:')
    for tl, pct in zip(target_labels, nan_pct):
        print(f'  {tl:22s}  {pct:.1f}%')

    return X_waveform, X_log_isi, X_both, WAVEFORM_LABELS, Y, target_names, target_labels


# ── Single-PSD alternative (fit Specparam once per pre/post window) ──────────
#
# build_ridge_matrices derives gamma_auc / exponent / theta_auc by averaging an
# already time-resolved sliding-window Specparam fit (win_avg(specparam_by_spike[i],
# ...)) within each pre/post/baseline window. The functions below offer an
# alternative: compute ONE PSD over the raw LFP segment inside each window and
# fit ONE static SpectralModel to it directly — this avoids blending together
# sliding-window epochs that straddle a window boundary, at the cost of a lower-
# resolution (single-spectrum) estimate per window.

def compute_pre_post_psd_features(hpf_lfp_by_spike, fs, pre_win, post_win, baseline_win,
                                   psd_seg_len_s=0.25,
                                   freq_range=(1, 90), n_freqs=256, time_bandwidth=2.0,
                                   band_dict=None, aperiodic_mode="fixed",
                                   peak_width_limits=(4.0, 8.0),
                                   max_n_peaks=4, min_peak_height=0.0, peak_threshold=2.0,
                                   verbose=False):
    """
    For each spike, compute ONE multitaper PSD per pre/baseline/post window and
    fit ONE static SpectralModel to it — returning single-value exponent/
    gamma_auc/theta_auc/r_squared per window (rather than averaging an already
    time-resolved fit within the window).

    The pre/post/baseline analysis windows used in this pipeline (e.g. 50-100 ms)
    are far too short to estimate a usable multitaper PSD across 1-90 Hz: the
    minimum viable spectral bandwidth is ~1/window_length, so a 50 ms window only
    supports ~20 Hz resolution — too coarse to separate theta from gamma or fit a
    credible aperiodic. To keep "one PSD per window" while still getting a usable
    spectral estimate, the PSD is computed over a `psd_seg_len_s`-long segment
    *centered on the analysis window's midpoint* (clipped to the available trace),
    rather than over the literal (typically much shorter) window bounds.

    Parameters
    ----------
    hpf_lfp_by_spike : list of dict, from load_hpf_lfp_windows
        {'t_bins_s': ndarray, 'lfp_hpf': ndarray} — full-resolution per-spike trace.
    fs : float
        Sampling rate of the LFP traces (Hz), e.g. config.LFP_FS.
    pre_win, post_win, baseline_win : (start_s, end_s)
        Analysis windows. Only their midpoints are used to center the PSD segment.
    psd_seg_len_s : float
        Length (s) of the segment the PSD/Specparam fit is computed over, centered
        on each analysis window's midpoint. Default 0.25 s (625 samples @ 2500 Hz)
        gives ~4 Hz spectral resolution with time_bandwidth=2.0 — enough to
        separate theta (4-10 Hz) from gamma (30-55 Hz).
    freq_range, n_freqs, time_bandwidth, band_dict, aperiodic_mode,
    peak_width_limits, max_n_peaks, min_peak_height, peak_threshold, verbose :
        Passed through to mne.time_frequency.psd_array_multitaper / SpectralModel,
        matching the conventions used in compute_lfp_windows / band_aucs (gamma=
        30-55 Hz, theta = 4-10 Hz, log10-power AUC between full and aperiodic fit).

    Returns
    -------
    list of dict, one per spike:
        {'pre': {...}, 'baseline': {...}, 'post': {...}}
    each sub-dict: {'exponent', 'gamma_auc', 'theta_auc', 'r_squared'} (NaN on failure).
    """
    import mne
    from specparam import SpectralModel

    if band_dict is None:
        band_dict = {"theta": (4, 15), "slow_gamma": (30, 60), "high_gamma": (60, 80), "total_gamma": (30, 80)}

    half_len = psd_seg_len_s / 2.0
    min_samples = int(round(fs * psd_seg_len_s * 0.9))   # allow slight edge clipping

    # Multitaper requires normalized half-bandwidth (bandwidth/2)*T >= 0.5,
    # i.e. bandwidth >= 1/T — bump up time_bandwidth if the segment is too
    # short for the requested value (e.g. the 2.0 Hz default needs T >= 0.5 s).
    bandwidth = max(float(time_bandwidth), 1.0 / psd_seg_len_s)

    try:
        from specparam.utils import interpolate_spectrum as _interp_spec
    except ImportError:
        from fooof.utils import interpolate_spectrum as _interp_spec

    _empty = {'exponent': np.nan, 'r_squared': np.nan}
    for _bname in band_dict:
        _empty[f'{_bname}_auc'] = np.nan

    def _fit_window(t_bins, trace, window):
        center = (window[0] + window[1]) / 2.0
        mask = (t_bins >= center - half_len) & (t_bins <= center + half_len)
        seg = trace[mask]
        if seg.size < min_samples:
            return dict(_empty)

        psd, freqs = mne.time_frequency.psd_array_multitaper(
            seg[np.newaxis, :], sfreq=fs, fmin=freq_range[0], fmax=freq_range[1],
            bandwidth=bandwidth, adaptive=False, normalization='full', verbose=False,
        )
        psd    = np.squeeze(psd, axis=0)
        freqs  = np.asarray(freqs, float)
        freqs_sm, psd_sm = _interp_spec(freqs, psd, [58, 62])

        sm = SpectralModel(
            aperiodic_mode=aperiodic_mode,
            peak_width_limits=peak_width_limits, max_n_peaks=max_n_peaks,
            min_peak_height=min_peak_height, peak_threshold=peak_threshold, verbose=verbose,
        )
        try:
            sm.fit(freqs_sm, psd_sm, freq_range=freq_range)
            full_log = np.asarray(sm.get_model(component="full",      space="log"))
            ape_log  = np.asarray(sm.get_model(component="aperiodic", space="log"))
            ff       = np.asarray(sm.freqs)
            out = {
                'exponent':  float(sm.get_params("aperiodic_params", "exponent")),
                'r_squared': float(sm.r_squared_),
            }
        except Exception:
            return dict(_empty)

        for bname, (f_lo, f_hi) in band_dict.items():
            bmask = (ff >= f_lo) & (ff <= f_hi)
            if np.any(bmask):
                diff = np.where(np.isfinite(full_log[bmask] - ape_log[bmask]),
                                full_log[bmask] - ape_log[bmask], 0.0)
                out[f'{bname}_auc'] = float(np.trapz(np.clip(diff, 0, None), ff[bmask]))
            else:
                out[f'{bname}_auc'] = np.nan
        for bname in band_dict:
            out.setdefault(f'{bname}_auc', np.nan)
        return out

    results = []
    for hw in hpf_lfp_by_spike:
        t_bins = np.asarray(hw['t_bins_s'], float)
        trace  = np.asarray(hw['lfp_hpf'], float)
        results.append({
            'pre':      _fit_window(t_bins, trace, pre_win),
            'baseline': _fit_window(t_bins, trace, baseline_win),
            'post':     _fit_window(t_bins, trace, post_win),
        })
    return results


def build_ridge_matrices_single_psd(df_reg, lfp_windows_by_spike, hpf_lfp_by_spike, fs,
                                     pre_win, post_win, baseline_win,
                                     psd_seg_len_s=0.25,
                                     freq_range=(1, 90), n_freqs=256, time_bandwidth=2.0,
                                     band_dict=None, **specparam_kwargs):
    """
    Alternative to build_ridge_matrices: gamma_auc / exponent / theta_auc come
    from compute_pre_post_psd_features (one PSD + one static SpectralModel fit
    per pre/baseline/post window per spike) rather than from averaging an
    already time-resolved sliding-window fit within each window. lfp_amp /
    lfp_std targets are computed identically to build_ridge_matrices, from the
    HPF-detrended trace via win_avg/win_std.

    Returns the same (X_waveform, X_log_isi, X_both, waveform_labels, Y,
    target_names, target_labels) tuple as build_ridge_matrices, so it can be fed
    into run_ridge_regression / plot_ridge_results unchanged — letting the two
    spectral-feature methods be compared side by side.
    """
    n = len(df_reg)
    psd_feats = compute_pre_post_psd_features(
        hpf_lfp_by_spike, fs, pre_win, post_win, baseline_win,
        psd_seg_len_s=psd_seg_len_s,
        freq_range=freq_range, n_freqs=n_freqs, time_bandwidth=time_bandwidth,
        band_dict=band_dict, **specparam_kwargs,
    )

    X_waveform = df_reg[WAVEFORM_COLS].values.astype(float)
    X_log_isi  = df_reg[['log_isi']].values.astype(float)
    X_both     = np.hstack([X_waveform, X_log_isi])

    target_names = (
        [f'pre_{k}'    for k in FEAT_KEYS] +
        [f'prebc_{k}'  for k in FEAT_KEYS] +
        [f'post_{k}'   for k in FEAT_KEYS] +
        [f'postbc_{k}' for k in FEAT_KEYS] +
        [f'delta_{k}'  for k in FEAT_KEYS]
    )
    target_labels = (
        [f'Pre {l}'      for l in FEAT_LABELS] +
        [f'Pre−BL {l}'   for l in FEAT_LABELS] +
        [f'Post {l}'     for l in FEAT_LABELS] +
        [f'Post−BL {l}'  for l in FEAT_LABELS] +
        [f'Δ {l}'        for l in FEAT_LABELS]
    )

    Y = np.full((n, 5 * len(FEAT_KEYS)), np.nan)

    for i in range(n):
        sw    = lfp_windows_by_spike[i]
        t_sw  = sw['t_bins_s'] if sw is not None else None
        pf    = psd_feats[i]
        hw    = hpf_lfp_by_spike[i]
        t_hpf = hw['t_bins_s']
        hpf   = hw['lfp_hpf']
        _amp  = lambda win: win_avg(t_hpf, hpf, win)
        _std  = lambda win: win_std(t_hpf, hpf, win)

        _nf = len(FEAT_KEYS)
        def _pf_vals(win_key):
            pfw = pf[win_key]
            return [_amp(locals()[f'{win_key}_win']), _std(locals()[f'{win_key}_win']),
                    pfw.get('slow_gamma_auc', np.nan), pfw.get('high_gamma_auc', np.nan),
                    pfw.get('total_gamma_auc', np.nan), pfw.get('exponent', np.nan),
                    pfw.get('theta_auc', np.nan)]
        pre_vals  = [_amp(pre_win),      _std(pre_win),
                     pf['pre']['slow_gamma_auc'],      pf['pre']['high_gamma_auc'],
                     pf['pre']['total_gamma_auc'],     pf['pre']['exponent'],      pf['pre']['theta_auc']]
        bl_vals   = [_amp(baseline_win), _std(baseline_win),
                     pf['baseline']['slow_gamma_auc'], pf['baseline']['high_gamma_auc'],
                     pf['baseline']['total_gamma_auc'],pf['baseline']['exponent'], pf['baseline']['theta_auc']]
        post_vals = [_amp(post_win),     _std(post_win),
                     pf['post']['slow_gamma_auc'],     pf['post']['high_gamma_auc'],
                     pf['post']['total_gamma_auc'],    pf['post']['exponent'],     pf['post']['theta_auc']]

        Y[i,       0:  _nf] = pre_vals
        Y[i,   _nf: 2*_nf] = [p - b if np.isfinite(p) and np.isfinite(b) else np.nan
                                for p, b in zip(pre_vals, bl_vals)]
        Y[i, 2*_nf: 3*_nf] = post_vals
        Y[i, 3*_nf: 4*_nf] = [po - b if np.isfinite(po) and np.isfinite(b) else np.nan
                                for po, b in zip(post_vals, bl_vals)]
        Y[i, 4*_nf: 5*_nf] = [po - pr if np.isfinite(po) and np.isfinite(pr) else np.nan
                                for po, pr in zip(post_vals, pre_vals)]

    nan_pct = np.isnan(Y).mean(axis=0) * 100
    print('Target NaN % (single-PSD method):')
    for tl, pct in zip(target_labels, nan_pct):
        print(f'  {tl:22s}  {pct:.1f}%')

    return X_waveform, X_log_isi, X_both, WAVEFORM_LABELS, Y, target_names, target_labels


# ── Regression ────────────────────────────────────────────────────────────────

def run_ridge_regression(Y, predictor_sets, target_names, n_perm, rng_seed, alphas,
                         save_path=None, force_recompute=False):
    """
    For each (target, predictor_set):
      1. Tune alpha via RidgeCV on full data
      2. 5-fold shuffled-CV permutation test with fixed alpha
      3. Store full-data beta weights for visualization

    Parameters
    ----------
    save_path : str or None
        Path to save/load results pickle. If None, no caching.
    force_recompute : bool
        If False and save_path exists, load from cache.

    Returns
    -------
    results : dict  results[tname][pname] = {r2_cv, p_val, best_alpha, beta, n_valid, ...}
    """
    if save_path and not force_recompute and os.path.exists(save_path):
        print(f'  [cache] {os.path.basename(save_path)}')
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    results = {tname: {} for tname in target_names}
    cv_splitter = KFold(n_splits=5, shuffle=True, random_state=rng_seed)
    n_total = len(target_names) * len(predictor_sets)
    pbar = tqdm(total=n_total, desc='ridge CV', unit='model')

    for t_idx, tname in enumerate(target_names):
        y_raw = Y[:, t_idx]

        for p_idx, (pname, (X_raw, plabels)) in enumerate(predictor_sets.items()):
            pbar.set_postfix(target=tname[:12], pred=pname[:10])
            valid   = np.isfinite(X_raw).all(axis=1) & np.isfinite(y_raw)
            n_valid = int(valid.sum())

            if n_valid < 20:
                results[tname][pname] = dict(r2_cv=np.nan, p_val=np.nan,
                                             null_mean=np.nan, null_std=np.nan,
                                             best_alpha=np.nan, beta=None,
                                             n_valid=n_valid, p_val_fdr=np.nan,
                                             sig_fdr=False)
                pbar.update(1)
                continue

            X, y = X_raw[valid], y_raw[valid]
            X_z  = StandardScaler().fit_transform(X)
            y_z  = StandardScaler().fit_transform(y.reshape(-1, 1)).ravel()

            alpha_cv   = RidgeCV(alphas=alphas, fit_intercept=True)
            alpha_cv.fit(X_z, y_z)
            best_alpha = float(alpha_cv.alpha_)

            pipe = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha, fit_intercept=True))
            cv_score, perm_scores, p_val = permutation_test_score(
                pipe, X_z, y_z, cv=cv_splitter,
                n_permutations=n_perm, scoring='r2',
                random_state=rng_seed * 100 + t_idx * 10 + p_idx,
                n_jobs=-1,
            )

            ridge_full = Ridge(alpha=best_alpha, fit_intercept=True)
            ridge_full.fit(X_z, y_z)
            y_pred_z = ridge_full.predict(X_z)

            results[tname][pname] = dict(
                r2_cv=float(cv_score), p_val=float(p_val),
                null_mean=float(np.mean(perm_scores)),
                null_std=float(np.std(perm_scores)),
                best_alpha=best_alpha,
                beta=dict(zip(plabels, ridge_full.coef_)),
                n_valid=n_valid,
                p_val_fdr=np.nan, sig_fdr=False,
                # scatter data: full-data z-scored actual + predicted (for pop scatter)
                y_actual=y_z.astype(np.float32),
                y_pred=y_pred_z.astype(np.float32),
            )
            pbar.update(1)

    pbar.close()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        print(f'  Saved: {os.path.basename(save_path)}')

    return results


# ── Prediction helper (for population scatter) ────────────────────────────────

def get_predictions(Y_col, X_raw, best_alpha):
    """
    Full-data Ridge predictions for one (target, cell) combination.
    Returns (y_actual_z, y_pred_z) — both z-scored within cell — or (None, None).
    """
    valid = np.isfinite(X_raw).all(axis=1) & np.isfinite(Y_col)
    if valid.sum() < 20:
        return None, None
    X, y = X_raw[valid], Y_col[valid]
    X_z      = StandardScaler().fit_transform(X)
    scaler_y = StandardScaler()
    y_z      = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    ridge    = Ridge(alpha=best_alpha, fit_intercept=True)
    ridge.fit(X_z, y_z)
    return y_z, ridge.predict(X_z)


# ── FDR correction ────────────────────────────────────────────────────────────

def apply_fdr(results, target_names, predictor_sets, q=0.05):
    """Apply Benjamini-Hochberg FDR correction across all tests for this cell."""
    test_keys  = [(tn, pn) for tn in target_names for pn in predictor_sets.keys()]
    raw_pvals  = np.array([results[tn][pn].get('p_val', np.nan) for tn, pn in test_keys])
    pvals_in   = np.where(np.isfinite(raw_pvals), raw_pvals, 1.0)
    rejected, pvals_fdr = fdrcorrection(pvals_in, alpha=q, method='indep')

    for (tn, pn), pfdr, rej in zip(test_keys, pvals_fdr, rejected):
        results[tn][pn]['p_val_fdr'] = float(pfdr)
        results[tn][pn]['sig_fdr']   = bool(rej)

    n_raw = int(np.sum(raw_pvals < 0.05))
    n_fdr = int(np.sum(rejected))
    print(f'FDR (BH, q={q}): {n_raw}/{len(test_keys)} raw p<0.05 → {n_fdr}/{len(test_keys)} after correction')

    # Summary table
    pred_names = list(predictor_sets.keys())
    col_w = 22
    header = f'{"Target":<22}' + ''.join(f'{p:>{col_w}}' for p in pred_names)
    print(f'\n{header}')
    print('-' * len(header))
    for tname in target_names:
        def _fmt(pn):
            r = results[tname][pn].get('r2_cv', np.nan)
            if not np.isfinite(r): return 'NaN'
            mk = '*' if results[tname][pn].get('sig_fdr') else \
                 ('~' if results[tname][pn].get('p_val', 1) < 0.05 else ' ')
            ba = results[tname][pn].get('best_alpha', np.nan)
            return f'{r:+.4f}{mk} α={ba:.2g}'
        print(f'{tname:<22}' + ''.join(f'{_fmt(p):>{col_w}}' for p in pred_names))
    print('* FDR q<0.05  ~ raw p<0.05 only')
    return results


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_ridge_results(cell_num, results, target_names, target_labels,
                       predictor_sets, waveform_labels, Y, X_waveform):
    """Heatmap, bar chart, beta weights (waveform only), and scatter plots."""
    pred_names = list(predictor_sets.keys())
    n_t = len(target_names)
    n_p = len(pred_names)

    r2_mat  = np.full((n_t, n_p), np.nan)
    sig_mat = np.zeros((n_t, n_p), dtype=bool)
    for t_idx, tn in enumerate(target_names):
        for p_idx, pn in enumerate(pred_names):
            r2_mat[t_idx, p_idx]  = results[tn][pn].get('r2_cv', np.nan)
            sig_mat[t_idx, p_idx] = results[tn][pn].get('sig_fdr', False)

    vmax = max(abs(np.nanmax(r2_mat)), abs(np.nanmin(r2_mat)), 0.01)

    # ── Heatmap ──
    fig, ax = plt.subplots(figsize=(8, 12))
    im = ax.imshow(r2_mat, aspect='auto', vmin=-vmax, vmax=vmax, cmap='RdBu_r')
    ax.set_xticks(range(n_p)); ax.set_xticklabels(pred_names, rotation=20, ha='right', fontsize=10)
    ax.set_yticks(range(n_t)); ax.set_yticklabels(target_labels, fontsize=9)
    for r in range(n_t):
        for c in range(n_p):
            v = r2_mat[r, c]
            if np.isfinite(v):
                star = '*' if sig_mat[r, c] else ''
                tc   = 'white' if abs(v) > vmax * 0.6 else 'black'
                ax.text(c, r, f'{v:+.3f}{star}', ha='center', va='center',
                        fontsize=7, color=tc, fontweight='bold' if star else 'normal')
    _grp_dividers = [4.5, 9.5, 14.5, 19.5]
    _grp_labels   = [(2.0, 'Pre\n(abs)'), (7.0, 'Pre\n−BL'),
                     (12.0,'Post\n(abs)'),(17.0,'Post\n−BL'),(22.0,'Δ\npost−pre')]
    for d in _grp_dividers: ax.axhline(d, color='white', lw=2, linestyle='--')
    for y, lbl in _grp_labels:
        ax.text(-0.8, y, lbl, va='center', ha='right', fontsize=8, color='gray')
    plt.colorbar(im, ax=ax, label='CV R²', shrink=0.5)
    ax.set_title(f'c{cell_num} – 5-fold CV R²: Spike Features → LFP  (* FDR q<0.05)', fontsize=11)
    fig.tight_layout(); plt.show()

    # ── Bar chart ──
    x, bw = np.arange(n_t), 0.25
    fig, ax = plt.subplots(figsize=(20, 4.5))
    for p_idx, (pn, col) in enumerate(zip(pred_names, ['#1976D2', '#E53935', '#43A047'])):
        bars = ax.bar(x + (p_idx - 1) * bw, r2_mat[:, p_idx], width=bw,
                      label=pn, color=col, alpha=0.85)
        for b_i, bar in enumerate(bars):
            if sig_mat[b_i, p_idx]:
                bar.set_edgecolor('black'); bar.set_linewidth(2.0)
    ax.set_xticks(x); ax.set_xticklabels(target_labels, rotation=35, ha='right', fontsize=8)
    ax.axhline(0, color='k', lw=1.2)
    for d in _grp_dividers: ax.axvline(d, color='gray', lw=1, linestyle='--')
    ax.set_ylabel('5-fold CV R²'); ax.legend(frameon=False, fontsize=9)
    ax.set_title(f'c{cell_num} – CV R² by predictor set  (bold border = FDR q<0.05)', fontsize=11)
    fig.tight_layout(); plt.show()

    # ── Beta weights (Waveform only) ──
    pname_wv = 'Waveform only'
    n_beta   = len(waveform_labels)
    ncols    = 5
    nrows    = math.ceil(n_t / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.2))
    axes = axes.flatten()
    fig.suptitle(f'c{cell_num} – Beta weights (Waveform only, full-data fit)', fontsize=11)
    for t_idx, (tn, tl) in enumerate(zip(target_names, target_labels)):
        ax   = axes[t_idx]
        res  = results[tn].get(pname_wv, {})
        beta = res.get('beta')
        r2   = res.get('r2_cv', np.nan)
        sig  = res.get('sig_fdr', False)
        if beta is None:
            ax.text(0.5, 0.5, 'no data', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8, color='gray')
            ax.axis('off'); continue
        vals     = [beta.get(l, np.nan) for l in waveform_labels]
        colors_b = ['#E53935' if v > 0 else '#1976D2' for v in vals]
        ax.barh(range(n_beta), vals, color=colors_b, alpha=0.85)
        ax.set_yticks(range(n_beta)); ax.set_yticklabels(waveform_labels, fontsize=7)
        ax.axvline(0, color='k', lw=0.8)
        ax.set_title(f'{tl}\nCV R²={r2:+.3f}{"*" if sig else ""}', fontsize=8,
                     fontweight='bold' if sig else 'normal')
        ax.set_xlabel('β (std units)', fontsize=7)
    for i in range(n_t, len(axes)): axes[i].set_visible(False)
    fig.tight_layout(); plt.show()

    # ── Scatter: FDR-significant waveform-only targets ──
    sig_targets = [(i, tn, tl) for i, (tn, tl) in enumerate(zip(target_names, target_labels))
                   if results[tn].get(pname_wv, {}).get('sig_fdr', False)]
    if not sig_targets:
        print('No FDR-significant targets for Waveform only model.')
        return
    ncols = min(len(sig_targets), 5)
    nrows = math.ceil(len(sig_targets) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes = np.array(axes).flatten() if len(sig_targets) > 1 else [axes]
    fig.suptitle(f'c{cell_num} – Actual vs Predicted (Waveform only, FDR q<0.05)', fontsize=11)
    for ax, (t_idx, tn, tl) in zip(axes, sig_targets):
        res   = results[tn][pname_wv]
        alpha = res.get('best_alpha', 1.0)
        y_raw = Y[:, t_idx]
        valid = np.isfinite(X_waveform).all(axis=1) & np.isfinite(y_raw)
        X, y  = X_waveform[valid], y_raw[valid]
        X_z   = StandardScaler().fit_transform(X)
        sc_y  = StandardScaler()
        y_z   = sc_y.fit_transform(y.reshape(-1, 1)).ravel()
        rr    = Ridge(alpha=alpha, fit_intercept=True); rr.fit(X_z, y_z)
        yp    = sc_y.inverse_transform(rr.predict(X_z).reshape(-1, 1)).ravel()
        ax.scatter(y, yp, alpha=0.15, s=3, rasterized=True, color='#1976D2')
        lo, hi = min(y.min(), yp.min()), max(y.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1)
        ax.set_xlabel('Actual', fontsize=8); ax.set_ylabel('Predicted', fontsize=8)
        ax.set_title(f'{tl}\nCV R²={res["r2_cv"]:+.3f}, p_fdr={res["p_val_fdr"]:.3f}', fontsize=8)
    for i in range(len(sig_targets), len(axes)): axes[i].set_visible(False)
    fig.tight_layout(); plt.show()
