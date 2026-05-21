"""
ridge_regression_utils.py
--------------------------
Helper functions for the per-cell spike-waveform → LFP ridge regression analysis.

Main entry point
----------------
build_ridge_matrices(df_reg, specparam_by_spike, lfp_windows_by_spike,
                     pre_win, post_win, baseline_win)
    → X_waveform, X_log_isi, X_both, waveform_labels,
      Y, target_names, target_labels
"""

import numpy as np


# ── Low-level helpers ─────────────────────────────────────────────────────────

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


# ── Matrix builder ────────────────────────────────────────────────────────────

WAVEFORM_COLS = [
    'ramp_amp', 'inflection_time', 'inflection_amp', 'peak_amp',
    'peak_width', 'peak_sharpness', 'exp_lambda', 'exp_const',
]
WAVEFORM_LABELS = [
    'Ramp Amp', 'Infl. Time', 'Infl. Amp', 'Peak Amp',
    'Peak Width', 'Sharpness', 'Decay λ', 'Decay Const',
]

FEAT_KEYS   = ['lfp_amp', 'lfp_std', 'gamma_auc', 'exponent', 'theta_auc']
FEAT_LABELS = ['LFP Amp', 'LFP Std', 'Gamma AUC', 'Exponent', 'Theta AUC']


def build_ridge_matrices(df_reg, specparam_by_spike, lfp_windows_by_spike,
                         pre_win, post_win, baseline_win):
    """
    Build predictor (X) and target (Y) matrices for the per-cell ridge regression.

    Parameters
    ----------
    df_reg : pd.DataFrame
        Spike feature dataframe (aligned to specparam_by_spike length).
    specparam_by_spike : list of dict
        Trimmed per-spike specparam results.
    lfp_windows_by_spike : list of dict
        Per-spike simple LFP results (lfp_mean, lfp_std, t_bins_s).
    pre_win, post_win, baseline_win : (float, float)
        Time windows in seconds.

    Returns
    -------
    X_waveform : ndarray (n, 8)
    X_log_isi  : ndarray (n, 1)
    X_both     : ndarray (n, 9)
    waveform_labels : list[str]
    Y           : ndarray (n, 15)  — 5 pre abs + 5 pre−BL + 5 delta
    target_names  : list[str]
    target_labels : list[str]
    """
    n = len(specparam_by_spike)

    # ── Predictors ──
    X_waveform = df_reg[WAVEFORM_COLS].values.astype(float)
    X_log_isi  = df_reg[['log_isi']].values.astype(float)
    X_both     = np.hstack([X_waveform, X_log_isi])

    # ── Target names / labels ──
    target_names = (
        [f'pre_{k}'   for k in FEAT_KEYS] +
        [f'prebc_{k}' for k in FEAT_KEYS] +
        [f'delta_{k}' for k in FEAT_KEYS]
    )
    target_labels = (
        [f'Pre {l}'    for l in FEAT_LABELS] +
        [f'Pre−BL {l}' for l in FEAT_LABELS] +
        [f'Δ {l}'      for l in FEAT_LABELS]
    )

    # ── Build Y ──
    Y = np.full((n, 15), np.nan)

    for i in range(n):
        sp        = specparam_by_spike[i]
        sw        = lfp_windows_by_spike[i]
        t_sp      = sp.get('t_bins_s')       if sp else None
        t_sw      = sw['t_bins_s']            if sw is not None else None
        band_aucs = sp.get('band_aucs', {})   if sp else {}

        def _sp(key):  return sp.get(key) if sp else None
        def _ba(band): return band_aucs.get(band)

        pre_vals = [
            win_avg(t_sw, sw['lfp_mean'] if sw else None, pre_win),
            win_avg(t_sw, sw['lfp_std']  if sw else None, pre_win),
            win_avg(t_sp, _ba('gamma'),                   pre_win),
            win_avg(t_sp, _sp('exponent'),                pre_win),
            win_avg(t_sp, _ba('theta'),                   pre_win),
        ]
        bl_vals = [
            win_avg(t_sw, sw['lfp_mean'] if sw else None, baseline_win),
            win_avg(t_sw, sw['lfp_std']  if sw else None, baseline_win),
            win_avg(t_sp, _ba('gamma'),                   baseline_win),
            win_avg(t_sp, _sp('exponent'),                baseline_win),
            win_avg(t_sp, _ba('theta'),                   baseline_win),
        ]
        post_vals = [
            win_avg(t_sw, sw['lfp_mean'] if sw else None, post_win),
            win_avg(t_sw, sw['lfp_std']  if sw else None, post_win),
            win_avg(t_sp, _ba('gamma'),                   post_win),
            win_avg(t_sp, _sp('exponent'),                post_win),
            win_avg(t_sp, _ba('theta'),                   post_win),
        ]

        Y[i,  0: 5] = pre_vals
        Y[i,  5:10] = [p - b if np.isfinite(p) and np.isfinite(b) else np.nan
                       for p, b in zip(pre_vals, bl_vals)]
        Y[i, 10:15] = [po - pr if np.isfinite(po) and np.isfinite(pr) else np.nan
                       for po, pr in zip(post_vals, pre_vals)]

    return (X_waveform, X_log_isi, X_both, WAVEFORM_LABELS,
            Y, target_names, target_labels)
