"""Spike features."""

import numpy as np
from scipy.optimize import curve_fit

from .window import window_spike
from .points import control_points
from .gen import exp_func



def compute_features(spike, fs, times=None, thresh_ms=1, thresh_zscore=40.,
                     smooth_frac=.008, poly_order=1, exp_shift_right=2.0, exp_duration=5.0):
    """"""

    if times is None:
        times = np.arange(0, len(spike)/fs, 1/fs)

    # Control points
    indices = control_points(spike, fs, times, thresh_ms, thresh_zscore,
                             smooth_frac, exp_shift_right, exp_duration)

    # Unpack indices
    idx_ramp_start, idx_inflection, idx_rise, \
            idx_peak, idx_decay, idx_exp_start, idx_exp_end = indices

    # Ramp features
    ramp_params = compute_ramp_features(spike, fs, idx_ramp_start, idx_inflection,
                                        idx_peak, poly_order=poly_order)

    # Peak features
    peak_params = compute_peak_features(spike, fs, idx_rise, idx_peak, idx_decay)

    # Exponential decay features
    exp_params = compute_decay_features(spike, fs, idx_exp_start, idx_exp_end)

    return indices, ramp_params, peak_params, exp_params


def compute_ramp_features(spike, fs, idx_ramp_start,
                          idx_inflection, idx_peak, poly_order=1):
    """Compute ramp features."""

    ramp = spike[idx_ramp_start:idx_inflection]

    times = np.arange(len(ramp)) * 1000 / fs

    poly_params = np.polyfit(times, ramp, poly_order)
    voltage_ramp = poly_params[0]

    inflection_time = (idx_peak - idx_inflection) / int(fs / 1000)
    inflection_mv = ramp[-1]

    return poly_params, voltage_ramp, inflection_time, inflection_mv


def compute_peak_features(spike, fs, idx_rise, idx_peak, idx_decay):
    """Compute peak features."""
    peak_width = (idx_decay - idx_rise) / int(fs/1000)

    peak_sharpness = ((spike[idx_peak]-spike[idx_peak-5]) +
                      (spike[idx_peak]-spike[idx_peak+5])) / 2

    return peak_width, peak_sharpness


def compute_decay_features(spike, fs, idx_exp_start, idx_exp_end):
    """Compute exponential decay features."""

    # Slice exponential portion
    exp = spike[idx_exp_start:idx_exp_end]

    # Times, in ms
    times = np.arange(len(exp)) * 1000 / fs

    # Initial guesses, derived from physiological estimating
    initial_guesses = np.array([50, 1, -60], dtype=np.float64)

    # Reasonable bounds
    bounds = ([0, 0, -100], [1000, 3, 50])

    # Fit
    exp_amp, exp_lambda, exp_const = \
        fit_exp_nonlinear(times, exp, initial_guesses, bounds)

    return exp_amp, exp_lambda, exp_const


def fit_exp_nonlinear(times, exp, initial_guesses, bounds):
    """Fit exponential decay."""
    popt, _ = curve_fit(exp_func, times, exp, p0=initial_guesses,
                        bounds=bounds, maxfev=10000)

    exp_amp, exp_lambda, exp_const = popt

    return exp_amp, exp_lambda, exp_const
