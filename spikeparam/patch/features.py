"""Spike features."""

import numpy as np
from scipy.optimize import curve_fit

from .gen import exp_func

def compute_ramp_features(times, spike, fs, idx_ramp_start,
                          idx_inflection, idx_peak, poly_order=1):
    """Compute ramp features."""

    ramp_times = times[idx_ramp_start:idx_inflection]
    ramp = spike[idx_ramp_start:idx_inflection]

    poly_params = np.polyfit(ramp_times, ramp, poly_order)
    voltage_ramp = poly_params[0]

    inflection_time = (idx_peak - idx_inflection) / int(fs / 1000)
    inflection_mv = ramp[-1]

    return poly_params, voltage_ramp, inflection_time, inflection_mv


def compute_peak_features(spike, fs, idx_decay, idx_peak):
    """Compute peak features."""
    peak_width = (idx_decay - idx_peak) / int(fs/1000)

    peak_sharpness = ((spike[idx_peak]-spike[idx_peak-5]) +
                      (spike[idx_peak]-spike[idx_peak+5])) / 2

    return peak_width, peak_sharpness


def compute_decay_features(times, spike, idx_exp_start, idx_exp_end):
    """Compute exponential decay features."""
    exp = spike[idx_exp_start:idx_exp_end]
    times = times[idx_exp_start:idx_exp_end]

    # Initial guesses, derived from physiological estimating
    initial_guesses = np.array([50, 1, times[0], -60], dtype=np.float64)

    # Reasonable bounds
    bounds = ([0, 0, 0, -100], [1000, 3, 10e9, 50])

    # Fit
    exp_amp, exp_lambda, exp_timeshift, exp_const = \
        fit_exp_nonlinear(times, exp, initial_guesses, bounds)

    return exp_amp, exp_lambda, exp_timeshift, exp_const


def fit_exp_nonlinear(times, exp, initial_guesses, bounds):
    """Fit exponential decay."""
    popt, _ = curve_fit(exp_func, times, exp, p0=initial_guesses,
                        bounds=bounds, maxfev=10000)

    exp_amp, exp_lambda, exp_timeshift, exp_const = popt

    return exp_amp, exp_lambda, exp_timeshift, exp_const
