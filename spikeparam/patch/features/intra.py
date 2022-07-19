"""Within spike features."""

import numpy as np
from scipy.optimize import curve_fit

from ..points import control_points
from ..gen import exp_func



def compute_features(spike, fs, pre_peak_ms=(-4., -1.), pre_inflection_ms=1.,
                     smooth_frac=.008, poly_order=1, exp_shift_right=2.0, exp_duration=5.0):
    """Compute features.

    Parameters
    ----------
    spike : 1d array
        Spike waveform.
    fs : float
        Sampling rate, in Hz.
    pre_peak_ms : tuple of (float, float)
        Time before the peak to estimate linear ramp fit as.
    pre_inflection_ms : float
        Time before the inflection point to define the ramp start.
    smooth_frac : float, optional, default: .008
        Smoothing fraction.
    exp_shift_right : float, optional, default: 2.
        Start time, in ms, to exponential start from peak.
    exp_duration : float, optional, default: 5.
        End time, in ms, of the exponential from the (shifted) peak.

    Returns
    -------
    indices : 1d array
        Control points indices, relative to spike.
        See return doc for points.control_points.
    ramp_params : 1d array
        Ramp features.
    peak_params : 1d array
        Peak width and sharpness.
    exp_params : 1d array
        Exponential decay parameters.
    """
    # Control points
    indices = control_points(spike, fs, pre_peak_ms, pre_inflection_ms,
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
    """Compute ramp features.

    Parameters
    ----------
    spike : 1d array
        Spike waveform.
    fs : float
        Sampling rate, in Hz.
    idx_ramp_start : int
        Start index of ramp section.
    idx_inflection : int
        End index of ramp section.
    idx_peak : int
        Index of the peak.
    poly_order : int, optional, default: 1
        Polynomial order to fit.

    Returns
    -------
    poly_params : 1d array
        Polynomial parameters.
    voltage_ramp : float
        First polynomial parameter (e.g. offset).
    inflection_time : float
        Time, in ms, of the inflection point.
    inflection_mv : float
        Voltage, in mv, at time of inflection.
    """

    ramp = spike[idx_ramp_start:idx_inflection]
    times = np.arange(len(ramp)) * 1000 / fs

    poly_params = np.polyfit(times, ramp, poly_order)
    voltage_ramp = poly_params[0]

    inflection_time = (idx_peak - idx_inflection) / int(fs / 1000)
    inflection_mv = ramp[-1]

    return poly_params, voltage_ramp, inflection_time, inflection_mv


def compute_peak_features(spike, fs, idx_rise, idx_peak, idx_decay):
    """Compute peak features.

    Parameters
    ----------
    spike : 1d array
        Spike waveform.
    fs : float
        Sampling rate, in Hz.
    idx_rise : int
        Index of rising midpoint (to peak).
    idx_peak : int
        Index of peak.
    idx_decay : int
        Index of decaying midpoint (from peak).

    Returns
    -------
    peak_width : float
        Width of peak, in ms.
    peak_sharpness : float
        Sharpness of peak.
    """
    peak_mv = spike[idx_peak]

    peak_width = (idx_decay - idx_rise) / int(fs/1000)

    peak_sharpness = ((spike[idx_peak]-spike[idx_peak-5]) +
                      (spike[idx_peak]-spike[idx_peak+5])) / 2

    return peak_mv, peak_width, peak_sharpness


def compute_decay_features(spike, fs, idx_exp_start, idx_exp_end):
    """Compute exponential decay features.

    Parameters
    ----------
    spike : 1d array
        Spike waveform.
    fs : float
        Sampling rate, in Hz.
    idx_exp_start : int
        Start index of exponential decay.
    idx_exp_end : int
        End index of exponential decay.

    Returns
    -------
    exp_amp : float
        Exponential amplitdue.
    exp_lambda : float
        Exponential decay.
    exp_const : float
        Exponential constant.
    """

    # Slice exponential portion
    exp = spike[idx_exp_start:idx_exp_end]

    # Times, in ms
    times = np.arange(len(exp)) * 1000 / fs

    # Initial guesses, derived from physiological estimating
    p0 = np.array([50, 1, -60], dtype=np.float64)

    # Reasonable bounds
    bounds = ([0, 0, -100], [1000, 10, 50])

    # Fit
    exp_amp, exp_lambda, exp_const = \
        fit_exp_nonlinear(times, exp, p0, bounds)

    return exp_amp, exp_lambda, exp_const


def fit_exp_nonlinear(times, exp, p0, bounds):
    """Fit exponential decay.

    Parameters
    ----------
    times : 1d array
        Time definition.
    exp : 1d array
        Voltage definition.
    p0 : tuple or list
        Initial parameter estimates.
    bounds : list of tuple
        Lower and upper bounds.

    Returns
    -------
    exp_amp : float
        Exponential amplitdue.
    exp_lambda : float
        Exponential decay.
    exp_const : float
        Exponential constant.
    """
    popt, _ = curve_fit(exp_func, times, exp, p0=p0,
                        bounds=bounds, maxfev=10000)

    exp_amp, exp_lambda, exp_const = popt

    return exp_amp, exp_lambda, exp_const
