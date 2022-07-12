"""Functions to compute spike control points."""

import numpy as np
import statsmodels.api as sm

from .window import find_spike_times



def control_points(spike, fs, thresh_ms=1., thresh_zscore=40.,
                   smooth_frac=.008, exp_shift_right=2., exp_duration=5.):
    """Compute spike control points.

    Parameters
    ----------
    spike : 1d array
        Spike waveform.
    fs : float
        Sampling rate, in Hz.
    thresh_ms : int, optional, default: 1.
        Minimum miliseconds between successive peaks.
    thresh_zscore : float, optional, default: 40.
        Peak z-score threshold.
    smooth_frac : float, optional, default: .008
        Smoothing fraction.
    exp_shift_right : float, optional, default: 2.
        Start time, in ms, to exponential start from peak.
    exp_duration : float, optional, default: 5.
        End time, in ms, of the exponential from the (shifted) peak.

    Returns
    -------
    indices : list of int
        Indices as:

        - idx_ramp_start : Index of ramp start.
        - idx_inflection : Index of inflection
        - idx_rise : Index of rise midpoint.
        - idx_peak : Index of peak.
        - idx_decay : Index of decay midpoint.
        - idx_exp_start : Index of exponential start
        - idx_exp_end : Index of exponential end

    """

    # Smoothed derivative
    times = np.arange(0, len(spike)/fs, 1/fs)
    _, d_smoothed_spike = diff_spike(times, spike, smooth_frac)

    # Get peak index, assumes array is centered on peak
    idx_peak = len(spike) // 2

    # Get ramping start and inflection
    idx_ramp_start, idx_inflection = inflection(d_smoothed_spike, fs, thresh_ms,
                                                thresh_zscore)

    # Get peak mid-points
    mid_amp = (spike[idx_inflection] + spike[idx_peak]) / 2

    _x = (spike[:int(idx_peak)] - mid_amp)[::-1]
    idx_rise  = idx_peak - np.where(_x <= _x/2)[0][0]

    _x = (spike[int(idx_peak):] - mid_amp)
    idx_decay = idx_peak + np.where(_x <= _x/2)[0][0]

    # Exponential window points
    one_ms = int(fs / 1000)

    decay_curve_shift = int(one_ms / exp_shift_right)

    decay_curve_floor_time = int(one_ms * exp_duration)

    idx_exp_start = idx_peak + decay_curve_shift
    idx_exp_end = idx_exp_start + decay_curve_floor_time

    # Collect indices
    indices = [
        idx_ramp_start, idx_inflection, idx_rise,
        idx_peak, idx_decay, idx_exp_start, idx_exp_end
    ]

    return indices


def inflection(d_smoothed_spike, fs, thresh_ms=1., thresh_zscore=40.):
    """Compute inflection points.

    Parameters
    ----------
    d_smoothed_spike : 1d array
        Smoothed spike deriviative.
    fs : float
        Sampling rate, in Hz.
    thresh_ms : float, optional, default: 1.
        Minimum miliseconds between successive peaks.
    thresh_zscore : float, optional, default: 40.
        Peak z-score threshold.

    Returns
    -------
    idx_ramp_start : int
        Start index of ramp, relative to windowed spike.
    idx_inflection : int
        End index of ramp, relative to windowed spike

    Notes
    -----
    Z-score threshold for inflection point
    is really high, since it's so low noise.
    """

    one_ms = int(fs / 1000)
    thresh_ms *= one_ms

    # Inflection time
    #   Get mean and std of first ms of data
    noise_window = d_smoothed_spike[0:one_ms]
    noise_mean = np.mean(noise_window)
    noise_std = np.std(noise_window)

    # Z-score spike relative to window above
    z_data = (d_smoothed_spike-noise_mean) / noise_std

    # Find the peak of the zscored data
    idx_z_peak, _ = find_spike_times(z_data, thresh_zscore, thresh_ms)

    # Find inflection voltage and time relative to spike peak
    inflection_time = np.abs(z_data[0:idx_z_peak[0]]-thresh_zscore)
    idx_inflection = np.argmin(inflection_time)

    # Get the voltage ramp slope for the 0.5 ms before inflection point
    idx_ramp_start = idx_inflection - int(0.5 * one_ms) # 0.5 ms before

    return idx_ramp_start, idx_inflection


def diff_spike(times, spike, smooth_frac=.008):
    """Compute the smoothed derivative of a spike.

    Parameters
    ----------
    times : 1d array
        Spike times.
    spike : 1d array
        Spike waveform.
    smooth_frac : float, optional, default: .008
        Smoothing fraction.

    Returns
    -------
    d_smoothed_times : 1d array
        Smoothed spike derivative times.
    d_smoothed_spike : 1d array
        Smoothed spike derivative.
    """

    # Difference of data and smooth it
    smoothed= sm.nonparametric.lowess(np.diff(spike), times[1:], frac=smooth_frac)

    # Unpack
    d_smoothed_times = smoothed[:, 0]
    d_smoothed_spike = smoothed[:, 1]

    return d_smoothed_times, d_smoothed_spike
