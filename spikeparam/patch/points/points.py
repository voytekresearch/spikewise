"""Functions to compute spike control points."""

import numpy as np
import statsmodels.api as sm



def control_points(spike, fs, pre_peak_ms=(-4., -1.), pre_inflection_ms=1.,
                   smooth_frac=.008, exp_shift_right=2., exp_duration=5., peak_ind=None):
    """Compute spike control points.

    Parameters
    ----------
    spike : 1d array
        Spike waveform.
    fs : float
        Sampling rate, in Hz.
    thresh_zscore : float, optional, default: 'auto'.
        Peak z-score threshold. Auto defaults to half z-score max.
    smooth_frac : float, optional, default: .008
        Smoothing fraction.
    exp_shift_right : float, optional, default: 2.
        Start time, in ms, to exponential start from peak.
    exp_duration : float, optional, default: 5.
        End time, in ms, of the exponential from the (shifted) peak.
    peak_ind : int, optional, default: None
        Peak index. None defaults to maximum.

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
    times = np.arange(0, len(spike)/fs, 1/fs)[:len(spike)]
    _, d_smoothed_spike = diff_spike(times, spike, smooth_frac)

    # Get peak index
    idx_peak = np.argmax(spike) if peak_ind is None else peak_ind

    # Get ramping start and inflection
    idx_ramp_start, idx_inflection = inflection(d_smoothed_spike, fs, idx_peak,
                                                pre_peak_ms, pre_inflection_ms)

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
    idx_exp_end = len(spike)-1 if idx_exp_end > len(spike) else idx_exp_end

    # Collect indices
    indices = [
        idx_ramp_start, idx_inflection, idx_rise,
        idx_peak, idx_decay, idx_exp_start, idx_exp_end
    ]

    return indices


def inflection(d_smoothed_spike, fs, peak_ind, pre_peak_ms=(-4., -1.), pre_inflection_ms=1.):
    """Compute inflection points.

    Parameters
    ----------
    spike : 1d array
        Spike waveform.
    fs : float
        Sampling rate, in Hz.
    peak_ind : int
        Peak index.
    pre_peak_ms : tuple of (float, float)
        Time before the peak to estimate linear ramp fit as.
    pre_inflection_ms : float
        Time before the inflection point to define the ramp start.

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

    # Segment spike
    xs = np.arange(len(d_smoothed_spike))

    d_peak_ind = np.argmax(d_smoothed_spike[peak_ind-one_ms:peak_ind+one_ms])
    d_peak_ind += peak_ind-one_ms

    # Rising half-max of derivative
    ind_rise = np.where(d_smoothed_spike[:d_peak_ind][::-1] <=
                       (d_smoothed_spike[:d_peak_ind].std()))[0]

    ind_rise = 1 if len(ind_rise) == 0 else ind_rise[0] + 1

    ind_rise = d_peak_ind - ind_rise

    # Ramp
    start_ramp = int(d_peak_ind + (pre_peak_ms[0] * one_ms))
    end_ramp   = int(d_peak_ind + (pre_peak_ms[1] * one_ms))

    start_ramp = 0 if start_ramp < 0 else start_ramp
    end_ramp = ind_rise if end_ramp > ind_rise else end_ramp

    # Linear fit
    p0 = np.polyfit(xs[start_ramp:end_ramp], d_smoothed_spike[start_ramp:end_ramp], 1)
    p1 = np.polyfit(xs[ind_rise:d_peak_ind], d_smoothed_spike[ind_rise:d_peak_ind], 1)

    # Take root of poly fits to find intersection
    idx_inflection = int(np.roots(p0-p1)[0])

    # Get the voltage ramp slope for the N ms before inflection point
    idx_ramp_start = idx_inflection - int(pre_inflection_ms * one_ms)

    if idx_ramp_start < 0:
        idx_ramp_start = 0

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
