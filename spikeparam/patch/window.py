"""Spike windowing functions."""

import numpy as np
from scipy.signal import find_peaks


def find_spike_times(sig, thresh_mv, thresh_ms):
    """Find spikes as peaks.

    Parameters
    ----------
    sig : 1d array
        Full signal.
    thresh_mv : float
        Voltage threshold.
    thresh_ms : float
        Minimum time between peaks, in ms.

    Returns
    -------
    idx_spikes : 1d array
        Indices of spikes.
    amp_spikes : 1d array
        Amplitude of spikes.
    """
    peaks = find_peaks(sig, height=thresh_mv, distance=thresh_ms)
    idx_spikes = peaks[0] # spike indices
    amp_spikes = peaks[1]['peak_heights'] # spike amplitudes

    return idx_spikes, amp_spikes


def window_spike(sig, times, fs, spike_ind, window_length=(10, 10), in_ms=True):
    """Isolate a spike from a full signal.

    Parameters
    ----------
    sig : 1d array
        Full signal.
    times : 1d array
        Time definition.
    fs : float
        Sampling rate, in Hz.
    spike_ind : int
        Index of spike in sig.
        Returned from find_spike_times.
    window_length : tuple of (float, float)
        Pre and post spike padding.
    """

    n_samples = int(fs / 1000) if in_ms else int(fs)

    # Get windows around spikes
    #   create window indices
    window_pre = int(n_samples * window_length[0])
    window_post = int(n_samples * window_length[1])

    # Get window
    window_spike_pre  = (spike_ind-window_pre)
    window_spike_post = (spike_ind+window_post)

    # Get window for times as well
    spike_times = times[(spike_ind-window_pre):
                        (spike_ind+window_post)]

    # Get data window
    spike = sig[window_spike_pre:window_spike_post]

    return spike, spike_times
