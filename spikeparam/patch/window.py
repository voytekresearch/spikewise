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


def window_spike(sig, fs, spike_ind, times=None, window_length=(10, 10), in_ms=True):
    """Isolate a spike from a full signal.

    Parameters
    ----------
    sig : 1d array
        Full signal.
    fs : float
        Sampling rate, in Hz.
    spike_ind : int
        Index of spike in sig.
        Returned from find_spike_times.
    times : 1d array, optional, default: None
        Time definition.
    window_length : tuple of (float, float), optional, default: (10, 10)
        Pre and post spike padding.
    in_ms : bool, optional, default: True
        Units of window_length.

    Returns
    -------
    spike : 1d array
        Isolated spike.
    spike_times : 1d array, optional
        Times of spike. Only returned if times is not None.
    """

    n_samples = int(fs / 1000) if in_ms else int(fs)

    # Get windows around spikes
    #   create window indices
    window_pre = int(n_samples * window_length[0])
    window_post = int(n_samples * window_length[1])

    # Get window
    window_spike_pre  = (spike_ind-window_pre)
    window_spike_post = (spike_ind+window_post)

    # Get data window
    spike = sig[window_spike_pre:window_spike_post]

    if times is not None:
        # Get window for times as well
        spike_times = times[(spike_ind-window_pre):
                            (spike_ind+window_post)]

        return spike, spike_times

    return spike
