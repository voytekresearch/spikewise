"""Spike windowing functions."""

import numpy as np
from scipy.signal import find_peaks



def find_spike_times(sig, thresh_amp, thresh_ms):
    """Find spikes as peaks.

    Parameters
    ----------
    sig : 1d array
        Full signal.
    thresh_amp : float
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
    peaks = find_peaks(sig, height=thresh_amp, distance=thresh_ms)
    idx_spikes = peaks[0] # spike indices
    amp_spikes = peaks[1]['peak_heights'] # spike amplitudes

    return idx_spikes, amp_spikes


def window_spike(sig, fs, spike_inds, times=None, window_length=(10., 10.), in_ms=True):
    """Isolate a spike from a full signal.

    Parameters
    ----------
    sig : 1d array
        Full signal.
    fs : float
        Sampling rate, in Hz.
    spike_ind : int or 1d array of int
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
    spikes : 1d or 2d array
        Isolated spikes.
    spike_times : 1d or 2d array, optional
        Times of spikes. Only returned if times is not None.

    Notes
    -----
    The dimensions of spikes and spike_times will match the spike_inds parameter.
    """

    int_input = False

    if isinstance(spike_inds, (int, np.int64)):
        spike_inds = [spike_inds]
        int_input = True

    initalized = False
    keep = np.array([True] * len(spike_inds))
    n_samples = fs / 1000 if in_ms else fs

    for ind in range(len(spike_inds)):

        # Get windows around spikes
        #   create window indices
        window_pre =  n_samples * window_length[0]
        window_post = n_samples * window_length[1]

        # Get window
        window_spike_pre  = int(spike_inds[ind]-window_pre)
        window_spike_post = int(spike_inds[ind]+window_post) + 1

        # Skip spike if full window can't be sliced
        if window_spike_pre < 0 or window_spike_post > len(sig):
            keep[ind] = False
            continue

        # Get data window
        _spike = sig[window_spike_pre:window_spike_post]

        if times is not None:
            # Get window for times as well
            _spike_times = times[int(spike_inds[ind]-window_pre):
                                 int(spike_inds[ind]+window_post)+1]

        # Return early if spike_times is an int
        if int_input and times is not None:
            return _spike, _spike_times
        elif int_input:
            return _spike

        # Initialize arrays
        if not initalized and times is not None:
            spike_times = np.zeros((len(spike_inds), len(_spike_times)))

        if not initalized:
            spikes = np.zeros((len(spike_inds), len(_spike)))
            initalized = True

        # Set arrays
        spikes[ind] = _spike

        if times is not None:
            spike_times[ind] = _spike_times

    if times is not None:
        return spikes[keep], spike_times[keep]

    return spikes[keep]
