"""Utility functions."""

import numpy as np



def create_times(sig, fs, in_ms=True):
    """Create spike times.

    Parameters
    ----------
    sig : 1d array
        Voltage time series.
    fs : float
        Sampling rate in Hz.
    in_ms : bool, optional, default: True
        Converts time array from seconds of milliseconds when True.

    Returns
    -------
    times : 1d array
        Time definition.
    """

    times = np.arange(0, len(sig)/fs, 1/fs)

    if in_ms:
        times *= 1000

    return times
