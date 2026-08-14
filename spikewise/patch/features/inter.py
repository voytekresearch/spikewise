"""Between spike features."""

import numpy as np


def compute_isi(spike_inds, fs, in_ms=True):
    """Compute inter-spike intervals.

    Parameters
    ----------
    spike_inds : 1d array or list of 1d array
        Sample indices where spikes occur.
    fs : float
        Sampling rate, in Hz.
    in_ms : bool, optional, default: True
        Scale time to ms if True.

    Returns
    -------
    isi : 1d array
        Time until next spike.
    """

    scale = 1000 if in_ms else 1

    if isinstance(spike_inds, np.ndarray) and spike_inds.ndim == 1:
        isi = np.zeros(len(spike_inds))
        isi[:-1] = (np.diff(spike_inds) / fs * scale)
        isi[-1] = np.nan
    else:
        # spike_inds is 2d and non=continuous between 1d slices
        isi = []
        for inds in spike_inds:
            _isi = np.zeros(len(inds))
            _isi[:-1] = (np.diff(inds) / fs * scale)
            _isi[-1] = np.nan
            isi.extend(_isi)
        isi = np.array(isi)

    return isi


def compute_isi_prev(spike_inds, fs, in_ms=True):
    """Compute preceding inter-spike intervals.

    Parameters
    ----------
    spike_inds : 1d array or list of 1d array
        Sample indices where spikes occur.
    fs : float
        Sampling rate, in Hz.
    in_ms : bool, optional, default: True
        Scale time to ms if True.

    Returns
    -------
    isi_prev : 1d array
        Time since the preceding spike. The first spike per recording is NaN.
    """

    scale = 1000 if in_ms else 1

    if isinstance(spike_inds, np.ndarray) and spike_inds.ndim == 1:
        isi_prev = np.zeros(len(spike_inds))
        isi_prev[1:] = (np.diff(spike_inds) / fs * scale)
        isi_prev[0] = np.nan
    else:
        # spike_inds is 2d and non-continuous between 1d slices
        isi_prev = []
        for inds in spike_inds:
            _isi_prev = np.zeros(len(inds))
            _isi_prev[1:] = (np.diff(inds) / fs * scale)
            _isi_prev[0] = np.nan
            isi_prev.extend(_isi_prev)
        isi_prev = np.array(isi_prev)

    return isi_prev


def compute_spike_count(spike_inds, fs, window_ms):
    """Count spikes in the window preceding each spike.

    Parameters
    ----------
    spike_inds : 1d array or list of 1d array
        Sample indices where spikes occur.
    fs : float
        Sampling rate, in Hz.
    window_ms : float
        Duration of the preceding window, in ms.

    Returns
    -------
    counts : 1d array
        Number of other spikes within `window_ms` before each spike, excluding
        the spike itself. Counts do not cross recording boundaries.
    """

    window_samples = window_ms * fs / 1000

    def _counts_1d(inds):
        inds = np.asarray(inds)
        lower_bounds = inds - window_samples
        left = np.searchsorted(inds, lower_bounds, side='right')
        return np.arange(len(inds)) - left

    if isinstance(spike_inds, np.ndarray) and spike_inds.ndim == 1:
        counts = _counts_1d(spike_inds)
    else:
        counts = []
        for inds in spike_inds:
            counts.extend(_counts_1d(inds))
        counts = np.array(counts)

    return counts
