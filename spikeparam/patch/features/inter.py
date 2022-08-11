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
