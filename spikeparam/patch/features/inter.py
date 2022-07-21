"""Between spike features."""

import numpy as np


def compute_isi(spike_inds, fs, in_ms=True, group=None):
    """Compute inter-spike intervals.

    Parameters
    ----------
    spike_inds : 1d array
        Sample indices where spikes occur.
    fs : float
        Sampling rate, in Hz.
    in_ms : bool, optional, default: True
        Scale time to ms if True.
    group : 1d array, optional, default: None
        Which group each spike each ind belongs to.
        If None, assumes a single group.

    Returns
    -------
    isi : 1d array
        Time until next spike.
    """

    scale = 1000 if in_ms else 1

    isi = np.zeros(len(spike_inds))
    isi[:-1] = (np.diff(spike_inds) / fs * scale)
    isi[-1] = np.nan

    # Nan the last sample of each group
    if group is not None:

        splits = np.where(np.diff(group))

        if len(splits) > 0:
           isi[splits[0]] = np.nan

    return isi
