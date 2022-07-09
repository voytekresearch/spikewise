"""Utility functions."""

import numpy as np


def create_times(sig, fs, in_ms=True):
    """Create spike times."""

    times = np.arange(0, len(sig)/fs, 1/fs)

    if in_ms:
        times *= 1000

    return times
