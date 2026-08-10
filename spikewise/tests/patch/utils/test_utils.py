"""Test utility functions."""

import pytest
import numpy as np
from spikewise.patch.utils import create_times



@pytest.mark.parametrize('in_ms', [True, False])
def test_create_times(check_param, in_ms):

    sig = np.zeros(10000)

    fs = 1000
    times = create_times(sig, fs, in_ms)

    check_param(times, np.ndarray, (0, len(sig) / fs * (1000 if in_ms else 1)), len(sig))
