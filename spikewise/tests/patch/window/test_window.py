"""Test windowing functions."""

import pytest
import numpy as np

from spikewise.patch.window import find_spike_times, window_spike



def test_find_spike_times():

    sig = np.zeros(1000)

    inds = np.arange(100, 1000, 100)

    sig[inds] = 20

    spike_inds, spike_amps = find_spike_times(sig, 10, 10)

    assert (spike_inds == inds).all()
    assert (spike_amps == 20).all()


@pytest.mark.parametrize('use_times', [True, False])
def test_window_spike(use_times):

    sig = np.arange(10000)
    fs = 1000
    spike_inds = np.arange(100, 1000, 100).astype(int)

    if use_times:
        spikes, spike_times = window_spike(sig, fs, spike_inds, times=sig,
                                           window_length=(2., 2.), in_ms=True)

        assert (spikes == spike_times).all()
    else:
        spikes = window_spike(sig, fs, spike_inds, times=None,
                              window_length=(2., 2.), in_ms=True)


    assert len(spikes) == len(spike_inds)
    assert len(spikes[0]) == (4 + 1)
    assert (spikes[:, 2] == spike_inds).all()

    # Int input
    if use_times:
        spikes, spike_times = window_spike(sig, fs, 100, times=sig,
                                           window_length=(2., 2.), in_ms=True)
        assert (spikes == spike_times).all()
    else:
        spikes = window_spike(sig, fs, 100, times=None,
                              window_length=(2., 2.), in_ms=True)

    assert spikes.ndim == 1 and len(spikes) == 5 and spikes[2] == 100

    # Edge
    fs = 1000

    spike_inds = np.arange(0, 1000, 100).astype(int)

    spikes, spike_times = window_spike(sig, fs, spike_inds, times=sig, window_length=(2., 2.))

    assert (spikes == spike_times).all()
    assert spikes[0][len(spikes)//2] != 0
