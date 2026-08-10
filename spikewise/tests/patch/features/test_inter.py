"""Test between spike features."""

import numpy as np

from spikewise.patch.features import compute_isi



def test_compute_isi():

    fs = 1000

    spike_inds = np.arange(0, 10 * fs, fs)

    isi = compute_isi(spike_inds, fs, in_ms=False)
    assert (isi[:-1] == 1).all()
    assert np.isnan(isi[-1])

    isi = compute_isi(spike_inds, fs, in_ms=True)
    assert (isi[:-1] == fs).all()
    assert np.isnan(isi[-1])


    spike_inds = [spike_inds,  spike_inds]

    isi = compute_isi(spike_inds, fs)

    assert (isi[:len(isi)//2-1] == isi[len(isi)//2:-1]).all()

    assert np.isnan(isi[(len(isi)//2)-1])
    assert np.isnan(isi[-1])
