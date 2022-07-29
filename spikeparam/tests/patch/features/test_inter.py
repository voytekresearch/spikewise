"""Test between spike features."""

import numpy as np

from spikeparam.patch.features import compute_isi



def test_compute_isi():

    fs = 1000

    spike_inds = np.arange(0, 10 * fs, fs)

    isi = compute_isi(spike_inds, fs, in_ms=False)
    assert (isi[:-1] == 1).all()
    assert np.isnan(isi[-1])

    isi = compute_isi(spike_inds, fs, in_ms=True)
    assert (isi[:-1] == fs).all()
    assert np.isnan(isi[-1])

    groups = np.concatenate((np.zeros(len(spike_inds)), np.ones(len(spike_inds))))
    spike_inds = np.concatenate((spike_inds,  spike_inds))

    isi = compute_isi(spike_inds, fs, group=groups)

    assert (isi[:len(isi)//2-1] == isi[len(isi)//2:-1]).all()

    assert np.isnan(isi[(len(isi)//2)-1])
    assert np.isnan(isi[-1])
