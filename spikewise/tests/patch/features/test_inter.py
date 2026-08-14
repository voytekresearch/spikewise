"""Test between spike features."""

import numpy as np

from spikewise.patch.features import compute_isi, compute_isi_prev, compute_spike_count



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


def test_compute_isi_prev():

    fs = 1000

    spike_inds = np.arange(0, 10 * fs, fs)

    isi_prev = compute_isi_prev(spike_inds, fs, in_ms=False)
    assert np.isnan(isi_prev[0])
    assert (isi_prev[1:] == 1).all()

    isi_prev = compute_isi_prev(spike_inds, fs, in_ms=True)
    assert np.isnan(isi_prev[0])
    assert (isi_prev[1:] == fs).all()

    # isi_prev is isi shifted by one index
    isi = compute_isi(spike_inds, fs)
    assert (isi_prev[1:] == isi[:-1]).all()

    spike_inds = [spike_inds, spike_inds]

    isi_prev = compute_isi_prev(spike_inds, fs)

    assert np.isnan(isi_prev[0])
    assert np.isnan(isi_prev[len(isi_prev)//2])
    assert (isi_prev[1:len(isi_prev)//2] == isi_prev[len(isi_prev)//2+1:]).all()


def test_compute_spike_count():

    fs = 1000

    # Evenly spaced at 1 spike / 10ms - a 50ms window should always see 4 preceding spikes
    # once enough history has accumulated, and fewer near the start
    spike_inds = np.arange(0, 100 * fs // 100, 10)

    counts = compute_spike_count(spike_inds, fs, window_ms=50.)

    assert counts[0] == 0
    assert counts[1] == 1
    assert counts[4] == 4
    assert counts[-1] == 4

    # A shorter window should see fewer preceding spikes than a longer one
    counts_short = compute_spike_count(spike_inds, fs, window_ms=20.)
    counts_long = compute_spike_count(spike_inds, fs, window_ms=200.)
    assert (counts_short <= counts).all()
    assert (counts <= counts_long).all()

    # Counts should not cross recording boundaries
    spike_inds_group = [spike_inds, spike_inds]
    counts_group = compute_spike_count(spike_inds_group, fs, window_ms=50.)
    assert counts_group[0] == 0
    assert counts_group[len(spike_inds)] == 0
