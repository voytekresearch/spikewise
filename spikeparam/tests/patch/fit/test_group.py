"""Test SpikeGroup class."""

import pytest
from functools import partial
import numpy as np

from spikeparam.patch.fit import SpikeGroup
from spikeparam.tests.utils import reader, alt_func, reader


@pytest.mark.parametrize('use_reader', [True, False])
def test_spikegroup_fit(sim_patch_spikes, use_reader):

    sig = sim_patch_spikes['sig']
    fs = sim_patch_spikes['fs']

    # Get two spikes
    sig = sig[10000:20000].copy()
    sig = np.pad(sig, 5000, constant_values=(sig[0], sig[-1]))

    # Create 2d copy
    sigs = np.array([sig, sig])

    sg = SpikeGroup()

    if use_reader:
        sg.fit(sigs, fs, n_jobs=-1, reader=reader)
    else:
        sg.fit(sigs, fs, n_jobs=-1)

    # Bypass spike detection
    spike_inds = sg.spike_inds
    sg = SpikeGroup()
    sg.fit(sigs, fs, spike_inds=spike_inds, n_jobs=-1)

    # No peaks found
    with pytest.raises(ValueError):
        sg = SpikeGroup(thresh_amp=np.inf)
        sg.fit(sigs, fs, n_jobs=-1, verbose=True)

    # Small max_gb
    with pytest.raises(ValueError):
        sg = SpikeGroup()
        sg.fit(sigs, fs, n_jobs=-1, max_gb=1e-9)


def test_spikegroup_alt(sim_patch_spikes):

    sig = sim_patch_spikes['sig']
    fs = sim_patch_spikes['fs']

    # Get two spikes
    sig = sig[10000:20000].copy()
    sig = np.pad(sig, 5000, constant_values=(sig[0], sig[-1]))

    # Create 2d copy
    sigs = np.array([sig, sig])

    # Without queue
    sg = SpikeGroup()
    sg.fit(sigs, fs, n_jobs=-1)
    sg.alt(sigs, fs, alt_func, param_keys=['alt_peak_amp'])

    # With queue
    _sg = SpikeGroup()
    _sg.alt(sigs, fs, alt_func, param_keys=['alt_peak_amp'], queue=True, reader=reader)
    _sg.fit(sigs, fs, n_jobs=-1)

    for i in sg.df_features.keys():
        if i != 'isi':
            # Don't test isi since it contains nans
            assert (sg.df_features[i] == _sg.df_features[i]).all()

    # Pre-windowed
    __sg = SpikeGroup()
    __sg.fit(sigs.copy(), fs, n_jobs=1)
    __sg.alt(sg.spikes.copy(), fs, alt_func, param_keys=['alt_peak_amp'], pre_windowed=True)

    for i in _sg.df_features.keys():
        if i != 'isi':
            # Don't test isi since it contains nans
            assert (_sg.df_features[i] == __sg.df_features[i]).all()
