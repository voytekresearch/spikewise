"""Test base Spike class."""

from tabnanny import verbose
import pytest
import numpy as np

from spikeparam.patch.fit import Spike


def test_spike_fit(sim_patch_spikes):

    sig = sim_patch_spikes['sig']
    fs = sim_patch_spikes['fs']

    # Get two spikes
    sig = sig[10000:20000].copy()
    sig = np.pad(sig, 5000, constant_values=(sig[0], sig[-1]))

    # 1 cpu
    sp = Spike()
    sp.fit(sig, fs, n_jobs=1)

    # Mutliproc
    sp = Spike()
    sp.fit(sig, fs, n_jobs=-1)
    peak_inds = sp.df_indices['peak'].values.copy()

    # Corr thresh
    _sp = Spike(corr_thresh=0)
    _sp.fit(sig, fs, peak_inds=peak_inds, n_jobs=-1)

    # No super corr thresh
    with pytest.raises(ValueError):
        _sp = Spike(corr_thresh=1.1)
        _sp.fit(sig, fs, peak_inds=peak_inds, n_jobs=-1)

    # Raise no peaks warning
    _sp = Spike()
    _sp.fit(sig, fs, peak_inds=[], n_jobs=-1)

    # Fail all fits
    with pytest.raises(ValueError):
        sp = Spike()
        _sp.fit(np.zeros(len(sig)), fs, peak_inds=peak_inds, n_jobs=1, verbose=True)

        _sp = Spike()
        _sp.fit(np.zeros(len(sig)), fs, peak_inds=peak_inds, n_jobs=-1, verbose=True)

    # Fail some fits
    _peak = peak_inds[0]
    _sig = sig.copy()
    _sig[_peak-1000:_peak+1000] = 0

    sp = Spike()
    _sp.fit(sig, fs, peak_inds=peak_inds, n_jobs=1, verbose=True)
    _sp.fit(sig, fs, peak_inds=peak_inds, n_jobs=-1, verbose=True)
