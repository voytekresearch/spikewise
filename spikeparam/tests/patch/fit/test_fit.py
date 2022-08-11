"""Test base Spike class."""

from tabnanny import verbose
import pytest
import numpy as np

from spikeparam.patch.fit import Spike
from spikeparam.tests.utils import plot_test, alt_func, pbar

@pytest.mark.parametrize('n_jobs', [1, 2])
def test_spike_fit(sim_patch_spikes, n_jobs):

    sig = sim_patch_spikes['sig']
    fs = sim_patch_spikes['fs']

    # Get two spikes
    sig = sig[10000:20000].copy()
    sig = np.pad(sig, 5000, constant_values=(sig[0], sig[-1]))

    # Test fit
    sp = Spike()
    sp.fit(sig, fs, n_jobs=n_jobs)
    peak_inds = sp.df_indices['peak'].values.copy()

    # Corr thresh
    _sp = Spike(corr_thresh=0)
    _sp.fit(sig, fs, peak_inds=peak_inds, n_jobs=n_jobs)

    # No super corr thresh
    with pytest.raises(ValueError):
        _sp = Spike(corr_thresh=1.1)
        _sp.fit(sig, fs, peak_inds=peak_inds, n_jobs=n_jobs)

    # Raise no peaks warning
    _sp = Spike()
    _sp.fit(sig, fs, peak_inds=[], n_jobs=n_jobs)

    # Fail all fits
    with pytest.raises(ValueError):
        _sp = Spike()
        _sp.fit(np.zeros(len(sig)), fs, peak_inds=peak_inds, n_jobs=n_jobs, verbose=True)

    # Fail some fits
    _peak = peak_inds[0]
    _sig = sig.copy()
    _sig[_peak-1000:_peak+1000] = 0

    _sp = Spike()
    _sp.fit(sig, fs, peak_inds=peak_inds, n_jobs=n_jobs, verbose=True)

    _sp = Spike()
    _sp.fit(sig, fs, peak_inds=peak_inds, n_jobs=n_jobs, progress=pbar)


@pytest.mark.parametrize('n_jobs', [1, 2])
def test_spike_alt(sim_patch_spikes, n_jobs):

    sig = sim_patch_spikes['sig']
    fs = sim_patch_spikes['fs']

    # Get two spikes
    sig = sig[10000:20000].copy()
    sig = np.pad(sig, 5000, constant_values=(sig[0], sig[-1]))

    # Fit
    sp = Spike()
    sp.fit(sig, fs, n_jobs=n_jobs)

    param_keys = ['alt_peak_amp']

    sp.alt(sig, fs, alt_func, func_args=(1,), n_jobs=n_jobs,
           param_keys=param_keys, progress=pbar)

    assert (sp.peak_amp == sp.alt_peak_amp).all()

    # Queue
    sp = Spike()
    sp.alt(sig, fs, alt_func, func_kwargs=dict(weight0=1),
           param_keys=param_keys, n_jobs=n_jobs, queue=True)
    sp.fit(sig, fs, n_jobs=n_jobs)

    assert (sp.peak_amp == sp.alt_peak_amp).all()

    # Type check various args
    for func_args in [None, (1,), 1]:
        sp = Spike()
        sp.fit(sig, fs, n_jobs=n_jobs)
        sp.alt(sig, fs, alt_func, func_args=func_args, n_jobs=n_jobs)


def test_spike_gen_fit(sim_patch_spikes):

    sig = sim_patch_spikes['sig']
    fs = sim_patch_spikes['fs']

    # Get two spikes
    sig = sig[10000:20000].copy()
    sig = np.pad(sig, 5000, constant_values=(sig[0], sig[-1]))

    # Test fit
    sp = Spike()
    sp.fit(sig, fs, n_jobs=-1)

    # Insert erroneous fit
    sp.inds_error = [0]
    sp.gen_fit()

    for i in ['fit_ramp', 'r_squared_ramp', 'fit_exp', 'r_squared_exp']:
        if i.startswith('fit'):
            assert np.isnan(getattr(sp, i)[0]).all()
        else:
            assert np.isnan(getattr(sp, i)[0])


@plot_test
@pytest.mark.parametrize('summary', [True, False])
def test_spike_plot(sim_patch_spikes, summary):

    sig = sim_patch_spikes['sig']
    fs = sim_patch_spikes['fs']

    # Get two spikes
    sig = sig[10000:20000].copy()
    sig = np.pad(sig, 5000, constant_values=(sig[0], sig[-1]))

    # Test fit
    sp = Spike()
    sp.fit(sig, fs, n_jobs=-1)

    sp.fit_ramp = None
    sp.fit_exp = None

    # Plot
    if summary:
        sp.plot_summary()
    else:
        sp.plot()
