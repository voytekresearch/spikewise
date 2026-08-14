"""Test base Spike class."""

import pytest
import numpy as np
import warnings

from spikewise.patch.fit import Spike
from spikewise.tests.utils import plot_test, alt_func, pbar

@pytest.mark.parametrize('n_jobs', [1, 2])
def test_spike_fit(sim_patch_spikes, n_jobs):

    sig = sim_patch_spikes['sig_short']
    fs = sim_patch_spikes['fs']

    # Test fit
    if n_jobs == 2:
        sp = Spike()
        sp.fit(sig, fs, n_jobs=n_jobs, progress=pbar)
    else:
        sp = sim_patch_spikes['sp']

    spike_inds = sp.df_indices['peak'].values.copy()

    # Corr thresh
    _sp = Spike(corr_thresh=0)
    _sp.fit(sig, fs, spike_inds=spike_inds, n_jobs=n_jobs)

    # No super corr thresh
    with pytest.raises(ValueError):
        _sp = Spike(corr_thresh=1.1)
        _sp.fit(sig, fs, spike_inds=spike_inds, n_jobs=n_jobs)

    # Raise no peaks warning
    _sp = Spike()
    _sp.fit(sig, fs, spike_inds=[], n_jobs=n_jobs)

    # Fail all fits
    with pytest.raises(ValueError):
        _sp = Spike()
        _sp.fit(np.zeros(len(sig)), fs, spike_inds=spike_inds, n_jobs=n_jobs, verbose=True)

    # Fail some fits
    _peak = spike_inds[0]
    _sig = sig.copy()
    _sig[_peak-1000:_peak+1000] = 0

    _sp = Spike()
    _sp.fit(sig, fs, spike_inds=spike_inds, n_jobs=n_jobs, verbose=True)


@pytest.mark.parametrize('n_jobs', [1, 2])
def test_spike_alt(sim_patch_spikes, n_jobs):

    sig = sim_patch_spikes['sig_short']
    fs = sim_patch_spikes['fs']
    sp = sim_patch_spikes['sp']

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
    sp = sim_patch_spikes['sp']

    for func_args in [None, (1,), 1]:
        sp.alt(sig, fs, alt_func, func_args=func_args, n_jobs=n_jobs)


def test_spike_gen_fit(sim_patch_spikes):

    sp = sim_patch_spikes['sp']

    # Insert erroneous fit
    sp.inds_error = [0]
    sp.gen_fit()

    for i in ['fit_ramp', 'r_squared_ramp', 'fit_exp', 'r_squared_exp']:
        if i.startswith('fit'):
            assert np.isnan(getattr(sp, i)[0]).all()
        else:
            assert np.isnan(getattr(sp, i)[0])


def test_spike_filter_features(sim_patch_spikes):

    sp = sim_patch_spikes['sp']

    assert 'isi_prev' in sp.df_features.columns
    assert 'n_spikes_50ms' in sp.df_features.columns
    assert 'n_spikes_200ms' in sp.df_features.columns

    assert np.isnan(sp.isi_prev[0])
    assert sp.n_spikes_50ms[0] == 0
    assert sp.n_spikes_200ms[0] == 0

    filtered = sp.filter_features(inplace=False, min_inflection=-10000, max_inflection=10000)

    assert filtered is not None
    assert filtered.df_features is not None
    assert 'log_isi' in filtered.df_features.columns
    assert 'log_isi_prev' in filtered.df_features.columns
    assert 'n_spikes_50ms' in filtered.df_features.columns
    assert 'n_spikes_200ms' in filtered.df_features.columns
    assert 'r_squared_exp' not in filtered.df_features.columns
    assert 'r_squared_ramp' not in filtered.df_features.columns


def test_spike_fit_thresh_ms_uses_sampling_rate():

    fs = 50000
    sig = np.zeros(5000)
    peaks = [500, 540, 600, 680, 800, 1000]
    sig[peaks] = 10

    sp = Spike(thresh_amp=1, thresh_ms=1.0, window_length=(0.2, 0.2))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        with pytest.raises(ValueError):
            sp.fit(sig, fs, n_jobs=1)

    assert sp.spike_inds.tolist() == [540, 600, 680, 800, 1000]


@plot_test
@pytest.mark.parametrize('summary', [True, False])
def test_spike_plot(sim_patch_spikes, summary):

    sp = sim_patch_spikes['sp']

    sp.fit_ramp = None
    sp.fit_exp = None

    # Plot
    if summary:
        sp.plot_summary()
    else:
        sp.plot()
