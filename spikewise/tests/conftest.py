"""Configuration file for pytest for spikewise."""

import os

import pytest

import numpy as np

from spikewise.gaussian import Spikes
from spikewise.gaussian.models.features.gaussians import _sim_ap_cycle
from spikewise.gaussian.models.cyclepoints import compute_spike_cyclepoints

from spikewise.patch.fit import Spike
from spikewise.patch.sim import sim_ppoly_dist, sim_patch
from spikewise.tests.utils import DATA_DIR



@pytest.fixture(scope='module')
def sim_patch_spikes():

    # Load param distribution
    fs = 200000
    poly_means = np.load(os.path.join(DATA_DIR, 'pvc-6_param_means.npy'))
    poly_cov = np.load(os.path.join(DATA_DIR, 'pvc-6_param_cov.npy'))
    degree = np.load(os.path.join(DATA_DIR, 'pvc-6_degree.npy'))

    # Simulate
    spikes, coeffs, knots = sim_ppoly_dist(poly_means, poly_cov, degree,
                                           20, seeds=np.arange(20))

    knot_keys = ['ramp_start', 'inflection', 'rise', 'peak',
                 'decay', 'tau', 'mtau', 'exp_end']

    # Define isi
    isi = np.random.exponential(scale=(fs / 1000) * 50, size=len(spikes)-1).astype(int)

    isi += 1000 # min refactory period

    sig = sim_patch(spikes, isi, 2500, pad=fs//20)

    # Get two spikes
    sig_short = sig[10000:20000].copy()
    sig_short = np.pad(sig_short, 5000, constant_values=(sig_short[0], sig_short[-1]))

    # Fit
    sp = Spike()
    sp.fit(sig_short, fs, n_jobs=-1)

    yield {'sig': sig, 'sig_short': sig_short, 'sp': sp, 'spikes': spikes, 'degree': degree,
           'coeffs': coeffs, 'knots': knots, 'knot_keys': knot_keys, 'fs': fs}


@pytest.fixture(scope='module')
def sim_spikes():

    # Simulate 5, 3-gaussian spikes
    cycle_params = {'centers': (.4, .5, .6), 'stds': (.1, .1, .1),
                    'alphas': (-1, 0, 1), 'heights': (10, -30, 20)}

    spike = _sim_ap_cycle(1, 100, **cycle_params)

    sig = np.zeros(1100)

    starts = np.arange(100, 1100, 200)
    ends = starts + 100

    for start in starts:
        sig[start:start+100] = spike

    # Pad edges
    pad = 500
    sig = np.pad(sig, pad)
    starts += pad

    # Simulate overlapping spikes
    cycle_params = {'centers': (.25, .5, .75), 'stds': (.05, .05, .05),
                    'alphas': (0, 0, 0), 'heights': (-14, -30, -14)}

    spike_overlap = _sim_ap_cycle(1, 100, **cycle_params)

    sig_overlap = np.zeros_like(sig)

    for start in starts:
        sig_overlap[start:start+100] = spike_overlap

    # Simulate prunable spikes
    cycle_params = {'centers': (.25, .5, .75), 'stds': (.05, .05, .05),
                    'alphas': (0, 0, 0), 'heights': (-30, -20, -30)}

    spike_prune = _sim_ap_cycle(1, 100, **cycle_params)

    sig_prune = np.zeros_like(sig)

    for start in starts:
        sig_prune[start:start+100] = spike_prune

    # Simulate Na current
    spike_na = _sim_ap_cycle(1, 100, .5, .1, 0, -20)

    sig_na = np.zeros_like(sig)

    for start in starts:
        sig_na[start:start+100] = spike_na

    # Simulate Na+K current
    cycle_params = {'centers': (.4, .6), 'stds': (.1, .1),
                    'alphas': (0, .2), 'heights': (-30, 15)}

    spike_na_k = _sim_ap_cycle(1, 100, **cycle_params)

    sig_na_k = np.zeros_like(sig)

    for start in starts:
        sig_na_k[start:start+100] = spike_na_k

    # Simulate Na+Conductive current
    cycle_params = {'centers': (.4, .5), 'stds': (.1, .1),
                    'alphas': (0, 0), 'heights': (15, -30)}

    spike_na_cond = _sim_ap_cycle(1, 100, **cycle_params)

    sig_na_cond = np.zeros_like(sig)

    for start in starts:
        sig_na_cond [start:start+100] = spike_na_cond

    yield {'sig': sig, 'sig_overlap': sig_overlap, 'sig_prune': sig_prune, 'sig_na': sig_na,
           'sig_na_k': sig_na_k, 'sig_na_cond': sig_na_cond, 'fs': 20000, 'f_range': (500, 3000),
           'spike':spike, 'locs':(starts, ends)}


@pytest.fixture(scope='module')
def sim_spikes_df(sim_spikes):

    sig = sim_spikes['sig']
    fs = sim_spikes['fs']
    f_range = sim_spikes['f_range']

    df_samples = compute_spike_cyclepoints(sig, fs, f_range, std=2)

    yield {'df_samples': df_samples}


@pytest.fixture(scope='module')
def sim_spikes_fit(sim_spikes):

    sig = sim_spikes['sig']
    fs = sim_spikes['fs']
    f_range = sim_spikes['f_range']

    spikes = Spikes()

    spikes.fit(sig, fs, f_range, n_gaussians=3, tol=1e-3)

    return {'spikes': spikes}


@pytest.fixture
def check_param():

    def check(param, ptype, prange=None, plen=None):

        assert param is not None
        assert isinstance(param, ptype)

        is_iterable = isinstance(param, (list, tuple, dict, np.ndarray))

        if prange is not None:

            if prange[0] is not None and not is_iterable:
                assert (param >= prange[0])
            else:
                for p in param:
                    assert (p >= prange[0])

            if prange[1] is not None and not is_iterable:
                assert (param <= prange[1])
            else:
                for p in param:
                    assert (p >= prange[0])

        if plen is not None:
            assert len(param) == plen

    return check
