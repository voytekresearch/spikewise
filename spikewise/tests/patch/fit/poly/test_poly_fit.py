"""Test polynomial model."""

import pytest
import numpy as np

from spikewise.patch.fit import PolySpike
from spikewise.tests.utils import plot_test, pbar



def test_polyspike_init():

    with pytest.raises(ValueError):
        ps = PolySpike(knots=[0, 1, 2], degree=[1])

    ps = PolySpike(knots=[0, 1, 2], degree=1)


@pytest.mark.parametrize('n_jobs', [1, -1])
@pytest.mark.parametrize('use_pbar', [True, False])
@pytest.mark.parametrize('gen_fits', [True, False])
def test_polyspike_fit(sim_patch_spikes, check_param, n_jobs, use_pbar, gen_fits):

    sig = sim_patch_spikes['sig_short']
    fs = sim_patch_spikes['fs']

    progress = pbar if use_pbar else None

    ps = PolySpike(degree=[1, 2, 2, 2, 2, 2, 2], window_length=(4., 4.))
    ps.fit(sig, fs, n_jobs=n_jobs, progress=progress, gen_fits=gen_fits)

    if gen_fits:
        assert (ps.poly_r_squared > .5).all()

    assert ps.df_poly is not None


def test_polyspike_simulate(sim_patch_spikes, check_param):
    sig = sim_patch_spikes['sig_short']
    fs = sim_patch_spikes['fs']

    ps = PolySpike(degree=[1, 2, 2, 2, 2, 2, 2], window_length=(4., 4.))
    ps.fit(sig, fs, n_jobs=-1)
    ps.simulate(n_sims=2, seeds=np.arange(2))

    for s in ps.spikes:
        check_param(s, np.ndarray, (sig.min()-100, sig.max()+100))

    for c in ps.sim_coeffs:
        check_param(c, np.ndarray, (-np.inf, np.inf), 20)

    for k in ps.sim_knots:
        check_param(k, np.ndarray, (0, len(ps.spikes[0]) * 2), 8)


@plot_test
def test_spike_plot(sim_patch_spikes):

    sig = sim_patch_spikes['sig_short']
    fs = sim_patch_spikes['fs']

    ps = PolySpike(degree=[1, 2, 2, 2, 2, 2, 2], window_length=(4., 4.))
    ps.fit(sig, fs, n_jobs=-1)
    ps.plot()
