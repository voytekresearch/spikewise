"""Test poly group fitting."""


import pytest
import numpy as np

from spikeparam.patch.fit import PolySpikeGroup
from spikeparam.tests.utils import plot_test, alt_func, pbar


def test_polyspikegroup_init():

    with pytest.raises(ValueError):
        ps = PolySpikeGroup(knots=[0, 1, 2], degree=[1])

    ps = PolySpikeGroup(knots=[0, 1, 2], degree=1)


def test_polyspikegroup_fit(sim_patch_spikes):

    sig = sim_patch_spikes['sig_short']
    sig = np.array([sig, sig])
    fs = sim_patch_spikes['fs']

    psg = PolySpikeGroup(degree=[1, 2, 2, 2, 2, 2, 2], window_length=(4., 4.))
    psg.fit(sig, fs)

    assert (psg.poly_r_squared > .5).all()
    assert psg.df_poly is not None
