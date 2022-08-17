"""Test plotting functions."""

import pytest
import numpy as np

from spikeparam.patch.plts import plot_model
from spikeparam.tests.utils import plot_test
from spikeparam.patch.fit import Spike



@plot_test
@pytest.mark.parametrize('mode', ['full', 'ramp', 'exp'])
@pytest.mark.parametrize('inds', [None, 0, [0, 1], [0, 'null']])
def test_plot_model(sim_patch_spikes, mode, inds):

    sp = sim_patch_spikes['sp']

    if isinstance(inds, list) and inds[1] == 'null':
        inds = [0, 1]
        sp.inds_error = [1]

    plot_model(sp, mode=mode, inds=inds, show_points=True)
