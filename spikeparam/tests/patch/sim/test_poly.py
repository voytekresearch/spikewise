"""Test polynomial simulations."""

import pytest
import numpy as np


from spikeparam.patch.sim import sim_ppoly, sim_ppoly_dist


def test_sim_ppoly():

    # Simulate a piecewise line; y=x
    xs = np.arange(100)
    ys = sim_ppoly(xs, np.array([0, 50, 100]), np.array([1, 0, 1, 0]), degree=1)

    assert (ys == xs).all()

    # Knots out-of-order
    with pytest.raises(ValueError):
        sim_ppoly(xs, np.array([0, 2, 1]), np.array([1, 0, 1, 0]), degree=1)



def test_sim_ppoly_dist():

    np.random.seed(0)

    intercepts = np.random.uniform(-1, 1, 100)
    slopes = np.random.uniform(.5, 1.5, 100)
    params = np.array([slopes, intercepts, slopes, intercepts])

    means = np.append(np.mean(params, axis=1), [0, 50, 100])

    cov = np.zeros((7, 7))
    cov[:4, :4] = np.cov(params, rowvar=1)

    spikes, sim_coeffs, sim_knots = sim_ppoly_dist(means, cov, np.array([1, 1]),
                                                   10, seeds=np.arange(10))

    for s in spikes:

        # Check monotonic
        assert (np.diff(s) > 0).all()

        # Check slope
        assert .4 <= (s[-1]-s[0]) / 100 <= 1.6

        # Check offset
        assert -1.1 <= s[0] <= 1.1


    assert .4 <= sim_coeffs[[0, 2]].mean() <= 1.6
    assert -1.1 <= sim_coeffs[[1, 3]].mean() <= 1.1
    assert (sim_knots == [0, 50, 100]).all()
