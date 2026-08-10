"""Test generating fits."""

import numpy as np
from spikewise.patch.gen import gen_fit_ramp, gen_fit_exp, exp_func


def test_gen_fit_ramp(check_param):
    sig_lin = np.arange(100)
    params = np.polyfit(sig_lin, sig_lin, 1)

    ramp_fit, ramp_r2 = gen_fit_ramp(sig_lin, sig_lin, params)

    check_param(ramp_fit, np.ndarray, (sig_lin[0]-1, sig_lin[-1]+1), len(sig_lin))
    check_param(ramp_r2, float, (.8, 1))


def test_gen_fit_exp(check_param):

    times = np.arange(100)

    exp = exp_func(times, 1, .1, 0)

    exp_fit, exp_r2 = gen_fit_exp(times, exp, (1, .1, 0))

    check_param(exp_fit, np.ndarray, (0, 1), len(exp))
    check_param(exp_r2, float, (.9, 1))

    assert (exp == exp_fit).all()
