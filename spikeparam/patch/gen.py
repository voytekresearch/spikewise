"""Generate fits."""

import numpy as np


def gen_fit_ramp(ramp_times, ramp, params):

    pfunc = np.poly1d(params)
    ramp_fit = pfunc(ramp_times)
    ramp_r2 = np.corrcoef(ramp, ramp_fit)[0][1] ** 2

    return ramp_fit, ramp_r2

def gen_fit_exp(exp_times, exp, params):

    exp_fit = exp_func(exp_times, *params)
    exp_r2 = np.corrcoef(exp_fit, exp)[0][1] ** 2

    return exp_fit, exp_r2


def exp_func(times, exp_amp, exp_lambda, exp_timeshift, exp_const):
    """Exponential function."""
    return exp_amp * np.exp(-exp_lambda * (times - exp_timeshift)) + exp_const
