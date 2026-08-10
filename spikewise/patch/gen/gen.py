"""Generate fits."""

import numpy as np



def gen_fit_ramp(ramp_times, ramp, params):
    """Generate the ramp fit.

    Parameters
    ----------
    ramp_times : 1d array
        Time definition.
    ramp : 1d array
        Voltage definition.
    params : list of float
        Polynomial parameters.

    Returns
    -------
    ramp_fit : 1d array
        Ramp fitted values.
    ramp_r2 : float
        Fit r-squared.
    """
    pfunc = np.poly1d(params)
    ramp_fit = pfunc(ramp_times)
    ramp_r2 = np.corrcoef(ramp, ramp_fit)[0][1] ** 2

    return ramp_fit, ramp_r2


def gen_fit_exp(exp_times, exp, params):
    """Generate the exponential fit.

    Parameters
    ----------
    exp_times : 1d array
        Time definition.
    exp : 1d array
        Voltage definition.
    params : list of float
        Exponential decay parameters.

    Returns
    -------
    exp_fit : 1d array
        Ramp fitted values.
    exp_r2 : float
        Fit r-squared.
    """
    exp_fit = exp_func(exp_times, *params)
    exp_r2 = np.corrcoef(exp_fit, exp)[0][1] ** 2

    return exp_fit, exp_r2


def exp_func(times, exp_amp, exp_lambda, exp_const):
    """Exponential function.

    Parameters
    ----------
    times : 1d array
        Time definition.
    exp_amp : float
        Exponential amplitude.
    exp_lambda : float
        Exponential decay.
    exp_const : float
        Exponential constant.

    Returns
    -------
    1d array
        Voltage values.
    """
    return exp_amp * np.exp(-exp_lambda * times) + exp_const
