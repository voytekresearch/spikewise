"""Plotting functions."""

from .gen import gen_fit_exp, gen_fit_ramp

import matpplotlib.pyplot as plt


def plot_control_points(spike, indices):

    plt.plot(spike)

    labels = ['Start', 'Inflection', 'Rise', 'Peak', 'Decay']

    for i, l in zip(indices, labels):
        plt.plot(i, spike[i], ls='', marker='.', ms=16, label=l)

    plt.legend()

    plt.ylabel('Voltage')
    plt.xlabel('Samples')


def plot_fit_ramp(times, spike, ramp_slice, poly_params):

    fit_ramp, _ = gen_fit_ramp(times[ramp_slice], spike[ramp_slice], poly_params)

    plt.plot(times[ramp_slice], spike[ramp_slice], label='Actual')
    plt.plot(times[ramp_slice], fit_ramp, label='Fit')

    plt.ylabel('Voltage')
    plt.xlabel('Samples')


def plot_fit_decay(times, spike, idx_exp_start, idx_exp_end, exp_params):

    fit_decay, _ = gen_fit_exp(times[idx_exp_start:idx_exp_end], exp_params)

    plt.plot(spike[idx_exp_start:idx_exp_end], label='Actual')
    plt.plot(fit_decay, label='Fit')

    plt.ylabel('Voltage')
    plt.xlabel('Samples')
