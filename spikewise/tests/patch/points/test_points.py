"""Test control point functions."""

import numpy as np

from spikewise.patch.points import control_points, inflection, diff_spike


def test_control_points(sim_patch_spikes):

    spike = sim_patch_spikes['spikes'][0]
    spike = spike[~np.isnan(spike)]

    fs = sim_patch_spikes['fs']

    indices = control_points(spike, fs)

    assert indices[0] < indices[1] < indices[2] < indices[3] < indices[4], indices[6]

    # idx_exp_start may not be monotonic (i.e. occuring before or after idx_decay)
    assert indices[3] < indices[5] < indices[6]


def test_inflection(sim_patch_spikes, check_param):

    spike = sim_patch_spikes['spikes'][0]
    spike = spike[~np.isnan(spike)]

    fs = sim_patch_spikes['fs']

    # Smoothed derivative
    times = np.arange(0, len(spike)/fs, 1/fs)[:len(spike)]
    _, d_smoothed_spike = diff_spike(times, spike, .008)

    # Get peak
    idx_peak = np.argmax(spike)

    pre_peak_ms=(-4., -1.)

    # Get ramping start and inflection
    idx_ramp_start, idx_inflection = inflection(d_smoothed_spike, fs, idx_peak,
                                                pre_peak_ms, 1.)

    check_param(idx_ramp_start, int, (0, idx_inflection))
    check_param(idx_inflection, int, (idx_ramp_start, len(spike)))


def test_diff_spike(sim_patch_spikes, check_param):

    spike = sim_patch_spikes['spikes'][0]
    spike = spike[~np.isnan(spike)]

    fs = sim_patch_spikes['fs']

    # Smoothed derivative
    times = np.arange(0, len(spike)/fs, 1/fs)[:len(spike)]
    d_smoothed_times, d_smoothed_spike = diff_spike(times, spike, .008)

    check_param(d_smoothed_times, np.ndarray, (times[0], times[-1]), len(times)-1)
    check_param(d_smoothed_spike, np.ndarray, (spike.min() - 10, spike.max()+10), len(times)-1)
