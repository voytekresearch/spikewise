"""Test within spike features."""

import numpy as np

from spikewise.patch.points import control_points
from spikewise.patch.features import (
    compute_features, compute_ramp_features, compute_peak_features,
    compute_decay_features, compute_poly_features
)



def test_compute_features(check_param, sim_patch_spikes):

    spikes = sim_patch_spikes['spikes']
    fs = sim_patch_spikes['fs']
    knots = sim_patch_spikes['knots']

    # Test all simulated spikes
    for ind in range(len(spikes)):

        spike = spikes[ind][~np.isnan(spikes[ind])]
        peak_ind = knots[ind][3]

        # Compute features
        indices, ramp_params, peak_params, exp_params = \
                compute_features(spike, fs, peak_ind=peak_ind, smooth_frac=.1)

        # Check indices
        check_param(indices, list, plen=7, prange=(0, len(spike)))
        assert (np.diff(indices[:5]) > 0).all()

        # Check ramp params
        poly_params, ramp_amp, inflection_time, inflection_amp = ramp_params

        check_param(poly_params, np.ndarray, plen=2)
        check_param(ramp_amp, float, prange=(spike.min(), spike.max()))
        check_param(inflection_time, float, prange=(0, (peak_ind/fs) * 1000))
        check_param(inflection_amp, float, prange=(spike.min(), spike.max()))

        # Check peak params
        peak_amp, peak_width, peak_sharpness = peak_params

        check_param(peak_amp, float, prange=(spike.min(), spike.max()))
        check_param(peak_width, float, prange=(0, len(spike)/fs*1000))
        check_param(peak_sharpness, float, prange=(0, spike.max()-spike.min()))

        # Check exp decay params
        exp_amp, exp_lambda, exp_const = exp_params
        check_param(exp_amp, float, prange=(spike.min(), spike.max()-spike.min()))
        check_param(exp_lambda, float, prange=(0, len(spike)))
        check_param(exp_const, float, prange=(-(spike.max()-spike.min()), spike.max()-spike.min()))


def test_compute_ramp_features(check_param, sim_patch_spikes):

    spikes = sim_patch_spikes['spikes']
    fs = sim_patch_spikes['fs']
    knots = sim_patch_spikes['knots']

    # Test all simulated spikes
    for ind in range(len(spikes)):

        spike = spikes[ind][~np.isnan(spikes[ind])]

        ramp_ind = knots[ind][0]
        infl_ind = knots[ind][1]
        peak_ind = knots[ind][3]

        poly_params, ramp_amp, inflection_time, inflection_amp = \
            compute_ramp_features(spike, fs, ramp_ind, infl_ind, peak_ind)

        check_param(poly_params, np.ndarray, plen=2)
        check_param(ramp_amp, float, prange=(spike.min(), spike.max()))
        check_param(inflection_time, float, prange=(0, (peak_ind/fs) * 1000))
        check_param(inflection_amp, float, prange=(spike.min(), spike.max()))


def test_compute_peak_features(check_param, sim_patch_spikes):

    spikes = sim_patch_spikes['spikes']
    fs = sim_patch_spikes['fs']
    knots = sim_patch_spikes['knots']

    # Test all simulated spikes
    for ind in range(len(spikes)):

        spike = spikes[ind][~np.isnan(spikes[ind])]

        rise_ind = knots[ind][2]
        peak_ind = knots[ind][3]
        decay_ind = knots[ind][4]

        peak_amp, peak_width, peak_sharpness = \
            compute_peak_features(spike, fs, rise_ind, peak_ind, decay_ind)

        check_param(peak_amp, float, prange=(spike.min(), spike.max()))
        check_param(peak_width, float, prange=(0, len(spike)/fs*1000))
        check_param(peak_sharpness, float, prange=(0, spike.max()-spike.min()))


def test_compute_decay_features(check_param, sim_patch_spikes):

    spikes = sim_patch_spikes['spikes']
    fs = sim_patch_spikes['fs']
    knots = sim_patch_spikes['knots']

    # Test all simulated spikes
    for ind in range(len(spikes)):

        spike = spikes[ind][~np.isnan(spikes[ind])]

        peak_ind = knots[ind][3]

        indices = control_points(spike, fs, peak_ind=peak_ind)

        exp_params = compute_decay_features(spike, fs, indices[5], indices[6])

        exp_amp, exp_lambda, exp_const = exp_params

        check_param(exp_amp, float, prange=(spike.min(), spike.max()-spike.min()))
        check_param(exp_lambda, float, prange=(0, len(spike)))
        check_param(exp_const, float, prange=(-(spike.max()-spike.min()), spike.max()-spike.min()))


def test_compute_poly_features(check_param, sim_patch_spikes):

    spikes = sim_patch_spikes['spikes']
    degree = sim_patch_spikes['degree']
    coeffs = sim_patch_spikes['coeffs']
    knots = sim_patch_spikes['knots']

    sigma = 1.

    # Test all simulated spikes
    for ind in range(len(spikes)):

        spike = spikes[ind][~np.isnan(spikes[ind])]

        _knots = knots[ind]

        _coeffs, ys_fit, r_squared = \
                compute_poly_features(spike, _knots, degree, gen_fit=True, sigma=sigma)

        check_param(ys_fit, np.ndarray, plen=len(spike))
        check_param(r_squared, float, prange=(-1, 1))
        check_param(_coeffs, np.ndarray, plen=len(coeffs[ind]))

        mae = np.abs(_coeffs-coeffs[ind]).mean()
        assert mae < 1e-6
