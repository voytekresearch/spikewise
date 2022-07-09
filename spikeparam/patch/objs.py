"""Spike class."""

import numpy as np
import pandas as pd

from spikeparam.patch.window import find_spike_times, window_spike
from spikeparam.patch.points import control_points
from spikeparam.patch.utils import create_times
from spikeparam.patch.features import (
    compute_ramp_features, compute_decay_features, compute_peak_features
)


class Spike:

    def __init__(self, window_length=(10, 10), thresh_mv=-10, thresh_ms=1.0,
                 thresh_zscore=40.0, smooth_frac=0.008, poly_order=1,
                 exp_shift_right=2.0, exp_duration=5.0):

        self.window_length = window_length
        self.thresh_mv = thresh_mv
        self.thresh_ms = thresh_ms
        self.thresh_zscore = thresh_zscore
        self.smooth_frac = smooth_frac
        self.poly_order = poly_order
        self.exp_shift_right = exp_shift_right
        self.exp_duration = exp_duration

    def fit(self, times, sig, fs, n_jobs=1):

        idx_spikes,  _= find_spike_times(sig, self.thresh_mv, self.thresh_ms)

        if n_jobs == 1:

            # Initalize arrays
            self.indices = np.zeros((len(idx_spikes), 7), dtype=int)
            self.poly_params = np.zeros((len(idx_spikes), self.poly_order + 1))

            self.voltage_ramp = np.zeros(len(idx_spikes))
            self.inflection_time = np.zeros(len(idx_spikes))
            self.inflection_mv = np.zeros(len(idx_spikes))
            self.peak_width = np.zeros(len(idx_spikes))
            self.peak_sharpness = np.zeros(len(idx_spikes))
            self.exp_params = np.zeros((len(idx_spikes), 4))

            self.exp_amp = np.zeros(len(idx_spikes))
            self.exp_lambda = np.zeros(len(idx_spikes))
            self.exp_timeshift = np.zeros(len(idx_spikes))
            self.exp_const = np.zeros(len(idx_spikes))


            for i in range(len(idx_spikes)):

                # Window
                spike, spike_times = window_spike(sig, times, fs, idx_spikes[i],
                                                  window_length=self.window_length)
                # Control points
                self.indices[i] = control_points(spike_times, spike, fs, thresh_ms=1,
                                          thresh_zscore=40., smooth_frac=.008)

                # Unpack indices
                idx_ramp_start, idx_inflection, idx_rise, \
                        idx_peak, idx_decay, idx_exp_start, idx_exp_end = self.indices[i]

                # Ramp features
                _ramp_params = compute_ramp_features(
                        spike_times, spike, fs, idx_ramp_start, idx_inflection, idx_peak)

                self.poly_params[i], self.voltage_ramp[i], self.inflection_time[i], \
                    self.inflection_mv[i] = _ramp_params

                # Peak features
                self.peak_width[i], self.peak_sharpness[i] = \
                    compute_peak_features(spike, fs, idx_decay, idx_peak)

                # Exponential decay features
                exp_params= compute_decay_features(spike_times, spike, idx_exp_start, idx_exp_end)

                self.exp_amp[i], self.exp_lambda[i], self.exp_timeshift[i], self.exp_const[i] = exp_params

            # Generate the dataframe
            self.gen_df()


    def gen_df(self):

        columns = ['voltage_ramp', 'inflection_time', 'inflection_mv', 'peak_width', 'peak_sharpness',
                   'exp_amp', 'exp_lambda', 'exp_timeshift', 'exp_const']

        self.df = pd.DataFrame()

        for c in columns:
            self.df[c] = getattr(self, c)