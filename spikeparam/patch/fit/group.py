import warnings

import numpy as np

from spikeparam.patch.window import find_spike_times, window_spike
from spikeparam.patch.fit import Spike


class SpikeGroup(Spike):

    def __init__(self, window_length=(10., 10.), thresh_amp=-10., thresh_ms=1.0,
                 pre_peak_ms=(-4., -1.), pre_inflection_ms=1., smooth_frac=0.008,
                 poly_order=1, exp_shift_right=2.0, exp_duration=5.0, corr_thresh=None):

        # Initalize super class
        super().__init__(self)

        # Settings
        self.window_length = window_length
        self.thresh_amp = thresh_amp
        self.thresh_ms = thresh_ms

        self.pre_peak_ms = pre_peak_ms
        self.pre_inflection_ms = pre_inflection_ms
        self.smooth_frac = smooth_frac
        self.poly_order = poly_order

        self.exp_shift_right = exp_shift_right
        self.exp_duration = exp_duration

        self.corr_thresh = corr_thresh


    def fit(self, sigs, fs, reader=None, gen_fits=True, gen_indices=True,
            low_mem=False, n_jobs=1, progress=None):
        """Fit the 2d spike array.

        Parameters
        ----------
        sigs : 2d array
            Voltage time series.
        fs : float
            Sampling rate, in Hz.
        gen_fit : bool, optional, default: True
            Generate fit arrays and r-squared values if True.
        gen_indices : bool, optional, default: True
            Generate sample indices of spike control points if True.
        low_mem : bool, optional, default: False
            Lowers memory usage at cost of increased runtime from
            repeat storage access.
        n_jobs : int, optional, 1
            Number of jobs to run in parallel.
            -1 default to cpu_count().
        progress : {tqdm.tqdm, tqdm.notebook.tqdm}
            Progress bar.
        """

        # Infer required shape
        n_sigs = len(sigs)

        self.spike_inds = []

        spikes = []

        for ind in range(n_sigs):

            # Read in signal
            if reader is not None:
                sig = reader(sigs[ind])

            # Find spikes
            idx_spikes,  _= find_spike_times(sig, self.thresh_amp, self.thresh_ms * int(fs / 1000))

            if len(idx_spikes) == 0:
                warnings.warn('No spikes detected.')
                self.spike_inds.append(None)
            else:
                self.spike_inds.append(idx_spikes)

            # Non-low memory mode:
            #   Store arrays to list, and then vstack them
            if not low_mem and self.spike_inds[-1] is not None:
                spikes.append(
                    window_spike(sig, fs, self.spike_inds[ind],
                                 window_length=self.window_length)
                )

        # Infer number of spikes
        self.n_spikes = sum([len(s) for s in self.spike_inds if s is not None])

        # Stack arrays
        if not low_mem:
            self.spikes = np.vstack(spikes)

        # Low-memory routine
        if low_mem:

            pos = 0
            for ind in range(n_sigs):

                if self.spike_inds[ind] is None:
                    continue

                if reader is not None:
                    sig = reader(sigs[ind])

                spikes = window_spike(sig, fs, self.spike_inds[ind],
                                      window_length=self.window_length)

                if self.spikes is None:
                    self.spikes = np.zeros((self.n_spikes, len(spikes[0])))

                self.spikes[pos:pos+len(spikes)] = spikes

                pos += len(spikes)

        # Track which spike belongs to which signal
        self.group = np.zeros(self.n_spikes, dtype=int)

        # Drop None spikes inds and track groups
        spike_inds = np.zeros(self.n_spikes, dtype=int)

        pos = 0
        group = 0

        for inds in self.spike_inds:

            if inds is None:
                continue

            spike_inds[pos:pos+len(inds)] = inds

            self.group[pos:pos+len(inds)] = group

            pos += len(inds)
            group += 1

        self.spike_inds = spike_inds

        # Call super's fit method
        super().fit(None, fs, gen_fits, gen_indices, True, n_jobs=n_jobs, progress=progress)
