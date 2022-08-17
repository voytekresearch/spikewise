"""Group polynomial fitting."""

from spikeparam.patch.fit import SpikeGroup

from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from spikeparam.patch.fit.poly import PolySpike



class PolySpikeGroup(PolySpike, SpikeGroup):

    def __init__(self, degree, knots=None, pad=None, sigma=None, fill=None,
                 window_length=(10., 10.), thresh_amp=-10., thresh_ms=1.0, pre_peak_ms=(-4., -1.),
                 pre_inflection_ms=1., smooth_frac=0.008, exp_shift_right=2.0, exp_duration=5.0,
                 corr_thresh=None):
        """Initialize object."""

        # Initalize super classes
        super(SpikeGroup, self).__init__()
        super(PolySpikeGroup, self).__init__(degree)

        # Poly settings
        self.degree = degree
        self.knots = knots
        self.pad = pad
        self.sigma = sigma

        if self.knots is None:
            self.knots = ['ramp_start', 'inflection', 'rise', 'peak',
                           'decay', 'tau', 'mtau', 'exp_end']

        # Repeat a single order
        if isinstance(self.degree, int):
            self.degree = np.tile(self.degree, len(self.knots)-1)

        if len(self.degree) != len(self.knots) - 1:
            raise ValueError("Orders must be one less then number of knots.")

        self.fill = fill

        # Super settings
        self.window_length = window_length
        self.thresh_amp = thresh_amp
        self.thresh_ms = thresh_ms

        self.pre_peak_ms = pre_peak_ms
        self.pre_inflection_ms = pre_inflection_ms
        self.smooth_frac = smooth_frac

        self.exp_shift_right = exp_shift_right
        self.exp_duration = exp_duration

        self.corr_thresh = corr_thresh

        # Poly results
        self.df_poly = None
        self.poly_coeffs = None
        self.poly_fit = None
        self.poly_r_squared = None


    def fit(self, sigs, fs, reader=None, peak_inds=None, gen_fits=True,
            gen_indices=True, max_gb=4, verbose=False, n_jobs=1, progress=None):
        """Fit the PolySpike object.

        Parameters
        ----------
        sigs : 1d or 2d array
            Alternative voltage time series if 2d.
            Indices to pass to reader if 1d.
        fs : float
            Sampling rate, in Hz.
        reader : function, optional, default: None
            Accepts sigs as the sole positional arguement and returns a 1d array.
        peak_inds : int or 1d array, optional, default: None
            Location of spike peaks, in samples. Bypasses spike detection.
            Use an int if the peak of the spike is in the same location.
            Use a 1d array for unique locations.
        gen_fit : bool, optional, default: True
            Generate fit arrays and r-squared values if True.
        gen_indices : bool, optional, default: True
            Generate sample indices of spike control knots if True.
        max_gb : float, optional, default: 4
            Maximum size of spike array, in gb.
        verbose : bool, optional, default: False
            Prints warnings if True.
        n_jobs : int, optional, 1
            Number of jobs to run in parallel.
            -1 default to cpu_count().
        progress : {tqdm.tqdm, tqdm.notebook.tqdm}
            Progress bar.
        """
        # Calls SpikeGroup's fit
        super(PolySpike, self).fit(sigs, fs, reader, peak_inds, gen_fits,
                                   gen_indices, max_gb, verbose, n_jobs, progress)

        # Calls PolySpike's fit
        super(PolySpikeGroup, self).fit(None, fs, peak_inds, gen_fits,
                                        gen_indices, n_jobs, progress)

