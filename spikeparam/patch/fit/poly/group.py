"""Group polynomial fitting."""

from spikeparam.patch.fit import SpikeGroup

from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from spikeparam.patch.fit.poly.fit import _fit



class PolySpikeGroup(SpikeGroup):

    def __init__(self,  orders, points=None, fill=None, window_length=(10., 10.), thresh_amp=-10.,
                 thresh_ms=1.0, pre_peak_ms=(-4., -1.), pre_inflection_ms=1., smooth_frac=0.008,
                 poly_order=1, exp_shift_right=2.0, exp_duration=5.0, corr_thresh=None):
        """Initialize object."""

        # Initalize super class
        super().__init__(self)

        # Poly settings
        self.orders = orders
        self.points = points
        self.fill = fill

        if self.points is None:
            self.points = ['ramp_start', 'inflection', 'rise', 'peak',
                           'decay', 'tau', 'mtau', 'exp_end']

        if len(self.orders) != len(self.points) - 1:
            raise ValueError("Orders must be one less then number of points.")

        # Super settings
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

        # Poly results
        self.df_poly = None
        self.poly_coeffs = None
        self.poly_fit = None
        self.poly_rsqs = None
        self.poly_rsq_full = None



    def fit(self, sigs, fs, reader=None, peak_inds=None, gen_fits=True,
            gen_indices=True, low_mem=False, verbose=False, n_jobs=1, progress=None):
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
            Generate sample indices of spike control points if True.
        low_mem : bool, optional, default: False
            Lowers memory usage at cost of increased runtime from
            repeat storage access.
        verbose : bool, optional, default: False
            Prints warnings if True.
        n_jobs : int, optional, 1
            Number of jobs to run in parallel.
            -1 default to cpu_count().
        progress : {tqdm.tqdm, tqdm.notebook.tqdm}
            Progress bar.
        """

        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        super().fit(sigs, fs, reader, peak_inds, gen_fits,
                    gen_indices, low_mem, verbose, n_jobs, progress)

        with Pool(processes=n_jobs) as pool:

            mapping = pool.imap(
                partial(_fit, orders=self.orders, points=self.points,
                        fill=self.fill, gen_fit=self.gen_fit),
                zip(self.spikes, self.indices)
            )

            if progress is None:
                results = list(mapping)
            else:
                results = progress(list(mapping), total=len(self.spikes))

        self.poly_indices = np.array([i[0] for i in results])
        params = [i[1] for i in results]

        del results

        if all([self.orders[0] == i for i in self.orders[1:]]):
            self.poly_coeffs = np.array([i[0] for i in params])
        else:
            self.poly_coeffs = [i[0] for i in params]

        self.poly_fit = np.array([i[1] for i in params])
        self.poly_rsqs = np.array([i[2] for i in params])
        self.poly_rsq_full = np.array([i[3] for i in params])

        del params

        # Create dataframe
        self.df_poly = pd.DataFrame()

        for i in range(len(self.points)-1):

            _coeffs = np.array([arr[i] for arr in self.poly_coeffs])

            for ind in range(len(_coeffs[0])):
                self.df_poly[f'poly{str(i).zfill(2)}_c{ind}'] = _coeffs[:, ind]


    def plot(self):
        """Plot the polynomial fit."""

        plt.figure(figsize=(10, 4))

        # True spike
        for i, s in enumerate(self.spikes):
            label = 'True' if i == 0 else ''
            plt.plot(self.times, s, color='C0', label=label)

        # Spike fit
        for i, p in enumerate(self.poly_fit):
            label = 'Fit' if i == 0 else ''
            plt.plot(self.times, p, color='C1', ls='--', label=label)

        # Spline points
        colors = ['C' + str(i) for i in range(2, len(self.poly_indices[0]) + 2)]

        for ind in range(len(self.spikes)):

            _spike = self.spikes[ind]

            for cind, j in enumerate(self.poly_indices[ind]):
                plt.scatter(self.times[j], _spike[j], color=colors[cind], zorder=3)

        plt.legend()
