"""Fit a spline polynomial to spike waveforms."""


from spikeparam.patch.fit import Spike

from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from spikeparam.patch.features import compute_poly_features



class PolySpike(Spike):
    """Polynomial Spike sub-class.

    Attributes
    ----------
    orders : list of int
        Polynomial order per segment. Should have length == len(points) - 1.
    points : list of str, optional, default: None
        Points to compute polynomials between. Select from:
        {'ramp_start', 'inflection', 'rise', 'peak',
         'decay', 'tau', 'mtau', 'exp_end'}
        None defaults to all points.
    poly_coeffs : 2d array or list of 1d array
        Polynomial coefficients, in increasing order.
        Warning: This is in reverse from what np.poly1d expects. This reverse order is used
        for ease of comparison between parameters (i.e. the first coefficient will always be
        the constant).
    poly_fit : 2d array
        Polynomial fit.
    poly_rsqs : 2d array
        R-squared for each spline.
    poly_rsq_full : 1d array
        R-squared for combined splines.
    **kwargs
        Additional settings passed to the Spike super class init.
    """
    def __init__(self,  orders, points=None, fill=None, window_length=(10., 10.), thresh_amp=-10.,
                 thresh_ms=1.0, pre_peak_ms=(-4., -1.), pre_inflection_ms=1., smooth_frac=0.008,
                 poly_order=1, exp_shift_right=2.0, exp_duration=5.0, corr_thresh=None):
        """Initialize object."""

        # Initalize super class
        super().__init__(self)

        # Poly settings
        self.orders = orders
        self.points = points

        if self.points is None:
            self.points = ['ramp_start', 'inflection', 'rise', 'peak',
                           'decay', 'tau', 'mtau', 'exp_end']

        if len(self.orders) != len(self.points) - 1:
            raise ValueError("Orders must be one less then number of points.")

        self.fill = fill

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


    def fit(self, sig, fs, peak_inds=None, gen_fits=True,
            gen_indices=True, n_jobs=1, progress=None):
        """Fit the PolySpike object.

        Parameters
        ----------
        sig : 1d array
            Voltage time series.
        fs : float
            Sampling rate, in Hz.
        reader : function, optional, default: None
            Accepts sigs as the sole positional arguement and returns a 1d array.
        peak_inds : int or 1d array, optional, default: None
            Location of spike peaks, in samples. Bypasses spike detection.
            Use an int if the peak of the spike is in the same location.
            Use a 1d array for unique locations.
        gen_fits : bool, optional, default: True
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

        if sig is not None:
            super().fit(sig, fs, peak_inds, gen_fits, gen_indices, preload=False,
                        n_jobs=n_jobs, progress=progress)

        with Pool(processes=n_jobs) as pool:

            mapping = pool.imap(
                partial(_fit, orders=self.orders, points=self.points,
                        fill=self.fill, gen_fit=gen_fits),
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
            self.poly_coeffs = np.array([i[0][::-1] for i in params])
        else:
            self.poly_coeffs = [i[0][::-1] for i in params]

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


def _poly_points(ys, spike_inds):
    """Get spline locations.

    Parameters
    ----------
    ys : 1d array
        Spike waveform.
    spike_inds : 1d array
        Point indices found by the Spike class.

    Returns
    -------
    inds : 1d array
        Updated spike indices.
    """

    select = [0, 1, 2, 3, 4, 6]

    inds = np.zeros(len(select) + 2, dtype=int)

    inds[:5] = spike_inds[select[:5]]
    inds[-1] = spike_inds[select[-1]]

    _ys = ys.copy()[inds[4]:]
    _ys -= _ys.min()

    tau_ind = np.where(_ys <= _ys[0] * np.exp(-1))[0]

    if len(tau_ind) == 0:
        tau_ind = len(ys)
    else:
        tau_ind = tau_ind[0] + inds[4]

    inds[-3] = tau_ind

    inds[-2] = tau_ind + int(inds[-1] - tau_ind)//2

    return inds


def _fit(ys_inds, orders=None, points=None, fill=None, gen_fit=None):
    """Proxy spike fit function."""

    ys, inds = ys_inds[0], ys_inds[1]

    # Get spline points
    inds = _poly_points(ys, inds)

    if points is not None:

        _inds = []

        names = ['ramp_start', 'inflection', 'rise', 'peak',
                 'decay', 'tau', 'mtau', 'exp_end']

        for i, name in enumerate(names):
            if name in points:
                _inds.append(i)

        inds = inds[_inds]

    # Compute features
    params = compute_poly_features(ys, inds, orders, fill, gen_fit)

    return inds, params
