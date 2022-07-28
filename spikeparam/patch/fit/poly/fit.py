"""Fit a spline polynomial to spike waveforms."""


from spikeparam.patch.fit import Spike

from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from spikeparam.patch.features import compute_poly_features
from spikeparam.patch.sim.poly import sim_ppoly_dist


class PolySpike(Spike):
    """Polynomial Spike sub-class.

    Attributes
    ----------
    degree : int or list of int
        Polynomial order per segment. Should have length == len(knots) - 1 if list.
    knots : list of str, optional, default: None
        Points to compute polynomials between. Select from:
        {'ramp_start', 'inflection', 'rise', 'peak',
         'decay', 'tau', 'mtau', 'exp_end'}
        None uses all default knots.
    pad : int
        Pad samples around knots for re-weighted (via sigma)
        error in optimization.
    sigma :float
        Standard deviation of error. Adds perference for optimized
        fit around knots, +/- pad.
    df_poly : pandas.DataFrame
        Dataframe representation of poly_coeffs.
        Warning: This is in reverse from what np.poly1d expects. This reverse order is used
        for ease of comparison between parameters (i.e. the first coefficient will always be
        the constant).
    poly_knots : 2d array
        Polynomial knot locations, in samples.
    poly_coeffs : 2d array or list of 1d array
        Polynomial coefficients.
    poly_fit : 2d array
        Polynomial fit.
    poly_rsqs : 2d array
        R-squared for each spline.
    poly_rsq_full : 1d array
        R-squared for combined splines.
    **kwargs
        Additional settings passed to the Spike super class init.
    """
    def __init__(self, degree, knots=None, pad=None, sigma=None, fill=None,
                 window_length=(10., 10.), thresh_amp=-10.,  thresh_ms=1.0,
                 pre_peak_ms=(-4., -1.), pre_inflection_ms=1., smooth_frac=0.008,
                 exp_shift_right=2.0, exp_duration=5.0, corr_thresh=None):
        """Initialize object."""

        # Initalize super class
        super().__init__(self)

        # Poly settings
        self.degree = degree
        self.knots = knots
        self.pad = pad
        self.sigma = sigma

        # Default knots
        if self.knots is None:
            self.knots = ['ramp_start', 'inflection', 'rise', 'peak',
                          'decay', 'tau', 'mtau', 'exp_end']

        if len(self.degree) != len(self.knots) - 1:
            raise ValueError("Orders must be one less then number of knots.")

        # Repeat a single order
        if isinstance(self.degree, int):
            self.degree = np.tile(self.degree, len(self.knots)-1)

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
            Generate sample indices of spike control knots if True.
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

        # In series
        if n_jobs == 1:

            iterable = range(len(self.spikes))

            if progress is not None:
                iterable = progress(iterable, total=len(self.spikes), desc='PolySpike')

            results = []

            for ind in iterable:

                _inds, _params = _fit((self.spikes[ind], self.indices[ind]), degree=self.degree,
                                       points=self.knots, pad=self.pad, sigma=self.sigma,
                                       fill=self.fill, gen_fit=gen_fits)

                results.append([_inds, _params])

        # In parallel
        else:

            with Pool(processes=n_jobs) as pool:

                mapping = pool.imap(
                    partial(_fit, degree=self.degree, points=self.knots,
                            pad=self.pad, sigma=self.sigma, fill=self.fill, gen_fit=gen_fits),
                    zip(self.spikes, self.indices)
                )

                if progress is None:
                    results = list(mapping)
                else:
                    results = progress(list(mapping), total=len(self.spikes), desc='PolySpike')

        # Sort results
        self.poly_knots = np.array([i[0] for i in results])
        params = [i[1] for i in results]

        del results

        if gen_fits:
            self.poly_coeffs = np.array([i[0] for i in params])
            self.poly_fit = np.array([i[1] for i in params])
            self.poly_r_squared = np.array([i[2] for i in params])
        else:
            self.poly_coeffs = params

        # Generate a dataframe
        pos = 0

        self.df_poly = pd.DataFrame()

        x = 0
        for i in self.degree:
            y = 0
            _params = np.array([j[pos:pos+i+1] for j in self.poly_coeffs])
            pos += i
            for j in _params.T:
                self.df_poly[f'poly{str(x).zfill(2)}_c{str(y).zfill(2)}'] = j
                y += 1
            x += 1


    def simulate(self, n_sims, means=None, cov=None, cov_weight=1, seeds=None):

        if means is None or cov is None:

            indices = self.poly_knots.copy()
            coeffs = self.poly_coeffs.copy()

            # Get mean and cov
            params = np.column_stack((coeffs, indices))

            if means is None:
                means = np.mean(params, axis=0)

            if cov is None:
                cov = np.cov(params, rowvar=0) * cov_weight

        spikes, sim_coeffs, sim_indices = sim_ppoly_dist(means, cov, self.degree,
                                                         n_sims, seeds=seeds)

        self.sim_spikes = spikes
        self.sim_coeffs = sim_coeffs
        self.sim_indices = sim_indices


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
        colors = ['C' + str(i) for i in range(2, len(self.poly_knots[0]) + 2)]

        for ind in range(len(self.spikes)):

            _spike = self.spikes[ind]

            for cind, j in enumerate(self.poly_knots[ind]):
                plt.scatter(self.times[j], _spike[j], color=colors[cind], zorder=3)

        plt.legend()


def _poly_points(ys, knots):
    """Get spline locations.

    Parameters
    ----------
    ys : 1d array
        Spike waveform.
    knots : 1d array
        Point indices found by the Spike class.

    Returns
    -------
    knots : 1d array
        Updated spike knots.
    """

    select = [0, 1, 2, 3, 4, 6]

    inds = np.zeros(len(select) + 2, dtype=int)

    inds[:5] = knots[select[:5]]
    inds[-1] = knots[select[-1]]

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


def _fit(ys_inds, degree=None, points=None, pad=None, sigma=None,
         fill=None, gen_fit=None):
    """Proxy spike fit function."""

    ys, inds = ys_inds[0], ys_inds[1]

    # Get spline points
    knots = _poly_points(ys, inds)

    if points is not None:

        _inds = []

        names = ['ramp_start', 'inflection', 'rise', 'peak',
                 'decay', 'tau', 'mtau', 'exp_end']

        for i, name in enumerate(names):
            if name in points:
                _inds.append(i)

        knots = knots[_inds]

    # Compute features
    params = compute_poly_features(ys, knots, degree, pad, sigma, fill, gen_fit)

    return knots, params
