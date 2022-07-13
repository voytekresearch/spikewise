"""Spike class."""

import warnings
from functools import partial
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spikeparam.patch.gen import gen_fit_ramp, gen_fit_exp
from spikeparam.patch.window import find_spike_times, window_spike
from spikeparam.patch.features import compute_features
from spikeparam.patch.plts import plot



class Spike:
    """Parametrize spike waveforms.

    Attributes
    ----------
    window_length : tuple of (float, float), optional, default: (10., 10.)
        Pre and post spike padding.
    thresh_mv : float, optional, default: -10
        Voltage threshold.
        Used in spike detection.
    thresh_ms : float, optional, default: 1.
        Minimum time between peaks, in ms.
        Used in spike detection.
    pre_peak_ms : tuple of (float, float), optional, default: (-4., -1.)
        Initial ramp window, relative to the max of the smoothed deriviate.
        Used to estimated the inflection point.
    pre_inflection_ms : float, optional, default:1.
        Time before the inflection point to define the ramp start.
    smooth_frac : float, optional, default: .008
        Smoothing fraction for the signal's derivative.
        Used to estimated the inflection point.
    exp_shift_right : float, optional, default: 2.
        Start time, in ms, to exponential start from peak.
    exp_duration : float, optional, default: 5.
        End time, in ms, of the exponential from the (shifted) peak.
    corr_thresh : float, optional, default: None
        Correlation coefficient threshold.
        Removes spikes with low mean correlation to other spikes.
    times : 1d array
        Time definition.
    spike_inds : 1d array
        Indices of spikes in sig.
    indices : 2d array
        Indices of control points per spike.
    poly_params : 2d array
        Polynomial parameters per spike.
    voltage_ramp : 1d array
        First polynomial parameter (e.g. offset) per spike.
    inflection_time : 1d array
        Time, in ms, of the inflection point per spike.
    inflection_mv : 1d array
        Voltage, in mv, at time of inflection per spike.
    peak_width : 1d array
        Width of peak, in ms, per spike.
    peak_sharpness : 1d array
        Sharpness of peak per spike.
    exp_amp : 1d array
        Exponential amplitude per spike.
    exp_lambda : 1d array
        Exponential decay per spike.
    exp_const : 1d array
        Exponential constant per spike.
    fit_ramp : 2d array
        Ramp fitted values per spike.
    fit_exp : 2d aray
        Exponential decay fitted values per spike.
    r_squared_ramp : 1d array
        Ramp r-squared per spike.
    r_squared_exp : 1d array
        Exponential r-squared per spike.
    fs : float
        Sampling rate, in Hz.
    inds_error : list of ind
        Spike indices of failed fits.
    df_features : pandas.DataFrame
        Waveform features per spike.
    """
    def __init__(self, window_length=(10., 10.), thresh_mv=-10., thresh_ms=1.0,
                 pre_peak_ms=(-4., -1.), pre_inflection_ms=1., smooth_frac=0.008,
                 poly_order=1, exp_shift_right=2.0, exp_duration=5.0, corr_thresh=None):

        # Settings
        self.window_length = window_length
        self.thresh_mv = thresh_mv
        self.thresh_ms = thresh_ms

        self.pre_peak_ms = pre_peak_ms
        self.pre_inflection_ms = pre_inflection_ms
        self.smooth_frac = smooth_frac
        self.poly_order = poly_order

        self.exp_shift_right = exp_shift_right
        self.exp_duration = exp_duration

        self.corr_thresh = corr_thresh

        # Results
        self.times = None
        self.spike_inds = None
        self.indices = None

        self.poly_params = None
        self.voltage_ramp = None
        self.inflection_time = None
        self.inflection_mv = None

        self.peak_width = None
        self.peak_sharpness = None

        self.exp_amp = None
        self.exp_lambda = None
        self.exp_const = None

        self.fit_ramp = None
        self.fit_exp = None

        self.r_squared_ramp = None
        self.r_squared_exp = None

        self.inds_error = None

        self.df_features = None


    def fit(self, sig, fs, gen_fits=True, n_jobs=1, progress=None):
        """Fit the 2d spike array.

        Parameters
        ----------
        sig : 1d array
            Voltage time series.
        fs : float
            Sampling rate, in Hz.
        gen_fit : bool, optional, default: True
            Generate fit arrays and r-squared values if True.
        n_jobs : int, optional, 1
            Number of jobs to run in parallel.
            -1 default to cpu_count().
        progress : {tqdm.tqdm, tqdm.notebook.tqdm}
            Progress bar.
        """
        self.fs = fs

        # Find spikes
        idx_spikes,  _= find_spike_times(sig, self.thresh_mv, self.thresh_ms * int(fs / 1000))
        self.spike_inds = idx_spikes

        # Get 2d array of spikes
        for i in range(len(idx_spikes)):

            spike = window_spike(sig, fs, idx_spikes[i],
                                 window_length=self.window_length)

            if i == 0:
                self.spikes = np.zeros((len(idx_spikes), len(spike)))

            self.spikes[i] = spike

        del sig

        # Remove outlier spikes
        if self.corr_thresh is not None:

            corrs = np.corrcoef(self.spikes)
            corrs = (corrs.sum(axis=0)-1) / (len(corrs)-1)

            inds = np.where(corrs > self.corr_thresh)[0]
            if len(inds) == 0:
                raise ValueError('No super-threshold spikes.')

            self.spikes = self.spikes[inds]

        # Initalize arrays
        self.indices = np.zeros((len(idx_spikes), 7), dtype=int)
        self.poly_params = np.zeros((len(idx_spikes), self.poly_order + 1))

        self.voltage_ramp = np.zeros(len(idx_spikes))
        self.inflection_time = np.zeros(len(idx_spikes))
        self.inflection_mv = np.zeros(len(idx_spikes))
        self.peak_width = np.zeros(len(idx_spikes))
        self.peak_sharpness = np.zeros(len(idx_spikes))

        self.exp_amp = np.zeros(len(idx_spikes))
        self.exp_lambda = np.zeros(len(idx_spikes))
        self.exp_const = np.zeros(len(idx_spikes))

        self.inds_error = []

        # Fit:
        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        # Collect kwargs
        kwargs = {
            'pre_peak_ms': self.pre_peak_ms,
            'pre_inflection_ms': self.pre_inflection_ms,
            'smooth_frac' : self.smooth_frac,
            'poly_order': self.poly_order,
            'exp_shift_right': self.exp_shift_right,
            'exp_duration': self.exp_duration
        }

        # In series
        if n_jobs == 1:

            for i in range(len(idx_spikes)):

                # Compute features
                indices, ramp_params, peak_params, exp_params = \
                    _compute_features(self.spikes[i], fs, **kwargs)

                # Unpack results
                if np.isnan(indices).any():
                    warnings.warn(f'Fail fit for spike: {i}')
                    self.indices[i] = [-999 for i in indices]
                    self.inds_error.append(i)
                else:
                    self.indices[i] = indices

                self.poly_params[i], self.voltage_ramp[i], self.inflection_time[i], \
                    self.inflection_mv[i] = ramp_params

                self.peak_width[i], self.peak_sharpness[i] = peak_params

                self.exp_amp[i], self.exp_lambda[i], self.exp_const[i] = exp_params

        # In parallel
        else:

            # Partial wrapper func
            pfunc = partial(_compute_features, fs=fs, **kwargs)

            # Run mp pool
            with Pool(processes=n_jobs) as pool:

                mapping = pool.imap(pfunc, self.spikes)

                if progress is None:
                    results = list(mapping)
                else:
                    results = list(progress(mapping, total=len(self.spikes)))

            # Unpack results
            for i in range(len(results)):

                indices, ramp_params, peak_params, exp_params = results[i]

                # Unpack results
                if np.isnan(indices).any():
                    warnings.warn(f'Fail fit for spike: {i}')
                    self.indices[i] = [-999 for i in indices]
                    self.inds_error.append(i)
                else:
                    self.indices[i] = indices

                self.poly_params[i], self.voltage_ramp[i], self.inflection_time[i], \
                    self.inflection_mv[i] = ramp_params

                self.peak_width[i], self.peak_sharpness[i] = peak_params

                self.exp_amp[i], self.exp_lambda[i], self.exp_const[i] = exp_params



        # Generate fits
        if gen_fits:
            self.gen_fit()

        # Generate the dataframe
        self.gen_df()


    def gen_fit(self, ramp=True, exp=True):
        """Generate arrays for ramp and exponential fits.

        Parameters
        ----------
        ramp : bool, optional, default: True
            Generate ramp fits if True.
        exp : bool, optional, default: True
            Generate exponential fits if True.
        """

        if self.times is None:
            self.times = np.arange(0, len(self.spikes[0])/self.fs, 1/self.fs)[:len(self.spikes[0])]

        for ind in range(len(self.spikes)):

            if ramp and ind not in self.inds_error:
                # Ramp
                start, end = self.indices[ind][0], self.indices[ind][1]

                _times = np.arange(end-start) * 1000 / self.fs

                _fit_ramp, _r2_ramp = gen_fit_ramp(_times, self.spikes[ind][start:end],
                                                   self.poly_params[ind])

                # Initalize arrays
                if self.fit_ramp is None:
                    self.fit_ramp = np.zeros((len(self.spikes), len(_fit_ramp)))
                    self.r_squared_ramp = np.zeros(len(self.spikes))

                # Store in attr
                self.fit_ramp[ind] = _fit_ramp
                self.r_squared_ramp[ind] = _r2_ramp

            if exp and ind not in self.inds_error:
                # Exponential decay
                start, end = self.indices[ind][-2], self.indices[ind][-1]

                _times = np.arange(end-start) * 1000 / self.fs

                _fit_exp, _r2_exp = gen_fit_exp(_times, self.spikes[ind][start:end],
                                                (self.exp_amp[ind], self.exp_lambda[ind], self.exp_const[ind]))

                # Initalize arrays
                if self.fit_exp is None:
                    self.fit_exp = np.zeros((len(self.spikes), len(_fit_exp)))
                    self.r_squared_exp = np.zeros(len(self.spikes))

                # Store in attr
                self.fit_exp[ind] = _fit_exp
                self.r_squared_exp[ind] = _r2_exp

        # Fill error fits with nans
        for ind in self.inds_error:

            self.fit_ramp[ind] = np.nan
            self.r_squared_ramp[ind] = np.nan
            self.fit_exp[ind] = np.nan
            self.r_squared_exp[ind] = np.nan


    def gen_df(self):
        """Generate feature dataframe."""

        columns = ['voltage_ramp', 'inflection_time', 'inflection_mv', 'peak_width',
                   'peak_sharpness', 'exp_amp', 'exp_lambda', 'exp_const']

        self.df_features = pd.DataFrame()

        for c in columns:
            self.df_features[c] = getattr(self, c)

        for _param in ['r_squared_ramp', 'r_squared_exp']:
            if hasattr(self, _param):
                self.df_features[_param] = getattr(self, _param)


    def plot(self, inds=None, mode='full', in_ms=True, show_points=False, ax=None):
        """Plot fits.

        Parameters
        ----------
        inds : int or list of ind
            Specific spikes to plot.
        mode : {'full', 'ramp', 'exp'}
            Plotting mode:

              - 'full' : plot the full spike and both fits.
              - 'ramp' : plot the ramping fit only.
              - 'exp'  : plot the exponential fit only.

        show_points : bool, optional, default: False
            Plots control points if True.
            Only used when mode is 'full'.
        ax : matplotlib AxesSubplot
            Axis to plot on.
        """
        # Generate fits if needed
        ramp = False
        exp = False

        if self.fit_ramp is None and mode in ['full', 'ramp']:
            ramp = True

        if self.fit_exp is None and mode in ['full', 'exp']:
            exp = True

        if ramp or exp:
            self.gen_fit(ramp, exp)

        # Plot
        plot(self, inds, mode, in_ms, show_points, ax)

    def plot_summary(self, axes=None):
        """Plot fit summary.

        Parameters
        ----------
        axes : list of ax
            Three axes to plot full, ramp, and exponential fits on.
        """
        if axes is None:
            fig = plt.figure(figsize=(12, 6), constrained_layout=True)

            spec = fig.add_gridspec(2, 2)

            ax0 = fig.add_subplot(spec[0, :])
            ax1 = fig.add_subplot(spec[1, 0])
            ax2 = fig.add_subplot(spec[1, 1])

            axes = [ax0, ax1, ax2]

        self.plot(ax=axes[0])
        self.plot(mode='ramp', ax=axes[1])
        self.plot(mode='exp', ax=axes[2])

        axes[0].set_title('Full Fit', size=18)
        axes[1].set_title('Ramp Fit', size=18)
        axes[2].set_title('Exponential Fit', size=18)


def _compute_features(spike, fs, **kwargs):
    """Wrapper function for compute_features."""
    poly_order = kwargs.pop('poly_order', 1)

    try:
        indices, ramp_params, peak_params, exp_params = \
            compute_features(spike, fs, poly_order=poly_order, **kwargs)
    except:

        indices = [np.nan] * 7

        ramp_params = [
            [np.nan] * (poly_order + 1),
            np.nan, np.nan, np.nan
        ]

        peak_params = [np.nan, np.nan]

        exp_params = [np.nan, np.nan, np.nan]

    return indices, ramp_params, peak_params, exp_params
