"""SpikeGroup class."""

import warnings

import numpy as np

from spikeparam.patch.window import find_spike_times, window_spike
from spikeparam.patch.fit import Spike


class SpikeGroup(Spike):

    def __init__(self, window_length=(10., 10.), thresh_amp=-10., thresh_ms=1.0,
                 pre_peak_ms=(-4., -1.), pre_inflection_ms=1., smooth_frac=0.008,
                 poly_order=1, exp_shift_right=2.0, exp_duration=5.0, corr_thresh=None):

        # Initalize super class
        super().__init__()

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


    def fit(self, sigs, fs, reader=None, spike_inds=None, gen_fits=True,
            gen_indices=True, max_gb=4, verbose=False, n_jobs=1, progress=None):
        """Fit the 2d spike array.

        Parameters
        ----------
        sigs : 2d array
            Voltage time series.
        fs : float
            Sampling rate, in Hz.
        reader : function, optional, default: None
            Accepts sigs as the sole positional arguement and returns a 1d array.
        spike_inds : 2d list or ragged array, optional, default: None
            Location of spike peaks, in samples. Bypasses spike detection.
        gen_fit : bool, optional, default: True
            Generate fit arrays and r-squared values if True.
        gen_indices : bool, optional, default: True
            Generate sample indices of spike control points if True.
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

        # Infer required shape
        n_sigs = len(sigs)

        n_spikes = np.array([0] * n_sigs)

        self.spike_inds = []

        # Pad around detected peak to find absolute peak
        pad = int(self.thresh_ms * fs / 1000)

        # Initalize spike array
        n_samples = int(
            (fs / 1000 * self.window_length[0]) + (fs / 1000 * self.window_length[1]) + 1
        )

        max_n_spikes = int(np.floor((max_gb * 1e9) / (8*n_samples)))

        self.spikes = np.zeros((max_n_spikes, n_samples))

        i = 0

        for ind in range(n_sigs):

            # Read in signal
            if reader is not None:
                sig = reader(sigs[ind])
            else:
                sig = sigs[ind]

            # Peak detection
            if spike_inds is None:

                # Find spikes
                _spike_inds,  _= find_spike_times(sig, self.thresh_amp, pad)

                if len(_spike_inds) == 0 :
                    if verbose:
                        warnings.warn(f'No spikes detected for spike: {ind}.')
                    continue

                # Ensure spike peak is the abs max
                starts = _spike_inds - pad//2
                ends = _spike_inds + pad//2

                for _ind in range(len(_spike_inds)):
                    _spike_inds[_ind] = int(starts[_ind] +
                                            np.argmax(sig[starts[_ind]:ends[_ind]]))

            else:
                _spike_inds = spike_inds[ind]

            # Ensure full windows can be created around spikes
            inds = np.where((_spike_inds >= 0) & (_spike_inds < len(sig)))[0]

            # At least 1 spike is found
            _n_spikes = len(inds)

            if  _n_spikes > 0:

                n_spikes[ind] =  _n_spikes

                self.spike_inds.append(_spike_inds)

                # Break if size of spikes exceeds max_gb
                try:
                    self.spikes[i:i+_n_spikes] =  window_spike(
                        sig, fs, _spike_inds[inds], window_length=self.window_length)
                except ValueError:
                    warnings.warn('Number of spikes exceeds allocated array size. Increase max_gb.')
                    break

                i += _n_spikes

        if i == 0:
            raise ValueError(
                'No spikes detected. Check thresh_amp and thresh_ms initalization settings.'
            )

        # Remove excess of the initalize array
        self.spikes = self.spikes[:i]

        # Track which spike belongs to which signal
        self.group = np.zeros(i, dtype=int)

        j = 0
        for i, n in enumerate(n_spikes):
            self.group[j:j+n] = i
            j += n

        # Call super's fit method
        self.n_spikes = sum(n_spikes)

        super().fit(None, fs, None, gen_fits, gen_indices, True,
                    n_jobs=n_jobs, progress=progress)

        # Update n_spikes attr
        self.n_spikes = n_spikes

        # Run alts
        if self.queue_group is not None:

            for locs in self.queue_group:

                args = [locs[k] for k in locs if k in ['sigs', 'fs', 'func']]
                kwargs =  {k:locs[k] for k in locs if k not in ['self', 'sigs', 'fs',
                                                                'func', '__class__']}

                self.alt(*args, **kwargs)


    def alt(self, sigs, fs, func, func_args=None, func_kwargs=None, reader=None, param_keys=None,
            ref='peak', window_length=(10., 10.), pre_windowed=False, n_jobs=1, progress=None,
            queue=False):
        """Compute features for an alternative/associated signal.

        Parameters
        ----------
        sigs : 1d or 2d array
            Alternative voltage time series if 2d.
            Indices to pass to reader if 1d.
        fs : float
            Alternative sampling rate, in Hz.
        func : function
            Computes features for each window. Each object returned should be {float, int, str}.
        func_args : tuple, optional, default: None
            Arguments to pass to func.
        func_kwargs : dict, optional, default: None
            Keyword arguments to pass to func.
        reader : function, optional, default: None
            Accepts sigs as the sole positional arguement and returns a 1d array.
        param_keys : list of str
            Names of features returned from func.
            These names become columns appended to df_features.
        ref : {'ramp_start', 'inflection', 'rise', 'peak', 'decay', 'exp_start', 'exp_end'}
            Reference used to create windows.
        window_length : tuple of (float, float)
            Number of milliseconds before and after the reference point to include.
        pre_windowed : bool, optional, default: False
            Sigs is assumed to already be windowed if True.
        n_jobs : int, optional, 1
            Number of jobs to run in parallel.
            -1 default to cpu_count().
        progress : {tqdm.tqdm, tqdm.notebook.tqdm}
            Progress bar.
        queue : bool, optional, default: False
            Queues method call to be executed when .fit is called.
        """
        # Queue call to be executed on .fit
        if queue:
            self.queue_group = [] if self.queue is None else self.queue

            _queue = {k: v for k, v in locals().items() if k != 'self'}
            _queue['queue'] = False

            self.queue_group.append(_queue)

            return

        self.alt_windows = None
        pos = 0

        if pre_windowed:
            # Signal is already windowed
            self.alt_windows = sigs
        else:
            # Stack alt signal into a 2d array
            for ind in range(len(sigs)):

                inds = np.where(self.df_features['group'].values == ind)[0]

                if len(inds) == 0:
                    continue

                # Read in signal
                if reader is not None:
                    sig = reader(sigs[ind])
                else:
                    sig = sigs[ind]

                alt_windows = window_spike(sig, fs, self.df_indices.iloc[inds][ref].values,
                                           window_length=window_length)

                if self.alt_windows is None:
                    n_nans = np.count_nonzero(np.isnan(self.df_features['peak_amp'].values))

                    self.alt_windows = np.zeros((len(self.df_features) - n_nans,
                                                 len(alt_windows[0])))

                self.alt_windows[pos:pos+len(alt_windows)] = alt_windows

                pos += len(alt_windows)

        super().alt(None, fs, func, func_args, func_kwargs, param_keys, ref, window_length,
                    True, n_jobs, progress, queue)
