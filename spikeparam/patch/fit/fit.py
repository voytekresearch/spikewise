"""Spike class."""

import os
import warnings
import inspect
import re
from functools import partial
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from spikeparam.patch.gen import gen_fit_ramp, gen_fit_exp
from spikeparam.patch.window import find_spike_times, window_spike
from spikeparam.patch.features import compute_features, compute_isi
from spikeparam.patch.plts import plot_model

class Spike:
    """Parametrize spike waveforms."""
    
    def __init__(self, window_length=(10., 10.), thresh_amp=-10., thresh_ms=1.0,
                 pre_peak_ms=(-4., -1.), pre_inflection_ms=1., smooth_frac=0.008,
                 poly_order=1, exp_shift_right=2.0, exp_duration=5.0, corr_thresh=None):
        
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

        # Filtering defaults
        self.default_filter_params = {
            'min_inflection': 0,
            'max_inflection': 2,
            'min_r2_exp': 0.5,
            'min_r2_ramp': 0.1,
            'log_isi': True,
            'drop_r_squared': True,
            'replace_inf': True
        }

        # Data storage
        self.spikes = None
        self.n_spikes = None
        self.times = None
        self.spike_inds = None
        self.indices = None
        self.ramp_poly_params = None
        self.ramp_amp = None
        self.inflection_time = None
        self.inflection_amp = None
        self.peak_amp = None
        self.peak_width = None
        self.peak_sharpness = None
        self.exp_amp = None
        self.exp_lambda = None
        self.exp_const = None
        self.fit_ramp = None
        self.fit_exp = None
        self.r_squared_ramp = None
        self.r_squared_exp = None
        self.isi = None
        self.inds_error = []
        self.df_features = None
        self.df_indices = None
        self.fs = None
        self.group = None
        self.queue = None
        self.queue_group = None
        self._filtered = False  

    def filter_features(self, inplace=True, **kwargs):
        """Filter spikes with single-use protection."""
        if self._filtered:
            print("Filtering already applied. Create new instance with inplace=False to re-filter.")
            return None if inplace else self
        
        # --- Original filtering logic ---
        params = {**self.default_filter_params, **kwargs}
        
        valid_mask = (
            (self.df_features['inflection_time'] > params['min_inflection']) &
            (self.df_features['inflection_time'] < params['max_inflection']) &
            (self.df_features['r_squared_exp'] >= params['min_r2_exp']) &
            (self.df_features['r_squared_ramp'] >= params['min_r2_ramp'])
        )
        valid_indices = self.df_features.index[valid_mask]

        if len(valid_indices) == 0:
            warnings.warn("No spikes remaining after filtering")
            if inplace:
                self._reset_attributes()
                return None
            return self._create_empty_instance()

        df_filtered = self._process_dataframe(valid_indices, params)
        filtered_attributes = self._get_filtered_attributes(valid_indices)
        # --- End original logic ---

        if inplace:
            self._update_instance(df_filtered, filtered_attributes, valid_indices)
            self._filtered = True  # Mark as filtered
            return None
        else:
            new_sp = self._create_filtered_instance(df_filtered, filtered_attributes)
            new_sp._filtered = True  # New instance also marked
            return new_sp

    def _process_dataframe(self, valid_indices, params):
        """Apply transformations with column safety checks."""
        df = self.df_features.loc[valid_indices].copy()
        
        # 1. Handle ISI logging only if 'isi' exists and not already logged
        if params['log_isi']:
            if 'isi' in df.columns:
                df['log_isi'] = np.log10(df['isi'])
                df.drop('isi', axis=1, inplace=True)
            elif 'log_isi' not in df.columns:
                warnings.warn("Cannot log ISI - 'isi' column missing")
        
        # 2. Drop R² columns only if they exist
        if params['drop_r_squared']:
            cols_to_drop = []
            if 'r_squared_exp' in df.columns:
                cols_to_drop.append('r_squared_exp')
            if 'r_squared_ramp' in df.columns:
                cols_to_drop.append('r_squared_ramp')
            if cols_to_drop:
                df.drop(cols_to_drop, axis=1, inplace=True)
        
        # 3. Replace infs if requested
        if params['replace_inf']:
            df = df.replace([np.inf, -np.inf], np.nan)
        
        return df

    def _get_filtered_attributes(self, valid_indices):
        """Slice all spike-related arrays."""
        return {
        # Add indices to filtered attributes
        'indices': self.indices[valid_indices],  
        'spikes': self.spikes[valid_indices],
        'spike_inds': self.spike_inds[valid_indices],
            'spikes': self.spikes[valid_indices],
            'spike_inds': self.spike_inds[valid_indices],
            'ramp_poly_params': self.ramp_poly_params[valid_indices],
            'ramp_amp': self.ramp_amp[valid_indices],
            'inflection_time': self.inflection_time[valid_indices],
            'inflection_amp': self.inflection_amp[valid_indices],
            'peak_amp': self.peak_amp[valid_indices],
            'peak_width': self.peak_width[valid_indices],
            'peak_sharpness': self.peak_sharpness[valid_indices],
            'exp_amp': self.exp_amp[valid_indices],
            'exp_lambda': self.exp_lambda[valid_indices],
            'exp_const': self.exp_const[valid_indices],
            'isi': self.isi[valid_indices],
            'fit_ramp': self.fit_ramp[valid_indices] if self.fit_ramp is not None else None,
            'fit_exp': self.fit_exp[valid_indices] if self.fit_exp is not None else None,
            'r_squared_ramp': self.r_squared_ramp[valid_indices] if self.r_squared_ramp is not None else None,
            'r_squared_exp': self.r_squared_exp[valid_indices] if self.r_squared_exp is not None else None,
            'inds_error': self._map_error_indices(valid_indices)
        }

    def _map_error_indices(self, valid_indices):
        """Convert original error indices to new positions."""
        index_map = {orig: new for new, orig in enumerate(valid_indices)}
        return [index_map[i] for i in self.inds_error if i in index_map]

    def _update_instance(self, df_filtered, filtered_attributes, valid_indices):
        """Update current instance with filtered data."""
        # Update core attributes
        for attr, value in filtered_attributes.items():
            setattr(self, attr, value)
            
        self.n_spikes = len(self.spikes)
        self.df_features = df_filtered
        self.gen_df_indices()
        
        # Reset derived data
        self.times = self._generate_times()

    def _create_filtered_instance(self, df_filtered, filtered_attributes):
        """Create new Spike instance with filtered data."""
        # Create new instance with same parameters
        new_sp = Spike(
            window_length=self.window_length,
            thresh_amp=self.thresh_amp,
            thresh_ms=self.thresh_ms,
            pre_peak_ms=self.pre_peak_ms,
            pre_inflection_ms=self.pre_inflection_ms,
            smooth_frac=self.smooth_frac,
            poly_order=self.poly_order,
            exp_shift_right=self.exp_shift_right,
            exp_duration=self.exp_duration,
            corr_thresh=self.corr_thresh
        )
        
        # Set filtered attributes
        new_sp.fs = self.fs
        new_sp.times = self.times.copy() if self.times is not None else None
        new_sp.group = self.group
        new_sp.queue = self.queue.copy() if self.queue is not None else None
        new_sp.queue_group = self.queue_group.copy() if self.queue_group is not None else None
        
        for attr, value in filtered_attributes.items():
            setattr(new_sp, attr, value)
            
        new_sp.n_spikes = len(new_sp.spikes)
        new_sp.df_features = df_filtered
        new_sp.gen_df_indices()
        
        return new_sp

    def _reset_attributes(self):
        """Reset all data attributes to initial state."""
        self.spikes = None
        self.n_spikes = 0
        self.spike_inds = np.array([])
        self.indices = None
        self.ramp_poly_params = None
        self.ramp_amp = None
        self.inflection_time = None
        self.inflection_amp = None
        self.peak_amp = None
        self.peak_width = None
        self.peak_sharpness = None
        self.exp_amp = None
        self.exp_lambda = None
        self.exp_const = None
        self.fit_ramp = None
        self.fit_exp = None
        self.r_squared_ramp = None
        self.r_squared_exp = None
        self.isi = None
        self.inds_error = []
        self.df_features = None
        self.df_indices = None

    def _create_empty_instance(self):
        """Create empty instance with matching parameters."""
        new_sp = Spike(
            window_length=self.window_length,
            thresh_amp=self.thresh_amp,
            thresh_ms=self.thresh_ms,
            pre_peak_ms=self.pre_peak_ms,
            pre_inflection_ms=self.pre_inflection_ms,
            smooth_frac=self.smooth_frac,
            poly_order=self.poly_order,
            exp_shift_right=self.exp_shift_right,
            exp_duration=self.exp_duration,
            corr_thresh=self.corr_thresh
        )
        new_sp.fs = self.fs
        return new_sp

    def _generate_times(self):
        """Regenerate time vector after filtering."""
        if self.spikes is None or len(self.spikes) == 0:
            return None
            
        return np.arange(0, len(self.spikes[0])/self.fs, 1/self.fs)[:len(self.spikes[0])]

    # Keep existing methods (fit, gen_fit, gen_df_features, etc.) unchanged below
    # [Previous fit(), gen_fit(), gen_df_features(), plot(), etc. methods here]

    def gen_df_indices(self):
        """Generate sample indices dataframe."""
        columns = ['ramp_start', 'inflection', 'rise', 
                   'peak', 'decay', 'exp_start', 'exp_end']
        
        if self.indices is None or len(self.indices) == 0:
            self.df_indices = pd.DataFrame(columns=columns)
            return
        
        # Use filtered indices
        ref_inds = self.indices[:, 3]  # Peak indices
        
        # Handle group vs single signal
        if isinstance(self.spike_inds, list) and self.spike_inds[0].ndim == 1:
            _spike_inds = np.concatenate(self.spike_inds)
        else:
            _spike_inds = self.spike_inds
        
        self.df_indices = pd.DataFrame()
        
        # Calculate absolute sample indices
        for col, inds in zip(columns, self.indices.T):
            self.df_indices[col] = _spike_inds + (inds - ref_inds)

    def fit(self, sig, fs, spike_inds=None, gen_fits=True, gen_indices=True,
            preload=False, verbose=False, n_jobs=1, progress=None, flip_signal=False):
        """Fit the 2d spike array.

        Parameters
        ----------
        sig : 1d array
            Voltage time series.
        fs : float
            Sampling rate, in Hz.
        spike_inds : int or 1d array, optional, default: None
            Location of spike peaks, in samples. Bypasses spike detection.
        gen_fit : bool, optional, default: True
            Generate fit arrays and r-squared values if True.
        gen_indices : bool, optional, default: True
            Generate sample indices of spike control points if True.
        preload : bool, optional, default: False
            If True, self.spikes and self.spike_inds have been set,
            ignoring the sig argument. Used for group sub-class.
        verbose : bool, optional, default: False
            Prints warnings if True.
        n_jobs : int, optional, 1
            Number of jobs to run in parallel.
            -1 default to cpu_count().
        progress : {tqdm.tqdm, tqdm.notebook.tqdm}
            Progress bar.

        flip_signal : {True, False}, optional, default: False
            Signal polarity handling:
            - True: Force signal inversion
            - False: Use original polarity
        """
        self.fs = fs
        
        

        if not preload:

            # ================== SIGNAL FLIPPING ================== 
            if flip_signal:
                sig = -sig  # Invert signal if needed
            # =====================================================
            
                # Find spikes
            if spike_inds is None:

                self.spike_inds,  _= find_spike_times(sig, self.thresh_amp,  self.thresh_ms*1000)
                
                # Ensure true max (take abs max around 20% of spike around peak)
                pad = int(sum(self.window_length) * fs / 1000) + 1
                pad = pad // 5

                starts = self.spike_inds - pad//2
                ends = self.spike_inds + pad//2

                for ind in range(len(self.spike_inds)):
                   self.spike_inds[ind] = starts[ind] + np.argmax(sig[starts[ind]:ends[ind]])

                #remove spike duplicates 
                duplicate_spike_times = pd.Series(self.spike_inds).duplicated(keep='first')
                # Remove duplicate spike times from the spike time array
                unique_spike_timestamps = [time for i, time in enumerate(self.spike_inds) if not duplicate_spike_times[i]]
                self.spike_inds = np.asarray(unique_spike_timestamps)
              
                   


            elif isinstance(spike_inds, (np.ndarray, list, int, np.int64)):
                self.spike_inds = spike_inds

            if len(self.spike_inds) == 0:
                warnings.warn('No spikes detected.')
                return

            


            # Get 2d array of spikes
            self.spikes = window_spike(sig, fs, self.spike_inds,
                                       window_length=self.window_length)

            del sig

        self.n_spikes = len(self.spikes)

        # Remove outlier spikes
        if self.corr_thresh is not None:

            corrs = np.corrcoef(self.spikes)
            corrs = (corrs.sum(axis=0)-1) / (len(corrs)-1)

            inds = np.where(corrs > self.corr_thresh)[0]
            if len(inds) == 0:
                raise ValueError('No super-threshold spikes.')

            self.spikes = self.spikes[inds]



        

        # Initalize arrays
        self.indices = np.zeros((self.n_spikes, 7), dtype=int)
        self.ramp_poly_params = np.zeros((self.n_spikes, self.poly_order + 1))

        self.ramp_amp = np.zeros(self.n_spikes)
        self.inflection_time = np.zeros(self.n_spikes)
        self.inflection_amp = np.zeros(self.n_spikes)

        self.peak_amp = np.zeros(self.n_spikes)
        self.peak_width = np.zeros(self.n_spikes)
        self.peak_sharpness = np.zeros(self.n_spikes)

        self.exp_amp = np.zeros(self.n_spikes)
        self.exp_lambda = np.zeros(self.n_spikes)
        self.exp_const = np.zeros(self.n_spikes)

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
            'exp_duration': self.exp_duration,
            'peak_ind': int(self.window_length[0] * fs / 1000),
            'verbose': verbose
        }

        # In series
        if n_jobs == 1:

            iterable = range(self.n_spikes)

            if progress is not None:
                iterable = progress(iterable, total=self.n_spikes, desc='Spike')

            for i in iterable:
                print(i)

                # Compute features
                indices, ramp_params, peak_params, exp_params = \
                    _compute_features(self.spikes[i], fs, **kwargs)

                # Unpack results
                if np.isnan(indices).any() or any([i < 0 for i in indices]):

                    if verbose:
                        warnings.warn(f'Fail fit for spike: {i}')

                    self.indices[i] = [-999 for i in indices]
                    self.inds_error.append(i)

                else:
                    self.indices[i] = indices

                self.ramp_poly_params[i], self.ramp_amp[i], self.inflection_time[i], \
                    self.inflection_amp[i] = ramp_params

                self.peak_amp[i], self.peak_width[i], self.peak_sharpness[i] = peak_params

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
                    results = list(progress(mapping, total=len(self.spikes), desc='Spike'))

            # Unpack results
            for i in range(len(results)):

                indices, ramp_params, peak_params, exp_params = results[i]

                # Unpack results
                if np.isnan(indices).any() or any([i < 0 for i in indices]):

                    if verbose:
                        warnings.warn(f'Fail fit for spike: {i}')

                    self.indices[i] = [-999 for i in indices]
                    self.inds_error.append(i)

                else:
                    self.indices[i] = indices

                self.ramp_poly_params[i], self.ramp_amp[i], self.inflection_time[i], \
                    self.inflection_amp[i] = ramp_params

                self.peak_amp[i], self.peak_width[i], self.peak_sharpness[i] = peak_params

                self.exp_amp[i], self.exp_lambda[i], self.exp_const[i] = exp_params

        # Check if all fits failed
        if (self.indices[:, 0] == -999).all():
            raise ValueError('All fits failed.')

        # Compute inter spike features

        self.isi = compute_isi(self.spike_inds, self.fs, True)

        # Generate fits
        if gen_fits:
            self.gen_fit()

        # Generate the dataframe
        self.gen_df_features()

        # Generate sample indices
        if gen_indices:
            self.gen_df_indices()

        # Run alts
        if self.queue is not None:

            for locs in self.queue:

                args = [locs[k] for k in locs if k in ['sig', 'fs', 'func']]
                kwargs =  {k:locs[k] for k in locs if k not in ['self', 'sig', 'fs', 'func']}

                self.alt(*args, **kwargs)

               


    def alt(self, sig, fs, func, func_args=None, func_kwargs=None, param_keys=None,
            ref='peak', window_length=(10., 10.), preload=False,
            n_jobs=1, progress=None, queue=False):
        """Compute features for an alternative/associated signal.

        Parameters
        ----------
        sig : 1d array
            Alternaitve voltage time series.
        fs : float
            Alternaitve sampling rate, in Hz.
        func : function
            Computes features for each window. Each object returned should be {float, int, str}.
            The signature must be: func(sig, fs, *arg, **kwargs).
        func_args : tuple, optional, default: None
            Arguments to pass to func.
        func_kwargs : dict, optional, default: None
            Keyword arguments to pass to func.
        param_keys : list of str
            Names of features returned from func.
            These names become columns appended to df_features.
        ref : {'ramp_start', 'inflection', 'rise', 'peak', 'decay', 'exp_start', 'exp_end'}
            Reference used to create windows.
        window_length : tuple of (float, float)
            Number of milliseconds before and after the reference point to include.
        preload : bool, optional, default: False
            If True, self.alt_windows have been set,
            ignoring the sig argument. Used for group sub-class.
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

            self.queue = [] if self.queue is None else self.queue

            _queue = {k: v for k, v in locals().items() if k != 'self'}
            _queue['queue'] = False

            self.queue.append(_queue)

            return

        # Window the alternative signal
        if not preload:
            inds = ((self.df_indices[ref].values / self.fs) * fs).astype(int)
            alt_windows = window_spike(sig, fs, inds, window_length=window_length)
        else:
            alt_windows = self.alt_windows

        # Handle args and kwargs
        if func_args is None:
            func_args = ()
        elif not isinstance(func_args, (tuple, list)):
            func_args = [func_args]

        if func_kwargs is None:
            func_kwargs = {}

        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        # In series
        if n_jobs == 1:

            iterable = range(len(alt_windows))

            if progress is not None:
                iterable = progress(iterable, total=len(alt_windows), desc='Alt')

            for ind in iterable:

                _params =  _compute_alt_features(fs, func, alt_windows[ind],
                                                 args=func_args, kwargs=func_kwargs)

                if ind == 0:
                    params = np.zeros((len(alt_windows), len(_params)), dtype='object')

                if ind == 0 and param_keys is None:
                    param_keys = ['alt_' + str(i) for i in range(len(_params))]

                params[ind] = _params

            # Add to dataframe
            for ind in range(len(param_keys)):
                self.df_features[param_keys[ind]] = params[:, ind].astype(type(params[0, ind]))

        # In parallel
        else:
            # Prevent all windows from being passed into pool
            self.alt_windows = None

            # Move func to a .py file
            if os.path.isfile('_tmp_funcs_mp.py'):
                os.remove('_tmp_funcs_mp.py')

            lines = inspect.getsource(func)

            # Ensure consistent func name for importing
            func_name = func.__name__

            lines = re.sub(f'def {func_name}', 'def func', lines)

            # Write function
            with open('_tmp_funcs_mp.py', 'w') as f:
                for line in lines:
                    f.write(line)

            # Import
            from _tmp_funcs_mp import func

            # Run mp pool
            with Pool(processes=n_jobs) as pool:

                pfunc = partial(_compute_alt_features, fs, func, args=func_args, kwargs=func_kwargs)

                mapping = pool.imap(pfunc, alt_windows)

                if progress is None:
                    results = list(mapping)
                else:
                    results = list(progress(mapping, total=len(alt_windows), desc='Alt'))

            # Reset
            self.alt_windows = alt_windows

            # Transpose results list
            params = [np.array(i) for i in zip(*results)]

            if param_keys is None:
                param_keys = ['alt_' + str(i) for i in range(len(params))]

            # Add to dataframe
            for ind in range(len(param_keys)):
                self.df_features[param_keys[ind]] = params[ind]

            # Remove temporary py file
            os.remove('_tmp_funcs_mp.py')

        del params

        # Track windows as an attribute
        if not preload and self.alt_windows is None:
            self.alt_windows = alt_windows
        elif not preload and isinstance(self.alt_windows, np.ndarray):
            self.alt_windows = [self.alt_windows]
            self.alt_windows.append(alt_windows)




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

            # Center times around peak
            for ind in self.indices:

                if ind[3] > 0:
                    self.times -= self.times[ind[3]]
                    break

        
        
        
        for ind in range(len(self.spikes)):

            if ramp and ind not in self.inds_error:
                # Ramp
                start, end = self.indices[ind][0], self.indices[ind][1]

                _times = np.arange(end-start) * 1000 / self.fs

                _fit_ramp, _r2_ramp = gen_fit_ramp(_times, self.spikes[ind][start:end],
                                                   self.ramp_poly_params[ind])

                # Initalize arrays
                if self.fit_ramp is None:
                    self.fit_ramp = np.zeros((len(self.spikes), len(_fit_ramp)))
                    self.r_squared_ramp = np.zeros(len(self.spikes))
                   
                    
               
                # Store in attr
                if len(_fit_ramp) != len(self.fit_ramp[ind]):
                    self.fit_ramp[ind] = np.nan
                    self.r_squared_ramp[ind] = np.nan
                    
               
                                          
                else:
                    self.fit_ramp[ind] = _fit_ramp
                    self.r_squared_ramp[ind] = _r2_ramp
          

            if exp and ind not in self.inds_error:
                # Exponential decay
                start, end = self.indices[ind][-2], self.indices[ind][-1]

                _times = np.arange(end-start) * 1000 / self.fs

                _fit_exp, _r2_exp = gen_fit_exp(_times, self.spikes[ind][start:end],
                                                (self.exp_amp[ind], self.exp_lambda[ind],
                                                self.exp_const[ind]))

                # Initalize arrays
                if self.fit_exp is None:
                    self.fit_exp = np.zeros((len(self.spikes), len(_fit_exp)))
                    self.r_squared_exp = np.zeros(len(self.spikes))

                # Store in attr
                if len(_fit_exp) != len(self.fit_exp[ind]):
                    self.fit_exp[ind] = np.nan
                    self.r_squared_exp[ind] = np.nan
               
                    
                else:
                    self.fit_exp[ind] = _fit_exp
                    self.r_squared_exp[ind] = _r2_exp
                
       
        # Fill error fits with nans
        for ind in self.inds_error:

            self.fit_ramp[ind] = np.nan
            self.r_squared_ramp[ind] = np.nan
            self.fit_exp[ind] = np.nan
            self.r_squared_exp[ind] = np.nan


    def gen_df_features(self):
        """Generate feature dataframe."""

        columns = ['ramp_amp', 'inflection_time', 'inflection_amp', 'peak_amp',
                   'peak_width', 'peak_sharpness', 'exp_lambda', 'exp_const', 'isi']

        self.df_features = pd.DataFrame()

        if self.group is not None:
            self.df_features['group'] = self.group

        for c in columns:
            self.df_features[c] = getattr(self, c)

        for _param in ['r_squared_ramp', 'r_squared_exp']:
            if hasattr(self, _param):
                self.df_features[_param] = getattr(self, _param)


    def gen_df_indices(self):
        """Generate sample indices dataframe."""

        columns  = ['ramp_start', 'inflection', 'rise',
                    'peak', 'decay', 'exp_start', 'exp_end']

        ref_inds = self.indices[:, 3]

        self.df_indices = pd.DataFrame()


        if isinstance(self.spike_inds, list) and self.spike_inds[0].ndim == 1:
             # If called from SpikeGroup
            _spike_inds = [j for i in self.spike_inds for j in i]
        else:
            _spike_inds = self.spike_inds

        for col, inds in zip(columns, self.indices.T):
            self.df_indices[col] = _spike_inds + (inds - ref_inds)


    def plot(self, inds=None, mode='full', in_ms=True, show_points=False, ax=None, groups=False, ind_groups=None, group_names=None, plot_average=False, plot_average_std=False, color='C0'):
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
        plot_model(self, inds, mode, in_ms, show_points, ax, groups, ind_groups, group_names, plot_average, plot_average_std, color)


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
    verbose = kwargs.pop('verbose')

    try:

        if verbose:
            indices, ramp_params, peak_params, exp_params = \
               compute_features(spike, fs, poly_order=poly_order, **kwargs)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                indices, ramp_params, peak_params, exp_params = \
                    compute_features(spike, fs, poly_order=poly_order, **kwargs)
    except:

        indices = [np.nan] * 7

        ramp_params = [
            [np.nan] * (poly_order + 1),
            np.nan, np.nan, np.nan
        ]

        peak_params = [np.nan, np.nan, np.nan]

        exp_params = [np.nan, np.nan, np.nan]

    return indices, ramp_params, peak_params, exp_params


def _compute_alt_features(fs, func, sig, args=None, kwargs=None):
    """Warpper function for computing alternative features."""
    res = func(sig, fs, *args, **kwargs)

    if not isinstance(res, (tuple, list, np.ndarray)):
        res = [res]

    return res
