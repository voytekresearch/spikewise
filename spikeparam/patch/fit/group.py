"""Group fitting."""

import warnings
from copy import copy
from functools import partial

from multiprocessing import Pool, cpu_count

import pandas as pd
import numpy as np

from spikeparam.patch.fit.fit import Spike




class SpikeGroup:

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

        # Results
        self.spikes = None
        self.times = None
        self.indices = None
        self.inds_error = None
        self.fit_ramp = None
        self.fit_exp = None

    def fit(self, sigs, fs, reader=None, gen_fits=True, gen_indices=True, verbose=False,
            n_jobs=1, progress=None):
        """Fit 2d signals.
        """

        # Initalize
        base_model = Spike(
            self.window_length, self.thresh_amp, self.thresh_ms,
            self.pre_peak_ms, self.pre_inflection_ms, self.smooth_frac,
            self.poly_order, self.exp_shift_right, self.exp_duration, self.corr_thresh
        )


        n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        if n_jobs == 1:
            # To-Do
            pass
        else:

            with Pool(processes=n_jobs) as pool:

                mapping = pool.imap(
                    partial(_fit, model=base_model, fs=fs, reader=reader,
                            gen_fits=gen_fits, gen_indices=gen_indices, verbose=self.verbose),
                    sigs
                )

                if progress is not None:
                    results = list(progress(mapping, total=len(sigs)))
                else:
                    results = list(mapping)

            self.results = results

            self.df_features = pd.concat(
                [r.df_features for r in results if r.df_features is not None]
            )

            # 2d array of spikes
            for r in results:
                if r.spikes is not None and self.spikes is None:
                    self.spikes = r.spikes
                elif r.spikes is not None:
                    self.spikes = np.vstack((self.spikes, r.spikes))

            # Spike times
            for r in results:
                if r.df_features is not None and self.times is None:
                    self.times = r.times
                    break


            # Combine results
            inds_error = []
            i=0

            for r in results:
                if r.indices is not None:

                    # Sample indices
                    if self.indices is None:
                        self.indices = r.indices
                    else:
                        self.indices = np.vstack((self.indices, r.indices))

                    # Error fits
                    for inds in r.indices:
                        if np.isnan(inds).any():
                            inds_error.append(i)
                        i+=1

                    # Combine fits
                    if r.fit_ramp is not None and self.fit_ramp is None:
                        self.fit_ramp = r.fit_ramp
                    elif r.fit_ramp is not None:
                        self.fit_ramp = np.vstack((self.fit_ramp, r.fit_ramp))

                    if r.fit_exp is not None and self.fit_exp is None:
                        self.fit_exp = r.fit_exp
                    elif r.fit_exp is not None:
                        self.fit_exp = np.vstack((self.fit_exp, r.fit_exp))


def _fit(ind, model=None, fs=None, reader=None, gen_fits=None, gen_indices=None, verbose=None):

    if reader is None:
        # To-Do
        pass
    else:
        arrays = reader(ind)

    if not verbose:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            model.fit(arrays['fit'], fs, gen_fits, gen_indices)

    else:
        model.fit(arrays['fit'], fs, gen_fits, gen_indices)

    if model.df_features is not None:
        model.df_features['index'] = ind

        cols = list(model.df_features.columns)
        cols = [cols[-1]] + cols[:-1]

        model.df_features = model.df_features[cols]

    return model
