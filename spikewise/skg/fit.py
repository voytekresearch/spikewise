"""Skewed Gaussian model."""

import numpy as np
from scipy.optimize import curve_fit
from neurodsp.sim.cycles import sim_skewed_gaussian_cycle


def sim_gaussian_spike(
    xs,
    gauss_a_ctr, gauss_a_std, gauss_a_alpha, gauss_a_scale,
    gauss_b_ctr, gauss_b_std, gauss_b_alpha, gauss_b_scale,
    scale, offset, return_separate=False
):
    """Simulate a spike as the sum of two skewed Gaussians.

    docme
    """
    gauss_a = sim_skewed_gaussian_cycle(
        1, len(xs),gauss_a_ctr, gauss_a_std, gauss_a_alpha
    )
    gauss_a = gauss_a * gauss_a_scale

    gauss_b = sim_skewed_gaussian_cycle(
        1, len(xs), gauss_b_ctr, gauss_b_std, gauss_b_alpha
    )
    gauss_b = gauss_b * gauss_b_scale

    gauss_a = gauss_a * scale + offset/2
    gauss_b = gauss_b * scale + offset/2

    if return_separate:
        return gauss_a, gauss_b

    return gauss_a + gauss_b


class SKG:

    def __init__(self, p0=None, bounds=None, maxfev=None):
        """Initialization.

        docme
        """
        # Inital parameters
        self.p0 = p0

        # Bounds to simulation as ((*lower), (*upper))
        self.bounds =  (
            (0, 1e-4, -1e4,   0, 0, 1e-4, -1e4,   0, 1e-1, -1e4),
            (1,    1,  1e4, 1e2, 1,    1,  1e4, 1e2, 1e4, 1e4)
        ) if bounds is None else bounds

        self.maxfev = 10_000 if maxfev is None else maxfev
        self.params_ = None

    def fit(self, X, progress=None):
        """Fit.

        docme
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        ts = np.arange(len(X[0]))

        self.params_ = np.zeros((len(X), 10))

        # Wrap iterable with tqdm
        iterable = enumerate(X)
        if progress is not None:
            iterable = progress(iterable, total=len(X))

        # Fit
        for ix, x in iterable:
            self.params_[ix], _ = curve_fit(
                sim_gaussian_spike, ts, x, maxfev=self.maxfev,
                bounds=self.bounds, p0=self.p0
            )
