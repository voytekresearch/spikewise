"""Simulate spikes as polynomials."""

from functools import partial
import numpy as np
from scipy.optimize import curve_fit



def sim_ppoly(xs, knots, coeffs, degree=2):
    """Simulate piecewise polynomials.

    Parameters
    ----------
    xs : 1d array
        Time definition.
    knots : 1d array
        Spike segmentation locations, in samples.
    coeffs : 1d array
        Series of polynomials coefficents.Is reshaped based on degree
        and must be ordered as expected by np.poly1d.
    degree : int of list of int, optional, default: 2
        Polynomial order for all segments (int) or a unique order for
        each segment (list).

    Returns
    -------
    ys : 1d array
        Voltage time series.
    """
    # Initalize array
    ys = np.zeros(len(xs))

    # Ensure degree is iterable
    if isinstance(degree, int):
        degree = np.tile(degree, len(knots)-1)

    # Reshape coefficients based on degree
    _coeffs = []
    pos = 0
    for i in degree:
        _coeffs.append(coeffs[pos:pos+i+1])
        pos += i+1

    # Simulate each segment
    starts = knots[:-1]
    ends = knots[1:] + 1

    for i in range(len(starts)):
        ys[starts[i]:ends[i]] = np.poly1d(_coeffs[i])(xs[starts[i]:ends[i]])

    return ys


def sim_ppoly_dist(means, cov, degree, n_sims, seeds=None):
    """Simulate a distribution of spikes.

    Parameters
    ----------
    means : 1d array
        Means of parameters (column stacked coefficients and indices).
    cov : 2d array
        Parameter covariance.
    degree : 1d array
        Polynomial degrees.
    n_sims : int
        Number of simulations.
    seeds : 1d array or int
        Simulation seeds.

    Returns
    -------
    spikes : 2d array
        Simulated spikes.
    sim_coeffs : 2d array
        Polynomial coefficients per simulation.
    sim_indices : 2d array
        Knot indices per simulation.
    """

    # Sample parameters from multivar norm
    sim_params = np.zeros((n_sims, len(means)))

    for n in range(n_sims):

        if seeds is not None and not isinstance(seeds, int):
            np.random.seed(seeds[n])
        elif seeds is not None:
            np.random.seed(seeds)

        sim_params[n] = np.random.multivariate_normal(means, cov)

    # Initalize
    n_coeffs = sum([i+1 for i in degree])
    sim_indices = sim_params[:, n_coeffs:].astype(int)
    sim_coeffs = sim_params[:, :n_coeffs]

    max_len = np.max(sim_indices[:, -1])
    spikes = np.zeros((n_sims, max_len))
    spikes[:] = np.nan

    # Run simulations
    for n in range(n_sims):

        xs = np.arange(sim_indices[n][-1])

        # Get the target array
        target = sim_ppoly_partial(xs, *sim_coeffs[n], knots=sim_indices[n], degree=degree)

        starts = sim_indices[n][:-1]
        diff = np.diff(target)

        for i in range(1, len(starts)):
            target[starts[i]:] += (target[starts[i]-1] + diff[starts[i]-2]) - target[starts[i]]

        # Solve and update constant coefficients
        sim_coeffs[n] = solve_constants(target, sim_coeffs[n], knots=sim_indices[n], degree=degree)

        # Regenerate new fit
        spike = sim_ppoly_partial(xs, *sim_coeffs[n], knots=sim_indices[n], degree=degree)

        spikes[n, max_len-sim_indices[n][-1]:] = spike

    return spikes, sim_coeffs, sim_indices


def solve_constants(ys, coeffs, knots=None, degree=None):
    """Solves constant polynomial parameter."""

    pfunc = partial(sim_ppoly_partial, knots=knots, degree=degree)

    xs = np.arange(len(ys))

    coeffs, _ = curve_fit(pfunc, xs, ys, p0=coeffs)

    return coeffs



def sim_ppoly_partial(xs, *coeffs, knots=None, degree=None):
    """Remapped arguments suitable for partial and curve_fit."""

    return sim_ppoly(xs, knots, coeffs, degree=degree)
