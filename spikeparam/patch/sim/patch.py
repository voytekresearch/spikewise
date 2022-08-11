"""Patch recording simulations."""

import numpy as np


def sim_patch(spikes, isi, tau, pad=None):
    """Simulate a patch recording.

    Parameters
    ----------
    spikes : 2d array
        Isolated spike waveforms (e.g from sim_ppoly_dist or sim_ppoly).
    isi : 1d array
        Interspike intervals, in samples.
    tau : float
        Hyperpolarization rate.
    pad : int
        Signal padding. The first and last values of the signal are set as constants.

    Returns
    -------
    sig : 1d array
        Voltage time series.
    """
    # Interspike intervals
    if len(isi) < len(spikes):
        isi = np.append(isi, isi.mean())

    isi = isi.round().astype(int)

    # Ensure spikes are initiated at the same resting voltage
    ind = np.where(np.isnan(spikes).sum(axis=0) == 0)[0][0] + 1

    rest_mv = np.nanmean(spikes[:, :ind], axis=1).mean()

    spikes = spikes + rest_mv - np.nanmean(spikes[:, :ind], axis=1)[:, None]

    # Simulate
    n_samples = np.count_nonzero(~np.isnan(spikes)) + isi.sum() + 1
    sig = np.zeros(n_samples)

    pos = 0
    for ind in range(len(spikes)):

        interval, spike = isi[ind], spikes[ind]

        # Strip nans
        _spike = spike[~np.isnan(spike)]

        # Fill spike
        sig[pos:pos+len(_spike)] = _spike

        # Define hyperpolarization & voltage between spikes
        if ind < len(spikes)-1:
            end = spikes[ind+1][~np.isnan(spikes[ind+1])][0]
        else:
            end = _spike[0]

        xs = np.arange(interval+1)

        hyper = -np.exp(-xs / tau)

        hyper = hyper - hyper.min()
        hyper = hyper / hyper.max()

        hyper *= abs(end - _spike[-1])
        hyper -= hyper[-1]-end

        sig[pos+len(_spike):pos+len(_spike)+interval+1] = hyper

        pos += len(_spike) + interval

    if pad is not None:
        sig = np.pad(sig, pad, constant_values=(sig[0], sig[-1]))

    return sig
