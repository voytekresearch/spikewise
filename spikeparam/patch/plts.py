"""Plotting functions."""

import matplotlib.pyplot as plt
import numpy as np



def plot(model, inds=None, mode='full', in_ms=True, show_points=False, ax=None):
    """Plot model results.

    Parameters
    ----------
    model : Spike
        Fit Spike object.
    inds : {None, int, list of int}
        Spike indices to plot.
    mode : {'full', 'ramp', 'exp'}
        Whether to plot the full spike, or ramping/exponential decay portions.
    in_ms : bool, optional, default: True
        Scales x-axis to ms if True.
    show_points : bool, optional, default: False
        Show control points used to segment the spike.
    ax : axis
        Axis to plot on.
    """
    # Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 4))

    wght = 1000 if in_ms else 1

    lab_true = 'Actual'
    lab_fit = 'Fit'

    if isinstance(inds, int):
        inds = [inds]
    elif inds is None:
        inds = range(len(model.spikes))

    alpha = 1/(len(inds)**.33)

    # Plot full fit
    if mode == 'full':

        for i in inds:
            _times = model.times * wght
            ax.plot(_times, model.spikes[i], color='C0', label=lab_true, alpha=alpha)
            lab_true = ''

            if show_points:
                _plot_control_points(_times, model.spikes[i], model.indices[i], ax)

        for i in inds:

            start, end = model.indices[i][0], model.indices[i][1]
            ax.plot(model.times[start:end] * wght, model.fit_ramp[i], color='C1', label=lab_fit, alpha=alpha)
            lab_fit = ''

            start, end = model.indices[i][-2], model.indices[i][-1]
            ax.plot(model.times[start:end]* wght, model.fit_exp[i], color='C1', label=lab_fit, alpha=alpha)

    # Only plot ramp fit
    elif mode == 'ramp':

        _times = np.arange(len(model.fit_ramp[0])) * wght / model.fs

        for i in inds:
            start, end = model.indices[i][0], model.indices[i][1]
            ax.plot(_times, model.spikes[i][start:end], color='C0', label=lab_true, alpha=alpha)
            lab_true = ''

        for i in inds:
            ax.plot(_times, model.fit_ramp[i], color='C1', label=lab_fit, alpha=alpha)
            lab_fit = ''

    # Only plot exp fit
    elif mode == 'exp':

        _times = np.arange(len(model.fit_exp[0])) * wght / model.fs

        for i in inds:
            start, end = model.indices[i][-2], model.indices[i][-1]
            ax.plot(_times, model.spikes[i][start:end], color='C0', label=lab_true, alpha=alpha)
            lab_true = ''

        for i in inds:
            ax.plot(_times, model.fit_exp[i], color='C1', label=lab_fit, alpha=alpha)
            lab_fit = ''

    ax.set_ylabel('Voltage')
    ax.set_xlabel('Time (ms)')
    ax.legend()


def _plot_control_points(times, spike, indices, ax):

    labels = ['Start', 'Inflection', 'Rise', 'Peak', 'Decay']
    colors = ['C' + str(i) for i in range(2, 7)]
    for i, l, c in zip(indices[:-2], labels, colors):
        ax.plot(times[i], spike[i], ls='', marker='.', ms=16, label=l, color=c)
