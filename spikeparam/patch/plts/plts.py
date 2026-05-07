"""Plotting functions."""

import matplotlib.pyplot as plt
import numpy as np


def _peak_align(waveforms, times, wght=1):
    """
    Shift each waveform so its peak (max |amplitude|) aligns to the median
    peak position across all waveforms.

    Parameters
    ----------
    waveforms : list of 1-D array-like
    times : 1-D array
        Original time array (same length as each waveform).
    wght : float
        Multiplier applied to times (1000 for ms).

    Returns
    -------
    aligned : np.ndarray, shape (n_spikes, output_len)
        Peak-aligned waveforms, NaN-padded at edges.
    t_axis : np.ndarray
        Time axis centered at 0 (peak = 0).
    """
    if not waveforms:
        return np.empty((0, len(times))), times * wght

    wfs       = [np.asarray(w, float) for w in waveforms]
    peak_idxs = [int(np.argmax(np.abs(w))) for w in wfs]
    ref       = int(np.median(peak_idxs))

    aligned = []
    for w, pk in zip(wfs, peak_idxs):
        shift = ref - pk
        if shift > 0:
            shifted = np.concatenate([np.full(shift, np.nan), w])
        elif shift < 0:
            shifted = w[-shift:]
        else:
            shifted = w.copy()
        aligned.append(shifted)

    min_len = min(len(a) for a in aligned)
    aligned = np.array([a[:min_len] for a in aligned])

    dt     = float(np.mean(np.diff(times))) if len(times) > 1 else 1.0
    t_axis = (np.arange(min_len) - ref) * dt * wght

    return aligned, t_axis


def plot_model(model, inds=None, mode='full', in_ms=True, show_points=False, ax=None, groups=False, ind_groups=None, group_names=None, plot_average=False, plot_average_std=False, color_spks='C0', peak_align=True):
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
        _, ax = plt.subplots(figsize=(14, 4))

    wght = 1000 if in_ms else 1

    lab_true = 'Actual'
    lab_fit = 'Fit'

    if isinstance(inds, int):
        inds = [inds]
    elif inds is None:
        inds = range(len(model.spikes))

    # Dynamic alpha
    alpha = 1/(len(inds)**.33)

    # Plot full fit
    _times = model.times * wght

    colors = ['firebrick', 'royalblue']
    if plot_average:

        # Create custom legend handles and labels
        custom_legend_handles = []

        if groups:
            if ind_groups is None:
                raise ValueError("ind_groups cannot be None when groups is True.")

            if group_names is None:
                group_names = [f'Group {i}' for i in range(len(ind_groups))]

            if plot_average:
                # Compute shared peak-alignment reference across ALL groups so
                # they share the same time axis when peak_align=True.
                all_wfs = [model.spikes[i] for grp in ind_groups for i in grp
                           if i not in model.inds_error]
                if peak_align and all_wfs:
                    _, t_plot = _peak_align(all_wfs, model.times, wght)
                else:
                    t_plot = _times

                offset = 0
                for idx, group in enumerate(ind_groups):
                    color = colors[idx] if idx < len(colors) else plt.cm.viridis(float(idx) / len(ind_groups))
                    wfs   = [model.spikes[i] for i in group if i not in model.inds_error]
                    if not wfs:
                        continue
                    if peak_align:
                        arr, _ = _peak_align(wfs, model.times, wght)
                    else:
                        arr = np.array(wfs)
                    n = min(arr.shape[1], len(t_plot))
                    avg_actual = np.nanmean(arr[:, :n], axis=0)
                    ax.plot(t_plot[:n], avg_actual, color=color, alpha=0.7,
                            label=f'Avg Actual {group_names[idx]}', linewidth=5)
                    custom_legend_handles.append(
                        plt.Line2D([0], [0], color=color, marker='.', markersize=8,
                                   label=f'Average Actual {group_names[idx]}'))
                    if plot_average_std:
                        std_dev = np.nanstd(arr[:, :n], axis=0)
                        ax.fill_between(t_plot[:n], avg_actual - std_dev,
                                        avg_actual + std_dev, color=color, alpha=0.3)
                        custom_legend_handles.append(
                            plt.fill_between([], [], [], color=color, alpha=0.3,
                                             label=f'Std Dev Band {group_names[idx]}'))

        else:
            wfs = [model.spikes[i] for i in range(len(model.spikes))]
            if peak_align and wfs:
                arr, t_plot = _peak_align(wfs, model.times, wght)
            else:
                arr    = np.array(wfs)
                t_plot = _times
            avg_actual = np.nanmean(arr, axis=0)
            ax.plot(t_plot, avg_actual, color=color_spks, label='Avg Actual', linewidth=5)
            if plot_average_std:
                std_dev = np.nanstd(arr, axis=0)
                ax.fill_between(t_plot, avg_actual - std_dev,
                                avg_actual + std_dev, color=color_spks, alpha=0.3)
            custom_legend_handles.append(
                plt.Line2D([0], [0], color=color_spks, marker='.', markersize=8,
                           label='Average Actual'))
            custom_legend_handles.append(
                plt.fill_between([], [], [], color=color_spks, alpha=0.3, label='Std Dev Band'))


    else:

        # Pre-compute peak-aligned waveforms, shared time axis, and per-spike shifts
        valid_inds = [i for i in inds if i not in model.inds_error]
        if peak_align and valid_inds:
            raw_wfs     = [model.spikes[i] for i in valid_inds]
            peak_idxs   = [int(np.argmax(np.abs(w))) for w in raw_wfs]
            ref         = int(np.median(peak_idxs))
            aligned_arr, t_aligned = _peak_align(raw_wfs, model.times, wght)
            aligned_map = {i: aligned_arr[k] for k, i in enumerate(valid_inds)}
            shift_map   = {i: ref - pk for i, pk in zip(valid_inds, peak_idxs)}
        else:
            aligned_map = {}
            shift_map   = {}
            t_aligned   = _times
            ref         = 0

        if mode == 'full':

            for i in inds:

                if i in model.inds_error:
                    continue

                wf   = aligned_map[i] if peak_align else model.spikes[i]
                t_wf = t_aligned[:len(wf)] if peak_align else _times
                ax.plot(t_wf, wf, color=color_spks, label=lab_true, alpha=alpha)
                lab_true = ''

                if show_points:
                    _plot_control_points(t_wf, wf, model.indices[i], ax)

            for i in inds:

                if i in model.inds_error:
                    continue

                sh = shift_map.get(i, 0)

                # Ramp — shift the time window by the same amount as the waveform
                start, end = model.indices[i][0], model.indices[i][1]
                if peak_align:
                    s2, e2 = start + sh, end + sh
                    t_seg  = t_aligned[max(0, s2):max(0, e2)]
                else:
                    t_seg = _times[start:end]
                if len(t_seg) != len(model.fit_ramp[i]):
                    continue
                ax.plot(t_seg, model.fit_ramp[i], color='C1',
                        label=lab_fit, alpha=alpha, ls='--')
                lab_fit = ''

                # Exponential
                start, end = model.indices[i][-2], model.indices[i][-1]
                if peak_align:
                    s2, e2 = start + sh, end + sh
                    t_seg  = t_aligned[max(0, s2):max(0, e2)]
                else:
                    t_seg = _times[start:end]
                if len(t_seg) == len(model.fit_exp[i]):
                    ax.plot(t_seg, model.fit_exp[i], color='C1',
                            label=lab_fit, alpha=alpha, ls='--')

        # Only plot ramp fit
        elif mode == 'ramp':

            for i in inds:
                if i in model.inds_error:
                    continue

                start, end = model.indices[i][0], model.indices[i][1]
                ax.plot(_times[start:end], model.spikes[i][start:end], color=color_spks,
                        label=lab_true, alpha=alpha)
                lab_true = ''

            for i in inds:

                if i in model.inds_error:
                    continue

                ax.plot(_times[start:end], model.fit_ramp[i], color='C1',
                        label=lab_fit, alpha=alpha, ls='--')
                lab_fit = ''

        # Only plot exp fit
        elif mode == 'exp':

            for i in inds:
                if i in model.inds_error:
                    continue

                start, end = model.indices[i][-2], model.indices[i][-1]
                ax.plot(_times[start:end], model.spikes[i][start:end], color=color_spks,
                        label=lab_true, alpha=alpha)
                lab_true = ''

            for i in inds:

                if i in model.inds_error:
                    continue

                ax.plot(_times[start:end], model.fit_exp[i], color='C1',
                        label=lab_fit, alpha=alpha, ls='--')
                lab_fit = ''

        ax.set_ylabel('Voltage')
        ax.set_xlabel('Time (ms)')

        # Create custom legend handles and labels
        custom_legend_handles = []

       
        
        custom_legend_handles.extend([plt.Line2D([0], [0], color=color_spks, marker='.', markersize=8, label='Actual'),
                                           plt.Line2D([0], [0], color='C1', linestyle='--', label='Fit')])

        if show_points:
            custom_legend_handles.extend([plt.Line2D([0], [0], marker='o', markersize=8, label='Start', color='C2'),
                                     plt.Line2D([0], [0], marker='o', markersize=8, label='Inflection', color='C3'),
                                     plt.Line2D([0], [0], marker='o', markersize=8, label='Rise', color='C4'),
                                     plt.Line2D([0], [0], marker='o', markersize=8, label='Peak', color='C5'),
                                     plt.Line2D([0], [0], marker='o', markersize=8, label='Decay', color='C6')])

        


        if groups:
            if ind_groups is None:
                raise ValueError("ind_groups cannot be None when groups is True. Please provide group indeces for the overlay group plots.")

            if group_names is None:
                group_names = [f'Group {i}' for i in range(len(ind_groups))]

            # If plot_average is False, plot individual spikes for each group
            for idx, group in enumerate(ind_groups):
                color = plt.cm.viridis(float(idx) / len(ind_groups))
                for i in group:
                    if i in model.inds_error:
                        continue

                    if mode == 'full':
                        ax.plot(_times, model.spikes[i], color=color, alpha=alpha, label=f'Actual {group_names[idx]}')
                        if show_points:
                            _plot_control_points(_times, model.spikes[i], model.indices[i], ax)
                        start, end = model.indices[i][0], model.indices[i][1]
                        if len(_times[start:end]) == len(model.fit_ramp[i]):
                            ax.plot(_times[start:end], model.fit_ramp[i], color=color, linestyle='--', label=f'Fit {group_names[idx]}')
                        start, end = model.indices[i][-2], model.indices[i][-1]
                        ax.plot(_times[start:end], model.fit_exp[i], color=color, linestyle='--')
                    elif mode == 'ramp':
                        start, end = model.indices[i][0], model.indices[i][1]
                        ax.plot(_times[start:end], model.spikes[i][start:end], color=color, alpha=alpha, label=f'Actual {group_names[idx]}')
                        ax.plot(_times[start:end], model.fit_ramp[i], color=color, linestyle='--', label=f'Fit {group_names[idx]}')
                    elif mode == 'exp':
                        start, end = model.indices[i][-2], model.indices[i][-1]
                        ax.plot(_times[start:end], model.spikes[i][start:end], color=color, alpha=alpha, label=f'Actual {group_names[idx]}')
                        ax.plot(_times[start:end], model.fit_exp[i], color=color, linestyle='--', label=f'Fit {group_names[idx]}')

            # Create custom legend handles and labels
            custom_legend_handles = []

            # Add handles and labels for actual data
            for idx, group in enumerate(ind_groups):
                color = plt.cm.viridis(float(idx) / len(ind_groups))
                custom_legend_handles.append(plt.Line2D([0], [0], color=color, label=f'Actual {group_names[idx]}'))

            # Add handles and labels for fit data
            for idx, group in enumerate(ind_groups):
                color = plt.cm.viridis(float(idx) / len(ind_groups))
                custom_legend_handles.append(plt.Line2D([0], [0], color=color, linestyle='--', label=f'Fit {group_names[idx]}'))

            # Add control points if show_points is True
            if show_points:
                for idx, group in enumerate(ind_groups):
                    color = plt.cm.viridis(float(idx) / len(ind_groups))
                    custom_legend_handles.extend([
                        plt.Line2D([0], [0], marker='o', markersize=8, label=f'Start {group_names[idx]}', color=color),
                        plt.Line2D([0], [0], marker='o', markersize=8, label=f'Inflection {group_names[idx]}', color=color),
                        plt.Line2D([0], [0], marker='o', markersize=8, label=f'Rise {group_names[idx]}', color=color),
                        plt.Line2D([0], [0], marker='o', markersize=8, label=f'Peak {group_names[idx]}', color=color),
                        plt.Line2D([0], [0], marker='o', markersize=8, label=f'Decay {group_names[idx]}', color=color)
                    ])

           




    

    ax.legend(handles=custom_legend_handles)
   


def _plot_control_points(times, spike, indices, ax):

    labels = ['Start', 'Inflection', 'Rise', 'Peak', 'Decay']
    colors = ['C' + str(i) for i in range(2, 7)]
    for i, l, c in zip(indices[:-2], labels, colors):
        ax.plot(times[i], spike[i], ls='', marker='.', ms=16, label=l, color=c)
