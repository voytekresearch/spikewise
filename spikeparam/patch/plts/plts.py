"""Plotting functions."""

import matplotlib.pyplot as plt
import numpy as np


def _half_width(w):
    w = np.asarray(w, float)
    peak = np.max(np.abs(w))
    return int(np.sum(np.abs(w) >= 0.5 * peak)) if peak > 0 else 0


def _filter_outlier_spikes(indices, spikes, iqr_thresh=1.5):
    """
    Return indices whose peak amplitude AND half-width are within
    Q1 ± iqr_thresh*IQR of the group. Catches narrow artifact spikes
    that have normal amplitude but biologically implausible width.
    """
    if not indices or iqr_thresh is None:
        return indices

    peaks  = np.array([np.max(np.abs(spikes[i])) for i in indices])
    widths = np.array([_half_width(spikes[i]) for i in indices])

    def _bounds(arr):
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        return q1 - iqr_thresh * iqr, q3 + iqr_thresh * iqr

    p_lo, p_hi = _bounds(peaks)
    w_lo, w_hi = _bounds(widths)

    return [i for i, p, wd in zip(indices, peaks, widths)
            if p_lo <= p <= p_hi and w_lo <= wd <= w_hi]


def _peak_align(waveforms, times, wght=1, peak_idxs=None):
    """
    Align each waveform so its peak falls at t=0.

    Creates a common grid spanning the widest pre-peak and post-peak window
    across all waveforms. Shorter waveforms are NaN-padded.

    Parameters
    ----------
    waveforms : list of 1-D array-like
    times : 1-D array
        Original time array (same length as each waveform).
    wght : float
        Multiplier applied to times (1000 for ms).
    peak_idxs : list of int, optional
        Pre-computed peak indices (one per waveform). When provided, these are
        used instead of argmax. Pass model.indices[:, 3] to avoid argmax
        picking a neighboring spike that entered the window.

    Returns
    -------
    aligned : np.ndarray, shape (n_spikes, output_len)
        Peak-aligned waveforms; peak of every spike is at index `pre`.
    t_axis : np.ndarray
        Time axis in ms (or samples if wght=1), with 0 at the peak.
    """
    if not waveforms:
        return np.empty((0, 0)), np.array([0.0])

    wfs = [np.asarray(w, float) for w in waveforms]
    if peak_idxs is None:
        peak_idxs = [int(np.argmax(w)) for w in wfs]
    else:
        peak_idxs = [int(p) for p in peak_idxs]

    pre   = max(peak_idxs)                                          # samples before peak
    post  = max(len(w) - pk - 1 for w, pk in zip(wfs, peak_idxs)) # samples after peak
    total = pre + post + 1

    aligned = np.full((len(wfs), total), np.nan)
    for k, (w, pk) in enumerate(zip(wfs, peak_idxs)):
        s = pre - pk                    # where this waveform starts in the grid
        aligned[k, s:s + len(w)] = w

    dt     = float(np.mean(np.diff(times))) if len(times) > 1 else 1.0
    t_axis = (np.arange(total) - pre) * dt * wght  # peak = 0

    return aligned, t_axis


def plot_model(model, inds=None, mode='full', in_ms=True, show_points=False, ax=None, groups=False, ind_groups=None, group_names=None, plot_average=False, plot_average_std=False, color_spks='C0', peak_align=True, outlier_thresh=3.0):
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
                n_spikes = len(model.spikes)
                all_wfs = [model.spikes[i] for grp in ind_groups for i in grp
                           if i not in model.inds_error and i < n_spikes]
                if peak_align and all_wfs:
                    _, t_plot = _peak_align(all_wfs, model.times, wght)
                    # Global pre = index where t=0 lives in the shared grid
                    global_pre = int(np.argmin(np.abs(t_plot)))
                else:
                    t_plot     = _times
                    global_pre = None

                offset = 0
                for idx, group in enumerate(ind_groups):
                    color = colors[idx] if idx < len(colors) else plt.cm.viridis(float(idx) / len(ind_groups))
                    clean_group = _filter_outlier_spikes(
                        [i for i in group if i not in model.inds_error and i < n_spikes],
                        model.spikes, outlier_thresh)
                    wfs   = [model.spikes[i] for i in clean_group]
                    if not wfs:
                        continue
                    if peak_align and global_pre is not None:
                        # Align each group using the GLOBAL pre so all groups
                        # share the same t=0 reference point.
                        wfs_arr   = [np.asarray(w, float) for w in wfs]
                        peak_idxs = [int(np.argmax(np.abs(w))) for w in wfs_arr]
                        global_post = max(len(w) - pk - 1
                                         for w, pk in zip(wfs_arr, peak_idxs))
                        total = global_pre + global_post + 1
                        arr   = np.full((len(wfs_arr), total), np.nan)
                        for k, (w, pk) in enumerate(zip(wfs_arr, peak_idxs)):
                            s = global_pre - pk
                            if s >= 0:
                                arr[k, s:s + len(w)] = w
                            else:
                                arr[k, :len(w) + s] = w[-s:]
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

        # Pre-compute peak-aligned waveforms and a shared time axis.
        # Use argmax(|w|) on each raw waveform — same as the May-7 working version.
        # (June-13 "fix" used model.indices[i][3] which is always the hardcoded
        # threshold-crossing index 150, not the actual spike peak.)
        valid_inds = [i for i in inds if i not in model.inds_error and i < len(model.spikes)]
        if peak_align and valid_inds:
            raw_wfs   = [model.spikes[i] for i in valid_inds]
            peak_idxs = [int(np.argmax(np.abs(w))) for w in raw_wfs]
            pre       = max(peak_idxs)
            aligned_arr, t_aligned = _peak_align(raw_wfs, model.times, wght,
                                                  peak_idxs=peak_idxs)
            aligned_map = {i: aligned_arr[k] for k, i in enumerate(valid_inds)}
            pk_map = {i: pk for i, pk in zip(valid_inds, peak_idxs)}
        else:
            aligned_map = {}
            pk_map      = {}
            t_aligned   = _times
            pre         = 0

        if mode == 'full':

            for i in inds:

                if i in model.inds_error:
                    continue

                wf   = aligned_map[i] if peak_align else model.spikes[i]
                t_wf = t_aligned[:len(wf)] if peak_align else _times
                ax.plot(t_wf, wf, color=color_spks, label=lab_true, alpha=alpha)
                lab_true = ''

                if show_points:
                    if peak_align:
                        # Shift indices to account for the alignment offset.
                        # model.indices[i] are positions in the ORIGINAL window;
                        # after alignment each waveform is shifted by (pre - pk_i)
                        # so we must offset indices accordingly.
                        pk_i   = pk_map.get(i, int(model.indices[i][3]))
                        shift  = pre - pk_i
                        adj    = model.indices[i] + shift
                    else:
                        adj = model.indices[i]
                    _plot_control_points(t_wf, wf, adj, ax)

            for i in inds:

                if i in model.inds_error:
                    continue

                pk_i = pk_map.get(i, 0)
                dt   = float(np.mean(np.diff(model.times))) if len(model.times) > 1 else 1.0

                # Ramp — time relative to this spike's own peak
                start, end = model.indices[i][0], model.indices[i][1]
                if peak_align:
                    t_seg = (np.arange(start, end) - pk_i) * dt * wght
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
                    t_seg = (np.arange(start, end) - pk_i) * dt * wght
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

        # Clip to the nominal window so the axis always shows ±window_length ms,
        # regardless of NaN-padded tails or edge spikes.
        if in_ms and hasattr(model, 'window_length'):
            ax.set_xlim(-model.window_length[0], model.window_length[1])

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
                clean_group = _filter_outlier_spikes(
                    [i for i in group if i not in model.inds_error and i < len(model.spikes)],
                    model.spikes, outlier_thresh)
                for i in clean_group:
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
